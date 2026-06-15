//! Bounded endpoint cache (Mooncake's `endpoint_store.h`) — a fixed-capacity,
//! evicting cache of per-peer connections, plus a **reclaim** waiting-list so a
//! connection still in flight is not torn down until its last user drops it
//! (Mooncake can't destroy a QP with outstanding work).
//!
//! Generic over the connection type `V`, so the eviction + reclaim logic is
//! unit-tested without an RDMA NIC; [`crate::transport::rdma`] instantiates it
//! with the real QP type behind `--features rdma`. Two policies, matching
//! Mooncake: **FIFO** and **SIEVE** (the second-chance lazy-promotion algorithm).
//!
//! This replaces the transport's earlier unbounded `HashMap` pool, which never
//! evicted or reclaimed — the gap called out in the Mooncake comparison.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Which endpoint a full cache evicts to make room.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Evict the oldest-inserted endpoint.
    Fifo,
    /// SIEVE: scan from a hand, giving a recently-used entry one second chance
    /// (clear its visited bit and skip) before evicting the first unvisited one.
    Sieve,
}

/// A bounded, evicting cache of `Arc<V>` connections keyed by a peer string.
#[derive(Debug)]
pub struct EndpointStore<V> {
    max_size: usize,
    policy: EvictionPolicy,
    map: HashMap<String, Arc<V>>,
    /// SIEVE second-chance bits, parallel to `map`.
    visited: HashMap<String, bool>,
    /// Insertion order (front = oldest); FIFO pops the front, SIEVE scans it.
    order: VecDeque<String>,
    /// SIEVE hand position into `order`.
    hand: usize,
    /// Evicted endpoints still referenced elsewhere (in flight) — kept alive
    /// until [`Self::reclaim`] finds them unreferenced, so a QP isn't destroyed
    /// mid-operation.
    waiting: Vec<Arc<V>>,
}

impl<V> EndpointStore<V> {
    pub fn new(max_size: usize, policy: EvictionPolicy) -> Self {
        Self {
            max_size: max_size.max(1),
            policy,
            map: HashMap::new(),
            visited: HashMap::new(),
            order: VecDeque::new(),
            hand: 0,
            waiting: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Number of evicted-but-not-yet-reclaimed endpoints (still in flight).
    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    /// Get `key`'s connection, or build + insert it via `make` on a miss. A hit
    /// marks the entry used (SIEVE second chance). Inserting past `max_size`
    /// evicts one entry per the policy first.
    pub fn get_or_insert<E, F>(&mut self, key: &str, make: F) -> Result<Arc<V>, E>
    where
        F: FnOnce() -> Result<V, E>,
    {
        if let Some(conn) = self.map.get(key) {
            self.visited.insert(key.to_string(), true);
            return Ok(conn.clone());
        }
        let conn = Arc::new(make()?);
        if self.map.len() >= self.max_size {
            self.evict_one();
        }
        self.map.insert(key.to_string(), conn.clone());
        self.visited.insert(key.to_string(), false);
        self.order.push_back(key.to_string());
        Ok(conn)
    }

    /// Read-only lookup — no second-chance bump (for tests / observability).
    pub fn peek(&self, key: &str) -> Option<Arc<V>> {
        self.map.get(key).cloned()
    }

    fn evict_one(&mut self) {
        let victim = match self.policy {
            EvictionPolicy::Fifo => self.order.pop_front(),
            EvictionPolicy::Sieve => self.sieve_victim(),
        };
        if let Some(key) = victim {
            self.visited.remove(&key);
            if let Some(conn) = self.map.remove(&key) {
                // Still referenced (a transfer holds it)? Defer destruction.
                if Arc::strong_count(&conn) > 1 {
                    self.waiting.push(conn);
                }
            }
        }
    }

    /// SIEVE victim: from the hand, clear+skip visited entries, evict the first
    /// unvisited one (removing it from `order`).
    fn sieve_victim(&mut self) -> Option<String> {
        if self.order.is_empty() {
            return None;
        }
        loop {
            if self.hand >= self.order.len() {
                self.hand = 0;
            }
            let key = self.order[self.hand].clone();
            if *self.visited.get(&key).unwrap_or(&false) {
                self.visited.insert(key, false); // used its second chance
                self.hand += 1;
            } else {
                self.order.remove(self.hand); // hand now points at the next entry
                return Some(key);
            }
        }
    }

    /// Drop any deferred (evicted-but-in-flight) endpoints now unreferenced —
    /// Mooncake's reclaim pass for safe QP destruction.
    pub fn reclaim(&mut self) {
        self.waiting.retain(|conn| Arc::strong_count(conn) > 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(max: usize, policy: EvictionPolicy) -> EndpointStore<i32> {
        EndpointStore::new(max, policy)
    }

    fn insert(s: &mut EndpointStore<i32>, key: &str, val: i32) -> Arc<i32> {
        s.get_or_insert::<(), _>(key, || Ok(val)).unwrap()
    }

    #[test]
    fn get_or_insert_builds_once_then_hits() {
        let mut s = store(4, EvictionPolicy::Fifo);
        let a1 = insert(&mut s, "a", 1);
        // A hit returns the same Arc and does NOT rebuild (closure would panic).
        let a2 = s
            .get_or_insert::<(), _>("a", || panic!("should not rebuild on a hit"))
            .unwrap();
        assert!(Arc::ptr_eq(&a1, &a2));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn fifo_evicts_oldest_and_stays_bounded() {
        let mut s = store(2, EvictionPolicy::Fifo);
        insert(&mut s, "a", 1);
        insert(&mut s, "b", 2);
        insert(&mut s, "c", 3); // over capacity → evict oldest ("a")
        assert_eq!(s.len(), 2, "stays bounded at max_size");
        assert!(s.peek("a").is_none(), "oldest evicted");
        assert!(s.peek("b").is_some() && s.peek("c").is_some());
    }

    #[test]
    fn sieve_gives_a_used_entry_a_second_chance() {
        let mut s = store(2, EvictionPolicy::Sieve);
        insert(&mut s, "a", 1);
        insert(&mut s, "b", 2);
        // Touch "a" → visited. Inserting "c" should spare "a" and evict "b".
        let _ = s.get_or_insert::<(), _>("a", || unreachable!()).unwrap();
        insert(&mut s, "c", 3);
        assert!(s.peek("a").is_some(), "recently-used entry kept");
        assert!(s.peek("b").is_none(), "unvisited entry evicted");
        assert!(s.peek("c").is_some());
    }

    #[test]
    fn reclaim_defers_destruction_until_inflight_ref_drops() {
        let mut s = store(1, EvictionPolicy::Fifo);
        let held = insert(&mut s, "a", 1); // hold an external ref (a "transfer")
        insert(&mut s, "b", 2); // evicts "a" while it is still referenced
        assert_eq!(
            s.waiting_len(),
            1,
            "in-flight eviction deferred, not dropped"
        );
        s.reclaim();
        assert_eq!(s.waiting_len(), 1, "still held → still deferred");
        drop(held);
        s.reclaim();
        assert_eq!(s.waiting_len(), 0, "ref dropped → reclaimed");
    }
}
