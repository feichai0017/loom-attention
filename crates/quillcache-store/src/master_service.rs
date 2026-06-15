//! MasterService (Mooncake's `mooncake-store/include/master_service.h`) — the
//! store's control plane: object metadata, replica allocation, the **two-phase
//! Put**, and lease-based eviction. **No object bytes flow through it** — clients
//! move bytes via the transfer engine directly to/from the allocated buffers;
//! the master only decides *where* and tracks *what is readable*.
//!
//! Two-phase Put (Mooncake's `PutStart` / `PutEnd` / `PutRevoke`):
//! 1. [`MasterService::put_start`] allocates `replica_num` replicas (distinct
//!    segments, via the [`AllocationStrategy`]) and returns their buffers; the
//!    object is `Initialized` (not yet readable).
//! 2. the client writes the bytes into those buffers (transfer engine);
//! 3. [`MasterService::put_end`] flips the replicas to `Complete` (readable), or
//!    [`MasterService::put_revoke`] aborts and frees them.
//!
//! **QuillCache's identity guard** is woven into [`MasterService::get_replica_list`]:
//! each object records the [`IdentityScope`] that wrote it, and a get from a
//! mismatched identity is refused with [`ErrorCode::UnsafeReuse`] — Mooncake
//! isolates by `tenant_id` but not by model / tokenizer / adapter, so extending
//! the guard to the full identity is our addition.

use crate::allocation_strategy::{create_allocation_strategy, AllocationStrategy};
use crate::allocator::{AllocatedBuffer, BufferAllocator};
use crate::count_min_sketch::CountMinSketch;
use crate::offset_allocator::OffsetBufferAllocator;
use crate::replica::{Replica, ReplicaData, ReplicaList, ReplicaStatus};
use crate::types::{ErrorCode, ObjectKey, ReplicaId, ReplicateConfig, SegmentName};
use quillcache_core::IdentityScope;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Per-object control-plane metadata (server-side; never sent to clients).
#[derive(Debug)]
struct ObjectMetadata {
    replicas: ReplicaList,
    /// The identity that wrote this object — QuillCache's guard.
    identity: IdentityScope,
    /// Logical time the read lease expires (blocks remove / eviction until then).
    lease_until: u64,
    /// Logical time of last access (approximate-LRU key).
    last_access: u64,
    soft_pinned: bool,
    hard_pinned: bool,
}

impl ObjectMetadata {
    fn has_complete_replica(&self) -> bool {
        self.replicas.values().any(Replica::is_complete)
    }
}

/// A serializable, consistent copy of the master's in-memory metadata — mounted
/// segments, object replicas, leases/pins, allocation strategy, and the clock.
/// Mooncake's periodic metadata snapshot, taken so a restarted master (or a
/// newly-elected leader, under etcd HA) can rebuild state; changes after the last
/// snapshot are lost (the same bound Mooncake documents).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MasterSnapshot {
    pub version: u32,
    pub strategy: String,
    pub clock: u64,
    pub lease_ttl: u64,
    pub high_watermark: f64,
    pub eviction_ratio: f64,
    pub segment_ttl: u64,
    pub next_replica_id: ReplicaId,
    pub segments: Vec<SegmentSnapshot>,
    pub objects: Vec<ObjectSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentSnapshot {
    pub name: SegmentName,
    pub capacity: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectSnapshot {
    pub key: ObjectKey,
    pub replicas: Vec<Replica>,
    pub identity: IdentityScope,
    pub lease_until: u64,
    pub last_access: u64,
    pub soft_pinned: bool,
    pub hard_pinned: bool,
}

/// The store's metadata / allocation / eviction authority.
#[derive(Debug)]
pub struct MasterService {
    /// Mounted segments → their buffer allocators (one per segment).
    allocators: Vec<Box<dyn BufferAllocator>>,
    objects: HashMap<ObjectKey, ObjectMetadata>,
    strategy: Box<dyn AllocationStrategy>,
    /// The allocation-strategy name, kept so a snapshot can rebuild the same one.
    strategy_name: String,
    next_replica_id: ReplicaId,
    clock: u64,
    lease_ttl: u64,
    high_watermark: f64,
    eviction_ratio: f64,
    /// Last logical-clock tick each segment's node heartbeated — Mooncake's
    /// periodic client heartbeats, the basis for failure detection.
    segment_heartbeat: HashMap<SegmentName, u64>,
    /// Ticks a segment may miss before it is treated as dead. `0` disables the
    /// health check (every mounted segment is considered alive).
    segment_ttl: u64,
    /// Approximate per-key access frequency (Mooncake's CountMinSketch), bumped on
    /// every guarded read so eviction / promotion can favour hot keys.
    hotness: CountMinSketch,
}

impl MasterService {
    pub fn new(strategy: &str) -> Self {
        Self {
            allocators: Vec::new(),
            objects: HashMap::new(),
            strategy: create_allocation_strategy(strategy),
            strategy_name: strategy.to_string(),
            next_replica_id: 0,
            clock: 0,
            lease_ttl: 5,
            high_watermark: 0.95,
            eviction_ratio: 0.1,
            segment_heartbeat: HashMap::new(),
            segment_ttl: 0,
            hotness: CountMinSketch::default(),
        }
    }

    /// Advance the logical clock (drives leases + LRU without wall-clock, so
    /// tests are deterministic).
    pub fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    // ---- segment lifecycle (Mooncake's MountSegment / UnmountSegment) ----

    pub fn mount_segment(&mut self, name: impl Into<SegmentName>, capacity: u64) {
        let name = name.into();
        // A freshly-mounted segment is alive as of now.
        self.segment_heartbeat.insert(name.clone(), self.clock);
        self.allocators
            .push(Box::new(OffsetBufferAllocator::new(name, capacity)));
    }

    /// Unmount a segment: drop any replicas living on it (they become
    /// unreadable), then remove its allocator.
    pub fn unmount_segment(&mut self, name: &str) -> Result<(), ErrorCode> {
        if !self.allocators.iter().any(|a| a.segment_name() == name) {
            return Err(ErrorCode::SegmentNotFound);
        }
        for obj in self.objects.values_mut() {
            obj.replicas.retain(|_, r| r.segment_name() != Some(name));
        }
        self.allocators.retain(|a| a.segment_name() != name);
        self.segment_heartbeat.remove(name);
        // Objects left with no replicas are gone.
        self.objects.retain(|_, o| !o.replicas.is_empty());
        Ok(())
    }

    // ---- HA: heartbeat-based segment health (Mooncake's client heartbeats) ----

    /// Enable failure detection: a segment that misses a heartbeat for more than
    /// `ttl` logical ticks is treated as dead and its replicas are not handed
    /// out. `0` disables the check (the default — every mounted segment is alive).
    pub fn set_segment_ttl(&mut self, ttl: u64) {
        self.segment_ttl = ttl;
    }

    /// Record a liveness heartbeat from a segment's node. Unknown segment → error.
    pub fn heartbeat(&mut self, segment: &str) -> Result<(), ErrorCode> {
        match self.segment_heartbeat.get_mut(segment) {
            Some(last) => {
                *last = self.clock;
                Ok(())
            }
            None => Err(ErrorCode::SegmentNotFound),
        }
    }

    /// Whether a mounted segment is alive (heartbeated within `segment_ttl`).
    /// With the check disabled (`segment_ttl == 0`), any mounted segment is alive.
    pub fn segment_alive(&self, segment: &str) -> bool {
        match self.segment_heartbeat.get(segment) {
            Some(&last) => {
                self.segment_ttl == 0 || self.clock.saturating_sub(last) <= self.segment_ttl
            }
            None => false,
        }
    }

    /// Mounted segments that have missed heartbeats past the TTL — failure
    /// detection surfaces them so the control plane can re-replicate / route away.
    pub fn dead_segments(&self) -> Vec<String> {
        self.allocators
            .iter()
            .map(|a| a.segment_name().to_string())
            .filter(|name| !self.segment_alive(name))
            .collect()
    }

    // ---- two-phase Put ----

    /// Phase 1: allocate `config.replica_num` replicas for `key` and return their
    /// buffers; the object is recorded `Initialized` (not yet readable).
    pub fn put_start(
        &mut self,
        key: ObjectKey,
        identity: IdentityScope,
        size: u64,
        config: &ReplicateConfig,
    ) -> Result<Vec<AllocatedBuffer>, ErrorCode> {
        if let Some(existing) = self.objects.get(&key) {
            if existing.has_complete_replica() {
                return Err(ErrorCode::ObjectAlreadyExists);
            }
        }
        // Reclaim any in-flight leftover for this key before re-allocating.
        if let Some(old) = self.objects.remove(&key) {
            self.free_replicas(&old.replicas);
        }

        let preferred = config.preferred_segment.as_deref();
        let buffers = self.allocate_replicas(size, config.replica_num, preferred)?;

        let mut replicas = ReplicaList::new();
        for buffer in &buffers {
            let id = self.next_replica_id;
            self.next_replica_id += 1;
            replicas.insert(id, Replica::new(id, ReplicaData::Memory(buffer.clone())));
        }
        let now = self.clock;
        self.objects.insert(
            key,
            ObjectMetadata {
                replicas,
                identity,
                lease_until: 0,
                last_access: now,
                soft_pinned: config.with_soft_pin,
                hard_pinned: config.with_hard_pin,
            },
        );
        Ok(buffers)
    }

    /// Phase 2: flip the object's replicas to `Complete` (readable).
    pub fn put_end(&mut self, key: &str) -> Result<(), ErrorCode> {
        let object = self.objects.get_mut(key).ok_or(ErrorCode::ObjectNotFound)?;
        for replica in object.replicas.values_mut() {
            replica.status = ReplicaStatus::Complete;
        }
        Ok(())
    }

    /// Abort an in-flight Put: free the allocation and drop the object.
    pub fn put_revoke(&mut self, key: &str) -> Result<(), ErrorCode> {
        let object = self.objects.remove(key).ok_or(ErrorCode::ObjectNotFound)?;
        self.free_replicas(&object.replicas);
        Ok(())
    }

    // ---- read path (identity-guarded) ----

    /// Return the object's complete replicas, **identity-guarded**, and grant a
    /// read lease. Refuses a requester whose identity doesn't match the writer's.
    pub fn get_replica_list(
        &mut self,
        key: &str,
        requester: &IdentityScope,
    ) -> Result<Vec<Replica>, ErrorCode> {
        let now = self.clock;
        let lease_ttl = self.lease_ttl;
        // Record this access in the frequency sketch (hot-key tracking).
        self.hotness.increment(key);
        // Failure detection: a Memory replica on a segment whose node stopped
        // heartbeating is treated as lost; a Disk replica is durable, so kept.
        let alive: HashSet<String> = self
            .allocators
            .iter()
            .map(|a| a.segment_name().to_string())
            .filter(|name| self.segment_alive(name))
            .collect();
        let object = self.objects.get_mut(key).ok_or(ErrorCode::ObjectNotFound)?;
        // QuillCache identity guard: a content-hash key can be requested under a
        // different identity — refuse cross-tenant / cross-adapter / cross-model.
        if let Some(violation) = object.identity.reuse_violation_against(requester) {
            return Err(ErrorCode::UnsafeReuse(violation));
        }
        let complete: Vec<Replica> = object
            .replicas
            .values()
            .filter(|r| r.is_complete())
            .filter(|r| match r.segment_name() {
                Some(seg) => alive.contains(seg),
                None => true,
            })
            .cloned()
            .collect();
        if complete.is_empty() {
            return Err(ErrorCode::ObjectNotReady);
        }
        object.last_access = now;
        object.lease_until = now + lease_ttl;
        Ok(complete)
    }

    // ---- batch APIs (Mooncake's BatchPut / BatchGet — one round-trip for many
    // keys; our connector offloads/loads a prefix's layers as one batch) ----

    /// Allocate replicas for many objects in one call. Transactional: if any key
    /// can't be allocated, the ones already started in this batch are revoked and
    /// the error is returned (no partial batch is left behind).
    pub fn batch_put_start(
        &mut self,
        items: Vec<(ObjectKey, IdentityScope, u64)>,
        config: &ReplicateConfig,
    ) -> Result<Vec<Vec<AllocatedBuffer>>, ErrorCode> {
        let mut out = Vec::with_capacity(items.len());
        let mut started: Vec<ObjectKey> = Vec::new();
        for (key, identity, size) in items {
            match self.put_start(key.clone(), identity, size, config) {
                Ok(buffers) => {
                    out.push(buffers);
                    started.push(key);
                }
                Err(e) => {
                    for k in &started {
                        let _ = self.put_revoke(k);
                    }
                    return Err(e);
                }
            }
        }
        Ok(out)
    }

    /// Commit many objects' replicas (flip to readable). Errors on the first key
    /// that isn't in flight.
    pub fn batch_put_end(&mut self, keys: &[String]) -> Result<(), ErrorCode> {
        for key in keys {
            self.put_end(key)?;
        }
        Ok(())
    }

    /// Revoke many in-flight Puts (free their allocations). Errors on the first
    /// key not in flight.
    pub fn batch_put_revoke(&mut self, keys: &[String]) -> Result<(), ErrorCode> {
        for key in keys {
            self.put_revoke(key)?;
        }
        Ok(())
    }

    // ---- upsert (Mooncake's UpsertStart/End/Revoke) ----

    /// Mooncake's `UpsertStart`. If the key is absent (or only in-flight) this is
    /// [`Self::put_start`]. If it exists complete at the **same** size, reuse its
    /// buffers in place — return them and flip the replicas back to `Initialized`
    /// so the client rewrites them (no re-allocation). If the size **differs**,
    /// free the old replicas and allocate new ones. Refused with
    /// [`ErrorCode::ObjectNotReady`] while a read lease is active (the object is
    /// busy), mirroring Mooncake's `OBJECT_REPLICA_BUSY`.
    pub fn upsert_start(
        &mut self,
        key: ObjectKey,
        identity: IdentityScope,
        size: u64,
        config: &ReplicateConfig,
    ) -> Result<Vec<AllocatedBuffer>, ErrorCode> {
        let (exists_complete, leased, existing_size) = match self.objects.get(&key) {
            None => (false, false, 0),
            Some(o) => (
                o.has_complete_replica(),
                o.lease_until > self.clock,
                o.replicas.values().next().map(|r| r.size()).unwrap_or(0),
            ),
        };
        if !exists_complete {
            // Absent or only an in-flight leftover — PutStart reclaims + allocates.
            return self.put_start(key, identity, size, config);
        }
        if leased {
            return Err(ErrorCode::ObjectNotReady);
        }
        if existing_size == size {
            // In-place: reuse the existing buffers, re-open for writing.
            let now = self.clock;
            let object = self.objects.get_mut(&key).expect("object present");
            object.identity = identity;
            object.last_access = now;
            let mut buffers = Vec::new();
            for replica in object.replicas.values_mut() {
                replica.status = ReplicaStatus::Initialized;
                if let ReplicaData::Memory(buffer) = &replica.data {
                    buffers.push(buffer.clone());
                }
            }
            Ok(buffers)
        } else {
            // Size changed: free the old replicas, then allocate fresh.
            let old = self.objects.remove(&key).expect("object present");
            self.free_replicas(&old.replicas);
            self.put_start(key, identity, size, config)
        }
    }

    /// Mooncake's `UpsertEnd` — same as committing a Put.
    pub fn upsert_end(&mut self, key: &str) -> Result<(), ErrorCode> {
        self.put_end(key)
    }

    /// Mooncake's `UpsertRevoke` — same as aborting a Put.
    pub fn upsert_revoke(&mut self, key: &str) -> Result<(), ErrorCode> {
        self.put_revoke(key)
    }

    /// Identity-guarded Get for many keys. Errors on the first key that is
    /// missing / not ready / refused (the connector wants all of a prefix's
    /// layers, or it recomputes the prefix).
    pub fn batch_get_replica_list(
        &mut self,
        keys: &[String],
        requester: &IdentityScope,
    ) -> Result<Vec<Vec<Replica>>, ErrorCode> {
        let mut out = Vec::with_capacity(keys.len());
        for key in keys {
            out.push(self.get_replica_list(key, requester)?);
        }
        Ok(out)
    }

    pub fn exist_key(&self, key: &str) -> bool {
        self.objects
            .get(key)
            .is_some_and(ObjectMetadata::has_complete_replica)
    }

    /// Existence for many keys in one call (Mooncake's `BatchExistKey`).
    pub fn batch_exist_key(&self, keys: &[String]) -> Vec<bool> {
        keys.iter().map(|k| self.exist_key(k)).collect()
    }

    /// Mooncake's `GetReplicaListByRegex`: every object whose key matches
    /// `pattern` and whose identity the requester is allowed to read, mapped to
    /// its complete replicas. Cross-identity matches are skipped (the guard), not
    /// errored — a bulk query returns what the caller may see.
    pub fn get_replica_list_by_regex(
        &mut self,
        pattern: &str,
        requester: &IdentityScope,
    ) -> Result<HashMap<String, Vec<Replica>>, ErrorCode> {
        let re = Regex::new(pattern).map_err(|e| ErrorCode::Io(format!("bad regex: {e}")))?;
        // Collect matching keys first so the &mut get_replica_list calls don't
        // alias the iteration over `objects`.
        let keys: Vec<String> = self
            .objects
            .keys()
            .filter(|k| re.is_match(k))
            .cloned()
            .collect();
        let mut out = HashMap::new();
        for key in keys {
            if let Ok(replicas) = self.get_replica_list(&key, requester) {
                out.insert(key, replicas);
            }
        }
        Ok(out)
    }

    /// Remove an object and free its replicas. Blocked while a read lease is
    /// active unless `force`.
    pub fn remove(&mut self, key: &str, force: bool) -> Result<(), ErrorCode> {
        let leased = {
            let object = self.objects.get(key).ok_or(ErrorCode::ObjectNotFound)?;
            object.lease_until > self.clock
        };
        if leased && !force {
            return Err(ErrorCode::ObjectNotReady);
        }
        let object = self.objects.remove(key).unwrap();
        self.free_replicas(&object.replicas);
        Ok(())
    }

    // ---- observability ----

    pub fn segment_count(&self) -> usize {
        self.allocators.len()
    }
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }
    pub fn capacity(&self) -> u64 {
        self.allocators.iter().map(|a| a.capacity()).sum()
    }
    pub fn allocated(&self) -> u64 {
        self.allocators.iter().map(|a| a.allocated()).sum()
    }

    /// Approximate access frequency for `key` (Mooncake's CountMinSketch
    /// estimate) — how hot the key is, for frequency-aware eviction / promotion.
    pub fn hotness(&self, key: &str) -> u8 {
        self.hotness.count(key)
    }

    // ---- HA: snapshot + recovery (Mooncake's metadata snapshot thread) ----

    /// Take a consistent [`MasterSnapshot`] of the current in-memory metadata.
    pub fn snapshot(&self) -> MasterSnapshot {
        MasterSnapshot {
            version: 1,
            strategy: self.strategy_name.clone(),
            clock: self.clock,
            lease_ttl: self.lease_ttl,
            high_watermark: self.high_watermark,
            eviction_ratio: self.eviction_ratio,
            segment_ttl: self.segment_ttl,
            next_replica_id: self.next_replica_id,
            segments: self
                .allocators
                .iter()
                .map(|a| SegmentSnapshot {
                    name: a.segment_name().to_string(),
                    capacity: a.capacity(),
                })
                .collect(),
            objects: self
                .objects
                .iter()
                .map(|(key, o)| ObjectSnapshot {
                    key: key.clone(),
                    replicas: o.replicas.values().cloned().collect(),
                    identity: o.identity.clone(),
                    lease_until: o.lease_until,
                    last_access: o.last_access,
                    soft_pinned: o.soft_pinned,
                    hard_pinned: o.hard_pinned,
                })
                .collect(),
        }
    }

    /// Rebuild a master from a snapshot: re-mount the segments and re-reserve each
    /// replica's exact `(offset, size)` so the allocator layout matches, then
    /// restore objects, leases, pins, and the clock.
    pub fn recover(snapshot: MasterSnapshot) -> Result<Self, ErrorCode> {
        let mut master = MasterService::new(&snapshot.strategy);
        master.clock = snapshot.clock;
        master.lease_ttl = snapshot.lease_ttl;
        master.high_watermark = snapshot.high_watermark;
        master.eviction_ratio = snapshot.eviction_ratio;
        master.segment_ttl = snapshot.segment_ttl;
        for seg in &snapshot.segments {
            master.mount_segment(seg.name.clone(), seg.capacity);
        }
        for obj in snapshot.objects {
            // Re-reserve each Memory replica's exact range so the allocator's
            // free-list reflects the recovered layout (Disk replicas are durable).
            for replica in &obj.replicas {
                if let ReplicaData::Memory(buf) = &replica.data {
                    let allocator = master
                        .allocators
                        .iter_mut()
                        .find(|a| a.segment_name() == buf.segment_name())
                        .ok_or(ErrorCode::SegmentNotFound)?;
                    if !allocator.reserve(buf.offset, buf.size) {
                        return Err(ErrorCode::InvalidReplica);
                    }
                }
            }
            let mut replicas = ReplicaList::new();
            for replica in obj.replicas {
                replicas.insert(replica.id, replica);
            }
            master.objects.insert(
                obj.key,
                ObjectMetadata {
                    replicas,
                    identity: obj.identity,
                    lease_until: obj.lease_until,
                    last_access: obj.last_access,
                    soft_pinned: obj.soft_pinned,
                    hard_pinned: obj.hard_pinned,
                },
            );
        }
        master.next_replica_id = snapshot.next_replica_id;
        Ok(master)
    }

    /// Persist a snapshot to `path` **atomically** — write a temp file, then
    /// rename — so a crash mid-write never leaves a torn snapshot (the QuillCache
    /// crash-consistency discipline applied to the master's metadata).
    pub fn save_snapshot(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let path = path.as_ref();
        let bytes = serde_json::to_vec(&self.snapshot())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let tmp = path.with_extension("snapshot.tmp");
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Recover a master from a snapshot file written by [`MasterService::save_snapshot`].
    pub fn load_snapshot(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let snapshot: MasterSnapshot = serde_json::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Self::recover(snapshot)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{e:?}")))
    }

    // ---- internals ----

    /// Allocate `replica_num` buffers of `size`; on segment exhaustion, evict the
    /// coldest unpinned objects to make room and retry once (Mooncake evicts at a
    /// high watermark or on put failure).
    fn allocate_replicas(
        &mut self,
        size: u64,
        replica_num: usize,
        preferred: Option<&str>,
    ) -> Result<Vec<AllocatedBuffer>, ErrorCode> {
        self.evict_if_needed();
        match self
            .strategy
            .allocate(&mut self.allocators, size, replica_num, preferred, &[])
        {
            Ok(buffers) => Ok(buffers),
            Err(ErrorCode::NoAvailableSegment) => {
                self.evict_to_fit(size.saturating_mul(replica_num as u64));
                self.strategy
                    .allocate(&mut self.allocators, size, replica_num, preferred, &[])
            }
            Err(other) => Err(other),
        }
    }

    fn free_replicas(&mut self, replicas: &ReplicaList) {
        for replica in replicas.values() {
            if let ReplicaData::Memory(buffer) = &replica.data {
                if let Some(allocator) = self
                    .allocators
                    .iter_mut()
                    .find(|a| a.segment_name() == buffer.segment_name)
                {
                    allocator.deallocate(buffer);
                }
            }
        }
    }

    /// Eviction candidates, coldest first; non-soft-pinned before soft-pinned;
    /// hard-pinned and currently-leased objects are never candidates.
    fn victims_coldest_first(&self) -> Vec<ObjectKey> {
        let now = self.clock;
        let mut victims: Vec<(ObjectKey, u64, bool)> = self
            .objects
            .iter()
            .filter(|(_, o)| !o.hard_pinned && o.lease_until <= now)
            .map(|(k, o)| (k.clone(), o.last_access, o.soft_pinned))
            .collect();
        victims.sort_by(|a, b| a.2.cmp(&b.2).then(a.1.cmp(&b.1)));
        victims.into_iter().map(|(k, _, _)| k).collect()
    }

    /// Proactive watermark eviction: when usage exceeds the high watermark, evict
    /// the coldest unpinned objects down below the target.
    pub fn evict_if_needed(&mut self) -> usize {
        let capacity = self.capacity();
        if capacity == 0 || (self.allocated() as f64) < self.high_watermark * capacity as f64 {
            return 0;
        }
        let target = (self.high_watermark * (1.0 - self.eviction_ratio) * capacity as f64) as u64;
        let mut evicted = 0;
        for key in self.victims_coldest_first() {
            if self.allocated() <= target {
                break;
            }
            if let Some(object) = self.objects.remove(&key) {
                self.free_replicas(&object.replicas);
                evicted += 1;
            }
        }
        evicted
    }

    /// On-demand eviction: evict coldest unpinned objects until at least `needed`
    /// bytes are free cluster-wide (best effort).
    fn evict_to_fit(&mut self, needed: u64) -> usize {
        let mut evicted = 0;
        for key in self.victims_coldest_first() {
            if self.capacity().saturating_sub(self.allocated()) >= needed {
                break;
            }
            if let Some(object) = self.objects.remove(&key) {
                self.free_replicas(&object.replicas);
                evicted += 1;
            }
        }
        evicted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quillcache_core::ReuseViolation;

    fn scope(tenant: &str) -> IdentityScope {
        IdentityScope {
            model_id: "m".into(),
            tokenizer_id: "t".into(),
            adapter_id: None,
            tenant_id: tenant.into(),
        }
    }

    #[test]
    fn two_phase_put_then_get_with_replication() {
        let mut m = MasterService::new("random");
        m.mount_segment("seg-0", 100);
        m.mount_segment("seg-1", 100);
        let id = scope("ten-a");

        // Phase 1: two replicas on distinct segments.
        let buffers = m
            .put_start("k".into(), id.clone(), 16, &ReplicateConfig::replicas(2))
            .unwrap();
        assert_eq!(buffers.len(), 2);
        assert_ne!(buffers[0].segment_name, buffers[1].segment_name);
        // Not readable until put_end.
        assert_eq!(m.get_replica_list("k", &id), Err(ErrorCode::ObjectNotReady));

        // Phase 2: now readable.
        m.put_end("k").unwrap();
        let replicas = m.get_replica_list("k", &id).unwrap();
        assert_eq!(replicas.len(), 2);
        assert!(replicas.iter().all(|r| r.is_complete()));
        assert!(m.exist_key("k"));
    }

    #[test]
    fn get_is_identity_guarded() {
        let mut m = MasterService::new("random");
        m.mount_segment("seg-0", 100);
        // Same content-hash key, written by tenant-a.
        m.put_start(
            "hot".into(),
            scope("ten-a"),
            10,
            &ReplicateConfig::replicas(1),
        )
        .unwrap();
        m.put_end("hot").unwrap();
        // tenant-b asks for the same key → refused (a prefix-cache privacy leak).
        assert_eq!(
            m.get_replica_list("hot", &scope("ten-b")),
            Err(ErrorCode::UnsafeReuse(ReuseViolation::Tenant))
        );
        // tenant-a (the writer) gets it.
        assert_eq!(m.get_replica_list("hot", &scope("ten-a")).unwrap().len(), 1);
    }

    #[test]
    fn put_revoke_frees_the_allocation() {
        let mut m = MasterService::new("random");
        m.mount_segment("seg-0", 100);
        m.put_start(
            "k".into(),
            scope("ten-a"),
            40,
            &ReplicateConfig::replicas(1),
        )
        .unwrap();
        assert_eq!(m.allocated(), 40);
        m.put_revoke("k").unwrap();
        assert_eq!(m.allocated(), 0);
        assert!(!m.exist_key("k"));
    }

    #[test]
    fn eviction_makes_room_under_pressure_and_keeps_hot_and_pinned() {
        let mut m = MasterService::new("random");
        m.mount_segment("seg-0", 100);
        let id = scope("ten-a");

        // A at t0 (coldest), B at t1.
        m.put_start("A".into(), id.clone(), 40, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("A").unwrap();
        m.tick();
        m.put_start("B".into(), id.clone(), 40, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("B").unwrap();
        assert_eq!(m.allocated(), 80);

        // C (40) doesn't fit (20 free) → evict the coldest (A) to make room.
        m.tick();
        m.put_start("C".into(), id.clone(), 40, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("C").unwrap();
        assert!(!m.exist_key("A"), "coldest object A should be evicted");
        assert!(m.exist_key("B") && m.exist_key("C"));
    }

    #[test]
    fn hard_pinned_object_is_never_evicted() {
        let mut m = MasterService::new("random");
        m.mount_segment("seg-0", 100);
        let id = scope("ten-a");
        // A is hard-pinned even though it is the coldest.
        let pinned = ReplicateConfig {
            with_hard_pin: true,
            ..ReplicateConfig::replicas(1)
        };
        m.put_start("A".into(), id.clone(), 40, &pinned).unwrap();
        m.put_end("A").unwrap();
        m.tick();
        m.put_start("B".into(), id.clone(), 40, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("B").unwrap();
        // C needs room: B (the only unpinned victim) is evicted, A survives.
        m.tick();
        m.put_start("C".into(), id.clone(), 40, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("C").unwrap();
        assert!(m.exist_key("A"), "hard-pinned A must survive");
        assert!(!m.exist_key("B"));
        assert!(m.exist_key("C"));
    }

    #[test]
    fn watermark_eviction_frees_down_to_target() {
        let mut m = MasterService::new("random");
        m.mount_segment("seg-0", 100);
        let id = scope("ten-a");
        for i in 0..10 {
            let k = format!("k{i}");
            m.put_start(k.clone(), id.clone(), 10, &ReplicateConfig::replicas(1))
                .unwrap();
            m.put_end(&k).unwrap();
            m.tick();
        }
        assert_eq!(m.allocated(), 100); // full, over the 95% watermark
        let evicted = m.evict_if_needed();
        assert!(evicted >= 2);
        assert_eq!(m.allocated(), 80); // freed down below target (85)
        assert!(!m.exist_key("k0") && !m.exist_key("k1")); // the two coldest
        assert!(m.exist_key("k2"));
    }

    #[test]
    fn read_lease_blocks_remove_until_it_expires() {
        let mut m = MasterService::new("random");
        m.mount_segment("seg-0", 100);
        let id = scope("ten-a");
        m.put_start("A".into(), id.clone(), 10, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("A").unwrap();
        // A get grants a lease (ttl 5).
        m.get_replica_list("A", &id).unwrap();
        assert_eq!(m.remove("A", false), Err(ErrorCode::ObjectNotReady));
        // Force ignores the lease.
        // (don't actually remove yet — test lease expiry path instead)
        for _ in 0..6 {
            m.tick();
        }
        assert!(m.remove("A", false).is_ok());
        assert!(!m.exist_key("A"));
    }

    #[test]
    fn snapshot_recovers_objects_segments_and_allocator_state() {
        let mut m = MasterService::new("random");
        m.mount_segment("seg-0", 1000);
        m.mount_segment("seg-1", 1000);
        let id = scope("ten-a");
        m.put_start("a".into(), id.clone(), 64, &ReplicateConfig::replicas(2))
            .unwrap();
        m.put_end("a").unwrap();
        m.put_start("b".into(), id.clone(), 128, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("b").unwrap();
        let allocated_before = m.allocated();

        // Round-trip through a snapshot (e.g. a restart / leader failover).
        let mut r = MasterService::recover(m.snapshot()).expect("recover");
        assert_eq!(r.object_count(), 2);
        assert_eq!(r.segment_count(), 2);
        assert_eq!(
            r.allocated(),
            allocated_before,
            "the allocator's allocated bytes are rebuilt exactly"
        );

        // Recovered objects are readable, still identity-guarded.
        assert_eq!(r.get_replica_list("a", &id).unwrap().len(), 2);
        assert!(matches!(
            r.get_replica_list("a", &scope("ten-b")),
            Err(ErrorCode::UnsafeReuse(_))
        ));

        // The rebuilt allocator won't hand out the recovered ranges again: a fresh
        // Put succeeds without overlapping the reserved offsets.
        r.put_start("c".into(), id.clone(), 64, &ReplicateConfig::replicas(1))
            .unwrap();
        r.put_end("c").unwrap();
        assert!(r.get_replica_list("c", &id).is_ok());
    }

    #[test]
    fn snapshot_file_round_trip_is_atomic() {
        let mut m = MasterService::new("random");
        m.mount_segment("seg-0", 1000);
        let id = scope("ten-a");
        m.put_start("k".into(), id.clone(), 64, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("k").unwrap();

        let path = std::env::temp_dir().join(format!("qc-master-snap-{}.json", std::process::id()));
        m.save_snapshot(&path).unwrap();
        let mut r = MasterService::load_snapshot(&path).unwrap();
        assert_eq!(r.get_replica_list("k", &id).unwrap().len(), 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn heartbeat_health_hides_replicas_on_a_dead_segment() {
        let mut m = MasterService::new("random");
        m.set_segment_ttl(5); // enable failure detection
        m.mount_segment("seg-0", 1000);
        let id = scope("ten-a");
        m.put_start("k".into(), id.clone(), 64, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("k").unwrap();
        assert!(m.segment_alive("seg-0"));
        assert!(m.get_replica_list("k", &id).is_ok());

        // Miss heartbeats past the TTL → the segment is dead and its only replica
        // is treated as lost, so the object becomes unservable.
        for _ in 0..6 {
            m.tick();
        }
        assert!(!m.segment_alive("seg-0"));
        assert_eq!(m.dead_segments(), vec!["seg-0".to_string()]);
        assert_eq!(m.get_replica_list("k", &id), Err(ErrorCode::ObjectNotReady));

        // A heartbeat brings the node back and its replica is served again.
        m.heartbeat("seg-0").unwrap();
        assert!(m.segment_alive("seg-0"));
        assert!(m.get_replica_list("k", &id).is_ok());
    }

    #[test]
    fn batch_put_then_batch_get_round_trips() {
        let mut m = MasterService::new("random");
        m.mount_segment("seg-0", 4096);
        let id = scope("ten-a");
        let keys: Vec<String> = vec!["k0".into(), "k1".into(), "k2".into()];
        let items = keys.iter().map(|k| (k.clone(), id.clone(), 64)).collect();

        let buffers = m
            .batch_put_start(items, &ReplicateConfig::replicas(1))
            .unwrap();
        assert_eq!(buffers.len(), 3);
        m.batch_put_end(&keys).unwrap();

        let got = m.batch_get_replica_list(&keys, &id).unwrap();
        assert_eq!(got.len(), 3);
        assert!(got.iter().all(|replicas| replicas.len() == 1));

        // The identity guard applies to the batch too.
        assert!(matches!(
            m.batch_get_replica_list(&["k0".into()], &scope("ten-b")),
            Err(ErrorCode::UnsafeReuse(_))
        ));
    }

    #[test]
    fn batch_put_start_rolls_back_on_failure() {
        let mut m = MasterService::new("random");
        m.mount_segment("seg-0", 100);
        let id = scope("ten-a");
        // The second object (200B) can't fit a 100B segment → the batch fails and
        // the first object's allocation is rolled back (no partial batch).
        let items = vec![
            ("a".to_string(), id.clone(), 64),
            ("b".to_string(), id, 200),
        ];
        assert!(m
            .batch_put_start(items, &ReplicateConfig::replicas(1))
            .is_err());
        assert_eq!(m.object_count(), 0);
        assert_eq!(m.allocated(), 0);
    }

    #[test]
    fn upsert_reuses_buffers_in_place_when_size_unchanged() {
        let mut m = MasterService::new("random");
        m.mount_segment("seg-0", 1000);
        let id = scope("ten-a");
        let bufs = m
            .put_start("k".into(), id.clone(), 64, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("k").unwrap();
        let off = bufs[0].offset;
        let used = m.allocated();
        // Same size → reuse the buffer in place, no extra allocation.
        let bufs2 = m
            .upsert_start("k".into(), id.clone(), 64, &ReplicateConfig::replicas(1))
            .unwrap();
        assert_eq!(bufs2[0].offset, off, "in-place upsert reuses the buffer");
        assert_eq!(
            m.allocated(),
            used,
            "no re-allocation for a same-size upsert"
        );
        assert!(!m.exist_key("k"), "re-opened for writing until committed");
        m.upsert_end("k").unwrap();
        assert!(m.exist_key("k"));
    }

    #[test]
    fn upsert_reallocates_when_size_changes() {
        let mut m = MasterService::new("random");
        m.mount_segment("seg-0", 1000);
        let id = scope("ten-a");
        m.put_start("k".into(), id.clone(), 40, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("k").unwrap();
        assert_eq!(m.allocated(), 40);
        m.upsert_start("k".into(), id.clone(), 100, &ReplicateConfig::replicas(1))
            .unwrap();
        m.upsert_end("k").unwrap();
        assert_eq!(m.allocated(), 100, "size change frees old + allocates new");
    }

    #[test]
    fn upsert_is_refused_while_leased() {
        let mut m = MasterService::new("random");
        m.mount_segment("seg-0", 1000);
        let id = scope("ten-a");
        m.put_start("k".into(), id.clone(), 64, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("k").unwrap();
        m.get_replica_list("k", &id).unwrap(); // grants a read lease → busy
        assert_eq!(
            m.upsert_start("k".into(), id.clone(), 64, &ReplicateConfig::replicas(1)),
            Err(ErrorCode::ObjectNotReady)
        );
    }

    #[test]
    fn get_replica_list_by_regex_matches_allowed_keys_only() {
        let mut m = MasterService::new("random");
        m.mount_segment("seg-0", 1000);
        let a = scope("ten-a");
        let b = scope("ten-b");
        for k in ["qc/a/1", "qc/a/2", "other"] {
            m.put_start(k.into(), a.clone(), 16, &ReplicateConfig::replicas(1))
                .unwrap();
            m.put_end(k).unwrap();
        }
        // Same-pattern key under a different tenant must NOT leak to tenant-a.
        m.put_start("qc/a/secret".into(), b, 16, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("qc/a/secret").unwrap();

        let got = m.get_replica_list_by_regex("^qc/a/.*", &a).unwrap();
        let mut keys: Vec<&str> = got.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["qc/a/1", "qc/a/2"]);
    }

    #[test]
    fn batch_exist_key_reports_committed_only() {
        let mut m = MasterService::new("random");
        m.mount_segment("seg-0", 1000);
        let id = scope("ten-a");
        let items = vec![
            ("a".to_string(), id.clone(), 16u64),
            ("b".to_string(), id.clone(), 16u64),
        ];
        m.batch_put_start(items, &ReplicateConfig::replicas(1))
            .unwrap();
        assert_eq!(
            m.batch_exist_key(&["a".into(), "b".into()]),
            vec![false, false]
        );
        m.batch_put_end(&["a".into(), "b".into()]).unwrap();
        assert_eq!(
            m.batch_exist_key(&["a".into(), "b".into(), "z".into()]),
            vec![true, true, false]
        );
    }

    #[test]
    fn batch_put_revoke_frees_inflight() {
        let mut m = MasterService::new("random");
        m.mount_segment("seg-0", 1000);
        let id = scope("ten-a");
        let items = vec![
            ("a".to_string(), id.clone(), 16u64),
            ("b".to_string(), id, 16u64),
        ];
        m.batch_put_start(items, &ReplicateConfig::replicas(1))
            .unwrap();
        m.batch_put_revoke(&["a".into(), "b".into()]).unwrap();
        assert_eq!(m.allocated(), 0);
        assert_eq!(m.object_count(), 0);
    }

    #[test]
    fn guarded_reads_bump_key_hotness() {
        let mut m = MasterService::new("random");
        m.mount_segment("seg-0", 1000);
        let id = scope("ten-a");
        m.put_start("hot".into(), id.clone(), 16, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("hot").unwrap();
        assert_eq!(m.hotness("hot"), 0);
        for _ in 0..5 {
            m.get_replica_list("hot", &id).unwrap();
        }
        assert_eq!(m.hotness("hot"), 5, "each guarded read bumps the sketch");
        assert_eq!(m.hotness("cold"), 0);
    }
}
