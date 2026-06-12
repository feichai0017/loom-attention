//! Allocation strategy (Mooncake's `mooncake-store/include/allocation_strategy.h`):
//! given the per-segment [`BufferAllocator`]s, choose which segment(s) to place
//! an object's replicas on. Each replica lands on a **distinct** segment.
//!
//! - [`RandomAllocationStrategy`] (Mooncake's default) — spread across segments.
//! - [`FreeRatioFirstAllocationStrategy`] — prefer the emptiest segments.
//!
//! (Mooncake's `Random` shuffles for load spread; this port picks distinct
//! candidates in a deterministic order so the placement is testable — the
//! invariant that matters, one segment per replica, is identical.)

use crate::allocator::{AllocatedBuffer, BufferAllocator};
use crate::types::ErrorCode;

/// Choose segments and allocate one buffer of `size` per replica.
pub trait AllocationStrategy: std::fmt::Debug + Send + Sync {
    fn name(&self) -> &str;

    /// Allocate `replica_num` buffers of `size`, each on a distinct segment,
    /// honouring `preferred` (placed first) and `excluded` (skipped). Fails with
    /// [`ErrorCode::NoAvailableSegment`] if fewer than `replica_num` segments can
    /// fit it.
    fn allocate(
        &self,
        allocators: &mut [Box<dyn BufferAllocator>],
        size: u64,
        replica_num: usize,
        preferred: Option<&str>,
        excluded: &[String],
    ) -> Result<Vec<AllocatedBuffer>, ErrorCode>;
}

/// Segment indices that aren't excluded and can fit `size`.
fn candidates(
    allocators: &[Box<dyn BufferAllocator>],
    size: u64,
    excluded: &[String],
) -> Vec<usize> {
    (0..allocators.len())
        .filter(|&i| {
            let a = &allocators[i];
            a.largest_free_region() >= size && !excluded.iter().any(|e| e == a.segment_name())
        })
        .collect()
}

fn free_ratio(a: &dyn BufferAllocator) -> f64 {
    if a.capacity() == 0 {
        0.0
    } else {
        (a.capacity() - a.allocated()) as f64 / a.capacity() as f64
    }
}

/// Allocate one buffer of `size` on the first `replica_num` of `ordered`.
fn place(
    allocators: &mut [Box<dyn BufferAllocator>],
    ordered: &[usize],
    size: u64,
    replica_num: usize,
) -> Result<Vec<AllocatedBuffer>, ErrorCode> {
    if ordered.len() < replica_num {
        return Err(ErrorCode::NoAvailableSegment);
    }
    let mut out = Vec::with_capacity(replica_num);
    for &i in ordered.iter().take(replica_num) {
        out.push(
            allocators[i]
                .allocate(size)
                .ok_or(ErrorCode::NoAvailableSegment)?,
        );
    }
    Ok(out)
}

#[derive(Debug, Default)]
pub struct RandomAllocationStrategy;

impl AllocationStrategy for RandomAllocationStrategy {
    fn name(&self) -> &str {
        "random"
    }

    fn allocate(
        &self,
        allocators: &mut [Box<dyn BufferAllocator>],
        size: u64,
        replica_num: usize,
        preferred: Option<&str>,
        excluded: &[String],
    ) -> Result<Vec<AllocatedBuffer>, ErrorCode> {
        let mut order = candidates(allocators, size, excluded);
        if let Some(pref) = preferred {
            // Stable sort: the preferred segment floats to the front, rest hold order.
            order.sort_by_key(|&i| allocators[i].segment_name() != pref);
        }
        place(allocators, &order, size, replica_num)
    }
}

#[derive(Debug, Default)]
pub struct FreeRatioFirstAllocationStrategy;

impl AllocationStrategy for FreeRatioFirstAllocationStrategy {
    fn name(&self) -> &str {
        "free_ratio_first"
    }

    fn allocate(
        &self,
        allocators: &mut [Box<dyn BufferAllocator>],
        size: u64,
        replica_num: usize,
        preferred: Option<&str>,
        excluded: &[String],
    ) -> Result<Vec<AllocatedBuffer>, ErrorCode> {
        let mut order = candidates(allocators, size, excluded);
        // Emptiest segment first (descending free ratio).
        order.sort_by(|&a, &b| {
            free_ratio(allocators[b].as_ref())
                .partial_cmp(&free_ratio(allocators[a].as_ref()))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(pref) = preferred {
            order.sort_by_key(|&i| allocators[i].segment_name() != pref);
        }
        place(allocators, &order, size, replica_num)
    }
}

/// Build a strategy from its config name (Mooncake's `CreateAllocationStrategy`).
pub fn create_allocation_strategy(name: &str) -> Box<dyn AllocationStrategy> {
    match name {
        "free_ratio_first" => Box::new(FreeRatioFirstAllocationStrategy),
        _ => Box::new(RandomAllocationStrategy),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::OffsetBufferAllocator;

    fn fleet() -> Vec<Box<dyn BufferAllocator>> {
        vec![
            Box::new(OffsetBufferAllocator::new("seg-0", 100)),
            Box::new(OffsetBufferAllocator::new("seg-1", 100)),
            Box::new(OffsetBufferAllocator::new("seg-2", 100)),
        ]
    }

    #[test]
    fn each_replica_lands_on_a_distinct_segment() {
        let mut a = fleet();
        let bufs = RandomAllocationStrategy
            .allocate(&mut a, 10, 2, None, &[])
            .unwrap();
        assert_eq!(bufs.len(), 2);
        assert_ne!(bufs[0].segment_name, bufs[1].segment_name);
    }

    #[test]
    fn not_enough_segments_is_an_error() {
        let mut a = fleet();
        // 4 replicas, only 3 segments → cannot place.
        assert_eq!(
            RandomAllocationStrategy.allocate(&mut a, 10, 4, None, &[]),
            Err(ErrorCode::NoAvailableSegment)
        );
    }

    #[test]
    fn excluded_segment_is_skipped() {
        let mut a = fleet();
        let bufs = RandomAllocationStrategy
            .allocate(&mut a, 10, 2, None, &["seg-1".to_string()])
            .unwrap();
        assert!(bufs.iter().all(|b| b.segment_name != "seg-1"));
    }

    #[test]
    fn free_ratio_first_prefers_the_emptiest_segments() {
        let mut a = fleet();
        // Fill seg-0 most, seg-1 a little; seg-2 stays empty.
        a[0].allocate(90);
        a[1].allocate(30);
        let bufs = FreeRatioFirstAllocationStrategy
            .allocate(&mut a, 10, 2, None, &[])
            .unwrap();
        let picked: Vec<&str> = bufs.iter().map(|b| b.segment_name.as_str()).collect();
        // The two emptiest (seg-2, seg-1) are chosen; the near-full seg-0 is not.
        assert!(picked.contains(&"seg-2") && picked.contains(&"seg-1"));
        assert!(!picked.contains(&"seg-0"));
    }
}
