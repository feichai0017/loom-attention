//! Buffer allocator (Mooncake's `mooncake-store/include/allocator.h` +
//! `offset_allocator/`). A mounted RAM segment is a fixed-size byte arena; the
//! allocator hands out offset ranges within it. [`OffsetBufferAllocator`] is the
//! default backend — Mooncake's is an O(1) size-binned offset allocator; this is
//! a first-fit free-list with coalescing-on-free, the same `allocate` /
//! `deallocate` / `largest_free_region` contract with simpler internals.

use crate::types::SegmentName;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A buffer allocated from a segment (Mooncake's `AllocatedBuffer`): a
/// `(segment, offset, size)` triple naming where an object's bytes live so the
/// transfer engine can address them.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AllocatedBuffer {
    pub segment_name: SegmentName,
    pub offset: u64,
    pub size: u64,
}

impl AllocatedBuffer {
    pub fn segment_name(&self) -> &str {
        &self.segment_name
    }
}

/// A per-segment buffer allocator (Mooncake's `BufferAllocatorBase`).
pub trait BufferAllocator: std::fmt::Debug + Send + Sync {
    fn segment_name(&self) -> &str;
    fn capacity(&self) -> u64;
    /// Bytes currently handed out.
    fn allocated(&self) -> u64;
    /// Allocate `size` bytes; `None` if no free region is large enough.
    fn allocate(&mut self, size: u64) -> Option<AllocatedBuffer>;
    /// Return a buffer's range to the free list (coalescing with neighbours).
    fn deallocate(&mut self, buffer: &AllocatedBuffer);
    /// The largest single contiguous free region (the biggest allocatable size).
    fn largest_free_region(&self) -> u64;
}

/// First-fit offset allocator over a sorted free-list of `offset -> length`
/// regions, coalescing adjacent free ranges on `deallocate`.
#[derive(Debug)]
pub struct OffsetBufferAllocator {
    segment_name: SegmentName,
    capacity: u64,
    /// Non-overlapping free regions, keyed by offset (so neighbours are adjacent).
    free: BTreeMap<u64, u64>,
    allocated: u64,
}

impl OffsetBufferAllocator {
    pub fn new(segment_name: impl Into<SegmentName>, capacity: u64) -> Self {
        let mut free = BTreeMap::new();
        if capacity > 0 {
            free.insert(0, capacity);
        }
        Self {
            segment_name: segment_name.into(),
            capacity,
            free,
            allocated: 0,
        }
    }
}

impl BufferAllocator for OffsetBufferAllocator {
    fn segment_name(&self) -> &str {
        &self.segment_name
    }

    fn capacity(&self) -> u64 {
        self.capacity
    }

    fn allocated(&self) -> u64 {
        self.allocated
    }

    fn allocate(&mut self, size: u64) -> Option<AllocatedBuffer> {
        if size == 0 {
            return None;
        }
        // First-fit: the first free region large enough.
        let (offset, region_len) = self
            .free
            .iter()
            .find(|(_, &len)| len >= size)
            .map(|(&off, &len)| (off, len))?;
        self.free.remove(&offset);
        if region_len > size {
            // The remainder stays free, starting just past the allocation.
            self.free.insert(offset + size, region_len - size);
        }
        self.allocated += size;
        Some(AllocatedBuffer {
            segment_name: self.segment_name.clone(),
            offset,
            size,
        })
    }

    fn deallocate(&mut self, buffer: &AllocatedBuffer) {
        let mut start = buffer.offset;
        let mut end = buffer.offset + buffer.size;
        // Coalesce with the region immediately before, if it abuts us.
        let prev = self.free.range(..start).next_back().map(|(&o, &l)| (o, l));
        if let Some((p_off, p_len)) = prev {
            if p_off + p_len == start {
                start = p_off;
                self.free.remove(&p_off);
            }
        }
        // Coalesce with the region immediately after, if it abuts us.
        let next = self.free.range(end..).next().map(|(&o, &l)| (o, l));
        if let Some((n_off, n_len)) = next {
            if n_off == end {
                end = n_off + n_len;
                self.free.remove(&n_off);
            }
        }
        self.free.insert(start, end - start);
        self.allocated = self.allocated.saturating_sub(buffer.size);
    }

    fn largest_free_region(&self) -> u64 {
        self.free.values().copied().max().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_split_and_fail_when_too_fragmented() {
        let mut a = OffsetBufferAllocator::new("seg-0", 100);
        let b1 = a.allocate(40).unwrap();
        assert_eq!(b1.offset, 0);
        let b2 = a.allocate(40).unwrap();
        assert_eq!(b2.offset, 40);
        assert_eq!(a.allocated(), 80);
        assert_eq!(a.largest_free_region(), 20);
        // Only 20 contiguous bytes left — a 30-byte request can't be served.
        assert!(a.allocate(30).is_none());
    }

    #[test]
    fn deallocate_coalesces_adjacent_free_regions() {
        let mut a = OffsetBufferAllocator::new("seg-0", 100);
        let b1 = a.allocate(50).unwrap(); // [0,50)
        let b2 = a.allocate(50).unwrap(); // [50,100) — segment full
        assert_eq!(a.largest_free_region(), 0);
        // Free both out of order; the two halves must coalesce back to one 100B
        // region, not stay as two 50B fragments.
        a.deallocate(&b2);
        a.deallocate(&b1);
        assert_eq!(a.allocated(), 0);
        assert_eq!(a.largest_free_region(), 100);
        // And a full-size allocation succeeds again.
        assert_eq!(a.allocate(100).unwrap().offset, 0);
    }
}
