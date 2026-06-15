//! Buffer allocator (Mooncake's `mooncake-store/include/allocator.h` +
//! `offset_allocator/`). A mounted RAM segment is a fixed-size byte arena; the
//! allocator hands out offset ranges within it. The store's default is the
//! faithful Aaltonen port in [`crate::offset_allocator`] (Mooncake's
//! `OffsetBufferAllocator`); these are two simpler alternatives behind the
//! [`BufferAllocator`] trait:
//! - [`FirstFitBufferAllocator`] — a first-fit free-list with coalescing-on-free
//!   (simple; alloc scans the free list — O(n)).
//! - [`BinnedBufferAllocator`] — a 64-bin power-of-two approximation of the
//!   offset allocator (O(1) bin-select via a `u64` bitmask, but the floor bin is
//!   scanned for a fit; superseded by the faithful 256-bin port).

use crate::types::SegmentName;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

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
    /// Reserve an exact `(offset, size)` range whose layout is already known —
    /// used to rebuild allocator state from a snapshot on master recovery.
    /// Returns false if the range is not entirely free.
    fn reserve(&mut self, offset: u64, size: u64) -> bool;
}

/// First-fit offset allocator over a sorted free-list of `offset -> length`
/// regions, coalescing adjacent free ranges on `deallocate`.
#[derive(Debug)]
pub struct FirstFitBufferAllocator {
    segment_name: SegmentName,
    capacity: u64,
    /// Non-overlapping free regions, keyed by offset (so neighbours are adjacent).
    free: BTreeMap<u64, u64>,
    allocated: u64,
}

impl FirstFitBufferAllocator {
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

impl BufferAllocator for FirstFitBufferAllocator {
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

    fn reserve(&mut self, offset: u64, size: u64) -> bool {
        if size == 0 {
            return false;
        }
        let end = offset + size;
        // The free region containing [offset, end): the one starting at or before
        // `offset` whose extent covers `end`.
        let containing = self
            .free
            .range(..=offset)
            .next_back()
            .map(|(&o, &l)| (o, l));
        let (r_off, r_len) = match containing {
            Some((o, l)) if o + l >= end => (o, l),
            _ => return false,
        };
        self.free.remove(&r_off);
        if offset > r_off {
            self.free.insert(r_off, offset - r_off); // free remainder before
        }
        if r_off + r_len > end {
            self.free.insert(end, (r_off + r_len) - end); // free remainder after
        }
        self.allocated += size;
        true
    }
}

/// Size-binned offset allocator — Mooncake's O(1)-style design. Free regions are
/// bucketed by size class (bin `k` holds regions of size in `[2^k, 2^(k+1))`); a
/// `u64` bitmask of non-empty bins finds the smallest fitting bin in O(1), so
/// `allocate` does not scan the whole free list. Coalescing on `deallocate` uses
/// a by-offset index. Same contract as [`FirstFitBufferAllocator`].
#[derive(Debug)]
pub struct BinnedBufferAllocator {
    segment_name: SegmentName,
    capacity: u64,
    allocated: u64,
    /// All free regions, `offset -> size` — for O(log n) neighbour coalescing.
    by_offset: BTreeMap<u64, u64>,
    /// Per-size-class sets of free-region offsets (bin = floor(log2(size))).
    bins: Vec<BTreeSet<u64>>,
    /// Bit `k` set ⇔ bin `k` is non-empty (the O(1) "smallest fitting bin" find).
    nonempty: u64,
}

impl BinnedBufferAllocator {
    pub fn new(segment_name: impl Into<SegmentName>, capacity: u64) -> Self {
        let mut a = Self {
            segment_name: segment_name.into(),
            capacity,
            allocated: 0,
            by_offset: BTreeMap::new(),
            bins: (0..64).map(|_| BTreeSet::new()).collect(),
            nonempty: 0,
        };
        if capacity > 0 {
            a.add_free(0, capacity);
        }
        a
    }

    /// Bin index for a region of `size` (≥ 1): floor(log2(size)), in 0..=63.
    fn bin_of(size: u64) -> usize {
        63 - size.leading_zeros() as usize
    }

    fn add_free(&mut self, offset: u64, size: u64) {
        if size == 0 {
            return;
        }
        self.by_offset.insert(offset, size);
        let b = Self::bin_of(size);
        self.bins[b].insert(offset);
        self.nonempty |= 1u64 << b;
    }

    fn remove_free(&mut self, offset: u64) -> Option<u64> {
        let size = self.by_offset.remove(&offset)?;
        let b = Self::bin_of(size);
        self.bins[b].remove(&offset);
        if self.bins[b].is_empty() {
            self.nonempty &= !(1u64 << b);
        }
        Some(size)
    }
}

impl BufferAllocator for BinnedBufferAllocator {
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
        let floor = Self::bin_of(size);
        // The floor bin may hold a region big enough — scan just that bin.
        let mut chosen = None;
        if (self.nonempty >> floor) & 1 == 1 {
            for &offset in &self.bins[floor] {
                if self.by_offset[&offset] >= size {
                    chosen = Some(offset);
                    break;
                }
            }
        }
        // Otherwise the lowest non-empty bin above `floor` is guaranteed to fit
        // (its regions are ≥ 2^(floor+1) > size) — found in O(1) via the bitmask.
        if chosen.is_none() {
            let higher = if floor + 1 >= 64 {
                0
            } else {
                self.nonempty & !((1u64 << (floor + 1)) - 1)
            };
            if higher != 0 {
                let b = higher.trailing_zeros() as usize;
                chosen = self.bins[b].iter().next().copied();
            }
        }
        let offset = chosen?;
        let region = self.remove_free(offset).unwrap();
        if region > size {
            self.add_free(offset + size, region - size);
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
        // Coalesce with the region immediately before / after, if they abut.
        let prev = self
            .by_offset
            .range(..start)
            .next_back()
            .map(|(&o, &l)| (o, l));
        if let Some((p_off, p_len)) = prev {
            if p_off + p_len == start {
                start = p_off;
                self.remove_free(p_off);
            }
        }
        let next = self.by_offset.range(end..).next().map(|(&o, &l)| (o, l));
        if let Some((n_off, n_len)) = next {
            if n_off == end {
                end = n_off + n_len;
                self.remove_free(n_off);
            }
        }
        self.add_free(start, end - start);
        self.allocated = self.allocated.saturating_sub(buffer.size);
    }

    fn largest_free_region(&self) -> u64 {
        self.by_offset.values().copied().max().unwrap_or(0)
    }

    fn reserve(&mut self, offset: u64, size: u64) -> bool {
        if size == 0 {
            return false;
        }
        let end = offset + size;
        let containing = self
            .by_offset
            .range(..=offset)
            .next_back()
            .map(|(&o, &l)| (o, l));
        let (r_off, r_len) = match containing {
            Some((o, l)) if o + l >= end => (o, l),
            _ => return false,
        };
        self.remove_free(r_off);
        if offset > r_off {
            self.add_free(r_off, offset - r_off);
        }
        if r_off + r_len > end {
            self.add_free(end, (r_off + r_len) - end);
        }
        self.allocated += size;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_split_and_fail_when_too_fragmented() {
        let mut a = FirstFitBufferAllocator::new("seg-0", 100);
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
        let mut a = FirstFitBufferAllocator::new("seg-0", 100);
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

    #[test]
    fn binned_allocate_split_then_coalesce() {
        let mut a = BinnedBufferAllocator::new("seg-0", 100);
        let b1 = a.allocate(40).unwrap();
        assert_eq!(b1.offset, 0);
        let b2 = a.allocate(40).unwrap();
        assert_eq!(b2.offset, 40);
        assert_eq!(a.allocated(), 80);
        // Only 20 contiguous bytes left.
        assert!(a.allocate(30).is_none());
        // Free both (out of order) → coalesce back to one 100B region.
        a.deallocate(&b1);
        a.deallocate(&b2);
        assert_eq!(a.allocated(), 0);
        assert_eq!(a.largest_free_region(), 100);
        assert_eq!(a.allocate(100).unwrap().offset, 0);
    }

    #[test]
    fn binned_reserve_carves_exact_ranges() {
        let mut a = BinnedBufferAllocator::new("seg-0", 1000);
        assert!(a.reserve(100, 64)); // carve [100, 164)
        assert_eq!(a.allocated(), 64);
        // A range overlapping the reserved one can't be reserved again.
        assert!(!a.reserve(120, 16));
        // A fresh allocation never overlaps the reserved range.
        let b = a.allocate(64).unwrap();
        assert!(b.offset + 64 <= 100 || b.offset >= 164);
    }
}
