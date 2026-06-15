//! Faithful Rust port of Sebastian Aaltonen's OffsetAllocator (MIT, 2023) — the
//! exact allocator Mooncake wraps in `OffsetBufferAllocator`
//! (`mooncake-store/src/offset_allocator.cpp`).
//!
//! Hard real-time **O(1)** allocate/free. Free regions are bucketed into 256
//! "small float" size bins (`NUM_TOP_BINS=32` × `BINS_PER_LEAF=8`); a two-level
//! bitmap (`used_bins_top: u32` + `used_bins[32]: u8`) finds the smallest fitting
//! bin with two bit-scans, and an intrusive node pool with neighbour links gives
//! O(1) coalescing on free. This is strictly more faithful than the project's
//! earlier `BinnedBufferAllocator` (64 power-of-2 bins, BTree free-lists, a scan
//! within the floor bin): the 3-bit mantissa rounds each request *up* to a bin
//! whose every member fits, so allocate never scans.
//!
//! Ported line-for-line from the C++; including Mooncake's one modification —
//! rounding an allocation's stored size up to the bin size so a later free lands
//! back in the same bin (`OFFSET_ALLOCATOR_NOT_ROUND_UP` off by default there).

use crate::allocator::{AllocatedBuffer, BufferAllocator};
use crate::types::SegmentName;
use std::collections::HashMap;

const NUM_TOP_BINS: usize = 32;
const BINS_PER_LEAF: usize = 8;
const TOP_BINS_INDEX_SHIFT: u32 = 3;
const LEAF_BINS_INDEX_MASK: u32 = 0x7;
const NUM_LEAF_BINS: usize = NUM_TOP_BINS * BINS_PER_LEAF; // 256

/// Sentinel for "no node" / "no free space" (C++ `Node::unused` / `NO_SPACE`).
const UNUSED: u32 = 0xffff_ffff;

/// `mooncake::offset_allocator::SmallFloat` — an 8-bit float (5-bit exponent +
/// 3-bit mantissa) binning so each size class carries the same average overhead.
mod small_float {
    pub const MANTISSA_BITS: u32 = 3;
    pub const MANTISSA_VALUE: u32 = 1 << MANTISSA_BITS; // 8
    pub const MANTISSA_MASK: u32 = MANTISSA_VALUE - 1;
    /// Largest size a single core allocator addresses (3.75 GiB); above this the
    /// outer `OffsetAllocator` scales sizes down by `multiplier_bits`.
    pub const MAX_BIN_SIZE: u64 = 4_026_531_840;

    /// Round a size **up** to the smallest bin index that still fits it.
    pub fn uint_to_float_round_up(size: u32) -> u32 {
        let mut exp = 0u32;
        let mut mantissa;
        if size < MANTISSA_VALUE {
            mantissa = size; // denorm: 0..(MANTISSA_VALUE-1)
        } else {
            // Normalized: hidden high bit always 1, not stored (just like float).
            let highest_set_bit = 31 - size.leading_zeros();
            let mantissa_start_bit = highest_set_bit - MANTISSA_BITS;
            exp = mantissa_start_bit + 1;
            mantissa = (size >> mantissa_start_bit) & MANTISSA_MASK;
            let low_bits_mask = (1u32 << mantissa_start_bit) - 1;
            // Round up!
            if (size & low_bits_mask) != 0 {
                mantissa += 1;
            }
        }
        // `+` (not `|`) lets a mantissa overflow carry into the exponent.
        (exp << MANTISSA_BITS) + mantissa
    }

    /// Round a size **down** to the largest bin index it covers.
    pub fn uint_to_float_round_down(size: u32) -> u32 {
        let mut exp = 0u32;
        let mantissa;
        if size < MANTISSA_VALUE {
            mantissa = size;
        } else {
            let highest_set_bit = 31 - size.leading_zeros();
            let mantissa_start_bit = highest_set_bit - MANTISSA_BITS;
            exp = mantissa_start_bit + 1;
            mantissa = (size >> mantissa_start_bit) & MANTISSA_MASK;
        }
        (exp << MANTISSA_BITS) | mantissa
    }

    /// Inverse: the byte size a bin index represents.
    pub fn float_to_uint(float_value: u32) -> u32 {
        let exponent = float_value >> MANTISSA_BITS;
        let mantissa = float_value & MANTISSA_MASK;
        if exponent == 0 {
            mantissa // denorms
        } else {
            (mantissa | MANTISSA_VALUE) << (exponent - 1)
        }
    }
}

/// Lowest set bit at-or-after `start_bit_index`, or [`UNUSED`] if none.
fn find_lowest_set_bit_after(bit_mask: u32, start_bit_index: u32) -> u32 {
    if start_bit_index >= 32 {
        return UNUSED;
    }
    let mask_before_start = (1u32 << start_bit_index) - 1;
    let bits_after = bit_mask & !mask_before_start;
    if bits_after == 0 {
        return UNUSED;
    }
    bits_after.trailing_zeros()
}

/// The result of an allocation: where it sits (`offset`) plus the node index that
/// owns it (`metadata`), which `free` needs. Mirrors C++ `OffsetAllocation`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OffsetAllocation {
    pub offset: u32,
    pub metadata: u32,
}

impl OffsetAllocation {
    pub const NO_SPACE: u32 = UNUSED;

    pub fn is_no_space(&self) -> bool {
        self.metadata == Self::NO_SPACE
    }
}

/// A free or allocated region. Two intrusive doubly-linked lists thread through
/// the pool: `bin_list_*` (other regions in the same size bin) and `neighbor_*`
/// (the physically adjacent regions, for O(1) coalescing on free).
#[derive(Debug, Clone, Copy)]
struct Node {
    data_offset: u32,
    data_size: u32,
    bin_list_prev: u32,
    bin_list_next: u32,
    neighbor_prev: u32,
    neighbor_next: u32,
    used: bool,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            data_offset: 0,
            data_size: 0,
            bin_list_prev: UNUSED,
            bin_list_next: UNUSED,
            neighbor_prev: UNUSED,
            neighbor_next: UNUSED,
            used: false,
        }
    }
}

/// Free-space totals (C++ `OffsetAllocStorageReport`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageReport {
    pub total_free_space: u32,
    pub largest_free_region: u32,
}

/// The core allocator (C++ `__Allocator`), addressing a `[0, size)` span in
/// abstract units. The `nodes`/`free_nodes` pool grows lazily up to
/// `max_capacity`.
#[derive(Debug)]
pub struct CoreAllocator {
    size: u32,
    current_capacity: u32,
    max_capacity: u32,
    free_storage: u32,
    used_bins_top: u32,
    used_bins: [u8; NUM_TOP_BINS],
    bin_indices: [u32; NUM_LEAF_BINS],
    nodes: Vec<Node>,
    /// Free-node stack: `free_nodes[free_offset]` is the next index to hand out.
    free_nodes: Vec<u32>,
    free_offset: u32,
}

impl CoreAllocator {
    pub fn new(size: u32, init_capacity: u32, max_capacity: u32) -> Self {
        let mut a = Self {
            size,
            current_capacity: init_capacity,
            max_capacity: init_capacity.max(max_capacity),
            free_storage: 0,
            used_bins_top: 0,
            used_bins: [0; NUM_TOP_BINS],
            bin_indices: [UNUSED; NUM_LEAF_BINS],
            nodes: Vec::new(),
            free_nodes: Vec::new(),
            free_offset: 0,
        };
        a.reset();
        a
    }

    fn reset(&mut self) {
        self.free_storage = 0;
        self.used_bins_top = 0;
        self.free_offset = 0;
        self.used_bins = [0; NUM_TOP_BINS];
        self.bin_indices = [UNUSED; NUM_LEAF_BINS];

        self.nodes.clear();
        self.free_nodes.clear();
        self.nodes.reserve(self.max_capacity as usize);
        self.free_nodes.reserve(self.max_capacity as usize);
        self.nodes
            .resize(self.current_capacity as usize, Node::default());
        self.free_nodes.resize(self.current_capacity as usize, 0);
        // Freelist is a stack; fill so index 0 pops first.
        for i in 0..self.current_capacity {
            self.free_nodes[i as usize] = i;
        }

        // Start state: the whole span as one big free node.
        self.insert_node_into_bin(self.size, 0);
    }

    /// Allocate `size` units; [`OffsetAllocation::is_no_space`] on failure.
    pub fn allocate(&mut self, size: u32) -> OffsetAllocation {
        // Out of node handles?
        if self.free_offset == self.max_capacity {
            return OffsetAllocation {
                offset: UNUSED,
                metadata: UNUSED,
            };
        }
        if self.free_offset == self.current_capacity {
            self.free_nodes.push(self.current_capacity);
            self.nodes.push(Node::default());
            self.current_capacity += 1;
        }

        // Round up to a bin whose every member fits, then find the smallest
        // non-empty such bin via the two-level bitmap.
        let min_bin_index = small_float::uint_to_float_round_up(size);
        let min_top_bin_index = min_bin_index >> TOP_BINS_INDEX_SHIFT;
        let min_leaf_bin_index = min_bin_index & LEAF_BINS_INDEX_MASK;

        let mut top_bin_index = min_top_bin_index;
        let mut leaf_bin_index = UNUSED;

        // If that top bin has any leaves, scan from the rounded-up leaf.
        if self.used_bins_top & (1 << top_bin_index) != 0 {
            leaf_bin_index = find_lowest_set_bit_after(
                self.used_bins[top_bin_index as usize] as u32,
                min_leaf_bin_index,
            );
        }

        // Otherwise search the next non-empty top bin (its leaves all fit).
        if leaf_bin_index == UNUSED {
            top_bin_index = find_lowest_set_bit_after(self.used_bins_top, min_top_bin_index + 1);
            if top_bin_index == UNUSED {
                return OffsetAllocation {
                    offset: UNUSED,
                    metadata: UNUSED,
                };
            }
            // The top bit was set, so at least one leaf bit is set — can't fail.
            leaf_bin_index = (self.used_bins[top_bin_index as usize] as u32).trailing_zeros();
        }

        let bin_index = (top_bin_index << TOP_BINS_INDEX_SHIFT) | leaf_bin_index;

        // Pop the bin's head node.
        let node_index = self.bin_indices[bin_index as usize];
        let node_total_size = self.nodes[node_index as usize].data_size;
        let node_data_offset = self.nodes[node_index as usize].data_offset;
        let node_bin_list_next = self.nodes[node_index as usize].bin_list_next;
        let node_neighbor_next = self.nodes[node_index as usize].neighbor_next;

        // Mooncake modification: store the bin-rounded size so a later free lands
        // back in this same bin instead of a smaller one.
        let roundup_size = small_float::float_to_uint(min_bin_index);
        self.nodes[node_index as usize].data_size = roundup_size;
        self.nodes[node_index as usize].used = true;

        self.bin_indices[bin_index as usize] = node_bin_list_next;
        if node_bin_list_next != UNUSED {
            self.nodes[node_bin_list_next as usize].bin_list_prev = UNUSED;
        }
        self.free_storage -= node_total_size;

        // Bin emptied? Clear the leaf bit, and the top bit if all leaves gone.
        if self.bin_indices[bin_index as usize] == UNUSED {
            self.used_bins[top_bin_index as usize] &= !(1 << leaf_bin_index);
            if self.used_bins[top_bin_index as usize] == 0 {
                self.used_bins_top &= !(1 << top_bin_index);
            }
        }

        // Push the remainder back as a smaller free node, linked as our neighbour.
        let reminder_size = node_total_size - roundup_size;
        if reminder_size > 0 {
            let new_node_index =
                self.insert_node_into_bin(reminder_size, node_data_offset + roundup_size);
            if node_neighbor_next != UNUSED {
                self.nodes[node_neighbor_next as usize].neighbor_prev = new_node_index;
            }
            self.nodes[new_node_index as usize].neighbor_prev = node_index;
            self.nodes[new_node_index as usize].neighbor_next = node_neighbor_next;
            self.nodes[node_index as usize].neighbor_next = new_node_index;
        }

        OffsetAllocation {
            offset: node_data_offset,
            metadata: node_index,
        }
    }

    /// Free a prior allocation, coalescing with free physical neighbours.
    pub fn free(&mut self, allocation: OffsetAllocation) {
        debug_assert!(allocation.metadata != UNUSED);
        if self.nodes.is_empty() {
            return;
        }
        let node_index = allocation.metadata;
        debug_assert!(self.nodes[node_index as usize].used);

        let mut offset = self.nodes[node_index as usize].data_offset;
        let mut size = self.nodes[node_index as usize].data_size;

        // Merge with the previous physical neighbour if it is free.
        let neighbor_prev = self.nodes[node_index as usize].neighbor_prev;
        if neighbor_prev != UNUSED && !self.nodes[neighbor_prev as usize].used {
            offset = self.nodes[neighbor_prev as usize].data_offset;
            size += self.nodes[neighbor_prev as usize].data_size;
            self.remove_node_from_bin(neighbor_prev);
            self.nodes[node_index as usize].neighbor_prev =
                self.nodes[neighbor_prev as usize].neighbor_prev;
        }

        // Merge with the next physical neighbour if it is free.
        let neighbor_next = self.nodes[node_index as usize].neighbor_next;
        if neighbor_next != UNUSED && !self.nodes[neighbor_next as usize].used {
            size += self.nodes[neighbor_next as usize].data_size;
            self.remove_node_from_bin(neighbor_next);
            self.nodes[node_index as usize].neighbor_next =
                self.nodes[neighbor_next as usize].neighbor_next;
        }

        let final_neighbor_next = self.nodes[node_index as usize].neighbor_next;
        let final_neighbor_prev = self.nodes[node_index as usize].neighbor_prev;

        // Return the node to the freelist, then re-insert the combined region.
        self.free_offset -= 1;
        self.free_nodes[self.free_offset as usize] = node_index;
        let combined_node_index = self.insert_node_into_bin(size, offset);

        // Relink the combined node with the surviving neighbours.
        if final_neighbor_next != UNUSED {
            self.nodes[combined_node_index as usize].neighbor_next = final_neighbor_next;
            self.nodes[final_neighbor_next as usize].neighbor_prev = combined_node_index;
        }
        if final_neighbor_prev != UNUSED {
            self.nodes[combined_node_index as usize].neighbor_prev = final_neighbor_prev;
            self.nodes[final_neighbor_prev as usize].neighbor_next = combined_node_index;
        }
    }

    fn insert_node_into_bin(&mut self, size: u32, data_offset: u32) -> u32 {
        // Round down so the bin's size class is <= the region (every member fits).
        let bin_index = small_float::uint_to_float_round_down(size);
        let top_bin_index = bin_index >> TOP_BINS_INDEX_SHIFT;
        let leaf_bin_index = bin_index & LEAF_BINS_INDEX_MASK;

        // Bin was empty? Set both bitmap levels.
        if self.bin_indices[bin_index as usize] == UNUSED {
            self.used_bins[top_bin_index as usize] |= 1 << leaf_bin_index;
            self.used_bins_top |= 1 << top_bin_index;
        }

        let top_node_index = self.bin_indices[bin_index as usize];
        let node_index = self.free_nodes[self.free_offset as usize];
        self.free_offset += 1;

        self.nodes[node_index as usize] = Node {
            data_offset,
            data_size: size,
            bin_list_prev: UNUSED,
            bin_list_next: top_node_index,
            neighbor_prev: UNUSED,
            neighbor_next: UNUSED,
            used: false,
        };
        if top_node_index != UNUSED {
            self.nodes[top_node_index as usize].bin_list_prev = node_index;
        }
        self.bin_indices[bin_index as usize] = node_index;
        self.free_storage += size;
        node_index
    }

    fn remove_node_from_bin(&mut self, node_index: u32) {
        let bin_list_prev = self.nodes[node_index as usize].bin_list_prev;
        let bin_list_next = self.nodes[node_index as usize].bin_list_next;

        if bin_list_prev != UNUSED {
            // Middle of the bin list — just unlink.
            self.nodes[bin_list_prev as usize].bin_list_next = bin_list_next;
            if bin_list_next != UNUSED {
                self.nodes[bin_list_next as usize].bin_list_prev = bin_list_prev;
            }
        } else {
            // Head of the bin — find the bin and repoint it.
            let bin_index =
                small_float::uint_to_float_round_down(self.nodes[node_index as usize].data_size);
            let top_bin_index = bin_index >> TOP_BINS_INDEX_SHIFT;
            let leaf_bin_index = bin_index & LEAF_BINS_INDEX_MASK;

            self.bin_indices[bin_index as usize] = bin_list_next;
            if bin_list_next != UNUSED {
                self.nodes[bin_list_next as usize].bin_list_prev = UNUSED;
            }
            if self.bin_indices[bin_index as usize] == UNUSED {
                self.used_bins[top_bin_index as usize] &= !(1 << leaf_bin_index);
                if self.used_bins[top_bin_index as usize] == 0 {
                    self.used_bins_top &= !(1 << top_bin_index);
                }
            }
        }

        self.free_offset -= 1;
        self.free_nodes[self.free_offset as usize] = node_index;
        self.free_storage -= self.nodes[node_index as usize].data_size;
    }

    pub fn storage_report(&self) -> StorageReport {
        let mut largest_free_region = 0;
        let mut free_storage = 0;
        if self.free_offset < self.max_capacity {
            free_storage = self.free_storage;
            if self.used_bins_top != 0 {
                let top_bin_index = 31 - self.used_bins_top.leading_zeros();
                let leaf_bin_index =
                    31 - (self.used_bins[top_bin_index as usize] as u32).leading_zeros();
                largest_free_region = small_float::float_to_uint(
                    (top_bin_index << TOP_BINS_INDEX_SHIFT) | leaf_bin_index,
                );
            }
        }
        StorageReport {
            total_free_space: free_storage,
            largest_free_region,
        }
    }

    /// Grow the lazily-allocated node pool so at least `n` slots are poppable,
    /// bounded by `max_capacity`. Returns false if the cap is reached.
    fn ensure_capacity(&mut self, n: u32) -> bool {
        while self.current_capacity - self.free_offset < n {
            if self.current_capacity == self.max_capacity {
                return false;
            }
            self.free_nodes.push(self.current_capacity);
            self.nodes.push(Node::default());
            self.current_capacity += 1;
        }
        true
    }

    /// Carve an exact `[offset, offset+size)` range out of whatever free region
    /// covers it, mark it used, and return its handle. Aaltonen has no
    /// addressed-reserve, so master recovery rebuilds an *equivalent* valid state
    /// from the snapshot's per-replica ranges this way. `None` if the range is not
    /// entirely free.
    pub fn reserve_exact(&mut self, offset: u32, size: u32) -> Option<OffsetAllocation> {
        if size == 0 {
            return None;
        }
        let end = offset.checked_add(size)?;
        // up to 3 nodes (before-free, used, after-free); we free 1 (the covering).
        if !self.ensure_capacity(3) {
            return None;
        }

        // Find the free region covering [offset, end) by walking the bin lists.
        let mut covering = UNUSED;
        'outer: for bin in 0..NUM_LEAF_BINS {
            let mut idx = self.bin_indices[bin];
            while idx != UNUSED {
                let n = &self.nodes[idx as usize];
                if n.data_offset <= offset && n.data_offset + n.data_size >= end {
                    covering = idx;
                    break 'outer;
                }
                idx = n.bin_list_next;
            }
        }
        if covering == UNUSED {
            return None;
        }

        let r_off = self.nodes[covering as usize].data_offset;
        let r_len = self.nodes[covering as usize].data_size;
        let nb_prev = self.nodes[covering as usize].neighbor_prev;
        let nb_next = self.nodes[covering as usize].neighbor_next;
        self.remove_node_from_bin(covering);

        // before-free [r_off, offset)
        let before = if offset > r_off {
            Some(self.insert_node_into_bin(offset - r_off, r_off))
        } else {
            None
        };

        // used [offset, end) — popped from the freelist, kept out of every bin.
        let used = self.free_nodes[self.free_offset as usize];
        self.free_offset += 1;
        self.nodes[used as usize] = Node {
            data_offset: offset,
            data_size: size,
            bin_list_prev: UNUSED,
            bin_list_next: UNUSED,
            neighbor_prev: UNUSED,
            neighbor_next: UNUSED,
            used: true,
        };

        // after-free [end, r_off+r_len)
        let after = if r_off + r_len > end {
            Some(self.insert_node_into_bin((r_off + r_len) - end, end))
        } else {
            None
        };

        // Re-thread neighbours: nb_prev <-> before <-> used <-> after <-> nb_next.
        let mut prev = nb_prev;
        for &cur in [before, Some(used), after].iter().flatten() {
            self.nodes[cur as usize].neighbor_prev = prev;
            if prev != UNUSED {
                self.nodes[prev as usize].neighbor_next = cur;
            }
            prev = cur;
        }
        self.nodes[prev as usize].neighbor_next = nb_next;
        if nb_next != UNUSED {
            self.nodes[nb_next as usize].neighbor_prev = prev;
        }

        Some(OffsetAllocation {
            offset,
            metadata: used,
        })
    }
}

/// Number of times `size` must be halved to fit under [`small_float::MAX_BIN_SIZE`].
fn calculate_multiplier(size: u64) -> u32 {
    let mut multiplier_bits = 0u32;
    while small_float::MAX_BIN_SIZE < (size >> multiplier_bits) {
        multiplier_bits += 1;
    }
    multiplier_bits
}

/// One allocation handed out by [`OffsetAllocator`]: the byte `offset` (already
/// shifted by the segment base) and `size` the caller asked for, plus the core
/// handle needed to free it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Allocation {
    pub offset: u64,
    pub size: u64,
    core: OffsetAllocation,
}

/// The byte-addressed allocator (C++ `OffsetAllocator`): adds a segment `base`
/// and a `multiplier_bits` size-scaling so spans larger than 3.75 GiB still map
/// onto the 32-bit core. Single-threaded; the store wraps it under its own lock.
#[derive(Debug)]
pub struct OffsetAllocator {
    core: CoreAllocator,
    base: u64,
    multiplier_bits: u32,
    capacity: u64,
    allocated_size: u64,
    allocated_num: u64,
}

impl OffsetAllocator {
    pub fn new(base: u64, size: u64, init_capacity: u32, max_capacity: u32) -> Self {
        let multiplier_bits = calculate_multiplier(size);
        let core = CoreAllocator::new(
            (size >> multiplier_bits) as u32,
            init_capacity,
            max_capacity,
        );
        Self {
            core,
            base,
            multiplier_bits,
            capacity: size,
            allocated_size: 0,
            allocated_num: 0,
        }
    }

    /// Allocate `size` bytes; `None` if no region fits or `size` is 0/too large.
    pub fn allocate(&mut self, size: u64) -> Option<Allocation> {
        if size == 0 {
            return None;
        }
        let fake_size = if self.multiplier_bits > 0 {
            (size + (1u64 << self.multiplier_bits) - 1) >> self.multiplier_bits
        } else {
            size
        };
        if fake_size > small_float::MAX_BIN_SIZE {
            return None;
        }
        let allocation = self.core.allocate(fake_size as u32);
        if allocation.is_no_space() {
            return None;
        }
        self.allocated_size += size;
        self.allocated_num += 1;
        Some(Allocation {
            offset: self.base + ((allocation.offset as u64) << self.multiplier_bits),
            size,
            core: allocation,
        })
    }

    pub fn free(&mut self, allocation: &Allocation) {
        self.core.free(allocation.core);
        self.allocated_size = self.allocated_size.saturating_sub(allocation.size);
        self.allocated_num = self.allocated_num.saturating_sub(1);
    }

    /// Reserve an exact byte range (master recovery). Only supported when sizes
    /// aren't scaled (`multiplier_bits == 0`, i.e. segments ≤ 3.75 GiB) — larger
    /// segments must restore allocator state directly rather than by range.
    pub fn reserve_exact(&mut self, offset: u64, size: u64) -> Option<Allocation> {
        if size == 0 || offset < self.base || self.multiplier_bits != 0 {
            return None;
        }
        let core_off = u32::try_from(offset - self.base).ok()?;
        let core_size = u32::try_from(size).ok()?;
        let core = self.core.reserve_exact(core_off, core_size)?;
        self.allocated_size += size;
        self.allocated_num += 1;
        Some(Allocation { offset, size, core })
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    pub fn allocated_size(&self) -> u64 {
        self.allocated_size
    }

    /// Free totals in bytes (scaled back up by `multiplier_bits`).
    pub fn storage_report(&self) -> (u64, u64) {
        let r = self.core.storage_report();
        (
            (r.total_free_space as u64) << self.multiplier_bits,
            (r.largest_free_region as u64) << self.multiplier_bits,
        )
    }
}

/// The store's per-segment `BufferAllocator`, backed by the faithful
/// [`OffsetAllocator`] — this is Mooncake's `OffsetBufferAllocator` (the
/// Aaltonen-backed one) and the store's default, since it strictly dominates the
/// earlier first-fit allocator on both speed (O(1)) and fragmentation. The trait
/// deallocates by `(offset, size)` while the core frees by handle, so a small
/// `offset -> handle` map bridges them.
#[derive(Debug)]
pub struct OffsetBufferAllocator {
    segment_name: SegmentName,
    inner: OffsetAllocator,
    live: HashMap<u64, Allocation>,
}

impl OffsetBufferAllocator {
    pub fn new(segment_name: impl Into<SegmentName>, capacity: u64) -> Self {
        Self {
            segment_name: segment_name.into(),
            // The node pool grows lazily (≈ one node per live region); cap high.
            inner: OffsetAllocator::new(0, capacity, 256, 1 << 22),
            live: HashMap::new(),
        }
    }
}

impl BufferAllocator for OffsetBufferAllocator {
    fn segment_name(&self) -> &str {
        &self.segment_name
    }

    fn capacity(&self) -> u64 {
        self.inner.capacity()
    }

    fn allocated(&self) -> u64 {
        self.inner.allocated_size()
    }

    fn allocate(&mut self, size: u64) -> Option<AllocatedBuffer> {
        let a = self.inner.allocate(size)?;
        self.live.insert(a.offset, a);
        Some(AllocatedBuffer {
            segment_name: self.segment_name.clone(),
            offset: a.offset,
            size,
        })
    }

    fn deallocate(&mut self, buffer: &AllocatedBuffer) {
        if let Some(a) = self.live.remove(&buffer.offset) {
            self.inner.free(&a);
        }
    }

    fn largest_free_region(&self) -> u64 {
        self.inner.storage_report().1
    }

    fn reserve(&mut self, offset: u64, size: u64) -> bool {
        match self.inner.reserve_exact(offset, size) {
            Some(a) => {
                self.live.insert(a.offset, a);
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_float_round_trips_and_orders() {
        // Denorm range (< 8) is exact.
        for s in 1..8u32 {
            assert_eq!(
                small_float::float_to_uint(small_float::uint_to_float_round_down(s)),
                s
            );
        }
        // Round-up bin always fits the request; round-down never exceeds it.
        for &s in &[8u32, 9, 15, 16, 100, 1000, 65537, 1 << 20] {
            let up = small_float::float_to_uint(small_float::uint_to_float_round_up(s));
            let down = small_float::float_to_uint(small_float::uint_to_float_round_down(s));
            assert!(up >= s, "round-up bin {up} must fit {s}");
            assert!(down <= s, "round-down bin {down} must not exceed {s}");
            // 3-bit mantissa ⇒ at most ~1/8 overhead.
            assert!(up <= s + s / 8 + 8, "round-up overhead too large for {s}");
        }
    }

    #[test]
    fn allocate_basic_offsets_and_free() {
        let mut a = OffsetAllocator::new(0, 1 << 16, 128, 4096);
        let x = a.allocate(1337).unwrap();
        assert_eq!(x.offset, 0);
        let y = a.allocate(123).unwrap();
        assert!(y.offset >= 1337); // past x's (bin-rounded) extent
        let z = a.allocate(64).unwrap();
        a.free(&y);
        // Freed region is reused by a same-class request.
        let y2 = a.allocate(123).unwrap();
        assert_eq!(y2.offset, y.offset);
        a.free(&x);
        a.free(&z);
        a.free(&y2);
        // Everything back ⇒ the whole span is one free region again.
        let (free, largest) = a.storage_report();
        assert_eq!(free, 1 << 16);
        assert_eq!(largest, 1 << 16);
    }

    #[test]
    fn coalesces_neighbours_on_free() {
        // Two adjacent half-segment allocations, freed in both orders, must merge
        // back into one full-size region (not stay as two fragments).
        for forward in [true, false] {
            let mut a = OffsetAllocator::new(0, 1024, 16, 256);
            let b1 = a.allocate(512).unwrap();
            let b2 = a.allocate(512).unwrap();
            assert_eq!(a.storage_report().1, 0); // full
            if forward {
                a.free(&b1);
                a.free(&b2);
            } else {
                a.free(&b2);
                a.free(&b1);
            }
            assert_eq!(a.storage_report().1, 1024, "neighbours must coalesce");
            assert!(a.allocate(1024).is_some());
        }
    }

    #[test]
    fn fragmentation_blocks_oversized_request() {
        let mut a = OffsetAllocator::new(0, 256, 16, 256);
        // Bins round up to multiples; carve the span into power-of-two chunks.
        let b1 = a.allocate(128).unwrap();
        let _b2 = a.allocate(64).unwrap();
        let _b3 = a.allocate(64).unwrap();
        // Span is full now — a fresh request must fail.
        assert!(a.allocate(64).is_none());
        // Free a hole; only that hole's worth can be re-allocated, not more.
        a.free(&b1);
        assert!(a.allocate(128).is_some());
        assert!(a.allocate(64).is_none());
    }

    #[test]
    fn base_offset_is_applied() {
        let mut a = OffsetAllocator::new(4096, 1 << 16, 64, 1024);
        let x = a.allocate(100).unwrap();
        assert_eq!(x.offset, 4096); // base + 0
        assert!(a.allocate(100).unwrap().offset > 4096);
    }

    #[test]
    fn large_span_uses_multiplier() {
        // 8 GiB span > MAX_BIN_SIZE (3.75 GiB) ⇒ multiplier_bits > 0, still works.
        let mut a = OffsetAllocator::new(0, 8u64 << 30, 64, 4096);
        let x = a.allocate(1 << 20).unwrap();
        assert_eq!(x.offset, 0);
        let y = a.allocate(1 << 20).unwrap();
        assert!(y.offset >= (1 << 20));
        a.free(&x);
        a.free(&y);
    }

    #[test]
    fn exhausts_node_pool_gracefully() {
        // max_capacity caps the node pool; allocation fails cleanly, never panics.
        let mut a = OffsetAllocator::new(0, 1 << 20, 4, 8);
        let mut held = Vec::new();
        for _ in 0..100 {
            match a.allocate(16) {
                Some(h) => held.push(h),
                None => break,
            }
        }
        // Freeing restores capacity.
        for h in &held {
            a.free(h);
        }
        assert_eq!(a.storage_report().0, 1 << 20);
    }
}
