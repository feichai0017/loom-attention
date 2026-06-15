//! Count-Min Sketch (Mooncake's `mooncake-store/include/count_min_sketch.h`) —
//! a compact, approximate key-frequency counter. Mooncake uses it for the
//! frequency-admission policy (whether a key is hot enough to promote into the
//! local hot cache); the [`crate::MasterService`] records every guarded `get` so
//! eviction / promotion can favour hot keys over a pure LRU.
//!
//! `depth` independent rows × `width` `u8` counters; `increment`/`count` take the
//! **min** across rows (the CMS estimate, which never under-counts). Counters
//! saturate at 255 and all halve (`>>= 1`) once `width * depth` increments have
//! accumulated, so the sketch ages instead of saturating — a faithful port,
//! single-threaded (the master holds its own lock, so no internal mutex).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const DEFAULT_WIDTH: usize = 4096;
const DEFAULT_DEPTH: usize = 4;

/// Approximate frequency counter over string keys.
#[derive(Debug)]
pub struct CountMinSketch {
    width: usize,
    depth: usize,
    table: Vec<Vec<u8>>,
    total_increments: usize,
}

impl Default for CountMinSketch {
    fn default() -> Self {
        Self::new(DEFAULT_WIDTH, DEFAULT_DEPTH)
    }
}

impl CountMinSketch {
    /// `width` counters per row × `depth` rows. Zero falls back to the defaults
    /// (4096 × 4), matching Mooncake.
    pub fn new(width: usize, depth: usize) -> Self {
        let width = if width > 0 { width } else { DEFAULT_WIDTH };
        let depth = if depth > 0 { depth } else { DEFAULT_DEPTH };
        Self {
            width,
            depth,
            table: vec![vec![0u8; width]; depth],
            total_increments: 0,
        }
    }

    /// Per-row hash: a base hash of the key mixed with the row seed (fmix64
    /// finalizer), so each row is an independent hash — faithfully Mooncake's.
    fn hash(&self, key: &str, seed: usize) -> usize {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let mut h = hasher.finish();
        h ^= (seed as u64)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(0x517c_c1b7_2722_0a95);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h as usize
    }

    /// Increment `key`'s count and return the (post-increment) min estimate.
    /// Triggers a decay once `width * depth` increments accumulate.
    pub fn increment(&mut self, key: &str) -> u8 {
        let mut min_val = u8::MAX;
        for i in 0..self.depth {
            let idx = self.hash(key, i) % self.width;
            // Saturating: counters cap at 255 (Mooncake's `< UINT8_MAX` guard).
            self.table[i][idx] = self.table[i][idx].saturating_add(1);
            min_val = min_val.min(self.table[i][idx]);
        }
        self.total_increments += 1;
        if self.total_increments >= self.width * self.depth {
            self.decay();
        }
        min_val
    }

    /// The estimated count for `key` (read-only; the min across rows).
    pub fn count(&self, key: &str) -> u8 {
        let mut min_val = u8::MAX;
        for i in 0..self.depth {
            let idx = self.hash(key, i) % self.width;
            min_val = min_val.min(self.table[i][idx]);
        }
        min_val
    }

    /// Halve every counter (periodic aging). Resets the decay accumulator.
    pub fn decay(&mut self) {
        for row in &mut self.table {
            for counter in row.iter_mut() {
                *counter >>= 1;
            }
        }
        self.total_increments = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_grow_with_repeated_increments() {
        let mut cms = CountMinSketch::new(256, 4);
        assert_eq!(cms.count("a"), 0);
        for expected in 1..=10u8 {
            assert_eq!(cms.increment("a"), expected);
        }
        assert_eq!(cms.count("a"), 10);
        // An un-incremented key reads ~0 (no collisions in a 256-wide sketch).
        assert_eq!(cms.count("never"), 0);
    }

    #[test]
    fn hotter_keys_estimate_higher() {
        let mut cms = CountMinSketch::new(1024, 4);
        for _ in 0..50 {
            cms.increment("hot");
        }
        for _ in 0..3 {
            cms.increment("cold");
        }
        assert!(cms.count("hot") > cms.count("cold"));
    }

    #[test]
    fn counters_saturate_without_overflow() {
        let mut cms = CountMinSketch::new(1024, 4); // big enough: no auto-decay
        for _ in 0..500 {
            cms.increment("x");
        }
        assert_eq!(
            cms.count("x"),
            u8::MAX,
            "counter saturates at 255, never wraps"
        );
    }

    #[test]
    fn decay_halves_counters() {
        let mut cms = CountMinSketch::new(1024, 4);
        for _ in 0..40 {
            cms.increment("x");
        }
        let before = cms.count("x");
        cms.decay();
        assert_eq!(cms.count("x"), before / 2);
    }

    #[test]
    fn auto_decay_keeps_counters_bounded() {
        // Tiny sketch: width*depth = 4, so decay fires every 4 increments.
        let mut cms = CountMinSketch::new(2, 2);
        for _ in 0..100 {
            cms.increment("x");
        }
        // Without aging this would have saturated; auto-decay keeps it small.
        assert!(cms.count("x") < u8::MAX);
    }
}
