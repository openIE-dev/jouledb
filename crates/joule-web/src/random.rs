//! Pseudorandom number generation — Xoshiro256**, distributions, shuffling.
//!
//! Pure-Rust replacement for seedrandom, chance.js, faker.js random utilities,
//! and similar JS/TS PRNG libraries.

use std::fmt;

// ── Xoshiro256** ───────────────────────────────────────────────

/// Xoshiro256** pseudorandom number generator.
///
/// Fast, high-quality, reproducible 64-bit PRNG with 256-bit state.
#[derive(Debug, Clone)]
pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    /// Create a new RNG from a 64-bit seed.
    pub fn new(seed: u64) -> Self {
        // Use SplitMix64 to initialize state from seed
        let mut sm = seed;
        let mut state = [0u64; 4];
        for s in &mut state {
            sm = sm.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *s = z ^ (z >> 31);
        }
        Self { state }
    }

    /// Create from explicit state.
    pub fn from_state(state: [u64; 4]) -> Self {
        Self { state }
    }

    /// Get current state for serialization.
    pub fn state(&self) -> [u64; 4] {
        self.state
    }

    /// Generate next u64.
    pub fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    /// Generate a random f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11; // 53 bits
        bits as f64 / (1u64 << 53) as f64
    }

    /// Generate a random boolean.
    pub fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    /// Uniform random integer in [lo, hi] (inclusive).
    pub fn uniform_int(&mut self, lo: i64, hi: i64) -> i64 {
        if lo >= hi {
            return lo;
        }
        let range = (hi - lo) as u64 + 1;
        lo + (self.next_u64() % range) as i64
    }

    /// Uniform random f64 in [lo, hi).
    pub fn uniform_f64(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }

    /// Normal distribution using Box-Muller transform.
    pub fn normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        let u1 = self.next_f64().max(1e-15); // avoid log(0)
        let u2 = self.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + z * std_dev
    }

    /// Weighted random selection from a slice of (weight, value) pairs.
    /// Returns `None` if the slice is empty or all weights are zero.
    pub fn weighted_choice<'a, T>(&mut self, items: &'a [(f64, T)]) -> Option<&'a T> {
        if items.is_empty() {
            return None;
        }
        let total: f64 = items.iter().map(|(w, _)| w.max(0.0)).sum();
        if total <= 0.0 {
            return None;
        }
        let mut threshold = self.next_f64() * total;
        for (w, item) in items {
            let weight = w.max(0.0);
            threshold -= weight;
            if threshold <= 0.0 {
                return Some(item);
            }
        }
        Some(&items.last().unwrap().1)
    }

    /// Fisher-Yates shuffle in place.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len();
        if n < 2 {
            return;
        }
        for i in (1..n).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }

    /// Sample N items from a collection without replacement.
    /// Returns up to `n` items. If `n >= collection.len()`, returns a shuffled copy of the entire collection.
    pub fn sample<T: Clone>(&mut self, collection: &[T], n: usize) -> Vec<T> {
        let mut pool: Vec<T> = collection.to_vec();
        self.shuffle(&mut pool);
        pool.truncate(n);
        pool
    }

    /// Generate a UUID v4 (random).
    pub fn uuid_v4(&mut self) -> String {
        let a = self.next_u64();
        let b = self.next_u64();
        // Set version (4) and variant (10xx)
        let hi = (a & 0xFFFFFFFF_FFFF0FFF) | 0x00000000_00004000;
        let lo = (b & 0x3FFFFFFF_FFFFFFFF) | 0x80000000_00000000;
        format!(
            "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
            (hi >> 32) as u32,
            (hi >> 16) as u16 & 0xFFFF,
            hi as u16,
            (lo >> 48) as u16,
            lo & 0xFFFFFFFFFFFF
        )
    }

    /// Fill a byte slice with random bytes.
    pub fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut i = 0;
        while i < dest.len() {
            let val = self.next_u64();
            let bytes = val.to_le_bytes();
            let remaining = dest.len() - i;
            let to_copy = remaining.min(8);
            dest[i..i + to_copy].copy_from_slice(&bytes[..to_copy]);
            i += to_copy;
        }
    }
}

impl fmt::Display for Rng {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rng(state=[{:x}, {:x}, {:x}, {:x}])",
            self.state[0], self.state[1], self.state[2], self.state[3])
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_sequences() {
        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn different_seeds_different_output() {
        let mut r1 = Rng::new(1);
        let mut r2 = Rng::new(2);
        let vals1: Vec<u64> = (0..10).map(|_| r1.next_u64()).collect();
        let vals2: Vec<u64> = (0..10).map(|_| r2.next_u64()).collect();
        assert_ne!(vals1, vals2);
    }

    #[test]
    fn next_f64_range() {
        let mut r = Rng::new(99);
        for _ in 0..1000 {
            let v = r.next_f64();
            assert!(v >= 0.0 && v < 1.0, "f64 out of range: {v}");
        }
    }

    #[test]
    fn uniform_int_range() {
        let mut r = Rng::new(7);
        for _ in 0..200 {
            let v = r.uniform_int(5, 10);
            assert!(v >= 5 && v <= 10, "uniform_int out of range: {v}");
        }
    }

    #[test]
    fn uniform_f64_range() {
        let mut r = Rng::new(13);
        for _ in 0..200 {
            let v = r.uniform_f64(-1.0, 1.0);
            assert!(v >= -1.0 && v < 1.0, "uniform_f64 out of range: {v}");
        }
    }

    #[test]
    fn normal_distribution() {
        let mut r = Rng::new(42);
        let mut sum = 0.0;
        let n = 10000;
        for _ in 0..n {
            sum += r.normal(0.0, 1.0);
        }
        let mean = sum / n as f64;
        assert!(mean.abs() < 0.1, "Normal mean too far from 0: {mean}");
    }

    #[test]
    fn weighted_choice_basic() {
        let mut r = Rng::new(42);
        let items = [(1.0, "rare"), (99.0, "common")];
        let mut common_count = 0;
        for _ in 0..1000 {
            if *r.weighted_choice(&items).unwrap() == "common" {
                common_count += 1;
            }
        }
        assert!(common_count > 900, "Common should appear most often: {common_count}");
    }

    #[test]
    fn shuffle_preserves_elements() {
        let mut r = Rng::new(42);
        let mut v = vec![1, 2, 3, 4, 5];
        r.shuffle(&mut v);
        v.sort();
        assert_eq!(v, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn sample_without_replacement() {
        let mut r = Rng::new(42);
        let collection = vec![10, 20, 30, 40, 50];
        let s = r.sample(&collection, 3);
        assert_eq!(s.len(), 3);
        for item in &s {
            assert!(collection.contains(item));
        }
        let mut sorted = s.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), s.len());
    }

    #[test]
    fn uuid_v4_format() {
        let mut r = Rng::new(42);
        let id = r.uuid_v4();
        assert_eq!(id.len(), 36);
        assert_eq!(id.chars().nth(8), Some('-'));
        assert_eq!(id.chars().nth(13), Some('-'));
        assert_eq!(id.chars().nth(14), Some('4')); // version 4
        assert_eq!(id.chars().nth(18), Some('-'));
        assert_eq!(id.chars().nth(23), Some('-'));
        let variant = id.chars().nth(19).unwrap();
        assert!(
            "89ab".contains(variant),
            "Invalid variant char: {variant}"
        );
    }

    #[test]
    fn fill_bytes() {
        let mut r = Rng::new(42);
        let mut buf = [0u8; 20];
        r.fill_bytes(&mut buf);
        assert!(buf.iter().any(|b| *b != 0));
    }

    #[test]
    fn state_round_trip() {
        let mut r1 = Rng::new(42);
        for _ in 0..50 {
            r1.next_u64();
        }
        let state = r1.state();
        let mut r2 = Rng::from_state(state);
        for _ in 0..50 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }
}
