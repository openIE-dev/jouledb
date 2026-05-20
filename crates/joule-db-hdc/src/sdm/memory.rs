//! Sparse Distributed Memory implementation
//!
//! Pure Rust implementation without WASM dependencies.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use thiserror::Error;

/// SDM-specific errors
#[derive(Error, Debug, Clone)]
pub enum SDMError {
    /// Data size doesn't match the expected dimension
    #[error("Data size mismatch: expected {expected}, got {actual}")]
    DataSizeMismatch {
        /// Expected size in bytes
        expected: usize,
        /// Actual size received
        actual: usize,
    },

    /// Not enough bytes to represent the required dimension
    #[error("Insufficient bytes for dimension: need {needed}, got {actual}")]
    InsufficientBytes {
        /// Bytes needed for the dimension
        needed: usize,
        /// Actual bytes provided
        actual: usize,
    },

    /// Internal lock was poisoned by a panicked thread
    #[error("Lock poisoned")]
    LockPoisoned,
}

/// Statistics about an SDM instance
#[derive(Debug, Clone)]
pub struct SDMStats {
    /// Number of hard locations in the memory
    pub num_locations: usize,
    /// Dimension of address space (bits)
    pub dimension: usize,
    /// Size of data stored at each location (bytes)
    pub data_size: usize,
    /// Hamming distance threshold for activation
    pub activation_radius: usize,
    /// Total number of write operations performed
    pub total_writes: u32,
    /// Number of locations that have been written to
    pub active_locations: usize,
}

/// Result of nearest location search
#[derive(Debug, Clone)]
pub struct NearestLocation {
    /// Index of the nearest hard location
    pub index: usize,
    /// Hamming distance to the query address
    pub distance: usize,
}

/// High-dimensional binary address (simulated with u64 chunks)
#[derive(Clone, Debug)]
pub struct SDMAddress {
    /// Internal bit storage as u64 chunks
    pub bits: Vec<u64>,
    dimension: usize,
}

impl SDMAddress {
    /// Create a new random SDM address using a seed
    pub fn random(dimension: usize, seed: u64) -> Self {
        Self::from_seed(dimension, seed)
    }

    /// Create address from data (content-addressable)
    pub fn from_data(data: &[u8], dimension: usize) -> Self {
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        let seed = hasher.finish();
        Self::from_seed(dimension, seed)
    }

    /// Create from seed (deterministic)
    pub fn from_seed(dimension: usize, seed: u64) -> Self {
        let num_chunks = (dimension + 63) / 64;
        let mut bits = Vec::with_capacity(num_chunks);
        let mut rng = seed;

        for _ in 0..num_chunks {
            // Simple LCG for deterministic random
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            bits.push(rng);
        }

        Self { bits, dimension }
    }

    /// Hamming distance between two addresses
    pub fn hamming_distance(&self, other: &SDMAddress) -> usize {
        self.bits
            .iter()
            .zip(other.bits.iter())
            .map(|(a, b)| (a ^ b).count_ones() as usize)
            .sum()
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Export as bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        self.bits.iter().flat_map(|&b| b.to_le_bytes()).collect()
    }

    /// Import from bytes
    pub fn from_bytes(bytes: &[u8], dimension: usize) -> Result<SDMAddress, SDMError> {
        let needed = ((dimension + 63) / 64) * 8;
        if bytes.len() < needed {
            return Err(SDMError::InsufficientBytes {
                needed,
                actual: bytes.len(),
            });
        }

        let bits: Vec<u64> = bytes
            .chunks(8)
            .map(|chunk| {
                let mut arr = [0u8; 8];
                let len = chunk.len().min(8);
                arr[..len].copy_from_slice(&chunk[..len]);
                u64::from_le_bytes(arr)
            })
            .collect();

        Ok(Self { bits, dimension })
    }
}

/// Hard location in SDM (physical storage)
pub(crate) struct HardLocation {
    pub(crate) address: SDMAddress,
    pub(crate) counters: Vec<i32>,
    pub(crate) write_count: u32,
}

/// Sparse Distributed Memory
///
/// A content-addressable memory that stores patterns in a distributed manner
/// across many "hard locations". Reading/writing to an address activates
/// all hard locations within a certain Hamming distance.
pub struct SparseDistributedMemory {
    locations: Arc<RwLock<Vec<HardLocation>>>,
    dimension: usize,
    data_size: usize,
    activation_radius: usize,
}

impl SparseDistributedMemory {
    /// Create new SDM
    ///
    /// # Arguments
    ///
    /// * `num_locations` - Number of hard locations (more = better capacity)
    /// * `dimension` - Address dimension in bits (256-1024 typical)
    /// * `data_size` - Size of data stored at each location
    pub fn new(num_locations: usize, dimension: usize, data_size: usize) -> Self {
        let locations: Vec<HardLocation> = (0..num_locations)
            .map(|i| HardLocation {
                address: SDMAddress::from_seed(dimension, i as u64 * 12345 + 67890),
                counters: vec![0; data_size],
                write_count: 0,
            })
            .collect();

        // Activation radius: ~45% of dimension ensures reasonable activation
        // For random addresses, expected Hamming distance is dimension/2
        // We want to activate ~10-20% of locations on average
        let activation_radius = (dimension as f64 * 0.45) as usize;

        Self {
            locations: Arc::new(RwLock::new(locations)),
            dimension,
            data_size,
            activation_radius,
        }
    }

    /// Write data to address
    ///
    /// Returns the number of hard locations that were activated.
    pub fn write(&self, address: &SDMAddress, data: &[i8]) -> Result<u32, SDMError> {
        if data.len() != self.data_size {
            return Err(SDMError::DataSizeMismatch {
                expected: self.data_size,
                actual: data.len(),
            });
        }

        let mut locations = self.locations.write().map_err(|_| SDMError::LockPoisoned)?;
        let mut activated = 0u32;

        for loc in locations.iter_mut() {
            let dist = address.hamming_distance(&loc.address);
            if dist < self.activation_radius {
                for (i, &d) in data.iter().enumerate() {
                    loc.counters[i] += d as i32;
                }
                loc.write_count += 1;
                activated += 1;
            }
        }

        Ok(activated)
    }

    /// Write from bytes (convenience)
    ///
    /// Converts bytes to bipolar representation (-128 to 127).
    pub fn write_bytes(&self, address: &SDMAddress, data: &[u8]) -> Result<u32, SDMError> {
        // Convert u8 to bipolar i8 (-128 to 127, centered around 0)
        let bipolar: Vec<i8> = data
            .iter()
            .map(|&b| ((b as i16) - 128) as i8)
            .take(self.data_size)
            .collect();

        let mut padded = bipolar;
        padded.resize(self.data_size, 0);

        self.write(address, &padded)
    }

    /// Read data from address
    ///
    /// Returns bipolar data (-1 or +1) based on accumulated counters.
    pub fn read(&self, address: &SDMAddress) -> Vec<i8> {
        let locations = self.locations.read().unwrap();
        let mut sum = vec![0i32; self.data_size];

        for loc in locations.iter() {
            let dist = address.hamming_distance(&loc.address);
            if dist < self.activation_radius {
                for (i, &c) in loc.counters.iter().enumerate() {
                    sum[i] += c;
                }
            }
        }

        // Threshold to bipolar
        sum.iter().map(|&s| if s >= 0 { 1 } else { -1 }).collect()
    }

    /// Read as bytes (convenience)
    pub fn read_bytes(&self, address: &SDMAddress) -> Vec<u8> {
        let bipolar = self.read(address);
        bipolar.iter().map(|&b| ((b as i16) + 128) as u8).collect()
    }

    /// Find k nearest hard locations
    pub fn nearest_locations(&self, address: &SDMAddress, k: usize) -> Vec<NearestLocation> {
        let locations = self.locations.read().unwrap();
        let mut distances: Vec<(usize, usize)> = locations
            .iter()
            .enumerate()
            .map(|(i, loc)| (i, address.hamming_distance(&loc.address)))
            .collect();

        distances.sort_by_key(|(_, d)| *d);

        distances
            .into_iter()
            .take(k)
            .map(|(index, distance)| NearestLocation { index, distance })
            .collect()
    }

    /// Content-addressable lookup
    pub fn content_lookup(&self, query_data: &[u8]) -> Vec<i8> {
        let address = SDMAddress::from_data(query_data, self.dimension);
        self.read(&address)
    }

    /// Get statistics
    pub fn stats(&self) -> SDMStats {
        let locations = self.locations.read().unwrap();
        let total_writes: u32 = locations.iter().map(|l| l.write_count).sum();
        let active_locations = locations.iter().filter(|l| l.write_count > 0).count();

        SDMStats {
            num_locations: locations.len(),
            dimension: self.dimension,
            data_size: self.data_size,
            activation_radius: self.activation_radius,
            total_writes,
            active_locations,
        }
    }

    /// Set activation radius (for tuning)
    pub fn set_activation_radius(&mut self, radius: usize) {
        self.activation_radius = radius.min(self.dimension);
    }

    /// Get current activation radius
    pub fn activation_radius(&self) -> usize {
        self.activation_radius
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get data size
    pub fn data_size(&self) -> usize {
        self.data_size
    }

    /// Direct access to hard locations (for attention bridge).
    pub fn locations_ref(&self) -> &Arc<RwLock<Vec<HardLocation>>> {
        &self.locations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdm_address_from_seed() {
        let addr1 = SDMAddress::from_seed(256, 12345);
        let addr2 = SDMAddress::from_seed(256, 12345);
        let addr3 = SDMAddress::from_seed(256, 12346);

        // Same seed = same address
        assert_eq!(addr1.hamming_distance(&addr2), 0);
        // Different seed = different address
        assert!(addr1.hamming_distance(&addr3) > 0);
    }

    #[test]
    fn test_sdm_address_from_data() {
        let addr1 = SDMAddress::from_data(b"hello", 256);
        let addr2 = SDMAddress::from_data(b"hello", 256);
        let addr3 = SDMAddress::from_data(b"world", 256);

        // Same data = same address
        assert_eq!(addr1.hamming_distance(&addr2), 0);
        // Different data = different address
        assert!(addr1.hamming_distance(&addr3) > 0);
    }

    #[test]
    fn test_sdm_write_read() {
        // Use more locations and larger activation radius for reliable activation
        let mut sdm = SparseDistributedMemory::new(10000, 256, 64);
        // Set activation radius to ~50% of dimension for better coverage
        sdm.set_activation_radius(128);

        let address = SDMAddress::from_data(b"test", 256);

        // Create bipolar data
        let data: Vec<i8> = (0..64).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();

        let activated = sdm.write(&address, &data).unwrap();
        assert!(
            activated > 0,
            "Should activate at least one location (got {})",
            activated
        );

        let read_data = sdm.read(&address);
        assert_eq!(read_data.len(), 64);

        // Data should be mostly similar (SDM is approximate)
        let matching: usize = data
            .iter()
            .zip(read_data.iter())
            .filter(|(a, b)| a == b)
            .count();
        assert!(
            matching > 32,
            "Should recall more than half correctly (got {}/64)",
            matching
        );
    }

    #[test]
    fn test_sdm_content_lookup() {
        let sdm = SparseDistributedMemory::new(1000, 256, 128);
        let data = b"Hello, SDM content addressable memory!";
        let address = SDMAddress::from_data(data, 256);

        sdm.write_bytes(&address, data).unwrap();

        // Content lookup should activate similar locations
        let result = sdm.content_lookup(data);
        assert_eq!(result.len(), 128);
    }

    #[test]
    fn test_sdm_nearest_locations() {
        let sdm = SparseDistributedMemory::new(100, 256, 64);
        let address = SDMAddress::from_data(b"query", 256);

        let nearest = sdm.nearest_locations(&address, 5);
        assert_eq!(nearest.len(), 5);

        // Should be sorted by distance
        for i in 1..nearest.len() {
            assert!(nearest[i].distance >= nearest[i - 1].distance);
        }
    }

    #[test]
    fn test_sdm_stats() {
        let mut sdm = SparseDistributedMemory::new(1000, 256, 64);
        // Use larger activation radius to ensure writes activate locations
        sdm.set_activation_radius(128);

        let stats = sdm.stats();

        assert_eq!(stats.num_locations, 1000);
        assert_eq!(stats.dimension, 256);
        assert_eq!(stats.data_size, 64);
        assert_eq!(stats.total_writes, 0);
        assert_eq!(stats.active_locations, 0);

        // Write something
        let address = SDMAddress::from_data(b"test", 256);
        let data: Vec<i8> = vec![1; 64];
        let activated = sdm.write(&address, &data).unwrap();

        let stats = sdm.stats();
        assert!(
            stats.total_writes > 0,
            "Should have writes (activated {} locations)",
            activated
        );
        assert!(stats.active_locations > 0, "Should have active locations");
    }

    #[test]
    fn test_sdm_bytes_roundtrip() {
        let addr = SDMAddress::from_seed(256, 42);
        let bytes = addr.to_bytes();
        let addr2 = SDMAddress::from_bytes(&bytes, 256).unwrap();

        assert_eq!(addr.hamming_distance(&addr2), 0);
    }

    #[test]
    fn test_data_size_mismatch() {
        let sdm = SparseDistributedMemory::new(100, 256, 64);
        let address = SDMAddress::from_data(b"test", 256);
        let wrong_size_data: Vec<i8> = vec![1; 32]; // Wrong size

        let result = sdm.write(&address, &wrong_size_data);
        assert!(matches!(result, Err(SDMError::DataSizeMismatch { .. })));
    }
}
