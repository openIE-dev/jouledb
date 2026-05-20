//! WASM linear memory — page-based allocation (64 KB pages), grow, bounds
//! checking, load/store (i32/i64/f32/f64), memory.fill, memory.copy,
//! initial/max pages, memory protection zones.

use std::fmt;

// ── Constants ──────────────────────────────────────────────────────────────

/// WASM page size: 64 KiB.
pub const PAGE_SIZE: usize = 65536;

/// Maximum pages allowed by the WASM spec (4 GiB / 64 KiB).
pub const MAX_PAGES_SPEC: u32 = 65536;

// ── Errors ─────────────────────────────────────────────────────────────────

/// Memory operation errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryError {
    OutOfBounds {
        offset: usize,
        size: usize,
        mem_size: usize,
    },
    GrowFailed {
        current: u32,
        requested_delta: u32,
        max: u32,
    },
    OverlapCopy {
        src: usize,
        dst: usize,
        len: usize,
    },
    ProtectedRegion {
        offset: usize,
        size: usize,
        zone_start: usize,
        zone_end: usize,
    },
    InvalidAlignment {
        offset: usize,
        required: usize,
    },
    ZeroPages,
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBounds {
                offset,
                size,
                mem_size,
            } => write!(
                f,
                "out of bounds: offset {offset} + size {size} > memory size {mem_size}"
            ),
            Self::GrowFailed {
                current,
                requested_delta,
                max,
            } => write!(
                f,
                "grow failed: {current} + {requested_delta} exceeds max {max}"
            ),
            Self::OverlapCopy { src, dst, len } => {
                write!(f, "overlapping copy: src={src}, dst={dst}, len={len}")
            }
            Self::ProtectedRegion {
                offset,
                size,
                zone_start,
                zone_end,
            } => write!(
                f,
                "access to protected region: [{offset}..{}] overlaps zone [{zone_start}..{zone_end}]",
                offset + size
            ),
            Self::InvalidAlignment { offset, required } => {
                write!(f, "invalid alignment: offset {offset} not aligned to {required}")
            }
            Self::ZeroPages => write!(f, "initial pages must be > 0 or explicitly 0"),
        }
    }
}

// ── Protection Zone ────────────────────────────────────────────────────────

/// A protection zone that forbids reads and writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtectionZone {
    pub start: usize,
    pub end: usize,
    pub label: String,
}

impl ProtectionZone {
    pub fn new(start: usize, end: usize, label: impl Into<String>) -> Self {
        Self {
            start,
            end: end.max(start),
            label: label.into(),
        }
    }

    /// Check if a byte range overlaps this zone.
    pub fn overlaps(&self, offset: usize, size: usize) -> bool {
        let access_end = offset + size;
        offset < self.end && access_end > self.start
    }
}

// ── Memory Statistics ──────────────────────────────────────────────────────

/// Memory usage statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryStats {
    pub current_pages: u32,
    pub max_pages: u32,
    pub byte_size: usize,
    pub grows: u64,
    pub loads: u64,
    pub stores: u64,
    pub fills: u64,
    pub copies: u64,
    pub protection_zones: usize,
}

// ── Linear Memory ──────────────────────────────────────────────────────────

/// WASM linear memory.
pub struct LinearMemory {
    data: Vec<u8>,
    current_pages: u32,
    max_pages: u32,
    protection_zones: Vec<ProtectionZone>,
    stats_grows: u64,
    stats_loads: u64,
    stats_stores: u64,
    stats_fills: u64,
    stats_copies: u64,
}

impl LinearMemory {
    /// Create a new linear memory with the given initial and maximum page counts.
    pub fn new(initial_pages: u32, max_pages: u32) -> Self {
        let max_pages = max_pages.min(MAX_PAGES_SPEC);
        let initial_pages = initial_pages.min(max_pages);
        let byte_size = initial_pages as usize * PAGE_SIZE;
        Self {
            data: vec![0u8; byte_size],
            current_pages: initial_pages,
            max_pages,
            protection_zones: Vec::new(),
            stats_grows: 0,
            stats_loads: 0,
            stats_stores: 0,
            stats_fills: 0,
            stats_copies: 0,
        }
    }

    /// Current number of pages.
    pub fn current_pages(&self) -> u32 {
        self.current_pages
    }

    /// Maximum number of pages.
    pub fn max_pages(&self) -> u32 {
        self.max_pages
    }

    /// Current byte size.
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }

    /// Grow memory by `delta_pages`. Returns the previous page count on success,
    /// or an error if the maximum would be exceeded.
    pub fn grow(&mut self, delta_pages: u32) -> Result<u32, MemoryError> {
        let new_pages = self.current_pages.checked_add(delta_pages).ok_or(
            MemoryError::GrowFailed {
                current: self.current_pages,
                requested_delta: delta_pages,
                max: self.max_pages,
            },
        )?;

        if new_pages > self.max_pages {
            return Err(MemoryError::GrowFailed {
                current: self.current_pages,
                requested_delta: delta_pages,
                max: self.max_pages,
            });
        }

        let new_byte_size = new_pages as usize * PAGE_SIZE;
        self.data.resize(new_byte_size, 0);
        let prev = self.current_pages;
        self.current_pages = new_pages;
        self.stats_grows += 1;
        Ok(prev)
    }

    // ── Bounds checking ────────────────────────────────────────────────

    /// Check that [offset, offset + size) is within bounds.
    fn check_bounds(&self, offset: usize, size: usize) -> Result<(), MemoryError> {
        let end = offset.checked_add(size).ok_or(MemoryError::OutOfBounds {
            offset,
            size,
            mem_size: self.data.len(),
        })?;
        if end > self.data.len() {
            return Err(MemoryError::OutOfBounds {
                offset,
                size,
                mem_size: self.data.len(),
            });
        }
        Ok(())
    }

    /// Check that an access does not hit a protection zone.
    fn check_protection(&self, offset: usize, size: usize) -> Result<(), MemoryError> {
        for zone in &self.protection_zones {
            if zone.overlaps(offset, size) {
                return Err(MemoryError::ProtectedRegion {
                    offset,
                    size,
                    zone_start: zone.start,
                    zone_end: zone.end,
                });
            }
        }
        Ok(())
    }

    // ── Load operations ────────────────────────────────────────────────

    /// Load a single byte.
    pub fn load_u8(&mut self, offset: usize) -> Result<u8, MemoryError> {
        self.check_bounds(offset, 1)?;
        self.check_protection(offset, 1)?;
        self.stats_loads += 1;
        Ok(self.data[offset])
    }

    /// Load an i32 (little-endian).
    pub fn load_i32(&mut self, offset: usize) -> Result<i32, MemoryError> {
        self.check_bounds(offset, 4)?;
        self.check_protection(offset, 4)?;
        self.stats_loads += 1;
        let bytes: [u8; 4] = [
            self.data[offset],
            self.data[offset + 1],
            self.data[offset + 2],
            self.data[offset + 3],
        ];
        Ok(i32::from_le_bytes(bytes))
    }

    /// Load an i64 (little-endian).
    pub fn load_i64(&mut self, offset: usize) -> Result<i64, MemoryError> {
        self.check_bounds(offset, 8)?;
        self.check_protection(offset, 8)?;
        self.stats_loads += 1;
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.data[offset..offset + 8]);
        Ok(i64::from_le_bytes(bytes))
    }

    /// Load an f32 (little-endian).
    pub fn load_f32(&mut self, offset: usize) -> Result<f32, MemoryError> {
        self.check_bounds(offset, 4)?;
        self.check_protection(offset, 4)?;
        self.stats_loads += 1;
        let bytes: [u8; 4] = [
            self.data[offset],
            self.data[offset + 1],
            self.data[offset + 2],
            self.data[offset + 3],
        ];
        Ok(f32::from_le_bytes(bytes))
    }

    /// Load an f64 (little-endian).
    pub fn load_f64(&mut self, offset: usize) -> Result<f64, MemoryError> {
        self.check_bounds(offset, 8)?;
        self.check_protection(offset, 8)?;
        self.stats_loads += 1;
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.data[offset..offset + 8]);
        Ok(f64::from_le_bytes(bytes))
    }

    /// Load a byte slice.
    pub fn load_bytes(&mut self, offset: usize, len: usize) -> Result<Vec<u8>, MemoryError> {
        self.check_bounds(offset, len)?;
        self.check_protection(offset, len)?;
        self.stats_loads += 1;
        Ok(self.data[offset..offset + len].to_vec())
    }

    // ── Store operations ───────────────────────────────────────────────

    /// Store a single byte.
    pub fn store_u8(&mut self, offset: usize, val: u8) -> Result<(), MemoryError> {
        self.check_bounds(offset, 1)?;
        self.check_protection(offset, 1)?;
        self.stats_stores += 1;
        self.data[offset] = val;
        Ok(())
    }

    /// Store an i32 (little-endian).
    pub fn store_i32(&mut self, offset: usize, val: i32) -> Result<(), MemoryError> {
        self.check_bounds(offset, 4)?;
        self.check_protection(offset, 4)?;
        self.stats_stores += 1;
        let bytes = val.to_le_bytes();
        self.data[offset..offset + 4].copy_from_slice(&bytes);
        Ok(())
    }

    /// Store an i64 (little-endian).
    pub fn store_i64(&mut self, offset: usize, val: i64) -> Result<(), MemoryError> {
        self.check_bounds(offset, 8)?;
        self.check_protection(offset, 8)?;
        self.stats_stores += 1;
        let bytes = val.to_le_bytes();
        self.data[offset..offset + 8].copy_from_slice(&bytes);
        Ok(())
    }

    /// Store an f32 (little-endian).
    pub fn store_f32(&mut self, offset: usize, val: f32) -> Result<(), MemoryError> {
        self.check_bounds(offset, 4)?;
        self.check_protection(offset, 4)?;
        self.stats_stores += 1;
        let bytes = val.to_le_bytes();
        self.data[offset..offset + 4].copy_from_slice(&bytes);
        Ok(())
    }

    /// Store an f64 (little-endian).
    pub fn store_f64(&mut self, offset: usize, val: f64) -> Result<(), MemoryError> {
        self.check_bounds(offset, 8)?;
        self.check_protection(offset, 8)?;
        self.stats_stores += 1;
        let bytes = val.to_le_bytes();
        self.data[offset..offset + 8].copy_from_slice(&bytes);
        Ok(())
    }

    /// Store a byte slice.
    pub fn store_bytes(&mut self, offset: usize, data: &[u8]) -> Result<(), MemoryError> {
        self.check_bounds(offset, data.len())?;
        self.check_protection(offset, data.len())?;
        self.stats_stores += 1;
        self.data[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    // ── Bulk operations ────────────────────────────────────────────────

    /// memory.fill — fill a region with a byte value.
    pub fn fill(&mut self, offset: usize, len: usize, val: u8) -> Result<(), MemoryError> {
        self.check_bounds(offset, len)?;
        self.check_protection(offset, len)?;
        self.stats_fills += 1;
        for byte in &mut self.data[offset..offset + len] {
            *byte = val;
        }
        Ok(())
    }

    /// memory.copy — copy bytes within memory (handles overlap correctly).
    pub fn copy_within(
        &mut self,
        dst: usize,
        src: usize,
        len: usize,
    ) -> Result<(), MemoryError> {
        self.check_bounds(src, len)?;
        self.check_bounds(dst, len)?;
        self.check_protection(src, len)?;
        self.check_protection(dst, len)?;
        self.stats_copies += 1;
        // Safe even with overlap because copy_within handles it.
        self.data.copy_within(src..src + len, dst);
        Ok(())
    }

    /// Initialize a region from an external data segment.
    pub fn init_segment(
        &mut self,
        dst_offset: usize,
        data: &[u8],
    ) -> Result<(), MemoryError> {
        self.check_bounds(dst_offset, data.len())?;
        self.data[dst_offset..dst_offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    // ── Protection zones ───────────────────────────────────────────────

    /// Add a protection zone. Any load/store overlapping this region will error.
    pub fn add_protection_zone(&mut self, zone: ProtectionZone) {
        self.protection_zones.push(zone);
    }

    /// Remove a protection zone by label.
    pub fn remove_protection_zone(&mut self, label: &str) -> bool {
        let before = self.protection_zones.len();
        self.protection_zones.retain(|z| z.label != label);
        self.protection_zones.len() < before
    }

    /// List all protection zones.
    pub fn protection_zones(&self) -> &[ProtectionZone] {
        &self.protection_zones
    }

    // ── Raw access (for debugging) ─────────────────────────────────────

    /// Get a read-only view of the raw data.
    pub fn raw_data(&self) -> &[u8] {
        &self.data
    }

    /// Hex dump of a region.
    pub fn hex_dump(&self, offset: usize, len: usize) -> Result<String, MemoryError> {
        self.check_bounds(offset, len)?;
        let mut out = String::new();
        for (i, &byte) in self.data[offset..offset + len].iter().enumerate() {
            if i > 0 && i % 16 == 0 {
                out.push('\n');
            } else if i > 0 {
                out.push(' ');
            }
            out.push_str(&format!("{byte:02x}"));
        }
        Ok(out)
    }

    // ── Statistics ─────────────────────────────────────────────────────

    /// Get memory statistics.
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            current_pages: self.current_pages,
            max_pages: self.max_pages,
            byte_size: self.data.len(),
            grows: self.stats_grows,
            loads: self.stats_loads,
            stores: self.stats_stores,
            fills: self.stats_fills,
            copies: self.stats_copies,
            protection_zones: self.protection_zones.len(),
        }
    }
}

impl fmt::Debug for LinearMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LinearMemory")
            .field("current_pages", &self.current_pages)
            .field("max_pages", &self.max_pages)
            .field("byte_size", &self.data.len())
            .field("protection_zones", &self.protection_zones.len())
            .finish()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_memory_size() {
        let mem = LinearMemory::new(2, 10);
        assert_eq!(mem.current_pages(), 2);
        assert_eq!(mem.byte_size(), 2 * PAGE_SIZE);
    }

    #[test]
    fn grow_success() {
        let mut mem = LinearMemory::new(1, 4);
        let prev = mem.grow(2).unwrap();
        assert_eq!(prev, 1);
        assert_eq!(mem.current_pages(), 3);
        assert_eq!(mem.byte_size(), 3 * PAGE_SIZE);
    }

    #[test]
    fn grow_beyond_max_fails() {
        let mut mem = LinearMemory::new(1, 2);
        let result = mem.grow(3);
        assert!(result.is_err());
        assert_eq!(mem.current_pages(), 1);
    }

    #[test]
    fn grow_zero_pages() {
        let mut mem = LinearMemory::new(1, 4);
        let prev = mem.grow(0).unwrap();
        assert_eq!(prev, 1);
        assert_eq!(mem.current_pages(), 1);
    }

    #[test]
    fn store_and_load_i32() {
        let mut mem = LinearMemory::new(1, 1);
        mem.store_i32(100, 42).unwrap();
        let val = mem.load_i32(100).unwrap();
        assert_eq!(val, 42);
    }

    #[test]
    fn store_and_load_i64() {
        let mut mem = LinearMemory::new(1, 1);
        mem.store_i64(200, 123456789_i64).unwrap();
        let val = mem.load_i64(200).unwrap();
        assert_eq!(val, 123456789);
    }

    #[test]
    fn store_and_load_f32() {
        let mut mem = LinearMemory::new(1, 1);
        mem.store_f32(0, 3.14_f32).unwrap();
        let val = mem.load_f32(0).unwrap();
        assert!((val - 3.14_f32).abs() < 0.001);
    }

    #[test]
    fn store_and_load_f64() {
        let mut mem = LinearMemory::new(1, 1);
        mem.store_f64(0, std::f64::consts::PI).unwrap();
        let val = mem.load_f64(0).unwrap();
        assert!((val - std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn store_and_load_u8() {
        let mut mem = LinearMemory::new(1, 1);
        mem.store_u8(0, 0xFF).unwrap();
        assert_eq!(mem.load_u8(0).unwrap(), 0xFF);
    }

    #[test]
    fn out_of_bounds_load() {
        let mut mem = LinearMemory::new(1, 1);
        let result = mem.load_i32(PAGE_SIZE - 2);
        assert!(matches!(result, Err(MemoryError::OutOfBounds { .. })));
    }

    #[test]
    fn out_of_bounds_store() {
        let mut mem = LinearMemory::new(1, 1);
        let result = mem.store_i32(PAGE_SIZE, 0);
        assert!(matches!(result, Err(MemoryError::OutOfBounds { .. })));
    }

    #[test]
    fn memory_fill() {
        let mut mem = LinearMemory::new(1, 1);
        mem.fill(0, 16, 0xAB).unwrap();
        for i in 0..16 {
            assert_eq!(mem.load_u8(i).unwrap(), 0xAB);
        }
        // Byte after fill region should be zero.
        assert_eq!(mem.load_u8(16).unwrap(), 0);
    }

    #[test]
    fn memory_copy() {
        let mut mem = LinearMemory::new(1, 1);
        mem.store_bytes(0, &[1, 2, 3, 4]).unwrap();
        mem.copy_within(10, 0, 4).unwrap();
        let copied = mem.load_bytes(10, 4).unwrap();
        assert_eq!(copied, vec![1, 2, 3, 4]);
    }

    #[test]
    fn overlapping_copy() {
        let mut mem = LinearMemory::new(1, 1);
        mem.store_bytes(0, &[1, 2, 3, 4, 5]).unwrap();
        // Copy [0..5] to [2..7] — overlapping.
        mem.copy_within(2, 0, 5).unwrap();
        let result = mem.load_bytes(2, 5).unwrap();
        assert_eq!(result, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn protection_zone_blocks_access() {
        let mut mem = LinearMemory::new(1, 1);
        mem.add_protection_zone(ProtectionZone::new(100, 200, "guard"));
        let result = mem.store_i32(150, 42);
        assert!(matches!(result, Err(MemoryError::ProtectedRegion { .. })));
    }

    #[test]
    fn protection_zone_edge() {
        let mut mem = LinearMemory::new(1, 1);
        mem.add_protection_zone(ProtectionZone::new(100, 200, "guard"));
        // Just before the zone should succeed.
        assert!(mem.store_i32(96, 1).is_ok());
        // Just at zone start should fail.
        assert!(mem.store_i32(100, 1).is_err());
        // Just after the zone should succeed.
        assert!(mem.store_i32(200, 1).is_ok());
    }

    #[test]
    fn remove_protection_zone() {
        let mut mem = LinearMemory::new(1, 1);
        mem.add_protection_zone(ProtectionZone::new(0, 100, "zone1"));
        assert!(mem.store_u8(50, 1).is_err());
        assert!(mem.remove_protection_zone("zone1"));
        assert!(mem.store_u8(50, 1).is_ok());
    }

    #[test]
    fn init_segment() {
        let mut mem = LinearMemory::new(1, 1);
        mem.init_segment(0, &[10, 20, 30]).unwrap();
        assert_eq!(mem.load_u8(0).unwrap(), 10);
        assert_eq!(mem.load_u8(1).unwrap(), 20);
        assert_eq!(mem.load_u8(2).unwrap(), 30);
    }

    #[test]
    fn hex_dump() {
        let mut mem = LinearMemory::new(1, 1);
        mem.store_bytes(0, &[0x00, 0xFF, 0xAB]).unwrap();
        let dump = mem.hex_dump(0, 3).unwrap();
        assert_eq!(dump, "00 ff ab");
    }

    #[test]
    fn stats_tracking() {
        let mut mem = LinearMemory::new(1, 4);
        mem.store_i32(0, 1).unwrap();
        mem.load_i32(0).unwrap();
        mem.grow(1).unwrap();
        mem.fill(0, 4, 0).unwrap();
        mem.copy_within(10, 0, 4).unwrap();
        let stats = mem.stats();
        assert_eq!(stats.stores, 1);
        assert_eq!(stats.loads, 1);
        assert_eq!(stats.grows, 1);
        assert_eq!(stats.fills, 1);
        assert_eq!(stats.copies, 1);
    }

    #[test]
    fn initial_clamped_to_max() {
        let mem = LinearMemory::new(10, 5);
        assert_eq!(mem.current_pages(), 5);
    }

    #[test]
    fn store_and_load_bytes() {
        let mut mem = LinearMemory::new(1, 1);
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        mem.store_bytes(100, &data).unwrap();
        let loaded = mem.load_bytes(100, 8).unwrap();
        assert_eq!(loaded, data);
    }

    #[test]
    fn memory_error_display() {
        let e = MemoryError::OutOfBounds {
            offset: 100,
            size: 4,
            mem_size: 50,
        };
        assert!(e.to_string().contains("out of bounds"));
    }

    #[test]
    fn protection_zone_overlap_detection() {
        let zone = ProtectionZone::new(100, 200, "test");
        assert!(!zone.overlaps(0, 100));
        assert!(zone.overlaps(99, 2));
        assert!(zone.overlaps(100, 1));
        assert!(zone.overlaps(150, 10));
        assert!(zone.overlaps(199, 2));
        assert!(!zone.overlaps(200, 10));
    }

    #[test]
    fn negative_i32_round_trip() {
        let mut mem = LinearMemory::new(1, 1);
        mem.store_i32(0, -12345).unwrap();
        assert_eq!(mem.load_i32(0).unwrap(), -12345);
    }
}
