//! CRC and non-cryptographic hash functions.
//!
//! CRC-32 (ISO 3309), CRC-32C (Castagnoli), CRC-16, Adler-32,
//! FNV-1/FNV-1a (32-bit and 64-bit), DJB2, SDBM — all with
//! streaming update support.

// ── CRC-32 (ISO 3309 / ITU-T V.42) ──────────────────────────────

const CRC32_POLY: u32 = 0xEDB88320; // Reversed polynomial

fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    for i in 0..256u32 {
        let mut crc = i;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ CRC32_POLY;
            } else {
                crc >>= 1;
            }
        }
        table[i as usize] = crc;
    }
    table
}

/// CRC-32 hasher (ISO 3309).
pub struct Crc32 {
    crc: u32,
    table: [u32; 256],
}

impl Crc32 {
    pub fn new() -> Self {
        Self { crc: 0xFFFFFFFF, table: crc32_table() }
    }

    pub fn update(&mut self, data: &[u8]) {
        for &byte in data {
            let idx = ((self.crc ^ byte as u32) & 0xFF) as usize;
            self.crc = (self.crc >> 8) ^ self.table[idx];
        }
    }

    pub fn finalize(&self) -> u32 {
        self.crc ^ 0xFFFFFFFF
    }

    pub fn reset(&mut self) {
        self.crc = 0xFFFFFFFF;
    }
}

/// Compute CRC-32 of a byte slice.
pub fn crc32(data: &[u8]) -> u32 {
    let mut h = Crc32::new();
    h.update(data);
    h.finalize()
}

// ── CRC-32C (Castagnoli) ─────────────────────────────────────────

const CRC32C_POLY: u32 = 0x82F63B78; // Reversed Castagnoli

fn crc32c_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    for i in 0..256u32 {
        let mut crc = i;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ CRC32C_POLY;
            } else {
                crc >>= 1;
            }
        }
        table[i as usize] = crc;
    }
    table
}

/// CRC-32C hasher (Castagnoli).
pub struct Crc32c {
    crc: u32,
    table: [u32; 256],
}

impl Crc32c {
    pub fn new() -> Self {
        Self { crc: 0xFFFFFFFF, table: crc32c_table() }
    }

    pub fn update(&mut self, data: &[u8]) {
        for &byte in data {
            let idx = ((self.crc ^ byte as u32) & 0xFF) as usize;
            self.crc = (self.crc >> 8) ^ self.table[idx];
        }
    }

    pub fn finalize(&self) -> u32 {
        self.crc ^ 0xFFFFFFFF
    }

    pub fn reset(&mut self) {
        self.crc = 0xFFFFFFFF;
    }
}

/// Compute CRC-32C of a byte slice.
pub fn crc32c(data: &[u8]) -> u32 {
    let mut h = Crc32c::new();
    h.update(data);
    h.finalize()
}

// ── CRC-16 (CCITT) ──────────────────────────────────────────────

const CRC16_POLY: u16 = 0xA001; // Reversed polynomial for CRC-16/ARC

/// CRC-16 hasher.
pub struct Crc16 {
    crc: u16,
}

impl Crc16 {
    pub fn new() -> Self {
        Self { crc: 0x0000 }
    }

    pub fn update(&mut self, data: &[u8]) {
        for &byte in data {
            self.crc ^= byte as u16;
            for _ in 0..8 {
                if self.crc & 1 != 0 {
                    self.crc = (self.crc >> 1) ^ CRC16_POLY;
                } else {
                    self.crc >>= 1;
                }
            }
        }
    }

    pub fn finalize(&self) -> u16 {
        self.crc
    }

    pub fn reset(&mut self) {
        self.crc = 0x0000;
    }
}

/// Compute CRC-16 of a byte slice.
pub fn crc16(data: &[u8]) -> u16 {
    let mut h = Crc16::new();
    h.update(data);
    h.finalize()
}

// ── Adler-32 ─────────────────────────────────────────────────────

const ADLER_MOD: u32 = 65521;

/// Adler-32 hasher.
pub struct Adler32 {
    a: u32,
    b: u32,
}

impl Adler32 {
    pub fn new() -> Self {
        Self { a: 1, b: 0 }
    }

    pub fn update(&mut self, data: &[u8]) {
        for &byte in data {
            self.a = (self.a + byte as u32) % ADLER_MOD;
            self.b = (self.b + self.a) % ADLER_MOD;
        }
    }

    pub fn finalize(&self) -> u32 {
        (self.b << 16) | self.a
    }

    pub fn reset(&mut self) {
        self.a = 1;
        self.b = 0;
    }
}

/// Compute Adler-32 of a byte slice.
pub fn adler32(data: &[u8]) -> u32 {
    let mut h = Adler32::new();
    h.update(data);
    h.finalize()
}

// ── FNV-1 / FNV-1a ──────────────────────────────────────────────

const FNV32_OFFSET: u32 = 0x811C9DC5;
const FNV32_PRIME: u32 = 0x01000193;
const FNV64_OFFSET: u64 = 0xCBF29CE484222325;
const FNV64_PRIME: u64 = 0x00000100000001B3;

/// Compute FNV-1 32-bit hash.
pub fn fnv1_32(data: &[u8]) -> u32 {
    let mut hash = FNV32_OFFSET;
    for &byte in data {
        hash = hash.wrapping_mul(FNV32_PRIME);
        hash ^= byte as u32;
    }
    hash
}

/// Compute FNV-1a 32-bit hash.
pub fn fnv1a_32(data: &[u8]) -> u32 {
    let mut hash = FNV32_OFFSET;
    for &byte in data {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(FNV32_PRIME);
    }
    hash
}

/// Compute FNV-1 64-bit hash.
pub fn fnv1_64(data: &[u8]) -> u64 {
    let mut hash = FNV64_OFFSET;
    for &byte in data {
        hash = hash.wrapping_mul(FNV64_PRIME);
        hash ^= byte as u64;
    }
    hash
}

/// Compute FNV-1a 64-bit hash.
pub fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash = FNV64_OFFSET;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV64_PRIME);
    }
    hash
}

// ── DJB2 ─────────────────────────────────────────────────────────

/// Compute DJB2 hash (Daniel J. Bernstein).
pub fn djb2(data: &[u8]) -> u32 {
    let mut hash: u32 = 5381;
    for &byte in data {
        hash = hash.wrapping_shl(5).wrapping_add(hash).wrapping_add(byte as u32);
    }
    hash
}

// ── SDBM ─────────────────────────────────────────────────────────

/// Compute SDBM hash.
pub fn sdbm(data: &[u8]) -> u32 {
    let mut hash: u32 = 0;
    for &byte in data {
        hash = (byte as u32)
            .wrapping_add(hash.wrapping_shl(6))
            .wrapping_add(hash.wrapping_shl(16))
            .wrapping_sub(hash);
    }
    hash
}

// ── Streaming Wrapper ────────────────────────────────────────────

/// Generic streaming hasher trait.
pub trait StreamingHash {
    type Output;
    fn update(&mut self, data: &[u8]);
    fn finalize(&self) -> Self::Output;
    fn reset(&mut self);
}

/// FNV-1a 32-bit streaming hasher.
pub struct Fnv1a32 {
    hash: u32,
}

impl Fnv1a32 {
    pub fn new() -> Self { Self { hash: FNV32_OFFSET } }
}

impl StreamingHash for Fnv1a32 {
    type Output = u32;

    fn update(&mut self, data: &[u8]) {
        for &byte in data {
            self.hash ^= byte as u32;
            self.hash = self.hash.wrapping_mul(FNV32_PRIME);
        }
    }

    fn finalize(&self) -> u32 { self.hash }

    fn reset(&mut self) { self.hash = FNV32_OFFSET; }
}

/// FNV-1a 64-bit streaming hasher.
pub struct Fnv1a64 {
    hash: u64,
}

impl Fnv1a64 {
    pub fn new() -> Self { Self { hash: FNV64_OFFSET } }
}

impl StreamingHash for Fnv1a64 {
    type Output = u64;

    fn update(&mut self, data: &[u8]) {
        for &byte in data {
            self.hash ^= byte as u64;
            self.hash = self.hash.wrapping_mul(FNV64_PRIME);
        }
    }

    fn finalize(&self) -> u64 { self.hash }

    fn reset(&mut self) { self.hash = FNV64_OFFSET; }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32_empty() {
        assert_eq!(crc32(b""), 0x00000000);
    }

    #[test]
    fn test_crc32_known() {
        // Well-known: CRC-32 of "123456789" = 0xCBF43926
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn test_crc32c_known() {
        // CRC-32C of "123456789" = 0xE3069283
        assert_eq!(crc32c(b"123456789"), 0xE3069283);
    }

    #[test]
    fn test_crc32_streaming() {
        let mut h = Crc32::new();
        h.update(b"1234");
        h.update(b"56789");
        assert_eq!(h.finalize(), crc32(b"123456789"));
    }

    #[test]
    fn test_crc16() {
        let c = crc16(b"123456789");
        // CRC-16/ARC of "123456789" = 0xBB3D
        assert_eq!(c, 0xBB3D);
    }

    #[test]
    fn test_adler32_known() {
        // Adler-32 of "Wikipedia" = 0x11E60398
        assert_eq!(adler32(b"Wikipedia"), 0x11E60398);
    }

    #[test]
    fn test_adler32_streaming() {
        let mut h = Adler32::new();
        h.update(b"Wiki");
        h.update(b"pedia");
        assert_eq!(h.finalize(), adler32(b"Wikipedia"));
    }

    #[test]
    fn test_fnv1a_32() {
        let h1 = fnv1a_32(b"hello");
        let h2 = fnv1a_32(b"world");
        assert_ne!(h1, h2);
        // Same input gives same hash
        assert_eq!(fnv1a_32(b"hello"), h1);
    }

    #[test]
    fn test_fnv1a_64() {
        let h = fnv1a_64(b"hello");
        assert_ne!(h, 0);
        assert_eq!(fnv1a_64(b"hello"), h);
    }

    #[test]
    fn test_fnv1_vs_fnv1a() {
        // FNV-1 and FNV-1a should produce different results for same input
        assert_ne!(fnv1_32(b"test"), fnv1a_32(b"test"));
        assert_ne!(fnv1_64(b"test"), fnv1a_64(b"test"));
    }

    #[test]
    fn test_djb2() {
        let h1 = djb2(b"hello");
        let h2 = djb2(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_sdbm() {
        let h1 = sdbm(b"hello");
        let h2 = sdbm(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_streaming_fnv1a32() {
        let mut h = Fnv1a32::new();
        h.update(b"hello");
        h.update(b" world");
        let combined = h.finalize();
        assert_eq!(combined, fnv1a_32(b"hello world"));
    }

    #[test]
    fn test_streaming_fnv1a64() {
        let mut h = Fnv1a64::new();
        h.update(b"hello");
        h.update(b" world");
        assert_eq!(h.finalize(), fnv1a_64(b"hello world"));
    }

    #[test]
    fn test_crc32_reset() {
        let mut h = Crc32::new();
        h.update(b"abc");
        h.reset();
        h.update(b"123456789");
        assert_eq!(h.finalize(), 0xCBF43926);
    }
}
