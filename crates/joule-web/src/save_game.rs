//! Game save/load system — versioned binary format, CRC32, save slots.
//!
//! Replaces save-state.js / FileSaver.js with pure Rust.
//! Versioned binary format with magic number, CRC32 checksums,
//! numbered/named save slots, metadata, data sections,
//! version migration, corruption detection, and auto-save.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SaveError {
    CorruptedData(String),
    InvalidMagic,
    ChecksumMismatch { expected: u32, actual: u32 },
    VersionTooNew { save_version: u32, max_supported: u32 },
    SlotNotFound(String),
    SlotFull(String),
    SectionNotFound(String),
    SerializationError(String),
    DataTooLarge { size: usize, limit: usize },
}

impl fmt::Display for SaveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CorruptedData(msg) => write!(f, "corrupted save data: {msg}"),
            Self::InvalidMagic => write!(f, "invalid save file magic number"),
            Self::ChecksumMismatch { expected, actual } => {
                write!(f, "checksum mismatch: expected {expected:#010x}, got {actual:#010x}")
            }
            Self::VersionTooNew { save_version, max_supported } => {
                write!(f, "save version {save_version} > max supported {max_supported}")
            }
            Self::SlotNotFound(name) => write!(f, "save slot not found: {name}"),
            Self::SlotFull(name) => write!(f, "save slot is full: {name}"),
            Self::SectionNotFound(name) => write!(f, "section not found: {name}"),
            Self::SerializationError(msg) => write!(f, "serialization error: {msg}"),
            Self::DataTooLarge { size, limit } => {
                write!(f, "data too large: {size} bytes (limit {limit})")
            }
        }
    }
}

impl std::error::Error for SaveError {}

// ── CRC32 ───────────────────────────────────────────────────────

const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFFu32;
    for &byte in data {
        let idx = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32_TABLE[idx];
    }
    crc ^ 0xFFFFFFFF
}

// ── Constants ───────────────────────────────────────────────────

const MAGIC: [u8; 4] = [0x4A, 0x53, 0x41, 0x56]; // "JSAV"
const CURRENT_VERSION: u32 = 3;
const MAX_SAVE_SIZE: usize = 16 * 1024 * 1024; // 16 MB
const QUICKSAVE_SLOT: &str = "__quicksave__";

// ── Data Sections ───────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct DataSection {
    pub name: String,
    pub data: Vec<u8>,
}

impl DataSection {
    pub fn new(name: &str, data: Vec<u8>) -> Self {
        Self { name: name.to_string(), data }
    }

    fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let name_bytes = self.name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf.extend_from_slice(&(self.data.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.data);
        buf
    }

    fn decode(data: &[u8], offset: &mut usize) -> Result<Self, SaveError> {
        if *offset + 4 > data.len() {
            return Err(SaveError::CorruptedData("truncated section name length".into()));
        }
        let name_len = u32::from_le_bytes([
            data[*offset], data[*offset + 1], data[*offset + 2], data[*offset + 3],
        ]) as usize;
        *offset += 4;
        if *offset + name_len > data.len() {
            return Err(SaveError::CorruptedData("truncated section name".into()));
        }
        let name = String::from_utf8(data[*offset..*offset + name_len].to_vec())
            .map_err(|_| SaveError::CorruptedData("invalid section name encoding".into()))?;
        *offset += name_len;
        if *offset + 4 > data.len() {
            return Err(SaveError::CorruptedData("truncated section data length".into()));
        }
        let data_len = u32::from_le_bytes([
            data[*offset], data[*offset + 1], data[*offset + 2], data[*offset + 3],
        ]) as usize;
        *offset += 4;
        if *offset + data_len > data.len() {
            return Err(SaveError::CorruptedData("truncated section data".into()));
        }
        let section_data = data[*offset..*offset + data_len].to_vec();
        *offset += data_len;
        Ok(Self { name, data: section_data })
    }
}

// ── Save Metadata ───────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct SaveMetadata {
    pub timestamp_ms: u64,
    pub play_time_ms: u64,
    pub player_level: u32,
    pub player_name: String,
    pub screenshot_placeholder: Vec<u8>,
}

impl SaveMetadata {
    pub fn new(timestamp_ms: u64) -> Self {
        Self {
            timestamp_ms,
            play_time_ms: 0,
            player_level: 1,
            player_name: String::new(),
            screenshot_placeholder: Vec::new(),
        }
    }

    pub fn with_play_time(mut self, ms: u64) -> Self { self.play_time_ms = ms; self }
    pub fn with_level(mut self, lvl: u32) -> Self { self.player_level = lvl; self }
    pub fn with_name(mut self, name: &str) -> Self { self.player_name = name.to_string(); self }
    pub fn with_screenshot(mut self, data: Vec<u8>) -> Self { self.screenshot_placeholder = data; self }

    fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&self.timestamp_ms.to_le_bytes());
        buf.extend_from_slice(&self.play_time_ms.to_le_bytes());
        buf.extend_from_slice(&self.player_level.to_le_bytes());
        let name_bytes = self.player_name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf.extend_from_slice(&(self.screenshot_placeholder.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.screenshot_placeholder);
        buf
    }

    fn decode(data: &[u8], offset: &mut usize) -> Result<Self, SaveError> {
        let err = |msg: &str| SaveError::CorruptedData(msg.to_string());
        if *offset + 20 > data.len() {
            return Err(err("truncated metadata header"));
        }
        let timestamp_ms = u64::from_le_bytes(data[*offset..*offset + 8].try_into().unwrap());
        *offset += 8;
        let play_time_ms = u64::from_le_bytes(data[*offset..*offset + 8].try_into().unwrap());
        *offset += 8;
        let player_level = u32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap());
        *offset += 4;
        if *offset + 4 > data.len() { return Err(err("truncated name length")); }
        let name_len = u32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap()) as usize;
        *offset += 4;
        if *offset + name_len > data.len() { return Err(err("truncated name")); }
        let player_name = String::from_utf8(data[*offset..*offset + name_len].to_vec())
            .map_err(|_| err("invalid name encoding"))?;
        *offset += name_len;
        if *offset + 4 > data.len() { return Err(err("truncated screenshot length")); }
        let screenshot_len = u32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap()) as usize;
        *offset += 4;
        if *offset + screenshot_len > data.len() { return Err(err("truncated screenshot")); }
        let screenshot_placeholder = data[*offset..*offset + screenshot_len].to_vec();
        *offset += screenshot_len;
        Ok(Self { timestamp_ms, play_time_ms, player_level, player_name, screenshot_placeholder })
    }
}

// ── Save File ───────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct SaveFile {
    pub version: u32,
    pub metadata: SaveMetadata,
    pub sections: Vec<DataSection>,
}

impl SaveFile {
    pub fn new(metadata: SaveMetadata) -> Self {
        Self { version: CURRENT_VERSION, metadata, sections: Vec::new() }
    }

    pub fn add_section(&mut self, section: DataSection) {
        self.sections.push(section);
    }

    pub fn get_section(&self, name: &str) -> Option<&DataSection> {
        self.sections.iter().find(|s| s.name == name)
    }

    pub fn remove_section(&mut self, name: &str) -> Option<DataSection> {
        if let Some(pos) = self.sections.iter().position(|s| s.name == name) {
            Some(self.sections.remove(pos))
        } else {
            None
        }
    }

    /// Encode to binary format.
    pub fn encode(&self) -> Result<Vec<u8>, SaveError> {
        let mut buf = Vec::new();
        // Magic (4 bytes)
        buf.extend_from_slice(&MAGIC);
        // Version (4 bytes)
        buf.extend_from_slice(&self.version.to_le_bytes());
        // Placeholder for checksum (4 bytes) — filled at end
        let checksum_offset = buf.len();
        buf.extend_from_slice(&[0u8; 4]);
        // Metadata
        buf.extend_from_slice(&self.metadata.encode());
        // Section count
        buf.extend_from_slice(&(self.sections.len() as u32).to_le_bytes());
        // Sections
        for section in &self.sections {
            buf.extend_from_slice(&section.encode());
        }
        // Check size
        if buf.len() > MAX_SAVE_SIZE {
            return Err(SaveError::DataTooLarge { size: buf.len(), limit: MAX_SAVE_SIZE });
        }
        // Compute checksum over everything after the checksum field
        let checksum = crc32(&buf[checksum_offset + 4..]);
        buf[checksum_offset..checksum_offset + 4].copy_from_slice(&checksum.to_le_bytes());
        Ok(buf)
    }

    /// Decode from binary format.
    pub fn decode(data: &[u8]) -> Result<Self, SaveError> {
        if data.len() < 12 {
            return Err(SaveError::CorruptedData("file too small".into()));
        }
        // Magic
        if data[0..4] != MAGIC {
            return Err(SaveError::InvalidMagic);
        }
        // Version
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version > CURRENT_VERSION {
            return Err(SaveError::VersionTooNew { save_version: version, max_supported: CURRENT_VERSION });
        }
        // Checksum
        let stored_checksum = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let computed_checksum = crc32(&data[12..]);
        if stored_checksum != computed_checksum {
            return Err(SaveError::ChecksumMismatch {
                expected: stored_checksum, actual: computed_checksum,
            });
        }
        let mut offset = 12;
        // Metadata
        let metadata = SaveMetadata::decode(data, &mut offset)?;
        // Apply version migrations
        let metadata = if version < 2 {
            // V1 didn't have play_time, default to 0
            SaveMetadata { play_time_ms: 0, ..metadata }
        } else {
            metadata
        };
        // Section count
        if offset + 4 > data.len() {
            return Err(SaveError::CorruptedData("truncated section count".into()));
        }
        let section_count = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let mut sections = Vec::with_capacity(section_count);
        for _ in 0..section_count {
            sections.push(DataSection::decode(data, &mut offset)?);
        }
        Ok(Self { version, metadata, sections })
    }

    pub fn file_size(&self) -> Result<usize, SaveError> {
        Ok(self.encode()?.len())
    }
}

// ── Save Slot Manager ───────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SaveSlotManager {
    slots: HashMap<String, Vec<u8>>,
    max_slots: usize,
    auto_save_interval_ms: u64,
    last_auto_save_ms: u64,
}

impl SaveSlotManager {
    pub fn new(max_slots: usize) -> Self {
        Self {
            slots: HashMap::new(),
            max_slots,
            auto_save_interval_ms: 300_000, // 5 minutes
            last_auto_save_ms: 0,
        }
    }

    pub fn with_auto_save_interval(mut self, interval_ms: u64) -> Self {
        self.auto_save_interval_ms = interval_ms;
        self
    }

    pub fn save_to_slot(&mut self, slot_name: &str, save: &SaveFile) -> Result<usize, SaveError> {
        // quicksave doesn't count toward max slots
        if slot_name != QUICKSAVE_SLOT && !self.slots.contains_key(slot_name) && self.slots.len() >= self.max_slots {
            return Err(SaveError::SlotFull(slot_name.to_string()));
        }
        let data = save.encode()?;
        let size = data.len();
        self.slots.insert(slot_name.to_string(), data);
        Ok(size)
    }

    pub fn load_from_slot(&self, slot_name: &str) -> Result<SaveFile, SaveError> {
        let data = self.slots.get(slot_name)
            .ok_or_else(|| SaveError::SlotNotFound(slot_name.to_string()))?;
        SaveFile::decode(data)
    }

    pub fn delete_slot(&mut self, slot_name: &str) -> Result<(), SaveError> {
        if self.slots.remove(slot_name).is_none() {
            return Err(SaveError::SlotNotFound(slot_name.to_string()));
        }
        Ok(())
    }

    pub fn slot_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.slots.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn slot_exists(&self, name: &str) -> bool {
        self.slots.contains_key(name)
    }

    pub fn slot_size(&self, name: &str) -> Option<usize> {
        self.slots.get(name).map(|d| d.len())
    }

    pub fn total_size(&self) -> usize {
        self.slots.values().map(|d| d.len()).sum()
    }

    pub fn slot_count(&self) -> usize { self.slots.len() }

    pub fn quick_save(&mut self, save: &SaveFile) -> Result<usize, SaveError> {
        self.save_to_slot(QUICKSAVE_SLOT, save)
    }

    pub fn quick_load(&self) -> Result<SaveFile, SaveError> {
        self.load_from_slot(QUICKSAVE_SLOT)
    }

    pub fn should_auto_save(&self, current_time_ms: u64) -> bool {
        current_time_ms >= self.last_auto_save_ms + self.auto_save_interval_ms
    }

    pub fn auto_save(&mut self, save: &SaveFile, current_time_ms: u64) -> Result<Option<usize>, SaveError> {
        if self.should_auto_save(current_time_ms) {
            self.last_auto_save_ms = current_time_ms;
            let size = self.save_to_slot("__autosave__", save)?;
            Ok(Some(size))
        } else {
            Ok(None)
        }
    }

    pub fn slot_metadata(&self, slot_name: &str) -> Result<SaveMetadata, SaveError> {
        let save = self.load_from_slot(slot_name)?;
        Ok(save.metadata)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_save() -> SaveFile {
        let meta = SaveMetadata::new(1000)
            .with_play_time(3600_000)
            .with_level(25)
            .with_name("TestHero");
        let mut save = SaveFile::new(meta);
        save.add_section(DataSection::new("player", vec![1, 2, 3, 4]));
        save.add_section(DataSection::new("world", vec![10, 20, 30]));
        save.add_section(DataSection::new("quests", vec![100, 200]));
        save
    }

    #[test]
    fn crc32_basic() {
        let c = crc32(b"hello");
        assert_eq!(c, 0x3610a686);
    }

    #[test]
    fn crc32_empty() {
        let c = crc32(b"");
        assert_eq!(c, 0x00000000);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let save = test_save();
        let encoded = save.encode().unwrap();
        let decoded = SaveFile::decode(&encoded).unwrap();
        assert_eq!(decoded.version, CURRENT_VERSION);
        assert_eq!(decoded.metadata.player_name, "TestHero");
        assert_eq!(decoded.metadata.player_level, 25);
        assert_eq!(decoded.metadata.play_time_ms, 3600_000);
        assert_eq!(decoded.sections.len(), 3);
        assert_eq!(decoded.sections[0].name, "player");
        assert_eq!(decoded.sections[0].data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn invalid_magic() {
        let mut data = test_save().encode().unwrap();
        data[0] = 0xFF;
        let err = SaveFile::decode(&data).unwrap_err();
        assert!(matches!(err, SaveError::InvalidMagic));
    }

    #[test]
    fn checksum_mismatch() {
        let mut data = test_save().encode().unwrap();
        // Corrupt the payload
        let last = data.len() - 1;
        data[last] ^= 0xFF;
        let err = SaveFile::decode(&data).unwrap_err();
        assert!(matches!(err, SaveError::ChecksumMismatch { .. }));
    }

    #[test]
    fn version_too_new() {
        let mut data = test_save().encode().unwrap();
        // Set version to 999
        data[4..8].copy_from_slice(&999u32.to_le_bytes());
        // Re-compute checksum won't match, but version check comes first
        let err = SaveFile::decode(&data).unwrap_err();
        assert!(matches!(err, SaveError::VersionTooNew { .. }));
    }

    #[test]
    fn truncated_data() {
        let err = SaveFile::decode(&[0x4A, 0x53]).unwrap_err();
        assert!(matches!(err, SaveError::CorruptedData(_)));
    }

    #[test]
    fn get_section() {
        let save = test_save();
        let section = save.get_section("world").unwrap();
        assert_eq!(section.data, vec![10, 20, 30]);
        assert!(save.get_section("missing").is_none());
    }

    #[test]
    fn remove_section() {
        let mut save = test_save();
        let removed = save.remove_section("world").unwrap();
        assert_eq!(removed.data, vec![10, 20, 30]);
        assert!(save.get_section("world").is_none());
        assert_eq!(save.sections.len(), 2);
    }

    #[test]
    fn file_size() {
        let save = test_save();
        let size = save.file_size().unwrap();
        assert!(size > 12); // at least header
    }

    #[test]
    fn slot_manager_save_load() {
        let mut mgr = SaveSlotManager::new(10);
        let save = test_save();
        mgr.save_to_slot("slot1", &save).unwrap();
        let loaded = mgr.load_from_slot("slot1").unwrap();
        assert_eq!(loaded.metadata.player_name, "TestHero");
    }

    #[test]
    fn slot_not_found() {
        let mgr = SaveSlotManager::new(10);
        let err = mgr.load_from_slot("missing").unwrap_err();
        assert!(matches!(err, SaveError::SlotNotFound(_)));
    }

    #[test]
    fn slot_limit() {
        let mut mgr = SaveSlotManager::new(2);
        let save = test_save();
        mgr.save_to_slot("slot1", &save).unwrap();
        mgr.save_to_slot("slot2", &save).unwrap();
        let err = mgr.save_to_slot("slot3", &save).unwrap_err();
        assert!(matches!(err, SaveError::SlotFull(_)));
    }

    #[test]
    fn overwrite_slot() {
        let mut mgr = SaveSlotManager::new(2);
        let save = test_save();
        mgr.save_to_slot("slot1", &save).unwrap();
        // Overwriting existing slot should work even at limit
        let mut save2 = test_save();
        save2.add_section(DataSection::new("extra", vec![42]));
        mgr.save_to_slot("slot2", &save).unwrap();
        mgr.save_to_slot("slot1", &save2).unwrap();
        let loaded = mgr.load_from_slot("slot1").unwrap();
        assert_eq!(loaded.sections.len(), 4);
    }

    #[test]
    fn delete_slot() {
        let mut mgr = SaveSlotManager::new(10);
        let save = test_save();
        mgr.save_to_slot("slot1", &save).unwrap();
        mgr.delete_slot("slot1").unwrap();
        assert!(!mgr.slot_exists("slot1"));
    }

    #[test]
    fn delete_nonexistent() {
        let mut mgr = SaveSlotManager::new(10);
        let err = mgr.delete_slot("missing").unwrap_err();
        assert!(matches!(err, SaveError::SlotNotFound(_)));
    }

    #[test]
    fn quick_save_load() {
        let mut mgr = SaveSlotManager::new(2);
        let save = test_save();
        mgr.quick_save(&save).unwrap();
        let loaded = mgr.quick_load().unwrap();
        assert_eq!(loaded.metadata.player_name, "TestHero");
    }

    #[test]
    fn auto_save_trigger() {
        let mut mgr = SaveSlotManager::new(10)
            .with_auto_save_interval(60_000);
        let save = test_save();
        // First auto-save at t=60_000 (>= 0 + 60_000)
        let result = mgr.auto_save(&save, 60_000).unwrap();
        assert!(result.is_some());
        // Too early for next auto-save
        let result2 = mgr.auto_save(&save, 90_000).unwrap();
        assert!(result2.is_none());
        // Time for another
        let result3 = mgr.auto_save(&save, 120_000).unwrap();
        assert!(result3.is_some());
    }

    #[test]
    fn should_auto_save() {
        let mgr = SaveSlotManager::new(10)
            .with_auto_save_interval(60_000);
        assert!(mgr.should_auto_save(60_000));
        assert!(!mgr.should_auto_save(30_000));
    }

    #[test]
    fn slot_names_sorted() {
        let mut mgr = SaveSlotManager::new(10);
        let save = test_save();
        mgr.save_to_slot("beta", &save).unwrap();
        mgr.save_to_slot("alpha", &save).unwrap();
        mgr.save_to_slot("gamma", &save).unwrap();
        assert_eq!(mgr.slot_names(), vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn total_size() {
        let mut mgr = SaveSlotManager::new(10);
        let save = test_save();
        let s1 = mgr.save_to_slot("s1", &save).unwrap();
        let s2 = mgr.save_to_slot("s2", &save).unwrap();
        assert_eq!(mgr.total_size(), s1 + s2);
    }

    #[test]
    fn slot_metadata() {
        let mut mgr = SaveSlotManager::new(10);
        let save = test_save();
        mgr.save_to_slot("s1", &save).unwrap();
        let meta = mgr.slot_metadata("s1").unwrap();
        assert_eq!(meta.player_level, 25);
    }

    #[test]
    fn metadata_with_screenshot() {
        let meta = SaveMetadata::new(100)
            .with_screenshot(vec![0xFF, 0xD8, 0xFF]); // fake JPEG header
        let mut save = SaveFile::new(meta);
        save.add_section(DataSection::new("test", vec![1]));
        let encoded = save.encode().unwrap();
        let decoded = SaveFile::decode(&encoded).unwrap();
        assert_eq!(decoded.metadata.screenshot_placeholder, vec![0xFF, 0xD8, 0xFF]);
    }

    #[test]
    fn empty_save() {
        let meta = SaveMetadata::new(0);
        let save = SaveFile::new(meta);
        let encoded = save.encode().unwrap();
        let decoded = SaveFile::decode(&encoded).unwrap();
        assert_eq!(decoded.sections.len(), 0);
        assert_eq!(decoded.metadata.player_name, "");
    }
}
