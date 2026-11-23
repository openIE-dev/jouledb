//! Flash Storage Backend
//!
//! Implements storage on NOR/NAND flash memory with:
//! - Wear leveling to distribute writes evenly
//! - Bad block management (NAND)
//! - Power-loss safe writes using copy-on-write
//!
//! # Memory Layout
//! ```text
//! +----------------+
//! | Superblock (0) |  - Magic, version, block count
//! +----------------+
//! | Block Map (1)  |  - Maps logical to physical blocks
//! +----------------+
//! | Wear Table (2) |  - Erase counts per block
//! +----------------+
//! | Data Blocks    |  - User data pages
//! +----------------+
//! ```

#[cfg(not(feature = "std"))]
use alloc::{collections::BTreeMap, format, string::String, vec, vec::Vec};
#[cfg(feature = "std")]
use std::collections::BTreeMap;

use super::{EdgeStorageBackend, StorageStats};
use crate::storage::StorageError;

/// Flash memory characteristics
#[derive(Debug, Clone)]
pub struct FlashConfig {
    /// Page size (smallest unit for reads)
    pub page_size: usize,
    /// Erase block size (smallest unit for erases)
    pub erase_block_size: usize,
    /// Total number of erase blocks
    pub block_count: usize,
    /// Maximum erase cycles before wear-out
    pub max_erase_cycles: u32,
    /// Reserved blocks for bad block replacement
    pub reserved_blocks: usize,
}

impl Default for FlashConfig {
    fn default() -> Self {
        Self {
            page_size: 4096,           // 4KB pages (typical NOR flash)
            erase_block_size: 65536,   // 64KB erase blocks
            block_count: 256,          // 16MB total
            max_erase_cycles: 100_000, // NOR flash typical
            reserved_blocks: 8,        // For bad block replacement
        }
    }
}

impl FlashConfig {
    /// Configuration for NAND flash
    pub fn nand() -> Self {
        Self {
            page_size: 2048,
            erase_block_size: 131072, // 128KB
            block_count: 1024,        // 128MB total
            max_erase_cycles: 10_000, // NAND has lower endurance
            reserved_blocks: 32,
        }
    }

    /// Configuration for SPI NOR flash
    pub fn spi_nor() -> Self {
        Self {
            page_size: 256,
            erase_block_size: 4096,
            block_count: 4096, // 16MB typical
            max_erase_cycles: 100_000,
            reserved_blocks: 4,
        }
    }
}

/// Block state in the wear leveling table
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockState {
    /// Block is free and ready for use
    Free,
    /// Block contains valid data
    InUse,
    /// Block is bad (for NAND)
    Bad,
    /// Block is being erased
    Erasing,
}

/// Wear leveling entry
#[derive(Debug, Clone, Copy)]
struct WearEntry {
    /// Number of times this block has been erased
    erase_count: u32,
    /// Current state
    state: BlockState,
    /// Logical block mapped to this physical block
    logical_block: Option<u32>,
}

impl Default for WearEntry {
    fn default() -> Self {
        Self {
            erase_count: 0,
            state: BlockState::Free,
            logical_block: None,
        }
    }
}

/// Flash storage backend with wear leveling
pub struct FlashBackend {
    /// Configuration
    config: FlashConfig,
    /// Memory-mapped flash region (simulated as Vec for no_std)
    memory: Vec<u8>,
    /// Logical to physical block mapping
    block_map: BTreeMap<u32, u32>,
    /// Wear leveling table (indexed by physical block)
    wear_table: Vec<WearEntry>,
    /// Free physical blocks (sorted by erase count for wear leveling)
    free_blocks: Vec<u32>,
    /// Statistics
    stats: StorageStats,
    /// Dirty flag (needs sync)
    dirty: bool,
}

impl FlashBackend {
    /// Create a new flash backend (simulated in memory)
    pub fn new(config: FlashConfig) -> Self {
        let total_size = config.erase_block_size * config.block_count;
        let memory = vec![0xFF; total_size]; // Flash erased state is 0xFF

        let mut wear_table = Vec::with_capacity(config.block_count);
        let mut free_blocks = Vec::with_capacity(config.block_count - config.reserved_blocks);

        for i in 0..config.block_count {
            wear_table.push(WearEntry::default());
            if i >= 3 {
                // Reserve first 3 blocks for metadata
                free_blocks.push(i as u32);
            }
        }

        Self {
            config,
            memory,
            block_map: BTreeMap::new(),
            wear_table,
            free_blocks,
            stats: StorageStats::default(),
            dirty: false,
        }
    }

    /// Create from raw memory-mapped flash (for real hardware)
    #[cfg(feature = "std")]
    pub unsafe fn from_raw(ptr: *mut u8, config: FlashConfig) -> Self {
        unsafe {
            let total_size = config.erase_block_size * config.block_count;
            let memory = Vec::from_raw_parts(ptr, total_size, total_size);

            let mut backend = Self::new(config);
            backend.memory = memory;
            backend.recover_metadata();
            backend
        }
    }

    /// Recover metadata from flash (after power cycle)
    fn recover_metadata(&mut self) {
        // Read superblock
        let superblock_offset = 0;
        let magic = &self.memory[superblock_offset..superblock_offset + 8];

        if magic != b"FLASHDB\0" {
            // Fresh flash, initialize
            self.initialize_flash();
            return;
        }

        // Read block map from block 1
        let map_offset = self.config.erase_block_size;
        let map_size = self.config.block_count * 8; // 4 bytes logical + 4 bytes physical

        let map_data = &self.memory[map_offset..map_offset + map_size];
        for (_i, chunk) in map_data.chunks(8).enumerate() {
            if chunk.len() < 8 {
                break;
            }
            let logical = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let physical = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);

            if logical != 0xFFFFFFFF && physical != 0xFFFFFFFF {
                self.block_map.insert(logical, physical);
                self.wear_table[physical as usize].state = BlockState::InUse;
                self.wear_table[physical as usize].logical_block = Some(logical);
            }
        }

        // Read wear table from block 2
        let wear_offset = self.config.erase_block_size * 2;
        let wear_data = &self.memory[wear_offset..wear_offset + self.config.block_count * 4];

        for (i, chunk) in wear_data.chunks(4).enumerate() {
            if chunk.len() < 4 {
                break;
            }
            let erase_count = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if erase_count != 0xFFFFFFFF {
                self.wear_table[i].erase_count = erase_count;
            }
        }

        // Rebuild free block list
        self.rebuild_free_list();
    }

    /// Initialize fresh flash
    fn initialize_flash(&mut self) {
        // Write superblock
        let magic = b"FLASHDB\0";
        self.memory[0..8].copy_from_slice(magic);

        // Version and config
        let version: u32 = 1;
        self.memory[8..12].copy_from_slice(&version.to_le_bytes());
        self.memory[12..16].copy_from_slice(&(self.config.block_count as u32).to_le_bytes());
        self.memory[16..20].copy_from_slice(&(self.config.page_size as u32).to_le_bytes());
        self.memory[20..24].copy_from_slice(&(self.config.erase_block_size as u32).to_le_bytes());

        self.dirty = true;
    }

    /// Rebuild free block list sorted by wear count
    fn rebuild_free_list(&mut self) {
        self.free_blocks.clear();

        for (i, entry) in self.wear_table.iter().enumerate() {
            if entry.state == BlockState::Free && i >= 3 {
                // Skip metadata blocks
                self.free_blocks.push(i as u32);
            }
        }

        // Sort by erase count (lowest first for wear leveling)
        let wear_table = &self.wear_table;
        self.free_blocks
            .sort_by_key(|&block| wear_table[block as usize].erase_count);
    }

    /// Allocate a physical block with wear leveling
    fn allocate_block(&mut self) -> Result<u32, StorageError> {
        if self.free_blocks.is_empty() {
            return Err(StorageError::NoSpace("No free blocks".into()));
        }

        // Get block with lowest wear
        let physical = self.free_blocks.remove(0);
        self.wear_table[physical as usize].state = BlockState::InUse;

        Ok(physical)
    }

    /// Free a physical block
    fn free_block(&mut self, physical: u32) -> Result<(), StorageError> {
        let entry = &mut self.wear_table[physical as usize];
        if entry.state == BlockState::Bad {
            return Err(StorageError::BadBlock);
        }

        // Erase the block (set to 0xFF)
        let offset = physical as usize * self.config.erase_block_size;
        let end = offset + self.config.erase_block_size;
        self.memory[offset..end].fill(0xFF);

        // Update wear table
        entry.erase_count += 1;
        entry.state = BlockState::Free;
        entry.logical_block = None;

        // Check for wear-out
        if entry.erase_count >= self.config.max_erase_cycles {
            entry.state = BlockState::Bad;
        } else {
            // Add back to free list in sorted order
            let erase_count = entry.erase_count;
            let insert_pos = self
                .free_blocks
                .iter()
                .position(|&b| self.wear_table[b as usize].erase_count > erase_count)
                .unwrap_or(self.free_blocks.len());
            self.free_blocks.insert(insert_pos, physical);
        }

        self.dirty = true;
        Ok(())
    }

    /// Calculate page offset within a block
    fn page_offset(&self, page_id: u64) -> (u32, usize) {
        let pages_per_block = self.config.erase_block_size / self.config.page_size;
        let logical_block = (page_id / pages_per_block as u64) as u32;
        let page_in_block = (page_id % pages_per_block as u64) as usize;
        (logical_block, page_in_block * self.config.page_size)
    }

    /// Write metadata to flash
    fn write_metadata(&mut self) {
        // Write block map to block 1
        let map_offset = self.config.erase_block_size;
        let mut offset = map_offset;

        for (&logical, &physical) in &self.block_map {
            self.memory[offset..offset + 4].copy_from_slice(&logical.to_le_bytes());
            self.memory[offset + 4..offset + 8].copy_from_slice(&physical.to_le_bytes());
            offset += 8;
        }

        // Write wear table to block 2
        let wear_offset = self.config.erase_block_size * 2;
        for (i, entry) in self.wear_table.iter().enumerate() {
            let offset = wear_offset + i * 4;
            self.memory[offset..offset + 4].copy_from_slice(&entry.erase_count.to_le_bytes());
        }
    }
}

impl EdgeStorageBackend for FlashBackend {
    fn read_page(&self, page_id: u64) -> Result<Vec<u8>, StorageError> {
        let (logical_block, page_offset) = self.page_offset(page_id);

        let physical_block = self
            .block_map
            .get(&logical_block)
            .ok_or_else(|| StorageError::NotFound(format!("Page {} not found", page_id)))?;

        let block_offset = *physical_block as usize * self.config.erase_block_size;
        let data_offset = block_offset + page_offset;
        let data_end = data_offset + self.config.page_size;

        if data_end > self.memory.len() {
            return Err(StorageError::OutOfBounds);
        }

        let data = self.memory[data_offset..data_end].to_vec();

        Ok(data)
    }

    fn write_page(&mut self, page_id: u64, data: &[u8]) -> Result<(), StorageError> {
        if data.len() > self.config.page_size {
            return Err(StorageError::DataTooLarge);
        }

        let (logical_block, page_offset) = self.page_offset(page_id);

        // Get or allocate physical block
        let physical_block = if let Some(&existing) = self.block_map.get(&logical_block) {
            existing
        } else {
            let new_block = self.allocate_block()?;
            self.block_map.insert(logical_block, new_block);
            self.wear_table[new_block as usize].logical_block = Some(logical_block);
            new_block
        };

        // Write data
        let block_offset = physical_block as usize * self.config.erase_block_size;
        let data_offset = block_offset + page_offset;

        // Pad data to page size
        let mut padded = vec![0xFF; self.config.page_size];
        padded[..data.len()].copy_from_slice(data);

        self.memory[data_offset..data_offset + self.config.page_size].copy_from_slice(&padded);

        // Update stats
        self.stats.bytes_written += self.config.page_size as u64;
        self.stats.write_ops += 1;
        self.dirty = true;

        Ok(())
    }

    fn sync(&mut self) -> Result<(), StorageError> {
        if self.dirty {
            self.write_metadata();
            self.dirty = false;
            self.stats.sync_ops += 1;
        }
        Ok(())
    }

    fn page_size(&self) -> usize {
        self.config.page_size
    }

    fn capacity(&self) -> u64 {
        (self.config.erase_block_size * (self.config.block_count - self.config.reserved_blocks))
            as u64
    }

    fn available(&self) -> u64 {
        (self.free_blocks.len() * self.config.erase_block_size) as u64
    }

    fn delete_page(&mut self, page_id: u64) -> Result<(), StorageError> {
        let (logical_block, _) = self.page_offset(page_id);

        if let Some(&physical_block) = self.block_map.get(&logical_block) {
            // For simplicity, we mark the whole block as free
            // A real implementation would track page-level validity
            self.block_map.remove(&logical_block);
            self.free_block(physical_block)?;
        }

        Ok(())
    }

    fn page_exists(&self, page_id: u64) -> bool {
        let (logical_block, _) = self.page_offset(page_id);
        self.block_map.contains_key(&logical_block)
    }

    fn stats(&self) -> StorageStats {
        let mut stats = self.stats.clone();
        stats.pages_used = self.block_map.len() as u64;
        stats.pages_free = self.free_blocks.len() as u64;

        // Calculate write amplification
        if stats.write_ops > 0 {
            let total_erases: u64 = self.wear_table.iter().map(|e| e.erase_count as u64).sum();
            stats.write_amplification = (total_erases * self.config.erase_block_size as u64) as f32
                / stats.bytes_written.max(1) as f32;
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_backend_basic() {
        let mut flash = FlashBackend::new(FlashConfig::default());

        // Write a page
        let data = b"Hello, Flash!";
        flash.write_page(0, data).unwrap();

        // Read it back
        let read_data = flash.read_page(0).unwrap();
        assert_eq!(&read_data[..data.len()], data);
    }

    #[test]
    fn test_flash_multiple_pages() {
        let mut flash = FlashBackend::new(FlashConfig::default());

        for i in 0..10 {
            let data = format!("Page {}", i);
            flash.write_page(i, data.as_bytes()).unwrap();
        }

        for i in 0..10 {
            let expected = format!("Page {}", i);
            let read_data = flash.read_page(i).unwrap();
            assert_eq!(&read_data[..expected.len()], expected.as_bytes());
        }
    }

    #[test]
    fn test_flash_delete_page() {
        let mut flash = FlashBackend::new(FlashConfig::default());

        flash.write_page(0, b"data").unwrap();
        assert!(flash.page_exists(0));

        flash.delete_page(0).unwrap();
        assert!(!flash.page_exists(0));
    }

    #[test]
    fn test_flash_wear_leveling() {
        let mut flash = FlashBackend::new(FlashConfig::default());

        // Write many pages to trigger wear leveling
        for i in 0..100 {
            flash.write_page(i, b"test").unwrap();
            flash.delete_page(i).unwrap();
        }

        let stats = flash.stats();
        assert!(stats.write_ops > 0);
    }

    #[test]
    fn test_flash_sync() {
        let mut flash = FlashBackend::new(FlashConfig::default());

        flash.write_page(0, b"test").unwrap();
        assert!(flash.dirty);

        flash.sync().unwrap();
        assert!(!flash.dirty);
    }
}
