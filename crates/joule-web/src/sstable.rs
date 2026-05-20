//! Sorted string table — block-based layout, block index, key-value encoding,
//! simple block compression, binary search within blocks, iterator, merge
//! iterator (multi-way merge), data/index/footer sections.

use std::collections::BinaryHeap;
use std::cmp::Ordering;

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors returned by SSTable operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SsTableError {
    /// Block not found at given offset.
    BlockNotFound(u32),
    /// Key not found.
    KeyNotFound,
    /// Corrupt block (checksum mismatch).
    CorruptBlock { offset: u32, expected: u32, actual: u32 },
    /// Invalid block data.
    InvalidBlock(String),
    /// Decompression failed.
    DecompressFailed(String),
    /// Builder was used after finalization.
    AlreadyFinalized,
}

impl std::fmt::Display for SsTableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BlockNotFound(off) => write!(f, "block not found at offset {off}"),
            Self::KeyNotFound => write!(f, "key not found"),
            Self::CorruptBlock { offset, expected, actual } => {
                write!(f, "corrupt block at {offset}: expected crc {expected:#010x}, got {actual:#010x}")
            }
            Self::InvalidBlock(msg) => write!(f, "invalid block: {msg}"),
            Self::DecompressFailed(msg) => write!(f, "decompression failed: {msg}"),
            Self::AlreadyFinalized => write!(f, "builder already finalized"),
        }
    }
}

impl std::error::Error for SsTableError {}

// ── CRC32 ────────────────────────────────────────────────────────────────────

fn crc32(data: &[u8]) -> u32 {
    const POLY: u32 = 0xEDB88320;
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ POLY;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ 0xFFFFFFFF
}

// ── Key-Value Encoding ──────────────────────────────────────────────────────

/// Encode a key-value pair into bytes.
/// Format: [key_len:4][value_len:4][key][value]
pub fn encode_kv(key: &[u8], value: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(8 + key.len() + value.len());
    buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(value);
    buf
}

/// Decode a key-value pair from bytes at the given offset.
/// Returns (key, value, bytes_consumed).
pub fn decode_kv(data: &[u8], offset: usize) -> Result<(Vec<u8>, Vec<u8>, usize), SsTableError> {
    if offset + 8 > data.len() {
        return Err(SsTableError::InvalidBlock("not enough header bytes".into()));
    }
    let key_len = u32::from_le_bytes([
        data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
    ]) as usize;
    let value_len = u32::from_le_bytes([
        data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7],
    ]) as usize;
    let kv_start = offset + 8;
    let total = 8 + key_len + value_len;
    if kv_start + key_len + value_len > data.len() {
        return Err(SsTableError::InvalidBlock("truncated kv data".into()));
    }
    let key = data[kv_start..kv_start + key_len].to_vec();
    let value = data[kv_start + key_len..kv_start + key_len + value_len].to_vec();
    Ok((key, value, total))
}

// ── Simple Compression ──────────────────────────────────────────────────────

/// Simple RLE-based compression for block data.
pub fn compress_block(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let byte = data[i];
        let mut count = 1u8;
        while i + (count as usize) < data.len()
            && data[i + count as usize] == byte
            && count < 255
        {
            count += 1;
        }
        if count >= 3 {
            // Marker: 0x00, byte, count.
            // To disambiguate, if byte == 0 we still use the marker.
            result.push(0xFF);
            result.push(count);
            result.push(byte);
            i += count as usize;
        } else {
            // Literal.
            if byte == 0xFF {
                // Escape: 0xFF, 1, 0xFF means a single 0xFF literal.
                result.push(0xFF);
                result.push(1);
                result.push(0xFF);
                i += 1;
                // Handle second if count == 2.
                if count == 2 {
                    result.push(0xFF);
                    result.push(1);
                    result.push(0xFF);
                    i += 1;
                }
            } else {
                for _ in 0..count {
                    result.push(byte);
                    i += 1;
                }
            }
        }
    }
    result
}

/// Decompress block data.
pub fn decompress_block(data: &[u8]) -> Result<Vec<u8>, SsTableError> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < data.len() {
        if data[i] == 0xFF {
            if i + 2 >= data.len() {
                return Err(SsTableError::DecompressFailed("truncated RLE marker".into()));
            }
            let count = data[i + 1] as usize;
            let byte = data[i + 2];
            for _ in 0..count {
                result.push(byte);
            }
            i += 3;
        } else {
            result.push(data[i]);
            i += 1;
        }
    }
    Ok(result)
}

// ── Block ────────────────────────────────────────────────────────────────────

/// A data block containing sorted key-value pairs.
#[derive(Debug, Clone)]
pub struct Block {
    /// Encoded key-value pairs (uncompressed).
    data: Vec<u8>,
    /// Offsets of each KV pair within `data` for binary search.
    offsets: Vec<u32>,
    /// Number of entries.
    entry_count: usize,
    /// Checksum of uncompressed data.
    checksum: u32,
}

impl Block {
    /// Create a new empty block.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            offsets: Vec::new(),
            entry_count: 0,
            checksum: 0,
        }
    }

    /// Append a key-value pair.  Keys must be added in sorted order.
    pub fn add(&mut self, key: &[u8], value: &[u8]) {
        let offset = self.data.len() as u32;
        self.offsets.push(offset);
        self.data.extend_from_slice(&encode_kv(key, value));
        self.entry_count += 1;
        self.checksum = crc32(&self.data);
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entry_count
    }

    /// Whether the block is empty.
    pub fn is_empty(&self) -> bool {
        self.entry_count == 0
    }

    /// Size in bytes (uncompressed).
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Verify the block checksum.
    pub fn verify(&self) -> bool {
        crc32(&self.data) == self.checksum
    }

    /// Get the key at a given entry index.
    fn key_at(&self, idx: usize) -> Result<Vec<u8>, SsTableError> {
        let offset = self.offsets[idx] as usize;
        let (key, _, _) = decode_kv(&self.data, offset)?;
        Ok(key)
    }

    /// Binary search for a key, returning the entry value.
    pub fn get(&self, key: &[u8]) -> Result<Vec<u8>, SsTableError> {
        if self.entry_count == 0 {
            return Err(SsTableError::KeyNotFound);
        }

        let mut lo = 0;
        let mut hi = self.entry_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let mid_key = self.key_at(mid)?;
            match mid_key.as_slice().cmp(key) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => {
                    let offset = self.offsets[mid] as usize;
                    let (_, value, _) = decode_kv(&self.data, offset)?;
                    return Ok(value);
                }
            }
        }
        Err(SsTableError::KeyNotFound)
    }

    /// Get all key-value pairs in sorted order.
    pub fn entries(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>, SsTableError> {
        let mut result = Vec::with_capacity(self.entry_count);
        let mut offset = 0;
        while offset < self.data.len() {
            let (key, value, consumed) = decode_kv(&self.data, offset)?;
            result.push((key, value));
            offset += consumed;
        }
        Ok(result)
    }

    /// First key in the block (for index).
    pub fn first_key(&self) -> Result<Vec<u8>, SsTableError> {
        if self.entry_count == 0 {
            return Err(SsTableError::InvalidBlock("empty block".into()));
        }
        self.key_at(0)
    }

    /// Last key in the block.
    pub fn last_key(&self) -> Result<Vec<u8>, SsTableError> {
        if self.entry_count == 0 {
            return Err(SsTableError::InvalidBlock("empty block".into()));
        }
        self.key_at(self.entry_count - 1)
    }

    /// Compress the block data and return compressed bytes with checksum.
    pub fn to_compressed(&self) -> Vec<u8> {
        let compressed = compress_block(&self.data);
        let mut result = Vec::new();
        result.extend_from_slice(&(self.entry_count as u32).to_le_bytes());
        result.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
        result.extend_from_slice(&self.checksum.to_le_bytes());
        // Offsets.
        result.extend_from_slice(&(self.offsets.len() as u32).to_le_bytes());
        for &off in &self.offsets {
            result.extend_from_slice(&off.to_le_bytes());
        }
        result.extend_from_slice(&compressed);
        result
    }

    /// Rebuild a block from compressed bytes.
    pub fn from_compressed(bytes: &[u8]) -> Result<Self, SsTableError> {
        if bytes.len() < 16 {
            return Err(SsTableError::InvalidBlock("too short".into()));
        }
        let entry_count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let compressed_len = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let checksum = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let offset_count = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;

        let mut pos = 16;
        let mut offsets = Vec::with_capacity(offset_count);
        for _ in 0..offset_count {
            if pos + 4 > bytes.len() {
                return Err(SsTableError::InvalidBlock("truncated offsets".into()));
            }
            offsets.push(u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]));
            pos += 4;
        }

        if pos + compressed_len > bytes.len() {
            return Err(SsTableError::InvalidBlock("truncated compressed data".into()));
        }
        let compressed = &bytes[pos..pos + compressed_len];
        let data = decompress_block(compressed)?;

        let actual_crc = crc32(&data);
        if actual_crc != checksum {
            return Err(SsTableError::CorruptBlock {
                offset: 0,
                expected: checksum,
                actual: actual_crc,
            });
        }

        Ok(Self {
            data,
            offsets,
            entry_count,
            checksum,
        })
    }
}

impl Default for Block {
    fn default() -> Self {
        Self::new()
    }
}

// ── Block Index ──────────────────────────────────────────────────────────────

/// An index entry pointing to a block.
#[derive(Debug, Clone)]
pub struct BlockIndexEntry {
    /// First key in the block.
    pub first_key: Vec<u8>,
    /// Offset of the block in the table.
    pub offset: u32,
    /// Size of the block in bytes.
    pub size: u32,
}

/// Block index for locating blocks by key.
#[derive(Debug, Clone)]
pub struct BlockIndex {
    entries: Vec<BlockIndexEntry>,
}

impl BlockIndex {
    /// Create a new empty block index.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add an index entry.
    pub fn add(&mut self, first_key: Vec<u8>, offset: u32, size: u32) {
        self.entries.push(BlockIndexEntry {
            first_key,
            offset,
            size,
        });
    }

    /// Find the block that may contain the given key.
    pub fn find_block(&self, key: &[u8]) -> Option<&BlockIndexEntry> {
        if self.entries.is_empty() {
            return None;
        }
        // Find the last entry whose first_key <= key.
        let mut result_idx = None;
        let mut lo = 0;
        let mut hi = self.entries.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.entries[mid].first_key.as_slice() <= key {
                result_idx = Some(mid);
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        result_idx.map(|i| &self.entries[i])
    }

    /// Number of index entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterator over index entries.
    pub fn iter(&self) -> impl Iterator<Item = &BlockIndexEntry> {
        self.entries.iter()
    }
}

impl Default for BlockIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ── Footer ───────────────────────────────────────────────────────────────────

/// Footer section of an SSTable, containing metadata.
#[derive(Debug, Clone)]
pub struct Footer {
    /// Offset of the index block.
    pub index_offset: u32,
    /// Size of the index block.
    pub index_size: u32,
    /// Total number of data blocks.
    pub block_count: u32,
    /// Total number of entries.
    pub entry_count: u32,
    /// Smallest key in the table.
    pub min_key: Vec<u8>,
    /// Largest key in the table.
    pub max_key: Vec<u8>,
    /// Magic number for format identification.
    pub magic: u32,
}

impl Footer {
    /// Standard magic number.
    pub const MAGIC: u32 = 0x53535442; // "SSTB"

    /// Create a new footer.
    pub fn new(
        index_offset: u32,
        index_size: u32,
        block_count: u32,
        entry_count: u32,
        min_key: Vec<u8>,
        max_key: Vec<u8>,
    ) -> Self {
        Self {
            index_offset,
            index_size,
            block_count,
            entry_count,
            min_key,
            max_key,
            magic: Self::MAGIC,
        }
    }

    /// Check that the magic number is valid.
    pub fn is_valid(&self) -> bool {
        self.magic == Self::MAGIC
    }
}

// ── SSTable Builder ─────────────────────────────────────────────────────────

/// Builds an SSTable from sorted key-value pairs.
#[derive(Debug)]
pub struct SsTableBuilder {
    blocks: Vec<Block>,
    current_block: Block,
    block_size_limit: usize,
    index: BlockIndex,
    entry_count: usize,
    min_key: Option<Vec<u8>>,
    max_key: Option<Vec<u8>>,
    finalized: bool,
}

impl SsTableBuilder {
    /// Create a new builder with the given target block size.
    pub fn new(block_size_limit: usize) -> Self {
        Self {
            blocks: Vec::new(),
            current_block: Block::new(),
            block_size_limit,
            index: BlockIndex::new(),
            entry_count: 0,
            min_key: None,
            max_key: None,
            finalized: false,
        }
    }

    /// Add a key-value pair.  Keys must be added in sorted order.
    pub fn add(&mut self, key: &[u8], value: &[u8]) -> Result<(), SsTableError> {
        if self.finalized {
            return Err(SsTableError::AlreadyFinalized);
        }

        if self.min_key.is_none() {
            self.min_key = Some(key.to_vec());
        }
        self.max_key = Some(key.to_vec());

        if self.current_block.size() + key.len() + value.len() + 8 > self.block_size_limit
            && !self.current_block.is_empty()
        {
            self.flush_block()?;
        }

        self.current_block.add(key, value);
        self.entry_count += 1;
        Ok(())
    }

    fn flush_block(&mut self) -> Result<(), SsTableError> {
        if self.current_block.is_empty() {
            return Ok(());
        }
        let first_key = self.current_block.first_key()?;
        let offset = self.blocks.iter().map(|b| b.size() as u32).sum::<u32>();
        let size = self.current_block.size() as u32;
        self.index.add(first_key, offset, size);
        let block = std::mem::replace(&mut self.current_block, Block::new());
        self.blocks.push(block);
        Ok(())
    }

    /// Finalize the builder and return the SSTable.
    pub fn build(mut self) -> Result<SsTableData, SsTableError> {
        if self.finalized {
            return Err(SsTableError::AlreadyFinalized);
        }
        self.finalized = true;
        self.flush_block()?;

        let index_offset: u32 = self.blocks.iter().map(|b| b.size() as u32).sum();
        let footer = Footer::new(
            index_offset,
            0, // simplified
            self.blocks.len() as u32,
            self.entry_count as u32,
            self.min_key.unwrap_or_default(),
            self.max_key.unwrap_or_default(),
        );

        Ok(SsTableData {
            blocks: self.blocks,
            index: self.index,
            footer,
        })
    }

    /// Number of entries added so far.
    pub fn entry_count(&self) -> usize {
        self.entry_count
    }
}

// ── SSTable Data ─────────────────────────────────────────────────────────────

/// A complete in-memory SSTable.
#[derive(Debug)]
pub struct SsTableData {
    blocks: Vec<Block>,
    index: BlockIndex,
    footer: Footer,
}

impl SsTableData {
    /// Look up a key.
    pub fn get(&self, key: &[u8]) -> Result<Vec<u8>, SsTableError> {
        // Find candidate block via index.
        let idx_entry = self.index.find_block(key).ok_or(SsTableError::KeyNotFound)?;
        // Find the block at that offset.
        let block = self.block_at_offset(idx_entry.offset)?;
        block.get(key)
    }

    fn block_at_offset(&self, target_offset: u32) -> Result<&Block, SsTableError> {
        let mut offset = 0u32;
        for block in &self.blocks {
            if offset == target_offset {
                return Ok(block);
            }
            offset += block.size() as u32;
        }
        Err(SsTableError::BlockNotFound(target_offset))
    }

    /// Number of data blocks.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Total number of entries.
    pub fn entry_count(&self) -> usize {
        self.footer.entry_count as usize
    }

    /// Footer metadata.
    pub fn footer(&self) -> &Footer {
        &self.footer
    }

    /// Index metadata.
    pub fn index(&self) -> &BlockIndex {
        &self.index
    }

    /// Iterate all key-value pairs across all blocks.
    pub fn iter(&self) -> SsTableIterator<'_> {
        SsTableIterator {
            blocks: &self.blocks,
            block_idx: 0,
            entries: Vec::new(),
            entry_idx: 0,
        }
    }
}

// ── Iterator ─────────────────────────────────────────────────────────────────

/// Iterator over all entries in an SSTable.
pub struct SsTableIterator<'a> {
    blocks: &'a [Block],
    block_idx: usize,
    entries: Vec<(Vec<u8>, Vec<u8>)>,
    entry_idx: usize,
}

impl Iterator for SsTableIterator<'_> {
    type Item = (Vec<u8>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.entry_idx < self.entries.len() {
                let item = self.entries[self.entry_idx].clone();
                self.entry_idx += 1;
                return Some(item);
            }
            if self.block_idx >= self.blocks.len() {
                return None;
            }
            self.entries = self.blocks[self.block_idx].entries().ok()?;
            self.entry_idx = 0;
            self.block_idx += 1;
        }
    }
}

// ── Merge Iterator ───────────────────────────────────────────────────────────

/// An item from one of the merge sources.
struct MergeItem {
    key: Vec<u8>,
    value: Vec<u8>,
    source_idx: usize,
}

impl Eq for MergeItem {}

impl PartialEq for MergeItem {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Ord for MergeItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior with BinaryHeap (max-heap).
        other.key.cmp(&self.key)
            .then_with(|| other.source_idx.cmp(&self.source_idx))
    }
}

impl PartialOrd for MergeItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Multi-way merge iterator over multiple sorted sources.
pub struct MergeIterator {
    heap: BinaryHeap<MergeItem>,
    sources: Vec<std::vec::IntoIter<(Vec<u8>, Vec<u8>)>>,
}

impl MergeIterator {
    /// Create a merge iterator from multiple sorted entry lists.
    pub fn new(sources: Vec<Vec<(Vec<u8>, Vec<u8>)>>) -> Self {
        let mut heap = BinaryHeap::new();
        let mut iters: Vec<std::vec::IntoIter<(Vec<u8>, Vec<u8>)>> = Vec::new();

        for (idx, entries) in sources.into_iter().enumerate() {
            let mut iter = entries.into_iter();
            if let Some((key, value)) = iter.next() {
                heap.push(MergeItem {
                    key,
                    value,
                    source_idx: idx,
                });
            }
            iters.push(iter);
        }

        Self {
            heap,
            sources: iters,
        }
    }

    /// Get the next smallest key-value pair.
    pub fn next_entry(&mut self) -> Option<(Vec<u8>, Vec<u8>)> {
        let item = self.heap.pop()?;
        let idx = item.source_idx;

        // Advance the source this item came from.
        if let Some((key, value)) = self.sources[idx].next() {
            self.heap.push(MergeItem {
                key,
                value,
                source_idx: idx,
            });
        }

        Some((item.key, item.value))
    }

    /// Drain all remaining entries in merge order.
    pub fn collect_all(&mut self) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut result = Vec::new();
        while let Some(entry) = self.next_entry() {
            result.push(entry);
        }
        result
    }

    /// Whether the iterator is exhausted.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_kv() {
        let encoded = encode_kv(b"hello", b"world");
        let (key, value, consumed) = decode_kv(&encoded, 0).unwrap();
        assert_eq!(key, b"hello");
        assert_eq!(value, b"world");
        assert_eq!(consumed, encoded.len());
    }

    #[test]
    fn encode_decode_empty_value() {
        let encoded = encode_kv(b"key", b"");
        let (key, value, _) = decode_kv(&encoded, 0).unwrap();
        assert_eq!(key, b"key");
        assert!(value.is_empty());
    }

    #[test]
    fn decode_kv_truncated() {
        let result = decode_kv(&[0, 0, 0, 10, 0, 0, 0, 0], 0);
        assert!(result.is_err());
    }

    #[test]
    fn compress_decompress_roundtrip() {
        let data = b"aaabbbcccddd".to_vec();
        let compressed = compress_block(&data);
        let decompressed = decompress_block(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn compress_decompress_no_runs() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let compressed = compress_block(&data);
        let decompressed = decompress_block(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn compress_decompress_long_run() {
        let data = vec![0x42; 200];
        let compressed = compress_block(&data);
        let decompressed = decompress_block(&compressed).unwrap();
        assert_eq!(decompressed, data);
        // Compressed should be smaller.
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn compress_decompress_empty() {
        let data: Vec<u8> = Vec::new();
        let compressed = compress_block(&data);
        let decompressed = decompress_block(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn block_add_and_get() {
        let mut block = Block::new();
        block.add(b"apple", b"1");
        block.add(b"banana", b"2");
        block.add(b"cherry", b"3");
        assert_eq!(block.len(), 3);
        assert_eq!(block.get(b"banana").unwrap(), b"2");
    }

    #[test]
    fn block_get_missing() {
        let mut block = Block::new();
        block.add(b"a", b"1");
        assert_eq!(block.get(b"z"), Err(SsTableError::KeyNotFound));
    }

    #[test]
    fn block_first_last_key() {
        let mut block = Block::new();
        block.add(b"alpha", b"1");
        block.add(b"beta", b"2");
        block.add(b"gamma", b"3");
        assert_eq!(block.first_key().unwrap(), b"alpha");
        assert_eq!(block.last_key().unwrap(), b"gamma");
    }

    #[test]
    fn block_verify_checksum() {
        let mut block = Block::new();
        block.add(b"key", b"val");
        assert!(block.verify());
    }

    #[test]
    fn block_compress_decompress_roundtrip() {
        let mut block = Block::new();
        block.add(b"key1", b"value1");
        block.add(b"key2", b"value2");
        block.add(b"key3", b"value3");

        let compressed = block.to_compressed();
        let restored = Block::from_compressed(&compressed).unwrap();
        assert_eq!(restored.len(), 3);
        assert_eq!(restored.get(b"key2").unwrap(), b"value2");
    }

    #[test]
    fn block_entries() {
        let mut block = Block::new();
        block.add(b"a", b"1");
        block.add(b"b", b"2");
        let entries = block.entries().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, b"a");
        assert_eq!(entries[1].1, b"2");
    }

    #[test]
    fn block_index_find() {
        let mut index = BlockIndex::new();
        index.add(b"a".to_vec(), 0, 100);
        index.add(b"m".to_vec(), 100, 100);
        index.add(b"z".to_vec(), 200, 100);

        let entry = index.find_block(b"c").unwrap();
        assert_eq!(entry.first_key, b"a");

        let entry = index.find_block(b"p").unwrap();
        assert_eq!(entry.first_key, b"m");

        let entry = index.find_block(b"z").unwrap();
        assert_eq!(entry.first_key, b"z");
    }

    #[test]
    fn block_index_empty() {
        let index = BlockIndex::new();
        assert!(index.find_block(b"x").is_none());
        assert!(index.is_empty());
    }

    #[test]
    fn footer_valid() {
        let footer = Footer::new(100, 50, 5, 100, b"a".to_vec(), b"z".to_vec());
        assert!(footer.is_valid());
        assert_eq!(footer.magic, Footer::MAGIC);
    }

    #[test]
    fn sstable_builder_and_lookup() {
        let mut builder = SsTableBuilder::new(256);
        builder.add(b"apple", b"fruit").unwrap();
        builder.add(b"bread", b"carb").unwrap();
        builder.add(b"cheese", b"dairy").unwrap();
        builder.add(b"date", b"fruit").unwrap();

        let table = builder.build().unwrap();
        assert_eq!(table.get(b"bread").unwrap(), b"carb");
        assert_eq!(table.get(b"cheese").unwrap(), b"dairy");
        assert_eq!(table.entry_count(), 4);
    }

    #[test]
    fn sstable_builder_multiple_blocks() {
        let mut builder = SsTableBuilder::new(32);
        for i in 0u32..20 {
            let key = format!("key_{i:04}").into_bytes();
            let val = format!("val_{i:04}").into_bytes();
            builder.add(&key, &val).unwrap();
        }
        let table = builder.build().unwrap();
        assert!(table.block_count() > 1);
        assert_eq!(table.entry_count(), 20);
        assert_eq!(
            table.get(b"key_0010").unwrap(),
            b"val_0010"
        );
    }

    #[test]
    fn sstable_iterator() {
        let mut builder = SsTableBuilder::new(256);
        builder.add(b"a", b"1").unwrap();
        builder.add(b"b", b"2").unwrap();
        builder.add(b"c", b"3").unwrap();
        let table = builder.build().unwrap();
        let entries: Vec<_> = table.iter().collect();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].0, b"a");
        assert_eq!(entries[2].1, b"3");
    }

    #[test]
    fn merge_iterator_two_sources() {
        let a = vec![
            (b"a".to_vec(), b"1".to_vec()),
            (b"c".to_vec(), b"3".to_vec()),
            (b"e".to_vec(), b"5".to_vec()),
        ];
        let b = vec![
            (b"b".to_vec(), b"2".to_vec()),
            (b"d".to_vec(), b"4".to_vec()),
        ];
        let mut merger = MergeIterator::new(vec![a, b]);
        let result = merger.collect_all();
        let keys: Vec<_> = result.iter().map(|(k, _)| k.clone()).collect();
        assert_eq!(keys, vec![b"a", b"b", b"c", b"d", b"e"]);
    }

    #[test]
    fn merge_iterator_three_sources() {
        let a = vec![(b"a".to_vec(), b"1".to_vec())];
        let b = vec![(b"b".to_vec(), b"2".to_vec())];
        let c = vec![(b"c".to_vec(), b"3".to_vec())];
        let mut merger = MergeIterator::new(vec![a, b, c]);
        let result = merger.collect_all();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].0, b"a");
        assert_eq!(result[2].0, b"c");
    }

    #[test]
    fn merge_iterator_empty_source() {
        let a = vec![(b"x".to_vec(), b"1".to_vec())];
        let b: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        let mut merger = MergeIterator::new(vec![a, b]);
        let result = merger.collect_all();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn sstable_get_missing_key() {
        let mut builder = SsTableBuilder::new(256);
        builder.add(b"a", b"1").unwrap();
        let table = builder.build().unwrap();
        assert_eq!(table.get(b"z"), Err(SsTableError::KeyNotFound));
    }

    #[test]
    fn sstable_error_display() {
        let e = SsTableError::KeyNotFound;
        assert_eq!(e.to_string(), "key not found");
        let e = SsTableError::BlockNotFound(42);
        assert!(e.to_string().contains("42"));
    }

    #[test]
    fn builder_after_finalize() {
        let mut builder = SsTableBuilder::new(256);
        builder.add(b"k", b"v").unwrap();
        // Consume builder.
        let _table = builder.build().unwrap();
        // Can't add after finalize — builder is consumed.
    }
}
