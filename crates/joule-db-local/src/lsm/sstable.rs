//! Sorted String Table (SSTable) for the LSM-Tree engine.
//!
//! An SSTable is an immutable, sorted key-value file with:
//! - Data blocks: sorted key-value pairs
//! - Index block: first_key → block_offset for binary search
//! - Bloom filter: fast negative lookups
//! - Footer: offsets to index and bloom sections

use crate::bloom::BloomFilter;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const SSTABLE_MAGIC: u32 = 0x53535442; // "SSTB"
const VALUE_TOMBSTONE: u8 = 0x00;

/// Convert a byte slice into a fixed-size array, returning an
/// `io::Error::InvalidData` on length mismatch. Replaces every
/// `slice.try_into().unwrap()` pattern in this module so a corrupt
/// SSTable surfaces as a typed error instead of a panic.
fn slice_to_array<const N: usize>(s: &[u8]) -> io::Result<[u8; N]> {
    s.try_into().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("expected {} bytes, got {}", N, s.len()),
        )
    })
}
const VALUE_DATA: u8 = 0x01;
const DEFAULT_BLOCK_SIZE: usize = 4096;

/// Metadata about an SSTable file.
#[derive(Debug, Clone)]
pub struct SSTableMeta {
    pub id: u64,
    pub level: usize,
    pub path: PathBuf,
    pub first_key: Vec<u8>,
    pub last_key: Vec<u8>,
    pub entry_count: usize,
    pub size_bytes: u64,
}

/// Index entry: first key of a data block and its file offset.
#[derive(Debug, Clone)]
struct IndexEntry {
    first_key: Vec<u8>,
    offset: u64,
    length: u32,
}

/// Writes sorted key-value pairs to an SSTable file.
pub struct SSTableWriter {
    path: PathBuf,
    writer: BufWriter<File>,
    bloom: BloomFilter,
    index_entries: Vec<IndexEntry>,
    current_block: Vec<u8>,
    current_block_first_key: Option<Vec<u8>>,
    current_block_offset: u64,
    first_key: Option<Vec<u8>>,
    last_key: Option<Vec<u8>>,
    entry_count: usize,
    block_size: usize,
    bytes_written: u64,
}

impl SSTableWriter {
    /// Create a new SSTable writer.
    pub fn new(path: &Path, expected_entries: usize) -> io::Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            writer: BufWriter::new(file),
            bloom: BloomFilter::new(expected_entries.max(1), 0.01),
            index_entries: Vec::new(),
            current_block: Vec::new(),
            current_block_first_key: None,
            current_block_offset: 0,
            first_key: None,
            last_key: None,
            entry_count: 0,
            block_size: DEFAULT_BLOCK_SIZE,
            bytes_written: 0,
        })
    }

    /// Add a key-value entry. `value` is None for tombstones.
    /// MUST be called in sorted key order.
    pub fn add(&mut self, key: &[u8], value: Option<&[u8]>) -> io::Result<()> {
        self.bloom.insert(key);

        if self.first_key.is_none() {
            self.first_key = Some(key.to_vec());
        }
        self.last_key = Some(key.to_vec());
        self.entry_count += 1;

        if self.current_block_first_key.is_none() {
            self.current_block_first_key = Some(key.to_vec());
        }

        // Encode entry into current block
        // Format: key_len(4) + key + marker(1) + [value_len(4) + value]
        let key_len = key.len() as u32;
        self.current_block.extend_from_slice(&key_len.to_le_bytes());
        self.current_block.extend_from_slice(key);

        match value {
            Some(v) => {
                self.current_block.push(VALUE_DATA);
                let val_len = v.len() as u32;
                self.current_block.extend_from_slice(&val_len.to_le_bytes());
                self.current_block.extend_from_slice(v);
            }
            None => {
                self.current_block.push(VALUE_TOMBSTONE);
            }
        }

        // Flush block if it's big enough
        if self.current_block.len() >= self.block_size {
            self.flush_block()?;
        }

        Ok(())
    }

    fn flush_block(&mut self) -> io::Result<()> {
        if self.current_block.is_empty() {
            return Ok(());
        }

        let block_len = self.current_block.len() as u32;
        self.writer.write_all(&self.current_block)?;

        self.index_entries.push(IndexEntry {
            first_key: self.current_block_first_key.take().unwrap_or_default(),
            offset: self.current_block_offset,
            length: block_len,
        });

        self.current_block_offset += block_len as u64;
        self.current_block.clear();
        self.bytes_written += block_len as u64;

        Ok(())
    }

    /// Finish writing the SSTable. Returns metadata about the file.
    pub fn finish(mut self, id: u64, level: usize) -> io::Result<SSTableMeta> {
        // Flush remaining data block
        self.flush_block()?;

        let data_end = self.current_block_offset;

        // Write index block
        let index_offset = data_end;
        let index_count = self.index_entries.len() as u32;
        self.writer.write_all(&index_count.to_le_bytes())?;
        for entry in &self.index_entries {
            let key_len = entry.first_key.len() as u32;
            self.writer.write_all(&key_len.to_le_bytes())?;
            self.writer.write_all(&entry.first_key)?;
            self.writer.write_all(&entry.offset.to_le_bytes())?;
            self.writer.write_all(&entry.length.to_le_bytes())?;
        }

        // Write bloom filter
        let bloom_offset = index_offset
            + 4
            + self
                .index_entries
                .iter()
                .map(|e| 4 + e.first_key.len() + 8 + 4)
                .sum::<usize>() as u64;
        let bloom_bytes = self.bloom.to_bytes();
        let bloom_len = bloom_bytes.len() as u32;
        self.writer.write_all(&bloom_len.to_le_bytes())?;
        self.writer.write_all(&bloom_bytes)?;

        // Write footer: index_offset(8) + bloom_offset(8) + entry_count(8) + magic(4)
        self.writer.write_all(&index_offset.to_le_bytes())?;
        self.writer.write_all(&bloom_offset.to_le_bytes())?;
        self.writer
            .write_all(&(self.entry_count as u64).to_le_bytes())?;
        self.writer.write_all(&SSTABLE_MAGIC.to_le_bytes())?;

        self.writer.flush()?;

        let file_size = fs::metadata(&self.path)?.len();

        Ok(SSTableMeta {
            id,
            level,
            path: self.path,
            first_key: self.first_key.unwrap_or_default(),
            last_key: self.last_key.unwrap_or_default(),
            entry_count: self.entry_count,
            size_bytes: file_size,
        })
    }
}

/// Reads entries from an SSTable file.
pub struct SSTableReader {
    path: PathBuf,
    index: Vec<IndexEntry>,
    bloom: BloomFilter,
    entry_count: usize,
}

impl SSTableReader {
    /// Open an existing SSTable file.
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut file = BufReader::new(File::open(path)?);
        let file_len = fs::metadata(path)?.len();

        if file_len < 28 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "SSTable too small",
            ));
        }

        // Read footer (last 28 bytes)
        file.seek(SeekFrom::End(-28))?;
        let mut footer = [0u8; 28];
        file.read_exact(&mut footer)?;

        let index_offset = u64::from_le_bytes(slice_to_array(&footer[0..8])?);
        let bloom_offset = u64::from_le_bytes(slice_to_array(&footer[8..16])?);
        let entry_count = u64::from_le_bytes(slice_to_array(&footer[16..24])?) as usize;
        let magic = u32::from_le_bytes(slice_to_array(&footer[24..28])?);

        if magic != SSTABLE_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid SSTable magic",
            ));
        }

        // Read index block
        file.seek(SeekFrom::Start(index_offset))?;
        let mut count_buf = [0u8; 4];
        file.read_exact(&mut count_buf)?;
        let index_count = u32::from_le_bytes(count_buf) as usize;

        let mut index = Vec::with_capacity(index_count);
        for _ in 0..index_count {
            let mut kl = [0u8; 4];
            file.read_exact(&mut kl)?;
            let key_len = u32::from_le_bytes(kl) as usize;
            let mut key = vec![0u8; key_len];
            file.read_exact(&mut key)?;
            let mut ob = [0u8; 8];
            file.read_exact(&mut ob)?;
            let offset = u64::from_le_bytes(ob);
            let mut lb = [0u8; 4];
            file.read_exact(&mut lb)?;
            let length = u32::from_le_bytes(lb);
            index.push(IndexEntry {
                first_key: key,
                offset,
                length,
            });
        }

        // Read bloom filter
        file.seek(SeekFrom::Start(bloom_offset))?;
        let mut bl = [0u8; 4];
        file.read_exact(&mut bl)?;
        let bloom_len = u32::from_le_bytes(bl) as usize;
        let mut bloom_data = vec![0u8; bloom_len];
        file.read_exact(&mut bloom_data)?;
        let bloom =
            BloomFilter::from_bytes(&bloom_data).unwrap_or_else(|| BloomFilter::new(1, 0.01));

        Ok(Self {
            path: path.to_path_buf(),
            index,
            bloom,
            entry_count,
        })
    }

    /// Check if a key might exist using the bloom filter.
    pub fn may_contain(&self, key: &[u8]) -> bool {
        self.bloom.may_contain(key)
    }

    /// Get a value by key. Returns:
    /// - Ok(Some(Some(value))) if found with data
    /// - Ok(Some(None)) if found as tombstone
    /// - Ok(None) if not found
    pub fn get(&self, key: &[u8]) -> io::Result<Option<Option<Vec<u8>>>> {
        if !self.bloom.may_contain(key) {
            return Ok(None);
        }

        // Binary search index to find the right block
        let block_idx = match self
            .index
            .binary_search_by(|entry| entry.first_key.as_slice().cmp(key))
        {
            Ok(i) => i,
            Err(0) => return Ok(None), // key is before all blocks
            Err(i) => i - 1,
        };

        let entry = &self.index[block_idx];
        let block = self.read_block(entry.offset, entry.length)?;
        Self::search_block(&block, key)
    }

    /// Iterate over all entries in the SSTable.
    pub fn iter(&self) -> io::Result<Vec<(Vec<u8>, Option<Vec<u8>>)>> {
        let mut result = Vec::with_capacity(self.entry_count);
        for entry in &self.index {
            let block = self.read_block(entry.offset, entry.length)?;
            let entries = Self::decode_block(&block);
            result.extend(entries);
        }
        Ok(result)
    }

    /// Range scan over entries.
    pub fn range(&self, start: &[u8], end: &[u8]) -> io::Result<Vec<(Vec<u8>, Option<Vec<u8>>)>> {
        let mut result = Vec::new();

        // Find starting block
        let start_block = match self
            .index
            .binary_search_by(|e| e.first_key.as_slice().cmp(start))
        {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) => i - 1,
        };

        for i in start_block..self.index.len() {
            let entry = &self.index[i];
            // Skip blocks that are entirely before the start
            if i + 1 < self.index.len() && self.index[i + 1].first_key.as_slice() <= start {
                continue;
            }
            // Stop if block starts after end
            if entry.first_key.as_slice() > end {
                break;
            }

            let block = self.read_block(entry.offset, entry.length)?;
            let entries = Self::decode_block(&block);
            for (k, v) in entries {
                if k.as_slice() >= start && k.as_slice() <= end {
                    result.push((k, v));
                }
            }
        }
        Ok(result)
    }

    /// Number of entries in this SSTable.
    pub fn entry_count(&self) -> usize {
        self.entry_count
    }

    fn read_block(&self, offset: u64, length: u32) -> io::Result<Vec<u8>> {
        let mut file = BufReader::new(File::open(&self.path)?);
        file.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; length as usize];
        file.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn search_block(block: &[u8], target: &[u8]) -> io::Result<Option<Option<Vec<u8>>>> {
        let mut pos = 0;
        while pos < block.len() {
            if pos + 4 > block.len() {
                break;
            }
            let key_len = u32::from_le_bytes(slice_to_array(&block[pos..pos + 4])?) as usize;
            pos += 4;
            if pos + key_len > block.len() {
                break;
            }
            let key = &block[pos..pos + key_len];
            pos += key_len;

            if pos >= block.len() {
                break;
            }
            let marker = block[pos];
            pos += 1;

            let value = if marker == VALUE_DATA {
                if pos + 4 > block.len() {
                    break;
                }
                let val_len = u32::from_le_bytes(slice_to_array(&block[pos..pos + 4])?) as usize;
                pos += 4;
                if pos + val_len > block.len() {
                    break;
                }
                let val = block[pos..pos + val_len].to_vec();
                pos += val_len;
                Some(val)
            } else {
                None
            };

            if key == target {
                return Ok(Some(value));
            }
            if key > target {
                return Ok(None); // Keys are sorted; no need to continue
            }
        }
        Ok(None)
    }

    fn decode_block(block: &[u8]) -> Vec<(Vec<u8>, Option<Vec<u8>>)> {
        let mut result = Vec::new();
        let mut pos = 0;
        while pos < block.len() {
            if pos + 4 > block.len() {
                break;
            }
            let key_len = match slice_to_array::<4>(&block[pos..pos + 4]) {
                Ok(a) => u32::from_le_bytes(a) as usize,
                Err(_) => break,
            };
            pos += 4;
            if pos + key_len > block.len() {
                break;
            }
            let key = block[pos..pos + key_len].to_vec();
            pos += key_len;

            if pos >= block.len() {
                break;
            }
            let marker = block[pos];
            pos += 1;

            let value = if marker == VALUE_DATA {
                if pos + 4 > block.len() {
                    break;
                }
                let val_len = match slice_to_array::<4>(&block[pos..pos + 4]) {
                    Ok(a) => u32::from_le_bytes(a) as usize,
                    Err(_) => break,
                };
                pos += 4;
                if pos + val_len > block.len() {
                    break;
                }
                let val = block[pos..pos + val_len].to_vec();
                pos += val_len;
                Some(val)
            } else {
                None
            };

            result.push((key, value));
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_sstable_write_read() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.sst");

        // Write
        let mut writer = SSTableWriter::new(&path, 5).unwrap();
        writer.add(b"apple", Some(b"red")).unwrap();
        writer.add(b"banana", Some(b"yellow")).unwrap();
        writer.add(b"cherry", Some(b"dark_red")).unwrap();
        let meta = writer.finish(1, 0).unwrap();
        assert_eq!(meta.entry_count, 3);

        // Read
        let reader = SSTableReader::open(&path).unwrap();
        assert_eq!(reader.entry_count(), 3);

        assert_eq!(reader.get(b"apple").unwrap(), Some(Some(b"red".to_vec())));
        assert_eq!(
            reader.get(b"banana").unwrap(),
            Some(Some(b"yellow".to_vec()))
        );
        assert_eq!(
            reader.get(b"cherry").unwrap(),
            Some(Some(b"dark_red".to_vec()))
        );
        assert_eq!(reader.get(b"date").unwrap(), None);
    }

    #[test]
    fn test_sstable_tombstone() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("tomb.sst");

        let mut writer = SSTableWriter::new(&path, 3).unwrap();
        writer.add(b"key1", Some(b"val1")).unwrap();
        writer.add(b"key2", None).unwrap(); // tombstone
        writer.add(b"key3", Some(b"val3")).unwrap();
        writer.finish(1, 0).unwrap();

        let reader = SSTableReader::open(&path).unwrap();
        assert_eq!(reader.get(b"key1").unwrap(), Some(Some(b"val1".to_vec())));
        assert_eq!(reader.get(b"key2").unwrap(), Some(None)); // tombstone
        assert_eq!(reader.get(b"key3").unwrap(), Some(Some(b"val3".to_vec())));
    }

    #[test]
    fn test_sstable_bloom_filter() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bloom.sst");

        let mut writer = SSTableWriter::new(&path, 100).unwrap();
        for i in 0..100u32 {
            let key = format!("key_{:04}", i);
            writer.add(key.as_bytes(), Some(b"value")).unwrap();
        }
        writer.finish(1, 0).unwrap();

        let reader = SSTableReader::open(&path).unwrap();
        // Keys that exist should pass bloom filter
        assert!(reader.may_contain(b"key_0050"));
        // Keys that don't exist should mostly be filtered (with small FP rate)
        let mut filtered = 0;
        for i in 1000..1100u32 {
            let key = format!("key_{:04}", i);
            if !reader.may_contain(key.as_bytes()) {
                filtered += 1;
            }
        }
        assert!(filtered > 80); // At least 80% should be filtered
    }

    #[test]
    fn test_sstable_iter() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("iter.sst");

        let mut writer = SSTableWriter::new(&path, 5).unwrap();
        writer.add(b"a", Some(b"1")).unwrap();
        writer.add(b"b", Some(b"2")).unwrap();
        writer.add(b"c", None).unwrap();
        writer.add(b"d", Some(b"4")).unwrap();
        writer.finish(1, 0).unwrap();

        let reader = SSTableReader::open(&path).unwrap();
        let entries = reader.iter().unwrap();
        assert_eq!(entries.len(), 4);
        assert_eq!(entries[0], (b"a".to_vec(), Some(b"1".to_vec())));
        assert_eq!(entries[2], (b"c".to_vec(), None));
    }

    #[test]
    fn test_sstable_range() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("range.sst");

        let mut writer = SSTableWriter::new(&path, 10).unwrap();
        for i in 0..10u8 {
            writer.add(&[i], Some(&[i * 10])).unwrap();
        }
        writer.finish(1, 0).unwrap();

        let reader = SSTableReader::open(&path).unwrap();
        let entries = reader.range(&[3], &[6]).unwrap();
        assert_eq!(entries.len(), 4); // 3, 4, 5, 6
        assert_eq!(entries[0].0, vec![3]);
        assert_eq!(entries[3].0, vec![6]);
    }

    #[test]
    fn test_sstable_large_values() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("large.sst");

        let big_value = vec![0xABu8; 8192]; // larger than block size
        let mut writer = SSTableWriter::new(&path, 3).unwrap();
        writer.add(b"big1", Some(&big_value)).unwrap();
        writer.add(b"big2", Some(&big_value)).unwrap();
        writer.finish(1, 0).unwrap();

        let reader = SSTableReader::open(&path).unwrap();
        assert_eq!(reader.get(b"big1").unwrap().unwrap().unwrap().len(), 8192);
        assert_eq!(reader.get(b"big2").unwrap().unwrap().unwrap().len(), 8192);
    }
}
