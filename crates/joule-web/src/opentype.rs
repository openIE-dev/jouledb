//! OpenType table parsing concepts — table directory, cmap, head, hhea,
//! hmtx, name, kern tables, and glyph bounding boxes.
//!
//! Replaces opentype.js with a pure Rust OpenType metadata model.
//! Implements parsing from byte slices for core font tables.

use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// Errors during OpenType parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OtError {
    /// Not enough bytes to read the expected structure.
    UnexpectedEof,
    /// Invalid or unsupported table format.
    InvalidFormat(String),
    /// Table not found.
    TableNotFound(String),
}

impl fmt::Display for OtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "unexpected end of data"),
            Self::InvalidFormat(msg) => write!(f, "invalid format: {msg}"),
            Self::TableNotFound(tag) => write!(f, "table not found: {tag}"),
        }
    }
}

impl std::error::Error for OtError {}

// ── Helpers ────────────────────────────────────────────────────

fn read_u16(data: &[u8], offset: usize) -> Result<u16, OtError> {
    if offset + 2 > data.len() {
        return Err(OtError::UnexpectedEof);
    }
    Ok(u16::from_be_bytes([data[offset], data[offset + 1]]))
}

fn read_i16(data: &[u8], offset: usize) -> Result<i16, OtError> {
    if offset + 2 > data.len() {
        return Err(OtError::UnexpectedEof);
    }
    Ok(i16::from_be_bytes([data[offset], data[offset + 1]]))
}

fn read_u32(data: &[u8], offset: usize) -> Result<u32, OtError> {
    if offset + 4 > data.len() {
        return Err(OtError::UnexpectedEof);
    }
    Ok(u32::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

fn read_tag(data: &[u8], offset: usize) -> Result<[u8; 4], OtError> {
    if offset + 4 > data.len() {
        return Err(OtError::UnexpectedEof);
    }
    Ok([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
}

fn tag_to_string(tag: &[u8; 4]) -> String {
    String::from_utf8_lossy(tag).to_string()
}

// ── Table Directory ────────────────────────────────────────────

/// A single table record in the OpenType table directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableRecord {
    pub tag: [u8; 4],
    pub checksum: u32,
    pub offset: u32,
    pub length: u32,
}

impl TableRecord {
    pub fn tag_string(&self) -> String {
        tag_to_string(&self.tag)
    }
}

/// The OpenType table directory (font header).
#[derive(Debug, Clone)]
pub struct TableDirectory {
    pub sfnt_version: u32,
    pub num_tables: u16,
    pub records: Vec<TableRecord>,
}

impl TableDirectory {
    /// Parse the table directory from raw font bytes.
    pub fn parse(data: &[u8]) -> Result<Self, OtError> {
        let sfnt_version = read_u32(data, 0)?;
        let num_tables = read_u16(data, 4)?;
        // skip search_range(2), entry_selector(2), range_shift(2)
        let mut records = Vec::with_capacity(num_tables as usize);
        for i in 0..num_tables as usize {
            let base = 12 + i * 16;
            let tag = read_tag(data, base)?;
            let checksum = read_u32(data, base + 4)?;
            let offset = read_u32(data, base + 8)?;
            let length = read_u32(data, base + 12)?;
            records.push(TableRecord {
                tag,
                checksum,
                offset,
                length,
            });
        }
        Ok(Self {
            sfnt_version,
            num_tables,
            records,
        })
    }

    /// Find a table record by tag string (e.g. "cmap", "head").
    pub fn find_table(&self, tag: &str) -> Option<&TableRecord> {
        let tag_bytes: [u8; 4] = if tag.len() >= 4 {
            [tag.as_bytes()[0], tag.as_bytes()[1], tag.as_bytes()[2], tag.as_bytes()[3]]
        } else {
            return None;
        };
        self.records.iter().find(|r| r.tag == tag_bytes)
    }

    /// Get the raw bytes for a table given font data.
    pub fn table_data<'a>(&self, font_data: &'a [u8], tag: &str) -> Result<&'a [u8], OtError> {
        let record = self
            .find_table(tag)
            .ok_or_else(|| OtError::TableNotFound(tag.to_string()))?;
        let start = record.offset as usize;
        let end = start + record.length as usize;
        if end > font_data.len() {
            return Err(OtError::UnexpectedEof);
        }
        Ok(&font_data[start..end])
    }
}

// ── Head Table ─────────────────────────────────────────────────

/// The `head` table — global font header.
#[derive(Debug, Clone, PartialEq)]
pub struct HeadTable {
    pub major_version: u16,
    pub minor_version: u16,
    pub font_revision: u32,
    pub flags: u16,
    pub units_per_em: u16,
    pub x_min: i16,
    pub y_min: i16,
    pub x_max: i16,
    pub y_max: i16,
    pub mac_style: u16,
    pub index_to_loc_format: i16,
}

impl HeadTable {
    /// Parse the head table from its raw bytes.
    pub fn parse(data: &[u8]) -> Result<Self, OtError> {
        if data.len() < 54 {
            return Err(OtError::UnexpectedEof);
        }
        Ok(Self {
            major_version: read_u16(data, 0)?,
            minor_version: read_u16(data, 2)?,
            font_revision: read_u32(data, 4)?,
            flags: read_u16(data, 16)?,
            units_per_em: read_u16(data, 18)?,
            x_min: read_i16(data, 36)?,
            y_min: read_i16(data, 38)?,
            x_max: read_i16(data, 40)?,
            y_max: read_i16(data, 42)?,
            mac_style: read_u16(data, 44)?,
            index_to_loc_format: read_i16(data, 50)?,
        })
    }

    /// Bounding box as (x_min, y_min, x_max, y_max).
    pub fn bbox(&self) -> (i16, i16, i16, i16) {
        (self.x_min, self.y_min, self.x_max, self.y_max)
    }

    /// Is the font bold (bit 0 of mac_style)?
    pub fn is_bold(&self) -> bool {
        self.mac_style & 1 != 0
    }

    /// Is the font italic (bit 1 of mac_style)?
    pub fn is_italic(&self) -> bool {
        self.mac_style & 2 != 0
    }
}

// ── Hhea Table ─────────────────────────────────────────────────

/// The `hhea` table — horizontal header.
#[derive(Debug, Clone, PartialEq)]
pub struct HheaTable {
    pub ascender: i16,
    pub descender: i16,
    pub line_gap: i16,
    pub advance_width_max: u16,
    pub number_of_h_metrics: u16,
}

impl HheaTable {
    pub fn parse(data: &[u8]) -> Result<Self, OtError> {
        if data.len() < 36 {
            return Err(OtError::UnexpectedEof);
        }
        Ok(Self {
            ascender: read_i16(data, 4)?,
            descender: read_i16(data, 6)?,
            line_gap: read_i16(data, 8)?,
            advance_width_max: read_u16(data, 10)?,
            number_of_h_metrics: read_u16(data, 34)?,
        })
    }
}

// ── Hmtx Table ─────────────────────────────────────────────────

/// Horizontal metrics for a single glyph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HMetric {
    pub advance_width: u16,
    pub lsb: i16,
}

/// The `hmtx` table — horizontal metrics.
#[derive(Debug, Clone)]
pub struct HmtxTable {
    pub metrics: Vec<HMetric>,
}

impl HmtxTable {
    /// Parse hmtx given the number of h_metrics from hhea.
    pub fn parse(data: &[u8], num_h_metrics: u16) -> Result<Self, OtError> {
        let n = num_h_metrics as usize;
        let required = n * 4;
        if data.len() < required {
            return Err(OtError::UnexpectedEof);
        }
        let mut metrics = Vec::with_capacity(n);
        for i in 0..n {
            let off = i * 4;
            metrics.push(HMetric {
                advance_width: read_u16(data, off)?,
                lsb: read_i16(data, off + 2)?,
            });
        }
        Ok(Self { metrics })
    }

    /// Get the metric for a glyph index.
    pub fn get(&self, glyph_id: u16) -> Option<&HMetric> {
        let idx = glyph_id as usize;
        if idx < self.metrics.len() {
            Some(&self.metrics[idx])
        } else {
            // Glyphs beyond num_h_metrics use the last entry's advance_width.
            self.metrics.last()
        }
    }
}

// ── Cmap Table (Format 4) ──────────────────────────────────────

/// Character to glyph mapping (Format 4 segment mapping).
#[derive(Debug, Clone)]
pub struct CmapFormat4 {
    pub segments: Vec<CmapSegment>,
}

/// A segment in cmap format 4.
#[derive(Debug, Clone)]
pub struct CmapSegment {
    pub start_code: u16,
    pub end_code: u16,
    pub id_delta: i16,
    pub id_range_offset: u16,
}

impl CmapFormat4 {
    /// Parse a cmap format 4 subtable.
    pub fn parse(data: &[u8]) -> Result<Self, OtError> {
        let format = read_u16(data, 0)?;
        if format != 4 {
            return Err(OtError::InvalidFormat(format!("expected cmap format 4, got {format}")));
        }
        let seg_count_x2 = read_u16(data, 6)?;
        let seg_count = (seg_count_x2 / 2) as usize;

        let end_codes_start = 14;
        // +2 for reserved pad
        let start_codes_start = end_codes_start + seg_count * 2 + 2;
        let id_deltas_start = start_codes_start + seg_count * 2;
        let id_range_offsets_start = id_deltas_start + seg_count * 2;

        let mut segments = Vec::with_capacity(seg_count);
        for i in 0..seg_count {
            let end_code = read_u16(data, end_codes_start + i * 2)?;
            let start_code = read_u16(data, start_codes_start + i * 2)?;
            let id_delta = read_i16(data, id_deltas_start + i * 2)?;
            let id_range_offset = read_u16(data, id_range_offsets_start + i * 2)?;
            segments.push(CmapSegment {
                start_code,
                end_code,
                id_delta,
                id_range_offset,
            });
        }

        Ok(Self { segments })
    }

    /// Map a Unicode code point to a glyph ID (simplified — only handles
    /// id_range_offset == 0 segments).
    pub fn map_char(&self, codepoint: u16) -> Option<u16> {
        for seg in &self.segments {
            if codepoint >= seg.start_code && codepoint <= seg.end_code {
                if seg.id_range_offset == 0 {
                    let glyph = (codepoint as i32 + seg.id_delta as i32) as u16;
                    return if glyph == 0 { None } else { Some(glyph) };
                }
                // Non-zero id_range_offset requires glyph index array lookup
                // which needs the subtable data — return None for now.
                return None;
            }
        }
        None
    }
}

// ── Name Table ─────────────────────────────────────────────────

/// Name IDs from the name table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NameId {
    Copyright = 0,
    FontFamily = 1,
    FontSubfamily = 2,
    UniqueId = 3,
    FullName = 4,
    Version = 5,
    PostScriptName = 6,
    Trademark = 7,
    Manufacturer = 8,
    Designer = 9,
    Description = 10,
    UrlVendor = 11,
    UrlDesigner = 12,
    License = 13,
    LicenseUrl = 14,
    TypographicFamily = 16,
    TypographicSubfamily = 17,
}

impl NameId {
    pub fn from_u16(val: u16) -> Option<Self> {
        match val {
            0 => Some(Self::Copyright),
            1 => Some(Self::FontFamily),
            2 => Some(Self::FontSubfamily),
            3 => Some(Self::UniqueId),
            4 => Some(Self::FullName),
            5 => Some(Self::Version),
            6 => Some(Self::PostScriptName),
            7 => Some(Self::Trademark),
            8 => Some(Self::Manufacturer),
            9 => Some(Self::Designer),
            10 => Some(Self::Description),
            11 => Some(Self::UrlVendor),
            12 => Some(Self::UrlDesigner),
            13 => Some(Self::License),
            14 => Some(Self::LicenseUrl),
            16 => Some(Self::TypographicFamily),
            17 => Some(Self::TypographicSubfamily),
            _ => None,
        }
    }
}

/// The `name` table.
#[derive(Debug, Clone)]
pub struct NameTable {
    pub entries: HashMap<u16, String>,
}

impl NameTable {
    /// Parse the name table from raw bytes.
    pub fn parse(data: &[u8]) -> Result<Self, OtError> {
        if data.len() < 6 {
            return Err(OtError::UnexpectedEof);
        }
        let count = read_u16(data, 2)? as usize;
        let string_offset = read_u16(data, 4)? as usize;

        let mut entries = HashMap::new();
        for i in 0..count {
            let rec_off = 6 + i * 12;
            if rec_off + 12 > data.len() {
                break;
            }
            let platform_id = read_u16(data, rec_off)?;
            let encoding_id = read_u16(data, rec_off + 2)?;
            let name_id = read_u16(data, rec_off + 6)?;
            let length = read_u16(data, rec_off + 8)? as usize;
            let offset = read_u16(data, rec_off + 10)? as usize;

            let str_start = string_offset + offset;
            if str_start + length > data.len() {
                continue;
            }
            let raw = &data[str_start..str_start + length];

            // Platform 3 (Windows) encoding 1 (Unicode BMP) = UTF-16BE.
            // Platform 1 (Mac) encoding 0 = MacRoman ≈ ASCII for Latin.
            let decoded = if platform_id == 3 && encoding_id == 1 {
                let chars: Vec<u16> = raw
                    .chunks(2)
                    .filter_map(|c| {
                        if c.len() == 2 {
                            Some(u16::from_be_bytes([c[0], c[1]]))
                        } else {
                            None
                        }
                    })
                    .collect();
                String::from_utf16_lossy(&chars)
            } else if platform_id == 1 {
                String::from_utf8_lossy(raw).to_string()
            } else {
                continue;
            };

            // Prefer Windows entries; don't overwrite with Mac.
            if platform_id == 3 || !entries.contains_key(&name_id) {
                entries.insert(name_id, decoded);
            }
        }

        Ok(Self { entries })
    }

    /// Get a name entry by NameId.
    pub fn get(&self, id: NameId) -> Option<&str> {
        self.entries.get(&(id as u16)).map(|s| s.as_str())
    }

    /// Get a name entry by raw ID.
    pub fn get_raw(&self, id: u16) -> Option<&str> {
        self.entries.get(&id).map(|s| s.as_str())
    }
}

// ── Kern Table ─────────────────────────────────────────────────

/// A kerning pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernPair {
    pub left: u16,
    pub right: u16,
    pub value: i16,
}

/// The `kern` table (format 0).
#[derive(Debug, Clone)]
pub struct KernTable {
    pub pairs: Vec<KernPair>,
    index: HashMap<(u16, u16), i16>,
}

impl KernTable {
    /// Parse kern table format 0 from raw bytes.
    pub fn parse(data: &[u8]) -> Result<Self, OtError> {
        if data.len() < 4 {
            return Err(OtError::UnexpectedEof);
        }
        // version(2) + nTables(2)
        let _version = read_u16(data, 0)?;
        let n_tables = read_u16(data, 2)?;
        if n_tables == 0 {
            return Ok(Self {
                pairs: Vec::new(),
                index: HashMap::new(),
            });
        }

        // First subtable starts at offset 4.
        // subtable header: version(2) + length(2) + coverage(2) = 6 bytes
        // Then for format 0: nPairs(2) + searchRange(2) + entrySelector(2) + rangeShift(2) = 8
        let sub_start = 4;
        if sub_start + 14 > data.len() {
            return Err(OtError::UnexpectedEof);
        }
        let n_pairs = read_u16(data, sub_start + 6)? as usize;
        let pairs_start = sub_start + 14;

        let mut pairs = Vec::with_capacity(n_pairs);
        let mut index = HashMap::with_capacity(n_pairs);
        for i in 0..n_pairs {
            let off = pairs_start + i * 6;
            if off + 6 > data.len() {
                break;
            }
            let left = read_u16(data, off)?;
            let right = read_u16(data, off + 2)?;
            let value = read_i16(data, off + 4)?;
            pairs.push(KernPair { left, right, value });
            index.insert((left, right), value);
        }

        Ok(Self { pairs, index })
    }

    /// Get the kerning value for a glyph pair.
    pub fn get_kerning(&self, left: u16, right: u16) -> i16 {
        self.index.get(&(left, right)).copied().unwrap_or(0)
    }
}

// ── Glyph Bounding Box ────────────────────────────────────────

/// Bounding box for a glyph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlyphBBox {
    pub x_min: i16,
    pub y_min: i16,
    pub x_max: i16,
    pub y_max: i16,
}

impl GlyphBBox {
    pub fn new(x_min: i16, y_min: i16, x_max: i16, y_max: i16) -> Self {
        Self {
            x_min,
            y_min,
            x_max,
            y_max,
        }
    }

    pub fn width(&self) -> i16 {
        self.x_max - self.x_min
    }

    pub fn height(&self) -> i16 {
        self.y_max - self.y_min
    }

    /// Parse from glyf table entry (first 10 bytes: numberOfContours + bbox).
    pub fn parse_from_glyf(data: &[u8]) -> Result<Self, OtError> {
        if data.len() < 10 {
            return Err(OtError::UnexpectedEof);
        }
        Ok(Self {
            x_min: read_i16(data, 2)?,
            y_min: read_i16(data, 4)?,
            x_max: read_i16(data, 6)?,
            y_max: read_i16(data, 8)?,
        })
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_table_directory_bytes(tables: &[(&[u8; 4], u32, u32)]) -> Vec<u8> {
        let num_tables = tables.len() as u16;
        let mut data = Vec::new();
        // sfnt_version = 0x00010000
        data.extend_from_slice(&0x00010000u32.to_be_bytes());
        data.extend_from_slice(&num_tables.to_be_bytes());
        data.extend_from_slice(&0u16.to_be_bytes()); // search_range
        data.extend_from_slice(&0u16.to_be_bytes()); // entry_selector
        data.extend_from_slice(&0u16.to_be_bytes()); // range_shift
        for (tag, offset, length) in tables {
            data.extend_from_slice(*tag);
            data.extend_from_slice(&0u32.to_be_bytes()); // checksum
            data.extend_from_slice(&offset.to_be_bytes());
            data.extend_from_slice(&length.to_be_bytes());
        }
        data
    }

    #[test]
    fn parse_table_directory() {
        let data = make_table_directory_bytes(&[
            (b"head", 100, 54),
            (b"cmap", 200, 80),
        ]);
        let dir = TableDirectory::parse(&data).unwrap();
        assert_eq!(dir.num_tables, 2);
        assert_eq!(dir.records[0].tag_string(), "head");
        assert_eq!(dir.records[1].tag_string(), "cmap");
    }

    #[test]
    fn find_table() {
        let data = make_table_directory_bytes(&[
            (b"head", 100, 54),
            (b"hhea", 160, 36),
        ]);
        let dir = TableDirectory::parse(&data).unwrap();
        assert!(dir.find_table("head").is_some());
        assert!(dir.find_table("hhea").is_some());
        assert!(dir.find_table("cmap").is_none());
    }

    #[test]
    fn parse_head_table() {
        let mut data = vec![0u8; 54];
        // major_version = 1
        data[0] = 0;
        data[1] = 1;
        // units_per_em = 1000 at offset 18
        let upm = 1000u16.to_be_bytes();
        data[18] = upm[0];
        data[19] = upm[1];
        // x_min = -100 at offset 36
        let xmin = (-100i16).to_be_bytes();
        data[36] = xmin[0];
        data[37] = xmin[1];
        // y_min = -200 at offset 38
        let ymin = (-200i16).to_be_bytes();
        data[38] = ymin[0];
        data[39] = ymin[1];
        // x_max = 1200 at offset 40
        let xmax = 1200i16.to_be_bytes();
        data[40] = xmax[0];
        data[41] = xmax[1];
        // y_max = 900 at offset 42
        let ymax = 900i16.to_be_bytes();
        data[42] = ymax[0];
        data[43] = ymax[1];

        let head = HeadTable::parse(&data).unwrap();
        assert_eq!(head.units_per_em, 1000);
        assert_eq!(head.bbox(), (-100, -200, 1200, 900));
    }

    #[test]
    fn parse_hhea_table() {
        let mut data = vec![0u8; 36];
        // ascender = 800 at offset 4
        let asc = 800i16.to_be_bytes();
        data[4] = asc[0];
        data[5] = asc[1];
        // descender = -200 at offset 6
        let desc = (-200i16).to_be_bytes();
        data[6] = desc[0];
        data[7] = desc[1];
        // number_of_h_metrics = 3 at offset 34
        let nhm = 3u16.to_be_bytes();
        data[34] = nhm[0];
        data[35] = nhm[1];

        let hhea = HheaTable::parse(&data).unwrap();
        assert_eq!(hhea.ascender, 800);
        assert_eq!(hhea.descender, -200);
        assert_eq!(hhea.number_of_h_metrics, 3);
    }

    #[test]
    fn parse_hmtx_table() {
        // 3 metrics: (600, 50), (500, 30), (400, 10)
        let mut data = Vec::new();
        for (aw, lsb) in [(600u16, 50i16), (500, 30), (400, 10)] {
            data.extend_from_slice(&aw.to_be_bytes());
            data.extend_from_slice(&lsb.to_be_bytes());
        }
        let hmtx = HmtxTable::parse(&data, 3).unwrap();
        assert_eq!(hmtx.metrics.len(), 3);
        assert_eq!(hmtx.metrics[0].advance_width, 600);
        assert_eq!(hmtx.metrics[0].lsb, 50);
        assert_eq!(hmtx.get(2).unwrap().advance_width, 400);
    }

    #[test]
    fn cmap_format4_simple() {
        // Build a minimal format 4 subtable.
        // 2 segments: [32..90] with delta=100, [0xFFFF..0xFFFF] sentinel
        let seg_count: u16 = 2;
        let seg_count_x2 = seg_count * 2;
        let mut data = Vec::new();
        data.extend_from_slice(&4u16.to_be_bytes()); // format
        data.extend_from_slice(&0u16.to_be_bytes()); // length (unused here)
        data.extend_from_slice(&0u16.to_be_bytes()); // language
        data.extend_from_slice(&seg_count_x2.to_be_bytes());
        data.extend_from_slice(&0u16.to_be_bytes()); // search_range
        data.extend_from_slice(&0u16.to_be_bytes()); // entry_selector
        data.extend_from_slice(&0u16.to_be_bytes()); // range_shift
        // end codes
        data.extend_from_slice(&90u16.to_be_bytes());
        data.extend_from_slice(&0xFFFFu16.to_be_bytes());
        // reserved pad
        data.extend_from_slice(&0u16.to_be_bytes());
        // start codes
        data.extend_from_slice(&32u16.to_be_bytes());
        data.extend_from_slice(&0xFFFFu16.to_be_bytes());
        // id deltas
        data.extend_from_slice(&100i16.to_be_bytes());
        data.extend_from_slice(&1i16.to_be_bytes());
        // id range offsets
        data.extend_from_slice(&0u16.to_be_bytes());
        data.extend_from_slice(&0u16.to_be_bytes());

        let cmap = CmapFormat4::parse(&data).unwrap();
        assert_eq!(cmap.segments.len(), 2);
        // 'A' = 65, glyph = 65 + 100 = 165
        assert_eq!(cmap.map_char(65), Some(165));
        // Below range
        assert_eq!(cmap.map_char(10), None);
    }

    #[test]
    fn name_table_mac_roman() {
        // Build a minimal name table with one Mac Roman entry.
        let name_str = b"TestFont";
        let str_offset = 6 + 12; // header(6) + 1 record(12)
        let mut data = Vec::new();
        data.extend_from_slice(&0u16.to_be_bytes()); // format 0
        data.extend_from_slice(&1u16.to_be_bytes()); // count = 1
        data.extend_from_slice(&(str_offset as u16).to_be_bytes());
        // Name record: platform=1(Mac), encoding=0, language=0, nameId=4(FullName), length, offset
        data.extend_from_slice(&1u16.to_be_bytes()); // platform
        data.extend_from_slice(&0u16.to_be_bytes()); // encoding
        data.extend_from_slice(&0u16.to_be_bytes()); // language
        data.extend_from_slice(&4u16.to_be_bytes()); // name ID
        data.extend_from_slice(&(name_str.len() as u16).to_be_bytes());
        data.extend_from_slice(&0u16.to_be_bytes()); // offset in string storage
        data.extend_from_slice(name_str);

        let name = NameTable::parse(&data).unwrap();
        assert_eq!(name.get(NameId::FullName), Some("TestFont"));
    }

    #[test]
    fn kern_table_basic() {
        let mut data = Vec::new();
        data.extend_from_slice(&0u16.to_be_bytes()); // version
        data.extend_from_slice(&1u16.to_be_bytes()); // nTables
        // subtable header
        data.extend_from_slice(&0u16.to_be_bytes()); // version
        data.extend_from_slice(&0u16.to_be_bytes()); // length
        data.extend_from_slice(&1u16.to_be_bytes()); // coverage (format 0, horizontal)
        // format 0 header
        data.extend_from_slice(&2u16.to_be_bytes()); // nPairs
        data.extend_from_slice(&0u16.to_be_bytes()); // search_range
        data.extend_from_slice(&0u16.to_be_bytes()); // entry_selector
        data.extend_from_slice(&0u16.to_be_bytes()); // range_shift
        // pair 1: glyph 10, glyph 20, value = -50
        data.extend_from_slice(&10u16.to_be_bytes());
        data.extend_from_slice(&20u16.to_be_bytes());
        data.extend_from_slice(&(-50i16).to_be_bytes());
        // pair 2: glyph 30, glyph 40, value = 25
        data.extend_from_slice(&30u16.to_be_bytes());
        data.extend_from_slice(&40u16.to_be_bytes());
        data.extend_from_slice(&25i16.to_be_bytes());

        let kern = KernTable::parse(&data).unwrap();
        assert_eq!(kern.pairs.len(), 2);
        assert_eq!(kern.get_kerning(10, 20), -50);
        assert_eq!(kern.get_kerning(30, 40), 25);
        assert_eq!(kern.get_kerning(99, 99), 0);
    }

    #[test]
    fn glyph_bbox() {
        let bbox = GlyphBBox::new(-10, -200, 800, 700);
        assert_eq!(bbox.width(), 810);
        assert_eq!(bbox.height(), 900);
    }

    #[test]
    fn glyph_bbox_parse() {
        let mut data = vec![0u8; 10];
        // num_contours at 0-1 (skip)
        let vals: [(i16, usize); 4] = [(-10, 2), (-200, 4), (800, 6), (700, 8)];
        for (val, off) in vals {
            let bytes = val.to_be_bytes();
            data[off] = bytes[0];
            data[off + 1] = bytes[1];
        }
        let bbox = GlyphBBox::parse_from_glyf(&data).unwrap();
        assert_eq!(bbox.x_min, -10);
        assert_eq!(bbox.y_max, 700);
    }

    #[test]
    fn error_display() {
        assert_eq!(OtError::UnexpectedEof.to_string(), "unexpected end of data");
        assert!(OtError::InvalidFormat("bad".into()).to_string().contains("bad"));
    }
}
