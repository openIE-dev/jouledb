//! SAM/BAM format parser — CIGAR strings, mapping flags, tag parsing.
//!
//! Parses the text-based SAM (Sequence Alignment/Map) format including
//! header lines, alignment records, CIGAR operation decomposition,
//! bitwise FLAG interpretation, and optional tag fields.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum SamError {
    EmptyInput,
    MalformedHeader(String),
    InsufficientFields { line: usize, found: usize },
    InvalidFlag { line: usize, value: String },
    InvalidPos { line: usize, value: String },
    InvalidMapq { line: usize, value: String },
    InvalidCigar(String),
    InvalidTag(String),
}

impl fmt::Display for SamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "empty SAM input"),
            Self::MalformedHeader(s) => write!(f, "malformed header: {s}"),
            Self::InsufficientFields { line, found } => {
                write!(f, "line {line}: expected >=11 fields, found {found}")
            }
            Self::InvalidFlag { line, value } => write!(f, "line {line}: invalid FLAG '{value}'"),
            Self::InvalidPos { line, value } => write!(f, "line {line}: invalid POS '{value}'"),
            Self::InvalidMapq { line, value } => write!(f, "line {line}: invalid MAPQ '{value}'"),
            Self::InvalidCigar(s) => write!(f, "invalid CIGAR: {s}"),
            Self::InvalidTag(s) => write!(f, "invalid tag: {s}"),
        }
    }
}

impl std::error::Error for SamError {}

// ── SAM flags ───────────────────────────────────────────────────

/// Bitwise FLAG constants from the SAM specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamFlags(pub u16);

impl SamFlags {
    pub const PAIRED: u16 = 0x1;
    pub const PROPER_PAIR: u16 = 0x2;
    pub const UNMAPPED: u16 = 0x4;
    pub const MATE_UNMAPPED: u16 = 0x8;
    pub const REVERSE_STRAND: u16 = 0x10;
    pub const MATE_REVERSE: u16 = 0x20;
    pub const READ1: u16 = 0x40;
    pub const READ2: u16 = 0x80;
    pub const SECONDARY: u16 = 0x100;
    pub const FAILED_QC: u16 = 0x200;
    pub const DUPLICATE: u16 = 0x400;
    pub const SUPPLEMENTARY: u16 = 0x800;

    pub fn has(self, flag: u16) -> bool { self.0 & flag != 0 }
    pub fn is_paired(self) -> bool { self.has(Self::PAIRED) }
    pub fn is_unmapped(self) -> bool { self.has(Self::UNMAPPED) }
    pub fn is_reverse(self) -> bool { self.has(Self::REVERSE_STRAND) }
    pub fn is_secondary(self) -> bool { self.has(Self::SECONDARY) }
    pub fn is_supplementary(self) -> bool { self.has(Self::SUPPLEMENTARY) }
    pub fn is_duplicate(self) -> bool { self.has(Self::DUPLICATE) }
    pub fn is_primary(self) -> bool { !self.is_secondary() && !self.is_supplementary() }
}

impl fmt::Display for SamFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:04x}", self.0)
    }
}

// ── CIGAR operations ────────────────────────────────────────────

/// A single CIGAR operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CigarOp {
    pub len: u32,
    pub op: char,
}

impl CigarOp {
    /// Whether this op consumes the reference.
    pub fn consumes_reference(self) -> bool {
        matches!(self.op, 'M' | 'D' | 'N' | '=' | 'X')
    }

    /// Whether this op consumes the query.
    pub fn consumes_query(self) -> bool {
        matches!(self.op, 'M' | 'I' | 'S' | '=' | 'X')
    }
}

impl fmt::Display for CigarOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.len, self.op)
    }
}

/// Parse a CIGAR string into operations.
pub fn parse_cigar(cigar: &str) -> Result<Vec<CigarOp>, SamError> {
    if cigar == "*" {
        return Ok(Vec::new());
    }
    let mut ops = Vec::new();
    let mut num_start = 0;
    for (i, ch) in cigar.chars().enumerate() {
        if ch.is_ascii_alphabetic() || ch == '=' {
            if i == num_start {
                return Err(SamError::InvalidCigar(format!("missing length before '{ch}'")));
            }
            let len: u32 = cigar[num_start..i]
                .parse()
                .map_err(|_| SamError::InvalidCigar(cigar.to_string()))?;
            if !matches!(ch, 'M' | 'I' | 'D' | 'N' | 'S' | 'H' | 'P' | '=' | 'X') {
                return Err(SamError::InvalidCigar(format!("unknown op '{ch}'")));
            }
            ops.push(CigarOp { len, op: ch });
            num_start = i + 1;
        }
    }
    if num_start != cigar.len() {
        return Err(SamError::InvalidCigar(cigar.to_string()));
    }
    Ok(ops)
}

/// Compute alignment length on the reference from CIGAR operations.
pub fn reference_length(ops: &[CigarOp]) -> u64 {
    ops.iter()
        .filter(|o| o.consumes_reference())
        .map(|o| o.len as u64)
        .sum()
}

/// Compute query consumed length from CIGAR operations.
pub fn query_length(ops: &[CigarOp]) -> u64 {
    ops.iter()
        .filter(|o| o.consumes_query())
        .map(|o| o.len as u64)
        .sum()
}

// ── Optional tags ───────────────────────────────────────────────

/// A typed SAM optional tag value.
#[derive(Debug, Clone, PartialEq)]
pub enum TagValue {
    Char(char),
    Int(i64),
    Float(f64),
    Str(String),
    Hex(String),
}

impl fmt::Display for TagValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Char(c) => write!(f, "A:{c}"),
            Self::Int(i) => write!(f, "i:{i}"),
            Self::Float(v) => write!(f, "f:{v}"),
            Self::Str(s) => write!(f, "Z:{s}"),
            Self::Hex(h) => write!(f, "H:{h}"),
        }
    }
}

fn parse_tag(field: &str) -> Result<(String, TagValue), SamError> {
    let parts: Vec<&str> = field.splitn(3, ':').collect();
    if parts.len() < 3 {
        return Err(SamError::InvalidTag(field.to_string()));
    }
    let key = format!("{}:{}", parts[0], parts[1]);
    let val = match parts[1] {
        "A" => TagValue::Char(parts[2].chars().next().unwrap_or('?')),
        "i" => TagValue::Int(
            parts[2].parse().map_err(|_| SamError::InvalidTag(field.to_string()))?,
        ),
        "f" => TagValue::Float(
            parts[2].parse().map_err(|_| SamError::InvalidTag(field.to_string()))?,
        ),
        "Z" => TagValue::Str(parts[2].to_string()),
        "H" => TagValue::Hex(parts[2].to_string()),
        _ => TagValue::Str(parts[2].to_string()),
    };
    Ok((key, val))
}

// ── SAM header ──────────────────────────────────────────────────

/// A parsed SAM header.
#[derive(Debug, Clone, Default)]
#[derive(PartialEq)]
pub struct SamHeader {
    pub hd: HashMap<String, String>,
    pub sequences: Vec<HashMap<String, String>>,
    pub read_groups: Vec<HashMap<String, String>>,
    pub programs: Vec<HashMap<String, String>>,
    pub comments: Vec<String>,
}

impl fmt::Display for SamHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SamHeader(seqs={}, rgs={}, pgs={})",
               self.sequences.len(), self.read_groups.len(), self.programs.len())
    }
}

// ── SAM alignment record ────────────────────────────────────────

/// A single SAM alignment record.
#[derive(Debug, Clone)]
#[derive(PartialEq)]
pub struct SamRecord {
    pub qname: String,
    pub flag: SamFlags,
    pub rname: String,
    pub pos: u64,
    pub mapq: u8,
    pub cigar: Vec<CigarOp>,
    pub rnext: String,
    pub pnext: u64,
    pub tlen: i64,
    pub seq: String,
    pub qual: String,
    pub tags: HashMap<String, TagValue>,
}

impl SamRecord {
    pub fn is_mapped(&self) -> bool { !self.flag.is_unmapped() }

    /// End position on the reference (0-based exclusive).
    pub fn end_pos(&self) -> u64 {
        self.pos.saturating_sub(1) + reference_length(&self.cigar)
    }

    /// Alignment identity from CIGAR (M= matches / alignment length).
    pub fn alignment_length(&self) -> u64 {
        reference_length(&self.cigar)
    }
}

impl fmt::Display for SamRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{} flag={} mapq={}",
               self.qname, self.rname, self.pos, self.flag, self.mapq)
    }
}

// ── Parser ──────────────────────────────────────────────────────

/// SAM format parser.
#[derive(Debug, Clone)]
pub struct SamParser {
    skip_unmapped: bool,
    min_mapq: u8,
}

impl SamParser {
    pub fn new() -> Self {
        Self { skip_unmapped: false, min_mapq: 0 }
    }

    pub fn with_skip_unmapped(mut self, s: bool) -> Self { self.skip_unmapped = s; self }
    pub fn with_min_mapq(mut self, m: u8) -> Self { self.min_mapq = m; self }

    /// Parse a complete SAM text input.
    pub fn parse(&self, input: &str) -> Result<(SamHeader, Vec<SamRecord>), SamError> {
        if input.trim().is_empty() {
            return Err(SamError::EmptyInput);
        }
        let mut header = SamHeader::default();
        let mut records = Vec::new();
        let mut line_num = 0_usize;

        for line in input.lines() {
            line_num += 1;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed.starts_with('@') {
                parse_header_line(trimmed, &mut header)?;
                continue;
            }
            let fields: Vec<&str> = trimmed.split('\t').collect();
            if fields.len() < 11 {
                return Err(SamError::InsufficientFields {
                    line: line_num,
                    found: fields.len(),
                });
            }
            let flag: u16 = fields[1]
                .parse()
                .map_err(|_| SamError::InvalidFlag { line: line_num, value: fields[1].to_string() })?;
            let pos: u64 = fields[3]
                .parse()
                .map_err(|_| SamError::InvalidPos { line: line_num, value: fields[3].to_string() })?;
            let mapq: u8 = fields[4]
                .parse()
                .map_err(|_| SamError::InvalidMapq { line: line_num, value: fields[4].to_string() })?;
            let pnext: u64 = fields[7].parse().unwrap_or(0);
            let tlen: i64 = fields[8].parse().unwrap_or(0);

            let sf = SamFlags(flag);
            if self.skip_unmapped && sf.is_unmapped() {
                continue;
            }
            if mapq < self.min_mapq {
                continue;
            }

            let cigar = parse_cigar(fields[5])?;
            let mut tags = HashMap::new();
            for f in &fields[11..] {
                if let Ok((k, v)) = parse_tag(f) {
                    tags.insert(k, v);
                }
            }

            records.push(SamRecord {
                qname: fields[0].to_string(),
                flag: sf,
                rname: fields[2].to_string(),
                pos,
                mapq,
                cigar,
                rnext: fields[6].to_string(),
                pnext,
                tlen,
                seq: fields[9].to_string(),
                qual: fields[10].to_string(),
                tags,
            });
        }
        Ok((header, records))
    }
}

impl fmt::Display for SamParser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SamParser(skip_unmapped={}, min_mapq={})", self.skip_unmapped, self.min_mapq)
    }
}

fn parse_header_line(line: &str, header: &mut SamHeader) -> Result<(), SamError> {
    let parts: Vec<&str> = line.split('\t').collect();
    if parts.is_empty() {
        return Err(SamError::MalformedHeader(line.to_string()));
    }
    let tag = parts[0];
    let mut fields = HashMap::new();
    for p in &parts[1..] {
        if let Some(pos) = p.find(':') {
            fields.insert(p[..pos].to_string(), p[pos + 1..].to_string());
        }
    }
    match tag {
        "@HD" => header.hd = fields,
        "@SQ" => header.sequences.push(fields),
        "@RG" => header.read_groups.push(fields),
        "@PG" => header.programs.push(fields),
        "@CO" => header.comments.push(parts[1..].join("\t")),
        _ => {}
    }
    Ok(())
}

// ── Convenience ─────────────────────────────────────────────────

/// Quick parse with default settings.
pub fn parse_sam(input: &str) -> Result<(SamHeader, Vec<SamRecord>), SamError> {
    SamParser::new().parse(input)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_line() -> String {
        "r001\t163\tchr1\t100\t60\t50M\t=\t200\t150\tACGT\tIIII\tNM:i:2\tMD:Z:48A1".to_string()
    }

    #[test]
    fn t01_parse_single_record() {
        let (_, recs) = parse_sam(&sample_line()).unwrap();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].qname, "r001");
    }

    #[test]
    fn t02_flag_parsing() {
        let (_, recs) = parse_sam(&sample_line()).unwrap();
        assert!(recs[0].flag.is_paired());
        assert!(!recs[0].flag.is_unmapped());
    }

    #[test]
    fn t03_cigar_parse_simple() {
        let ops = parse_cigar("50M").unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].len, 50);
        assert_eq!(ops[0].op, 'M');
    }

    #[test]
    fn t04_cigar_complex() {
        let ops = parse_cigar("3S47M2I1D5M").unwrap();
        assert_eq!(ops.len(), 5);
        assert_eq!(reference_length(&ops), 53);
    }

    #[test]
    fn t05_cigar_star() {
        let ops = parse_cigar("*").unwrap();
        assert!(ops.is_empty());
    }

    #[test]
    fn t06_cigar_invalid() {
        assert!(parse_cigar("M50").is_err());
    }

    #[test]
    fn t07_reference_length() {
        let ops = parse_cigar("10M5I10M").unwrap();
        assert_eq!(reference_length(&ops), 20);
    }

    #[test]
    fn t08_query_length() {
        let ops = parse_cigar("10M5I10M3D").unwrap();
        assert_eq!(query_length(&ops), 25);
    }

    #[test]
    fn t09_tag_parsing() {
        let (_, recs) = parse_sam(&sample_line()).unwrap();
        assert_eq!(recs[0].tags.get("NM:i"), Some(&TagValue::Int(2)));
    }

    #[test]
    fn t10_tag_string() {
        let (_, recs) = parse_sam(&sample_line()).unwrap();
        assert_eq!(recs[0].tags.get("MD:Z"), Some(&TagValue::Str("48A1".to_string())));
    }

    #[test]
    fn t11_header_sq() {
        let input = "@SQ\tSN:chr1\tLN:248956422\nr001\t0\tchr1\t1\t60\t4M\t*\t0\t0\tACGT\tIIII\n";
        let (hdr, _) = parse_sam(input).unwrap();
        assert_eq!(hdr.sequences.len(), 1);
        assert_eq!(hdr.sequences[0].get("SN").map(|s| s.as_str()), Some("chr1"));
    }

    #[test]
    fn t12_header_hd() {
        let input = "@HD\tVN:1.6\tSO:coordinate\nr001\t0\tchr1\t1\t60\t4M\t*\t0\t0\tACGT\tIIII\n";
        let (hdr, _) = parse_sam(input).unwrap();
        assert_eq!(hdr.hd.get("VN").map(|s| s.as_str()), Some("1.6"));
    }

    #[test]
    fn t13_skip_unmapped() {
        let input = "r1\t4\t*\t0\t0\t*\t*\t0\t0\tACGT\tIIII\n";
        let parser = SamParser::new().with_skip_unmapped(true);
        let (_, recs) = parser.parse(input).unwrap();
        assert!(recs.is_empty());
    }

    #[test]
    fn t14_min_mapq_filter() {
        let input = "r1\t0\tchr1\t1\t10\t4M\t*\t0\t0\tACGT\tIIII\n";
        let parser = SamParser::new().with_min_mapq(30);
        let (_, recs) = parser.parse(input).unwrap();
        assert!(recs.is_empty());
    }

    #[test]
    fn t15_flags_secondary() {
        let f = SamFlags(0x100);
        assert!(f.is_secondary());
        assert!(!f.is_primary());
    }

    #[test]
    fn t16_flags_supplementary() {
        let f = SamFlags(0x800);
        assert!(f.is_supplementary());
    }

    #[test]
    fn t17_flags_duplicate() {
        let f = SamFlags(0x400);
        assert!(f.is_duplicate());
    }

    #[test]
    fn t18_empty_input() {
        assert_eq!(parse_sam(""), Err(SamError::EmptyInput));
    }

    #[test]
    fn t19_display_record() {
        let (_, recs) = parse_sam(&sample_line()).unwrap();
        let s = format!("{}", recs[0]);
        assert!(s.contains("r001"));
        assert!(s.contains("chr1"));
    }

    #[test]
    fn t20_display_parser() {
        let p = SamParser::new().with_min_mapq(20);
        let s = format!("{p}");
        assert!(s.contains("min_mapq=20"));
    }
}
