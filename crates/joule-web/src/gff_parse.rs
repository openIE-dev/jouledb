//! GFF3/GTF annotation parser — feature hierarchy, attribute parsing.
//!
//! Parses General Feature Format version 3 (GFF3) and Gene Transfer Format
//! (GTF) annotation files. Supports parent-child feature hierarchies,
//! attribute key=value parsing, feature filtering, and coordinate queries.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum GffError {
    EmptyInput,
    InsufficientFields { line: usize, found: usize },
    InvalidCoordinate { line: usize, field: String, value: String },
    InvalidStrand { line: usize, value: String },
    MalformedAttribute { line: usize, attr: String },
    CircularHierarchy(String),
}

impl fmt::Display for GffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "empty GFF input"),
            Self::InsufficientFields { line, found } => {
                write!(f, "line {line}: expected 9 fields, found {found}")
            }
            Self::InvalidCoordinate { line, field, value } => {
                write!(f, "line {line}: invalid {field} '{value}'")
            }
            Self::InvalidStrand { line, value } => {
                write!(f, "line {line}: invalid strand '{value}'")
            }
            Self::MalformedAttribute { line, attr } => {
                write!(f, "line {line}: malformed attribute '{attr}'")
            }
            Self::CircularHierarchy(id) => write!(f, "circular hierarchy at '{id}'"),
        }
    }
}

impl std::error::Error for GffError {}

// ── Format variant ──────────────────────────────────────────────

/// GFF3 vs GTF attribute style.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GffFormat {
    /// GFF3: `Key=Value` pairs separated by `;`.
    Gff3,
    /// GTF: `key "value";` pairs separated by ` `.
    Gtf,
}

impl fmt::Display for GffFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gff3 => write!(f, "GFF3"),
            Self::Gtf => write!(f, "GTF"),
        }
    }
}

// ── Strand ──────────────────────────────────────────────────────

/// Genomic strand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strand {
    Plus,
    Minus,
    Unstranded,
    Unknown,
}

impl Strand {
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            '+' => Some(Self::Plus),
            '-' => Some(Self::Minus),
            '.' => Some(Self::Unstranded),
            '?' => Some(Self::Unknown),
            _ => None,
        }
    }
}

impl fmt::Display for Strand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plus => write!(f, "+"),
            Self::Minus => write!(f, "-"),
            Self::Unstranded => write!(f, "."),
            Self::Unknown => write!(f, "?"),
        }
    }
}

// ── Feature record ──────────────────────────────────────────────

/// A single GFF/GTF feature record.
#[derive(Debug, Clone, PartialEq)]
pub struct GffRecord {
    pub seqid: String,
    pub source: String,
    pub feature_type: String,
    pub start: u64,
    pub end: u64,
    pub score: Option<f64>,
    pub strand: Strand,
    pub phase: Option<u8>,
    pub attributes: HashMap<String, Vec<String>>,
}

impl GffRecord {
    pub fn new(seqid: &str, feature_type: &str, start: u64, end: u64) -> Self {
        Self {
            seqid: seqid.to_string(),
            source: ".".to_string(),
            feature_type: feature_type.to_string(),
            start,
            end,
            score: None,
            strand: Strand::Unstranded,
            phase: None,
            attributes: HashMap::new(),
        }
    }

    pub fn with_source(mut self, s: &str) -> Self {
        self.source = s.to_string();
        self
    }

    pub fn with_strand(mut self, s: Strand) -> Self {
        self.strand = s;
        self
    }

    pub fn with_score(mut self, s: f64) -> Self {
        self.score = Some(s);
        self
    }

    pub fn with_phase(mut self, p: u8) -> Self {
        self.phase = Some(p);
        self
    }

    pub fn with_attribute(mut self, key: &str, value: &str) -> Self {
        self.attributes
            .entry(key.to_string())
            .or_default()
            .push(value.to_string());
        self
    }

    /// Feature length in bases (1-based inclusive).
    pub fn length(&self) -> u64 {
        if self.end >= self.start {
            self.end - self.start + 1
        } else {
            0
        }
    }

    /// True if this feature overlaps the given interval.
    pub fn overlaps(&self, start: u64, end: u64) -> bool {
        self.start <= end && self.end >= start
    }

    /// True if this feature fully contains the given interval.
    pub fn contains(&self, start: u64, end: u64) -> bool {
        self.start <= start && self.end >= end
    }

    /// Get the GFF3 ID attribute.
    pub fn id(&self) -> Option<&str> {
        self.attributes.get("ID").and_then(|v| v.first()).map(|s| s.as_str())
    }

    /// Get the GFF3 Parent attribute(s).
    pub fn parents(&self) -> Vec<&str> {
        self.attributes
            .get("Parent")
            .map(|v| v.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get the GTF gene_id attribute.
    pub fn gene_id(&self) -> Option<&str> {
        self.attributes.get("gene_id").and_then(|v| v.first()).map(|s| s.as_str())
    }

    /// Get the GTF transcript_id attribute.
    pub fn transcript_id(&self) -> Option<&str> {
        self.attributes.get("transcript_id").and_then(|v| v.first()).map(|s| s.as_str())
    }

    /// Get any attribute by key.
    pub fn attr(&self, key: &str) -> Option<&str> {
        self.attributes.get(key).and_then(|v| v.first()).map(|s| s.as_str())
    }
}

impl fmt::Display for GffRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}-{} {} ({})",
               self.seqid, self.start, self.end, self.feature_type, self.strand)
    }
}

// ── Feature collection ──────────────────────────────────────────

/// A collection of GFF features with hierarchy support.
#[derive(Debug, Clone)]
#[derive(PartialEq)]
pub struct GffDocument {
    pub records: Vec<GffRecord>,
    pub directives: Vec<String>,
}

impl GffDocument {
    /// Filter features by type.
    pub fn by_type(&self, ft: &str) -> Vec<&GffRecord> {
        self.records.iter().filter(|r| r.feature_type == ft).collect()
    }

    /// Filter features by seqid.
    pub fn by_seqid(&self, seqid: &str) -> Vec<&GffRecord> {
        self.records.iter().filter(|r| r.seqid == seqid).collect()
    }

    /// Find features overlapping a region.
    pub fn overlapping(&self, seqid: &str, start: u64, end: u64) -> Vec<&GffRecord> {
        self.records
            .iter()
            .filter(|r| r.seqid == seqid && r.overlaps(start, end))
            .collect()
    }

    /// Build parent-to-children map (GFF3 ID/Parent).
    pub fn children_of(&self, parent_id: &str) -> Vec<&GffRecord> {
        self.records
            .iter()
            .filter(|r| r.parents().contains(&parent_id))
            .collect()
    }
}

impl fmt::Display for GffDocument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GffDocument({} features)", self.records.len())
    }
}

// ── Parser ──────────────────────────────────────────────────────

/// GFF3/GTF parser.
#[derive(Debug, Clone)]
pub struct GffParser {
    format: GffFormat,
    feature_filter: Option<String>,
}

impl GffParser {
    pub fn new(format: GffFormat) -> Self {
        Self { format, feature_filter: None }
    }

    pub fn with_feature_filter(mut self, ft: &str) -> Self {
        self.feature_filter = Some(ft.to_string());
        self
    }

    /// Parse a complete GFF/GTF string.
    pub fn parse(&self, input: &str) -> Result<GffDocument, GffError> {
        if input.trim().is_empty() {
            return Err(GffError::EmptyInput);
        }
        let mut records = Vec::new();
        let mut directives = Vec::new();
        let mut line_num = 0_usize;

        for line in input.lines() {
            line_num += 1;
            let trimmed = line.trim();
            if trimmed.is_empty() { continue; }
            if trimmed.starts_with("##") {
                directives.push(trimmed.to_string());
                continue;
            }
            if trimmed.starts_with('#') { continue; }

            let fields: Vec<&str> = trimmed.split('\t').collect();
            if fields.len() < 9 {
                return Err(GffError::InsufficientFields { line: line_num, found: fields.len() });
            }

            let feature_type = fields[2].to_string();
            if let Some(ref ft) = self.feature_filter {
                if feature_type != *ft { continue; }
            }

            let start: u64 = fields[3]
                .parse()
                .map_err(|_| GffError::InvalidCoordinate {
                    line: line_num,
                    field: "start".to_string(),
                    value: fields[3].to_string(),
                })?;
            let end: u64 = fields[4]
                .parse()
                .map_err(|_| GffError::InvalidCoordinate {
                    line: line_num,
                    field: "end".to_string(),
                    value: fields[4].to_string(),
                })?;
            let score = if fields[5] == "." { None } else {
                Some(fields[5].parse::<f64>().map_err(|_| GffError::InvalidCoordinate {
                    line: line_num,
                    field: "score".to_string(),
                    value: fields[5].to_string(),
                })?)
            };
            let strand = if fields[6].len() == 1 {
                Strand::from_char(fields[6].chars().next().unwrap())
                    .ok_or_else(|| GffError::InvalidStrand {
                        line: line_num,
                        value: fields[6].to_string(),
                    })?
            } else {
                return Err(GffError::InvalidStrand { line: line_num, value: fields[6].to_string() });
            };
            let phase = if fields[7] == "." { None } else {
                Some(fields[7].parse::<u8>().map_err(|_| GffError::InvalidCoordinate {
                    line: line_num,
                    field: "phase".to_string(),
                    value: fields[7].to_string(),
                })?)
            };

            let attributes = match self.format {
                GffFormat::Gff3 => parse_gff3_attributes(fields[8]),
                GffFormat::Gtf => parse_gtf_attributes(fields[8]),
            };

            records.push(GffRecord {
                seqid: fields[0].to_string(),
                source: fields[1].to_string(),
                feature_type,
                start,
                end,
                score,
                strand,
                phase,
                attributes,
            });
        }
        Ok(GffDocument { records, directives })
    }
}

impl fmt::Display for GffParser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GffParser(format={}, filter={:?})", self.format, self.feature_filter)
    }
}

// ── Attribute parsers ───────────────────────────────────────────

fn parse_gff3_attributes(field: &str) -> HashMap<String, Vec<String>> {
    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    if field == "." { return map; }
    for pair in field.split(';') {
        let pair = pair.trim();
        if pair.is_empty() { continue; }
        if let Some(eq) = pair.find('=') {
            let key = pair[..eq].to_string();
            let value = pair[eq + 1..].to_string();
            // GFF3 allows comma-separated multi-values
            for v in value.split(',') {
                map.entry(key.clone()).or_default().push(url_decode(v.trim()));
            }
        }
    }
    map
}

fn parse_gtf_attributes(field: &str) -> HashMap<String, Vec<String>> {
    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    if field == "." { return map; }
    for pair in field.split(';') {
        let pair = pair.trim();
        if pair.is_empty() { continue; }
        // GTF format: key "value"
        let parts: Vec<&str> = pair.splitn(2, ' ').collect();
        if parts.len() == 2 {
            let key = parts[0].to_string();
            let value = parts[1].trim_matches('"').to_string();
            map.entry(key).or_default().push(value);
        }
    }
    map
}

fn url_decode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '%' {
            let h: String = chars.by_ref().take(2).collect();
            if let Ok(b) = u8::from_str_radix(&h, 16) {
                out.push(b as char);
            } else {
                out.push('%');
                out.push_str(&h);
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Quick GFF3 parse.
pub fn parse_gff3(input: &str) -> Result<GffDocument, GffError> {
    GffParser::new(GffFormat::Gff3).parse(input)
}

/// Quick GTF parse.
pub fn parse_gtf(input: &str) -> Result<GffDocument, GffError> {
    GffParser::new(GffFormat::Gtf).parse(input)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_gff3() -> String {
        "##gff-version 3\n\
         chr1\tENSEMBL\tgene\t1000\t5000\t.\t+\t.\tID=gene01;Name=BRCA1\n\
         chr1\tENSEMBL\tmRNA\t1000\t5000\t.\t+\t.\tID=mrna01;Parent=gene01\n\
         chr1\tENSEMBL\texon\t1000\t1200\t.\t+\t.\tID=exon01;Parent=mrna01\n\
         chr1\tENSEMBL\texon\t3000\t5000\t.\t+\t.\tID=exon02;Parent=mrna01\n"
            .to_string()
    }

    #[test]
    fn t01_parse_gff3() {
        let doc = parse_gff3(&sample_gff3()).unwrap();
        assert_eq!(doc.records.len(), 4);
    }

    #[test]
    fn t02_feature_type() {
        let doc = parse_gff3(&sample_gff3()).unwrap();
        assert_eq!(doc.records[0].feature_type, "gene");
    }

    #[test]
    fn t03_coordinates() {
        let doc = parse_gff3(&sample_gff3()).unwrap();
        assert_eq!(doc.records[0].start, 1000);
        assert_eq!(doc.records[0].end, 5000);
    }

    #[test]
    fn t04_feature_length() {
        let doc = parse_gff3(&sample_gff3()).unwrap();
        assert_eq!(doc.records[0].length(), 4001);
    }

    #[test]
    fn t05_strand() {
        let doc = parse_gff3(&sample_gff3()).unwrap();
        assert_eq!(doc.records[0].strand, Strand::Plus);
    }

    #[test]
    fn t06_attributes_id() {
        let doc = parse_gff3(&sample_gff3()).unwrap();
        assert_eq!(doc.records[0].id(), Some("gene01"));
    }

    #[test]
    fn t07_attributes_name() {
        let doc = parse_gff3(&sample_gff3()).unwrap();
        assert_eq!(doc.records[0].attr("Name"), Some("BRCA1"));
    }

    #[test]
    fn t08_parent_attribute() {
        let doc = parse_gff3(&sample_gff3()).unwrap();
        assert_eq!(doc.records[1].parents(), vec!["gene01"]);
    }

    #[test]
    fn t09_children_of() {
        let doc = parse_gff3(&sample_gff3()).unwrap();
        let children = doc.children_of("mrna01");
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn t10_by_type() {
        let doc = parse_gff3(&sample_gff3()).unwrap();
        assert_eq!(doc.by_type("exon").len(), 2);
    }

    #[test]
    fn t11_overlapping() {
        let doc = parse_gff3(&sample_gff3()).unwrap();
        let hits = doc.overlapping("chr1", 1100, 1150);
        assert!(hits.len() >= 2); // gene + exon01
    }

    #[test]
    fn t12_contains() {
        let rec = GffRecord::new("chr1", "gene", 100, 500);
        assert!(rec.contains(200, 400));
        assert!(!rec.contains(50, 200));
    }

    #[test]
    fn t13_feature_filter() {
        let parser = GffParser::new(GffFormat::Gff3).with_feature_filter("exon");
        let doc = parser.parse(&sample_gff3()).unwrap();
        assert_eq!(doc.records.len(), 2);
    }

    #[test]
    fn t14_gtf_parsing() {
        let input = "chr1\tENSEMBL\texon\t100\t200\t.\t+\t.\tgene_id \"G1\"; transcript_id \"T1\";\n";
        let doc = parse_gtf(input).unwrap();
        assert_eq!(doc.records[0].gene_id(), Some("G1"));
        assert_eq!(doc.records[0].transcript_id(), Some("T1"));
    }

    #[test]
    fn t15_directives() {
        let doc = parse_gff3(&sample_gff3()).unwrap();
        assert_eq!(doc.directives.len(), 1);
        assert!(doc.directives[0].contains("gff-version"));
    }

    #[test]
    fn t16_empty_input() {
        assert_eq!(parse_gff3(""), Err(GffError::EmptyInput));
    }

    #[test]
    fn t17_invalid_strand() {
        let input = "chr1\t.\tgene\t1\t100\t.\tX\t.\t.\n";
        assert!(matches!(
            parse_gff3(input),
            Err(GffError::InvalidStrand { .. })
        ));
    }

    #[test]
    fn t18_by_seqid() {
        let doc = parse_gff3(&sample_gff3()).unwrap();
        assert_eq!(doc.by_seqid("chr1").len(), 4);
        assert_eq!(doc.by_seqid("chr2").len(), 0);
    }

    #[test]
    fn t19_display_record() {
        let rec = GffRecord::new("chr1", "gene", 100, 500).with_strand(Strand::Minus);
        let s = format!("{rec}");
        assert!(s.contains("chr1:100-500"));
        assert!(s.contains("-"));
    }

    #[test]
    fn t20_display_parser() {
        let p = GffParser::new(GffFormat::Gff3);
        let s = format!("{p}");
        assert!(s.contains("GFF3"));
    }
}
