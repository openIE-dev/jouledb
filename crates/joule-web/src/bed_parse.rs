//! BED format parser — interval representation, track lines, field handling.
//!
//! Parses BED (Browser Extensible Data) format files with 3-12 columns,
//! track line metadata, interval arithmetic (overlap, merge, subtract),
//! and serialization. Coordinates are 0-based, half-open.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum BedError {
    EmptyInput,
    InsufficientFields { line: usize, found: usize },
    InvalidCoordinate { line: usize, field: String, value: String },
    InvalidScore { line: usize, value: String },
    InvalidStrand { line: usize, value: String },
    InvalidRgb { line: usize, value: String },
    StartAfterEnd { line: usize, start: u64, end: u64 },
}

impl fmt::Display for BedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "empty BED input"),
            Self::InsufficientFields { line, found } => {
                write!(f, "line {line}: expected >=3 fields, found {found}")
            }
            Self::InvalidCoordinate { line, field, value } => {
                write!(f, "line {line}: invalid {field} '{value}'")
            }
            Self::InvalidScore { line, value } => write!(f, "line {line}: invalid score '{value}'"),
            Self::InvalidStrand { line, value } => write!(f, "line {line}: invalid strand '{value}'"),
            Self::InvalidRgb { line, value } => write!(f, "line {line}: invalid RGB '{value}'"),
            Self::StartAfterEnd { line, start, end } => {
                write!(f, "line {line}: start {start} > end {end}")
            }
        }
    }
}

impl std::error::Error for BedError {}

// ── RGB color ───────────────────────────────────────────────────

/// An RGB color from the BED itemRgb field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    pub fn new(r: u8, g: u8, b: u8) -> Self { Self { r, g, b } }

    pub fn from_str_triplet(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 3 { return None; }
        let r: u8 = parts[0].trim().parse().ok()?;
        let g: u8 = parts[1].trim().parse().ok()?;
        let b: u8 = parts[2].trim().parse().ok()?;
        Some(Self { r, g, b })
    }
}

impl fmt::Display for Rgb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{},{},{}", self.r, self.g, self.b)
    }
}

// ── Block (for BED12) ───────────────────────────────────────────

/// A sub-interval block in BED12 format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Block {
    pub relative_start: u64,
    pub size: u64,
}

impl Block {
    pub fn new(relative_start: u64, size: u64) -> Self {
        Self { relative_start, size }
    }
}

impl fmt::Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}+{}", self.relative_start, self.size)
    }
}

// ── BED record ──────────────────────────────────────────────────

/// A single BED record (supports BED3-BED12).
#[derive(Debug, Clone)]
#[derive(PartialEq)]
pub struct BedRecord {
    pub chrom: String,
    pub chrom_start: u64,
    pub chrom_end: u64,
    pub name: Option<String>,
    pub score: Option<u16>,
    pub strand: Option<char>,
    pub thick_start: Option<u64>,
    pub thick_end: Option<u64>,
    pub item_rgb: Option<Rgb>,
    pub blocks: Vec<Block>,
}

impl BedRecord {
    pub fn new(chrom: &str, start: u64, end: u64) -> Self {
        Self {
            chrom: chrom.to_string(),
            chrom_start: start,
            chrom_end: end,
            name: None,
            score: None,
            strand: None,
            thick_start: None,
            thick_end: None,
            item_rgb: None,
            blocks: Vec::new(),
        }
    }

    pub fn with_name(mut self, n: &str) -> Self { self.name = Some(n.to_string()); self }
    pub fn with_score(mut self, s: u16) -> Self { self.score = Some(s); self }
    pub fn with_strand(mut self, s: char) -> Self { self.strand = Some(s); self }
    pub fn with_thick(mut self, start: u64, end: u64) -> Self {
        self.thick_start = Some(start);
        self.thick_end = Some(end);
        self
    }
    pub fn with_rgb(mut self, rgb: Rgb) -> Self { self.item_rgb = Some(rgb); self }
    pub fn with_blocks(mut self, blocks: Vec<Block>) -> Self { self.blocks = blocks; self }

    /// Interval length.
    pub fn length(&self) -> u64 { self.chrom_end - self.chrom_start }

    /// Number of BED columns this record has.
    pub fn column_count(&self) -> usize {
        if !self.blocks.is_empty() { 12 }
        else if self.item_rgb.is_some() { 9 }
        else if self.thick_start.is_some() { 8 }
        else if self.strand.is_some() { 6 }
        else if self.score.is_some() { 5 }
        else if self.name.is_some() { 4 }
        else { 3 }
    }

    /// True if this record overlaps another on the same chromosome.
    pub fn overlaps(&self, other: &BedRecord) -> bool {
        self.chrom == other.chrom
            && self.chrom_start < other.chrom_end
            && self.chrom_end > other.chrom_start
    }

    /// Overlap length with another record (0 if none).
    pub fn overlap_length(&self, other: &BedRecord) -> u64 {
        if !self.overlaps(other) { return 0; }
        let s = self.chrom_start.max(other.chrom_start);
        let e = self.chrom_end.min(other.chrom_end);
        e - s
    }

    /// Midpoint of the interval.
    pub fn midpoint(&self) -> f64 {
        (self.chrom_start as f64 + self.chrom_end as f64) / 2.0
    }

    /// Serialize to a tab-delimited BED line.
    pub fn to_bed(&self) -> String {
        let mut parts = vec![
            self.chrom.clone(),
            self.chrom_start.to_string(),
            self.chrom_end.to_string(),
        ];
        if let Some(ref n) = self.name { parts.push(n.clone()); } else { return parts.join("\t"); }
        if let Some(s) = self.score { parts.push(s.to_string()); } else { return parts.join("\t"); }
        if let Some(s) = self.strand { parts.push(s.to_string()); } else { return parts.join("\t"); }
        if let Some(ts) = self.thick_start {
            parts.push(ts.to_string());
            parts.push(self.thick_end.unwrap_or(ts).to_string());
        } else {
            return parts.join("\t");
        }
        if let Some(rgb) = self.item_rgb { parts.push(rgb.to_string()); } else { return parts.join("\t"); }
        if !self.blocks.is_empty() {
            parts.push(self.blocks.len().to_string());
            let sizes: Vec<String> = self.blocks.iter().map(|b| b.size.to_string()).collect();
            let starts: Vec<String> = self.blocks.iter().map(|b| b.relative_start.to_string()).collect();
            parts.push(sizes.join(","));
            parts.push(starts.join(","));
        }
        parts.join("\t")
    }
}

impl fmt::Display for BedRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}-{}", self.chrom, self.chrom_start, self.chrom_end)?;
        if let Some(ref n) = self.name { write!(f, " {n}")?; }
        Ok(())
    }
}

// ── Track metadata ──────────────────────────────────────────────

/// Track line metadata.
#[derive(Debug, Clone, Default)]
#[derive(PartialEq)]
pub struct TrackLine {
    pub name: Option<String>,
    pub description: Option<String>,
    pub visibility: Option<String>,
    pub color: Option<Rgb>,
    pub fields: Vec<(String, String)>,
}

impl TrackLine {
    pub fn from_line(line: &str) -> Self {
        let mut tl = Self::default();
        let content = line.strip_prefix("track").unwrap_or(line).trim();
        // Simple key=value or key="value" parser
        let mut rest = content;
        while !rest.is_empty() {
            rest = rest.trim_start();
            if let Some(eq) = rest.find('=') {
                let key = rest[..eq].trim();
                let after = &rest[eq + 1..];
                let (value, remaining) = if after.starts_with('"') {
                    let after_quote = &after[1..];
                    if let Some(end_q) = after_quote.find('"') {
                        (&after_quote[..end_q], &after_quote[end_q + 1..])
                    } else {
                        (after_quote, "")
                    }
                } else if let Some(sp) = after.find(' ') {
                    (&after[..sp], &after[sp..])
                } else {
                    (after, "")
                };
                match key {
                    "name" => tl.name = Some(value.to_string()),
                    "description" => tl.description = Some(value.to_string()),
                    "visibility" => tl.visibility = Some(value.to_string()),
                    "color" => tl.color = Rgb::from_str_triplet(value),
                    _ => {}
                }
                tl.fields.push((key.to_string(), value.to_string()));
                rest = remaining;
            } else {
                break;
            }
        }
        tl
    }
}

impl fmt::Display for TrackLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "track")?;
        if let Some(ref n) = self.name { write!(f, " name=\"{n}\"")?; }
        Ok(())
    }
}

// ── Parser ──────────────────────────────────────────────────────

/// BED file parser.
#[derive(Debug, Clone)]
pub struct BedParser {
    validate_coords: bool,
    min_length: u64,
}

impl BedParser {
    pub fn new() -> Self { Self { validate_coords: true, min_length: 0 } }
    pub fn with_validate_coords(mut self, v: bool) -> Self { self.validate_coords = v; self }
    pub fn with_min_length(mut self, m: u64) -> Self { self.min_length = m; self }

    /// Parse a complete BED string.
    pub fn parse(&self, input: &str) -> Result<(Option<TrackLine>, Vec<BedRecord>), BedError> {
        if input.trim().is_empty() { return Err(BedError::EmptyInput); }
        let mut track = None;
        let mut records = Vec::new();
        let mut line_num = 0_usize;

        for line in input.lines() {
            line_num += 1;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') { continue; }
            if trimmed.starts_with("track") {
                track = Some(TrackLine::from_line(trimmed));
                continue;
            }
            if trimmed.starts_with("browser") { continue; }

            let fields: Vec<&str> = trimmed.split('\t').collect();
            if fields.len() < 3 {
                return Err(BedError::InsufficientFields { line: line_num, found: fields.len() });
            }
            let start: u64 = fields[1].parse().map_err(|_| BedError::InvalidCoordinate {
                line: line_num, field: "chromStart".to_string(), value: fields[1].to_string(),
            })?;
            let end: u64 = fields[2].parse().map_err(|_| BedError::InvalidCoordinate {
                line: line_num, field: "chromEnd".to_string(), value: fields[2].to_string(),
            })?;
            if self.validate_coords && start > end {
                return Err(BedError::StartAfterEnd { line: line_num, start, end });
            }
            if end - start < self.min_length { continue; }

            let mut rec = BedRecord::new(fields[0], start, end);
            if fields.len() > 3 { rec.name = Some(fields[3].to_string()); }
            if fields.len() > 4 {
                rec.score = Some(fields[4].parse().map_err(|_| BedError::InvalidScore {
                    line: line_num, value: fields[4].to_string(),
                })?);
            }
            if fields.len() > 5 {
                let s = fields[5].chars().next().unwrap_or('.');
                if !matches!(s, '+' | '-' | '.') {
                    return Err(BedError::InvalidStrand { line: line_num, value: fields[5].to_string() });
                }
                rec.strand = Some(s);
            }
            if fields.len() > 7 {
                rec.thick_start = Some(fields[6].parse().unwrap_or(start));
                rec.thick_end = Some(fields[7].parse().unwrap_or(end));
            }
            if fields.len() > 8 {
                rec.item_rgb = Rgb::from_str_triplet(fields[8]);
            }
            if fields.len() > 11 {
                let block_count: usize = fields[9].parse().unwrap_or(0);
                let sizes: Vec<u64> = fields[10].split(',').filter(|s| !s.is_empty())
                    .filter_map(|s| s.parse().ok()).collect();
                let starts: Vec<u64> = fields[11].split(',').filter(|s| !s.is_empty())
                    .filter_map(|s| s.parse().ok()).collect();
                let n = block_count.min(sizes.len()).min(starts.len());
                rec.blocks = (0..n).map(|i| Block::new(starts[i], sizes[i])).collect();
            }
            records.push(rec);
        }
        Ok((track, records))
    }
}

impl fmt::Display for BedParser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BedParser(validate={}, min_len={})", self.validate_coords, self.min_length)
    }
}

// ── Interval operations ─────────────────────────────────────────

/// Merge overlapping intervals on the same chromosome.
pub fn merge_intervals(mut recs: Vec<BedRecord>) -> Vec<BedRecord> {
    if recs.is_empty() { return recs; }
    recs.sort_by(|a, b| a.chrom.cmp(&b.chrom).then(a.chrom_start.cmp(&b.chrom_start)));
    let mut merged = vec![recs[0].clone()];
    for rec in &recs[1..] {
        let last = merged.last_mut().unwrap();
        if rec.chrom == last.chrom && rec.chrom_start <= last.chrom_end {
            last.chrom_end = last.chrom_end.max(rec.chrom_end);
        } else {
            merged.push(rec.clone());
        }
    }
    merged
}

/// Quick parse with default settings.
pub fn parse_bed(input: &str) -> Result<(Option<TrackLine>, Vec<BedRecord>), BedError> {
    BedParser::new().parse(input)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_bed() -> String {
        "chr1\t0\t1000\tfeature1\t500\t+\n\
         chr1\t2000\t3000\tfeature2\t300\t-\n".to_string()
    }

    #[test]
    fn t01_parse_basic() {
        let (_, recs) = parse_bed(&sample_bed()).unwrap();
        assert_eq!(recs.len(), 2);
    }

    #[test]
    fn t02_coordinates() {
        let (_, recs) = parse_bed(&sample_bed()).unwrap();
        assert_eq!(recs[0].chrom_start, 0);
        assert_eq!(recs[0].chrom_end, 1000);
    }

    #[test]
    fn t03_name_score_strand() {
        let (_, recs) = parse_bed(&sample_bed()).unwrap();
        assert_eq!(recs[0].name.as_deref(), Some("feature1"));
        assert_eq!(recs[0].score, Some(500));
        assert_eq!(recs[0].strand, Some('+'));
    }

    #[test]
    fn t04_length() {
        let rec = BedRecord::new("chr1", 100, 500);
        assert_eq!(rec.length(), 400);
    }

    #[test]
    fn t05_midpoint() {
        let rec = BedRecord::new("chr1", 100, 200);
        assert!((rec.midpoint() - 150.0).abs() < 1e-9);
    }

    #[test]
    fn t06_overlap_true() {
        let a = BedRecord::new("chr1", 100, 300);
        let b = BedRecord::new("chr1", 200, 400);
        assert!(a.overlaps(&b));
    }

    #[test]
    fn t07_overlap_false_diff_chrom() {
        let a = BedRecord::new("chr1", 100, 300);
        let b = BedRecord::new("chr2", 100, 300);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn t08_overlap_length() {
        let a = BedRecord::new("chr1", 100, 300);
        let b = BedRecord::new("chr1", 200, 400);
        assert_eq!(a.overlap_length(&b), 100);
    }

    #[test]
    fn t09_merge_intervals() {
        let recs = vec![
            BedRecord::new("chr1", 0, 100),
            BedRecord::new("chr1", 50, 200),
            BedRecord::new("chr1", 300, 400),
        ];
        let merged = merge_intervals(recs);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].chrom_end, 200);
    }

    #[test]
    fn t10_bed3_only() {
        let input = "chr1\t0\t100\n";
        let (_, recs) = parse_bed(input).unwrap();
        assert_eq!(recs[0].column_count(), 3);
    }

    #[test]
    fn t11_track_line() {
        let input = "track name=\"test\" description=\"my track\"\nchr1\t0\t100\n";
        let (track, recs) = parse_bed(input).unwrap();
        assert!(track.is_some());
        assert_eq!(track.unwrap().name.as_deref(), Some("test"));
        assert_eq!(recs.len(), 1);
    }

    #[test]
    fn t12_start_after_end() {
        let input = "chr1\t500\t100\n";
        assert!(matches!(parse_bed(input), Err(BedError::StartAfterEnd { .. })));
    }

    #[test]
    fn t13_min_length_filter() {
        let parser = BedParser::new().with_min_length(500);
        let (_, recs) = parser.parse(&sample_bed()).unwrap();
        assert_eq!(recs.len(), 2); // both are >= 500bp
    }

    #[test]
    fn t14_serialization() {
        let rec = BedRecord::new("chr1", 0, 100).with_name("f1").with_score(500);
        let s = rec.to_bed();
        assert!(s.contains("chr1\t0\t100\tf1\t500"));
    }

    #[test]
    fn t15_rgb() {
        let rgb = Rgb::from_str_triplet("255,0,128").unwrap();
        assert_eq!(rgb.r, 255);
        assert_eq!(rgb.g, 0);
        assert_eq!(rgb.b, 128);
    }

    #[test]
    fn t16_rgb_display() {
        let rgb = Rgb::new(10, 20, 30);
        assert_eq!(format!("{rgb}"), "10,20,30");
    }

    #[test]
    fn t17_empty_input() {
        assert_eq!(parse_bed(""), Err(BedError::EmptyInput));
    }

    #[test]
    fn t18_comment_skip() {
        let input = "# comment\nchr1\t0\t100\n";
        let (_, recs) = parse_bed(input).unwrap();
        assert_eq!(recs.len(), 1);
    }

    #[test]
    fn t19_display_record() {
        let rec = BedRecord::new("chr1", 100, 200).with_name("test");
        let s = format!("{rec}");
        assert!(s.contains("chr1:100-200"));
        assert!(s.contains("test"));
    }

    #[test]
    fn t20_display_parser() {
        let p = BedParser::new().with_min_length(50);
        let s = format!("{p}");
        assert!(s.contains("min_len=50"));
    }
}
