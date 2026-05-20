//! FASTQ format parser — quality scores, Phred encoding, paired-end support.
//!
//! Handles standard FASTQ four-line records (`@header`, sequence, `+`,
//! quality), Phred+33 / Phred+64 quality decoding, per-base and per-read
//! statistics, paired-end file validation, and quality trimming.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum FastqError {
    EmptyInput,
    MissingHeader(usize),
    MissingSeparator(usize),
    LengthMismatch { record: usize, seq_len: usize, qual_len: usize },
    InvalidQuality { record: usize, position: usize, ch: char },
    PairedEndMismatch { r1_count: usize, r2_count: usize },
    MalformedRecord(String),
}

impl fmt::Display for FastqError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "empty FASTQ input"),
            Self::MissingHeader(n) => write!(f, "missing '@' header at record {n}"),
            Self::MissingSeparator(n) => write!(f, "missing '+' separator at record {n}"),
            Self::LengthMismatch { record, seq_len, qual_len } => {
                write!(f, "record {record}: sequence length {seq_len} != quality length {qual_len}")
            }
            Self::InvalidQuality { record, position, ch } => {
                write!(f, "record {record}: invalid quality char '{ch}' at position {position}")
            }
            Self::PairedEndMismatch { r1_count, r2_count } => {
                write!(f, "paired-end mismatch: R1 has {r1_count}, R2 has {r2_count} records")
            }
            Self::MalformedRecord(s) => write!(f, "malformed record: {s}"),
        }
    }
}

impl std::error::Error for FastqError {}

// ── Quality encoding ────────────────────────────────────────────

/// Phred quality score encoding scheme.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QualityEncoding {
    /// Sanger / Illumina 1.8+ (ASCII 33 offset).
    Phred33,
    /// Old Illumina / Solexa (ASCII 64 offset).
    Phred64,
}

impl QualityEncoding {
    /// ASCII offset for this encoding.
    pub fn offset(self) -> u8 {
        match self {
            Self::Phred33 => 33,
            Self::Phred64 => 64,
        }
    }

    /// Decode a single quality character to a Phred score.
    pub fn decode(self, ch: char) -> Option<u8> {
        let val = ch as u8;
        if val < self.offset() {
            return None;
        }
        Some(val - self.offset())
    }

    /// Encode a Phred score to a quality character.
    pub fn encode(self, score: u8) -> char {
        (score + self.offset()) as char
    }

    /// Decode an entire quality string to scores.
    pub fn decode_all(self, quals: &str) -> Option<Vec<u8>> {
        quals.chars().map(|c| self.decode(c)).collect()
    }

    /// Error probability from Phred score: 10^(-Q/10).
    pub fn error_probability(score: u8) -> f64 {
        10.0_f64.powf(-(score as f64) / 10.0)
    }
}

impl fmt::Display for QualityEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Phred33 => write!(f, "Phred+33"),
            Self::Phred64 => write!(f, "Phred+64"),
        }
    }
}

// ── FASTQ record ────────────────────────────────────────────────

/// A single FASTQ record.
#[derive(Debug, Clone, PartialEq)]
pub struct FastqRecord {
    pub id: String,
    pub description: String,
    pub sequence: String,
    pub quality: String,
}

impl FastqRecord {
    pub fn new(id: &str, sequence: &str, quality: &str) -> Self {
        Self {
            id: id.to_string(),
            description: String::new(),
            sequence: sequence.to_string(),
            quality: quality.to_string(),
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    /// Decode quality scores with the given encoding.
    pub fn quality_scores(&self, enc: QualityEncoding) -> Option<Vec<u8>> {
        enc.decode_all(&self.quality)
    }

    /// Mean quality score.
    pub fn mean_quality(&self, enc: QualityEncoding) -> f64 {
        let scores = match self.quality_scores(enc) {
            Some(s) => s,
            None => return 0.0,
        };
        if scores.is_empty() {
            return 0.0;
        }
        scores.iter().map(|s| *s as f64).sum::<f64>() / scores.len() as f64
    }

    /// Minimum quality score.
    pub fn min_quality(&self, enc: QualityEncoding) -> Option<u8> {
        self.quality_scores(enc).and_then(|s| s.into_iter().min())
    }

    /// Trim bases from the 3' end where quality falls below `threshold`.
    pub fn quality_trim(&self, enc: QualityEncoding, threshold: u8) -> Self {
        let scores = match self.quality_scores(enc) {
            Some(s) => s,
            None => return self.clone(),
        };
        let mut end = scores.len();
        while end > 0 && scores[end - 1] < threshold {
            end -= 1;
        }
        Self {
            id: self.id.clone(),
            description: self.description.clone(),
            sequence: self.sequence[..end].to_string(),
            quality: self.quality[..end].to_string(),
        }
    }

    /// Serialize to four-line FASTQ format.
    pub fn to_fastq(&self) -> String {
        let mut out = String::new();
        out.push('@');
        out.push_str(&self.id);
        if !self.description.is_empty() {
            out.push(' ');
            out.push_str(&self.description);
        }
        out.push('\n');
        out.push_str(&self.sequence);
        out.push_str("\n+\n");
        out.push_str(&self.quality);
        out.push('\n');
        out
    }
}

impl fmt::Display for FastqRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "@{} ({}bp, mean_q={:.1})", self.id, self.len(),
               self.mean_quality(QualityEncoding::Phred33))
    }
}

// ── Parser ──────────────────────────────────────────────────────

/// FASTQ parser with configurable validation.
#[derive(Debug, Clone)]
pub struct FastqParser {
    encoding: QualityEncoding,
    validate_quality: bool,
    min_length: usize,
}

impl FastqParser {
    pub fn new() -> Self {
        Self {
            encoding: QualityEncoding::Phred33,
            validate_quality: true,
            min_length: 0,
        }
    }

    pub fn with_encoding(mut self, enc: QualityEncoding) -> Self {
        self.encoding = enc;
        self
    }

    pub fn with_validate_quality(mut self, v: bool) -> Self {
        self.validate_quality = v;
        self
    }

    pub fn with_min_length(mut self, m: usize) -> Self {
        self.min_length = m;
        self
    }

    /// Parse a complete FASTQ string.
    pub fn parse(&self, input: &str) -> Result<Vec<FastqRecord>, FastqError> {
        if input.trim().is_empty() {
            return Err(FastqError::EmptyInput);
        }
        let lines: Vec<&str> = input.lines().collect();
        let mut records = Vec::new();
        let mut i = 0;
        let mut rec_num = 0_usize;

        while i < lines.len() {
            // skip empty lines between records
            if lines[i].trim().is_empty() {
                i += 1;
                continue;
            }
            if i + 3 >= lines.len() {
                return Err(FastqError::MalformedRecord(
                    format!("incomplete record at line {}", i + 1),
                ));
            }
            // Line 1: header
            let header = lines[i].trim();
            if !header.starts_with('@') {
                return Err(FastqError::MissingHeader(rec_num));
            }
            let hdr = &header[1..];
            let (id, desc) = if let Some(pos) = hdr.find(|c: char| c.is_whitespace()) {
                (hdr[..pos].to_string(), hdr[pos..].trim().to_string())
            } else {
                (hdr.to_string(), String::new())
            };
            // Line 2: sequence
            let seq = lines[i + 1].trim().to_string();
            // Line 3: separator
            let sep = lines[i + 2].trim();
            if !sep.starts_with('+') {
                return Err(FastqError::MissingSeparator(rec_num));
            }
            // Line 4: quality
            let qual = lines[i + 3].trim().to_string();
            if seq.len() != qual.len() {
                return Err(FastqError::LengthMismatch {
                    record: rec_num,
                    seq_len: seq.len(),
                    qual_len: qual.len(),
                });
            }
            if self.validate_quality {
                for (pos, ch) in qual.chars().enumerate() {
                    if self.encoding.decode(ch).is_none() {
                        return Err(FastqError::InvalidQuality {
                            record: rec_num,
                            position: pos,
                            ch,
                        });
                    }
                }
            }
            if seq.len() >= self.min_length {
                records.push(FastqRecord {
                    id,
                    description: desc,
                    sequence: seq,
                    quality: qual,
                });
            }
            rec_num += 1;
            i += 4;
        }
        Ok(records)
    }
}

impl fmt::Display for FastqParser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FastqParser(encoding={}, min_len={})", self.encoding, self.min_length)
    }
}

// ── Paired-end helpers ──────────────────────────────────────────

/// Validate that two FASTQ record sets form a valid paired-end pair.
pub fn validate_paired(r1: &[FastqRecord], r2: &[FastqRecord]) -> Result<(), FastqError> {
    if r1.len() != r2.len() {
        return Err(FastqError::PairedEndMismatch {
            r1_count: r1.len(),
            r2_count: r2.len(),
        });
    }
    Ok(())
}

/// Interleave paired-end records (R1_1, R2_1, R1_2, R2_2, ...).
pub fn interleave_paired(r1: &[FastqRecord], r2: &[FastqRecord]) -> Vec<FastqRecord> {
    let mut out = Vec::with_capacity(r1.len() + r2.len());
    for (a, b) in r1.iter().zip(r2.iter()) {
        out.push(a.clone());
        out.push(b.clone());
    }
    out
}

// ── Convenience ─────────────────────────────────────────────────

/// Quick parse with default Phred+33 settings.
pub fn parse_fastq(input: &str) -> Result<Vec<FastqRecord>, FastqError> {
    FastqParser::new().parse(input)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "@read1 desc\nACGTACGT\n+\nIIIIIIII\n";

    #[test]
    fn t01_single_record() {
        let recs = parse_fastq(SAMPLE).unwrap();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].id, "read1");
        assert_eq!(recs[0].sequence, "ACGTACGT");
    }

    #[test]
    fn t02_quality_decode_phred33() {
        let scores = QualityEncoding::Phred33.decode_all("IIIIIIII").unwrap();
        assert!(scores.iter().all(|s| *s == 40));
    }

    #[test]
    fn t03_quality_decode_phred64() {
        let score = QualityEncoding::Phred64.decode('h').unwrap();
        assert_eq!(score, 40);
    }

    #[test]
    fn t04_encode_roundtrip() {
        let enc = QualityEncoding::Phred33;
        for q in 0..42 {
            let ch = enc.encode(q);
            assert_eq!(enc.decode(ch).unwrap(), q);
        }
    }

    #[test]
    fn t05_error_probability() {
        let p = QualityEncoding::error_probability(20);
        assert!((p - 0.01).abs() < 1e-9);
    }

    #[test]
    fn t06_mean_quality() {
        let rec = FastqRecord::new("r", "ACGT", "IIII");
        assert!((rec.mean_quality(QualityEncoding::Phred33) - 40.0).abs() < 1e-9);
    }

    #[test]
    fn t07_min_quality() {
        let rec = FastqRecord::new("r", "ACGT", "!5II");
        assert_eq!(rec.min_quality(QualityEncoding::Phred33), Some(0));
    }

    #[test]
    fn t08_quality_trim() {
        // '!' = Q0, 'I' = Q40
        let rec = FastqRecord::new("r", "ACGTNN", "IIII!!");
        let trimmed = rec.quality_trim(QualityEncoding::Phred33, 20);
        assert_eq!(trimmed.sequence, "ACGT");
        assert_eq!(trimmed.quality, "IIII");
    }

    #[test]
    fn t09_empty_input() {
        assert_eq!(parse_fastq(""), Err(FastqError::EmptyInput));
    }

    #[test]
    fn t10_missing_header() {
        let input = "ACGT\n+\nIIII\nXXXX\n";
        assert!(matches!(parse_fastq(input), Err(FastqError::MissingHeader(_))));
    }

    #[test]
    fn t11_length_mismatch() {
        let input = "@r\nACGT\n+\nIII\n";
        assert!(matches!(parse_fastq(input), Err(FastqError::LengthMismatch { .. })));
    }

    #[test]
    fn t12_multi_record() {
        let input = "@a\nAA\n+\nII\n@b\nCC\n+\nII\n";
        let recs = parse_fastq(input).unwrap();
        assert_eq!(recs.len(), 2);
    }

    #[test]
    fn t13_min_length_filter() {
        let parser = FastqParser::new().with_min_length(5);
        let input = "@a\nAC\n+\nII\n@b\nACGTACGT\n+\nIIIIIIII\n";
        let recs = parser.parse(input).unwrap();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].id, "b");
    }

    #[test]
    fn t14_paired_validate_ok() {
        let r1 = vec![FastqRecord::new("a/1", "AC", "II")];
        let r2 = vec![FastqRecord::new("a/2", "GT", "II")];
        assert!(validate_paired(&r1, &r2).is_ok());
    }

    #[test]
    fn t15_paired_mismatch() {
        let r1 = vec![FastqRecord::new("a/1", "AC", "II")];
        let r2 = Vec::new();
        assert!(validate_paired(&r1, &r2).is_err());
    }

    #[test]
    fn t16_interleave() {
        let r1 = vec![FastqRecord::new("a/1", "AA", "II")];
        let r2 = vec![FastqRecord::new("a/2", "CC", "II")];
        let il = interleave_paired(&r1, &r2);
        assert_eq!(il.len(), 2);
        assert_eq!(il[0].id, "a/1");
        assert_eq!(il[1].id, "a/2");
    }

    #[test]
    fn t17_serialization_roundtrip() {
        let rec = FastqRecord::new("x", "ACGT", "IIII").with_description("test");
        let s = rec.to_fastq();
        let parsed = parse_fastq(&s).unwrap();
        assert_eq!(parsed[0].sequence, "ACGT");
    }

    #[test]
    fn t18_display_record() {
        let rec = FastqRecord::new("r1", "ACGT", "IIII");
        let s = format!("{rec}");
        assert!(s.contains("r1"));
        assert!(s.contains("4bp"));
    }

    #[test]
    fn t19_display_parser() {
        let p = FastqParser::new().with_encoding(QualityEncoding::Phred64);
        let s = format!("{p}");
        assert!(s.contains("Phred+64"));
    }

    #[test]
    fn t20_missing_separator() {
        let input = "@r\nACGT\nNOSEP\nIIII\n";
        assert!(matches!(parse_fastq(input), Err(FastqError::MissingSeparator(_))));
    }
}
