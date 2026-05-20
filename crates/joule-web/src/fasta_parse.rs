//! FASTA format parser — multi-sequence, header parsing, sequence validation.
//!
//! Parses standard FASTA files with `>` header lines and wrapped sequence
//! data. Supports DNA, RNA, and protein alphabets, multi-record iteration,
//! header field extraction, and round-trip serialization.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum FastaError {
    EmptyInput,
    MissingHeader,
    InvalidCharacter { record: String, position: usize, ch: char },
    DuplicateId(String),
    MalformedHeader(String),
}

impl fmt::Display for FastaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "empty FASTA input"),
            Self::MissingHeader => write!(f, "first non-empty line must start with '>'"),
            Self::InvalidCharacter { record, position, ch } => {
                write!(f, "invalid character '{ch}' at position {position} in record '{record}'")
            }
            Self::DuplicateId(id) => write!(f, "duplicate sequence id: {id}"),
            Self::MalformedHeader(h) => write!(f, "malformed header: {h}"),
        }
    }
}

impl std::error::Error for FastaError {}

// ── Alphabet ────────────────────────────────────────────────────

/// Sequence alphabet for validation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Alphabet {
    Dna,
    Rna,
    Protein,
    Any,
}

impl Alphabet {
    /// Returns true if `ch` is valid for this alphabet.
    pub fn is_valid(self, ch: char) -> bool {
        let upper = ch.to_ascii_uppercase();
        match self {
            Self::Dna => matches!(upper, 'A' | 'C' | 'G' | 'T' | 'N' | '-' | '.'),
            Self::Rna => matches!(upper, 'A' | 'C' | 'G' | 'U' | 'N' | '-' | '.'),
            Self::Protein => {
                matches!(upper,
                    'A' | 'R' | 'N' | 'D' | 'C' | 'E' | 'Q' | 'G' | 'H' | 'I'
                    | 'L' | 'K' | 'M' | 'F' | 'P' | 'S' | 'T' | 'W' | 'Y' | 'V'
                    | 'B' | 'Z' | 'X' | '*' | '-' | '.')
            }
            Self::Any => true,
        }
    }
}

impl fmt::Display for Alphabet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dna => write!(f, "DNA"),
            Self::Rna => write!(f, "RNA"),
            Self::Protein => write!(f, "Protein"),
            Self::Any => write!(f, "Any"),
        }
    }
}

// ── FASTA record ────────────────────────────────────────────────

/// A single FASTA sequence record.
#[derive(Debug, Clone, PartialEq)]
pub struct FastaRecord {
    pub id: String,
    pub description: String,
    pub sequence: String,
}

impl FastaRecord {
    pub fn new(id: &str, sequence: &str) -> Self {
        Self {
            id: id.to_string(),
            description: String::new(),
            sequence: sequence.to_string(),
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Sequence length excluding gaps.
    pub fn ungapped_len(&self) -> usize {
        self.sequence.chars().filter(|c| *c != '-' && *c != '.').count()
    }

    /// Total sequence length including gaps.
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    /// True if the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    /// GC content for nucleotide sequences (fraction of G+C).
    pub fn gc_content(&self) -> f64 {
        let total = self.ungapped_len() as f64;
        if total == 0.0 {
            return 0.0;
        }
        let gc = self
            .sequence
            .chars()
            .filter(|c| {
                let u = c.to_ascii_uppercase();
                u == 'G' || u == 'C'
            })
            .count() as f64;
        gc / total
    }

    /// Reverse complement (DNA).
    pub fn reverse_complement(&self) -> String {
        self.sequence
            .chars()
            .rev()
            .map(|c| match c.to_ascii_uppercase() {
                'A' => 'T',
                'T' => 'A',
                'C' => 'G',
                'G' => 'C',
                'N' => 'N',
                other => other,
            })
            .collect()
    }

    /// Validate every character against an alphabet.
    pub fn validate(&self, alphabet: Alphabet) -> Result<(), FastaError> {
        for (i, ch) in self.sequence.chars().enumerate() {
            if !alphabet.is_valid(ch) {
                return Err(FastaError::InvalidCharacter {
                    record: self.id.clone(),
                    position: i,
                    ch,
                });
            }
        }
        Ok(())
    }

    /// Format as FASTA with the given line width.
    pub fn to_fasta(&self, line_width: usize) -> String {
        let mut out = String::new();
        out.push('>');
        out.push_str(&self.id);
        if !self.description.is_empty() {
            out.push(' ');
            out.push_str(&self.description);
        }
        out.push('\n');
        let width = if line_width == 0 { 80 } else { line_width };
        let seq = self.sequence.as_bytes();
        let mut offset = 0;
        while offset < seq.len() {
            let end = (offset + width).min(seq.len());
            // SAFETY: FASTA sequences are ASCII
            out.push_str(&self.sequence[offset..end]);
            out.push('\n');
            offset = end;
        }
        out
    }
}

impl fmt::Display for FastaRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, ">{}", self.id)?;
        if !self.description.is_empty() {
            write!(f, " {}", self.description)?;
        }
        write!(f, " ({}bp)", self.sequence.len())
    }
}

// ── Parser configuration ────────────────────────────────────────

/// Builder for FASTA parsing options.
#[derive(Debug, Clone)]
pub struct FastaParser {
    alphabet: Alphabet,
    uppercase: bool,
    reject_duplicates: bool,
}

impl FastaParser {
    pub fn new() -> Self {
        Self {
            alphabet: Alphabet::Any,
            uppercase: false,
            reject_duplicates: false,
        }
    }

    pub fn with_alphabet(mut self, a: Alphabet) -> Self {
        self.alphabet = a;
        self
    }

    pub fn with_uppercase(mut self, u: bool) -> Self {
        self.uppercase = u;
        self
    }

    pub fn with_reject_duplicates(mut self, r: bool) -> Self {
        self.reject_duplicates = r;
        self
    }

    /// Parse a complete FASTA string into records.
    pub fn parse(&self, input: &str) -> Result<Vec<FastaRecord>, FastaError> {
        if input.trim().is_empty() {
            return Err(FastaError::EmptyInput);
        }
        let mut records = Vec::new();
        let mut seen: HashMap<String, bool> = HashMap::new();
        let mut current_id: Option<String> = None;
        let mut current_desc = String::new();
        let mut current_seq = String::new();
        let mut first_non_empty = true;

        for line in input.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed.starts_with('>') {
                if first_non_empty {
                    first_non_empty = false;
                } else if current_id.is_none() {
                    return Err(FastaError::MissingHeader);
                }
                // Flush previous record
                if let Some(ref id) = current_id {
                    let seq = if self.uppercase {
                        current_seq.to_ascii_uppercase()
                    } else {
                        current_seq.clone()
                    };
                    let rec = FastaRecord {
                        id: id.clone(),
                        description: current_desc.clone(),
                        sequence: seq,
                    };
                    rec.validate(self.alphabet)?;
                    records.push(rec);
                }
                // Parse header
                let header = &trimmed[1..];
                let (id, desc) = parse_header(header)?;
                if self.reject_duplicates {
                    if seen.contains_key(&id) {
                        return Err(FastaError::DuplicateId(id));
                    }
                    seen.insert(id.clone(), true);
                }
                current_id = Some(id);
                current_desc = desc;
                current_seq.clear();
            } else {
                if first_non_empty {
                    return Err(FastaError::MissingHeader);
                }
                current_seq.push_str(trimmed);
            }
        }
        // Flush last
        if let Some(id) = current_id {
            let seq = if self.uppercase {
                current_seq.to_ascii_uppercase()
            } else {
                current_seq
            };
            let rec = FastaRecord {
                id,
                description: current_desc,
                sequence: seq,
            };
            rec.validate(self.alphabet)?;
            records.push(rec);
        }
        Ok(records)
    }
}

impl fmt::Display for FastaParser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FastaParser(alphabet={}, uppercase={}, reject_dups={})",
            self.alphabet, self.uppercase, self.reject_duplicates
        )
    }
}

// ── Header parsing helper ───────────────────────────────────────

fn parse_header(header: &str) -> Result<(String, String), FastaError> {
    let trimmed = header.trim();
    if trimmed.is_empty() {
        return Err(FastaError::MalformedHeader(">".to_string()));
    }
    if let Some(pos) = trimmed.find(|c: char| c.is_whitespace()) {
        let id = trimmed[..pos].to_string();
        let desc = trimmed[pos..].trim().to_string();
        Ok((id, desc))
    } else {
        Ok((trimmed.to_string(), String::new()))
    }
}

// ── Convenience ─────────────────────────────────────────────────

/// Quick parse with default settings.
pub fn parse_fasta(input: &str) -> Result<Vec<FastaRecord>, FastaError> {
    FastaParser::new().parse(input)
}

/// Serialize records to a FASTA string.
pub fn write_fasta(records: &[FastaRecord], line_width: usize) -> String {
    let mut out = String::new();
    for rec in records {
        out.push_str(&rec.to_fasta(line_width));
    }
    out
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t01_single_record() {
        let input = ">seq1 test\nACGTACGT\n";
        let recs = parse_fasta(input).unwrap();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].id, "seq1");
        assert_eq!(recs[0].description, "test");
        assert_eq!(recs[0].sequence, "ACGTACGT");
    }

    #[test]
    fn t02_multi_record() {
        let input = ">a\nACGT\n>b\nTGCA\n";
        let recs = parse_fasta(input).unwrap();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[1].id, "b");
    }

    #[test]
    fn t03_wrapped_sequence() {
        let input = ">x\nACGT\nTGCA\nAAAA\n";
        let recs = parse_fasta(input).unwrap();
        assert_eq!(recs[0].sequence, "ACGTTGCAAAAA");
    }

    #[test]
    fn t04_empty_input() {
        assert_eq!(parse_fasta(""), Err(FastaError::EmptyInput));
    }

    #[test]
    fn t05_missing_header() {
        assert_eq!(parse_fasta("ACGT\n"), Err(FastaError::MissingHeader));
    }

    #[test]
    fn t06_dna_validation() {
        let parser = FastaParser::new().with_alphabet(Alphabet::Dna);
        let input = ">s1\nACGTN\n";
        assert!(parser.parse(input).is_ok());
    }

    #[test]
    fn t07_dna_invalid_char() {
        let parser = FastaParser::new().with_alphabet(Alphabet::Dna);
        let input = ">s1\nACGU\n";
        assert!(parser.parse(input).is_err());
    }

    #[test]
    fn t08_rna_alphabet() {
        let parser = FastaParser::new().with_alphabet(Alphabet::Rna);
        let input = ">r1\nAUGCN\n";
        assert!(parser.parse(input).is_ok());
    }

    #[test]
    fn t09_protein_alphabet() {
        let parser = FastaParser::new().with_alphabet(Alphabet::Protein);
        let input = ">p1\nMKWVTFISLLFLFSSAYS\n";
        assert!(parser.parse(input).is_ok());
    }

    #[test]
    fn t10_uppercase_option() {
        let parser = FastaParser::new().with_uppercase(true);
        let input = ">s\nacgt\n";
        let recs = parser.parse(input).unwrap();
        assert_eq!(recs[0].sequence, "ACGT");
    }

    #[test]
    fn t11_gc_content() {
        let rec = FastaRecord::new("test", "AACCGGTT");
        assert!((rec.gc_content() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn t12_gc_content_empty() {
        let rec = FastaRecord::new("e", "");
        assert!((rec.gc_content() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn t13_reverse_complement() {
        let rec = FastaRecord::new("rc", "ACGT");
        assert_eq!(rec.reverse_complement(), "ACGT");
    }

    #[test]
    fn t14_reverse_complement_asym() {
        let rec = FastaRecord::new("rc", "AACG");
        assert_eq!(rec.reverse_complement(), "CGTT");
    }

    #[test]
    fn t15_ungapped_len() {
        let rec = FastaRecord::new("g", "AC-GT.A");
        assert_eq!(rec.ungapped_len(), 5);
    }

    #[test]
    fn t16_duplicate_rejection() {
        let parser = FastaParser::new().with_reject_duplicates(true);
        let input = ">dup\nACGT\n>dup\nTGCA\n";
        assert_eq!(parser.parse(input), Err(FastaError::DuplicateId("dup".to_string())));
    }

    #[test]
    fn t17_roundtrip_serialization() {
        let rec = FastaRecord::new("s1", "ACGTACGTACGTACGT")
            .with_description("round trip");
        let serialized = rec.to_fasta(8);
        let parsed = parse_fasta(&serialized).unwrap();
        assert_eq!(parsed[0].sequence, rec.sequence);
    }

    #[test]
    fn t18_write_multi() {
        let recs = vec![
            FastaRecord::new("a", "AAAA"),
            FastaRecord::new("b", "CCCC"),
        ];
        let out = write_fasta(&recs, 80);
        assert!(out.contains(">a\n"));
        assert!(out.contains(">b\n"));
    }

    #[test]
    fn t19_display_record() {
        let rec = FastaRecord::new("abc", "ACGT");
        let s = format!("{rec}");
        assert!(s.contains("abc"));
        assert!(s.contains("4bp"));
    }

    #[test]
    fn t20_display_parser() {
        let p = FastaParser::new().with_alphabet(Alphabet::Dna);
        let s = format!("{p}");
        assert!(s.contains("DNA"));
    }
}
