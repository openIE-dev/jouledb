//! VCF variant format parser — genotype fields, INFO/FORMAT parsing.
//!
//! Parses Variant Call Format (VCF) files including header meta-lines,
//! INFO and FORMAT field definitions, variant records, genotype decoding,
//! allele frequency computation, and filtering support.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum VcfError {
    EmptyInput,
    MissingColumnHeader,
    InsufficientFields { line: usize, found: usize },
    InvalidPos { line: usize, value: String },
    InvalidQual { line: usize, value: String },
    InvalidGenotype(String),
    MalformedMeta(String),
}

impl fmt::Display for VcfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "empty VCF input"),
            Self::MissingColumnHeader => write!(f, "missing #CHROM column header"),
            Self::InsufficientFields { line, found } => {
                write!(f, "line {line}: expected >=8 fields, found {found}")
            }
            Self::InvalidPos { line, value } => write!(f, "line {line}: invalid POS '{value}'"),
            Self::InvalidQual { line, value } => write!(f, "line {line}: invalid QUAL '{value}'"),
            Self::InvalidGenotype(s) => write!(f, "invalid genotype: {s}"),
            Self::MalformedMeta(s) => write!(f, "malformed meta: {s}"),
        }
    }
}

impl std::error::Error for VcfError {}

// ── Variant type ────────────────────────────────────────────────

/// Classification of a variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariantType {
    Snp,
    Insertion,
    Deletion,
    Mnp,
    Complex,
    Structural,
    Reference,
}

impl fmt::Display for VariantType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Snp => write!(f, "SNP"),
            Self::Insertion => write!(f, "INS"),
            Self::Deletion => write!(f, "DEL"),
            Self::Mnp => write!(f, "MNP"),
            Self::Complex => write!(f, "COMPLEX"),
            Self::Structural => write!(f, "SV"),
            Self::Reference => write!(f, "REF"),
        }
    }
}

// ── Genotype ────────────────────────────────────────────────────

/// A parsed genotype for one sample.
#[derive(Debug, Clone, PartialEq)]
pub struct Genotype {
    pub alleles: Vec<Option<usize>>,
    pub phased: bool,
    pub fields: HashMap<String, String>,
}

impl Genotype {
    /// Parse a GT field like "0/1" or "0|1".
    pub fn from_gt(gt: &str) -> Result<Self, VcfError> {
        if gt == "." || gt == "./." || gt == ".|." {
            return Ok(Self { alleles: vec![None, None], phased: false, fields: HashMap::new() });
        }
        let phased = gt.contains('|');
        let sep = if phased { '|' } else { '/' };
        let alleles: Result<Vec<Option<usize>>, _> = gt
            .split(sep)
            .map(|a| {
                if a == "." {
                    Ok(None)
                } else {
                    a.parse::<usize>().map(Some).map_err(|_| VcfError::InvalidGenotype(gt.to_string()))
                }
            })
            .collect();
        Ok(Self { alleles: alleles?, phased, fields: HashMap::new() })
    }

    pub fn with_field(mut self, key: &str, value: &str) -> Self {
        self.fields.insert(key.to_string(), value.to_string());
        self
    }

    /// True if heterozygous.
    pub fn is_het(&self) -> bool {
        let a: Vec<_> = self.alleles.iter().flatten().collect();
        a.len() >= 2 && a[0] != a[1]
    }

    /// True if homozygous alternate.
    pub fn is_hom_alt(&self) -> bool {
        let a: Vec<_> = self.alleles.iter().flatten().collect();
        a.len() >= 2 && a[0] == a[1] && *a[0] > 0
    }

    /// True if homozygous reference.
    pub fn is_hom_ref(&self) -> bool {
        let a: Vec<_> = self.alleles.iter().flatten().collect();
        a.len() >= 2 && a[0] == a[1] && *a[0] == 0
    }

    /// Ploidy.
    pub fn ploidy(&self) -> usize { self.alleles.len() }
}

impl fmt::Display for Genotype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sep = if self.phased { "|" } else { "/" };
        let parts: Vec<String> = self.alleles.iter().map(|a| {
            match a {
                Some(v) => v.to_string(),
                None => ".".to_string(),
            }
        }).collect();
        write!(f, "{}", parts.join(sep))
    }
}

// ── VCF header ──────────────────────────────────────────────────

/// VCF file header metadata.
#[derive(Debug, Clone, Default)]
#[derive(PartialEq)]
pub struct VcfHeader {
    pub file_format: String,
    pub info_fields: HashMap<String, VcfFieldDef>,
    pub format_fields: HashMap<String, VcfFieldDef>,
    pub filter_fields: HashMap<String, String>,
    pub contigs: Vec<HashMap<String, String>>,
    pub samples: Vec<String>,
}

/// Definition of an INFO or FORMAT field.
#[derive(Debug, Clone, PartialEq)]
pub struct VcfFieldDef {
    pub id: String,
    pub number: String,
    pub field_type: String,
    pub description: String,
}

impl fmt::Display for VcfHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VcfHeader(format={}, samples={}, info={}, fmt={})",
               self.file_format, self.samples.len(),
               self.info_fields.len(), self.format_fields.len())
    }
}

// ── VCF record ──────────────────────────────────────────────────

/// A single VCF variant record.
#[derive(Debug, Clone)]
#[derive(PartialEq)]
pub struct VcfRecord {
    pub chrom: String,
    pub pos: u64,
    pub id: String,
    pub reference: String,
    pub alt: Vec<String>,
    pub qual: Option<f64>,
    pub filter: Vec<String>,
    pub info: HashMap<String, String>,
    pub format: Vec<String>,
    pub genotypes: Vec<Genotype>,
}

impl VcfRecord {
    /// Classify the variant type.
    pub fn variant_type(&self) -> VariantType {
        if self.alt.is_empty() || (self.alt.len() == 1 && self.alt[0] == ".") {
            return VariantType::Reference;
        }
        let alt0 = &self.alt[0];
        if alt0.starts_with('<') {
            return VariantType::Structural;
        }
        let ref_len = self.reference.len();
        let alt_len = alt0.len();
        if ref_len == 1 && alt_len == 1 {
            VariantType::Snp
        } else if ref_len == alt_len {
            VariantType::Mnp
        } else if ref_len < alt_len {
            VariantType::Insertion
        } else if ref_len > alt_len {
            VariantType::Deletion
        } else {
            VariantType::Complex
        }
    }

    /// Allele frequency of the first ALT across all genotypes.
    pub fn allele_frequency(&self) -> f64 {
        let mut alt_count = 0_usize;
        let mut total = 0_usize;
        for gt in &self.genotypes {
            for a in &gt.alleles {
                if let Some(idx) = a {
                    total += 1;
                    if *idx > 0 {
                        alt_count += 1;
                    }
                }
            }
        }
        if total == 0 { 0.0 } else { alt_count as f64 / total as f64 }
    }

    /// True if the variant passes all filters.
    pub fn is_pass(&self) -> bool {
        self.filter.is_empty() || (self.filter.len() == 1 && self.filter[0] == "PASS")
    }

    /// Get an INFO field value.
    pub fn info_value(&self, key: &str) -> Option<&str> {
        self.info.get(key).map(|s| s.as_str())
    }
}

impl fmt::Display for VcfRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{} {}->{} type={} AF={:.3}",
               self.chrom, self.pos, self.reference,
               self.alt.join(","), self.variant_type(), self.allele_frequency())
    }
}

// ── Parser ──────────────────────────────────────────────────────

/// VCF parser.
#[derive(Debug, Clone)]
pub struct VcfParser {
    pass_only: bool,
    min_qual: Option<f64>,
}

impl VcfParser {
    pub fn new() -> Self { Self { pass_only: false, min_qual: None } }
    pub fn with_pass_only(mut self, p: bool) -> Self { self.pass_only = p; self }
    pub fn with_min_qual(mut self, q: f64) -> Self { self.min_qual = Some(q); self }

    /// Parse a complete VCF string.
    pub fn parse(&self, input: &str) -> Result<(VcfHeader, Vec<VcfRecord>), VcfError> {
        if input.trim().is_empty() {
            return Err(VcfError::EmptyInput);
        }
        let mut header = VcfHeader::default();
        let mut records = Vec::new();
        let mut found_col_header = false;
        let mut line_num = 0_usize;

        for line in input.lines() {
            line_num += 1;
            let trimmed = line.trim();
            if trimmed.is_empty() { continue; }

            if trimmed.starts_with("##") {
                parse_meta_line(trimmed, &mut header);
                continue;
            }
            if trimmed.starts_with("#CHROM") {
                found_col_header = true;
                let cols: Vec<&str> = trimmed.split('\t').collect();
                if cols.len() > 9 {
                    header.samples = cols[9..].iter().map(|s| s.to_string()).collect();
                }
                continue;
            }
            if !found_col_header {
                return Err(VcfError::MissingColumnHeader);
            }

            let fields: Vec<&str> = trimmed.split('\t').collect();
            if fields.len() < 8 {
                return Err(VcfError::InsufficientFields { line: line_num, found: fields.len() });
            }
            let pos: u64 = fields[1]
                .parse()
                .map_err(|_| VcfError::InvalidPos { line: line_num, value: fields[1].to_string() })?;
            let qual = if fields[5] == "." { None } else {
                Some(fields[5].parse::<f64>()
                    .map_err(|_| VcfError::InvalidQual { line: line_num, value: fields[5].to_string() })?)
            };
            let filter: Vec<String> = if fields[6] == "." {
                Vec::new()
            } else {
                fields[6].split(';').map(|s| s.to_string()).collect()
            };
            let info = parse_info(fields[7]);
            let alt: Vec<String> = fields[4].split(',').map(|s| s.to_string()).collect();

            // FORMAT + genotypes
            let mut format_keys = Vec::new();
            let mut genotypes = Vec::new();
            if fields.len() > 8 {
                format_keys = fields[8].split(':').map(|s| s.to_string()).collect();
                for f in &fields[9..] {
                    let gt = parse_genotype_field(f, &format_keys)?;
                    genotypes.push(gt);
                }
            }

            let rec = VcfRecord {
                chrom: fields[0].to_string(),
                pos,
                id: fields[2].to_string(),
                reference: fields[3].to_string(),
                alt,
                qual,
                filter,
                info,
                format: format_keys.clone(),
                genotypes,
            };

            if self.pass_only && !rec.is_pass() { continue; }
            if let Some(mq) = self.min_qual {
                if rec.qual.unwrap_or(0.0) < mq { continue; }
            }
            records.push(rec);
        }
        Ok((header, records))
    }
}

impl fmt::Display for VcfParser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VcfParser(pass_only={}, min_qual={:?})", self.pass_only, self.min_qual)
    }
}

// ── Helpers ─────────────────────────────────────────────────────

fn parse_meta_line(line: &str, header: &mut VcfHeader) {
    let content = &line[2..];
    if let Some(rest) = content.strip_prefix("fileformat=") {
        header.file_format = rest.to_string();
    } else if let Some(rest) = content.strip_prefix("INFO=<") {
        if let Some(def) = parse_field_def(rest) {
            header.info_fields.insert(def.id.clone(), def);
        }
    } else if let Some(rest) = content.strip_prefix("FORMAT=<") {
        if let Some(def) = parse_field_def(rest) {
            header.format_fields.insert(def.id.clone(), def);
        }
    } else if let Some(rest) = content.strip_prefix("FILTER=<") {
        let inner = rest.trim_end_matches('>');
        let fields = parse_meta_fields(inner);
        if let (Some(id), Some(desc)) = (fields.get("ID"), fields.get("Description")) {
            header.filter_fields.insert(id.clone(), desc.clone());
        }
    }
}

fn parse_field_def(raw: &str) -> Option<VcfFieldDef> {
    let inner = raw.trim_end_matches('>');
    let fields = parse_meta_fields(inner);
    Some(VcfFieldDef {
        id: fields.get("ID")?.clone(),
        number: fields.get("Number").cloned().unwrap_or_default(),
        field_type: fields.get("Type").cloned().unwrap_or_default(),
        description: fields.get("Description").cloned().unwrap_or_default(),
    })
}

fn parse_meta_fields(s: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut key = String::new();
    let mut val = String::new();
    let mut in_key = true;
    let mut in_quote = false;
    for ch in s.chars() {
        if in_key {
            if ch == '=' {
                in_key = false;
            } else if ch != ',' {
                key.push(ch);
            }
        } else if ch == '"' {
            in_quote = !in_quote;
        } else if ch == ',' && !in_quote {
            map.insert(key.trim().to_string(), val.trim().to_string());
            key.clear();
            val.clear();
            in_key = true;
        } else {
            val.push(ch);
        }
    }
    if !key.is_empty() {
        map.insert(key.trim().to_string(), val.trim().to_string());
    }
    map
}

fn parse_info(field: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    if field == "." { return map; }
    for item in field.split(';') {
        if let Some(pos) = item.find('=') {
            map.insert(item[..pos].to_string(), item[pos + 1..].to_string());
        } else {
            map.insert(item.to_string(), String::new());
        }
    }
    map
}

fn parse_genotype_field(field: &str, keys: &[String]) -> Result<Genotype, VcfError> {
    let values: Vec<&str> = field.split(':').collect();
    let gt_str = values.first().copied().unwrap_or("./.");
    let mut gt = Genotype::from_gt(gt_str)?;
    for (i, k) in keys.iter().enumerate().skip(1) {
        if let Some(v) = values.get(i) {
            gt.fields.insert(k.clone(), v.to_string());
        }
    }
    Ok(gt)
}

/// Quick parse with default settings.
pub fn parse_vcf(input: &str) -> Result<(VcfHeader, Vec<VcfRecord>), VcfError> {
    VcfParser::new().parse(input)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_vcf() -> String {
        "##fileformat=VCFv4.3\n\
         ##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Total Depth\">\n\
         ##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n\
         #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\n\
         chr1\t100\trs1\tA\tG\t30\tPASS\tDP=50\tGT\t0/1\n".to_string()
    }

    #[test]
    fn t01_parse_basic() {
        let (hdr, recs) = parse_vcf(&sample_vcf()).unwrap();
        assert_eq!(hdr.file_format, "VCFv4.3");
        assert_eq!(recs.len(), 1);
    }

    #[test]
    fn t02_variant_fields() {
        let (_, recs) = parse_vcf(&sample_vcf()).unwrap();
        assert_eq!(recs[0].chrom, "chr1");
        assert_eq!(recs[0].pos, 100);
        assert_eq!(recs[0].reference, "A");
    }

    #[test]
    fn t03_variant_type_snp() {
        let (_, recs) = parse_vcf(&sample_vcf()).unwrap();
        assert_eq!(recs[0].variant_type(), VariantType::Snp);
    }

    #[test]
    fn t04_genotype_het() {
        let gt = Genotype::from_gt("0/1").unwrap();
        assert!(gt.is_het());
        assert!(!gt.is_hom_ref());
    }

    #[test]
    fn t05_genotype_hom_ref() {
        let gt = Genotype::from_gt("0/0").unwrap();
        assert!(gt.is_hom_ref());
    }

    #[test]
    fn t06_genotype_hom_alt() {
        let gt = Genotype::from_gt("1/1").unwrap();
        assert!(gt.is_hom_alt());
    }

    #[test]
    fn t07_genotype_phased() {
        let gt = Genotype::from_gt("0|1").unwrap();
        assert!(gt.phased);
    }

    #[test]
    fn t08_genotype_missing() {
        let gt = Genotype::from_gt("./.").unwrap();
        assert_eq!(gt.alleles, vec![None, None]);
    }

    #[test]
    fn t09_allele_frequency() {
        let (_, recs) = parse_vcf(&sample_vcf()).unwrap();
        assert!((recs[0].allele_frequency() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn t10_info_parsing() {
        let (_, recs) = parse_vcf(&sample_vcf()).unwrap();
        assert_eq!(recs[0].info_value("DP"), Some("50"));
    }

    #[test]
    fn t11_filter_pass() {
        let (_, recs) = parse_vcf(&sample_vcf()).unwrap();
        assert!(recs[0].is_pass());
    }

    #[test]
    fn t12_header_info_def() {
        let (hdr, _) = parse_vcf(&sample_vcf()).unwrap();
        assert!(hdr.info_fields.contains_key("DP"));
    }

    #[test]
    fn t13_samples() {
        let (hdr, _) = parse_vcf(&sample_vcf()).unwrap();
        assert_eq!(hdr.samples, vec!["SAMPLE1"]);
    }

    #[test]
    fn t14_pass_only_filter() {
        let input = format!(
            "##fileformat=VCFv4.3\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n\
             chr1\t100\t.\tA\tG\t30\tLowQual\t.\n"
        );
        let parser = VcfParser::new().with_pass_only(true);
        let (_, recs) = parser.parse(&input).unwrap();
        assert!(recs.is_empty());
    }

    #[test]
    fn t15_min_qual_filter() {
        let parser = VcfParser::new().with_min_qual(50.0);
        let (_, recs) = parser.parse(&sample_vcf()).unwrap();
        assert!(recs.is_empty());
    }

    #[test]
    fn t16_variant_type_insertion() {
        let input = "##fileformat=VCFv4.3\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n\
                     chr1\t100\t.\tA\tACG\t30\tPASS\t.\n";
        let (_, recs) = parse_vcf(input).unwrap();
        assert_eq!(recs[0].variant_type(), VariantType::Insertion);
    }

    #[test]
    fn t17_variant_type_deletion() {
        let input = "##fileformat=VCFv4.3\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n\
                     chr1\t100\t.\tACG\tA\t30\tPASS\t.\n";
        let (_, recs) = parse_vcf(input).unwrap();
        assert_eq!(recs[0].variant_type(), VariantType::Deletion);
    }

    #[test]
    fn t18_empty_input() {
        assert_eq!(parse_vcf(""), Err(VcfError::EmptyInput));
    }

    #[test]
    fn t19_display_record() {
        let (_, recs) = parse_vcf(&sample_vcf()).unwrap();
        let s = format!("{}", recs[0]);
        assert!(s.contains("chr1:100"));
        assert!(s.contains("SNP"));
    }

    #[test]
    fn t20_genotype_display() {
        let gt = Genotype::from_gt("0/1").unwrap();
        assert_eq!(format!("{gt}"), "0/1");
    }
}
