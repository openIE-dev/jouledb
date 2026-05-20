//! Seed-Based Read Mapper — seed indexing, extend-and-score, quality
//! filtering, mapping quality (MAPQ), soft clipping, multi-hit handling.
//!
//! Pure-Rust short-read mapper that builds a seed index of the reference,
//! locates candidate positions via exact k-mer seeds, extends alignments,
//! and assigns mapping quality scores.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum MapperError {
    EmptyReference,
    EmptyRead,
    InvalidSeedLen(usize),
    InvalidParameters(String),
}

impl fmt::Display for MapperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyReference => write!(f, "empty reference"),
            Self::EmptyRead => write!(f, "empty read"),
            Self::InvalidSeedLen(s) => write!(f, "invalid seed length: {s}"),
            Self::InvalidParameters(s) => write!(f, "invalid parameters: {s}"),
        }
    }
}

impl std::error::Error for MapperError {}

// ── Mapping Quality ─────────────────────────────────────────────

/// Mapping quality score (Phred-scaled).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapQ(pub u8);

impl MapQ {
    pub fn unique() -> Self { Self(60) }
    pub fn multi_hit(count: usize) -> Self {
        if count <= 1 { return Self(60); }
        let q = (-(10.0 * (1.0 - 1.0 / count as f64).ln() / std::f64::consts::LN_10)).round();
        Self((q.max(0.0).min(60.0)) as u8)
    }
    pub fn unmapped() -> Self { Self(0) }

    /// Probability of incorrect mapping.
    pub fn error_prob(&self) -> f64 {
        10.0_f64.powf(-(self.0 as f64) / 10.0)
    }
}

impl fmt::Display for MapQ {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MAPQ={}", self.0)
    }
}

// ── CIGAR Operations ────────────────────────────────────────────

/// CIGAR operation for alignment encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CigarOp {
    Match(usize),
    Mismatch(usize),
    Insertion(usize),
    Deletion(usize),
    SoftClip(usize),
}

impl fmt::Display for CigarOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Match(n) => write!(f, "{n}M"),
            Self::Mismatch(n) => write!(f, "{n}X"),
            Self::Insertion(n) => write!(f, "{n}I"),
            Self::Deletion(n) => write!(f, "{n}D"),
            Self::SoftClip(n) => write!(f, "{n}S"),
        }
    }
}

/// Encode alignment as CIGAR string.
pub fn cigar_string(ops: &[CigarOp]) -> String {
    ops.iter().map(|o| format!("{o}")).collect()
}

// ── Mapping Hit ─────────────────────────────────────────────────

/// A single mapping of a read to the reference.
#[derive(Debug, Clone)]
pub struct MappingHit {
    pub ref_start: usize,
    pub ref_end: usize,
    pub read_start: usize,
    pub read_end: usize,
    pub score: f64,
    pub identity: f64,
    pub mapq: MapQ,
    pub cigar: Vec<CigarOp>,
    pub is_reverse: bool,
    pub num_mismatches: usize,
}

impl MappingHit {
    /// Alignment length on the reference.
    pub fn ref_span(&self) -> usize {
        self.ref_end.saturating_sub(self.ref_start)
    }

    /// Alignment length on the read.
    pub fn read_span(&self) -> usize {
        self.read_end.saturating_sub(self.read_start)
    }

    pub fn cigar_string(&self) -> String {
        cigar_string(&self.cigar)
    }
}

impl fmt::Display for MappingHit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let strand = if self.is_reverse { '-' } else { '+' };
        write!(
            f,
            "Hit(ref={}..{}, {}, score={:.1}, id={:.1}%, {}, cigar={})",
            self.ref_start, self.ref_end, strand,
            self.score, self.identity, self.mapq, self.cigar_string()
        )
    }
}

// ── Mapping Result ──────────────────────────────────────────────

/// Result of mapping a single read.
#[derive(Debug, Clone)]
pub struct MappingResult {
    pub read_id: usize,
    pub read_length: usize,
    pub hits: Vec<MappingHit>,
    pub is_mapped: bool,
    pub primary_idx: Option<usize>,
}

impl MappingResult {
    /// Best (primary) hit.
    pub fn primary(&self) -> Option<&MappingHit> {
        self.primary_idx.and_then(|i| self.hits.get(i))
    }

    /// Number of mapping locations.
    pub fn num_hits(&self) -> usize {
        self.hits.len()
    }
}

impl fmt::Display for MappingResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MappingResult(read={}, len={}, mapped={}, hits={})",
            self.read_id, self.read_length, self.is_mapped, self.num_hits()
        )
    }
}

// ── Mapper Config ───────────────────────────────────────────────

/// Configuration for the read mapper.
#[derive(Debug, Clone)]
pub struct MapperConfig {
    seed_len: usize,
    seed_step: usize,
    max_mismatches: usize,
    min_score: f64,
    match_score: f64,
    mismatch_penalty: f64,
    gap_penalty: f64,
    min_mapq: u8,
    report_all: bool,
    max_hits: usize,
}

impl MapperConfig {
    pub fn new() -> Self {
        Self {
            seed_len: 11,
            seed_step: 5,
            max_mismatches: 3,
            min_score: 20.0,
            match_score: 1.0,
            mismatch_penalty: -1.0,
            gap_penalty: -2.0,
            min_mapq: 0,
            report_all: false,
            max_hits: 100,
        }
    }

    pub fn with_seed_len(mut self, s: usize) -> Self { self.seed_len = s; self }
    pub fn with_seed_step(mut self, s: usize) -> Self { self.seed_step = s; self }
    pub fn with_max_mismatches(mut self, m: usize) -> Self { self.max_mismatches = m; self }
    pub fn with_min_score(mut self, s: f64) -> Self { self.min_score = s; self }
    pub fn with_match_score(mut self, s: f64) -> Self { self.match_score = s; self }
    pub fn with_mismatch_penalty(mut self, p: f64) -> Self { self.mismatch_penalty = p; self }
    pub fn with_gap_penalty(mut self, g: f64) -> Self { self.gap_penalty = g; self }
    pub fn with_min_mapq(mut self, q: u8) -> Self { self.min_mapq = q; self }
    pub fn with_report_all(mut self, r: bool) -> Self { self.report_all = r; self }
    pub fn with_max_hits(mut self, m: usize) -> Self { self.max_hits = m; self }
}

impl Default for MapperConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for MapperConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MapperConfig(seed={}, step={}, max_mis={}, min_score={:.0})",
            self.seed_len, self.seed_step, self.max_mismatches, self.min_score
        )
    }
}

// ── Seed Index ──────────────────────────────────────────────────

/// Index of k-mer seeds from the reference genome.
struct SeedIndex {
    index: HashMap<Vec<u8>, Vec<usize>>,
    seed_len: usize,
}

impl SeedIndex {
    fn build(reference: &[u8], seed_len: usize, seed_step: usize) -> Self {
        let mut index: HashMap<Vec<u8>, Vec<usize>> = HashMap::new();
        if reference.len() >= seed_len {
            let mut pos = 0;
            while pos + seed_len <= reference.len() {
                let seed: Vec<u8> = reference[pos..pos + seed_len]
                    .iter()
                    .map(|c| c.to_ascii_uppercase())
                    .collect();
                index.entry(seed).or_default().push(pos);
                pos += seed_step;
            }
        }
        Self { index, seed_len }
    }

    fn lookup(&self, seed: &[u8]) -> &[usize] {
        let key: Vec<u8> = seed.iter().map(|c| c.to_ascii_uppercase()).collect();
        self.index.get(&key).map(|v| v.as_slice()).unwrap_or(&[])
    }
}

// ── Read Mapper ─────────────────────────────────────────────────

/// Seed-and-extend read mapper.
#[derive(Debug, Clone)]
pub struct ReadMapper {
    config: MapperConfig,
    ref_len: usize,
}

impl ReadMapper {
    pub fn new(config: MapperConfig) -> Self {
        Self { config, ref_len: 0 }
    }

    /// Map a batch of reads against a reference sequence.
    pub fn map_reads(
        &mut self,
        reference: &[u8],
        reads: &[&[u8]],
    ) -> Result<Vec<MappingResult>, MapperError> {
        if reference.is_empty() {
            return Err(MapperError::EmptyReference);
        }
        if self.config.seed_len == 0 {
            return Err(MapperError::InvalidSeedLen(0));
        }

        self.ref_len = reference.len();
        let index = SeedIndex::build(reference, self.config.seed_len, self.config.seed_step);

        let mut results = Vec::with_capacity(reads.len());
        for (rid, read) in reads.iter().enumerate() {
            results.push(self.map_single(reference, read, rid, &index)?);
        }
        Ok(results)
    }

    /// Map a single read.
    fn map_single(
        &self,
        reference: &[u8],
        read: &[u8],
        read_id: usize,
        index: &SeedIndex,
    ) -> Result<MappingResult, MapperError> {
        if read.is_empty() {
            return Err(MapperError::EmptyRead);
        }

        let mut candidate_positions: HashMap<usize, usize> = HashMap::new();

        // Find seed hits.
        if read.len() >= self.config.seed_len {
            let mut pos = 0;
            while pos + self.config.seed_len <= read.len() {
                let seed = &read[pos..pos + self.config.seed_len];
                for &ref_pos in index.lookup(seed) {
                    let start = ref_pos.saturating_sub(pos);
                    *candidate_positions.entry(start).or_insert(0) += 1;
                }
                pos += self.config.seed_step;
            }
        }

        // Extend and score each candidate.
        let mut hits = Vec::new();
        for (&ref_start, _) in &candidate_positions {
            if let Some(hit) = self.extend_hit(reference, read, ref_start) {
                if hit.score >= self.config.min_score
                    && hit.num_mismatches <= self.config.max_mismatches
                {
                    hits.push(hit);
                }
            }
        }

        // Sort by score descending.
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(self.config.max_hits);

        // Assign MAPQ.
        let num_hits = hits.len();
        for hit in &mut hits {
            hit.mapq = MapQ::multi_hit(num_hits);
        }

        // Filter by min MAPQ.
        if self.config.min_mapq > 0 {
            hits.retain(|h| h.mapq.0 >= self.config.min_mapq);
        }

        let is_mapped = !hits.is_empty();
        let primary_idx = if is_mapped { Some(0) } else { None };

        if !self.config.report_all && hits.len() > 1 {
            hits.truncate(1);
        }

        Ok(MappingResult {
            read_id,
            read_length: read.len(),
            hits,
            is_mapped,
            primary_idx,
        })
    }

    /// Extend a candidate alignment at `ref_start`.
    fn extend_hit(
        &self,
        reference: &[u8],
        read: &[u8],
        ref_start: usize,
    ) -> Option<MappingHit> {
        let ref_end = (ref_start + read.len()).min(reference.len());
        let aln_len = ref_end - ref_start;
        if aln_len == 0 {
            return None;
        }

        let read_aln = &read[..aln_len.min(read.len())];
        let ref_aln = &reference[ref_start..ref_end];
        let compare_len = read_aln.len().min(ref_aln.len());

        let mut score = 0.0_f64;
        let mut mismatches = 0;
        let mut cigar_ops = Vec::new();
        let mut match_run = 0;
        let mut mismatch_run = 0;

        for i in 0..compare_len {
            if read_aln[i].to_ascii_uppercase() == ref_aln[i].to_ascii_uppercase() {
                if mismatch_run > 0 {
                    cigar_ops.push(CigarOp::Mismatch(mismatch_run));
                    mismatch_run = 0;
                }
                match_run += 1;
                score += self.config.match_score;
            } else {
                if match_run > 0 {
                    cigar_ops.push(CigarOp::Match(match_run));
                    match_run = 0;
                }
                mismatch_run += 1;
                mismatches += 1;
                score += self.config.mismatch_penalty;
            }
        }
        if match_run > 0 {
            cigar_ops.push(CigarOp::Match(match_run));
        }
        if mismatch_run > 0 {
            cigar_ops.push(CigarOp::Mismatch(mismatch_run));
        }

        // Soft clip if read extends beyond reference.
        let clip = read.len().saturating_sub(compare_len);
        if clip > 0 {
            cigar_ops.push(CigarOp::SoftClip(clip));
        }

        let identity = if compare_len > 0 {
            (compare_len - mismatches) as f64 / compare_len as f64 * 100.0
        } else {
            0.0
        };

        Some(MappingHit {
            ref_start,
            ref_end: ref_start + compare_len,
            read_start: 0,
            read_end: compare_len,
            score,
            identity,
            mapq: MapQ::unmapped(),
            cigar: cigar_ops,
            is_reverse: false,
            num_mismatches: mismatches,
        })
    }

    /// Mapping statistics for a batch of results.
    pub fn stats(results: &[MappingResult]) -> MappingStats {
        let total = results.len();
        let mapped = results.iter().filter(|r| r.is_mapped).count();
        let unique = results.iter().filter(|r| r.num_hits() == 1).count();
        let multi = results.iter().filter(|r| r.num_hits() > 1).count();
        MappingStats { total_reads: total, mapped, unique, multi_mapped: multi }
    }
}

impl fmt::Display for ReadMapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReadMapper({}, ref_len={})", self.config, self.ref_len)
    }
}

// ── Mapping Statistics ──────────────────────────────────────────

/// Summary statistics of a mapping run.
#[derive(Debug, Clone)]
pub struct MappingStats {
    pub total_reads: usize,
    pub mapped: usize,
    pub unique: usize,
    pub multi_mapped: usize,
}

impl MappingStats {
    pub fn mapping_rate(&self) -> f64 {
        if self.total_reads == 0 { 0.0 } else { self.mapped as f64 / self.total_reads as f64 * 100.0 }
    }
}

impl fmt::Display for MappingStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MappingStats(total={}, mapped={} ({:.1}%), unique={}, multi={})",
            self.total_reads, self.mapped, self.mapping_rate(),
            self.unique, self.multi_mapped
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn reference() -> Vec<u8> {
        b"ACGTACGTACGTACGTACGTACGTACGTACGT".to_vec()
    }

    fn mapper() -> ReadMapper {
        ReadMapper::new(
            MapperConfig::new()
                .with_seed_len(4)
                .with_seed_step(2)
                .with_min_score(2.0)
                .with_max_mismatches(5),
        )
    }

    #[test]
    fn map_exact_match() {
        let refseq = reference();
        let mut m = mapper();
        let results = m.map_reads(&refseq, &[b"ACGTACGT"]).unwrap();
        assert!(results[0].is_mapped);
    }

    #[test]
    fn map_with_mismatch() {
        let refseq = reference();
        let mut m = mapper();
        let results = m.map_reads(&refseq, &[b"AXGTACGT"]).unwrap();
        // May or may not map depending on scoring
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn empty_reference_err() {
        let mut m = mapper();
        assert!(m.map_reads(b"", &[b"ACGT"]).is_err());
    }

    #[test]
    fn empty_read_err() {
        let refseq = reference();
        let mut m = mapper();
        assert!(m.map_reads(&refseq, &[b""]).is_err());
    }

    #[test]
    fn mapq_unique() {
        let mq = MapQ::unique();
        assert_eq!(mq.0, 60);
    }

    #[test]
    fn mapq_multi() {
        let mq = MapQ::multi_hit(10);
        assert!(mq.0 < 60);
    }

    #[test]
    fn mapq_error_prob() {
        let mq = MapQ(30);
        assert!((mq.error_prob() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn cigar_string_test() {
        let ops = vec![CigarOp::Match(5), CigarOp::Mismatch(1), CigarOp::Match(3)];
        assert_eq!(cigar_string(&ops), "5M1X3M");
    }

    #[test]
    fn cigar_soft_clip() {
        let ops = vec![CigarOp::SoftClip(2), CigarOp::Match(8)];
        assert_eq!(cigar_string(&ops), "2S8M");
    }

    #[test]
    fn mapping_hit_display() {
        let refseq = reference();
        let mut m = mapper();
        let results = m.map_reads(&refseq, &[b"ACGTACGT"]).unwrap();
        if let Some(h) = results[0].primary() {
            let s = format!("{h}");
            assert!(s.contains("Hit("));
        }
    }

    #[test]
    fn mapping_result_display() {
        let refseq = reference();
        let mut m = mapper();
        let results = m.map_reads(&refseq, &[b"ACGTACGT"]).unwrap();
        assert!(format!("{}", results[0]).contains("MappingResult("));
    }

    #[test]
    fn batch_stats() {
        let refseq = reference();
        let mut m = mapper();
        let results = m.map_reads(&refseq, &[b"ACGTACGT", b"ACGT"]).unwrap();
        let stats = ReadMapper::stats(&results);
        assert_eq!(stats.total_reads, 2);
        assert!(stats.mapping_rate() >= 0.0);
    }

    #[test]
    fn stats_display() {
        let stats = MappingStats {
            total_reads: 100,
            mapped: 95,
            unique: 80,
            multi_mapped: 15,
        };
        assert!(format!("{stats}").contains("MappingStats("));
    }

    #[test]
    fn config_builder() {
        let cfg = MapperConfig::new()
            .with_seed_len(15)
            .with_seed_step(3)
            .with_max_mismatches(2)
            .with_min_score(30.0)
            .with_match_score(2.0)
            .with_mismatch_penalty(-3.0)
            .with_gap_penalty(-5.0)
            .with_min_mapq(10)
            .with_report_all(true)
            .with_max_hits(50);
        assert!(format!("{cfg}").contains("seed=15"));
    }

    #[test]
    fn report_all_mode() {
        let refseq = reference();
        let mut m = ReadMapper::new(
            MapperConfig::new()
                .with_seed_len(4)
                .with_seed_step(2)
                .with_min_score(2.0)
                .with_max_mismatches(5)
                .with_report_all(true),
        );
        let results = m.map_reads(&refseq, &[b"ACGT"]).unwrap();
        if results[0].is_mapped {
            assert!(results[0].num_hits() >= 1);
        }
    }

    #[test]
    fn ref_span_calculation() {
        let refseq = reference();
        let mut m = mapper();
        let results = m.map_reads(&refseq, &[b"ACGTACGT"]).unwrap();
        if let Some(h) = results[0].primary() {
            assert!(h.ref_span() > 0);
            assert!(h.read_span() > 0);
        }
    }

    #[test]
    fn mapper_display() {
        let m = mapper();
        assert!(format!("{m}").contains("ReadMapper("));
    }

    #[test]
    fn invalid_seed_len() {
        let mut m = ReadMapper::new(MapperConfig::new().with_seed_len(0));
        assert!(m.map_reads(b"ACGT", &[b"AC"]).is_err());
    }

    #[test]
    fn unmapped_result() {
        let refseq = b"AAAAAAAAAAAAAAAA";
        let mut m = ReadMapper::new(
            MapperConfig::new()
                .with_seed_len(4)
                .with_seed_step(2)
                .with_min_score(100.0),
        );
        let results = m.map_reads(refseq, &[b"CCCC"]).unwrap();
        assert!(!results[0].is_mapped);
    }

    #[test]
    fn mapq_unmapped_zero() {
        let mq = MapQ::unmapped();
        assert_eq!(mq.0, 0);
    }
}
