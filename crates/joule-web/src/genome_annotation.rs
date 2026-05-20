//! Genome annotation engine for gene features, tracks, and annotation merging.
//!
//! Provides a complete gene annotation pipeline including feature creation
//! (CDS, exon, intron, UTR, promoter, terminator), track management with
//! strand-aware indexing, annotation merging with conflict resolution,
//! GFF3-style serialization, and interval overlap queries.

use std::fmt;
use std::collections::HashMap;

// ── Feature Kind ────────────────────────────────────────────────

/// Biological feature type on the genome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FeatureKind {
    Gene,
    Mrna,
    Cds,
    Exon,
    Intron,
    FivePrimeUtr,
    ThreePrimeUtr,
    Promoter,
    Terminator,
    RepeatRegion,
    Custom,
}

impl fmt::Display for FeatureKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gene => write!(f, "gene"),
            Self::Mrna => write!(f, "mRNA"),
            Self::Cds => write!(f, "CDS"),
            Self::Exon => write!(f, "exon"),
            Self::Intron => write!(f, "intron"),
            Self::FivePrimeUtr => write!(f, "five_prime_UTR"),
            Self::ThreePrimeUtr => write!(f, "three_prime_UTR"),
            Self::Promoter => write!(f, "promoter"),
            Self::Terminator => write!(f, "terminator"),
            Self::RepeatRegion => write!(f, "repeat_region"),
            Self::Custom => write!(f, "region"),
        }
    }
}

// ── Strand ──────────────────────────────────────────────────────

/// DNA strand orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Strand {
    Forward,
    Reverse,
    Unknown,
}

impl fmt::Display for Strand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Forward => write!(f, "+"),
            Self::Reverse => write!(f, "-"),
            Self::Unknown => write!(f, "."),
        }
    }
}

// ── Genomic Feature ─────────────────────────────────────────────

/// A single annotated feature on a reference sequence.
#[derive(Debug, Clone)]
pub struct GenomicFeature {
    pub id: String,
    pub seqid: String,
    pub kind: FeatureKind,
    pub start: usize,
    pub end: usize,
    pub strand: Strand,
    pub score: f64,
    pub phase: Option<u8>,
    pub parent_id: Option<String>,
    pub attributes: HashMap<String, String>,
}

impl GenomicFeature {
    /// Create a new feature at a genomic interval.
    pub fn new(seqid: &str, kind: FeatureKind, start: usize, end: usize) -> Self {
        Self {
            id: String::new(),
            seqid: seqid.to_string(),
            kind,
            start,
            end,
            strand: Strand::Unknown,
            score: 0.0,
            phase: None,
            parent_id: None,
            attributes: HashMap::new(),
        }
    }

    pub fn with_id(mut self, id: &str) -> Self {
        self.id = id.to_string();
        self
    }

    pub fn with_strand(mut self, strand: Strand) -> Self {
        self.strand = strand;
        self
    }

    pub fn with_score(mut self, score: f64) -> Self {
        self.score = score;
        self
    }

    pub fn with_phase(mut self, phase: u8) -> Self {
        self.phase = Some(phase);
        self
    }

    pub fn with_parent(mut self, parent_id: &str) -> Self {
        self.parent_id = Some(parent_id.to_string());
        self
    }

    pub fn with_attribute(mut self, key: &str, value: &str) -> Self {
        self.attributes.insert(key.to_string(), value.to_string());
        self
    }

    /// Length of this feature in base pairs.
    pub fn length(&self) -> usize {
        if self.end >= self.start { self.end - self.start + 1 } else { 0 }
    }

    /// Whether this feature overlaps the given interval.
    pub fn overlaps(&self, start: usize, end: usize) -> bool {
        self.start <= end && start <= self.end
    }

    /// Reciprocal overlap fraction between this feature and another.
    pub fn reciprocal_overlap(&self, other: &GenomicFeature) -> f64 {
        if self.seqid != other.seqid {
            return 0.0;
        }
        let ov_start = self.start.max(other.start);
        let ov_end = self.end.min(other.end);
        if ov_start > ov_end {
            return 0.0;
        }
        let overlap_len = (ov_end - ov_start + 1) as f64;
        let union_len = (self.length().max(other.length())) as f64;
        if union_len == 0.0 { 0.0 } else { overlap_len / union_len }
    }
}

impl fmt::Display for GenomicFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}\t.\t{}\t{}\t{}\t{:.1}\t{}\t{}\tID={}",
            self.seqid,
            self.kind,
            self.start,
            self.end,
            self.score,
            self.strand,
            self.phase.map_or(".".to_string(), |p| p.to_string()),
            self.id,
        )
    }
}

// ── Annotation Track ────────────────────────────────────────────

/// A named track grouping features of the same type.
#[derive(Debug, Clone)]
pub struct AnnotationTrack {
    pub name: String,
    pub kind: FeatureKind,
    pub features: Vec<GenomicFeature>,
    pub description: String,
}

impl AnnotationTrack {
    pub fn new(name: &str, kind: FeatureKind) -> Self {
        Self {
            name: name.to_string(),
            kind,
            features: Vec::new(),
            description: String::new(),
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Add a feature to this track.
    pub fn add_feature(&mut self, feature: GenomicFeature) {
        self.features.push(feature);
    }

    /// Sort features by start position.
    pub fn sort_by_position(&mut self) {
        self.features.sort_by_key(|f| (f.start, f.end));
    }

    /// Find all features overlapping a query interval.
    pub fn query_overlap(&self, start: usize, end: usize) -> Vec<&GenomicFeature> {
        self.features.iter().filter(|f| f.overlaps(start, end)).collect()
    }

    /// Features on a specific strand.
    pub fn on_strand(&self, strand: Strand) -> Vec<&GenomicFeature> {
        self.features.iter().filter(|f| f.strand == strand).collect()
    }

    /// Total base-pair coverage (simple sum, may double-count overlaps).
    pub fn total_coverage_bp(&self) -> usize {
        self.features.iter().map(|f| f.length()).sum()
    }
}

impl fmt::Display for AnnotationTrack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Track({}, {} features, kind={})", self.name, self.features.len(), self.kind)
    }
}

// ── Merge Strategy ──────────────────────────────────────────────

/// Strategy for resolving conflicts when merging overlapping annotations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MergeStrategy {
    /// Keep both annotations.
    KeepBoth,
    /// Keep the feature with the higher score.
    HighestScore,
    /// Merge into a single feature spanning both intervals.
    Union,
    /// Keep only the overlapping region.
    Intersection,
}

impl fmt::Display for MergeStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KeepBoth => write!(f, "keep_both"),
            Self::HighestScore => write!(f, "highest_score"),
            Self::Union => write!(f, "union"),
            Self::Intersection => write!(f, "intersection"),
        }
    }
}

// ── Annotation Set ──────────────────────────────────────────────

/// A collection of annotation tracks for a genome assembly.
#[derive(Debug, Clone)]
pub struct AnnotationSet {
    pub assembly: String,
    pub tracks: Vec<AnnotationTrack>,
    pub merge_strategy: MergeStrategy,
}

impl AnnotationSet {
    pub fn new(assembly: &str) -> Self {
        Self {
            assembly: assembly.to_string(),
            tracks: Vec::new(),
            merge_strategy: MergeStrategy::KeepBoth,
        }
    }

    pub fn with_merge_strategy(mut self, strategy: MergeStrategy) -> Self {
        self.merge_strategy = strategy;
        self
    }

    pub fn add_track(&mut self, track: AnnotationTrack) {
        self.tracks.push(track);
    }

    /// Total feature count across all tracks.
    pub fn feature_count(&self) -> usize {
        self.tracks.iter().map(|t| t.features.len()).sum()
    }

    /// Merge two annotation sets using the configured strategy.
    pub fn merge(&self, other: &AnnotationSet) -> AnnotationSet {
        let mut result = AnnotationSet::new(&self.assembly)
            .with_merge_strategy(self.merge_strategy);

        // Collect features by kind
        let mut by_kind: HashMap<FeatureKind, Vec<GenomicFeature>> = HashMap::new();
        for track in self.tracks.iter().chain(other.tracks.iter()) {
            for feat in &track.features {
                by_kind.entry(track.kind).or_default().push(feat.clone());
            }
        }

        for (kind, mut features) in by_kind {
            features.sort_by_key(|f| (f.start, f.end));
            let merged = self.apply_merge_strategy(&features);
            let mut track = AnnotationTrack::new(&format!("merged_{}", kind), kind);
            for f in merged {
                track.add_feature(f);
            }
            result.add_track(track);
        }
        result
    }

    fn apply_merge_strategy(&self, features: &[GenomicFeature]) -> Vec<GenomicFeature> {
        match self.merge_strategy {
            MergeStrategy::KeepBoth => features.to_vec(),
            MergeStrategy::HighestScore => self.merge_highest_score(features),
            MergeStrategy::Union => self.merge_union(features),
            MergeStrategy::Intersection => self.merge_intersection(features),
        }
    }

    fn merge_highest_score(&self, features: &[GenomicFeature]) -> Vec<GenomicFeature> {
        let mut result: Vec<GenomicFeature> = Vec::new();
        for feat in features {
            let mut dominated = false;
            result.retain(|existing: &GenomicFeature| {
                if feat.overlaps(existing.start, existing.end) && feat.seqid == existing.seqid {
                    if feat.score > existing.score {
                        return false; // remove existing
                    } else {
                        dominated = true;
                    }
                }
                true
            });
            if !dominated {
                result.push(feat.clone());
            }
        }
        result
    }

    fn merge_union(&self, features: &[GenomicFeature]) -> Vec<GenomicFeature> {
        let mut result: Vec<GenomicFeature> = Vec::new();
        for feat in features {
            let mut merged = false;
            for existing in result.iter_mut() {
                if feat.overlaps(existing.start, existing.end) && feat.seqid == existing.seqid {
                    existing.start = existing.start.min(feat.start);
                    existing.end = existing.end.max(feat.end);
                    existing.score = existing.score.max(feat.score);
                    merged = true;
                    break;
                }
            }
            if !merged {
                result.push(feat.clone());
            }
        }
        result
    }

    fn merge_intersection(&self, features: &[GenomicFeature]) -> Vec<GenomicFeature> {
        let mut result: Vec<GenomicFeature> = Vec::new();
        for i in 0..features.len() {
            for j in (i + 1)..features.len() {
                let a = &features[i];
                let b = &features[j];
                if a.overlaps(b.start, b.end) && a.seqid == b.seqid {
                    let start = a.start.max(b.start);
                    let end = a.end.min(b.end);
                    let mut inter = GenomicFeature::new(&a.seqid, a.kind, start, end)
                        .with_strand(a.strand)
                        .with_score((a.score + b.score) / 2.0);
                    inter.id = format!("{}_{}_inter", a.id, b.id);
                    result.push(inter);
                }
            }
        }
        result
    }
}

impl fmt::Display for AnnotationSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AnnotationSet(assembly={}, tracks={}, features={}, merge={})",
            self.assembly,
            self.tracks.len(),
            self.feature_count(),
            self.merge_strategy,
        )
    }
}

// ── GFF3 Serialization ─────────────────────────────────────────

/// Serialize an annotation set to GFF3 format lines.
pub fn to_gff3_lines(set: &AnnotationSet) -> Vec<String> {
    let mut lines = vec!["##gff-version 3".to_string()];
    for track in &set.tracks {
        for feat in &track.features {
            lines.push(format!("{}", feat));
        }
    }
    lines
}

/// Count features grouped by kind across an annotation set.
pub fn feature_summary(set: &AnnotationSet) -> HashMap<FeatureKind, usize> {
    let mut counts = HashMap::new();
    for track in &set.tracks {
        for feat in &track.features {
            *counts.entry(feat.kind).or_insert(0) += 1;
        }
    }
    counts
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_length() {
        let f = GenomicFeature::new("chr1", FeatureKind::Gene, 100, 200);
        assert_eq!(f.length(), 101);
    }

    #[test]
    fn test_feature_overlap() {
        let f = GenomicFeature::new("chr1", FeatureKind::Exon, 100, 200);
        assert!(f.overlaps(150, 250));
        assert!(f.overlaps(50, 150));
        assert!(!f.overlaps(201, 300));
        assert!(!f.overlaps(10, 99));
    }

    #[test]
    fn test_reciprocal_overlap_same() {
        let a = GenomicFeature::new("chr1", FeatureKind::Cds, 100, 200);
        let b = GenomicFeature::new("chr1", FeatureKind::Cds, 100, 200);
        assert!((a.reciprocal_overlap(&b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_reciprocal_overlap_none() {
        let a = GenomicFeature::new("chr1", FeatureKind::Cds, 100, 200);
        let b = GenomicFeature::new("chr1", FeatureKind::Cds, 300, 400);
        assert_eq!(a.reciprocal_overlap(&b), 0.0);
    }

    #[test]
    fn test_reciprocal_overlap_diff_seqid() {
        let a = GenomicFeature::new("chr1", FeatureKind::Cds, 100, 200);
        let b = GenomicFeature::new("chr2", FeatureKind::Cds, 100, 200);
        assert_eq!(a.reciprocal_overlap(&b), 0.0);
    }

    #[test]
    fn test_feature_with_builders() {
        let f = GenomicFeature::new("chr2", FeatureKind::Gene, 500, 1000)
            .with_id("gene_001")
            .with_strand(Strand::Forward)
            .with_score(42.5)
            .with_phase(0)
            .with_parent("mRNA_001")
            .with_attribute("Name", "BRCA1");
        assert_eq!(f.id, "gene_001");
        assert_eq!(f.strand, Strand::Forward);
        assert!((f.score - 42.5).abs() < 1e-9);
        assert_eq!(f.phase, Some(0));
        assert_eq!(f.parent_id.as_deref(), Some("mRNA_001"));
        assert_eq!(f.attributes.get("Name").map(|s| s.as_str()), Some("BRCA1"));
    }

    #[test]
    fn test_strand_display() {
        assert_eq!(format!("{}", Strand::Forward), "+");
        assert_eq!(format!("{}", Strand::Reverse), "-");
        assert_eq!(format!("{}", Strand::Unknown), ".");
    }

    #[test]
    fn test_feature_kind_display() {
        assert_eq!(format!("{}", FeatureKind::Gene), "gene");
        assert_eq!(format!("{}", FeatureKind::Cds), "CDS");
        assert_eq!(format!("{}", FeatureKind::FivePrimeUtr), "five_prime_UTR");
    }

    #[test]
    fn test_track_query_overlap() {
        let mut track = AnnotationTrack::new("genes", FeatureKind::Gene);
        track.add_feature(GenomicFeature::new("chr1", FeatureKind::Gene, 100, 200));
        track.add_feature(GenomicFeature::new("chr1", FeatureKind::Gene, 300, 400));
        track.add_feature(GenomicFeature::new("chr1", FeatureKind::Gene, 500, 600));
        let hits = track.query_overlap(150, 350);
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn test_track_on_strand() {
        let mut track = AnnotationTrack::new("cds", FeatureKind::Cds);
        track.add_feature(GenomicFeature::new("chr1", FeatureKind::Cds, 100, 200).with_strand(Strand::Forward));
        track.add_feature(GenomicFeature::new("chr1", FeatureKind::Cds, 300, 400).with_strand(Strand::Reverse));
        track.add_feature(GenomicFeature::new("chr1", FeatureKind::Cds, 500, 600).with_strand(Strand::Forward));
        assert_eq!(track.on_strand(Strand::Forward).len(), 2);
        assert_eq!(track.on_strand(Strand::Reverse).len(), 1);
    }

    #[test]
    fn test_track_sort_by_position() {
        let mut track = AnnotationTrack::new("exons", FeatureKind::Exon);
        track.add_feature(GenomicFeature::new("chr1", FeatureKind::Exon, 500, 600));
        track.add_feature(GenomicFeature::new("chr1", FeatureKind::Exon, 100, 200));
        track.add_feature(GenomicFeature::new("chr1", FeatureKind::Exon, 300, 400));
        track.sort_by_position();
        assert_eq!(track.features[0].start, 100);
        assert_eq!(track.features[2].start, 500);
    }

    #[test]
    fn test_annotation_set_merge_keep_both() {
        let mut set_a = AnnotationSet::new("GRCh38");
        let mut t = AnnotationTrack::new("genes_a", FeatureKind::Gene);
        t.add_feature(GenomicFeature::new("chr1", FeatureKind::Gene, 100, 200).with_id("a1"));
        set_a.add_track(t);

        let mut set_b = AnnotationSet::new("GRCh38");
        let mut t2 = AnnotationTrack::new("genes_b", FeatureKind::Gene);
        t2.add_feature(GenomicFeature::new("chr1", FeatureKind::Gene, 150, 250).with_id("b1"));
        set_b.add_track(t2);

        let merged = set_a.merge(&set_b);
        assert_eq!(merged.feature_count(), 2);
    }

    #[test]
    fn test_annotation_set_merge_highest_score() {
        let mut set_a = AnnotationSet::new("GRCh38")
            .with_merge_strategy(MergeStrategy::HighestScore);
        let mut t = AnnotationTrack::new("g_a", FeatureKind::Gene);
        t.add_feature(GenomicFeature::new("chr1", FeatureKind::Gene, 100, 200).with_id("a1").with_score(10.0));
        set_a.add_track(t);

        let mut set_b = AnnotationSet::new("GRCh38");
        let mut t2 = AnnotationTrack::new("g_b", FeatureKind::Gene);
        t2.add_feature(GenomicFeature::new("chr1", FeatureKind::Gene, 150, 250).with_id("b1").with_score(20.0));
        set_b.add_track(t2);

        let merged = set_a.merge(&set_b);
        assert_eq!(merged.feature_count(), 1);
    }

    #[test]
    fn test_annotation_set_merge_union() {
        let mut set_a = AnnotationSet::new("GRCh38")
            .with_merge_strategy(MergeStrategy::Union);
        let mut t = AnnotationTrack::new("g_a", FeatureKind::Gene);
        t.add_feature(GenomicFeature::new("chr1", FeatureKind::Gene, 100, 200).with_id("a1"));
        set_a.add_track(t);

        let mut set_b = AnnotationSet::new("GRCh38");
        let mut t2 = AnnotationTrack::new("g_b", FeatureKind::Gene);
        t2.add_feature(GenomicFeature::new("chr1", FeatureKind::Gene, 150, 250).with_id("b1"));
        set_b.add_track(t2);

        let merged = set_a.merge(&set_b);
        assert_eq!(merged.feature_count(), 1);
        let f = &merged.tracks[0].features[0];
        assert_eq!(f.start, 100);
        assert_eq!(f.end, 250);
    }

    #[test]
    fn test_gff3_serialization() {
        let mut set = AnnotationSet::new("test");
        let mut t = AnnotationTrack::new("genes", FeatureKind::Gene);
        t.add_feature(GenomicFeature::new("chr1", FeatureKind::Gene, 1, 100).with_id("g1").with_strand(Strand::Forward));
        set.add_track(t);
        let lines = to_gff3_lines(&set);
        assert_eq!(lines[0], "##gff-version 3");
        assert!(lines[1].contains("chr1"));
        assert!(lines[1].contains("gene"));
    }

    #[test]
    fn test_feature_summary() {
        let mut set = AnnotationSet::new("test");
        let mut t1 = AnnotationTrack::new("genes", FeatureKind::Gene);
        t1.add_feature(GenomicFeature::new("chr1", FeatureKind::Gene, 1, 100));
        t1.add_feature(GenomicFeature::new("chr1", FeatureKind::Gene, 200, 300));
        let mut t2 = AnnotationTrack::new("exons", FeatureKind::Exon);
        t2.add_feature(GenomicFeature::new("chr1", FeatureKind::Exon, 10, 50));
        set.add_track(t1);
        set.add_track(t2);
        let summary = feature_summary(&set);
        assert_eq!(summary[&FeatureKind::Gene], 2);
        assert_eq!(summary[&FeatureKind::Exon], 1);
    }

    #[test]
    fn test_track_total_coverage() {
        let mut track = AnnotationTrack::new("exons", FeatureKind::Exon);
        track.add_feature(GenomicFeature::new("chr1", FeatureKind::Exon, 100, 199));
        track.add_feature(GenomicFeature::new("chr1", FeatureKind::Exon, 300, 399));
        assert_eq!(track.total_coverage_bp(), 200);
    }

    #[test]
    fn test_annotation_set_display() {
        let set = AnnotationSet::new("GRCh38");
        let s = format!("{}", set);
        assert!(s.contains("GRCh38"));
        assert!(s.contains("tracks=0"));
    }

    #[test]
    fn test_feature_display_gff3() {
        let f = GenomicFeature::new("chr1", FeatureKind::Cds, 100, 200)
            .with_id("cds_1")
            .with_strand(Strand::Reverse)
            .with_phase(2);
        let s = format!("{}", f);
        assert!(s.contains("chr1"));
        assert!(s.contains("CDS"));
        assert!(s.contains("-"));
        assert!(s.contains("2"));
    }
}
