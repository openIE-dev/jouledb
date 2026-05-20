//! JouleDB Genomics Link
//!
//! HDC-powered Genomics and Bioinformatics module.
//! Provides O(1) sequence similarity, k-mer encoding, and variant detection.

pub use joule_db_hdc::{BinaryHV, BundleAccumulator};
use std::collections::HashMap;

pub const DIMENSION: usize = 10000;

// ============================================================================
// Core Types
// ============================================================================

/// DNA nucleotide bases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Base {
    A, // Adenine
    T, // Thymine
    G, // Guanine
    C, // Cytosine
    N, // Unknown
}

impl Base {
    pub fn from_char(c: char) -> Self {
        match c.to_ascii_uppercase() {
            'A' => Base::A,
            'T' => Base::T,
            'G' => Base::G,
            'C' => Base::C,
            _ => Base::N,
        }
    }

    pub fn complement(&self) -> Self {
        match self {
            Base::A => Base::T,
            Base::T => Base::A,
            Base::G => Base::C,
            Base::C => Base::G,
            Base::N => Base::N,
        }
    }
}

/// A DNA sequence
#[derive(Debug, Clone)]
pub struct DnaSequence {
    pub id: String,
    pub sequence: Vec<Base>,
    pub quality: Option<Vec<u8>>,
    pub description: Option<String>,
}

impl DnaSequence {
    pub fn from_string(id: &str, seq: &str) -> Self {
        Self {
            id: id.to_string(),
            sequence: seq.chars().map(Base::from_char).collect(),
            quality: None,
            description: None,
        }
    }

    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    pub fn reverse_complement(&self) -> Self {
        Self {
            id: format!("{}_rc", self.id),
            sequence: self.sequence.iter().rev().map(|b| b.complement()).collect(),
            quality: self
                .quality
                .as_ref()
                .map(|q| q.iter().rev().copied().collect()),
            description: self.description.clone(),
        }
    }
}

/// A variant (mutation)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Variant {
    pub chromosome: String,
    pub position: u64,
    pub reference: String,
    pub alternate: String,
    pub variant_type: VariantType,
    pub quality: f64,
    pub annotations: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum VariantType {
    Snp, // Single Nucleotide Polymorphism
    Insertion,
    Deletion,
    Mnp, // Multi-Nucleotide Polymorphism
    Complex,
}

/// A gene annotation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Gene {
    pub id: String,
    pub name: String,
    pub chromosome: String,
    pub start: u64,
    pub end: u64,
    pub strand: Strand,
    pub biotype: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Strand {
    Forward,
    Reverse,
}

/// A protein sequence
#[derive(Debug, Clone)]
pub struct ProteinSequence {
    pub id: String,
    pub sequence: String,
    pub gene_id: Option<String>,
}

// ============================================================================
// Genomics Link Encoder (macro-generated core)
// ============================================================================

joule_db_hdc::define_domain_module! {
    /// HDC encoder core for genomics domain data
    pub struct GenomicsLinkCore {
        seed: 0x6E00_01C5,
        dimension: 10000,
        fields: ["chromosome", "position", "reference", "alternate", "gene", "biotype"],
        scalars: ["position_base"],
        enums: {
            base_vectors: Base => [Base::A, Base::T, Base::G, Base::C, Base::N],
            variant_type_vectors: VariantType => [VariantType::Snp, VariantType::Insertion, VariantType::Deletion, VariantType::Mnp, VariantType::Complex],
            strand_vectors: Strand => [Strand::Forward, Strand::Reverse]
        },
    }
}

// ============================================================================
// GenomicsLink - public API wrapper with kmer_size support
// ============================================================================

/// VSA Encoder for genomics data
pub struct GenomicsLink {
    core: GenomicsLinkCore,
    kmer_size: usize,
}

impl GenomicsLink {
    pub fn new() -> Self {
        Self::with_kmer_size(21) // Default k-mer size
    }

    pub fn with_kmer_size(kmer_size: usize) -> Self {
        Self {
            core: GenomicsLinkCore::new(),
            kmer_size,
        }
    }

    /// Encode a k-mer into a hypervector
    /// Uses positional binding: H(kmer) = H(b0)*P^0 + H(b1)*P^1 + ... + H(bk)*P^k
    pub fn encode_kmer(&self, kmer: &[Base]) -> BinaryHV {
        let mut acc = BundleAccumulator::new(DIMENSION);

        for (pos, base) in kmer.iter().enumerate() {
            let base_vec = &self.core.base_vectors[base];
            // Positional binding using permutation
            let positioned =
                base_vec.bind(&self.core.scalar_bases["position_base"].permute_words(pos % 157));
            acc.add(&positioned);
        }

        acc.threshold()
    }

    /// Encode an entire DNA sequence as superposition of k-mers
    pub fn encode_sequence(&self, seq: &DnaSequence) -> BinaryHV {
        if seq.len() < self.kmer_size {
            // Sequence shorter than k-mer, encode directly
            return self.encode_kmer(&seq.sequence);
        }

        let mut acc = BundleAccumulator::new(DIMENSION);

        // Sliding window of k-mers
        for window in seq.sequence.windows(self.kmer_size) {
            acc.add(&self.encode_kmer(window));
        }

        acc.threshold()
    }

    /// Encode a variant
    pub fn encode_variant(&self, variant: &Variant) -> BinaryHV {
        let mut acc = BundleAccumulator::new(DIMENSION);

        // Chromosome
        let chr_hv = BinaryHV::from_hash(variant.chromosome.as_bytes(), DIMENSION);
        acc.add(&self.core.field_vectors["chromosome"].bind(&chr_hv));

        // Position (scaled)
        let pos_shift = (variant.position / 1000) as usize % 157; // 1kb resolution
        let pos_vec = self.core.scalar_bases["position_base"].permute_words(pos_shift);
        acc.add(&self.core.field_vectors["position"].bind(&pos_vec));

        // Variant type
        acc.add(&self.core.variant_type_vectors[&variant.variant_type]);

        // Reference allele
        let ref_hv = BinaryHV::from_hash(variant.reference.as_bytes(), DIMENSION);
        acc.add(&self.core.field_vectors["reference"].bind(&ref_hv));

        // Alternate allele
        let alt_hv = BinaryHV::from_hash(variant.alternate.as_bytes(), DIMENSION);
        acc.add(&self.core.field_vectors["alternate"].bind(&alt_hv));

        acc.threshold()
    }

    /// Encode a gene
    pub fn encode_gene(&self, gene: &Gene) -> BinaryHV {
        let mut acc = BundleAccumulator::new(DIMENSION);

        // Gene name/ID
        let name_hv = BinaryHV::from_hash(gene.name.as_bytes(), DIMENSION);
        acc.add(&self.core.field_vectors["gene"].bind(&name_hv));

        // Chromosome
        let chr_hv = BinaryHV::from_hash(gene.chromosome.as_bytes(), DIMENSION);
        acc.add(&self.core.field_vectors["chromosome"].bind(&chr_hv));

        // Strand
        acc.add(&self.core.strand_vectors[&gene.strand]);

        // Biotype
        let bio_hv = BinaryHV::from_hash(gene.biotype.as_bytes(), DIMENSION);
        acc.add(&self.core.field_vectors["biotype"].bind(&bio_hv));

        acc.threshold()
    }

    /// Encode a protein sequence
    pub fn encode_protein(&self, protein: &ProteinSequence) -> BinaryHV {
        let mut acc = BundleAccumulator::new(DIMENSION);

        // 3-mer sliding window for protein
        let seq_chars: Vec<char> = protein.sequence.chars().collect();

        if seq_chars.len() >= 3 {
            for window in seq_chars.windows(3) {
                let mut kmer_acc = BundleAccumulator::new(DIMENSION);
                for (pos, aa) in window.iter().enumerate() {
                    let aa_vec = BinaryHV::from_hash(&[*aa as u8], DIMENSION);
                    let positioned =
                        aa_vec.bind(&self.core.scalar_bases["position_base"].permute_words(pos));
                    kmer_acc.add(&positioned);
                }
                acc.add(&kmer_acc.threshold());
            }
        }

        acc.threshold()
    }

    /// Get k-mer size
    pub fn kmer_size(&self) -> usize {
        self.kmer_size
    }
}

impl Default for GenomicsLink {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Sequence Database
// ============================================================================

/// Holographic sequence database for similarity search
pub struct SequenceDb {
    /// Sequence vectors indexed by ID
    sequences: HashMap<String, BinaryHV>,
    /// Bundle of all sequences for quick membership test
    all_sequences: BundleAccumulator,
    /// Encoder
    encoder: GenomicsLink,
}

impl SequenceDb {
    pub fn new() -> Self {
        Self {
            sequences: HashMap::new(),
            all_sequences: BundleAccumulator::new(DIMENSION),
            encoder: GenomicsLink::new(),
        }
    }

    /// Add a sequence to the database
    pub fn add_sequence(&mut self, seq: &DnaSequence) {
        let hv = self.encoder.encode_sequence(seq);
        self.all_sequences.add(&hv);
        self.sequences.insert(seq.id.clone(), hv);
    }

    /// Find similar sequences (O(N) but each comparison is O(1))
    pub fn find_similar(&self, query: &DnaSequence, threshold: f32) -> Vec<(String, f32)> {
        let query_hv = self.encoder.encode_sequence(query);

        let mut matches: Vec<(String, f32)> = self
            .sequences
            .iter()
            .map(|(id, hv)| (id.clone(), query_hv.similarity(hv)))
            .filter(|(_, sim)| *sim > threshold)
            .collect();

        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        matches
    }

    /// Check if a sequence is similar to any in the database (O(1))
    pub fn contains_similar(&self, query: &DnaSequence, threshold: f32) -> bool {
        let query_hv = self.encoder.encode_sequence(query);
        let bundle_hv = self.all_sequences.threshold();
        query_hv.similarity(&bundle_hv) > threshold
    }

    /// Get sequence count
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }
}

impl Default for SequenceDb {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Variant Database
// ============================================================================

/// Holographic variant database
pub struct VariantDb {
    /// Variants by chromosome
    by_chromosome: HashMap<String, BundleAccumulator>,
    /// All variant vectors
    variants: HashMap<String, BinaryHV>,
    /// Encoder
    encoder: GenomicsLink,
}

impl VariantDb {
    pub fn new() -> Self {
        Self {
            by_chromosome: HashMap::new(),
            variants: HashMap::new(),
            encoder: GenomicsLink::new(),
        }
    }

    /// Add a variant
    pub fn add_variant(&mut self, variant: &Variant) {
        let hv = self.encoder.encode_variant(variant);

        let key = format!("{}:{}", variant.chromosome, variant.position);

        // Add to chromosome bundle
        let bundle = self
            .by_chromosome
            .entry(variant.chromosome.clone())
            .or_insert_with(|| BundleAccumulator::new(DIMENSION));
        bundle.add(&hv);

        self.variants.insert(key, hv);
    }

    /// Check if a variant exists (or similar)
    pub fn check_variant(&self, variant: &Variant) -> Option<f32> {
        let query_hv = self.encoder.encode_variant(variant);

        if let Some(bundle) = self.by_chromosome.get(&variant.chromosome) {
            let sim = query_hv.similarity(&bundle.threshold());
            if sim > 0.6 {
                return Some(sim);
            }
        }

        None
    }

    /// Get variant count
    pub fn len(&self) -> usize {
        self.variants.len()
    }

    pub fn is_empty(&self) -> bool {
        self.variants.is_empty()
    }
}

impl Default for VariantDb {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmer_encoding() {
        let link = GenomicsLink::with_kmer_size(5);

        let kmer = vec![Base::A, Base::T, Base::G, Base::C, Base::A];
        let hv = link.encode_kmer(&kmer);

        // Same k-mer should give same vector
        let hv2 = link.encode_kmer(&kmer);
        let sim = hv.similarity(&hv2);
        assert!(
            (sim - 1.0).abs() < 0.01,
            "Same kmer should have similarity 1.0"
        );
    }

    #[test]
    fn test_similar_sequences() {
        let link = GenomicsLink::with_kmer_size(7);

        // Two very similar sequences (differ by 1 base)
        let seq1 = DnaSequence::from_string("seq1", "ATGCATGCATGCATGCATGC");
        let seq2 = DnaSequence::from_string("seq2", "ATGCATGCTTGCATGCATGC"); // One T->T change

        let hv1 = link.encode_sequence(&seq1);
        let hv2 = link.encode_sequence(&seq2);

        let similarity = hv1.similarity(&hv2);
        println!("Similar sequence similarity: {}", similarity);
        assert!(
            similarity > 0.7,
            "Similar sequences should have high similarity"
        );
    }

    #[test]
    fn test_different_sequences() {
        let link = GenomicsLink::with_kmer_size(7);

        let seq1 = DnaSequence::from_string("seq1", "AAAAAAAAAAAAAAAAAAAA");
        let seq2 = DnaSequence::from_string("seq2", "TTTTTTTTTTTTTTTTTTTT");

        let hv1 = link.encode_sequence(&seq1);
        let hv2 = link.encode_sequence(&seq2);

        let similarity = hv1.similarity(&hv2);
        println!("Different sequence similarity: {}", similarity);
        assert!(
            similarity < 0.7,
            "Different sequences should have low similarity"
        );
    }

    #[test]
    fn test_reverse_complement() {
        let seq = DnaSequence::from_string("test", "ATGC");
        let rc = seq.reverse_complement();

        assert_eq!(rc.sequence, vec![Base::G, Base::C, Base::A, Base::T]);
    }

    #[test]
    fn test_sequence_db() {
        let mut db = SequenceDb::new();

        db.add_sequence(&DnaSequence::from_string(
            "gene1",
            "ATGCATGCATGCATGCATGCATGC",
        ));
        db.add_sequence(&DnaSequence::from_string(
            "gene2",
            "GCTAGCTAGCTAGCTAGCTAGCTA",
        ));
        db.add_sequence(&DnaSequence::from_string(
            "gene3",
            "ATGCATGCATGCATGCATGCATGC",
        )); // Same as gene1

        assert_eq!(db.len(), 3);

        // Find similar to gene1
        let query = DnaSequence::from_string("query", "ATGCATGCATGCATGCATGCATGC");
        let matches = db.find_similar(&query, 0.8);

        assert!(!matches.is_empty());
        println!("Matches: {:?}", matches);
    }

    #[test]
    fn test_variant_encoding() {
        let link = GenomicsLink::new();

        let snp = Variant {
            chromosome: "chr1".to_string(),
            position: 12345678,
            reference: "A".to_string(),
            alternate: "G".to_string(),
            variant_type: VariantType::Snp,
            quality: 99.0,
            annotations: vec![],
        };

        let hv = link.encode_variant(&snp);

        // Same variant should encode consistently
        let hv2 = link.encode_variant(&snp);
        assert!((hv.similarity(&hv2) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_protein_encoding() {
        let link = GenomicsLink::new();

        let protein = ProteinSequence {
            id: "protein1".to_string(),
            sequence: "MKWVTFISLLFLFSSAYS".to_string(),
            gene_id: None,
        };

        let hv = link.encode_protein(&protein);

        // Similar protein
        let protein2 = ProteinSequence {
            id: "protein2".to_string(),
            sequence: "MKWVTFISLLFLFSSAYS".to_string(), // Same
            gene_id: None,
        };

        let hv2 = link.encode_protein(&protein2);
        let sim = hv.similarity(&hv2);
        assert!((sim - 1.0).abs() < 0.01);
    }
}
