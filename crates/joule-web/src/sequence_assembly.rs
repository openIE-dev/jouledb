//! Sequence Assembly — overlap-layout-consensus (OLC), contig building,
//! scaffolding, overlap graph construction, greedy and graph-based assembly.
//!
//! Pure-Rust de novo sequence assembler implementing the overlap-layout-consensus
//! paradigm with configurable overlap detection, graph layout, and consensus
//! generation for short and medium read assembly.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum AssemblyError {
    InsufficientReads(usize),
    NoOverlaps,
    InvalidMinOverlap(usize),
    InvalidParameters(String),
}

impl fmt::Display for AssemblyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientReads(n) => write!(f, "insufficient reads: {n}"),
            Self::NoOverlaps => write!(f, "no overlaps found"),
            Self::InvalidMinOverlap(m) => write!(f, "invalid min overlap: {m}"),
            Self::InvalidParameters(s) => write!(f, "invalid parameters: {s}"),
        }
    }
}

impl std::error::Error for AssemblyError {}

// ── Overlap ─────────────────────────────────────────────────────

/// An overlap between two reads.
#[derive(Debug, Clone)]
pub struct Overlap {
    pub read_a: usize,
    pub read_b: usize,
    pub offset: usize,
    pub length: usize,
    pub identity: f64,
    pub is_suffix_prefix: bool,
}

impl Overlap {
    pub fn new(
        read_a: usize,
        read_b: usize,
        offset: usize,
        length: usize,
        identity: f64,
    ) -> Self {
        Self {
            read_a,
            read_b,
            offset,
            length,
            identity,
            is_suffix_prefix: true,
        }
    }
}

impl fmt::Display for Overlap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Overlap({}->{}, len={}, off={}, id={:.1}%)",
            self.read_a, self.read_b, self.length, self.offset, self.identity
        )
    }
}

// ── Contig ──────────────────────────────────────────────────────

/// An assembled contiguous sequence.
#[derive(Debug, Clone)]
pub struct Contig {
    pub id: usize,
    pub sequence: Vec<u8>,
    pub read_ids: Vec<usize>,
    pub coverage: f64,
}

impl Contig {
    pub fn new(id: usize, sequence: Vec<u8>) -> Self {
        Self { id, sequence, read_ids: Vec::new(), coverage: 0.0 }
    }

    pub fn with_reads(mut self, reads: Vec<usize>) -> Self {
        self.read_ids = reads;
        self
    }

    pub fn with_coverage(mut self, c: f64) -> Self {
        self.coverage = c;
        self
    }

    /// Length in base pairs.
    pub fn length(&self) -> usize {
        self.sequence.len()
    }

    /// Number of reads contributing.
    pub fn num_reads(&self) -> usize {
        self.read_ids.len()
    }
}

impl fmt::Display for Contig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Contig({}: {} bp, {} reads, {:.1}x cov)",
            self.id, self.length(), self.num_reads(), self.coverage
        )
    }
}

// ── Scaffold ────────────────────────────────────────────────────

/// A scaffold: ordered contigs with estimated gap sizes.
#[derive(Debug, Clone)]
pub struct Scaffold {
    pub id: usize,
    pub components: Vec<ScaffoldComponent>,
    pub total_length: usize,
}

/// A component within a scaffold.
#[derive(Debug, Clone)]
pub enum ScaffoldComponent {
    Contig { contig_id: usize, reversed: bool },
    Gap { estimated_size: usize },
}

impl Scaffold {
    pub fn new(id: usize) -> Self {
        Self { id, components: Vec::new(), total_length: 0 }
    }

    pub fn add_contig(&mut self, contig_id: usize, length: usize, reversed: bool) {
        self.components.push(ScaffoldComponent::Contig { contig_id, reversed });
        self.total_length += length;
    }

    pub fn add_gap(&mut self, size: usize) {
        self.components.push(ScaffoldComponent::Gap { estimated_size: size });
        self.total_length += size;
    }

    /// Number of contigs in the scaffold.
    pub fn num_contigs(&self) -> usize {
        self.components
            .iter()
            .filter(|c| matches!(c, ScaffoldComponent::Contig { .. }))
            .count()
    }
}

impl fmt::Display for Scaffold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Scaffold({}: {} bp, {} contigs)",
            self.id, self.total_length, self.num_contigs()
        )
    }
}

// ── Assembly Config ─────────────────────────────────────────────

/// Configuration for the assembler.
#[derive(Debug, Clone)]
pub struct AssemblyConfig {
    min_overlap: usize,
    min_identity: f64,
    max_overhang: usize,
    min_contig_length: usize,
    scaffold_gap_size: usize,
}

impl AssemblyConfig {
    pub fn new() -> Self {
        Self {
            min_overlap: 20,
            min_identity: 90.0,
            max_overhang: 5,
            min_contig_length: 50,
            scaffold_gap_size: 100,
        }
    }

    pub fn with_min_overlap(mut self, m: usize) -> Self { self.min_overlap = m; self }
    pub fn with_min_identity(mut self, i: f64) -> Self { self.min_identity = i; self }
    pub fn with_max_overhang(mut self, o: usize) -> Self { self.max_overhang = o; self }
    pub fn with_min_contig_length(mut self, l: usize) -> Self { self.min_contig_length = l; self }
    pub fn with_scaffold_gap_size(mut self, g: usize) -> Self { self.scaffold_gap_size = g; self }
}

impl Default for AssemblyConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for AssemblyConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AssemblyConfig(min_ovlp={}, min_id={:.0}%, min_ctg={})",
            self.min_overlap, self.min_identity, self.min_contig_length
        )
    }
}

// ── Assembly Result ─────────────────────────────────────────────

/// Result of a sequence assembly run.
#[derive(Debug, Clone)]
pub struct AssemblyResult {
    pub contigs: Vec<Contig>,
    pub scaffolds: Vec<Scaffold>,
    pub num_reads: usize,
    pub num_overlaps: usize,
    pub n50: usize,
    pub total_length: usize,
}

impl AssemblyResult {
    /// Largest contig length.
    pub fn max_contig_length(&self) -> usize {
        self.contigs.iter().map(|c| c.length()).max().unwrap_or(0)
    }

    /// Number of contigs.
    pub fn num_contigs(&self) -> usize {
        self.contigs.len()
    }
}

impl fmt::Display for AssemblyResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Assembly({} contigs, N50={}, total={} bp, {} reads, {} overlaps)",
            self.num_contigs(), self.n50, self.total_length,
            self.num_reads, self.num_overlaps
        )
    }
}

// ── Overlap Graph ───────────────────────────────────────────────

/// Directed overlap graph for OLC assembly.
struct OverlapGraph {
    edges: HashMap<usize, Vec<(usize, usize)>>, // node -> [(target, overlap_len)]
    in_degree: HashMap<usize, usize>,
}

impl OverlapGraph {
    fn new(num_reads: usize) -> Self {
        let mut in_degree = HashMap::new();
        for i in 0..num_reads {
            in_degree.insert(i, 0);
        }
        Self { edges: HashMap::new(), in_degree }
    }

    fn add_edge(&mut self, from: usize, to: usize, overlap_len: usize) {
        self.edges.entry(from).or_default().push((to, overlap_len));
        *self.in_degree.entry(to).or_insert(0) += 1;
    }

    /// Greedily traverse from sources to build linear paths.
    fn greedy_paths(&self) -> Vec<Vec<(usize, usize)>> {
        let mut visited = std::collections::HashSet::new();
        let mut paths = Vec::new();

        // Start from nodes with in_degree == 0.
        let mut sources: Vec<usize> = self
            .in_degree
            .iter()
            .filter(|&(_, &d)| d == 0)
            .map(|(&n, _)| n)
            .collect();
        sources.sort();

        for start in sources {
            if visited.contains(&start) {
                continue;
            }
            let mut path = Vec::new();
            let mut current = start;
            visited.insert(current);

            loop {
                if let Some(neighbors) = self.edges.get(&current) {
                    // Pick the neighbor with the longest overlap.
                    if let Some(&(next, ovlp)) = neighbors
                        .iter()
                        .filter(|(n, _)| !visited.contains(n))
                        .max_by_key(|&&(_, o)| o)
                    {
                        path.push((current, ovlp));
                        visited.insert(next);
                        current = next;
                    } else {
                        path.push((current, 0));
                        break;
                    }
                } else {
                    path.push((current, 0));
                    break;
                }
            }
            paths.push(path);
        }

        // Also handle isolated nodes.
        for (&node, _) in &self.in_degree {
            if !visited.contains(&node) {
                paths.push(vec![(node, 0)]);
            }
        }

        paths
    }
}

// ── Assembler ───────────────────────────────────────────────────

/// OLC sequence assembler.
#[derive(Debug, Clone)]
pub struct SequenceAssembler {
    config: AssemblyConfig,
}

impl SequenceAssembler {
    pub fn new(config: AssemblyConfig) -> Self {
        Self { config }
    }

    /// Compute suffix-prefix overlap between read_a and read_b.
    fn suffix_prefix_overlap(&self, a: &[u8], b: &[u8]) -> Option<Overlap> {
        let min_ovl = self.config.min_overlap;
        let max_ovl = a.len().min(b.len());

        let mut best_len = 0;
        for olen in min_ovl..=max_ovl {
            let suffix = &a[a.len() - olen..];
            let prefix = &b[..olen];
            let matches = suffix
                .iter()
                .zip(prefix.iter())
                .filter(|(sa, sb)| sa.to_ascii_uppercase() == sb.to_ascii_uppercase())
                .count();
            let identity = matches as f64 / olen as f64 * 100.0;
            if identity >= self.config.min_identity {
                best_len = olen;
            }
        }

        if best_len >= min_ovl {
            let identity = {
                let suffix = &a[a.len() - best_len..];
                let prefix = &b[..best_len];
                let matches = suffix
                    .iter()
                    .zip(prefix.iter())
                    .filter(|(sa, sb)| sa.to_ascii_uppercase() == sb.to_ascii_uppercase())
                    .count();
                matches as f64 / best_len as f64 * 100.0
            };
            Some(Overlap::new(0, 0, a.len() - best_len, best_len, identity))
        } else {
            None
        }
    }

    /// Assemble reads into contigs using OLC.
    pub fn assemble(&self, reads: &[&[u8]]) -> Result<AssemblyResult, AssemblyError> {
        if reads.len() < 2 {
            return Err(AssemblyError::InsufficientReads(reads.len()));
        }
        if self.config.min_overlap == 0 {
            return Err(AssemblyError::InvalidMinOverlap(0));
        }

        // Phase 1: Overlap detection.
        let mut overlaps = Vec::new();
        let mut graph = OverlapGraph::new(reads.len());

        for i in 0..reads.len() {
            for j in 0..reads.len() {
                if i == j {
                    continue;
                }
                if let Some(mut ovl) = self.suffix_prefix_overlap(reads[i], reads[j]) {
                    ovl.read_a = i;
                    ovl.read_b = j;
                    graph.add_edge(i, j, ovl.length);
                    overlaps.push(ovl);
                }
            }
        }

        if overlaps.is_empty() {
            return Err(AssemblyError::NoOverlaps);
        }

        // Phase 2: Layout — greedy path traversal.
        let paths = graph.greedy_paths();

        // Phase 3: Consensus — merge reads along each path.
        let mut contigs = Vec::new();
        for (cid, path) in paths.iter().enumerate() {
            if path.is_empty() {
                continue;
            }
            let mut seq = reads[path[0].0].to_vec();
            let mut read_ids = vec![path[0].0];

            for &(node, ovlp) in path.iter().skip(1) {
                if ovlp > 0 && ovlp < reads[node].len() {
                    seq.extend_from_slice(&reads[node][ovlp..]);
                } else {
                    seq.extend_from_slice(reads[node]);
                }
                read_ids.push(node);
            }

            if seq.len() >= self.config.min_contig_length {
                let cov = read_ids.len() as f64;
                contigs.push(
                    Contig::new(cid, seq).with_reads(read_ids).with_coverage(cov),
                );
            }
        }

        // Compute N50.
        let total_length: usize = contigs.iter().map(|c| c.length()).sum();
        let n50 = Self::compute_n50(&contigs, total_length);

        Ok(AssemblyResult {
            num_reads: reads.len(),
            num_overlaps: overlaps.len(),
            n50,
            total_length,
            scaffolds: Vec::new(),
            contigs,
        })
    }

    /// Compute the N50 statistic.
    fn compute_n50(contigs: &[Contig], total: usize) -> usize {
        let mut lengths: Vec<usize> = contigs.iter().map(|c| c.length()).collect();
        lengths.sort_unstable_by(|a, b| b.cmp(a));
        let half = total / 2;
        let mut cumulative = 0;
        for &l in &lengths {
            cumulative += l;
            if cumulative >= half {
                return l;
            }
        }
        lengths.first().copied().unwrap_or(0)
    }

    /// Build scaffolds from contigs using paired read information.
    pub fn scaffold(
        &self,
        contigs: &[Contig],
        links: &[(usize, usize, bool)], // (contig_a, contig_b, reverse_b)
    ) -> Vec<Scaffold> {
        let mut scaffolds = Vec::new();
        let mut used = std::collections::HashSet::new();

        for &(a, b, rev_b) in links {
            if used.contains(&a) || used.contains(&b) {
                continue;
            }
            let mut scaf = Scaffold::new(scaffolds.len());
            scaf.add_contig(a, contigs.get(a).map(|c| c.length()).unwrap_or(0), false);
            scaf.add_gap(self.config.scaffold_gap_size);
            scaf.add_contig(b, contigs.get(b).map(|c| c.length()).unwrap_or(0), rev_b);
            scaffolds.push(scaf);
            used.insert(a);
            used.insert(b);
        }

        scaffolds
    }
}

impl fmt::Display for SequenceAssembler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SequenceAssembler({})", self.config)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn overlapping_reads() -> Vec<Vec<u8>> {
        vec![
            b"ACGTACGTACGT".to_vec(),
            b"ACGTACGTTTTT".to_vec(),
            b"TTTTTGGGGGAA".to_vec(),
        ]
    }

    fn assembler() -> SequenceAssembler {
        SequenceAssembler::new(
            AssemblyConfig::new()
                .with_min_overlap(4)
                .with_min_identity(80.0)
                .with_min_contig_length(5),
        )
    }

    #[test]
    fn basic_assembly() {
        let reads = overlapping_reads();
        let refs: Vec<&[u8]> = reads.iter().map(|r| r.as_slice()).collect();
        let result = assembler().assemble(&refs).unwrap();
        assert!(!result.contigs.is_empty());
    }

    #[test]
    fn contig_length_valid() {
        let reads = overlapping_reads();
        let refs: Vec<&[u8]> = reads.iter().map(|r| r.as_slice()).collect();
        let result = assembler().assemble(&refs).unwrap();
        for c in &result.contigs {
            assert!(c.length() >= 5);
        }
    }

    #[test]
    fn insufficient_reads() {
        let result = assembler().assemble(&[b"ACGT".as_slice()]);
        assert!(matches!(result, Err(AssemblyError::InsufficientReads(_))));
    }

    #[test]
    fn no_overlaps_err() {
        let asm = SequenceAssembler::new(
            AssemblyConfig::new().with_min_overlap(100).with_min_contig_length(1),
        );
        let r1 = b"AAAA";
        let r2 = b"CCCC";
        let result = asm.assemble(&[r1.as_slice(), r2.as_slice()]);
        assert!(matches!(result, Err(AssemblyError::NoOverlaps)));
    }

    #[test]
    fn n50_computed() {
        let reads = overlapping_reads();
        let refs: Vec<&[u8]> = reads.iter().map(|r| r.as_slice()).collect();
        let result = assembler().assemble(&refs).unwrap();
        assert!(result.n50 > 0);
    }

    #[test]
    fn total_length_positive() {
        let reads = overlapping_reads();
        let refs: Vec<&[u8]> = reads.iter().map(|r| r.as_slice()).collect();
        let result = assembler().assemble(&refs).unwrap();
        assert!(result.total_length > 0);
    }

    #[test]
    fn overlap_display() {
        let o = Overlap::new(0, 1, 5, 10, 95.0);
        assert!(format!("{o}").contains("Overlap("));
    }

    #[test]
    fn contig_display() {
        let c = Contig::new(0, b"ACGT".to_vec()).with_coverage(3.0);
        assert!(format!("{c}").contains("Contig("));
    }

    #[test]
    fn scaffold_basic() {
        let c1 = Contig::new(0, b"ACGTACGT".to_vec());
        let c2 = Contig::new(1, b"TTTTGGGG".to_vec());
        let contigs = vec![c1, c2];
        let links = vec![(0, 1, false)];
        let scafs = assembler().scaffold(&contigs, &links);
        assert_eq!(scafs.len(), 1);
        assert_eq!(scafs[0].num_contigs(), 2);
    }

    #[test]
    fn scaffold_display() {
        let mut s = Scaffold::new(0);
        s.add_contig(0, 100, false);
        assert!(format!("{s}").contains("Scaffold("));
    }

    #[test]
    fn config_builder() {
        let cfg = AssemblyConfig::new()
            .with_min_overlap(30)
            .with_min_identity(95.0)
            .with_max_overhang(10)
            .with_min_contig_length(100)
            .with_scaffold_gap_size(200);
        assert!(format!("{cfg}").contains("min_ovlp=30"));
    }

    #[test]
    fn result_display() {
        let reads = overlapping_reads();
        let refs: Vec<&[u8]> = reads.iter().map(|r| r.as_slice()).collect();
        let result = assembler().assemble(&refs).unwrap();
        assert!(format!("{result}").contains("Assembly("));
    }

    #[test]
    fn max_contig_length() {
        let reads = overlapping_reads();
        let refs: Vec<&[u8]> = reads.iter().map(|r| r.as_slice()).collect();
        let result = assembler().assemble(&refs).unwrap();
        assert!(result.max_contig_length() > 0);
    }

    #[test]
    fn assembler_display() {
        assert!(format!("{}", assembler()).contains("SequenceAssembler"));
    }

    #[test]
    fn identical_reads_assembly() {
        let r = b"ACGTACGTACGTACGT";
        let result = assembler().assemble(&[r.as_slice(), r.as_slice()]).unwrap();
        assert!(!result.contigs.is_empty());
    }

    #[test]
    fn contig_with_reads() {
        let c = Contig::new(0, b"ACGT".to_vec()).with_reads(vec![0, 1, 2]);
        assert_eq!(c.num_reads(), 3);
    }

    #[test]
    fn scaffold_gap() {
        let mut s = Scaffold::new(0);
        s.add_contig(0, 50, false);
        s.add_gap(100);
        s.add_contig(1, 75, true);
        assert_eq!(s.total_length, 225);
    }

    #[test]
    fn invalid_min_overlap_zero() {
        let asm = SequenceAssembler::new(AssemblyConfig::new().with_min_overlap(0));
        let r = b"ACGT";
        assert!(asm.assemble(&[r.as_slice(), r.as_slice()]).is_err());
    }

    #[test]
    fn suffix_prefix_overlap_basic() {
        let asm = assembler();
        let ovl = asm.suffix_prefix_overlap(b"ACGTACGT", b"ACGTTTTT");
        assert!(ovl.is_some());
    }

    #[test]
    fn num_contigs_result() {
        let reads = overlapping_reads();
        let refs: Vec<&[u8]> = reads.iter().map(|r| r.as_slice()).collect();
        let result = assembler().assemble(&refs).unwrap();
        assert!(result.num_contigs() > 0);
    }
}
