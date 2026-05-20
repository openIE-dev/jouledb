//! Loop closure detection — place recognition and pose-graph correction.
//!
//! Bag-of-words descriptor matching, geometric consistency verification,
//! and integration with a pose graph for trajectory correction upon
//! detecting revisited locations.

use std::collections::HashMap;
use std::fmt;

// ── Feature descriptor ────────────────────────────────────────────

/// A visual feature descriptor (simplified fixed-length vector).
#[derive(Debug, Clone)]
pub struct Descriptor {
    pub data: Vec<f64>,
}

impl Descriptor {
    pub fn new(data: Vec<f64>) -> Self { Self { data } }

    pub fn zeros(dim: usize) -> Self { Self { data: vec![0.0; dim] } }

    pub fn dim(&self) -> usize { self.data.len() }

    /// L2 distance to another descriptor.
    pub fn distance(&self, other: &Descriptor) -> f64 {
        assert_eq!(self.data.len(), other.data.len());
        self.data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }

    /// Normalize to unit length.
    pub fn normalize(&mut self) {
        let norm: f64 = self.data.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for v in &mut self.data { *v /= norm; }
        }
    }
}

impl fmt::Display for Descriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Descriptor(dim={})", self.dim())
    }
}

// ── Visual word / vocabulary ──────────────────────────────────────

/// A visual word (cluster center) in a bag-of-words vocabulary.
#[derive(Debug, Clone)]
pub struct VisualWord {
    pub id: usize,
    pub center: Descriptor,
}

/// Bag-of-words vocabulary built from descriptor clusters.
#[derive(Debug, Clone)]
pub struct Vocabulary {
    pub words: Vec<VisualWord>,
}

impl Vocabulary {
    pub fn new(words: Vec<VisualWord>) -> Self { Self { words } }

    pub fn size(&self) -> usize { self.words.len() }

    /// Quantize a descriptor to the nearest visual word.
    pub fn quantize(&self, desc: &Descriptor) -> usize {
        let mut best_id = 0;
        let mut best_dist = f64::MAX;
        for w in &self.words {
            let d = w.center.distance(desc);
            if d < best_dist {
                best_dist = d;
                best_id = w.id;
            }
        }
        best_id
    }

    /// Build a simple vocabulary from descriptors using k-means.
    pub fn from_descriptors(descriptors: &[Descriptor], k: usize, max_iter: usize, seed: u64) -> Self {
        if descriptors.is_empty() || k == 0 {
            return Self { words: Vec::new() };
        }
        let dim = descriptors[0].dim();
        let mut rng_state = seed.wrapping_add(1);
        let mut next_rng = || -> u64 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            rng_state
        };

        // Initialize centers randomly
        let mut centers: Vec<Vec<f64>> = (0..k)
            .map(|_| {
                let idx = (next_rng() % descriptors.len() as u64) as usize;
                descriptors[idx].data.clone()
            })
            .collect();

        for _ in 0..max_iter {
            // Assign
            let mut assignments = vec![0usize; descriptors.len()];
            for (i, desc) in descriptors.iter().enumerate() {
                let mut best = 0;
                let mut best_d = f64::MAX;
                for (c, center) in centers.iter().enumerate() {
                    let d: f64 = desc.data.iter().zip(center).map(|(a, b)| (a - b) * (a - b)).sum();
                    if d < best_d { best_d = d; best = c; }
                }
                assignments[i] = best;
            }

            // Update centers
            let mut new_centers = vec![vec![0.0; dim]; k];
            let mut counts = vec![0usize; k];
            for (i, desc) in descriptors.iter().enumerate() {
                let c = assignments[i];
                counts[c] += 1;
                for (j, v) in desc.data.iter().enumerate() {
                    new_centers[c][j] += v;
                }
            }
            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..dim { new_centers[c][j] /= counts[c] as f64; }
                }
            }
            centers = new_centers;
        }

        let words = centers
            .into_iter()
            .enumerate()
            .map(|(id, center)| VisualWord { id, center: Descriptor::new(center) })
            .collect();
        Self { words }
    }
}

impl fmt::Display for Vocabulary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vocabulary(words={})", self.words.len())
    }
}

// ── Bag-of-words vector ───────────────────────────────────────────

/// Bag-of-words representation of a place/image.
#[derive(Debug, Clone)]
pub struct BowVector {
    pub frame_id: usize,
    /// Word ID → TF-IDF weight.
    pub weights: HashMap<usize, f64>,
}

impl BowVector {
    /// Build from descriptors + vocabulary (TF weighting).
    pub fn from_descriptors(frame_id: usize, descriptors: &[Descriptor], vocab: &Vocabulary) -> Self {
        let mut word_counts: HashMap<usize, usize> = HashMap::new();
        for desc in descriptors {
            let word = vocab.quantize(desc);
            *word_counts.entry(word).or_insert(0) += 1;
        }
        let n = descriptors.len() as f64;
        let weights: HashMap<usize, f64> = word_counts
            .into_iter()
            .map(|(word, count)| (word, count as f64 / n))
            .collect();
        Self { frame_id, weights }
    }

    /// L1-normalized similarity (0 = different, 1 = identical).
    pub fn similarity(&self, other: &BowVector) -> f64 {
        let mut score = 0.0;
        for (word, &w1) in &self.weights {
            if let Some(&w2) = other.weights.get(word) {
                score += w1.min(w2);
            }
        }
        score
    }

    /// L2 norm of the weight vector.
    pub fn norm(&self) -> f64 {
        self.weights.values().map(|v| v * v).sum::<f64>().sqrt()
    }
}

impl fmt::Display for BowVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BowVector(frame={}, words={})", self.frame_id, self.weights.len())
    }
}

// ── Loop closure candidate ────────────────────────────────────────

/// A candidate loop closure between two frames.
#[derive(Debug, Clone)]
pub struct LoopCandidate {
    pub query_frame: usize,
    pub match_frame: usize,
    pub score: f64,
    pub verified: bool,
}

impl fmt::Display for LoopCandidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LoopCandidate({} <-> {}, score={:.4}, verified={})",
            self.query_frame, self.match_frame, self.score, self.verified
        )
    }
}

// ── Loop closure detector ─────────────────────────────────────────

/// Configuration for loop closure detection.
#[derive(Debug, Clone)]
pub struct LoopClosureConfig {
    pub min_score: f64,
    pub min_frame_gap: usize,
    pub consistency_window: usize,
    pub consistency_threshold: usize,
    pub max_candidates: usize,
}

impl Default for LoopClosureConfig {
    fn default() -> Self {
        Self {
            min_score: 0.3,
            min_frame_gap: 20,
            consistency_window: 3,
            consistency_threshold: 2,
            max_candidates: 5,
        }
    }
}

impl LoopClosureConfig {
    pub fn new() -> Self { Self::default() }
    pub fn with_min_score(mut self, s: f64) -> Self { self.min_score = s; self }
    pub fn with_min_frame_gap(mut self, g: usize) -> Self { self.min_frame_gap = g; self }
    pub fn with_consistency_window(mut self, w: usize) -> Self { self.consistency_window = w; self }
    pub fn with_consistency_threshold(mut self, t: usize) -> Self { self.consistency_threshold = t; self }
    pub fn with_max_candidates(mut self, m: usize) -> Self { self.max_candidates = m; self }
}

/// Loop closure detector using bag-of-words.
#[derive(Debug, Clone)]
pub struct LoopClosureDetector {
    pub config: LoopClosureConfig,
    pub database: Vec<BowVector>,
    pub detections: Vec<LoopCandidate>,
}

impl LoopClosureDetector {
    pub fn new(config: LoopClosureConfig) -> Self {
        Self { config, database: Vec::new(), detections: Vec::new() }
    }

    /// Add a frame to the database.
    pub fn add_frame(&mut self, bow: BowVector) {
        self.database.push(bow);
    }

    /// Query for loop closures against the database.
    pub fn detect(&mut self, query: &BowVector) -> Vec<LoopCandidate> {
        let mut candidates: Vec<LoopCandidate> = Vec::new();

        for entry in &self.database {
            // Minimum temporal gap
            let gap = if query.frame_id > entry.frame_id {
                query.frame_id - entry.frame_id
            } else {
                entry.frame_id - query.frame_id
            };
            if gap < self.config.min_frame_gap { continue; }

            let score = query.similarity(entry);
            if score >= self.config.min_score {
                candidates.push(LoopCandidate {
                    query_frame: query.frame_id,
                    match_frame: entry.frame_id,
                    score,
                    verified: false,
                });
            }
        }

        // Sort by score descending
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(self.config.max_candidates);

        // Temporal consistency check
        for cand in &mut candidates {
            cand.verified = self.check_consistency(cand);
        }

        let verified: Vec<LoopCandidate> = candidates.iter().filter(|c| c.verified).cloned().collect();
        self.detections.extend(verified.iter().cloned());
        candidates
    }

    /// Check temporal consistency: do neighbors of the match also match?
    fn check_consistency(&self, candidate: &LoopCandidate) -> bool {
        let window = self.config.consistency_window;
        let mut consistent_count = 0;

        for prev in &self.detections {
            let frame_diff = if candidate.match_frame > prev.match_frame {
                candidate.match_frame - prev.match_frame
            } else {
                prev.match_frame - candidate.match_frame
            };
            if frame_diff <= window && frame_diff > 0 {
                consistent_count += 1;
            }
        }

        // First detection is always accepted, subsequent require consistency
        self.detections.is_empty() || consistent_count >= self.config.consistency_threshold
    }

    /// Number of verified loop closures.
    pub fn num_detections(&self) -> usize {
        self.detections.len()
    }

    /// Database size.
    pub fn database_size(&self) -> usize {
        self.database.len()
    }
}

impl fmt::Display for LoopClosureDetector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LoopClosureDetector(db={}, detections={})",
            self.database.len(),
            self.detections.len()
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_desc(vals: &[f64]) -> Descriptor {
        Descriptor::new(vals.to_vec())
    }

    fn simple_vocab() -> Vocabulary {
        Vocabulary::new(vec![
            VisualWord { id: 0, center: make_desc(&[1.0, 0.0, 0.0]) },
            VisualWord { id: 1, center: make_desc(&[0.0, 1.0, 0.0]) },
            VisualWord { id: 2, center: make_desc(&[0.0, 0.0, 1.0]) },
        ])
    }

    #[test]
    fn test_descriptor_distance() {
        let a = make_desc(&[1.0, 0.0, 0.0]);
        let b = make_desc(&[0.0, 1.0, 0.0]);
        let d = a.distance(&b);
        assert!((d - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_descriptor_normalize() {
        let mut d = make_desc(&[3.0, 4.0]);
        d.normalize();
        let norm: f64 = d.data.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_descriptor_display() {
        let d = make_desc(&[1.0, 2.0, 3.0]);
        assert!(format!("{}", d).contains("dim=3"));
    }

    #[test]
    fn test_vocabulary_quantize() {
        let vocab = simple_vocab();
        let d = make_desc(&[0.9, 0.1, 0.0]);
        assert_eq!(vocab.quantize(&d), 0);
    }

    #[test]
    fn test_vocabulary_size() {
        let vocab = simple_vocab();
        assert_eq!(vocab.size(), 3);
    }

    #[test]
    fn test_vocabulary_from_descriptors() {
        let descs: Vec<Descriptor> = (0..30)
            .map(|i| {
                let mut d = vec![0.0; 4];
                d[i % 3] = 1.0;
                d[3] = (i as f64) * 0.01;
                Descriptor::new(d)
            })
            .collect();
        let vocab = Vocabulary::from_descriptors(&descs, 3, 10, 42);
        assert_eq!(vocab.size(), 3);
    }

    #[test]
    fn test_vocabulary_display() {
        let vocab = simple_vocab();
        assert!(format!("{}", vocab).contains("words=3"));
    }

    #[test]
    fn test_bow_vector_creation() {
        let vocab = simple_vocab();
        let descs = vec![
            make_desc(&[1.0, 0.0, 0.0]),
            make_desc(&[1.0, 0.1, 0.0]),
            make_desc(&[0.0, 1.0, 0.0]),
        ];
        let bow = BowVector::from_descriptors(0, &descs, &vocab);
        assert_eq!(bow.weights.len(), 2); // words 0 and 1
    }

    #[test]
    fn test_bow_self_similarity() {
        let vocab = simple_vocab();
        let descs = vec![make_desc(&[1.0, 0.0, 0.0])];
        let bow = BowVector::from_descriptors(0, &descs, &vocab);
        let sim = bow.similarity(&bow);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bow_different_similarity() {
        let vocab = simple_vocab();
        let a = BowVector::from_descriptors(0, &[make_desc(&[1.0, 0.0, 0.0])], &vocab);
        let b = BowVector::from_descriptors(1, &[make_desc(&[0.0, 1.0, 0.0])], &vocab);
        let sim = a.similarity(&b);
        assert!(sim < 0.01);
    }

    #[test]
    fn test_bow_display() {
        let vocab = simple_vocab();
        let bow = BowVector::from_descriptors(5, &[make_desc(&[1.0, 0.0, 0.0])], &vocab);
        assert!(format!("{}", bow).contains("frame=5"));
    }

    #[test]
    fn test_loop_candidate_display() {
        let c = LoopCandidate { query_frame: 10, match_frame: 50, score: 0.85, verified: true };
        let s = format!("{}", c);
        assert!(s.contains("10"));
        assert!(s.contains("50"));
    }

    #[test]
    fn test_detector_creation() {
        let det = LoopClosureDetector::new(LoopClosureConfig::default());
        assert_eq!(det.database_size(), 0);
        assert_eq!(det.num_detections(), 0);
    }

    #[test]
    fn test_detector_add_frame() {
        let mut det = LoopClosureDetector::new(LoopClosureConfig::default());
        let vocab = simple_vocab();
        let bow = BowVector::from_descriptors(0, &[make_desc(&[1.0, 0.0, 0.0])], &vocab);
        det.add_frame(bow);
        assert_eq!(det.database_size(), 1);
    }

    #[test]
    fn test_detect_no_loop() {
        let mut det = LoopClosureDetector::new(LoopClosureConfig::new().with_min_frame_gap(5));
        let vocab = simple_vocab();
        // Add frames that are too close
        for i in 0..3 {
            det.add_frame(BowVector::from_descriptors(i, &[make_desc(&[1.0, 0.0, 0.0])], &vocab));
        }
        let query = BowVector::from_descriptors(4, &[make_desc(&[1.0, 0.0, 0.0])], &vocab);
        let cands = det.detect(&query);
        assert!(cands.is_empty());
    }

    #[test]
    fn test_detect_loop_found() {
        let mut det = LoopClosureDetector::new(
            LoopClosureConfig::new().with_min_frame_gap(5).with_min_score(0.5),
        );
        let vocab = simple_vocab();
        det.add_frame(BowVector::from_descriptors(0, &[make_desc(&[1.0, 0.0, 0.0])], &vocab));
        let query = BowVector::from_descriptors(100, &[make_desc(&[1.0, 0.0, 0.0])], &vocab);
        let cands = det.detect(&query);
        assert!(!cands.is_empty());
        assert!(cands[0].score > 0.5);
    }

    #[test]
    fn test_config_builder() {
        let cfg = LoopClosureConfig::new()
            .with_min_score(0.5)
            .with_min_frame_gap(30)
            .with_max_candidates(10);
        assert!((cfg.min_score - 0.5).abs() < 1e-10);
        assert_eq!(cfg.min_frame_gap, 30);
        assert_eq!(cfg.max_candidates, 10);
    }

    #[test]
    fn test_detector_display() {
        let det = LoopClosureDetector::new(LoopClosureConfig::default());
        let s = format!("{}", det);
        assert!(s.contains("LoopClosureDetector"));
    }

    #[test]
    fn test_bow_norm() {
        let vocab = simple_vocab();
        let bow = BowVector::from_descriptors(0, &[make_desc(&[1.0, 0.0, 0.0])], &vocab);
        assert!(bow.norm() > 0.0);
    }

    #[test]
    fn test_descriptor_zeros() {
        let d = Descriptor::zeros(10);
        assert_eq!(d.dim(), 10);
        assert!(d.data.iter().all(|v| *v == 0.0));
    }
}
