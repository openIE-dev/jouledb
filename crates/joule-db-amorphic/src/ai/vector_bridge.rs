//! FHRR ↔ BinaryHV Bridge
//!
//! Two hyperdimensional representations need to interoperate:
//! - **BinaryHV** (JouleDB): 10,000 binary bits, Hamming distance, XOR binding
//! - **FHRR/Phasor** (codegraph): 10,000 complex phases, cosine similarity, phase multiply
//!
//! This bridge enables JouleDB records to be queried against codegraph's FHRR index
//! and vice versa, preserving similarity ordering across representations.

use crate::BinaryHV;
use std::f32::consts::PI;

/// A phasor vector: each component is an angle θ ∈ [0, 2π) on the unit circle.
/// This is a lightweight representation — the full FhrrVector in codegraph
/// stores (re, im) pairs, but angles are sufficient for bridging.
#[derive(Clone, Debug)]
pub struct PhasorVector {
    pub angles: Vec<f32>,
}

impl PhasorVector {
    pub fn new(angles: Vec<f32>) -> Self {
        Self { angles }
    }

    pub fn dim(&self) -> usize {
        self.angles.len()
    }

    /// Cosine similarity in phasor space: mean(cos(θ_a - θ_b))
    pub fn similarity(&self, other: &Self) -> f32 {
        assert_eq!(self.angles.len(), other.angles.len());
        let sum: f32 = self
            .angles
            .iter()
            .zip(other.angles.iter())
            .map(|(a, b)| (a - b).cos())
            .sum();
        sum / self.angles.len() as f32
    }

    /// Bind: element-wise phase addition (complex multiply in angle space)
    pub fn bind(&self, other: &Self) -> Self {
        assert_eq!(self.angles.len(), other.angles.len());
        let angles = self
            .angles
            .iter()
            .zip(other.angles.iter())
            .map(|(a, b)| (a + b) % (2.0 * PI))
            .collect();
        Self { angles }
    }

    /// Unbind: element-wise phase subtraction (conjugate multiply)
    pub fn unbind(&self, key: &Self) -> Self {
        assert_eq!(self.angles.len(), key.angles.len());
        let angles = self
            .angles
            .iter()
            .zip(key.angles.iter())
            .map(|(a, b)| ((a - b) % (2.0 * PI) + 2.0 * PI) % (2.0 * PI))
            .collect();
        Self { angles }
    }

    /// Bundle: circular mean of angles (sum of unit vectors, take angle of result)
    pub fn bundle(vecs: &[&Self]) -> Self {
        if vecs.is_empty() {
            return Self {
                angles: Vec::new(),
            };
        }
        let dim = vecs[0].dim();
        let mut angles = Vec::with_capacity(dim);
        for i in 0..dim {
            let (mut sin_sum, mut cos_sum) = (0.0f32, 0.0f32);
            for v in vecs {
                sin_sum += v.angles[i].sin();
                cos_sum += v.angles[i].cos();
            }
            let angle = sin_sum.atan2(cos_sum);
            angles.push(if angle < 0.0 { angle + 2.0 * PI } else { angle });
        }
        Self { angles }
    }

    /// Convert from (re, im) pairs — the format used by FhrrVector in codegraph.
    pub fn from_complex_pairs(pairs: &[(f32, f32)]) -> Self {
        let angles = pairs.iter().map(|(re, im)| im.atan2(*re)).collect();
        Self { angles }
    }

    /// Convert to (re, im) pairs for codegraph compatibility.
    pub fn to_complex_pairs(&self) -> Vec<(f32, f32)> {
        self.angles
            .iter()
            .map(|a| (a.cos(), a.sin()))
            .collect()
    }
}

/// Convert BinaryHV → PhasorVector.
///
/// Binary bit → phase: 0 → 0°, 1 → 180° (π).
/// This maps the binary {0, 1} domain to the phasor {0, π} domain,
/// preserving similarity: Hamming distance ∝ phase disagreement.
pub fn binaryhv_to_phasor(hv: &BinaryHV) -> PhasorVector {
    let dim = hv.dimension();
    let words = hv.as_words();
    let mut angles = Vec::with_capacity(dim);
    for i in 0..dim {
        let word_idx = i / 64;
        let bit_idx = i % 64;
        let bit = (words[word_idx] >> bit_idx) & 1;
        angles.push(if bit == 1 { PI } else { 0.0 });
    }
    PhasorVector { angles }
}

/// Convert PhasorVector → BinaryHV.
///
/// Threshold phase at π/2 (90°): angles in [π/2, 3π/2) → 1, else → 0.
/// This corresponds to thresholding the real part: cos(θ) < 0 → 1.
pub fn phasor_to_binaryhv(pv: &PhasorVector) -> BinaryHV {
    let dim = pv.dim();
    let num_words = (dim + 63) / 64;
    let mut words = vec![0u64; num_words];
    for (i, angle) in pv.angles.iter().enumerate() {
        // cos(θ) < 0 means θ ∈ (π/2, 3π/2) — set bit to 1
        if angle.cos() < 0.0 {
            words[i / 64] |= 1u64 << (i % 64);
        }
    }
    BinaryHV::from_words(words, dim)
}

/// Cross-format similarity: compare a BinaryHV against a PhasorVector
/// without full conversion. Uses the cosine-of-phase-difference formula
/// but with the binary vector's implied phases (0 or π).
///
/// Result is in [-1, 1] (phasor scale). To convert to BinaryHV scale [0, 1]:
/// `(result + 1.0) / 2.0`
pub fn cross_similarity(hv: &BinaryHV, pv: &PhasorVector) -> f32 {
    assert_eq!(hv.dimension(), pv.dim());
    let words = hv.as_words();
    let dim = hv.dimension();
    let mut sum = 0.0f32;
    for i in 0..dim {
        let bit = (words[i / 64] >> (i % 64)) & 1;
        // cos(bit_phase - phasor_angle) where bit_phase is 0 or π
        if bit == 0 {
            sum += pv.angles[i].cos(); // cos(0 - θ) = cos(θ)
        } else {
            sum -= pv.angles[i].cos(); // cos(π - θ) = -cos(θ)
        }
    }
    sum / dim as f32
}

/// Batch convert BinaryHV records to PhasorVectors for FHRR index queries.
pub fn batch_to_phasor(hvs: &[BinaryHV]) -> Vec<PhasorVector> {
    hvs.iter().map(binaryhv_to_phasor).collect()
}

/// Batch convert PhasorVectors to BinaryHVs for amorphic store ingestion.
pub fn batch_to_binaryhv(pvs: &[PhasorVector]) -> Vec<BinaryHV> {
    pvs.iter().map(phasor_to_binaryhv).collect()
}

/// Similarity-preserving roundtrip quality metric.
/// Measures how well similarity ordering is preserved across BinaryHV → Phasor → BinaryHV.
/// Returns (correlation, max_distortion) where correlation should be ≥ 0.95 and
/// max_distortion should be ≤ 0.05 for the bridge to be useful.
pub fn roundtrip_quality(originals: &[BinaryHV]) -> (f32, f32) {
    if originals.len() < 2 {
        return (1.0, 0.0);
    }
    let mut original_sims = Vec::new();
    let mut roundtrip_sims = Vec::new();
    let roundtripped: Vec<BinaryHV> = originals
        .iter()
        .map(|hv| phasor_to_binaryhv(&binaryhv_to_phasor(hv)))
        .collect();

    for i in 0..originals.len() {
        for j in (i + 1)..originals.len() {
            original_sims.push(originals[i].similarity(&originals[j]));
            roundtrip_sims.push(roundtripped[i].similarity(&roundtripped[j]));
        }
    }

    // Pearson correlation
    let n = original_sims.len() as f32;
    let mean_o: f32 = original_sims.iter().sum::<f32>() / n;
    let mean_r: f32 = roundtrip_sims.iter().sum::<f32>() / n;
    let mut cov = 0.0f32;
    let mut var_o = 0.0f32;
    let mut var_r = 0.0f32;
    for i in 0..original_sims.len() {
        let do_ = original_sims[i] - mean_o;
        let dr = roundtrip_sims[i] - mean_r;
        cov += do_ * dr;
        var_o += do_ * do_;
        var_r += dr * dr;
    }
    let correlation = if var_o > 0.0 && var_r > 0.0 {
        cov / (var_o.sqrt() * var_r.sqrt())
    } else {
        1.0
    };

    let max_distortion = original_sims
        .iter()
        .zip(roundtrip_sims.iter())
        .map(|(o, r)| (o - r).abs())
        .fold(0.0f32, f32::max);

    (correlation, max_distortion)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binaryhv_to_phasor_zeros() {
        let hv = BinaryHV::zeros(64);
        let pv = binaryhv_to_phasor(&hv);
        assert_eq!(pv.dim(), 64);
        for angle in &pv.angles {
            assert!((*angle - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_roundtrip_identity() {
        // All zeros: 0 → phase 0 → cos(0)=1 > 0 → bit 0 ✓
        let hv = BinaryHV::zeros(128);
        let rt = phasor_to_binaryhv(&binaryhv_to_phasor(&hv));
        assert_eq!(hv.hamming_distance(&rt), 0);
    }

    #[test]
    fn test_roundtrip_random() {
        // Random vector should survive roundtrip perfectly
        // (binary phases are exactly 0 or π, thresholding is exact)
        let hv = BinaryHV::random(1024, 42);
        let rt = phasor_to_binaryhv(&binaryhv_to_phasor(&hv));
        assert_eq!(hv.hamming_distance(&rt), 0);
    }

    #[test]
    fn test_cross_similarity_identical() {
        let hv = BinaryHV::random(256, 99);
        let pv = binaryhv_to_phasor(&hv);
        let sim = cross_similarity(&hv, &pv);
        assert!((sim - 1.0).abs() < 1e-5, "self-similarity should be 1.0, got {sim}");
    }

    #[test]
    fn test_cross_similarity_orthogonal() {
        // Two random vectors should have similarity near 0
        let hv = BinaryHV::random(10000, 1);
        let other = BinaryHV::random(10000, 2);
        let pv = binaryhv_to_phasor(&other);
        let sim = cross_similarity(&hv, &pv);
        assert!(sim.abs() < 0.1, "random vectors should be near-orthogonal, got {sim}");
    }

    #[test]
    fn test_phasor_similarity() {
        let a = PhasorVector::new(vec![0.0, PI, 0.0, PI]);
        let b = PhasorVector::new(vec![0.0, PI, 0.0, PI]);
        assert!((a.similarity(&b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_phasor_bind_unbind() {
        let a = PhasorVector::new(vec![0.5, 1.0, 1.5, 2.0]);
        let key = PhasorVector::new(vec![0.1, 0.2, 0.3, 0.4]);
        let bound = a.bind(&key);
        let unbound = bound.unbind(&key);
        for (orig, recovered) in a.angles.iter().zip(unbound.angles.iter()) {
            let diff = (orig - recovered).abs();
            assert!(diff < 1e-5 || (diff - 2.0 * PI).abs() < 1e-5);
        }
    }

    #[test]
    fn test_roundtrip_quality_perfect() {
        let hvs: Vec<BinaryHV> = (0..10).map(|s| BinaryHV::random(512, s)).collect();
        let (corr, dist) = roundtrip_quality(&hvs);
        assert!(
            (corr - 1.0).abs() < 1e-5,
            "binary roundtrip should be perfect, correlation={corr}"
        );
        assert!(
            dist < 1e-5,
            "binary roundtrip should have zero distortion, got {dist}"
        );
    }

    #[test]
    fn test_from_complex_pairs() {
        let pairs = vec![(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
        let pv = PhasorVector::from_complex_pairs(&pairs);
        assert!((pv.angles[0] - 0.0).abs() < 1e-5); // (1, 0) → 0
        assert!((pv.angles[1] - PI / 2.0).abs() < 1e-5); // (0, 1) → π/2
        assert!((pv.angles[2].abs() - PI).abs() < 1e-5); // (-1, 0) → ±π
    }
}
