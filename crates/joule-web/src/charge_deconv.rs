//! Charge state deconvolution, envelope detection, and mass reconstruction.
//!
//! Determines the charge state of multiply-charged ions from
//! isotope spacing, deconvolves raw m/z spectra to neutral mass
//! spectra, detects charge-state envelopes, and merges multiple
//! charge states of the same species.

use std::fmt;

// ── Constants ───────────────────────────────────────────────────

/// Proton mass in Daltons.
const PROTON_MASS: f64 = 1.007_276_47;

/// ¹³C spacing in Daltons.
const C13_SPACING: f64 = 1.003_355;

// ── ChargeState ─────────────────────────────────────────────────

/// A determined charge state with its associated confidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChargeState {
    pub charge: u8,
    pub confidence: f64,
}

impl ChargeState {
    pub fn new(charge: u8, confidence: f64) -> Self {
        Self { charge, confidence }
    }
}

impl fmt::Display for ChargeState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "z={}+ ({:.1}%)", self.charge, self.confidence * 100.0)
    }
}

// ── MzPeak ──────────────────────────────────────────────────────

/// A peak in the raw m/z spectrum used for deconvolution input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MzPeak {
    pub mz: f64,
    pub intensity: f64,
}

impl MzPeak {
    pub fn new(mz: f64, intensity: f64) -> Self {
        Self { mz, intensity }
    }

    /// Convert this m/z to neutral mass given a charge.
    pub fn to_neutral_mass(&self, charge: u8) -> f64 {
        self.mz * charge as f64 - charge as f64 * PROTON_MASS
    }
}

impl fmt::Display for MzPeak {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "m/z={:.4} I={:.1}", self.mz, self.intensity)
    }
}

// ── DeconvConfig ────────────────────────────────────────────────

/// Configuration for charge-state deconvolution.
#[derive(Debug, Clone)]
pub struct DeconvConfig {
    pub min_charge: u8,
    pub max_charge: u8,
    pub mz_tolerance_da: f64,
    pub min_isotope_peaks: usize,
    pub min_intensity: f64,
    pub merge_tolerance_da: f64,
}

impl DeconvConfig {
    pub fn new() -> Self {
        Self {
            min_charge: 1,
            max_charge: 6,
            mz_tolerance_da: 0.02,
            min_isotope_peaks: 3,
            min_intensity: 100.0,
            merge_tolerance_da: 0.5,
        }
    }

    pub fn with_charge_range(mut self, lo: u8, hi: u8) -> Self {
        self.min_charge = lo.max(1);
        self.max_charge = hi.max(lo);
        self
    }

    pub fn with_mz_tolerance(mut self, tol: f64) -> Self {
        self.mz_tolerance_da = tol;
        self
    }

    pub fn with_min_isotope_peaks(mut self, n: usize) -> Self {
        self.min_isotope_peaks = n.max(2);
        self
    }

    pub fn with_min_intensity(mut self, i: f64) -> Self {
        self.min_intensity = i;
        self
    }

    pub fn with_merge_tolerance(mut self, tol: f64) -> Self {
        self.merge_tolerance_da = tol;
        self
    }
}

impl Default for DeconvConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for DeconvConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Deconv(z={}-{}+, tol={:.3}Da, min_peaks={})",
            self.min_charge, self.max_charge, self.mz_tolerance_da, self.min_isotope_peaks,
        )
    }
}

// ── IsotopeEnvelope ─────────────────────────────────────────────

/// A detected isotope envelope at a specific charge state.
#[derive(Debug, Clone)]
pub struct IsotopeEnvelope {
    pub peaks: Vec<MzPeak>,
    pub charge: u8,
    pub monoisotopic_mz: f64,
    pub neutral_mass: f64,
    pub total_intensity: f64,
}

impl IsotopeEnvelope {
    pub fn new(peaks: Vec<MzPeak>, charge: u8) -> Self {
        let monoisotopic_mz = peaks.first().map(|p| p.mz).unwrap_or(0.0);
        let neutral_mass = monoisotopic_mz * charge as f64 - charge as f64 * PROTON_MASS;
        let total_intensity = peaks.iter().map(|p| p.intensity).sum();
        Self { peaks, charge, monoisotopic_mz, neutral_mass, total_intensity }
    }

    /// Number of isotope peaks in this envelope.
    pub fn len(&self) -> usize {
        self.peaks.len()
    }

    /// Whether the envelope has no peaks.
    pub fn is_empty(&self) -> bool {
        self.peaks.is_empty()
    }
}

impl fmt::Display for IsotopeEnvelope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Envelope z={}+ mono_mz={:.4} M={:.2} peaks={} I_sum={:.0}",
            self.charge,
            self.monoisotopic_mz,
            self.neutral_mass,
            self.peaks.len(),
            self.total_intensity,
        )
    }
}

// ── DeconvolvedMass ─────────────────────────────────────────────

/// A neutral mass after deconvolution, possibly merged from
/// multiple charge states.
#[derive(Debug, Clone)]
pub struct DeconvolvedMass {
    pub neutral_mass: f64,
    pub total_intensity: f64,
    pub charge_states: Vec<u8>,
    pub envelope_count: usize,
}

impl DeconvolvedMass {
    pub fn new(mass: f64, intensity: f64, charge: u8) -> Self {
        Self {
            neutral_mass: mass,
            total_intensity: intensity,
            charge_states: vec![charge],
            envelope_count: 1,
        }
    }

    /// Merge another observation of the same species.
    pub fn merge(&mut self, other: &DeconvolvedMass) {
        self.total_intensity += other.total_intensity;
        for &z in &other.charge_states {
            if !self.charge_states.contains(&z) {
                self.charge_states.push(z);
            }
        }
        self.envelope_count += other.envelope_count;
        // Weighted average of masses.
        let w1 = self.total_intensity - other.total_intensity;
        let w2 = other.total_intensity;
        let total_w = w1 + w2;
        if total_w > 0.0 {
            self.neutral_mass = (self.neutral_mass * w1 + other.neutral_mass * w2) / total_w;
        }
    }
}

impl fmt::Display for DeconvolvedMass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let charges: Vec<String> = self.charge_states.iter().map(|z| format!("{}+", z)).collect();
        write!(
            f,
            "M={:.2} I={:.0} charges=[{}] n={}",
            self.neutral_mass,
            self.total_intensity,
            charges.join(","),
            self.envelope_count,
        )
    }
}

// ── Charge determination ────────────────────────────────────────

/// Determine the charge state from the spacing between adjacent
/// isotope peaks in m/z.
pub fn determine_charge_from_spacing(spacing: f64, max_charge: u8) -> Option<ChargeState> {
    if spacing <= 0.0 {
        return None;
    }
    let mut best_charge = 0u8;
    let mut best_error = f64::INFINITY;

    for z in 1..=max_charge {
        let expected = C13_SPACING / z as f64;
        let error = (spacing - expected).abs();
        if error < best_error {
            best_error = error;
            best_charge = z;
        }
    }

    if best_charge == 0 {
        return None;
    }

    let expected = C13_SPACING / best_charge as f64;
    let confidence = 1.0 - (best_error / expected).min(1.0);
    Some(ChargeState::new(best_charge, confidence))
}

/// Convert m/z to neutral mass.
pub fn mz_to_neutral_mass(mz: f64, charge: u8) -> f64 {
    mz * charge as f64 - charge as f64 * PROTON_MASS
}

/// Convert neutral mass to m/z.
pub fn neutral_mass_to_mz(mass: f64, charge: u8) -> f64 {
    (mass + charge as f64 * PROTON_MASS) / charge as f64
}

// ── Envelope detection ──────────────────────────────────────────

/// Detect isotope envelopes in sorted peak list.
///
/// For each candidate peak and each charge state, look for
/// successive peaks spaced by C13_SPACING/z.
pub fn detect_envelopes(peaks: &[MzPeak], config: &DeconvConfig) -> Vec<IsotopeEnvelope> {
    let mut envelopes = Vec::new();
    let mut used = vec![false; peaks.len()];

    for z in (config.min_charge..=config.max_charge).rev() {
        let spacing = C13_SPACING / z as f64;

        for i in 0..peaks.len() {
            if used[i] || peaks[i].intensity < config.min_intensity {
                continue;
            }

            let mut env_peaks = vec![peaks[i]];
            let mut last_mz = peaks[i].mz;
            let mut env_indices = vec![i];

            // Search forward for successive isotope peaks.
            for j in (i + 1)..peaks.len() {
                if used[j] {
                    continue;
                }
                let expected_mz = last_mz + spacing;
                if (peaks[j].mz - expected_mz).abs() <= config.mz_tolerance_da {
                    env_peaks.push(peaks[j]);
                    env_indices.push(j);
                    last_mz = peaks[j].mz;
                } else if peaks[j].mz > expected_mz + config.mz_tolerance_da {
                    break;
                }
            }

            if env_peaks.len() >= config.min_isotope_peaks {
                for &idx in &env_indices {
                    used[idx] = true;
                }
                envelopes.push(IsotopeEnvelope::new(env_peaks, z));
            }
        }
    }

    envelopes
}

// ── Mass merging ────────────────────────────────────────────────

/// Merge envelopes with compatible neutral masses.
pub fn merge_envelopes(envelopes: &[IsotopeEnvelope], tolerance: f64) -> Vec<DeconvolvedMass> {
    let mut masses: Vec<DeconvolvedMass> = Vec::new();

    for env in envelopes {
        let mut merged = false;
        for dm in &mut masses {
            if (dm.neutral_mass - env.neutral_mass).abs() <= tolerance {
                let other = DeconvolvedMass::new(env.neutral_mass, env.total_intensity, env.charge);
                dm.merge(&other);
                merged = true;
                break;
            }
        }
        if !merged {
            masses.push(DeconvolvedMass::new(
                env.neutral_mass,
                env.total_intensity,
                env.charge,
            ));
        }
    }

    masses.sort_by(|a, b| a.neutral_mass.partial_cmp(&b.neutral_mass).unwrap());
    masses
}

/// Full deconvolution pipeline: detect envelopes and merge.
pub fn deconvolve(peaks: &[MzPeak], config: &DeconvConfig) -> Vec<DeconvolvedMass> {
    let envelopes = detect_envelopes(peaks, config);
    merge_envelopes(&envelopes, config.merge_tolerance_da)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_charge_from_spacing_z1() {
        let cs = determine_charge_from_spacing(1.003, 4).unwrap();
        assert_eq!(cs.charge, 1);
        assert!(cs.confidence > 0.99);
    }

    #[test]
    fn test_charge_from_spacing_z2() {
        let cs = determine_charge_from_spacing(0.5017, 4).unwrap();
        assert_eq!(cs.charge, 2);
        assert!(cs.confidence > 0.95);
    }

    #[test]
    fn test_charge_from_spacing_z3() {
        let spacing = C13_SPACING / 3.0;
        let cs = determine_charge_from_spacing(spacing, 6).unwrap();
        assert_eq!(cs.charge, 3);
    }

    #[test]
    fn test_charge_from_spacing_zero() {
        assert!(determine_charge_from_spacing(0.0, 4).is_none());
    }

    #[test]
    fn test_mz_to_neutral() {
        let mz = 500.5;
        let mass = mz_to_neutral_mass(mz, 2);
        let expected = mz * 2.0 - 2.0 * PROTON_MASS;
        assert!((mass - expected).abs() < 1e-6);
    }

    #[test]
    fn test_neutral_to_mz_roundtrip() {
        let mass = 1000.0;
        let mz = neutral_mass_to_mz(mass, 3);
        let back = mz_to_neutral_mass(mz, 3);
        assert!((back - mass).abs() < 1e-9);
    }

    #[test]
    fn test_detect_z1_envelope() {
        let peaks = vec![
            MzPeak::new(500.0, 1000.0),
            MzPeak::new(501.003, 800.0),
            MzPeak::new(502.007, 400.0),
            MzPeak::new(503.010, 150.0),
        ];
        let cfg = DeconvConfig::new()
            .with_charge_range(1, 4)
            .with_mz_tolerance(0.02)
            .with_min_isotope_peaks(3);
        let envs = detect_envelopes(&peaks, &cfg);
        assert_eq!(envs.len(), 1);
        assert_eq!(envs[0].charge, 1);
        assert!(envs[0].peaks.len() >= 3);
    }

    #[test]
    fn test_detect_z2_envelope() {
        let spacing = C13_SPACING / 2.0;
        let peaks = vec![
            MzPeak::new(500.0, 1000.0),
            MzPeak::new(500.0 + spacing, 900.0),
            MzPeak::new(500.0 + 2.0 * spacing, 600.0),
            MzPeak::new(500.0 + 3.0 * spacing, 300.0),
        ];
        let cfg = DeconvConfig::new()
            .with_charge_range(1, 4)
            .with_min_isotope_peaks(3);
        let envs = detect_envelopes(&peaks, &cfg);
        assert!(!envs.is_empty());
        assert!(envs.iter().any(|e| e.charge == 2));
    }

    #[test]
    fn test_envelope_neutral_mass() {
        let peaks = vec![
            MzPeak::new(500.0, 1000.0),
            MzPeak::new(501.003, 800.0),
            MzPeak::new(502.007, 400.0),
        ];
        let env = IsotopeEnvelope::new(peaks, 1);
        let expected = 500.0 - PROTON_MASS;
        assert!((env.neutral_mass - expected).abs() < 0.01);
    }

    #[test]
    fn test_merge_same_species() {
        let env1 = IsotopeEnvelope::new(
            vec![MzPeak::new(500.0, 1000.0), MzPeak::new(501.003, 800.0), MzPeak::new(502.007, 400.0)],
            1,
        );
        let env2 = IsotopeEnvelope::new(
            vec![MzPeak::new(250.5, 600.0), MzPeak::new(251.0, 500.0), MzPeak::new(251.5, 300.0)],
            2,
        );
        let merged = merge_envelopes(&[env1, env2], 1.0);
        // Their neutral masses are within tolerance (~0.007 Da apart), so should be merged.
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].charge_states.len(), 2);
    }

    #[test]
    fn test_deconvolve_pipeline() {
        let peaks = vec![
            MzPeak::new(400.0, 500.0),
            MzPeak::new(401.003, 400.0),
            MzPeak::new(402.007, 200.0),
        ];
        let cfg = DeconvConfig::new().with_min_isotope_peaks(3);
        let masses = deconvolve(&peaks, &cfg);
        assert!(!masses.is_empty());
    }

    #[test]
    fn test_mz_peak_display() {
        let p = MzPeak::new(500.1234, 999.0);
        let d = format!("{}", p);
        assert!(d.contains("500.1234"));
    }

    #[test]
    fn test_charge_state_display() {
        let cs = ChargeState::new(3, 0.95);
        let d = format!("{}", cs);
        assert!(d.contains("3+"));
        assert!(d.contains("95.0%"));
    }

    #[test]
    fn test_envelope_display() {
        let env = IsotopeEnvelope::new(
            vec![MzPeak::new(500.0, 1000.0), MzPeak::new(501.0, 500.0), MzPeak::new(502.0, 200.0)],
            2,
        );
        let d = format!("{}", env);
        assert!(d.contains("z=2+"));
    }

    #[test]
    fn test_deconvolved_mass_display() {
        let dm = DeconvolvedMass::new(999.5, 5000.0, 2);
        let d = format!("{}", dm);
        assert!(d.contains("999.5"));
        assert!(d.contains("2+"));
    }

    #[test]
    fn test_config_display() {
        let cfg = DeconvConfig::new();
        let d = format!("{}", cfg);
        assert!(d.contains("Deconv"));
    }

    #[test]
    fn test_min_isotope_peaks_filter() {
        let peaks = vec![
            MzPeak::new(500.0, 1000.0),
            MzPeak::new(501.003, 800.0),
        ];
        let cfg = DeconvConfig::new().with_min_isotope_peaks(3);
        let envs = detect_envelopes(&peaks, &cfg);
        assert!(envs.is_empty());
    }

    #[test]
    fn test_intensity_filter() {
        let peaks = vec![
            MzPeak::new(500.0, 50.0),
            MzPeak::new(501.003, 40.0),
            MzPeak::new(502.007, 30.0),
        ];
        let cfg = DeconvConfig::new().with_min_intensity(100.0);
        let envs = detect_envelopes(&peaks, &cfg);
        assert!(envs.is_empty());
    }
}
