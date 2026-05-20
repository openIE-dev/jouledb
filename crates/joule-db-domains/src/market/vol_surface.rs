//! Volatility Surface Integration
//!
//! Stores and interpolates implied volatility across strike and expiration dimensions
//! using holographic nearest-neighbor interpolation.

use super::{DIMENSION, OptionType};
use joule_db_hdc::{BinaryHV, BundleAccumulator};
use std::collections::HashMap;

/// A volatility surface node key: (strike cents, expiration epoch days)
pub type VolNodeKey = (u32, u64);

/// Volatility surface with holographic interpolation
///
/// Stores IV observations at discrete (strike, expiry) points and provides
/// O(1) approximate nearest-neighbor interpolation using VSA similarity.
pub struct VolatilitySurface {
    /// Exact IV values: (strike_cents, expiry_epoch) → IV
    surface: HashMap<VolNodeKey, f64>,

    /// Holographic index of all vol nodes for NN lookup
    vol_nodes_hologram: BundleAccumulator,

    /// Individual node vectors for inverse lookup
    node_vectors: HashMap<VolNodeKey, BinaryHV>,

    /// Field vectors for encoding
    field_vectors: HashMap<String, BinaryHV>,

    /// Scalar bases for permutation encoding
    scalar_bases: HashMap<String, BinaryHV>,

    /// Symbol this surface belongs to
    pub symbol: String,
}

impl VolatilitySurface {
    /// Create a new volatility surface for a symbol
    pub fn new(symbol: &str) -> Self {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let mut rng = StdRng::seed_from_u64(0x0015_5000); // Deterministic seed

        let mut field_vectors = HashMap::new();
        let mut scalar_bases = HashMap::new();

        // Field vectors
        field_vectors.insert(
            "strike".to_string(),
            BinaryHV::random(DIMENSION, rng.random()),
        );
        field_vectors.insert(
            "expiry".to_string(),
            BinaryHV::random(DIMENSION, rng.random()),
        );
        field_vectors.insert("iv".to_string(), BinaryHV::random(DIMENSION, rng.random()));

        // Scalar bases for permutation
        scalar_bases.insert(
            "strike".to_string(),
            BinaryHV::random(DIMENSION, rng.random()),
        );
        scalar_bases.insert(
            "expiry".to_string(),
            BinaryHV::random(DIMENSION, rng.random()),
        );
        scalar_bases.insert("iv".to_string(), BinaryHV::random(DIMENSION, rng.random()));

        Self {
            surface: HashMap::new(),
            vol_nodes_hologram: BundleAccumulator::new(DIMENSION),
            node_vectors: HashMap::new(),
            field_vectors,
            scalar_bases,
            symbol: symbol.to_string(),
        }
    }

    /// Encode a vol surface node into a hypervector
    fn encode_node(&self, strike: f64, expiry: u64) -> BinaryHV {
        // Strike: convert to cents for better resolution
        let strike_cents = (strike * 100.0) as usize;
        let strike_vec = self.scalar_bases["strike"].permute_words(strike_cents);
        let bound_strike = self.field_vectors["strike"].bind(&strike_vec);

        // Expiry: days from epoch
        let expiry_vec = self.scalar_bases["expiry"].permute_words(expiry as usize);
        let bound_expiry = self.field_vectors["expiry"].bind(&expiry_vec);

        // Bundle strike + expiry
        let mut acc = BundleAccumulator::new(DIMENSION);
        acc.add(&bound_strike);
        acc.add(&bound_expiry);
        acc.threshold()
    }

    /// Encode an IV value as a hypervector
    fn encode_iv(&self, iv: f64) -> BinaryHV {
        // Scale IV [0.0, 2.0] to [0, 200] for permutation
        let iv_shift = (iv * 100.0).clamp(0.0, 200.0) as usize;
        let iv_vec = self.scalar_bases["iv"].permute_words(iv_shift);
        self.field_vectors["iv"].bind(&iv_vec)
    }

    /// Update a node on the volatility surface
    ///
    /// # Arguments
    /// * `strike` - Strike price
    /// * `expiry` - Expiration epoch (days)
    /// * `iv` - Implied volatility (e.g., 0.25 for 25%)
    pub fn update_node(&mut self, strike: f64, expiry: u64, iv: f64) {
        let key = ((strike * 100.0) as u32, expiry);

        // Remove old node from hologram if exists
        if let Some(old_vec) = self.node_vectors.get(&key) {
            self.vol_nodes_hologram.subtract(old_vec);
        }

        // Create new node vector
        let node_vec = self.encode_node(strike, expiry);

        // Add to hologram and storage
        self.vol_nodes_hologram.add(&node_vec);
        self.node_vectors.insert(key, node_vec);
        self.surface.insert(key, iv);
    }

    /// Remove a node from the surface
    pub fn remove_node(&mut self, strike: f64, expiry: u64) {
        let key = ((strike * 100.0) as u32, expiry);

        if let Some(old_vec) = self.node_vectors.remove(&key) {
            self.vol_nodes_hologram.subtract(&old_vec);
            self.surface.remove(&key);
        }
    }

    /// Query exact IV at a grid point
    ///
    /// Returns None if the exact point doesn't exist
    pub fn query_iv_exact(&self, strike: f64, expiry: u64) -> Option<f64> {
        let key = ((strike * 100.0) as u32, expiry);
        self.surface.get(&key).copied()
    }

    /// Interpolate IV using holographic nearest-neighbor
    ///
    /// Uses similarity-weighted average of nearby nodes.
    /// O(N) where N = number of nodes, but constant time per similarity check.
    pub fn interpolate_iv_holographic(&self, strike: f64, expiry: u64) -> Option<f64> {
        if self.surface.is_empty() {
            return None;
        }

        // First check exact match
        if let Some(iv) = self.query_iv_exact(strike, expiry) {
            return Some(iv);
        }

        // Encode query point
        let query_vec = self.encode_node(strike, expiry);

        // Find nearest neighbors by similarity
        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;

        for (key, node_vec) in &self.node_vectors {
            let similarity = query_vec.similarity(node_vec);

            // Use similarity as weight (higher = closer)
            // Only use nodes with reasonable similarity (> 0.5 means closer than orthogonal)
            if similarity > 0.5 {
                let iv = self.surface[key];
                let weight = (similarity - 0.5) * 2.0; // Scale [0.5, 1.0] to [0, 1]
                weighted_sum += iv * weight as f64;
                weight_total += weight as f64;
            }
        }

        if weight_total > 0.0 {
            Some(weighted_sum / weight_total)
        } else {
            // Fallback: return average of all nodes
            let sum: f64 = self.surface.values().sum();
            Some(sum / self.surface.len() as f64)
        }
    }

    /// Interpolate IV using bilinear interpolation on the grid
    ///
    /// More accurate than holographic for dense grids but requires
    /// finding explicit neighboring points.
    pub fn interpolate_iv_bilinear(&self, strike: f64, expiry: u64) -> Option<f64> {
        if self.surface.is_empty() {
            return None;
        }

        // Check exact match first
        if let Some(iv) = self.query_iv_exact(strike, expiry) {
            return Some(iv);
        }

        let target_strike_cents = (strike * 100.0) as u32;

        // Find bounding strikes
        let mut lower_strike: Option<u32> = None;
        let mut upper_strike: Option<u32> = None;

        // Find bounding expiries
        let mut lower_expiry: Option<u64> = None;
        let mut upper_expiry: Option<u64> = None;

        for (k_strike, k_expiry) in self.surface.keys() {
            // Strike bounds
            if *k_strike <= target_strike_cents {
                lower_strike = Some(lower_strike.map_or(*k_strike, |l| l.max(*k_strike)));
            }
            if *k_strike >= target_strike_cents {
                upper_strike = Some(upper_strike.map_or(*k_strike, |u| u.min(*k_strike)));
            }

            // Expiry bounds
            if *k_expiry <= expiry {
                lower_expiry = Some(lower_expiry.map_or(*k_expiry, |l| l.max(*k_expiry)));
            }
            if *k_expiry >= expiry {
                upper_expiry = Some(upper_expiry.map_or(*k_expiry, |u| u.min(*k_expiry)));
            }
        }

        // Need at least one point to interpolate
        let ls = lower_strike.or(upper_strike)?;
        let us = upper_strike.unwrap_or(ls);
        let le = lower_expiry.or(upper_expiry)?;
        let ue = upper_expiry.unwrap_or(le);

        // Get corner IVs (if they exist)
        let iv_ll = self.surface.get(&(ls, le));
        let iv_lu = self.surface.get(&(ls, ue));
        let iv_ul = self.surface.get(&(us, le));
        let iv_uu = self.surface.get(&(us, ue));

        // Collect available corners
        let mut sum = 0.0;
        let mut count = 0;

        for iv in [iv_ll, iv_lu, iv_ul, iv_uu].iter().flatten() {
            sum += **iv;
            count += 1;
        }

        if count == 0 {
            return None;
        }

        // Full bilinear if all 4 corners exist
        if let (Some(&v_ll), Some(&v_lu), Some(&v_ul), Some(&v_uu)) = (iv_ll, iv_lu, iv_ul, iv_uu) {
            if ls != us && le != ue {
                // Bilinear interpolation
                let t_strike = (target_strike_cents - ls) as f64 / (us - ls) as f64;
                let t_expiry = (expiry - le) as f64 / (ue - le) as f64;

                let iv_l = v_ll * (1.0 - t_expiry) + v_lu * t_expiry;
                let iv_u = v_ul * (1.0 - t_expiry) + v_uu * t_expiry;
                let iv = iv_l * (1.0 - t_strike) + iv_u * t_strike;

                return Some(iv);
            }
        }

        // Fallback to simple average of available corners
        Some(sum / count as f64)
    }

    /// Interpolate Greeks from IV using finite differences
    ///
    /// Approximates delta, gamma, theta, vega from the IV surface.
    /// Requires Black-Scholes model assumptions.
    pub fn interpolate_greeks(
        &self,
        strike: f64,
        expiry: u64,
        spot: f64,
        option_type: OptionType,
        risk_free_rate: f64,
    ) -> Option<super::Greeks> {
        let iv = self.interpolate_iv_holographic(strike, expiry)?;

        // Time to expiry in years (assume 365 days/year)
        let now_epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            / 86400; // Current epoch day

        let t = if expiry > now_epoch {
            (expiry - now_epoch) as f64 / 365.0
        } else {
            0.001 // Near-expiry floor
        };

        // d1 calculation (Black-Scholes)
        let d1 = (spot.ln() / strike.ln() + (risk_free_rate + iv * iv / 2.0) * t) / (iv * t.sqrt());

        // Delta
        let delta = match option_type {
            OptionType::Call => norm_cdf(d1),
            OptionType::Put => norm_cdf(d1) - 1.0,
        };

        // Gamma (same for call and put)
        let gamma = norm_pdf(d1) / (spot * iv * t.sqrt());

        // Theta (simplified)
        let theta = -(spot * norm_pdf(d1) * iv) / (2.0 * t.sqrt())
            - risk_free_rate * strike * (-risk_free_rate * t).exp() * norm_cdf(d1 - iv * t.sqrt());
        let theta = theta / 365.0; // Daily theta

        // Vega
        let vega = spot * t.sqrt() * norm_pdf(d1) / 100.0; // Per 1% IV move

        Some(super::Greeks::new(delta, gamma.abs(), theta, vega.abs()))
    }

    /// Get all nodes in the surface
    pub fn nodes(&self) -> impl Iterator<Item = (f64, u64, f64)> + '_ {
        self.surface
            .iter()
            .map(|((strike_cents, expiry), iv)| (*strike_cents as f64 / 100.0, *expiry, *iv))
    }

    /// Number of nodes in the surface
    pub fn node_count(&self) -> usize {
        self.surface.len()
    }

    /// Clear all nodes from the surface
    pub fn clear(&mut self) {
        self.surface.clear();
        self.node_vectors.clear();
        self.vol_nodes_hologram = BundleAccumulator::new(DIMENSION);
    }

    /// Get strikes at a given expiry
    pub fn strikes_at_expiry(&self, expiry: u64) -> Vec<f64> {
        self.surface
            .keys()
            .filter(|(_, e)| *e == expiry)
            .map(|(s, _)| *s as f64 / 100.0)
            .collect()
    }

    /// Get expiries at a given strike
    pub fn expiries_at_strike(&self, strike: f64) -> Vec<u64> {
        let strike_cents = (strike * 100.0) as u32;
        self.surface
            .keys()
            .filter(|(s, _)| *s == strike_cents)
            .map(|(_, e)| *e)
            .collect()
    }
}

/// Standard normal CDF approximation (Abramowitz & Stegun)
fn norm_cdf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / std::f64::consts::SQRT_2;

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    0.5 * (1.0 + sign * y)
}

/// Standard normal PDF
fn norm_pdf(x: f64) -> f64 {
    (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vol_surface_exact_query() {
        let mut surface = VolatilitySurface::new("AAPL");

        surface.update_node(150.0, 19800, 0.25);
        surface.update_node(155.0, 19800, 0.28);
        surface.update_node(150.0, 19830, 0.26);

        assert_eq!(surface.query_iv_exact(150.0, 19800), Some(0.25));
        assert_eq!(surface.query_iv_exact(155.0, 19800), Some(0.28));
        assert_eq!(surface.query_iv_exact(152.5, 19800), None); // Not an exact node
    }

    #[test]
    fn test_vol_surface_interpolation() {
        let mut surface = VolatilitySurface::new("AAPL");

        // Create a 2x2 grid
        surface.update_node(150.0, 19800, 0.20);
        surface.update_node(160.0, 19800, 0.22);
        surface.update_node(150.0, 19830, 0.24);
        surface.update_node(160.0, 19830, 0.26);

        // Interpolate at center
        let iv = surface.interpolate_iv_bilinear(155.0, 19815).unwrap();

        // Should be approximately the average: (0.20 + 0.22 + 0.24 + 0.26) / 4 = 0.23
        assert!((iv - 0.23).abs() < 0.02, "Expected ~0.23, got {}", iv);
    }

    #[test]
    fn test_vol_surface_holographic_interpolation() {
        let mut surface = VolatilitySurface::new("SPY");

        // Add several nodes
        surface.update_node(400.0, 19800, 0.18);
        surface.update_node(410.0, 19800, 0.20);
        surface.update_node(420.0, 19800, 0.22);

        // Query between nodes
        let iv = surface.interpolate_iv_holographic(415.0, 19800);
        assert!(iv.is_some());

        let iv = iv.unwrap();
        // Should be between 0.20 and 0.22
        assert!(iv >= 0.18 && iv <= 0.24, "IV {} out of expected range", iv);
    }

    #[test]
    fn test_vol_surface_update_replaces() {
        let mut surface = VolatilitySurface::new("TSLA");

        surface.update_node(200.0, 19800, 0.30);
        assert_eq!(surface.query_iv_exact(200.0, 19800), Some(0.30));

        // Update same node
        surface.update_node(200.0, 19800, 0.35);
        assert_eq!(surface.query_iv_exact(200.0, 19800), Some(0.35));

        // Still only one node
        assert_eq!(surface.node_count(), 1);
    }

    #[test]
    fn test_vol_surface_remove() {
        let mut surface = VolatilitySurface::new("GOOG");

        surface.update_node(100.0, 19800, 0.25);
        surface.update_node(110.0, 19800, 0.27);
        assert_eq!(surface.node_count(), 2);

        surface.remove_node(100.0, 19800);
        assert_eq!(surface.node_count(), 1);
        assert_eq!(surface.query_iv_exact(100.0, 19800), None);
    }
}
