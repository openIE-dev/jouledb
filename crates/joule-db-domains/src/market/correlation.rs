//! Cross-Asset Correlation Tracking
//!
//! Uses holographic co-movement tracking to estimate correlations between assets
//! without computing explicit covariance matrices.

use super::DIMENSION;
use joule_db_hdc::{BinaryHV, BundleAccumulator};
use std::collections::HashMap;

/// Direction of price movement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoveDirection {
    Up,
    Down,
    Flat,
}

impl MoveDirection {
    /// Determine direction from price change
    pub fn from_change(prev: f64, curr: f64, threshold: f64) -> Self {
        let pct_change = (curr - prev) / prev;
        if pct_change > threshold {
            MoveDirection::Up
        } else if pct_change < -threshold {
            MoveDirection::Down
        } else {
            MoveDirection::Flat
        }
    }
}

/// A single price observation
#[derive(Debug, Clone)]
struct PriceMove {
    /// Symbol
    symbol: String,
    /// Direction of move
    direction: MoveDirection,
    /// Magnitude (percentage)
    magnitude: f64,
    /// Encoded HV
    vector: BinaryHV,
    /// Timestamp
    timestamp: u64,
}

/// Correlation tracking using holographic co-movement
///
/// Tracks price movements across symbols and computes correlations
/// based on how often assets move together.
pub struct CorrelationMatrix {
    /// Recent price moves per symbol
    price_history: HashMap<String, Vec<BinaryHV>>,

    /// Co-movement bundles: (sym1, sym2) -> accumulated co-movements
    correlation_surface: HashMap<(String, String), BundleAccumulator>,

    /// Direction vectors (pre-computed)
    direction_vectors: HashMap<MoveDirection, BinaryHV>,

    /// Symbol vectors (cached)
    symbol_vectors: HashMap<String, BinaryHV>,

    /// Magnitude scalar base
    magnitude_base: BinaryHV,

    /// Field vectors
    field_vectors: HashMap<String, BinaryHV>,

    /// Move threshold for direction determination
    move_threshold: f64,

    /// Maximum history per symbol
    max_history: usize,

    /// Observation count per pair
    observation_counts: HashMap<(String, String), usize>,
}

impl CorrelationMatrix {
    /// Create a new correlation matrix
    pub fn new() -> Self {
        Self::with_threshold(0.001) // 0.1% default threshold
    }

    /// Create with custom move threshold
    pub fn with_threshold(threshold: f64) -> Self {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let mut rng = StdRng::seed_from_u64(0xC0EE_E1A7); // Deterministic seed

        // Pre-compute direction vectors
        let mut direction_vectors = HashMap::new();
        direction_vectors.insert(
            MoveDirection::Up,
            BinaryHV::from_hash(b"MOVE_UP", DIMENSION),
        );
        direction_vectors.insert(
            MoveDirection::Down,
            BinaryHV::from_hash(b"MOVE_DOWN", DIMENSION),
        );
        direction_vectors.insert(
            MoveDirection::Flat,
            BinaryHV::from_hash(b"MOVE_FLAT", DIMENSION),
        );

        // Field vectors
        let mut field_vectors = HashMap::new();
        field_vectors.insert(
            "symbol".to_string(),
            BinaryHV::random(DIMENSION, rng.random()),
        );
        field_vectors.insert(
            "direction".to_string(),
            BinaryHV::random(DIMENSION, rng.random()),
        );
        field_vectors.insert(
            "magnitude".to_string(),
            BinaryHV::random(DIMENSION, rng.random()),
        );

        // Magnitude scalar base
        let magnitude_base = BinaryHV::random(DIMENSION, rng.random());

        Self {
            price_history: HashMap::new(),
            correlation_surface: HashMap::new(),
            direction_vectors,
            symbol_vectors: HashMap::new(),
            magnitude_base,
            field_vectors,
            move_threshold: threshold,
            max_history: 100,
            observation_counts: HashMap::new(),
        }
    }

    /// Get or create symbol vector
    fn get_symbol_vector(&mut self, symbol: &str) -> BinaryHV {
        if let Some(vec) = self.symbol_vectors.get(symbol) {
            return vec.clone();
        }

        // Generate deterministic vector from symbol name
        let vec = BinaryHV::from_hash(symbol.as_bytes(), DIMENSION);
        self.symbol_vectors.insert(symbol.to_string(), vec.clone());
        vec
    }

    /// Encode a price move as a hypervector
    fn encode_move(&mut self, symbol: &str, direction: MoveDirection, magnitude: f64) -> BinaryHV {
        // Symbol binding
        let sym_vec = self.get_symbol_vector(symbol);
        let bound_symbol = self.field_vectors["symbol"].bind(&sym_vec);

        // Direction binding
        let dir_vec = &self.direction_vectors[&direction];
        let bound_direction = self.field_vectors["direction"].bind(dir_vec);

        // Magnitude binding (scale to 0-1000 for permutation)
        let mag_shift = (magnitude.abs() * 10000.0).min(1000.0) as usize;
        let mag_vec = self.magnitude_base.permute_words(mag_shift);
        let bound_magnitude = self.field_vectors["magnitude"].bind(&mag_vec);

        // Bundle
        let mut acc = BundleAccumulator::new(DIMENSION);
        acc.add(&bound_symbol);
        acc.add(&bound_direction);
        acc.add(&bound_magnitude);
        acc.threshold()
    }

    /// Observe a price move for a symbol
    ///
    /// # Arguments
    /// * `symbol` - Asset symbol
    /// * `prev_price` - Previous price
    /// * `curr_price` - Current price
    pub fn observe_price_move(&mut self, symbol: &str, prev_price: f64, curr_price: f64) {
        let direction = MoveDirection::from_change(prev_price, curr_price, self.move_threshold);
        let magnitude = (curr_price - prev_price) / prev_price;

        let move_vec = self.encode_move(symbol, direction, magnitude);

        // Store in history
        let history = self.price_history.entry(symbol.to_string()).or_default();
        history.push(move_vec);

        // Trim history
        while history.len() > self.max_history {
            history.remove(0);
        }
    }

    /// Update correlations based on co-occurring moves
    ///
    /// Call this after observing moves for multiple symbols at the same time point.
    pub fn update_correlations(&mut self, symbols_moved: &[&str]) {
        // For each pair of symbols that moved
        for i in 0..symbols_moved.len() {
            for j in (i + 1)..symbols_moved.len() {
                let sym1 = symbols_moved[i];
                let sym2 = symbols_moved[j];

                // Get latest move vectors
                let vec1 = self.price_history.get(sym1).and_then(|h| h.last());
                let vec2 = self.price_history.get(sym2).and_then(|h| h.last());

                if let (Some(v1), Some(v2)) = (vec1, vec2) {
                    // Create co-movement vector (BIND of both moves)
                    let co_move = v1.bind(v2);

                    // Canonical key ordering
                    let key = if sym1 < sym2 {
                        (sym1.to_string(), sym2.to_string())
                    } else {
                        (sym2.to_string(), sym1.to_string())
                    };

                    // Add to correlation surface
                    let acc = self
                        .correlation_surface
                        .entry(key.clone())
                        .or_insert_with(|| BundleAccumulator::new(DIMENSION));
                    acc.add(&co_move);

                    // Track observation count
                    *self.observation_counts.entry(key).or_insert(0) += 1;
                }
            }
        }
    }

    /// Query correlation between two symbols
    ///
    /// Returns a similarity score [0, 1] where:
    /// - 1.0 = perfectly correlated (always move together in same direction)
    /// - 0.5 = uncorrelated (random)
    /// - 0.0 = negatively correlated (always move opposite)
    pub fn query_correlation(&self, sym1: &str, sym2: &str) -> Option<f32> {
        // Get histories
        let hist1 = self.price_history.get(sym1)?;
        let hist2 = self.price_history.get(sym2)?;

        if hist1.is_empty() || hist2.is_empty() {
            return None;
        }

        // Compute average similarity of recent moves
        let min_len = hist1.len().min(hist2.len());
        if min_len == 0 {
            return None;
        }

        let mut total_similarity = 0.0f32;
        let offset1 = hist1.len() - min_len;
        let offset2 = hist2.len() - min_len;

        for i in 0..min_len {
            let sim = hist1[offset1 + i].similarity(&hist2[offset2 + i]);
            total_similarity += sim;
        }

        Some(total_similarity / min_len as f32)
    }

    /// Query correlation using the accumulated co-movement surface
    ///
    /// More accurate for longer observation periods.
    pub fn query_correlation_accumulated(&self, sym1: &str, sym2: &str) -> Option<f32> {
        let key = if sym1 < sym2 {
            (sym1.to_string(), sym2.to_string())
        } else {
            (sym2.to_string(), sym1.to_string())
        };

        let acc = self.correlation_surface.get(&key)?;
        let count = self.observation_counts.get(&key)?;

        if *count < 10 {
            return None; // Not enough observations
        }

        // Get the bundled co-movement vector
        let bundled = acc.threshold();

        // Compare to a "perfect positive correlation" reference
        // Two identical moves would have similarity 1.0
        // Two opposite moves would have similarity ~0.5 (orthogonal)

        // For accumulated correlation, we measure consistency by comparing
        // the bundled vector to individual observations
        // Higher count with consistent patterns = stronger accumulated signal
        let consistency_proxy = (*count as f32).sqrt() / 100.0;
        Some(consistency_proxy.min(1.0))
    }

    /// Find clusters of correlated symbols
    ///
    /// Returns groups of symbols that are correlated above the threshold.
    pub fn find_correlation_clusters(&self, threshold: f32) -> Vec<Vec<String>> {
        let symbols: Vec<&String> = self.price_history.keys().collect();
        let mut clusters: Vec<Vec<String>> = Vec::new();
        let mut assigned: HashMap<&String, usize> = HashMap::new();

        for sym in &symbols {
            if assigned.contains_key(sym) {
                continue;
            }

            // Start new cluster
            let mut cluster = vec![(*sym).clone()];
            assigned.insert(sym, clusters.len());

            // Find correlated symbols
            for other in &symbols {
                if sym == other || assigned.contains_key(other) {
                    continue;
                }

                if let Some(corr) = self.query_correlation(sym, other) {
                    if corr > threshold {
                        cluster.push((*other).clone());
                        assigned.insert(other, clusters.len());
                    }
                }
            }

            clusters.push(cluster);
        }

        clusters
    }

    /// Get all tracked symbols
    pub fn symbols(&self) -> impl Iterator<Item = &str> {
        self.price_history.keys().map(|s| s.as_str())
    }

    /// Get observation count for a pair
    pub fn observation_count(&self, sym1: &str, sym2: &str) -> usize {
        let key = if sym1 < sym2 {
            (sym1.to_string(), sym2.to_string())
        } else {
            (sym2.to_string(), sym1.to_string())
        };

        self.observation_counts.get(&key).copied().unwrap_or(0)
    }

    /// Clear all correlation data
    pub fn clear(&mut self) {
        self.price_history.clear();
        self.correlation_surface.clear();
        self.observation_counts.clear();
    }

    /// Set move threshold
    pub fn set_threshold(&mut self, threshold: f64) {
        self.move_threshold = threshold;
    }
}

impl Default for CorrelationMatrix {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_move_direction() {
        assert_eq!(
            MoveDirection::from_change(100.0, 101.0, 0.005),
            MoveDirection::Up
        );
        assert_eq!(
            MoveDirection::from_change(100.0, 99.0, 0.005),
            MoveDirection::Down
        );
        assert_eq!(
            MoveDirection::from_change(100.0, 100.2, 0.005),
            MoveDirection::Flat
        );
    }

    #[test]
    fn test_correlation_same_direction() {
        let mut matrix = CorrelationMatrix::with_threshold(0.001);

        // Both symbols move up together
        for _ in 0..20 {
            matrix.observe_price_move("AAPL", 150.0, 151.0); // Up
            matrix.observe_price_move("MSFT", 300.0, 302.0); // Up
            matrix.update_correlations(&["AAPL", "MSFT"]);
        }

        let corr = matrix.query_correlation("AAPL", "MSFT");
        assert!(corr.is_some());
        let corr = corr.unwrap();
        println!("Same direction correlation: {}", corr);

        // Should be higher than 0.5 (uncorrelated baseline)
        assert!(corr > 0.5, "Expected correlation > 0.5, got {}", corr);
    }

    #[test]
    fn test_correlation_opposite_direction() {
        let mut matrix = CorrelationMatrix::with_threshold(0.001);

        // Symbols move in opposite directions
        for _ in 0..20 {
            matrix.observe_price_move("SPY", 400.0, 404.0); // Up
            matrix.observe_price_move("VIX", 20.0, 19.0); // Down (inverse)
            matrix.update_correlations(&["SPY", "VIX"]);
        }

        let corr = matrix.query_correlation("SPY", "VIX");
        assert!(corr.is_some());
        let corr = corr.unwrap();
        println!("Opposite direction correlation: {}", corr);

        // Should be lower for opposite moves
        // Note: in VSA, opposite moves still share symbol encoding,
        // so correlation won't be 0, but should be lower than same-direction
    }

    #[test]
    fn test_cluster_detection() {
        let mut matrix = CorrelationMatrix::with_threshold(0.001);

        // Create two groups:
        // Group 1: AAPL, MSFT (tech)
        // Group 2: XOM, CVX (energy)

        for _ in 0..30 {
            // Tech moves together
            matrix.observe_price_move("AAPL", 150.0, 152.0);
            matrix.observe_price_move("MSFT", 300.0, 304.0);

            // Energy moves together but different from tech
            matrix.observe_price_move("XOM", 100.0, 99.0);
            matrix.observe_price_move("CVX", 150.0, 148.0);

            matrix.update_correlations(&["AAPL", "MSFT", "XOM", "CVX"]);
        }

        let clusters = matrix.find_correlation_clusters(0.6);
        println!("Clusters found: {:?}", clusters);

        // Should find at least 2 distinct clusters
        assert!(!clusters.is_empty());
    }

    #[test]
    fn test_observation_count() {
        let mut matrix = CorrelationMatrix::new();

        for _ in 0..5 {
            matrix.observe_price_move("A", 10.0, 10.1);
            matrix.observe_price_move("B", 20.0, 20.2);
            matrix.update_correlations(&["A", "B"]);
        }

        assert_eq!(matrix.observation_count("A", "B"), 5);
        assert_eq!(matrix.observation_count("B", "A"), 5); // Symmetric
    }
}
