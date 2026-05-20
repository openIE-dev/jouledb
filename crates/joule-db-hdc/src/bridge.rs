//! Bridge to invisible-os `Hypervector` trait.
//!
//! Provides zero-copy conversions between joule-db-hdc's `BinaryHV` and
//! inv-hdc-core's `BinaryHV`, enabling joule-db's hyperdimensional index
//! to participate in the cross-layer HDC pipeline:
//!
//! **ask-davidc (Phasor) → joule-db (BinaryHV) → ai-os NeuralFS (semantic search)**

use crate::turbo_holographic::BinaryHV as JdbBinaryHV;
use inv_hdc_core::BinaryHV as CoreBinaryHV;
use inv_hdc_core::Hypervector;

/// Convert joule-db-hdc's `BinaryHV` to inv-hdc-core's `BinaryHV`.
///
/// Zero-cost data copy. Both types use the same packed u64 representation.
/// Note: Core's `similarity()` has a known XNOR over-count for non-64-aligned
/// dimensions. Use `HypervectorBridge::cross_similarity()` for accurate results.
impl From<&JdbBinaryHV> for CoreBinaryHV {
    fn from(jdb: &JdbBinaryHV) -> Self {
        CoreBinaryHV {
            data: jdb.as_words().to_vec(),
            dim: jdb.dimension(),
        }
    }
}

/// Convert inv-hdc-core's `BinaryHV` to joule-db-hdc's `BinaryHV`.
impl From<&CoreBinaryHV> for JdbBinaryHV {
    fn from(core: &CoreBinaryHV) -> Self {
        JdbBinaryHV::from_words(core.data.clone(), core.dim)
    }
}

/// Extension trait for joule-db's `BinaryHV` to use inv-hdc-core operations.
pub trait HypervectorBridge {
    /// Convert to the shared `BinaryHV` type for cross-layer operations.
    fn to_core(&self) -> CoreBinaryHV;

    /// Compute cross-algebra similarity via the shared trait.
    ///
    /// Uses JDB's hamming-based formula (correct for non-64-aligned dims)
    /// rather than Core's XNOR formula (which over-counts matching padding bits).
    fn cross_similarity(&self, other: &CoreBinaryHV) -> f64;
}

impl HypervectorBridge for JdbBinaryHV {
    fn to_core(&self) -> CoreBinaryHV {
        CoreBinaryHV::from(self)
    }

    fn cross_similarity(&self, other: &CoreBinaryHV) -> f64 {
        // Use JDB's similarity via conversion — JDB's formula handles
        // non-64-aligned dimensions correctly (XOR popcount / dim).
        let other_jdb = JdbBinaryHV::from(other);
        self.similarity(&other_jdb) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jdb_to_core_roundtrip() {
        let jdb = JdbBinaryHV::random(10000, 42);
        let core = CoreBinaryHV::from(&jdb);
        let back = JdbBinaryHV::from(&core);

        // Perfect roundtrip — data is copied verbatim both ways
        assert_eq!(jdb.dimension(), back.dimension());
        assert_eq!(jdb.hamming_distance(&back), 0);
    }

    #[test]
    fn core_to_jdb_roundtrip() {
        let core = CoreBinaryHV::from_seed(10000, 99);
        let jdb = JdbBinaryHV::from(&core);
        let back = CoreBinaryHV::from(&jdb);

        assert_eq!(core.dim, back.dim);
        assert_eq!(core.data, back.data); // Verbatim data preservation
    }

    #[test]
    fn dimension_preserved() {
        for dim in [64, 1000, 4096, 10000] {
            let jdb = JdbBinaryHV::random(dim, 123);
            let core = CoreBinaryHV::from(&jdb);
            assert_eq!(core.dim, dim);

            let back = JdbBinaryHV::from(&core);
            assert_eq!(back.dimension(), dim);
        }
    }

    #[test]
    fn similarity_matches() {
        // Compare similarity via JDB (correct for non-64-aligned dims)
        // and cross_similarity bridge (which also uses JDB's formula).
        let a_jdb = JdbBinaryHV::random(10000, 1);
        let b_jdb = JdbBinaryHV::random(10000, 2);
        let b_core = CoreBinaryHV::from(&b_jdb);

        let jdb_sim = a_jdb.similarity(&b_jdb) as f64;
        let bridge_sim = a_jdb.cross_similarity(&b_core);

        assert!(
            (jdb_sim - bridge_sim).abs() < 0.001,
            "bridge similarity should match JDB: jdb={jdb_sim}, bridge={bridge_sim}"
        );
    }

    #[test]
    fn cross_similarity_works() {
        let jdb = JdbBinaryHV::random(10000, 42);
        let core = CoreBinaryHV::from(&jdb);

        // Self-similarity via cross bridge
        assert!((jdb.cross_similarity(&core) - 1.0).abs() < 0.001);

        // Different vector
        let other_core = CoreBinaryHV::from_seed(10000, 99);
        let sim = jdb.cross_similarity(&other_core);
        assert!(sim > 0.0 && sim < 1.0); // Should be near 0.5 for random
    }

    #[test]
    fn bind_interop() {
        // Bind in jdb space, convert, check equivalence with core space.
        let a_jdb = JdbBinaryHV::random(10000, 10);
        let b_jdb = JdbBinaryHV::random(10000, 20);
        let c_jdb = a_jdb.bind(&b_jdb);

        let a_core = CoreBinaryHV::from(&a_jdb);
        let b_core = CoreBinaryHV::from(&b_jdb);
        let c_core = a_core.bind(&b_core);

        // Compare via JDB's similarity (correct for non-64-aligned dims)
        let c_core_as_jdb = JdbBinaryHV::from(&c_core);
        assert!(
            (c_jdb.similarity(&c_core_as_jdb) - 1.0).abs() < 0.001,
            "bind results should match across algebras"
        );
    }
}
