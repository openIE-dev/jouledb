//! Common trait for domain-specific HDC encoding modules.
//!
//! All domain-link modules (cyber, health, iot, etc.) implement these traits
//! to provide a consistent encoding interface for their domain types.

use crate::turbo_holographic::{BinaryHV, BundleAccumulator};

/// Common trait for domain-specific HDC encoders.
///
/// Each domain module has an encoder struct that converts domain-specific
/// data types into hyperdimensional vectors. This trait provides the
/// standard interface for encoding and similarity operations.
///
/// # Example
///
/// ```rust,ignore
/// use joule_db_hdc::DomainEncoder;
///
/// struct MyEncoder { /* ... */ }
///
/// impl DomainEncoder for MyEncoder {
///     fn dimension(&self) -> usize { 10000 }
///
///     fn bundle(&self, components: &[BinaryHV]) -> BinaryHV {
///         let mut acc = BundleAccumulator::new(self.dimension());
///         for c in components { acc.add(c); }
///         acc.threshold()
///     }
///
///     fn encode_scalar(&self, base: &str, value: u32, max: u32) -> BinaryHV {
///         self.scalar_base(base).permute(((value as f64 / max as f64) * 100.0) as usize)
///     }
/// }
/// ```
pub trait DomainEncoder {
    /// Get the dimension used by this encoder (typically 10000).
    fn dimension(&self) -> usize;

    /// Bundle multiple component vectors into a single vector using majority vote.
    fn bundle(&self, components: &[BinaryHV]) -> BinaryHV {
        let mut acc = BundleAccumulator::new(self.dimension());
        for c in components {
            acc.add(c);
        }
        acc.threshold()
    }

    /// Encode a scalar value using permutation-based encoding.
    ///
    /// Maps a value in [0, max] to a permuted version of a base vector,
    /// where similar values produce similar vectors.
    fn encode_scalar_with_base(&self, base: &BinaryHV, value: u32, max: u32) -> BinaryHV {
        let shift = ((value as f64 / max as f64) * 100.0) as usize;
        base.permute(shift)
    }
}
