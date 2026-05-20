//! Macro for generating domain-specific HDC encoder structs.
//!
//! The [`define_domain_module!`] macro eliminates boilerplate code across
//! the 19 domain-link modules by generating the common encoder struct,
//! constructor, `bundle()`, `encode_scalar()`, and dynamic vector cache methods.
//!
//! # Usage
//!
//! ```rust,ignore
//! use joule_db_hdc::define_domain_module;
//!
//! define_domain_module! {
//!     /// My domain encoder
//!     pub struct MyLink {
//!         seed: 0xABCD_0001,
//!         dimension: 10000,
//!         fields: ["entity", "value", "category", "timestamp"],
//!         scalars: ["amount", "duration", "count"],
//!         enums: {
//!             status_vectors: StatusType => [Active, Inactive, Pending],
//!             priority_vectors: Priority => [Low, Medium, High],
//!         },
//!         dynamic: {
//!             location_vectors: "location",
//!             tag_vectors: "tag",
//!         },
//!     }
//! }
//! ```
//!
//! This generates:
//! - A struct with `field_vectors`, `scalar_bases`, enum vector maps, dynamic caches, and `rng`
//! - `new()` constructor with deterministic seeded initialization
//! - `bundle(&self, components: &[BinaryHV]) -> BinaryHV`
//! - `encode_scalar(&self, base: &str, value: u32, max: u32) -> BinaryHV`
//! - `get_<name>(&mut self, key: &str) -> BinaryHV` for each dynamic cache
//! - `Default` impl

/// Generate a domain-specific HDC encoder struct with all common boilerplate.
///
/// See [module-level documentation](self) for full usage.
#[macro_export]
macro_rules! define_domain_module {
    (
        $(#[$meta:meta])*
        pub struct $name:ident {
            seed: $seed:expr,
            dimension: $dim:expr,
            fields: [$($field:expr),* $(,)?],
            scalars: [$($scalar:expr),* $(,)?],
            enums: {
                $(
                    $enum_map:ident : $enum_type:ty => [$($variant:expr),* $(,)?]
                ),* $(,)?
            }
            $(,dynamic: {
                $(
                    $dyn_map:ident : $dyn_label:expr
                ),* $(,)?
            })? $(,)?
        }
    ) => {
        $(#[$meta])*
        pub struct $name {
            field_vectors: std::collections::HashMap<String, $crate::BinaryHV>,
            scalar_bases: std::collections::HashMap<String, $crate::BinaryHV>,
            $(
                $enum_map: std::collections::HashMap<$enum_type, $crate::BinaryHV>,
            )*
            $($(
                $dyn_map: std::collections::HashMap<String, $crate::BinaryHV>,
            )*)?
            rng: rand::rngs::StdRng,
        }

        impl $name {
            /// Create a new encoder with deterministic seeded initialization.
            pub fn new() -> Self {
                use rand::{RngExt as _, SeedableRng as _};

                let mut rng = rand::rngs::StdRng::seed_from_u64($seed);

                let field_vectors: std::collections::HashMap<String, $crate::BinaryHV> =
                    [$($field),*].iter()
                        .map(|f: &&str| (f.to_string(), $crate::BinaryHV::random($dim, rng.random())))
                        .collect();

                let scalar_bases: std::collections::HashMap<String, $crate::BinaryHV> =
                    [$($scalar),*].iter()
                        .map(|n: &&str| (n.to_string(), $crate::BinaryHV::random($dim, rng.random())))
                        .collect();

                $(
                    let $enum_map: std::collections::HashMap<$enum_type, $crate::BinaryHV> =
                        [$($variant),*].iter()
                            .map(|v| (v.clone(), $crate::BinaryHV::random($dim, rng.random())))
                            .collect();
                )*

                Self {
                    field_vectors,
                    scalar_bases,
                    $(
                        $enum_map,
                    )*
                    $($(
                        $dyn_map: std::collections::HashMap::new(),
                    )*)?
                    rng,
                }
            }

            /// Bundle multiple component vectors into a single vector using majority vote.
            pub fn bundle(&self, components: &[$crate::BinaryHV]) -> $crate::BinaryHV {
                let mut acc = $crate::BundleAccumulator::new($dim);
                for c in components {
                    acc.add(c);
                }
                acc.threshold()
            }

            /// Encode a scalar value using permutation-based encoding.
            ///
            /// Maps a value in `[0, max]` to a permuted version of the named base vector.
            pub fn encode_scalar(&self, base: &str, value: u32, max: u32) -> $crate::BinaryHV {
                self.scalar_bases.get(base).unwrap()
                    .permute(((value as f64 / max as f64) * 100.0) as usize)
            }

            // Generate get_* methods for dynamic caches
            $crate::__domain_module_dynamic_getters!($dim $(, $($dyn_map, $dyn_label),*)?);
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

/// Internal helper macro to generate dynamic getter methods.
/// Not intended for direct use.
#[macro_export]
#[doc(hidden)]
macro_rules! __domain_module_dynamic_getters {
    // No dynamic fields
    ($dim:expr) => {};
    // With dynamic fields
    ($dim:expr, $($dyn_map:ident, $dyn_label:expr),*) => {
        $(
            /// Get or create a vector for the given key in this dynamic cache.
            pub fn $dyn_map(&mut self, key: &str) -> $crate::BinaryHV {
                use rand::RngExt as _;
                if !self.$dyn_map.contains_key(key) {
                    self.$dyn_map.insert(
                        key.to_string(),
                        $crate::BinaryHV::random($dim, self.rng.random()),
                    );
                }
                self.$dyn_map.get(key).unwrap().clone()
            }
        )*
    };
}

#[cfg(test)]
mod tests {
    use crate::{BinaryHV, BundleAccumulator};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum TestCategory {
        Alpha,
        Beta,
        Gamma,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum TestStatus {
        Active,
        Inactive,
    }

    define_domain_module! {
        /// Test encoder for macro validation
        pub struct TestEncoder {
            seed: 0xDEAD_BEEF,
            dimension: 10000,
            fields: ["entity", "value", "label"],
            scalars: ["amount", "count"],
            enums: {
                category_vectors: TestCategory => [TestCategory::Alpha, TestCategory::Beta, TestCategory::Gamma],
                status_vectors: TestStatus => [TestStatus::Active, TestStatus::Inactive]
            },
            dynamic: {
                tag_vectors: "tag",
                location_vectors: "location"
            },
        }
    }

    #[test]
    fn test_macro_creates_encoder() {
        let encoder = TestEncoder::new();
        // Field vectors should be created
        assert!(encoder.field_vectors.contains_key("entity"));
        assert!(encoder.field_vectors.contains_key("value"));
        assert!(encoder.field_vectors.contains_key("label"));

        // Scalar bases should be created
        assert!(encoder.scalar_bases.contains_key("amount"));
        assert!(encoder.scalar_bases.contains_key("count"));

        // Enum vectors should be created
        assert!(encoder.category_vectors.contains_key(&TestCategory::Alpha));
        assert!(encoder.category_vectors.contains_key(&TestCategory::Beta));
        assert!(encoder.category_vectors.contains_key(&TestCategory::Gamma));
        assert!(encoder.status_vectors.contains_key(&TestStatus::Active));
        assert!(encoder.status_vectors.contains_key(&TestStatus::Inactive));
    }

    #[test]
    fn test_macro_bundle() {
        let encoder = TestEncoder::new();
        let a = BinaryHV::random(10000, 1);
        let b = BinaryHV::random(10000, 2);
        let result = encoder.bundle(&[a, b]);
        assert_eq!(result.dimension(), 10000);
    }

    #[test]
    fn test_macro_encode_scalar() {
        let encoder = TestEncoder::new();
        let hv = encoder.encode_scalar("amount", 50, 100);
        assert_eq!(hv.dimension(), 10000);

        // Same value should produce same vector
        let hv2 = encoder.encode_scalar("amount", 50, 100);
        assert_eq!(hv.similarity(&hv2), 1.0);

        // Different values should produce different vectors
        let hv3 = encoder.encode_scalar("amount", 10, 100);
        let sim = hv.similarity(&hv3);
        assert!(sim < 1.0);
    }

    #[test]
    fn test_macro_dynamic_getters() {
        let mut encoder = TestEncoder::new();

        // First call creates the vector
        let tag1 = encoder.tag_vectors("rust");
        assert_eq!(tag1.dimension(), 10000);

        // Second call returns the same vector
        let tag1_again = encoder.tag_vectors("rust");
        assert_eq!(tag1.similarity(&tag1_again), 1.0);

        // Different key returns different vector
        let tag2 = encoder.tag_vectors("python");
        let sim = tag1.similarity(&tag2);
        assert!(
            sim < 0.7,
            "Different tags should have low similarity: {}",
            sim
        );

        // Location cache works independently
        let loc = encoder.location_vectors("NYC");
        assert_eq!(loc.dimension(), 10000);
    }

    #[test]
    fn test_macro_default_impl() {
        let encoder = TestEncoder::default();
        assert!(encoder.field_vectors.contains_key("entity"));
    }

    #[test]
    fn test_macro_deterministic_initialization() {
        let enc1 = TestEncoder::new();
        let enc2 = TestEncoder::new();

        // Same seed should produce identical vectors
        let sim = enc1.field_vectors["entity"].similarity(&enc2.field_vectors["entity"]);
        assert_eq!(
            sim, 1.0,
            "Deterministic initialization should produce identical vectors"
        );
    }

    #[test]
    fn test_macro_enum_binding() {
        let encoder = TestEncoder::new();

        // Bind a field with a category - standard pattern across all modules
        let category_hv =
            encoder.field_vectors["entity"].bind(&encoder.category_vectors[&TestCategory::Alpha]);
        assert_eq!(category_hv.dimension(), 10000);

        // Same binding should always produce same result
        let category_hv2 =
            encoder.field_vectors["entity"].bind(&encoder.category_vectors[&TestCategory::Alpha]);
        assert_eq!(category_hv.similarity(&category_hv2), 1.0);

        // Different category should produce different binding
        let other_hv =
            encoder.field_vectors["entity"].bind(&encoder.category_vectors[&TestCategory::Beta]);
        let sim = category_hv.similarity(&other_hv);
        assert!(sim < 0.7);
    }
}
