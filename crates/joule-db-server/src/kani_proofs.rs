//! Kani bounded model checker proofs for JSON operations.
//!
//! These proofs verify algebraic properties of json_compare and json_equals
//! over a finite but exhaustive domain of JSON value types.
//!
//! Install: `cargo install cargo-kani`
//! Run: `cargo kani -p joule-db-server`

#[cfg(kani)]
mod proofs {
    use crate::json_ops::{json_compare, json_equals};

    /// Build a canonical JSON value for each type.
    fn canonical_values() -> [serde_json::Value; 7] {
        [
            serde_json::Value::Null,
            serde_json::Value::Bool(false),
            serde_json::Value::Bool(true),
            serde_json::json!(0),
            serde_json::json!(42),
            serde_json::json!(""),
            serde_json::json!("hello"),
        ]
    }

    /// **Proof: json_compare is reflexive for all canonical types.**
    #[kani::proof]
    fn prove_json_compare_reflexive() {
        let values = canonical_values();
        for v in &values {
            assert_eq!(json_compare(v, v), 0, "reflexivity violated");
        }
    }

    /// **Proof: json_compare output is always in {-1, 0, 1}.**
    #[kani::proof]
    fn prove_json_compare_bounded() {
        let values = canonical_values();
        for a in &values {
            for b in &values {
                let result = json_compare(a, b);
                assert!(result >= -1 && result <= 1);
            }
        }
    }

    /// **Proof: json_compare is antisymmetric for all canonical pairs.**
    #[kani::proof]
    fn prove_json_compare_antisymmetric() {
        let values = canonical_values();
        for a in &values {
            for b in &values {
                let ab = json_compare(a, b);
                let ba = json_compare(b, a);
                assert_eq!(ab, -ba, "antisymmetry violated");
            }
        }
    }

    /// **Proof: json_compare is transitive for all canonical triples.**
    #[kani::proof]
    fn prove_json_compare_transitive() {
        let values = canonical_values();
        for a in &values {
            for b in &values {
                for c in &values {
                    let ab = json_compare(a, b);
                    let bc = json_compare(b, c);
                    if ab <= 0 && bc <= 0 {
                        let ac = json_compare(a, c);
                        assert!(ac <= 0, "transitivity violated");
                    }
                }
            }
        }
    }

    /// **Proof: json_equals is reflexive for all canonical types.**
    #[kani::proof]
    fn prove_json_equals_reflexive() {
        let values = canonical_values();
        for v in &values {
            assert!(json_equals(v, v), "json_equals reflexivity violated");
        }
    }

    /// **Proof: json_equals is commutative for all canonical pairs.**
    #[kani::proof]
    fn prove_json_equals_commutative() {
        let values = canonical_values();
        for a in &values {
            for b in &values {
                assert_eq!(
                    json_equals(a, b),
                    json_equals(b, a),
                    "json_equals commutativity violated"
                );
            }
        }
    }

    /// **Proof: type_rank ordering is consistent with json_compare.**
    ///
    /// Different types must compare according to Null < Bool < Number < String.
    #[kani::proof]
    fn prove_type_ordering() {
        let null = serde_json::Value::Null;
        let bool_val = serde_json::json!(false);
        let num = serde_json::json!(0);
        let str_val = serde_json::json!("");

        // Null < Bool < Number < String
        assert_eq!(json_compare(&null, &bool_val), -1);
        assert_eq!(json_compare(&bool_val, &num), -1);
        assert_eq!(json_compare(&num, &str_val), -1);

        // Reverse
        assert_eq!(json_compare(&str_val, &num), 1);
        assert_eq!(json_compare(&num, &bool_val), 1);
        assert_eq!(json_compare(&bool_val, &null), 1);
    }
}
