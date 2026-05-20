//! Property-based tests for JouleDB JSON operations.
//!
//! Verifies algebraic properties of json_equals, json_compare, json arithmetic,
//! and type coercion functions using `proptest`.

use proptest::prelude::*;
use serde_json::json;

use crate::json_ops::*;

// ═══════════════════════════════════════════════════════════════════════════
// Strategies
// ═══════════════════════════════════════════════════════════════════════════

/// Generate a leaf-level JSON value (no nesting).
fn arb_json_leaf() -> impl Strategy<Value = serde_json::Value> {
    prop_oneof![
        Just(serde_json::Value::Null),
        prop::bool::ANY.prop_map(|b| json!(b)),
        prop::num::i64::ANY.prop_map(|n| json!(n)),
        // Use finite f64 only (NaN/Inf can't be represented in JSON)
        (-1e15f64..1e15).prop_map(|f| {
            serde_json::Number::from_f64(f)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null)
        }),
        "[a-zA-Z0-9 ]{0,20}".prop_map(|s| json!(s)),
    ]
}

/// Generate a numeric JSON value (for arithmetic tests).
fn arb_json_number() -> impl Strategy<Value = serde_json::Value> {
    prop_oneof![
        prop::num::i64::ANY.prop_map(|n| json!(n)),
        (-1e12f64..1e12)
            .prop_filter("must be finite", |f| f.is_finite())
            .prop_map(|f| {
                serde_json::Number::from_f64(f)
                    .map(serde_json::Value::Number)
                    .unwrap_or(json!(0))
            }),
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// Properties — Equality
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    /// **Property 1: json_equals is commutative.**
    #[test]
    fn prop_json_equals_commutative(a in arb_json_leaf(), b in arb_json_leaf()) {
        let ab = json_equals(&a, &b);
        let ba = json_equals(&b, &a);
        prop_assert!(ab == ba, "json_equals commutativity violated for {:?} and {:?}", a, b);
    }

    /// **Property 2: json_equals is reflexive.**
    #[test]
    fn prop_json_equals_reflexive(a in arb_json_leaf()) {
        prop_assert!(json_equals(&a, &a), "json_equals reflexivity violated for {:?}", a);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Properties — Comparison
    // ═══════════════════════════════════════════════════════════════════════

    /// **Property 3: json_compare is reflexive (zero for self).**
    #[test]
    fn prop_json_compare_reflexive(a in arb_json_leaf()) {
        let result = json_compare(&a, &a);
        prop_assert!(result == 0, "json_compare reflexivity violated: got {} for {:?}", result, a);
    }

    /// **Property 4: json_compare is antisymmetric.**
    #[test]
    fn prop_json_compare_antisymmetric(a in arb_json_leaf(), b in arb_json_leaf()) {
        let ab = json_compare(&a, &b);
        let ba = json_compare(&b, &a);
        prop_assert!(ab == -ba, "antisymmetry violated: compare({:?},{:?})={}, compare({:?},{:?})={}", a, b, ab, b, a, ba);
    }

    /// **Property 5: json_compare is transitive.**
    #[test]
    fn prop_json_compare_transitive(
        a in arb_json_leaf(),
        b in arb_json_leaf(),
        c in arb_json_leaf(),
    ) {
        let ab = json_compare(&a, &b);
        let bc = json_compare(&b, &c);
        if ab <= 0 && bc <= 0 {
            let ac = json_compare(&a, &c);
            prop_assert!(ac <= 0, "transitivity violated");
        }
    }

    /// **Property 6: json_compare output is always in {-1, 0, 1}.**
    #[test]
    fn prop_json_compare_bounded(a in arb_json_leaf(), b in arb_json_leaf()) {
        let result = json_compare(&a, &b);
        prop_assert!(result >= -1 && result <= 1, "json_compare must return -1, 0, or 1, got {}", result);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Properties — Arithmetic
    // ═══════════════════════════════════════════════════════════════════════

    /// **Property 7: Integer addition is commutative.**
    #[test]
    fn prop_json_add_commutative(a in arb_json_number(), b in arb_json_number()) {
        let ab = json_add(&a, &b);
        let ba = json_add(&b, &a);
        let af = try_coerce_to_f64(&ab);
        let bf = try_coerce_to_f64(&ba);
        match (af, bf) {
            (Some(x), Some(y)) => {
                prop_assert!((x - y).abs() < 1e-6 || (x == 0.0 && y == 0.0), "add commutativity violated");
            }
            (None, None) => {}
            _ => prop_assert!(false, "add commutativity type mismatch"),
        }
    }

    /// **Property 8: Division by zero returns Null.**
    #[test]
    fn prop_json_div_by_zero(a in arb_json_number()) {
        let result = json_div(&a, &json!(0));
        prop_assert!(result.is_null(), "division by zero must return Null, got {:?}", result);
    }

    /// **Property 9: Multiplication by 1 is identity.**
    #[test]
    fn prop_json_mul_identity(a in (-1_000_000i64..1_000_000i64)) {
        let val = json!(a);
        let result = json_mul(&val, &json!(1));
        prop_assert!(result == val, "x * 1 must equal x: {} * 1 = {:?}", a, result);
    }

    /// **Property 10: Addition with 0 is identity.**
    #[test]
    fn prop_json_add_identity(a in (-1_000_000i64..1_000_000i64)) {
        let val = json!(a);
        let result = json_add(&val, &json!(0));
        prop_assert!(result == val, "x + 0 must equal x: {} + 0 = {:?}", a, result);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Properties — Coercion
    // ═══════════════════════════════════════════════════════════════════════

    /// **Property 11: try_coerce_to_f64 is consistent with try_coerce_to_i64.**
    #[test]
    fn prop_coercion_consistency(n in (-1_000_000i64..1_000_000i64)) {
        let val = json!(n);
        let as_f = try_coerce_to_f64(&val).unwrap();
        let as_i = try_coerce_to_i64(&val).unwrap();
        prop_assert!(as_f == as_i as f64, "coercion mismatch: f64={}, i64={}", as_f, as_i);
    }

    /// **Property 12: normalize_int is idempotent.**
    #[test]
    fn prop_normalize_int_idempotent(v in arb_json_leaf()) {
        let once = normalize_int(v.clone());
        let twice = normalize_int(once.clone());
        prop_assert!(once == twice, "normalize_int must be idempotent");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Properties — AST conversion roundtrip
    // ═══════════════════════════════════════════════════════════════════════

    /// **Property 13: ast_value↔json roundtrip for simple values.**
    #[test]
    fn prop_ast_json_roundtrip(n in prop::num::i64::ANY) {
        let json_val = json!(n);
        let ast_val = json_to_ast_value(&json_val);
        let back = ast_value_to_json(&ast_val);
        prop_assert!(json_val == back, "ast_value↔json roundtrip failed for {}", n);
    }
}
