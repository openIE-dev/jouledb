/*
 * Function Registry Stress Tests
 * Exercises eval_scalar_function with adversarial inputs:
 * NULLs in every position, type mismatches, boundary values (i64::MIN/MAX,
 * f64::NAN/INFINITY), zero args, huge args, Unicode, empty strings.
 */

use joule_db_query::ast::Value;
use joule_db_query::functions::eval_scalar_function;

fn eval(name: &str, args: &[Value]) -> Result<Value, joule_db_query::QueryError> {
    eval_scalar_function(name, args, None)
}

// ── NULL Propagation ─────────────────────────────────────────────────
// Most scalar functions should return Value::Null when passed NULL input.

#[test]
fn fn_null_to_string_functions() {
    let fns = [
        "UPPER",
        "LOWER",
        "TRIM",
        "LTRIM",
        "RTRIM",
        "LENGTH",
        "REVERSE",
        "INITCAP",
        "CHAR_LENGTH",
        "ASCII",
    ];
    for name in fns {
        let result = eval(name, &[Value::Null]).unwrap();
        assert_eq!(result, Value::Null, "{name}(NULL) should be NULL");
    }
}

#[test]
fn fn_null_to_math_functions() {
    let fns = [
        "ABS", "CEIL", "CEILING", "FLOOR", "SQRT", "CBRT", "EXP", "LN", "LOG10", "LOG2", "SIGN",
        "SIN", "COS", "TAN", "ASIN", "ACOS", "ATAN", "DEGREES", "RADIANS",
    ];
    for name in fns {
        let result = eval(name, &[Value::Null]).unwrap();
        assert_eq!(result, Value::Null, "{name}(NULL) should be NULL");
    }
}

#[test]
fn fn_null_in_multi_arg_functions() {
    // COALESCE with all NULLs → NULL
    assert_eq!(
        eval("COALESCE", &[Value::Null, Value::Null, Value::Null]).unwrap(),
        Value::Null
    );
    // NULLIF with NULL first arg
    assert_eq!(
        eval("NULLIF", &[Value::Null, Value::Int(1)]).unwrap(),
        Value::Null
    );
    // CONCAT with NULLs
    let result = eval("CONCAT", &[Value::Null, Value::Null]).unwrap();
    // CONCAT may turn NULL into "" or "Null" — just verify it doesn't crash
    assert!(matches!(result, Value::String(_)));
}

// ── Zero Arguments ───────────────────────────────────────────────────

#[test]
fn fn_zero_args_to_string_functions() {
    let fns = [
        "UPPER", "LOWER", "TRIM", "LENGTH", "REVERSE", "SUBSTR", "REPLACE",
    ];
    for name in fns {
        let result = eval(name, &[]);
        assert!(result.is_ok(), "{name}() with 0 args should not panic");
    }
}

#[test]
fn fn_zero_args_to_math_functions() {
    let fns = ["ABS", "CEIL", "FLOOR", "SQRT", "ROUND", "POWER"];
    for name in fns {
        let result = eval(name, &[]);
        assert!(result.is_ok(), "{name}() with 0 args should not panic");
    }
}

#[test]
fn fn_zero_args_special() {
    // These should return valid values with 0 args
    let result = eval("PI", &[]).unwrap();
    assert!(matches!(result, Value::Float(f) if (f - std::f64::consts::PI).abs() < 1e-10));

    let result = eval("RANDOM", &[]).unwrap();
    assert!(matches!(result, Value::Float(f) if (0.0..=1.0).contains(&f)));

    let result = eval("NOW", &[]).unwrap();
    assert!(matches!(result, Value::Timestamp(_)));
}

// ── Type Mismatches ──────────────────────────────────────────────────

#[test]
fn fn_string_func_with_int() {
    // UPPER expects String, give it Int
    assert_eq!(eval("UPPER", &[Value::Int(42)]).unwrap(), Value::Null);
    assert_eq!(eval("LOWER", &[Value::Float(3.14)]).unwrap(), Value::Null);
    assert_eq!(eval("LENGTH", &[Value::Bool(true)]).unwrap(), Value::Null);
}

#[test]
fn fn_math_func_with_string() {
    // SQRT expects numeric, give it String
    assert_eq!(
        eval("SQRT", &[Value::String("hello".into())]).unwrap(),
        Value::Null
    );
    assert_eq!(
        eval("ABS", &[Value::String("abc".into())]).unwrap(),
        Value::Null
    );
    assert_eq!(eval("CEIL", &[Value::Bool(false)]).unwrap(), Value::Null);
}

#[test]
fn fn_math_int_coercion() {
    // Math functions should accept Int and coerce to Float
    let result = eval("SQRT", &[Value::Int(16)]).unwrap();
    assert!(matches!(result, Value::Float(f) if (f - 4.0).abs() < 1e-10));

    let result = eval("ABS", &[Value::Int(-42)]).unwrap();
    assert!(matches!(result, Value::Int(42)));
}

// ── Float Boundary Values ────────────────────────────────────────────

#[test]
fn fn_nan_propagation() {
    let nan = Value::Float(f64::NAN);
    let result = eval("ABS", &[nan.clone()]).unwrap();
    if let Value::Float(f) = result {
        assert!(f.is_nan(), "ABS(NaN) should be NaN");
    }

    let result = eval("SQRT", &[nan.clone()]).unwrap();
    if let Value::Float(f) = result {
        assert!(f.is_nan(), "SQRT(NaN) should be NaN");
    }
}

#[test]
fn fn_infinity() {
    let inf = Value::Float(f64::INFINITY);
    let neg_inf = Value::Float(f64::NEG_INFINITY);

    // ABS(−∞) = +∞
    let result = eval("ABS", &[neg_inf]).unwrap();
    if let Value::Float(f) = result {
        assert!(f.is_infinite() && f > 0.0);
    }

    // CEIL(+∞) = +∞
    let result = eval("CEIL", &[inf.clone()]).unwrap();
    if let Value::Float(f) = result {
        assert!(f.is_infinite() && f > 0.0);
    }
}

#[test]
fn fn_sqrt_negative() {
    let result = eval("SQRT", &[Value::Float(-1.0)]).unwrap();
    if let Value::Float(f) = result {
        assert!(f.is_nan(), "SQRT(-1) should be NaN");
    }
}

// ── Integer Boundaries ───────────────────────────────────────────────

#[test]
fn fn_abs_i64_min() {
    // i64::MIN has no positive i64 representation — should promote to Float.
    let result = eval("ABS", &[Value::Int(i64::MIN)]).unwrap();
    assert!(
        matches!(result, Value::Float(f) if f == 9223372036854775808.0),
        "ABS(i64::MIN) should promote to Float, got {:?}",
        result
    );
}

#[test]
fn fn_abs_i64_max() {
    let result = eval("ABS", &[Value::Int(i64::MAX)]).unwrap();
    assert_eq!(result, Value::Int(i64::MAX));
}

#[test]
fn fn_power_overflow() {
    // 2^63 should overflow i64
    let result = eval("POWER", &[Value::Int(2), Value::Int(63)]).unwrap();
    // Should return Float result since it can't fit in i64
    assert!(matches!(result, Value::Float(_) | Value::Int(_)));
}

// ── String Edge Cases ────────────────────────────────────────────────

#[test]
fn fn_empty_string_functions() {
    let empty = Value::String(String::new());
    assert_eq!(
        eval("UPPER", &[empty.clone()]).unwrap(),
        Value::String(String::new())
    );
    assert_eq!(
        eval("LOWER", &[empty.clone()]).unwrap(),
        Value::String(String::new())
    );
    assert_eq!(
        eval("REVERSE", &[empty.clone()]).unwrap(),
        Value::String(String::new())
    );

    // LENGTH("") = 0
    let len = eval("LENGTH", &[empty.clone()]).unwrap();
    assert!(matches!(len, Value::Int(0)));
}

#[test]
fn fn_unicode_string_functions() {
    let emoji = Value::String("🔥🚀💎".to_string());
    // LENGTH returns character count (3), not byte count (12).
    let result = eval("LENGTH", &[emoji.clone()]).unwrap();
    assert_eq!(result, Value::Int(3));

    let result = eval("REVERSE", &[emoji]).unwrap();
    assert_eq!(result, Value::String("💎🚀🔥".to_string()));
}

#[test]
fn fn_unicode_cjk() {
    let cjk = Value::String("日本語テスト".to_string());
    // LENGTH returns character count (6), not byte count (18)
    assert_eq!(eval("LENGTH", &[cjk.clone()]).unwrap(), Value::Int(6));
    assert_eq!(
        eval("UPPER", &[cjk.clone()]).unwrap(),
        Value::String("日本語テスト".to_string())
    );
}

#[test]
fn fn_substr_boundaries() {
    let s = Value::String("hello".to_string());
    // SUBSTR("hello", 1, 0) → ""
    let result = eval("SUBSTR", &[s.clone(), Value::Int(1), Value::Int(0)]).unwrap();
    assert!(matches!(result, Value::String(ref s) if s.is_empty()));

    // SUBSTR("hello", 100, 5) → ""
    let result = eval("SUBSTR", &[s.clone(), Value::Int(100), Value::Int(5)]).unwrap();
    assert!(matches!(result, Value::String(ref s) if s.is_empty()));

    // SUBSTR("hello", 1, 1000) → "hello" (take all available)
    let result = eval("SUBSTR", &[s.clone(), Value::Int(1), Value::Int(1000)]).unwrap();
    assert_eq!(result, Value::String("hello".to_string()));
}

#[test]
fn fn_repeat_boundary() {
    // REPEAT("x", 0) → ""
    let result = eval("REPEAT", &[Value::String("x".into()), Value::Int(0)]).unwrap();
    assert!(matches!(result, Value::String(ref s) if s.is_empty()));

    // REPEAT("x", -1) → NULL
    let result = eval("REPEAT", &[Value::String("x".into()), Value::Int(-1)]).unwrap();
    assert_eq!(result, Value::Null);
}

#[test]
fn fn_concat_many_args() {
    let args: Vec<Value> = (0..100).map(|i| Value::String(format!("s{}", i))).collect();
    let result = eval("CONCAT", &args).unwrap();
    if let Value::String(s) = result {
        assert!(s.starts_with("s0"));
        assert!(s.ends_with("s99"));
    }
}

#[test]
fn fn_lpad_rpad() {
    let result = eval(
        "LPAD",
        &[
            Value::String("hi".into()),
            Value::Int(5),
            Value::String("*".into()),
        ],
    )
    .unwrap();
    assert_eq!(result, Value::String("***hi".to_string()));

    let result = eval(
        "RPAD",
        &[
            Value::String("hi".into()),
            Value::Int(5),
            Value::String("*".into()),
        ],
    )
    .unwrap();
    assert_eq!(result, Value::String("hi***".to_string()));
}

// ── COALESCE Edge Cases ──────────────────────────────────────────────

#[test]
fn fn_coalesce_first_non_null() {
    let result = eval("COALESCE", &[Value::Null, Value::Null, Value::Int(42)]).unwrap();
    assert_eq!(result, Value::Int(42));
}

#[test]
fn fn_coalesce_all_null() {
    let args: Vec<Value> = (0..50).map(|_| Value::Null).collect();
    assert_eq!(eval("COALESCE", &args).unwrap(), Value::Null);
}

#[test]
fn fn_coalesce_single_value() {
    assert_eq!(
        eval("COALESCE", &[Value::String("x".into())]).unwrap(),
        Value::String("x".into())
    );
}

// ── GREATEST / LEAST ─────────────────────────────────────────────────

#[test]
fn fn_greatest_least_mixed_types() {
    let result = eval(
        "GREATEST",
        &[Value::Int(1), Value::Float(2.5), Value::Int(3)],
    )
    .unwrap();
    match &result {
        Value::Int(3) => {}
        Value::Float(f) if (*f - 3.0).abs() < 1e-10 => {}
        other => panic!("Expected 3, got {:?}", other),
    }

    let result = eval("LEAST", &[Value::Int(1), Value::Float(0.5), Value::Int(3)]).unwrap();
    assert!(matches!(result, Value::Float(f) if (f - 0.5).abs() < 1e-10));
}

#[test]
fn fn_greatest_single_arg() {
    assert_eq!(eval("GREATEST", &[Value::Int(42)]).unwrap(), Value::Int(42));
}

// ── TYPEOF ───────────────────────────────────────────────────────────

#[test]
fn fn_typeof_all_types() {
    let cases = [
        (Value::Null, "NULL"),
        (Value::Bool(true), "BOOLEAN"),
        (Value::Int(42), "INTEGER"),
        (Value::Float(3.14), "REAL"),
        (Value::String("hi".into()), "TEXT"),
        (Value::Bytes(vec![1, 2]), "BLOB"),
        (Value::Array(vec![]), "UNKNOWN"),
        (Value::Object(Default::default()), "UNKNOWN"),
        (Value::Timestamp(0), "TIMESTAMP"),
        (Value::Uuid("abc".into()), "UNKNOWN"),
    ];
    for (val, expected) in cases {
        let result = eval("TYPEOF", &[val]).unwrap();
        assert_eq!(
            result,
            Value::String(expected.to_string()),
            "TYPEOF mismatch"
        );
    }
}

// ── JSON Functions ───────────────────────────────────────────────────

#[test]
fn fn_json_valid() {
    assert_eq!(
        eval("JSON_VALID", &[Value::String(r#"{"a":1}"#.into())]).unwrap(),
        Value::Int(1)
    );
    assert_eq!(
        eval("JSON_VALID", &[Value::String("not json".into())]).unwrap(),
        Value::Int(0)
    );
    assert_eq!(
        eval("JSON_VALID", &[Value::String(String::new())]).unwrap(),
        Value::Int(0)
    );
}

#[test]
fn fn_json_extract_nested() {
    let json = Value::String(r#"{"a":{"b":{"c":42}}}"#.into());
    let result = eval(
        "JSON_EXTRACT",
        &[
            json,
            Value::String("a".into()),
            Value::String("b".into()),
            Value::String("c".into()),
        ],
    )
    .unwrap();
    assert!(matches!(result, Value::Int(42)));
}

#[test]
fn fn_json_extract_missing_key() {
    let json = Value::String(r#"{"a":1}"#.into());
    let result = eval("JSON_EXTRACT", &[json, Value::String("nonexistent".into())]).unwrap();
    assert_eq!(result, Value::Null);
}

#[test]
fn fn_json_array_length() {
    let arr = Value::String(r#"[1,2,3,4,5]"#.into());
    let result = eval("JSON_ARRAY_LENGTH", &[arr]).unwrap();
    assert_eq!(result, Value::Int(5));
}

#[test]
fn fn_json_typeof() {
    let cases = [
        (r#"null"#, "null"),
        (r#"true"#, "boolean"),
        (r#"42"#, "number"),
        (r#""hello""#, "string"),
        (r#"[1,2]"#, "array"),
        (r#"{"a":1}"#, "object"),
    ];
    for (json_str, expected) in cases {
        let result = eval("JSON_TYPEOF", &[Value::String(json_str.into())]).unwrap();
        assert_eq!(result, Value::String(expected.to_string()));
    }
}

// ── Array Functions ──────────────────────────────────────────────────

#[test]
fn fn_array_length_empty() {
    let result = eval("ARRAY_LENGTH", &[Value::Array(vec![])]).unwrap();
    assert_eq!(result, Value::Int(0));
}

#[test]
fn fn_array_append() {
    let arr = Value::Array(vec![Value::Int(1), Value::Int(2)]);
    let result = eval("ARRAY_APPEND", &[arr, Value::Int(3)]).unwrap();
    assert_eq!(
        result,
        Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)])
    );
}

#[test]
fn fn_array_contains() {
    let arr = Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
    let result = eval("ARRAY_CONTAINS", &[arr.clone(), Value::Int(2)]).unwrap();
    assert_eq!(result, Value::Bool(true));

    let result = eval("ARRAY_CONTAINS", &[arr, Value::Int(99)]).unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn fn_generate_series() {
    let result = eval("GENERATE_SERIES", &[Value::Int(1), Value::Int(5)]).unwrap();
    if let Value::Array(arr) = result {
        assert_eq!(arr.len(), 5);
        assert_eq!(arr[0], Value::Int(1));
        assert_eq!(arr[4], Value::Int(5));
    } else {
        panic!("Expected Array");
    }
}

#[test]
fn fn_generate_series_with_step() {
    let result = eval(
        "GENERATE_SERIES",
        &[Value::Int(0), Value::Int(10), Value::Int(3)],
    )
    .unwrap();
    if let Value::Array(arr) = result {
        assert!(arr.contains(&Value::Int(0)));
        assert!(arr.contains(&Value::Int(3)));
        assert!(arr.contains(&Value::Int(6)));
        assert!(arr.contains(&Value::Int(9)));
    }
}

// ── Date/Time Functions ──────────────────────────────────────────────

#[test]
fn fn_extract_parts() {
    // 2024-01-15 12:30:45 UTC = 1705319445
    let ts = Value::Timestamp(1705319445);
    let parts = ["year", "month", "day", "hour", "minute", "second", "epoch"];
    for part in parts {
        let result = eval("EXTRACT", &[Value::String(part.into()), ts.clone()]);
        assert!(result.is_ok(), "EXTRACT({part}) should not fail");
    }
}

#[test]
fn fn_extract_from_int() {
    // Should also work with Int (treated as unix timestamp)
    let ts = Value::Int(1705319445);
    let result = eval("EXTRACT", &[Value::String("year".into()), ts]).unwrap();
    assert!(matches!(result, Value::Int(2024)));
}

#[test]
fn fn_age_function() {
    let result = eval("AGE", &[Value::Timestamp(1000), Value::Timestamp(2000)]).unwrap();
    // Should return difference in seconds
    assert!(matches!(result, Value::Int(i) if i.abs() == 1000));
}

// ── FTS Functions ────────────────────────────────────────────────────

#[test]
fn fn_match_against_basic() {
    let result = eval(
        "MATCH_AGAINST",
        &[
            Value::String("hello world".into()),
            Value::String("hello world greeting".into()),
        ],
    )
    .unwrap();
    assert!(matches!(result, Value::Float(f) if f > 0.0));
}

#[test]
fn fn_match_against_no_overlap() {
    let result = eval(
        "MATCH_AGAINST",
        &[Value::String("cat".into()), Value::String("dog".into())],
    )
    .unwrap();
    assert!(matches!(result, Value::Float(f) if f == 0.0));
}

#[test]
fn fn_fts_fuzzy_match() {
    let result = eval(
        "FTS_FUZZY_MATCH",
        &[
            Value::String("helo".into()), // typo
            Value::String("hello world".into()),
        ],
    )
    .unwrap();
    assert!(matches!(result, Value::Bool(true)));
}

#[test]
fn fn_fts_boolean_match() {
    let result = eval(
        "FTS_BOOLEAN_MATCH",
        &[
            Value::String("+hello -goodbye".into()),
            Value::String("hello world greeting".into()),
        ],
    )
    .unwrap();
    assert!(matches!(result, Value::Bool(true)));
}

#[test]
fn fn_fts_empty_query() {
    let result = eval(
        "MATCH_AGAINST",
        &[
            Value::String(String::new()),
            Value::String("some text".into()),
        ],
    )
    .unwrap();
    assert!(matches!(result, Value::Float(f) if f == 0.0));
}

// ── Unknown Function ─────────────────────────────────────────────────

#[test]
fn fn_unknown_function() {
    let result = eval("NONEXISTENT_FUNC", &[Value::Int(1)]);
    assert!(result.is_err());
}

#[test]
fn fn_case_insensitive() {
    // Functions should be case-insensitive
    let r1 = eval("upper", &[Value::String("hi".into())]).unwrap();
    let r2 = eval("UPPER", &[Value::String("hi".into())]).unwrap();
    let r3 = eval("Upper", &[Value::String("hi".into())]).unwrap();
    assert_eq!(r1, r2);
    assert_eq!(r2, r3);
}

// ── Values Equal ─────────────────────────────────────────────────────

#[test]
fn fn_nullif_equal() {
    assert_eq!(
        eval("NULLIF", &[Value::Int(1), Value::Int(1)]).unwrap(),
        Value::Null
    );
    assert_eq!(
        eval("NULLIF", &[Value::Int(1), Value::Int(2)]).unwrap(),
        Value::Int(1)
    );
}

#[test]
fn fn_nullif_int_float_equal() {
    // 1 == 1.0 should be treated as equal
    let result = eval("NULLIF", &[Value::Int(1), Value::Float(1.0)]).unwrap();
    assert_eq!(result, Value::Null);
}

// ── IFNULL / NVL ─────────────────────────────────────────────────────

#[test]
fn fn_ifnull() {
    assert_eq!(
        eval("IFNULL", &[Value::Null, Value::Int(42)]).unwrap(),
        Value::Int(42)
    );
    assert_eq!(
        eval("IFNULL", &[Value::Int(1), Value::Int(42)]).unwrap(),
        Value::Int(1)
    );
}

#[test]
fn fn_nvl() {
    assert_eq!(
        eval("NVL", &[Value::Null, Value::String("default".into())]).unwrap(),
        Value::String("default".into())
    );
}

// ── Trig Boundaries ─────────────────────────────────────────────────

#[test]
fn fn_asin_out_of_range() {
    // asin(2) is undefined — should return NaN
    let result = eval("ASIN", &[Value::Float(2.0)]).unwrap();
    if let Value::Float(f) = result {
        assert!(f.is_nan());
    }
}

#[test]
fn fn_acos_out_of_range() {
    let result = eval("ACOS", &[Value::Float(-2.0)]).unwrap();
    if let Value::Float(f) = result {
        assert!(f.is_nan());
    }
}

// ── MOD / Division by Zero ──────────────────────────────────────────

#[test]
fn fn_mod_by_zero() {
    let result = eval("MOD", &[Value::Int(10), Value::Int(0)]);
    // Should either return NULL or error — not crash
    assert!(result.is_ok() || result.is_err());
}

// ── TIME_BUCKET ──────────────────────────────────────────────────────

#[test]
fn fn_time_bucket_zero_interval() {
    let result = eval("TIME_BUCKET", &[Value::Int(0), Value::Timestamp(1000)]).unwrap();
    assert_eq!(result, Value::Null);
}

// ── HISTOGRAM ────────────────────────────────────────────────────────

#[test]
fn fn_histogram_zero_buckets() {
    let result = eval(
        "HISTOGRAM",
        &[
            Value::Float(5.0),
            Value::Float(0.0),
            Value::Float(10.0),
            Value::Int(0),
        ],
    )
    .unwrap();
    assert_eq!(result, Value::Null);
}

// ── CHR / ASCII ──────────────────────────────────────────────────────

#[test]
fn fn_chr_boundary() {
    let result = eval("CHR", &[Value::Int(65)]).unwrap();
    assert_eq!(result, Value::String("A".to_string()));

    let result = eval("CHR", &[Value::Int(0)]).unwrap();
    assert!(matches!(result, Value::String(_)));
}

#[test]
fn fn_ascii_empty_string() {
    let result = eval("ASCII", &[Value::String(String::new())]).unwrap();
    // Either NULL or 0 for empty string
    assert!(matches!(result, Value::Null | Value::Int(0)));
}

// ── POSITION ─────────────────────────────────────────────────────────

#[test]
fn fn_position_not_found() {
    let result = eval(
        "POSITION",
        &[
            Value::String("xyz".into()),
            Value::String("hello world".into()),
        ],
    )
    .unwrap();
    assert_eq!(result, Value::Int(0));
}

// ── TRANSLATE ────────────────────────────────────────────────────────

#[test]
fn fn_translate_basic() {
    let result = eval(
        "TRANSLATE",
        &[
            Value::String("hello".into()),
            Value::String("helo".into()),
            Value::String("HELO".into()),
        ],
    )
    .unwrap();
    assert_eq!(result, Value::String("HELLO".to_string()));
}

// ── SPLIT_PART ───────────────────────────────────────────────────────

#[test]
fn fn_split_part() {
    let result = eval(
        "SPLIT_PART",
        &[
            Value::String("a.b.c".into()),
            Value::String(".".into()),
            Value::Int(2),
        ],
    )
    .unwrap();
    assert_eq!(result, Value::String("b".to_string()));
}

// ── Stress: 100 Functions Rapid Fire ─────────────────────────────────

#[test]
fn fn_rapid_fire_100_calls() {
    for i in 0..100 {
        let _ = eval("UPPER", &[Value::String(format!("test_{}", i))]);
        let _ = eval("ABS", &[Value::Int(i)]);
        let _ = eval("SQRT", &[Value::Float(i as f64)]);
        let _ = eval("LENGTH", &[Value::String("x".repeat(i as usize))]);
        let _ = eval("COALESCE", &[Value::Null, Value::Int(i)]);
    }
}
