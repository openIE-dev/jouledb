//! Pattern matching DSL for `serde_json::Value`.
//!
//! Provides a builder-style API for matching JSON values against patterns
//! with wildcards, guards, nested destructuring, pattern variables (captures),
//! and exhaustiveness checking.

use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ── Pattern ─────────────────────────────────────────────────────────────────

/// A pattern that can be matched against a `serde_json::Value`.
#[derive(Clone)]
pub enum Pattern {
    /// Matches any value.
    Wildcard,
    /// Matches a specific literal value.
    Literal(Value),
    /// Captures the matched value under a named variable.
    Variable(String),
    /// Matches null.
    Null,
    /// Matches any boolean.
    AnyBool,
    /// Matches any number.
    AnyNumber,
    /// Matches any string.
    AnyString,
    /// Matches any array.
    AnyArray,
    /// Matches any object.
    AnyObject,
    /// Matches an array with specific element patterns (positional).
    Array(Vec<Pattern>),
    /// Matches an array where every element matches the inner pattern.
    ArrayAll(Box<Pattern>),
    /// Matches an object with specific field patterns.
    Object(Vec<(String, Pattern)>),
    /// Matches if either pattern matches (disjunction).
    Or(Box<Pattern>, Box<Pattern>),
    /// Matches if both patterns match (conjunction).
    And(Box<Pattern>, Box<Pattern>),
    /// Matches if the inner pattern does NOT match.
    Not(Box<Pattern>),
    /// Matches with a guard condition applied after the pattern.
    Guard(Box<Pattern>, GuardFn),
    /// Matches a string against a prefix.
    StringPrefix(String),
    /// Matches a string against a suffix.
    StringSuffix(String),
    /// Matches a number in range [lo, hi].
    NumberRange(f64, f64),
}

/// A guard function: takes the matched value and captured variables, returns bool.
#[derive(Clone)]
pub struct GuardFn {
    inner: fn(&Value, &Captures) -> bool,
}

impl GuardFn {
    /// Create a guard from a function pointer.
    pub fn new(f: fn(&Value, &Captures) -> bool) -> Self {
        Self { inner: f }
    }

    fn eval(&self, val: &Value, caps: &Captures) -> bool {
        (self.inner)(val, caps)
    }
}

impl fmt::Debug for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Pattern::Wildcard => write!(f, "_"),
            Pattern::Literal(v) => write!(f, "Literal({v})"),
            Pattern::Variable(name) => write!(f, "Var({name})"),
            Pattern::Null => write!(f, "Null"),
            Pattern::AnyBool => write!(f, "AnyBool"),
            Pattern::AnyNumber => write!(f, "AnyNumber"),
            Pattern::AnyString => write!(f, "AnyString"),
            Pattern::AnyArray => write!(f, "AnyArray"),
            Pattern::AnyObject => write!(f, "AnyObject"),
            Pattern::Array(pats) => write!(f, "Array({pats:?})"),
            Pattern::ArrayAll(p) => write!(f, "ArrayAll({p:?})"),
            Pattern::Object(fields) => write!(f, "Object({fields:?})"),
            Pattern::Or(a, b) => write!(f, "Or({a:?}, {b:?})"),
            Pattern::And(a, b) => write!(f, "And({a:?}, {b:?})"),
            Pattern::Not(p) => write!(f, "Not({p:?})"),
            Pattern::Guard(p, _) => write!(f, "Guard({p:?}, <fn>)"),
            Pattern::StringPrefix(s) => write!(f, "StringPrefix({s:?})"),
            Pattern::StringSuffix(s) => write!(f, "StringSuffix({s:?})"),
            Pattern::NumberRange(lo, hi) => write!(f, "NumberRange({lo}..{hi})"),
        }
    }
}

// ── Captures ────────────────────────────────────────────────────────────────

/// Captured variable bindings from a pattern match.
#[derive(Debug, Clone, Default)]
pub struct Captures {
    bindings: HashMap<String, Value>,
}

impl Captures {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a captured value by name.
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.bindings.get(name)
    }

    /// Insert a capture.
    pub fn insert(&mut self, name: String, value: Value) {
        self.bindings.insert(name, value);
    }

    /// Merge another set of captures into this one.
    pub fn merge(&mut self, other: &Captures) {
        for (k, v) in &other.bindings {
            self.bindings.insert(k.clone(), v.clone());
        }
    }

    /// Number of captured variables.
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Whether there are no captures.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// All captured variable names.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.bindings.keys().cloned().collect();
        names.sort();
        names
    }
}

// ── Pattern matching ────────────────────────────────────────────────────────

/// Try to match a value against a pattern, returning captured variables on success.
pub fn match_pattern(pattern: &Pattern, value: &Value) -> Option<Captures> {
    let mut caps = Captures::new();
    if do_match(pattern, value, &mut caps) {
        Some(caps)
    } else {
        None
    }
}

fn do_match(pattern: &Pattern, value: &Value, caps: &mut Captures) -> bool {
    match pattern {
        Pattern::Wildcard => true,
        Pattern::Literal(expected) => value == expected,
        Pattern::Variable(name) => {
            caps.insert(name.clone(), value.clone());
            true
        }
        Pattern::Null => value.is_null(),
        Pattern::AnyBool => value.is_boolean(),
        Pattern::AnyNumber => value.is_number(),
        Pattern::AnyString => value.is_string(),
        Pattern::AnyArray => value.is_array(),
        Pattern::AnyObject => value.is_object(),
        Pattern::Array(pats) => {
            if let Value::Array(arr) = value {
                if arr.len() != pats.len() {
                    return false;
                }
                for (p, v) in pats.iter().zip(arr.iter()) {
                    if !do_match(p, v, caps) {
                        return false;
                    }
                }
                true
            } else {
                false
            }
        }
        Pattern::ArrayAll(pat) => {
            if let Value::Array(arr) = value {
                for v in arr {
                    if !do_match(pat, v, caps) {
                        return false;
                    }
                }
                true
            } else {
                false
            }
        }
        Pattern::Object(fields) => {
            if let Value::Object(map) = value {
                for (key, pat) in fields {
                    match map.get(key) {
                        Some(v) => {
                            if !do_match(pat, v, caps) {
                                return false;
                            }
                        }
                        None => return false,
                    }
                }
                true
            } else {
                false
            }
        }
        Pattern::Or(a, b) => {
            let mut caps_a = Captures::new();
            if do_match(a, value, &mut caps_a) {
                caps.merge(&caps_a);
                return true;
            }
            let mut caps_b = Captures::new();
            if do_match(b, value, &mut caps_b) {
                caps.merge(&caps_b);
                return true;
            }
            false
        }
        Pattern::And(a, b) => {
            let mut caps_a = Captures::new();
            let mut caps_b = Captures::new();
            if do_match(a, value, &mut caps_a) && do_match(b, value, &mut caps_b) {
                caps.merge(&caps_a);
                caps.merge(&caps_b);
                true
            } else {
                false
            }
        }
        Pattern::Not(p) => {
            let mut tmp = Captures::new();
            !do_match(p, value, &mut tmp)
        }
        Pattern::Guard(p, guard) => {
            let mut inner_caps = Captures::new();
            if do_match(p, value, &mut inner_caps) && guard.eval(value, &inner_caps) {
                caps.merge(&inner_caps);
                true
            } else {
                false
            }
        }
        Pattern::StringPrefix(prefix) => {
            value
                .as_str()
                .map(|s| s.starts_with(prefix.as_str()))
                .unwrap_or(false)
        }
        Pattern::StringSuffix(suffix) => {
            value
                .as_str()
                .map(|s| s.ends_with(suffix.as_str()))
                .unwrap_or(false)
        }
        Pattern::NumberRange(lo, hi) => {
            value
                .as_f64()
                .map(|n| n >= *lo && n <= *hi)
                .unwrap_or(false)
        }
    }
}

// ── MatchExpr: multi-arm matcher ────────────────────────────────────────────

/// A match expression with multiple arms, evaluated top-to-bottom.
pub struct MatchExpr<T> {
    arms: Vec<MatchArm<T>>,
    default: Option<Box<dyn Fn(&Value) -> T>>,
}

struct MatchArm<T> {
    pattern: Pattern,
    handler: Box<dyn Fn(&Value, &Captures) -> T>,
}

impl<T> MatchExpr<T> {
    /// Create a new empty match expression.
    pub fn new() -> Self {
        Self {
            arms: Vec::new(),
            default: None,
        }
    }

    /// Add a match arm with a pattern and handler.
    pub fn arm(
        mut self,
        pattern: Pattern,
        handler: impl Fn(&Value, &Captures) -> T + 'static,
    ) -> Self {
        self.arms.push(MatchArm {
            pattern,
            handler: Box::new(handler),
        });
        self
    }

    /// Set the default (wildcard / else) handler.
    pub fn default(mut self, handler: impl Fn(&Value) -> T + 'static) -> Self {
        self.default = Some(Box::new(handler));
        self
    }

    /// Evaluate the match expression against a value.
    /// Returns `None` if no arm matches and no default is set.
    pub fn eval(&self, value: &Value) -> Option<T> {
        for arm in &self.arms {
            if let Some(caps) = match_pattern(&arm.pattern, value) {
                return Some((arm.handler)(value, &caps));
            }
        }
        self.default.as_ref().map(|d| d(value))
    }

    /// Number of arms (excluding default).
    pub fn arm_count(&self) -> usize {
        self.arms.len()
    }

    /// Whether a default handler is set.
    pub fn has_default(&self) -> bool {
        self.default.is_some()
    }
}

impl<T> Default for MatchExpr<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Exhaustiveness checking ─────────────────────────────────────────────────

/// The set of JSON value categories we check for exhaustiveness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueCategory {
    Null,
    Bool,
    Number,
    String,
    Array,
    Object,
}

impl ValueCategory {
    pub fn all() -> Vec<Self> {
        vec![
            Self::Null,
            Self::Bool,
            Self::Number,
            Self::String,
            Self::Array,
            Self::Object,
        ]
    }
}

/// Check which value categories a set of patterns covers.
pub fn covered_categories(patterns: &[Pattern]) -> Vec<ValueCategory> {
    let mut covered = std::collections::HashSet::new();
    for p in patterns {
        collect_covered(p, &mut covered);
    }
    let mut result: Vec<ValueCategory> = covered.into_iter().collect();
    result.sort_by_key(|c| format!("{c:?}"));
    result
}

fn collect_covered(pattern: &Pattern, covered: &mut std::collections::HashSet<ValueCategory>) {
    match pattern {
        Pattern::Wildcard | Pattern::Variable(_) => {
            for c in ValueCategory::all() {
                covered.insert(c);
            }
        }
        Pattern::Null => { covered.insert(ValueCategory::Null); }
        Pattern::AnyBool => { covered.insert(ValueCategory::Bool); }
        Pattern::AnyNumber | Pattern::NumberRange(_, _) => { covered.insert(ValueCategory::Number); }
        Pattern::AnyString | Pattern::StringPrefix(_) | Pattern::StringSuffix(_) => {
            covered.insert(ValueCategory::String);
        }
        Pattern::AnyArray | Pattern::Array(_) | Pattern::ArrayAll(_) => {
            covered.insert(ValueCategory::Array);
        }
        Pattern::AnyObject | Pattern::Object(_) => { covered.insert(ValueCategory::Object); }
        Pattern::Literal(v) => {
            let cat = match v {
                Value::Null => ValueCategory::Null,
                Value::Bool(_) => ValueCategory::Bool,
                Value::Number(_) => ValueCategory::Number,
                Value::String(_) => ValueCategory::String,
                Value::Array(_) => ValueCategory::Array,
                Value::Object(_) => ValueCategory::Object,
            };
            covered.insert(cat);
        }
        Pattern::Or(a, b) => {
            collect_covered(a, covered);
            collect_covered(b, covered);
        }
        Pattern::And(a, _) => {
            // And is restrictive — only the first pattern's categories count.
            collect_covered(a, covered);
        }
        Pattern::Not(_) => {
            // Not is hard to reason about statically, treat as covering nothing.
        }
        Pattern::Guard(p, _) => {
            // Guards can fail, so they don't guarantee coverage.
            // But we conservatively mark what the inner pattern covers.
            collect_covered(p, covered);
        }
    }
}

/// Check if a set of patterns is exhaustive (covers all value categories).
pub fn is_exhaustive(patterns: &[Pattern]) -> bool {
    let covered = covered_categories(patterns);
    covered.len() == ValueCategory::all().len()
}

/// Return the missing categories.
pub fn missing_categories(patterns: &[Pattern]) -> Vec<ValueCategory> {
    let covered: std::collections::HashSet<_> =
        covered_categories(patterns).into_iter().collect();
    let mut missing: Vec<ValueCategory> = ValueCategory::all()
        .into_iter()
        .filter(|c| !covered.contains(c))
        .collect();
    missing.sort_by_key(|c| format!("{c:?}"));
    missing
}

// ── Builder helpers ─────────────────────────────────────────────────────────

/// Create a literal pattern.
pub fn lit(v: Value) -> Pattern {
    Pattern::Literal(v)
}

/// Create a variable pattern.
pub fn var(name: &str) -> Pattern {
    Pattern::Variable(name.to_string())
}

/// Create an object pattern from field-pattern pairs.
pub fn obj(fields: Vec<(&str, Pattern)>) -> Pattern {
    Pattern::Object(
        fields
            .into_iter()
            .map(|(k, p)| (k.to_string(), p))
            .collect(),
    )
}

/// Create an array pattern.
pub fn arr(pats: Vec<Pattern>) -> Pattern {
    Pattern::Array(pats)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn wildcard_matches_anything() {
        assert!(match_pattern(&Pattern::Wildcard, &json!(42)).is_some());
        assert!(match_pattern(&Pattern::Wildcard, &json!("hi")).is_some());
        assert!(match_pattern(&Pattern::Wildcard, &json!(null)).is_some());
    }

    #[test]
    fn literal_match() {
        assert!(match_pattern(&lit(json!(42)), &json!(42)).is_some());
        assert!(match_pattern(&lit(json!(42)), &json!(43)).is_none());
        assert!(match_pattern(&lit(json!("hello")), &json!("hello")).is_some());
    }

    #[test]
    fn null_pattern() {
        assert!(match_pattern(&Pattern::Null, &json!(null)).is_some());
        assert!(match_pattern(&Pattern::Null, &json!(0)).is_none());
    }

    #[test]
    fn type_patterns() {
        assert!(match_pattern(&Pattern::AnyBool, &json!(true)).is_some());
        assert!(match_pattern(&Pattern::AnyBool, &json!(1)).is_none());
        assert!(match_pattern(&Pattern::AnyNumber, &json!(3.14)).is_some());
        assert!(match_pattern(&Pattern::AnyString, &json!("hi")).is_some());
        assert!(match_pattern(&Pattern::AnyArray, &json!([1, 2])).is_some());
        assert!(match_pattern(&Pattern::AnyObject, &json!({"a": 1})).is_some());
    }

    #[test]
    fn variable_captures() {
        let caps = match_pattern(&var("x"), &json!(42)).unwrap();
        assert_eq!(caps.get("x"), Some(&json!(42)));
    }

    #[test]
    fn array_pattern() {
        let pat = arr(vec![lit(json!(1)), var("second")]);
        let caps = match_pattern(&pat, &json!([1, 2])).unwrap();
        assert_eq!(caps.get("second"), Some(&json!(2)));
        assert!(match_pattern(&pat, &json!([1, 2, 3])).is_none()); // wrong length
    }

    #[test]
    fn array_all_pattern() {
        let pat = Pattern::ArrayAll(Box::new(Pattern::AnyNumber));
        assert!(match_pattern(&pat, &json!([1, 2, 3])).is_some());
        assert!(match_pattern(&pat, &json!([1, "x", 3])).is_none());
        assert!(match_pattern(&pat, &json!([])).is_some()); // vacuously true
    }

    #[test]
    fn object_pattern() {
        let pat = obj(vec![
            ("name", var("n")),
            ("age", Pattern::AnyNumber),
        ]);
        let val = json!({"name": "Alice", "age": 30, "extra": true});
        let caps = match_pattern(&pat, &val).unwrap();
        assert_eq!(caps.get("n"), Some(&json!("Alice")));
    }

    #[test]
    fn object_pattern_missing_field() {
        let pat = obj(vec![("name", Pattern::Wildcard)]);
        assert!(match_pattern(&pat, &json!({"age": 30})).is_none());
    }

    #[test]
    fn nested_pattern() {
        let pat = obj(vec![
            ("user", obj(vec![
                ("name", var("name")),
                ("address", obj(vec![("city", var("city"))])),
            ])),
        ]);
        let val = json!({
            "user": {
                "name": "Alice",
                "address": {"city": "Boston"}
            }
        });
        let caps = match_pattern(&pat, &val).unwrap();
        assert_eq!(caps.get("name"), Some(&json!("Alice")));
        assert_eq!(caps.get("city"), Some(&json!("Boston")));
    }

    #[test]
    fn or_pattern() {
        let pat = Pattern::Or(
            Box::new(lit(json!(1))),
            Box::new(lit(json!(2))),
        );
        assert!(match_pattern(&pat, &json!(1)).is_some());
        assert!(match_pattern(&pat, &json!(2)).is_some());
        assert!(match_pattern(&pat, &json!(3)).is_none());
    }

    #[test]
    fn and_pattern() {
        let pat = Pattern::And(
            Box::new(Pattern::AnyNumber),
            Box::new(Pattern::NumberRange(0.0, 100.0)),
        );
        assert!(match_pattern(&pat, &json!(50)).is_some());
        assert!(match_pattern(&pat, &json!(150)).is_none());
        assert!(match_pattern(&pat, &json!("hi")).is_none());
    }

    #[test]
    fn not_pattern() {
        let pat = Pattern::Not(Box::new(Pattern::Null));
        assert!(match_pattern(&pat, &json!(42)).is_some());
        assert!(match_pattern(&pat, &json!(null)).is_none());
    }

    #[test]
    fn guard_pattern() {
        let guard = GuardFn::new(|v, _caps| {
            v.as_i64().map(|n| n > 10).unwrap_or(false)
        });
        let pat = Pattern::Guard(Box::new(Pattern::AnyNumber), guard);
        assert!(match_pattern(&pat, &json!(20)).is_some());
        assert!(match_pattern(&pat, &json!(5)).is_none());
    }

    #[test]
    fn string_prefix_suffix() {
        assert!(match_pattern(&Pattern::StringPrefix("hel".into()), &json!("hello")).is_some());
        assert!(match_pattern(&Pattern::StringPrefix("xyz".into()), &json!("hello")).is_none());
        assert!(match_pattern(&Pattern::StringSuffix("llo".into()), &json!("hello")).is_some());
    }

    #[test]
    fn number_range() {
        let pat = Pattern::NumberRange(1.0, 10.0);
        assert!(match_pattern(&pat, &json!(5)).is_some());
        assert!(match_pattern(&pat, &json!(1)).is_some());
        assert!(match_pattern(&pat, &json!(10)).is_some());
        assert!(match_pattern(&pat, &json!(0)).is_none());
        assert!(match_pattern(&pat, &json!(11)).is_none());
    }

    #[test]
    fn match_expr_first_arm() {
        let expr = MatchExpr::new()
            .arm(Pattern::AnyNumber, |v, _| format!("num: {v}"))
            .arm(Pattern::AnyString, |v, _| format!("str: {v}"))
            .default(|_| "other".to_string());
        assert_eq!(expr.eval(&json!(42)), Some("num: 42".to_string()));
        assert!(expr.eval(&json!("hi")).unwrap().starts_with("str: "));
        assert_eq!(expr.eval(&json!(null)), Some("other".to_string()));
    }

    #[test]
    fn match_expr_no_default_no_match() {
        let expr: MatchExpr<i32> = MatchExpr::new()
            .arm(Pattern::AnyNumber, |_, _| 1);
        assert_eq!(expr.eval(&json!("hi")), None);
    }

    #[test]
    fn match_expr_with_captures() {
        let expr = MatchExpr::new()
            .arm(
                obj(vec![("name", var("n"))]),
                |_v, caps| caps.get("n").unwrap().as_str().unwrap().to_string(),
            );
        assert_eq!(
            expr.eval(&json!({"name": "Alice"})),
            Some("Alice".to_string())
        );
    }

    #[test]
    fn exhaustiveness_all_covered() {
        let patterns = vec![
            Pattern::Null,
            Pattern::AnyBool,
            Pattern::AnyNumber,
            Pattern::AnyString,
            Pattern::AnyArray,
            Pattern::AnyObject,
        ];
        assert!(is_exhaustive(&patterns));
        assert!(missing_categories(&patterns).is_empty());
    }

    #[test]
    fn exhaustiveness_missing_some() {
        let patterns = vec![Pattern::AnyNumber, Pattern::AnyString];
        assert!(!is_exhaustive(&patterns));
        let missing = missing_categories(&patterns);
        assert!(missing.contains(&ValueCategory::Null));
        assert!(missing.contains(&ValueCategory::Bool));
    }

    #[test]
    fn exhaustiveness_wildcard_covers_all() {
        assert!(is_exhaustive(&[Pattern::Wildcard]));
    }

    #[test]
    fn match_expr_arm_count() {
        let expr: MatchExpr<i32> = MatchExpr::new()
            .arm(Pattern::Null, |_, _| 0)
            .arm(Pattern::AnyNumber, |_, _| 1);
        assert_eq!(expr.arm_count(), 2);
        assert!(!expr.has_default());
    }

    #[test]
    fn captures_names() {
        let pat = obj(vec![("x", var("a")), ("y", var("b"))]);
        let caps = match_pattern(&pat, &json!({"x": 1, "y": 2})).unwrap();
        assert_eq!(caps.names(), vec!["a", "b"]);
    }

    #[test]
    fn captures_len() {
        let caps = match_pattern(&var("v"), &json!(42)).unwrap();
        assert_eq!(caps.len(), 1);
        assert!(!caps.is_empty());
    }

    #[test]
    fn pattern_debug() {
        let p = Pattern::Wildcard;
        assert_eq!(format!("{p:?}"), "_");
        let p2 = Pattern::NumberRange(1.0, 10.0);
        assert!(format!("{p2:?}").contains("NumberRange"));
    }
}
