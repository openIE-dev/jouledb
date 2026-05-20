//! Newtype pattern helpers — newtype wrappers with validation, Display/FromStr
//! delegation, arithmetic delegation, comparison delegation, serde transparent
//! support, collection of newtype utilities, and branded types.
//!
//! Replaces TypeScript branded types (Nominal<T>), opaque types, and
//! newtype boilerplate with a pure-Rust newtype system.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::str::FromStr;

// ── Errors ──────────────────────────────────────────────────────

/// Errors from newtype operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NewtypeError {
    /// Validation failed.
    ValidationFailed { type_name: String, reason: String },
    /// Parse failed.
    ParseFailed { type_name: String, input: String, reason: String },
    /// Arithmetic overflow.
    Overflow { type_name: String, operation: String },
}

impl fmt::Display for NewtypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ValidationFailed { type_name, reason } => {
                write!(f, "{type_name} validation failed: {reason}")
            }
            Self::ParseFailed { type_name, input, reason } => {
                write!(f, "failed to parse {type_name} from \"{input}\": {reason}")
            }
            Self::Overflow { type_name, operation } => {
                write!(f, "{type_name} overflow in {operation}")
            }
        }
    }
}

impl std::error::Error for NewtypeError {}

// ── Validated Newtype ───────────────────────────────────────────

/// A validated newtype wrapper. Construction enforces a validator function.
/// The inner value is immutable after construction.
#[derive(Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Validated<T> {
    value: T,
}

impl<T: fmt::Debug> fmt::Debug for Validated<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Validated({:?})", self.value)
    }
}

impl<T: PartialEq> PartialEq for Validated<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T: Eq> Eq for Validated<T> {}

impl<T: PartialOrd> PartialOrd for Validated<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<T: Ord> Ord for Validated<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

impl<T: Hash> Hash for Validated<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl<T: fmt::Display> fmt::Display for Validated<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.value.fmt(f)
    }
}

impl<T> Validated<T> {
    /// Create a validated newtype. Returns error if the validator rejects.
    pub fn new(value: T, validator: impl FnOnce(&T) -> Result<(), String>) -> Result<Self, NewtypeError> {
        validator(&value).map_err(|reason| NewtypeError::ValidationFailed {
            type_name: std::any::type_name::<T>().to_string(),
            reason,
        })?;
        Ok(Self { value })
    }

    /// Create without validation (for trusted inputs).
    pub fn new_unchecked(value: T) -> Self {
        Self { value }
    }

    /// Borrow the inner value.
    pub fn inner(&self) -> &T {
        &self.value
    }

    /// Consume and return the inner value.
    pub fn into_inner(self) -> T {
        self.value
    }

    /// Map the inner value through a function, re-validating.
    pub fn map<U>(
        self,
        f: impl FnOnce(T) -> U,
        validator: impl FnOnce(&U) -> Result<(), String>,
    ) -> Result<Validated<U>, NewtypeError> {
        Validated::new(f(self.value), validator)
    }

    /// Map without re-validation.
    pub fn map_unchecked<U>(self, f: impl FnOnce(T) -> U) -> Validated<U> {
        Validated::new_unchecked(f(self.value))
    }
}

impl<T: FromStr> Validated<T> {
    /// Parse from a string, then validate.
    pub fn parse(
        s: &str,
        validator: impl FnOnce(&T) -> Result<(), String>,
    ) -> Result<Self, NewtypeError>
    where
        T: FromStr,
        T::Err: fmt::Display,
    {
        let value = s.parse::<T>().map_err(|e| NewtypeError::ParseFailed {
            type_name: std::any::type_name::<T>().to_string(),
            input: s.to_string(),
            reason: e.to_string(),
        })?;
        Self::new(value, validator)
    }
}

// ── Branded Type ────────────────────────────────────────────────

/// A branded type: a value tagged with a phantom brand for type-level
/// distinction. `Branded<i32, UserId>` and `Branded<i32, ProductId>`
/// are different types even though both wrap `i32`.
#[derive(Serialize, Deserialize)]
#[serde(transparent)]
pub struct Branded<T, Brand> {
    value: T,
    #[serde(skip)]
    _brand: PhantomData<Brand>,
}

impl<T: Clone, Brand> Clone for Branded<T, Brand> {
    fn clone(&self) -> Self {
        Self { value: self.value.clone(), _brand: PhantomData }
    }
}

impl<T: Copy, Brand> Copy for Branded<T, Brand> {}

impl<T: fmt::Debug, Brand> fmt::Debug for Branded<T, Brand> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Branded({:?})", self.value)
    }
}

impl<T: PartialEq, Brand> PartialEq for Branded<T, Brand> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T: Eq, Brand> Eq for Branded<T, Brand> {}

impl<T: PartialOrd, Brand> PartialOrd for Branded<T, Brand> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<T: Ord, Brand> Ord for Branded<T, Brand> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

impl<T: Hash, Brand> Hash for Branded<T, Brand> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl<T: fmt::Display, Brand> fmt::Display for Branded<T, Brand> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.value.fmt(f)
    }
}

impl<T, Brand> Branded<T, Brand> {
    /// Create a branded value.
    pub fn new(value: T) -> Self {
        Self { value, _brand: PhantomData }
    }

    /// Borrow the inner value.
    pub fn inner(&self) -> &T {
        &self.value
    }

    /// Consume and return the inner value.
    pub fn into_inner(self) -> T {
        self.value
    }

    /// Map the inner value, preserving the brand.
    pub fn map(self, f: impl FnOnce(T) -> T) -> Self {
        Self::new(f(self.value))
    }

    /// Re-brand to a different brand (explicit cast).
    pub fn rebrand<NewBrand>(self) -> Branded<T, NewBrand> {
        Branded::new(self.value)
    }
}

impl<T: FromStr, Brand> FromStr for Branded<T, Brand> {
    type Err = T::Err;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::new(s.parse::<T>()?))
    }
}

// ── NonEmpty String ─────────────────────────────────────────────

/// A string that is guaranteed to be non-empty.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NonEmptyString {
    value: String,
}

impl NonEmptyString {
    /// Create a non-empty string. Returns `None` if empty.
    pub fn new(s: impl Into<String>) -> Option<Self> {
        let value = s.into();
        if value.is_empty() { None } else { Some(Self { value }) }
    }

    /// Borrow the inner string.
    pub fn as_str(&self) -> &str {
        &self.value
    }

    /// Consume and return the inner string.
    pub fn into_inner(self) -> String {
        self.value
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.value.len()
    }

    /// Append a string.
    pub fn push_str(&mut self, s: &str) {
        self.value.push_str(s);
    }
}

impl fmt::Display for NonEmptyString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.value.fmt(f)
    }
}

impl FromStr for NonEmptyString {
    type Err = NewtypeError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s).ok_or_else(|| NewtypeError::ValidationFailed {
            type_name: "NonEmptyString".to_string(),
            reason: "string is empty".to_string(),
        })
    }
}

impl AsRef<str> for NonEmptyString {
    fn as_ref(&self) -> &str {
        &self.value
    }
}

// ── Bounded Number ──────────────────────────────────────────────

/// A number constrained to a [min, max] range.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundedF64 {
    value: f64,
    min: f64,
    max: f64,
}

impl BoundedF64 {
    /// Create a bounded f64. Returns error if out of range.
    pub fn new(value: f64, min: f64, max: f64) -> Result<Self, NewtypeError> {
        if value < min || value > max {
            return Err(NewtypeError::ValidationFailed {
                type_name: "BoundedF64".to_string(),
                reason: format!("{value} is not in [{min}, {max}]"),
            });
        }
        Ok(Self { value, min, max })
    }

    /// Clamp a value into range.
    pub fn clamped(value: f64, min: f64, max: f64) -> Self {
        Self {
            value: value.clamp(min, max),
            min,
            max,
        }
    }

    /// Get the value.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Get the min.
    pub fn min(&self) -> f64 {
        self.min
    }

    /// Get the max.
    pub fn max(&self) -> f64 {
        self.max
    }

    /// Percentage within range [0.0, 1.0].
    pub fn fraction(&self) -> f64 {
        if (self.max - self.min).abs() < f64::EPSILON {
            0.0
        } else {
            (self.value - self.min) / (self.max - self.min)
        }
    }

    /// Try to set a new value.
    pub fn set(&mut self, value: f64) -> Result<(), NewtypeError> {
        if value < self.min || value > self.max {
            return Err(NewtypeError::ValidationFailed {
                type_name: "BoundedF64".to_string(),
                reason: format!("{value} is not in [{}, {}]", self.min, self.max),
            });
        }
        self.value = value;
        Ok(())
    }
}

impl fmt::Display for BoundedF64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

// ── Bounded i64 ─────────────────────────────────────────────────

/// An integer constrained to a [min, max] range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BoundedI64 {
    value: i64,
    min: i64,
    max: i64,
}

impl BoundedI64 {
    /// Create a bounded i64.
    pub fn new(value: i64, min: i64, max: i64) -> Result<Self, NewtypeError> {
        if value < min || value > max {
            return Err(NewtypeError::ValidationFailed {
                type_name: "BoundedI64".to_string(),
                reason: format!("{value} is not in [{min}, {max}]"),
            });
        }
        Ok(Self { value, min, max })
    }

    /// Clamp into range.
    pub fn clamped(value: i64, min: i64, max: i64) -> Self {
        Self { value: value.clamp(min, max), min, max }
    }

    /// Get the value.
    pub fn value(&self) -> i64 {
        self.value
    }

    /// Checked addition within bounds.
    pub fn checked_add(&self, rhs: i64) -> Result<Self, NewtypeError> {
        let result = self.value.checked_add(rhs).ok_or_else(|| NewtypeError::Overflow {
            type_name: "BoundedI64".to_string(),
            operation: "add".to_string(),
        })?;
        Self::new(result, self.min, self.max)
    }

    /// Checked subtraction within bounds.
    pub fn checked_sub(&self, rhs: i64) -> Result<Self, NewtypeError> {
        let result = self.value.checked_sub(rhs).ok_or_else(|| NewtypeError::Overflow {
            type_name: "BoundedI64".to_string(),
            operation: "sub".to_string(),
        })?;
        Self::new(result, self.min, self.max)
    }

    /// Checked multiplication within bounds.
    pub fn checked_mul(&self, rhs: i64) -> Result<Self, NewtypeError> {
        let result = self.value.checked_mul(rhs).ok_or_else(|| NewtypeError::Overflow {
            type_name: "BoundedI64".to_string(),
            operation: "mul".to_string(),
        })?;
        Self::new(result, self.min, self.max)
    }
}

impl fmt::Display for BoundedI64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

// ── Positive / NonNegative ──────────────────────────────────────

/// A strictly positive integer (> 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Positive {
    value: u64,
}

impl Positive {
    /// Create a positive integer. Returns None if zero.
    pub fn new(value: u64) -> Option<Self> {
        if value == 0 { None } else { Some(Self { value }) }
    }

    /// Get the value.
    pub fn value(&self) -> u64 {
        self.value
    }

    /// Get as usize.
    pub fn as_usize(&self) -> usize {
        self.value as usize
    }

    /// Checked addition.
    pub fn checked_add(&self, rhs: u64) -> Option<Self> {
        self.value.checked_add(rhs).and_then(Self::new)
    }

    /// Checked multiplication.
    pub fn checked_mul(&self, rhs: u64) -> Option<Self> {
        self.value.checked_mul(rhs).and_then(Self::new)
    }
}

impl fmt::Display for Positive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl From<Positive> for u64 {
    fn from(p: Positive) -> u64 {
        p.value
    }
}

// ── Percentage ──────────────────────────────────────────────────

/// A percentage value clamped to [0.0, 100.0].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Percentage {
    value: f64,
}

impl Percentage {
    /// Create a percentage, clamping to [0.0, 100.0].
    pub fn new(value: f64) -> Self {
        Self { value: value.clamp(0.0, 100.0) }
    }

    /// Create from a fraction [0.0, 1.0].
    pub fn from_fraction(fraction: f64) -> Self {
        Self::new(fraction * 100.0)
    }

    /// Get the percentage value [0.0, 100.0].
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Get as fraction [0.0, 1.0].
    pub fn as_fraction(&self) -> f64 {
        self.value / 100.0
    }

    /// Apply this percentage to a value.
    pub fn of(&self, base: f64) -> f64 {
        base * self.as_fraction()
    }

    /// Complement (100 - self).
    pub fn complement(&self) -> Self {
        Self::new(100.0 - self.value)
    }
}

impl fmt::Display for Percentage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1}%", self.value)
    }
}

// ── Identifier ──────────────────────────────────────────────────

/// An identifier string: must be non-empty, start with a letter or underscore,
/// and contain only alphanumeric chars and underscores.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Identifier {
    value: String,
}

impl Identifier {
    /// Create an identifier from a string.
    pub fn new(s: impl Into<String>) -> Result<Self, NewtypeError> {
        let value = s.into();
        if value.is_empty() {
            return Err(NewtypeError::ValidationFailed {
                type_name: "Identifier".to_string(),
                reason: "empty string".to_string(),
            });
        }
        let first = value.chars().next().unwrap();
        if !first.is_alphabetic() && first != '_' {
            return Err(NewtypeError::ValidationFailed {
                type_name: "Identifier".to_string(),
                reason: format!("must start with letter or underscore, got '{first}'"),
            });
        }
        if !value.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Err(NewtypeError::ValidationFailed {
                type_name: "Identifier".to_string(),
                reason: "must contain only alphanumeric chars and underscores".to_string(),
            });
        }
        Ok(Self { value })
    }

    /// Borrow the inner string.
    pub fn as_str(&self) -> &str {
        &self.value
    }

    /// Consume and return.
    pub fn into_inner(self) -> String {
        self.value
    }
}

impl fmt::Display for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.value.fmt(f)
    }
}

impl FromStr for Identifier {
    type Err = NewtypeError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Identifier::new(s)
    }
}

impl AsRef<str> for Identifier {
    fn as_ref(&self) -> &str {
        &self.value
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validated_ok() {
        let v = Validated::new(42i32, |x| {
            if *x > 0 { Ok(()) } else { Err("must be positive".into()) }
        });
        assert!(v.is_ok());
        assert_eq!(*v.unwrap().inner(), 42);
    }

    #[test]
    fn test_validated_fail() {
        let v = Validated::new(-1i32, |x| {
            if *x > 0 { Ok(()) } else { Err("must be positive".into()) }
        });
        assert!(v.is_err());
    }

    #[test]
    fn test_validated_unchecked() {
        let v = Validated::new_unchecked(0i32);
        assert_eq!(*v.inner(), 0);
    }

    #[test]
    fn test_validated_into_inner() {
        let v = Validated::new_unchecked("hello".to_string());
        assert_eq!(v.into_inner(), "hello");
    }

    #[test]
    fn test_validated_map() {
        let v = Validated::new_unchecked(5i32);
        let doubled = v.map(|x| x * 2, |x| {
            if *x <= 100 { Ok(()) } else { Err("too big".into()) }
        });
        assert_eq!(*doubled.unwrap().inner(), 10);
    }

    #[test]
    fn test_validated_display() {
        let v = Validated::new_unchecked(42);
        assert_eq!(v.to_string(), "42");
    }

    #[test]
    fn test_validated_parse() {
        let v = Validated::<i32>::parse("42", |x| {
            if *x > 0 { Ok(()) } else { Err("must be positive".into()) }
        });
        assert_eq!(*v.unwrap().inner(), 42);
    }

    #[test]
    fn test_branded_basic() {
        struct UserId;
        struct ProductId;
        let user: Branded<i32, UserId> = Branded::new(1);
        let _product: Branded<i32, ProductId> = Branded::new(1);
        assert_eq!(*user.inner(), 1);
    }

    #[test]
    fn test_branded_display() {
        struct Tag;
        let b: Branded<i32, Tag> = Branded::new(42);
        assert_eq!(b.to_string(), "42");
    }

    #[test]
    fn test_branded_from_str() {
        struct Tag;
        let b: Branded<i32, Tag> = "42".parse().unwrap();
        assert_eq!(*b.inner(), 42);
    }

    #[test]
    fn test_branded_into_inner() {
        struct Tag;
        let b: Branded<String, Tag> = Branded::new("hello".to_string());
        assert_eq!(b.into_inner(), "hello");
    }

    #[test]
    fn test_branded_rebrand() {
        struct A;
        struct B;
        let a: Branded<i32, A> = Branded::new(5);
        let b: Branded<i32, B> = a.rebrand();
        assert_eq!(*b.inner(), 5);
    }

    #[test]
    fn test_non_empty_string_ok() {
        let s = NonEmptyString::new("hello").unwrap();
        assert_eq!(s.as_str(), "hello");
        assert_eq!(s.len(), 5);
    }

    #[test]
    fn test_non_empty_string_empty() {
        assert!(NonEmptyString::new("").is_none());
    }

    #[test]
    fn test_non_empty_string_from_str() {
        let s: Result<NonEmptyString, _> = "hi".parse();
        assert!(s.is_ok());
        let s: Result<NonEmptyString, _> = "".parse();
        assert!(s.is_err());
    }

    #[test]
    fn test_non_empty_string_push() {
        let mut s = NonEmptyString::new("hello").unwrap();
        s.push_str(" world");
        assert_eq!(s.as_str(), "hello world");
    }

    #[test]
    fn test_bounded_f64_ok() {
        let b = BoundedF64::new(5.0, 0.0, 10.0).unwrap();
        assert_eq!(b.value(), 5.0);
        assert_eq!(b.fraction(), 0.5);
    }

    #[test]
    fn test_bounded_f64_out_of_range() {
        assert!(BoundedF64::new(11.0, 0.0, 10.0).is_err());
        assert!(BoundedF64::new(-1.0, 0.0, 10.0).is_err());
    }

    #[test]
    fn test_bounded_f64_clamped() {
        let b = BoundedF64::clamped(15.0, 0.0, 10.0);
        assert_eq!(b.value(), 10.0);
    }

    #[test]
    fn test_bounded_i64_arithmetic() {
        let b = BoundedI64::new(5, 0, 10).unwrap();
        assert_eq!(b.checked_add(3).unwrap().value(), 8);
        assert!(b.checked_add(6).is_err()); // 11 > 10
        assert_eq!(b.checked_sub(3).unwrap().value(), 2);
        assert!(b.checked_sub(6).is_err()); // -1 < 0
    }

    #[test]
    fn test_positive_ok() {
        assert!(Positive::new(1).is_some());
        assert_eq!(Positive::new(1).unwrap().value(), 1);
    }

    #[test]
    fn test_positive_zero() {
        assert!(Positive::new(0).is_none());
    }

    #[test]
    fn test_positive_checked_add() {
        let p = Positive::new(5).unwrap();
        assert_eq!(p.checked_add(3).unwrap().value(), 8);
    }

    #[test]
    fn test_percentage() {
        let p = Percentage::new(75.0);
        assert_eq!(p.value(), 75.0);
        assert_eq!(p.as_fraction(), 0.75);
        assert_eq!(p.of(200.0), 150.0);
        assert_eq!(p.complement().value(), 25.0);
    }

    #[test]
    fn test_percentage_clamping() {
        assert_eq!(Percentage::new(150.0).value(), 100.0);
        assert_eq!(Percentage::new(-10.0).value(), 0.0);
    }

    #[test]
    fn test_percentage_from_fraction() {
        let p = Percentage::from_fraction(0.5);
        assert_eq!(p.value(), 50.0);
    }

    #[test]
    fn test_percentage_display() {
        let p = Percentage::new(42.5);
        assert_eq!(p.to_string(), "42.5%");
    }

    #[test]
    fn test_identifier_ok() {
        let id = Identifier::new("foo_bar").unwrap();
        assert_eq!(id.as_str(), "foo_bar");
    }

    #[test]
    fn test_identifier_underscore_start() {
        assert!(Identifier::new("_private").is_ok());
    }

    #[test]
    fn test_identifier_digit_start() {
        assert!(Identifier::new("1bad").is_err());
    }

    #[test]
    fn test_identifier_empty() {
        assert!(Identifier::new("").is_err());
    }

    #[test]
    fn test_identifier_special_chars() {
        assert!(Identifier::new("foo-bar").is_err());
        assert!(Identifier::new("foo bar").is_err());
    }

    #[test]
    fn test_validated_ordering() {
        let a = Validated::new_unchecked(1i32);
        let b = Validated::new_unchecked(2i32);
        assert!(a < b);
    }

    #[test]
    fn test_branded_ordering() {
        struct Tag;
        let a: Branded<i32, Tag> = Branded::new(1);
        let b: Branded<i32, Tag> = Branded::new(2);
        assert!(a < b);
    }

    #[test]
    fn test_branded_map() {
        struct Tag;
        let a: Branded<i32, Tag> = Branded::new(5);
        let b = a.map(|x| x * 2);
        assert_eq!(*b.inner(), 10);
    }
}
