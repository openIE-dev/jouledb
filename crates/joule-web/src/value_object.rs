//! DDD Value Objects — equality by value, immutability, self-validation,
//! common value objects (Money, EmailAddress, PhoneNumber, Address,
//! DateRange, Percentage), factory methods, and conversion traits.
//!
//! Replaces ad-hoc value types in JS/TS with a pure-Rust value object
//! framework that enforces immutability and validation at construction.

use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Value object domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueError {
    /// Validation failed.
    Invalid { field: String, reason: String },
    /// Empty value not allowed.
    Empty(String),
    /// Value out of range.
    OutOfRange { field: String, min: String, max: String, actual: String },
    /// Parse error.
    ParseError(String),
    /// Currency mismatch.
    CurrencyMismatch { expected: String, actual: String },
    /// Invalid date range.
    InvalidDateRange { start: String, end: String },
}

impl fmt::Display for ValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid { field, reason } => write!(f, "invalid {field}: {reason}"),
            Self::Empty(field) => write!(f, "{field} cannot be empty"),
            Self::OutOfRange { field, min, max, actual } => {
                write!(f, "{field} out of range [{min}, {max}]: {actual}")
            }
            Self::ParseError(msg) => write!(f, "parse error: {msg}"),
            Self::CurrencyMismatch { expected, actual } => {
                write!(f, "currency mismatch: expected {expected}, got {actual}")
            }
            Self::InvalidDateRange { start, end } => {
                write!(f, "invalid date range: {start} > {end}")
            }
        }
    }
}

impl std::error::Error for ValueError {}

// ── Money ───────────────────────────────────────────────────────

/// Immutable money value object. Amount stored as cents (minor units).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Money {
    amount_cents: i64,
    currency: String,
}

impl Money {
    /// Create from a major-unit amount (e.g. dollars).
    pub fn from_major(amount: i64, currency: impl Into<String>) -> Result<Self, ValueError> {
        let currency = currency.into();
        if currency.is_empty() {
            return Err(ValueError::Empty("currency".to_string()));
        }
        Ok(Self { amount_cents: amount * 100, currency })
    }

    /// Create from minor units (e.g. cents).
    pub fn from_minor(amount_cents: i64, currency: impl Into<String>) -> Result<Self, ValueError> {
        let currency = currency.into();
        if currency.is_empty() {
            return Err(ValueError::Empty("currency".to_string()));
        }
        Ok(Self { amount_cents, currency })
    }

    /// Zero money in the given currency.
    pub fn zero(currency: impl Into<String>) -> Result<Self, ValueError> {
        Self::from_minor(0, currency)
    }

    pub fn amount_cents(&self) -> i64 {
        self.amount_cents
    }

    pub fn currency(&self) -> &str {
        &self.currency
    }

    /// Major units (e.g. dollars).
    pub fn major(&self) -> i64 {
        self.amount_cents / 100
    }

    /// Minor remainder (e.g. cents portion).
    pub fn minor(&self) -> i64 {
        self.amount_cents.abs() % 100
    }

    /// Add two money values (same currency).
    pub fn add(&self, other: &Money) -> Result<Self, ValueError> {
        if self.currency != other.currency {
            return Err(ValueError::CurrencyMismatch {
                expected: self.currency.clone(),
                actual: other.currency.clone(),
            });
        }
        Ok(Self {
            amount_cents: self.amount_cents + other.amount_cents,
            currency: self.currency.clone(),
        })
    }

    /// Subtract.
    pub fn sub(&self, other: &Money) -> Result<Self, ValueError> {
        if self.currency != other.currency {
            return Err(ValueError::CurrencyMismatch {
                expected: self.currency.clone(),
                actual: other.currency.clone(),
            });
        }
        Ok(Self {
            amount_cents: self.amount_cents - other.amount_cents,
            currency: self.currency.clone(),
        })
    }

    /// Multiply by scalar.
    pub fn multiply(&self, factor: i64) -> Self {
        Self {
            amount_cents: self.amount_cents * factor,
            currency: self.currency.clone(),
        }
    }

    /// Whether negative.
    pub fn is_negative(&self) -> bool {
        self.amount_cents < 0
    }

    /// Whether zero.
    pub fn is_zero(&self) -> bool {
        self.amount_cents == 0
    }

    /// Negate.
    pub fn negate(&self) -> Self {
        Self {
            amount_cents: -self.amount_cents,
            currency: self.currency.clone(),
        }
    }

    /// Absolute value.
    pub fn abs(&self) -> Self {
        Self {
            amount_cents: self.amount_cents.abs(),
            currency: self.currency.clone(),
        }
    }
}

impl fmt::Display for Money {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sign = if self.amount_cents < 0 { "-" } else { "" };
        let major = (self.amount_cents.abs()) / 100;
        let minor = (self.amount_cents.abs()) % 100;
        write!(f, "{sign}{} {major}.{minor:02}", self.currency)
    }
}

impl PartialOrd for Money {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.currency != other.currency {
            None
        } else {
            Some(self.amount_cents.cmp(&other.amount_cents))
        }
    }
}

// ── EmailAddress ────────────────────────────────────────────────

/// Validated email address value object.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EmailAddress {
    value: String,
}

impl EmailAddress {
    /// Create a validated email address.
    pub fn new(email: impl Into<String>) -> Result<Self, ValueError> {
        let value = email.into().trim().to_lowercase();
        if value.is_empty() {
            return Err(ValueError::Empty("email".to_string()));
        }
        let at_pos = value.find('@').ok_or_else(|| ValueError::Invalid {
            field: "email".to_string(),
            reason: "missing @ symbol".to_string(),
        })?;
        let local = &value[..at_pos];
        let domain = &value[at_pos + 1..];
        if local.is_empty() {
            return Err(ValueError::Invalid {
                field: "email".to_string(),
                reason: "empty local part".to_string(),
            });
        }
        if domain.is_empty() || !domain.contains('.') {
            return Err(ValueError::Invalid {
                field: "email".to_string(),
                reason: "invalid domain".to_string(),
            });
        }
        Ok(Self { value })
    }

    pub fn value(&self) -> &str {
        &self.value
    }

    pub fn local_part(&self) -> &str {
        &self.value[..self.value.find('@').unwrap()]
    }

    pub fn domain(&self) -> &str {
        &self.value[self.value.find('@').unwrap() + 1..]
    }
}

impl fmt::Display for EmailAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

// ── PhoneNumber ─────────────────────────────────────────────────

/// Validated phone number value object.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PhoneNumber {
    country_code: String,
    number: String,
}

impl PhoneNumber {
    /// Create a phone number with country code and number.
    pub fn new(
        country_code: impl Into<String>,
        number: impl Into<String>,
    ) -> Result<Self, ValueError> {
        let country_code = country_code.into().trim().to_string();
        let number: String = number.into().chars().filter(|c| c.is_ascii_digit()).collect();
        if country_code.is_empty() {
            return Err(ValueError::Empty("country_code".to_string()));
        }
        if number.len() < 4 {
            return Err(ValueError::Invalid {
                field: "phone_number".to_string(),
                reason: "too short (min 4 digits)".to_string(),
            });
        }
        if number.len() > 15 {
            return Err(ValueError::Invalid {
                field: "phone_number".to_string(),
                reason: "too long (max 15 digits)".to_string(),
            });
        }
        Ok(Self { country_code, number })
    }

    pub fn country_code(&self) -> &str {
        &self.country_code
    }

    pub fn number(&self) -> &str {
        &self.number
    }

    pub fn full_number(&self) -> String {
        format!("+{}{}", self.country_code, self.number)
    }
}

impl fmt::Display for PhoneNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "+{} {}", self.country_code, self.number)
    }
}

// ── Address ─────────────────────────────────────────────────────

/// Address value object.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Address {
    street: String,
    city: String,
    state: String,
    postal_code: String,
    country: String,
}

impl Address {
    pub fn new(
        street: impl Into<String>,
        city: impl Into<String>,
        state: impl Into<String>,
        postal_code: impl Into<String>,
        country: impl Into<String>,
    ) -> Result<Self, ValueError> {
        let street = street.into();
        let city = city.into();
        let country = country.into();
        if street.is_empty() {
            return Err(ValueError::Empty("street".to_string()));
        }
        if city.is_empty() {
            return Err(ValueError::Empty("city".to_string()));
        }
        if country.is_empty() {
            return Err(ValueError::Empty("country".to_string()));
        }
        Ok(Self {
            street,
            city,
            state: state.into(),
            postal_code: postal_code.into(),
            country,
        })
    }

    pub fn street(&self) -> &str { &self.street }
    pub fn city(&self) -> &str { &self.city }
    pub fn state(&self) -> &str { &self.state }
    pub fn postal_code(&self) -> &str { &self.postal_code }
    pub fn country(&self) -> &str { &self.country }

    /// Single-line formatted address.
    pub fn one_line(&self) -> String {
        let mut parts = vec![self.street.clone(), self.city.clone()];
        if !self.state.is_empty() {
            parts.push(self.state.clone());
        }
        if !self.postal_code.is_empty() {
            parts.push(self.postal_code.clone());
        }
        parts.push(self.country.clone());
        parts.join(", ")
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.one_line())
    }
}

// ── DateRange ───────────────────────────────────────────────────

/// Immutable date range value object.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DateRange {
    start: NaiveDate,
    end: NaiveDate,
}

impl DateRange {
    /// Create a date range (inclusive start, inclusive end).
    pub fn new(start: NaiveDate, end: NaiveDate) -> Result<Self, ValueError> {
        if start > end {
            return Err(ValueError::InvalidDateRange {
                start: start.to_string(),
                end: end.to_string(),
            });
        }
        Ok(Self { start, end })
    }

    /// Single-day range.
    pub fn single_day(day: NaiveDate) -> Self {
        Self { start: day, end: day }
    }

    pub fn start(&self) -> NaiveDate { self.start }
    pub fn end(&self) -> NaiveDate { self.end }

    /// Number of days in the range (inclusive).
    pub fn days(&self) -> i64 {
        (self.end - self.start).num_days() + 1
    }

    /// Whether a date falls within this range.
    pub fn contains(&self, date: NaiveDate) -> bool {
        date >= self.start && date <= self.end
    }

    /// Whether two ranges overlap.
    pub fn overlaps(&self, other: &DateRange) -> bool {
        self.start <= other.end && other.start <= self.end
    }

    /// Compute the intersection of two ranges, if they overlap.
    pub fn intersection(&self, other: &DateRange) -> Option<DateRange> {
        if !self.overlaps(other) {
            return None;
        }
        let s = self.start.max(other.start);
        let e = self.end.min(other.end);
        Some(DateRange { start: s, end: e })
    }

    /// Merge two overlapping or adjacent ranges.
    pub fn merge(&self, other: &DateRange) -> Option<DateRange> {
        // Check overlap or adjacency (end + 1 day == other start).
        let adjacent = self.end.succ_opt() == Some(other.start)
            || other.end.succ_opt() == Some(self.start);
        if !self.overlaps(other) && !adjacent {
            return None;
        }
        let s = self.start.min(other.start);
        let e = self.end.max(other.end);
        Some(DateRange { start: s, end: e })
    }
}

impl fmt::Display for DateRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} to {}", self.start, self.end)
    }
}

// ── Percentage ──────────────────────────────────────────────────

/// Percentage value object (0.0 to 100.0 by default, but can be unclamped).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Percentage {
    value: f64,
}

impl Percentage {
    /// Create a percentage in the range [0, 100].
    pub fn new(value: f64) -> Result<Self, ValueError> {
        if value < 0.0 || value > 100.0 {
            return Err(ValueError::OutOfRange {
                field: "percentage".to_string(),
                min: "0".to_string(),
                max: "100".to_string(),
                actual: value.to_string(),
            });
        }
        Ok(Self { value })
    }

    /// Create without range check.
    pub fn unclamped(value: f64) -> Self {
        Self { value }
    }

    /// Create from a decimal fraction (0.0 to 1.0 -> 0% to 100%).
    pub fn from_fraction(fraction: f64) -> Result<Self, ValueError> {
        Self::new(fraction * 100.0)
    }

    pub fn value(&self) -> f64 {
        self.value
    }

    /// As fraction (0.0 to 1.0).
    pub fn as_fraction(&self) -> f64 {
        self.value / 100.0
    }

    /// Apply percentage to a value.
    pub fn apply(&self, base: f64) -> f64 {
        base * self.as_fraction()
    }

    /// Complement (100 - self).
    pub fn complement(&self) -> Self {
        Self { value: 100.0 - self.value }
    }
}

impl fmt::Display for Percentage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}%", self.value)
    }
}

// ── Timestamp value object ──────────────────────────────────────

/// An immutable timestamp value object wrapping DateTime<Utc>.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Timestamp {
    value: DateTime<Utc>,
}

impl Timestamp {
    pub fn now() -> Self {
        Self { value: Utc::now() }
    }

    pub fn from_datetime(dt: DateTime<Utc>) -> Self {
        Self { value: dt }
    }

    pub fn value(&self) -> DateTime<Utc> {
        self.value
    }

    /// Whether this timestamp is before another.
    pub fn is_before(&self, other: &Timestamp) -> bool {
        self.value < other.value
    }

    /// Whether this timestamp is after another.
    pub fn is_after(&self, other: &Timestamp) -> bool {
        self.value > other.value
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value.to_rfc3339())
    }
}

// ── NonEmptyString ──────────────────────────────────────────────

/// A string that is guaranteed to be non-empty after trimming.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NonEmptyString {
    value: String,
}

impl NonEmptyString {
    pub fn new(value: impl Into<String>) -> Result<Self, ValueError> {
        let value = value.into().trim().to_string();
        if value.is_empty() {
            return Err(ValueError::Empty("string".to_string()));
        }
        Ok(Self { value })
    }

    pub fn value(&self) -> &str {
        &self.value
    }

    pub fn len(&self) -> usize {
        self.value.len()
    }
}

impl fmt::Display for NonEmptyString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Money tests
    #[test]
    fn test_money_from_major() {
        let m = Money::from_major(10, "USD").unwrap();
        assert_eq!(m.amount_cents(), 1000);
        assert_eq!(m.major(), 10);
        assert_eq!(m.minor(), 0);
    }

    #[test]
    fn test_money_from_minor() {
        let m = Money::from_minor(1050, "USD").unwrap();
        assert_eq!(m.major(), 10);
        assert_eq!(m.minor(), 50);
    }

    #[test]
    fn test_money_add_same_currency() {
        let a = Money::from_major(10, "USD").unwrap();
        let b = Money::from_major(5, "USD").unwrap();
        let sum = a.add(&b).unwrap();
        assert_eq!(sum.amount_cents(), 1500);
    }

    #[test]
    fn test_money_add_different_currency() {
        let a = Money::from_major(10, "USD").unwrap();
        let b = Money::from_major(5, "EUR").unwrap();
        assert!(matches!(a.add(&b), Err(ValueError::CurrencyMismatch { .. })));
    }

    #[test]
    fn test_money_sub() {
        let a = Money::from_major(10, "USD").unwrap();
        let b = Money::from_major(3, "USD").unwrap();
        let diff = a.sub(&b).unwrap();
        assert_eq!(diff.amount_cents(), 700);
    }

    #[test]
    fn test_money_multiply() {
        let m = Money::from_major(10, "USD").unwrap();
        let doubled = m.multiply(2);
        assert_eq!(doubled.amount_cents(), 2000);
    }

    #[test]
    fn test_money_negate() {
        let m = Money::from_major(10, "USD").unwrap();
        let neg = m.negate();
        assert!(neg.is_negative());
        assert_eq!(neg.amount_cents(), -1000);
    }

    #[test]
    fn test_money_display() {
        let m = Money::from_minor(1050, "USD").unwrap();
        assert_eq!(format!("{m}"), "USD 10.50");
    }

    #[test]
    fn test_money_partial_ord() {
        let a = Money::from_major(10, "USD").unwrap();
        let b = Money::from_major(5, "USD").unwrap();
        assert!(a > b);
    }

    #[test]
    fn test_money_partial_ord_different_currency() {
        let a = Money::from_major(10, "USD").unwrap();
        let b = Money::from_major(5, "EUR").unwrap();
        assert_eq!(a.partial_cmp(&b), None);
    }

    #[test]
    fn test_money_empty_currency() {
        assert!(matches!(Money::from_major(10, ""), Err(ValueError::Empty(_))));
    }

    // EmailAddress tests
    #[test]
    fn test_email_valid() {
        let email = EmailAddress::new("User@Example.COM").unwrap();
        assert_eq!(email.value(), "user@example.com");
        assert_eq!(email.local_part(), "user");
        assert_eq!(email.domain(), "example.com");
    }

    #[test]
    fn test_email_missing_at() {
        assert!(matches!(EmailAddress::new("invalid"), Err(ValueError::Invalid { .. })));
    }

    #[test]
    fn test_email_empty_local() {
        assert!(matches!(EmailAddress::new("@example.com"), Err(ValueError::Invalid { .. })));
    }

    #[test]
    fn test_email_invalid_domain() {
        assert!(matches!(EmailAddress::new("user@nodot"), Err(ValueError::Invalid { .. })));
    }

    // PhoneNumber tests
    #[test]
    fn test_phone_valid() {
        let p = PhoneNumber::new("1", "555-123-4567").unwrap();
        assert_eq!(p.country_code(), "1");
        assert_eq!(p.number(), "5551234567");
        assert_eq!(p.full_number(), "+15551234567");
    }

    #[test]
    fn test_phone_too_short() {
        assert!(matches!(PhoneNumber::new("1", "123"), Err(ValueError::Invalid { .. })));
    }

    // Address tests
    #[test]
    fn test_address_valid() {
        let addr = Address::new("123 Main St", "Springfield", "IL", "62701", "US").unwrap();
        assert_eq!(addr.city(), "Springfield");
        assert!(addr.one_line().contains("Springfield"));
    }

    #[test]
    fn test_address_empty_street() {
        assert!(matches!(
            Address::new("", "City", "ST", "12345", "US"),
            Err(ValueError::Empty(_))
        ));
    }

    // DateRange tests
    #[test]
    fn test_date_range_valid() {
        let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(2024, 1, 31).unwrap();
        let range = DateRange::new(start, end).unwrap();
        assert_eq!(range.days(), 31);
    }

    #[test]
    fn test_date_range_invalid() {
        let start = NaiveDate::from_ymd_opt(2024, 2, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        assert!(matches!(DateRange::new(start, end), Err(ValueError::InvalidDateRange { .. })));
    }

    #[test]
    fn test_date_range_contains() {
        let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(2024, 1, 31).unwrap();
        let range = DateRange::new(start, end).unwrap();
        let mid = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
        assert!(range.contains(mid));
        let outside = NaiveDate::from_ymd_opt(2024, 2, 1).unwrap();
        assert!(!range.contains(outside));
    }

    #[test]
    fn test_date_range_overlaps() {
        let r1 = DateRange::new(
            NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 15).unwrap(),
        ).unwrap();
        let r2 = DateRange::new(
            NaiveDate::from_ymd_opt(2024, 1, 10).unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 20).unwrap(),
        ).unwrap();
        assert!(r1.overlaps(&r2));
    }

    #[test]
    fn test_date_range_intersection() {
        let r1 = DateRange::new(
            NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 15).unwrap(),
        ).unwrap();
        let r2 = DateRange::new(
            NaiveDate::from_ymd_opt(2024, 1, 10).unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 20).unwrap(),
        ).unwrap();
        let inter = r1.intersection(&r2).unwrap();
        assert_eq!(inter.start(), NaiveDate::from_ymd_opt(2024, 1, 10).unwrap());
        assert_eq!(inter.end(), NaiveDate::from_ymd_opt(2024, 1, 15).unwrap());
    }

    #[test]
    fn test_date_range_single_day() {
        let d = NaiveDate::from_ymd_opt(2024, 6, 15).unwrap();
        let range = DateRange::single_day(d);
        assert_eq!(range.days(), 1);
        assert!(range.contains(d));
    }

    // Percentage tests
    #[test]
    fn test_percentage_valid() {
        let p = Percentage::new(50.0).unwrap();
        assert!((p.as_fraction() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_percentage_out_of_range() {
        assert!(matches!(Percentage::new(101.0), Err(ValueError::OutOfRange { .. })));
        assert!(matches!(Percentage::new(-1.0), Err(ValueError::OutOfRange { .. })));
    }

    #[test]
    fn test_percentage_apply() {
        let p = Percentage::new(25.0).unwrap();
        assert!((p.apply(200.0) - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_percentage_complement() {
        let p = Percentage::new(30.0).unwrap();
        assert!((p.complement().value() - 70.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_percentage_from_fraction() {
        let p = Percentage::from_fraction(0.75).unwrap();
        assert!((p.value() - 75.0).abs() < f64::EPSILON);
    }

    // NonEmptyString tests
    #[test]
    fn test_non_empty_string_valid() {
        let s = NonEmptyString::new("hello").unwrap();
        assert_eq!(s.value(), "hello");
    }

    #[test]
    fn test_non_empty_string_empty() {
        assert!(matches!(NonEmptyString::new(""), Err(ValueError::Empty(_))));
    }

    #[test]
    fn test_non_empty_string_whitespace_only() {
        assert!(matches!(NonEmptyString::new("   "), Err(ValueError::Empty(_))));
    }

    // Timestamp tests
    #[test]
    fn test_timestamp_ordering() {
        let t1 = Timestamp::now();
        let t2 = Timestamp::now();
        // t1 is before or equal to t2
        assert!(!t1.is_after(&t2));
    }

    // Value equality tests
    #[test]
    fn test_value_equality_by_content() {
        let a1 = Address::new("123 Main", "City", "ST", "12345", "US").unwrap();
        let a2 = Address::new("123 Main", "City", "ST", "12345", "US").unwrap();
        assert_eq!(a1, a2);
    }

    #[test]
    fn test_money_zero() {
        let z = Money::zero("USD").unwrap();
        assert!(z.is_zero());
        assert!(!z.is_negative());
    }
}
