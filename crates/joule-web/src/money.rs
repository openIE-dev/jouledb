//! Money/currency handling — ISO 4217 currency codes, money arithmetic
//! (add/sub/mul/div), rounding modes (half-up, half-even, banker's),
//! currency conversion, money allocation (split evenly handling remainders),
//! formatting, and comparison.
//!
//! Replaces dinero.js / currency.js with a pure-Rust money model that
//! uses integer minor units to avoid floating-point errors.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Money domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoneyError {
    /// Cannot perform arithmetic on different currencies.
    CurrencyMismatch { lhs: String, rhs: String },
    /// Unknown currency code.
    UnknownCurrency(String),
    /// Division by zero.
    DivisionByZero,
    /// Allocation error.
    AllocationError(String),
    /// No exchange rate available.
    NoExchangeRate { from: String, to: String },
    /// Negative allocation count.
    InvalidAllocation(String),
    /// Parse error.
    ParseError(String),
    /// Overflow.
    Overflow(String),
}

impl fmt::Display for MoneyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CurrencyMismatch { lhs, rhs } => {
                write!(f, "currency mismatch: {lhs} vs {rhs}")
            }
            Self::UnknownCurrency(c) => write!(f, "unknown currency: {c}"),
            Self::DivisionByZero => write!(f, "division by zero"),
            Self::AllocationError(msg) => write!(f, "allocation error: {msg}"),
            Self::NoExchangeRate { from, to } => {
                write!(f, "no exchange rate from {from} to {to}")
            }
            Self::InvalidAllocation(msg) => write!(f, "invalid allocation: {msg}"),
            Self::ParseError(msg) => write!(f, "parse error: {msg}"),
            Self::Overflow(msg) => write!(f, "overflow: {msg}"),
        }
    }
}

impl std::error::Error for MoneyError {}

// ── Rounding Mode ───────────────────────────────────────────────

/// Rounding modes for money operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RoundingMode {
    /// Round towards positive infinity.
    Up,
    /// Round towards negative infinity.
    Down,
    /// Round half up (standard rounding).
    HalfUp,
    /// Round half down.
    HalfDown,
    /// Round half to even (banker's rounding).
    HalfEven,
    /// Round towards zero (truncate).
    Truncate,
}

impl RoundingMode {
    /// Round a value given a divisor using this mode.
    /// `numerator / divisor` rounded according to the mode.
    pub fn divide_round(self, numerator: i64, divisor: i64) -> i64 {
        if divisor == 0 {
            return 0;
        }
        let quotient = numerator / divisor;
        let remainder = numerator % divisor;
        if remainder == 0 {
            return quotient;
        }
        let abs_remainder = remainder.unsigned_abs();
        let abs_divisor = divisor.unsigned_abs();
        let double_rem = abs_remainder * 2;
        let is_negative = (numerator < 0) != (divisor < 0);

        match self {
            RoundingMode::Up => {
                if is_negative {
                    quotient
                } else {
                    quotient + 1
                }
            }
            RoundingMode::Down => {
                if is_negative {
                    quotient - 1
                } else {
                    quotient
                }
            }
            RoundingMode::HalfUp => {
                if double_rem >= abs_divisor {
                    if is_negative { quotient - 1 } else { quotient + 1 }
                } else {
                    quotient
                }
            }
            RoundingMode::HalfDown => {
                if double_rem > abs_divisor {
                    if is_negative { quotient - 1 } else { quotient + 1 }
                } else {
                    quotient
                }
            }
            RoundingMode::HalfEven => {
                if double_rem > abs_divisor {
                    if is_negative { quotient - 1 } else { quotient + 1 }
                } else if double_rem == abs_divisor {
                    // Round to even.
                    if quotient % 2 != 0 {
                        if is_negative { quotient - 1 } else { quotient + 1 }
                    } else {
                        quotient
                    }
                } else {
                    quotient
                }
            }
            RoundingMode::Truncate => quotient,
        }
    }
}

// ── Currency ────────────────────────────────────────────────────

/// Currency definition with ISO 4217 info.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Currency {
    pub code: String,
    pub name: String,
    pub symbol: String,
    pub minor_units: u8,
    pub numeric_code: u16,
}

impl Currency {
    pub fn new(
        code: impl Into<String>,
        name: impl Into<String>,
        symbol: impl Into<String>,
        minor_units: u8,
        numeric_code: u16,
    ) -> Self {
        Self {
            code: code.into(),
            name: name.into(),
            symbol: symbol.into(),
            minor_units,
            numeric_code,
        }
    }

    /// The multiplier to convert major units to minor units.
    pub fn multiplier(&self) -> i64 {
        10_i64.pow(self.minor_units as u32)
    }
}

/// Common currencies.
pub fn usd() -> Currency {
    Currency::new("USD", "US Dollar", "$", 2, 840)
}

pub fn eur() -> Currency {
    Currency::new("EUR", "Euro", "\u{20ac}", 2, 978)
}

pub fn gbp() -> Currency {
    Currency::new("GBP", "British Pound", "\u{00a3}", 2, 826)
}

pub fn jpy() -> Currency {
    Currency::new("JPY", "Japanese Yen", "\u{00a5}", 0, 392)
}

pub fn chf() -> Currency {
    Currency::new("CHF", "Swiss Franc", "CHF", 2, 756)
}

pub fn cad() -> Currency {
    Currency::new("CAD", "Canadian Dollar", "CA$", 2, 124)
}

pub fn aud() -> Currency {
    Currency::new("AUD", "Australian Dollar", "A$", 2, 036)
}

pub fn inr() -> Currency {
    Currency::new("INR", "Indian Rupee", "\u{20b9}", 2, 356)
}

pub fn brl() -> Currency {
    Currency::new("BRL", "Brazilian Real", "R$", 2, 986)
}

pub fn krw() -> Currency {
    Currency::new("KRW", "South Korean Won", "\u{20a9}", 0, 410)
}

// ── Money ───────────────────────────────────────────────────────

/// Money value with integer minor units to avoid floating-point errors.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Money {
    amount: i64,
    currency_code: String,
    minor_units: u8,
}

impl Money {
    /// Create money from minor units (e.g., cents for USD).
    pub fn from_minor(amount: i64, currency: &Currency) -> Self {
        Self {
            amount,
            currency_code: currency.code.clone(),
            minor_units: currency.minor_units,
        }
    }

    /// Create money from major units (e.g., dollars for USD).
    pub fn from_major(major: i64, currency: &Currency) -> Self {
        Self {
            amount: major * currency.multiplier(),
            currency_code: currency.code.clone(),
            minor_units: currency.minor_units,
        }
    }

    /// Create zero money.
    pub fn zero(currency: &Currency) -> Self {
        Self::from_minor(0, currency)
    }

    /// Create from a fractional amount (e.g. 10.50).
    pub fn from_decimal(whole: i64, frac: u64, currency: &Currency) -> Self {
        let mult = currency.multiplier();
        let frac_value = if currency.minor_units == 0 {
            0
        } else {
            let max_frac = 10_u64.pow(currency.minor_units as u32);
            (frac % max_frac) as i64
        };
        let sign = if whole < 0 { -1 } else { 1 };
        let abs_amount = whole.abs() * mult + frac_value;
        Self {
            amount: sign * abs_amount,
            currency_code: currency.code.clone(),
            minor_units: currency.minor_units,
        }
    }

    /// Amount in minor units.
    pub fn amount(&self) -> i64 {
        self.amount
    }

    /// Currency code.
    pub fn currency_code(&self) -> &str {
        &self.currency_code
    }

    /// Minor units (decimal places).
    pub fn minor_units(&self) -> u8 {
        self.minor_units
    }

    /// Major unit part.
    pub fn major_part(&self) -> i64 {
        let mult = 10_i64.pow(self.minor_units as u32);
        self.amount / mult
    }

    /// Minor unit remainder.
    pub fn minor_part(&self) -> i64 {
        let mult = 10_i64.pow(self.minor_units as u32);
        self.amount.abs() % mult
    }

    /// As f64 value (lossy).
    pub fn as_f64(&self) -> f64 {
        let mult = 10_f64.powi(self.minor_units as i32);
        self.amount as f64 / mult
    }

    fn ensure_same_currency(&self, other: &Money) -> Result<(), MoneyError> {
        if self.currency_code != other.currency_code {
            Err(MoneyError::CurrencyMismatch {
                lhs: self.currency_code.clone(),
                rhs: other.currency_code.clone(),
            })
        } else {
            Ok(())
        }
    }

    /// Add two money values.
    pub fn add(&self, other: &Money) -> Result<Money, MoneyError> {
        self.ensure_same_currency(other)?;
        Ok(Money {
            amount: self.amount + other.amount,
            currency_code: self.currency_code.clone(),
            minor_units: self.minor_units,
        })
    }

    /// Subtract.
    pub fn sub(&self, other: &Money) -> Result<Money, MoneyError> {
        self.ensure_same_currency(other)?;
        Ok(Money {
            amount: self.amount - other.amount,
            currency_code: self.currency_code.clone(),
            minor_units: self.minor_units,
        })
    }

    /// Multiply by an integer scalar.
    pub fn multiply(&self, factor: i64) -> Money {
        Money {
            amount: self.amount * factor,
            currency_code: self.currency_code.clone(),
            minor_units: self.minor_units,
        }
    }

    /// Multiply by a fractional factor with rounding.
    pub fn multiply_rounded(&self, numerator: i64, denominator: i64, mode: RoundingMode) -> Result<Money, MoneyError> {
        if denominator == 0 {
            return Err(MoneyError::DivisionByZero);
        }
        let product = self.amount * numerator;
        let result = mode.divide_round(product, denominator);
        Ok(Money {
            amount: result,
            currency_code: self.currency_code.clone(),
            minor_units: self.minor_units,
        })
    }

    /// Divide by an integer with rounding.
    pub fn divide(&self, divisor: i64, mode: RoundingMode) -> Result<Money, MoneyError> {
        if divisor == 0 {
            return Err(MoneyError::DivisionByZero);
        }
        let result = mode.divide_round(self.amount, divisor);
        Ok(Money {
            amount: result,
            currency_code: self.currency_code.clone(),
            minor_units: self.minor_units,
        })
    }

    /// Negate.
    pub fn negate(&self) -> Money {
        Money {
            amount: -self.amount,
            currency_code: self.currency_code.clone(),
            minor_units: self.minor_units,
        }
    }

    /// Absolute value.
    pub fn abs(&self) -> Money {
        Money {
            amount: self.amount.abs(),
            currency_code: self.currency_code.clone(),
            minor_units: self.minor_units,
        }
    }

    /// Whether negative.
    pub fn is_negative(&self) -> bool {
        self.amount < 0
    }

    /// Whether zero.
    pub fn is_zero(&self) -> bool {
        self.amount == 0
    }

    /// Whether positive.
    pub fn is_positive(&self) -> bool {
        self.amount > 0
    }

    /// Min of two money values.
    pub fn min(&self, other: &Money) -> Result<Money, MoneyError> {
        self.ensure_same_currency(other)?;
        if self.amount <= other.amount {
            Ok(self.clone())
        } else {
            Ok(other.clone())
        }
    }

    /// Max of two money values.
    pub fn max(&self, other: &Money) -> Result<Money, MoneyError> {
        self.ensure_same_currency(other)?;
        if self.amount >= other.amount {
            Ok(self.clone())
        } else {
            Ok(other.clone())
        }
    }

    /// Allocate evenly among `n` parts, distributing remainder cent-by-cent.
    pub fn allocate(&self, n: usize) -> Result<Vec<Money>, MoneyError> {
        if n == 0 {
            return Err(MoneyError::InvalidAllocation("cannot allocate to 0 parts".to_string()));
        }
        let base = self.amount / n as i64;
        let remainder = (self.amount % n as i64).unsigned_abs() as usize;
        let sign = if self.amount < 0 { -1 } else { 1 };
        let mut parts = Vec::with_capacity(n);
        for i in 0..n {
            let extra = if i < remainder { sign } else { 0 };
            parts.push(Money {
                amount: base + extra,
                currency_code: self.currency_code.clone(),
                minor_units: self.minor_units,
            });
        }
        Ok(parts)
    }

    /// Allocate by ratios (e.g., [50, 30, 20] for 50%/30%/20%).
    pub fn allocate_by_ratios(&self, ratios: &[u64]) -> Result<Vec<Money>, MoneyError> {
        if ratios.is_empty() {
            return Err(MoneyError::InvalidAllocation("empty ratios".to_string()));
        }
        let total_ratio: u64 = ratios.iter().sum();
        if total_ratio == 0 {
            return Err(MoneyError::InvalidAllocation("total ratio is zero".to_string()));
        }
        let mut parts = Vec::with_capacity(ratios.len());
        let mut allocated = 0i64;
        for (i, ratio) in ratios.iter().enumerate() {
            if i == ratios.len() - 1 {
                // Last part gets the remainder.
                parts.push(Money {
                    amount: self.amount - allocated,
                    currency_code: self.currency_code.clone(),
                    minor_units: self.minor_units,
                });
            } else {
                let part = RoundingMode::Truncate.divide_round(
                    self.amount * *ratio as i64,
                    total_ratio as i64,
                );
                allocated += part;
                parts.push(Money {
                    amount: part,
                    currency_code: self.currency_code.clone(),
                    minor_units: self.minor_units,
                });
            }
        }
        Ok(parts)
    }

    /// Format the money value.
    pub fn format_display(&self) -> String {
        let sign = if self.amount < 0 { "-" } else { "" };
        let mult = 10_i64.pow(self.minor_units as u32);
        let major = (self.amount.abs()) / mult;
        let minor = (self.amount.abs()) % mult;
        if self.minor_units == 0 {
            format!("{sign}{} {major}", self.currency_code)
        } else {
            let width = self.minor_units as usize;
            format!("{sign}{} {major}.{minor:0>width$}", self.currency_code)
        }
    }
}

impl fmt::Display for Money {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format_display())
    }
}

impl PartialOrd for Money {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.currency_code != other.currency_code {
            None
        } else {
            Some(self.amount.cmp(&other.amount))
        }
    }
}

// ── ExchangeRate ────────────────────────────────────────────────

/// An exchange rate between two currencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeRate {
    pub from: String,
    pub to: String,
    /// Rate as a fraction: numerator / denominator.
    pub rate_numerator: i64,
    pub rate_denominator: i64,
}

impl ExchangeRate {
    /// Create a rate. E.g., 1 USD = 0.85 EUR => numerator=85, denominator=100.
    pub fn new(
        from: impl Into<String>,
        to: impl Into<String>,
        rate_numerator: i64,
        rate_denominator: i64,
    ) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            rate_numerator,
            rate_denominator,
        }
    }

    /// Convert money using this rate.
    pub fn convert(&self, money: &Money, target_currency: &Currency, mode: RoundingMode) -> Result<Money, MoneyError> {
        if money.currency_code != self.from {
            return Err(MoneyError::CurrencyMismatch {
                lhs: money.currency_code.clone(),
                rhs: self.from.clone(),
            });
        }
        if self.rate_denominator == 0 {
            return Err(MoneyError::DivisionByZero);
        }
        // Handle different minor units between source and target.
        let source_mult = 10_i64.pow(money.minor_units as u32);
        let target_mult = 10_i64.pow(target_currency.minor_units as u32);

        // Convert: amount_in_target = amount_in_source * rate * (target_mult / source_mult)
        let numerator = money.amount * self.rate_numerator * target_mult;
        let denominator = self.rate_denominator * source_mult;

        let converted = mode.divide_round(numerator, denominator);
        Ok(Money {
            amount: converted,
            currency_code: target_currency.code.clone(),
            minor_units: target_currency.minor_units,
        })
    }
}

// ── CurrencyConverter ───────────────────────────────────────────

/// Currency converter with registered exchange rates.
#[derive(Debug, Default)]
pub struct CurrencyConverter {
    rates: HashMap<(String, String), ExchangeRate>,
}

impl CurrencyConverter {
    pub fn new() -> Self {
        Self { rates: HashMap::new() }
    }

    /// Register an exchange rate.
    pub fn register_rate(&mut self, rate: ExchangeRate) {
        self.rates.insert((rate.from.clone(), rate.to.clone()), rate);
    }

    /// Convert money to a target currency.
    pub fn convert(
        &self,
        money: &Money,
        target: &Currency,
        mode: RoundingMode,
    ) -> Result<Money, MoneyError> {
        let key = (money.currency_code.clone(), target.code.clone());
        let rate = self.rates.get(&key).ok_or_else(|| MoneyError::NoExchangeRate {
            from: money.currency_code.clone(),
            to: target.code.clone(),
        })?;
        rate.convert(money, target, mode)
    }

    /// Number of registered rates.
    pub fn rate_count(&self) -> usize {
        self.rates.len()
    }
}

// ── Sum helper ──────────────────────────────────────────────────

/// Sum a slice of money values (all must be the same currency).
pub fn sum(values: &[Money]) -> Result<Money, MoneyError> {
    if values.is_empty() {
        return Err(MoneyError::AllocationError("cannot sum empty list".to_string()));
    }
    let mut total = values[0].clone();
    for v in &values[1..] {
        total = total.add(v)?;
    }
    Ok(total)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_money_from_major() {
        let m = Money::from_major(10, &usd());
        assert_eq!(m.amount(), 1000);
        assert_eq!(m.major_part(), 10);
        assert_eq!(m.minor_part(), 0);
    }

    #[test]
    fn test_money_from_minor() {
        let m = Money::from_minor(1050, &usd());
        assert_eq!(m.major_part(), 10);
        assert_eq!(m.minor_part(), 50);
    }

    #[test]
    fn test_money_from_decimal() {
        let m = Money::from_decimal(10, 50, &usd());
        assert_eq!(m.amount(), 1050);
        assert_eq!(m.major_part(), 10);
        assert_eq!(m.minor_part(), 50);
    }

    #[test]
    fn test_money_add() {
        let a = Money::from_major(10, &usd());
        let b = Money::from_major(5, &usd());
        let result = a.add(&b).unwrap();
        assert_eq!(result.amount(), 1500);
    }

    #[test]
    fn test_money_add_currency_mismatch() {
        let a = Money::from_major(10, &usd());
        let b = Money::from_major(5, &eur());
        assert!(matches!(a.add(&b), Err(MoneyError::CurrencyMismatch { .. })));
    }

    #[test]
    fn test_money_sub() {
        let a = Money::from_major(10, &usd());
        let b = Money::from_major(3, &usd());
        let result = a.sub(&b).unwrap();
        assert_eq!(result.amount(), 700);
    }

    #[test]
    fn test_money_multiply() {
        let m = Money::from_major(10, &usd());
        let result = m.multiply(3);
        assert_eq!(result.amount(), 3000);
    }

    #[test]
    fn test_money_divide_half_up() {
        let m = Money::from_minor(100, &usd());
        let result = m.divide(3, RoundingMode::HalfUp).unwrap();
        assert_eq!(result.amount(), 33); // 100/3 = 33.33 -> 33
    }

    #[test]
    fn test_money_divide_by_zero() {
        let m = Money::from_minor(100, &usd());
        assert!(matches!(m.divide(0, RoundingMode::HalfUp), Err(MoneyError::DivisionByZero)));
    }

    #[test]
    fn test_money_negate() {
        let m = Money::from_major(10, &usd());
        let neg = m.negate();
        assert!(neg.is_negative());
        assert_eq!(neg.amount(), -1000);
    }

    #[test]
    fn test_money_abs() {
        let m = Money::from_minor(-500, &usd());
        let a = m.abs();
        assert_eq!(a.amount(), 500);
    }

    #[test]
    fn test_money_is_zero() {
        let m = Money::zero(&usd());
        assert!(m.is_zero());
        assert!(!m.is_positive());
        assert!(!m.is_negative());
    }

    #[test]
    fn test_money_display() {
        let m = Money::from_minor(1050, &usd());
        assert_eq!(format!("{m}"), "USD 10.50");
    }

    #[test]
    fn test_money_display_jpy() {
        let m = Money::from_major(1000, &jpy());
        assert_eq!(format!("{m}"), "JPY 1000");
    }

    #[test]
    fn test_money_display_negative() {
        let m = Money::from_minor(-500, &usd());
        assert_eq!(format!("{m}"), "-USD 5.00");
    }

    #[test]
    fn test_money_partial_ord() {
        let a = Money::from_major(10, &usd());
        let b = Money::from_major(5, &usd());
        assert!(a > b);
    }

    #[test]
    fn test_money_partial_ord_different_currency() {
        let a = Money::from_major(10, &usd());
        let b = Money::from_major(5, &eur());
        assert_eq!(a.partial_cmp(&b), None);
    }

    #[test]
    fn test_allocate_evenly() {
        let m = Money::from_minor(100, &usd());
        let parts = m.allocate(3).unwrap();
        assert_eq!(parts.len(), 3);
        let total: i64 = parts.iter().map(|p| p.amount()).sum();
        assert_eq!(total, 100);
        // First part gets the extra cent.
        assert_eq!(parts[0].amount(), 34);
        assert_eq!(parts[1].amount(), 33);
        assert_eq!(parts[2].amount(), 33);
    }

    #[test]
    fn test_allocate_zero_parts() {
        let m = Money::from_major(10, &usd());
        assert!(matches!(m.allocate(0), Err(MoneyError::InvalidAllocation(_))));
    }

    #[test]
    fn test_allocate_by_ratios() {
        let m = Money::from_minor(100, &usd());
        let parts = m.allocate_by_ratios(&[50, 30, 20]).unwrap();
        assert_eq!(parts.len(), 3);
        let total: i64 = parts.iter().map(|p| p.amount()).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_allocate_by_ratios_empty() {
        let m = Money::from_major(10, &usd());
        assert!(matches!(m.allocate_by_ratios(&[]), Err(MoneyError::InvalidAllocation(_))));
    }

    #[test]
    fn test_rounding_half_up() {
        assert_eq!(RoundingMode::HalfUp.divide_round(5, 2), 3);
        assert_eq!(RoundingMode::HalfUp.divide_round(7, 2), 4);
    }

    #[test]
    fn test_rounding_half_even() {
        // 5/2 = 2.5 -> rounds to 2 (even)
        assert_eq!(RoundingMode::HalfEven.divide_round(5, 2), 2);
        // 7/2 = 3.5 -> rounds to 4 (even)
        assert_eq!(RoundingMode::HalfEven.divide_round(7, 2), 4);
    }

    #[test]
    fn test_rounding_truncate() {
        assert_eq!(RoundingMode::Truncate.divide_round(7, 2), 3);
        assert_eq!(RoundingMode::Truncate.divide_round(-7, 2), -3);
    }

    #[test]
    fn test_rounding_up_down() {
        assert_eq!(RoundingMode::Up.divide_round(7, 2), 4);
        assert_eq!(RoundingMode::Down.divide_round(7, 2), 3);
    }

    #[test]
    fn test_exchange_rate_conversion() {
        // 1 USD = 0.85 EUR (85/100)
        let rate = ExchangeRate::new("USD", "EUR", 85, 100);
        let usd_money = Money::from_major(100, &usd());
        let eur_money = rate.convert(&usd_money, &eur(), RoundingMode::HalfUp).unwrap();
        assert_eq!(eur_money.amount(), 8500); // 100.00 USD -> 85.00 EUR
        assert_eq!(eur_money.currency_code(), "EUR");
    }

    #[test]
    fn test_currency_converter() {
        let mut converter = CurrencyConverter::new();
        converter.register_rate(ExchangeRate::new("USD", "EUR", 85, 100));
        let money = Money::from_major(100, &usd());
        let converted = converter.convert(&money, &eur(), RoundingMode::HalfUp).unwrap();
        assert_eq!(converted.amount(), 8500);
    }

    #[test]
    fn test_currency_converter_no_rate() {
        let converter = CurrencyConverter::new();
        let money = Money::from_major(100, &usd());
        let result = converter.convert(&money, &eur(), RoundingMode::HalfUp);
        assert!(matches!(result, Err(MoneyError::NoExchangeRate { .. })));
    }

    #[test]
    fn test_sum_money() {
        let values = vec![
            Money::from_major(10, &usd()),
            Money::from_major(20, &usd()),
            Money::from_major(30, &usd()),
        ];
        let total = sum(&values).unwrap();
        assert_eq!(total.amount(), 6000);
    }

    #[test]
    fn test_money_min_max() {
        let a = Money::from_major(10, &usd());
        let b = Money::from_major(20, &usd());
        assert_eq!(a.min(&b).unwrap().amount(), 1000);
        assert_eq!(a.max(&b).unwrap().amount(), 2000);
    }

    #[test]
    fn test_multiply_rounded() {
        // $10.00 * 1/3 = $3.33
        let m = Money::from_major(10, &usd());
        let result = m.multiply_rounded(1, 3, RoundingMode::HalfUp).unwrap();
        assert_eq!(result.amount(), 333);
    }

    #[test]
    fn test_as_f64() {
        let m = Money::from_minor(1050, &usd());
        assert!((m.as_f64() - 10.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_negative_money_display() {
        let m = Money::from_decimal(-10, 50, &usd());
        assert_eq!(m.major_part(), -10);
        assert_eq!(m.minor_part(), 50);
    }

    #[test]
    fn test_jpy_no_minor_units() {
        let m = Money::from_major(1000, &jpy());
        assert_eq!(m.amount(), 1000);
        assert_eq!(m.major_part(), 1000);
        assert_eq!(m.minor_part(), 0);
    }

    #[test]
    fn test_currency_multiplier() {
        assert_eq!(usd().multiplier(), 100);
        assert_eq!(jpy().multiplier(), 1);
    }
}
