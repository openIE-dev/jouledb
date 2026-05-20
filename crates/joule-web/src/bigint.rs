//! Arbitrary precision integers — pure-Rust replacement for bignumber.js, BigInt polyfill.
//!
//! Sign + Vec<u32> limbs representation. Add, subtract, multiply (Karatsuba for large),
//! divide, modular exponentiation, GCD, comparison, string conversion (base 10/16), factorial.

use std::cmp::Ordering;
use std::fmt;

/// The base for each limb: 2^32.
const BASE: u64 = 1u64 << 32;

// ── BigInt ────────────────────────────────────────────────────

/// An arbitrary-precision integer stored as sign + little-endian u32 limbs.
/// limbs[0] is the least significant limb.
#[derive(Debug, Clone)]
pub struct BigInt {
    /// true if negative, false if zero or positive.
    pub negative: bool,
    /// Limbs in little-endian order (limbs[0] = least significant).
    pub limbs: Vec<u32>,
}

impl BigInt {
    /// Zero.
    pub fn zero() -> Self {
        Self { negative: false, limbs: Vec::new() }
    }

    /// One.
    pub fn one() -> Self {
        Self { negative: false, limbs: vec![1] }
    }

    /// Create from an i64.
    pub fn from_i64(val: i64) -> Self {
        if val == 0 {
            return Self::zero();
        }
        let negative = val < 0;
        let abs_val = (val as i128).unsigned_abs() as u64;
        let mut limbs = Vec::new();
        let lo = abs_val as u32;
        let hi = (abs_val >> 32) as u32;
        limbs.push(lo);
        if hi != 0 {
            limbs.push(hi);
        }
        Self { negative, limbs }
    }

    /// Create from a u64.
    pub fn from_u64(val: u64) -> Self {
        if val == 0 {
            return Self::zero();
        }
        let mut limbs = Vec::new();
        let lo = val as u32;
        let hi = (val >> 32) as u32;
        limbs.push(lo);
        if hi != 0 {
            limbs.push(hi);
        }
        Self { negative: false, limbs }
    }

    /// Remove leading zero limbs.
    fn normalize(&mut self) {
        while self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
        if self.limbs.is_empty() {
            self.negative = false;
        }
    }

    /// Whether this is zero.
    pub fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    /// Whether this is positive (> 0).
    pub fn is_positive(&self) -> bool {
        !self.negative && !self.is_zero()
    }

    /// Whether this is negative (< 0).
    pub fn is_negative(&self) -> bool {
        self.negative
    }

    /// Absolute value.
    pub fn abs(&self) -> Self {
        Self { negative: false, limbs: self.limbs.clone() }
    }

    /// Negate.
    pub fn negate(&self) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        Self { negative: !self.negative, limbs: self.limbs.clone() }
    }

    /// Compare magnitudes (ignoring sign).
    fn cmp_magnitude(&self, other: &Self) -> Ordering {
        if self.limbs.len() != other.limbs.len() {
            return self.limbs.len().cmp(&other.limbs.len());
        }
        for i in (0..self.limbs.len()).rev() {
            if self.limbs[i] != other.limbs[i] {
                return self.limbs[i].cmp(&other.limbs[i]);
            }
        }
        Ordering::Equal
    }

    /// Add magnitudes (both treated as positive).
    fn add_magnitudes(a: &[u32], b: &[u32]) -> Vec<u32> {
        let max_len = a.len().max(b.len());
        let mut result = Vec::with_capacity(max_len + 1);
        let mut carry: u64 = 0;
        for i in 0..max_len {
            let av = if i < a.len() { a[i] as u64 } else { 0 };
            let bv = if i < b.len() { b[i] as u64 } else { 0 };
            let sum = av + bv + carry;
            result.push(sum as u32);
            carry = sum >> 32;
        }
        if carry != 0 {
            result.push(carry as u32);
        }
        result
    }

    /// Subtract magnitudes (a >= b assumed). Returns a - b.
    fn sub_magnitudes(a: &[u32], b: &[u32]) -> Vec<u32> {
        let mut result = Vec::with_capacity(a.len());
        let mut borrow: i64 = 0;
        for i in 0..a.len() {
            let av = a[i] as i64;
            let bv = if i < b.len() { b[i] as i64 } else { 0 };
            let mut diff = av - bv - borrow;
            if diff < 0 {
                diff += BASE as i64;
                borrow = 1;
            } else {
                borrow = 0;
            }
            result.push(diff as u32);
        }
        // Remove leading zeros
        while result.last() == Some(&0) {
            result.pop();
        }
        result
    }

    /// Addition.
    pub fn add(&self, other: &Self) -> Self {
        if self.is_zero() { return other.clone(); }
        if other.is_zero() { return self.clone(); }

        if self.negative == other.negative {
            // Same sign: add magnitudes
            let limbs = Self::add_magnitudes(&self.limbs, &other.limbs);
            Self { negative: self.negative, limbs }
        } else {
            // Different signs: subtract magnitudes
            match self.cmp_magnitude(other) {
                Ordering::Greater | Ordering::Equal => {
                    let limbs = Self::sub_magnitudes(&self.limbs, &other.limbs);
                    let mut result = Self { negative: self.negative, limbs };
                    result.normalize();
                    result
                }
                Ordering::Less => {
                    let limbs = Self::sub_magnitudes(&other.limbs, &self.limbs);
                    let mut result = Self { negative: other.negative, limbs };
                    result.normalize();
                    result
                }
            }
        }
    }

    /// Subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.negate())
    }

    /// Schoolbook multiplication.
    fn mul_schoolbook(a: &[u32], b: &[u32]) -> Vec<u32> {
        if a.is_empty() || b.is_empty() {
            return Vec::new();
        }
        let mut result = vec![0u32; a.len() + b.len()];
        for i in 0..a.len() {
            let mut carry: u64 = 0;
            for j in 0..b.len() {
                let prod = a[i] as u64 * b[j] as u64 + result[i + j] as u64 + carry;
                result[i + j] = prod as u32;
                carry = prod >> 32;
            }
            if carry != 0 {
                result[i + b.len()] += carry as u32;
            }
        }
        // Remove leading zeros
        while result.last() == Some(&0) {
            result.pop();
        }
        result
    }

    /// Karatsuba multiplication for large numbers.
    fn mul_karatsuba(a: &[u32], b: &[u32]) -> Vec<u32> {
        let n = a.len().max(b.len());
        if n < 32 {
            return Self::mul_schoolbook(a, b);
        }

        let mid = n / 2;

        let (a_low, a_high) = if a.len() <= mid {
            (a, &[] as &[u32])
        } else {
            (&a[..mid], &a[mid..])
        };
        let (b_low, b_high) = if b.len() <= mid {
            (b, &[] as &[u32])
        } else {
            (&b[..mid], &b[mid..])
        };

        let z0 = Self::mul_karatsuba(a_low, b_low);
        let z2 = Self::mul_karatsuba(a_high, b_high);

        let a_sum = Self::add_magnitudes(a_low, a_high);
        let b_sum = Self::add_magnitudes(b_low, b_high);
        let z1_full = Self::mul_karatsuba(&a_sum, &b_sum);

        // z1 = z1_full - z2 - z0
        let z1_temp = Self::sub_magnitudes_safe(&z1_full, &z2);
        let z1 = Self::sub_magnitudes_safe(&z1_temp, &z0);

        // result = z0 + z1 * BASE^mid + z2 * BASE^(2*mid)
        let mut result = z0;
        // Add z1 shifted by mid
        Self::add_shifted(&mut result, &z1, mid);
        // Add z2 shifted by 2*mid
        Self::add_shifted(&mut result, &z2, 2 * mid);

        while result.last() == Some(&0) {
            result.pop();
        }
        result
    }

    /// Safe magnitude subtraction (handles a < b by returning empty).
    fn sub_magnitudes_safe(a: &[u32], b: &[u32]) -> Vec<u32> {
        // Check if a >= b
        let a_len = {
            let mut l = a.len();
            while l > 0 && a[l - 1] == 0 { l -= 1; }
            l
        };
        let b_len = {
            let mut l = b.len();
            while l > 0 && b[l - 1] == 0 { l -= 1; }
            l
        };
        if a_len < b_len {
            return Vec::new();
        }
        Self::sub_magnitudes(a, b)
    }

    /// Add `other` shifted left by `shift` limbs to `result`.
    fn add_shifted(result: &mut Vec<u32>, other: &[u32], shift: usize) {
        while result.len() < shift + other.len() + 1 {
            result.push(0);
        }
        let mut carry: u64 = 0;
        for i in 0..other.len() {
            let sum = result[shift + i] as u64 + other[i] as u64 + carry;
            result[shift + i] = sum as u32;
            carry = sum >> 32;
        }
        let mut idx = shift + other.len();
        while carry != 0 && idx < result.len() {
            let sum = result[idx] as u64 + carry;
            result[idx] = sum as u32;
            carry = sum >> 32;
            idx += 1;
        }
        if carry != 0 {
            result.push(carry as u32);
        }
    }

    /// Multiplication.
    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        let limbs = Self::mul_karatsuba(&self.limbs, &other.limbs);
        let negative = self.negative != other.negative;
        let mut result = Self { negative, limbs };
        result.normalize();
        result
    }

    /// Division with remainder. Returns (quotient, remainder).
    /// Panics if divisor is zero.
    pub fn div_rem(&self, other: &Self) -> (Self, Self) {
        assert!(!other.is_zero(), "Division by zero");

        if self.is_zero() {
            return (Self::zero(), Self::zero());
        }

        match self.abs().cmp_magnitude(&other.abs()) {
            Ordering::Less => {
                return (Self::zero(), self.clone());
            }
            Ordering::Equal => {
                let neg = self.negative != other.negative;
                return (Self { negative: neg, limbs: vec![1] }, Self::zero());
            }
            _ => {}
        }

        // Long division
        let (q_limbs, r_limbs) = Self::long_division(&self.limbs, &other.limbs);
        let q_neg = self.negative != other.negative;
        let r_neg = self.negative;

        let mut quotient = Self { negative: q_neg, limbs: q_limbs };
        let mut remainder = Self { negative: r_neg, limbs: r_limbs };
        quotient.normalize();
        remainder.normalize();
        (quotient, remainder)
    }

    /// Long division of magnitudes. Returns (quotient_limbs, remainder_limbs).
    fn long_division(numerator: &[u32], denominator: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let n = numerator.len();
        let m = denominator.len();

        if m == 1 {
            // Simple single-limb division
            let d = denominator[0] as u64;
            let mut quotient = vec![0u32; n];
            let mut rem: u64 = 0;
            for i in (0..n).rev() {
                let cur = rem * BASE + numerator[i] as u64;
                quotient[i] = (cur / d) as u32;
                rem = cur % d;
            }
            while quotient.last() == Some(&0) {
                quotient.pop();
            }
            let r = if rem == 0 { Vec::new() } else { vec![rem as u32] };
            return (quotient, r);
        }

        // Multi-limb: use shifting approach
        let mut remainder = BigInt { negative: false, limbs: Vec::new() };
        let divisor = BigInt { negative: false, limbs: denominator.to_vec() };

        let total_bits = n * 32;
        let mut quotient_bits = vec![false; total_bits];

        for i in (0..total_bits).rev() {
            // Shift remainder left by 1 bit
            let mut carry = 0u32;
            for limb in &mut remainder.limbs {
                let new_carry = *limb >> 31;
                *limb = (*limb << 1) | carry;
                carry = new_carry;
            }
            if carry != 0 {
                remainder.limbs.push(carry);
            }

            // Bring down next bit from numerator
            let limb_idx = i / 32;
            let bit_idx = i % 32;
            let bit = (numerator[limb_idx] >> bit_idx) & 1;
            if remainder.limbs.is_empty() {
                if bit != 0 {
                    remainder.limbs.push(bit);
                }
            } else {
                remainder.limbs[0] |= bit;
            }
            remainder.normalize();

            // If remainder >= divisor, subtract and set quotient bit
            if remainder.cmp_magnitude(&divisor) != Ordering::Less {
                remainder.limbs = Self::sub_magnitudes(&remainder.limbs, &divisor.limbs);
                remainder.normalize();
                quotient_bits[i] = true;
            }
        }

        // Convert quotient bits to limbs
        let q_limb_count = (total_bits + 31) / 32;
        let mut q_limbs = vec![0u32; q_limb_count];
        for (i, &bit) in quotient_bits.iter().enumerate() {
            if bit {
                q_limbs[i / 32] |= 1u32 << (i % 32);
            }
        }
        while q_limbs.last() == Some(&0) {
            q_limbs.pop();
        }

        (q_limbs, remainder.limbs)
    }

    /// Division (quotient only).
    pub fn div(&self, other: &Self) -> Self {
        self.div_rem(other).0
    }

    /// Remainder.
    pub fn rem(&self, other: &Self) -> Self {
        self.div_rem(other).1
    }

    /// Modular exponentiation: self^exp mod modulus.
    pub fn mod_pow(&self, exp: &Self, modulus: &Self) -> Self {
        assert!(!modulus.is_zero(), "Modulus cannot be zero");
        if exp.is_zero() {
            return Self::one().div_rem(modulus).1;
        }

        let mut base = self.div_rem(modulus).1;
        let mut result = Self::one();
        let mut e = exp.abs();

        while !e.is_zero() {
            // Check if lowest bit is set
            let lowest = if e.limbs.is_empty() { 0 } else { e.limbs[0] & 1 };
            if lowest == 1 {
                result = result.mul(&base).div_rem(modulus).1;
            }
            base = base.mul(&base).div_rem(modulus).1;
            // Right-shift e by 1
            let mut carry = 0u32;
            for i in (0..e.limbs.len()).rev() {
                let new_carry = e.limbs[i] & 1;
                e.limbs[i] = (e.limbs[i] >> 1) | (carry << 31);
                carry = new_carry;
            }
            e.normalize();
        }

        result
    }

    /// Greatest common divisor (Euclidean algorithm).
    pub fn gcd(a: &Self, b: &Self) -> Self {
        let mut x = a.abs();
        let mut y = b.abs();
        while !y.is_zero() {
            let r = x.rem(&y);
            x = y;
            y = r;
        }
        x
    }

    /// Factorial: n!
    pub fn factorial(n: u64) -> Self {
        let mut result = Self::one();
        for i in 2..=n {
            result = result.mul(&Self::from_u64(i));
        }
        result
    }

    /// Convert to base-10 string.
    pub fn to_base10(&self) -> String {
        if self.is_zero() {
            return "0".to_string();
        }

        let mut digits = Vec::new();
        let mut val = self.abs();
        let ten = Self::from_u64(10);

        while !val.is_zero() {
            let (q, r) = val.div_rem(&ten);
            let digit = if r.limbs.is_empty() { 0 } else { r.limbs[0] };
            digits.push(std::char::from_digit(digit, 10).unwrap_or('0'));
            val = q;
        }

        if self.negative {
            digits.push('-');
        }
        digits.reverse();
        digits.into_iter().collect()
    }

    /// Convert to base-16 (hex) string with "0x" prefix.
    pub fn to_hex(&self) -> String {
        if self.is_zero() {
            return "0x0".to_string();
        }

        let mut hex = String::new();
        let mut first = true;
        for &limb in self.limbs.iter().rev() {
            if first {
                if limb != 0 {
                    hex.push_str(&format!("{:x}", limb));
                    first = false;
                }
            } else {
                hex.push_str(&format!("{:08x}", limb));
            }
        }
        if hex.is_empty() {
            hex.push('0');
        }

        let prefix = if self.negative { "-0x" } else { "0x" };
        format!("{prefix}{hex}")
    }

    /// Parse from a decimal string.
    pub fn from_str_radix10(s: &str) -> Option<Self> {
        let s = s.trim();
        if s.is_empty() { return None; }

        let (negative, digits) = if let Some(rest) = s.strip_prefix('-') {
            (true, rest)
        } else if let Some(rest) = s.strip_prefix('+') {
            (false, rest)
        } else {
            (false, s)
        };

        if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }

        let ten = Self::from_u64(10);
        let mut result = Self::zero();
        for ch in digits.chars() {
            let d = ch.to_digit(10)? as u64;
            result = result.mul(&ten).add(&Self::from_u64(d));
        }

        if negative && !result.is_zero() {
            result.negative = true;
        }
        Some(result)
    }

    /// Parse from a hex string (with or without "0x" prefix).
    pub fn from_str_radix16(s: &str) -> Option<Self> {
        let s = s.trim();
        let (negative, hex) = if let Some(rest) = s.strip_prefix('-') {
            (true, rest)
        } else {
            (false, s)
        };

        let hex = hex.strip_prefix("0x").or_else(|| hex.strip_prefix("0X")).unwrap_or(hex);
        if hex.is_empty() || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }

        let sixteen = Self::from_u64(16);
        let mut result = Self::zero();
        for ch in hex.chars() {
            let d = ch.to_digit(16)? as u64;
            result = result.mul(&sixteen).add(&Self::from_u64(d));
        }

        if negative && !result.is_zero() {
            result.negative = true;
        }
        Some(result)
    }

    /// Number of bits needed to represent this value.
    pub fn bit_length(&self) -> usize {
        if self.limbs.is_empty() { return 0; }
        let top = self.limbs.len() - 1;
        top * 32 + (32 - self.limbs[top].leading_zeros() as usize)
    }
}

impl PartialEq for BigInt {
    fn eq(&self, other: &Self) -> bool {
        self.negative == other.negative && self.limbs == other.limbs
    }
}

impl Eq for BigInt {}

impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BigInt {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.negative, other.negative) {
            (true, false) => {
                if self.is_zero() && other.is_zero() { Ordering::Equal }
                else { Ordering::Less }
            }
            (false, true) => {
                if self.is_zero() && other.is_zero() { Ordering::Equal }
                else { Ordering::Greater }
            }
            (false, false) => self.cmp_magnitude(other),
            (true, true) => other.cmp_magnitude(self), // reversed for negatives
        }
    }
}

impl fmt::Display for BigInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_base10())
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_i64_and_display() {
        assert_eq!(BigInt::from_i64(0).to_string(), "0");
        assert_eq!(BigInt::from_i64(42).to_string(), "42");
        assert_eq!(BigInt::from_i64(-99).to_string(), "-99");
    }

    #[test]
    fn addition_positive() {
        let a = BigInt::from_i64(123);
        let b = BigInt::from_i64(456);
        assert_eq!(a.add(&b).to_string(), "579");
    }

    #[test]
    fn addition_negative() {
        let a = BigInt::from_i64(-10);
        let b = BigInt::from_i64(3);
        assert_eq!(a.add(&b).to_string(), "-7");
    }

    #[test]
    fn subtraction() {
        let a = BigInt::from_i64(100);
        let b = BigInt::from_i64(30);
        assert_eq!(a.sub(&b).to_string(), "70");
        assert_eq!(b.sub(&a).to_string(), "-70");
    }

    #[test]
    fn multiplication() {
        let a = BigInt::from_i64(12345);
        let b = BigInt::from_i64(67890);
        assert_eq!(a.mul(&b).to_string(), "838102050");
    }

    #[test]
    fn multiplication_large() {
        // Test Karatsuba path by multiplying large numbers
        let a = BigInt::factorial(20);
        let b = BigInt::from_i64(2);
        let double = a.mul(&b);
        // 20! = 2432902008176640000
        assert_eq!(a.to_string(), "2432902008176640000");
        assert_eq!(double.to_string(), "4865804016353280000");
    }

    #[test]
    fn division() {
        let a = BigInt::from_i64(100);
        let b = BigInt::from_i64(7);
        let (q, r) = a.div_rem(&b);
        assert_eq!(q.to_string(), "14");
        assert_eq!(r.to_string(), "2");
    }

    #[test]
    fn division_exact() {
        let a = BigInt::from_i64(144);
        let b = BigInt::from_i64(12);
        let (q, r) = a.div_rem(&b);
        assert_eq!(q.to_string(), "12");
        assert!(r.is_zero());
    }

    #[test]
    fn comparison() {
        let a = BigInt::from_i64(100);
        let b = BigInt::from_i64(200);
        assert!(a < b);
        assert!(b > a);
        assert_eq!(BigInt::from_i64(5), BigInt::from_i64(5));
    }

    #[test]
    fn comparison_negative() {
        let a = BigInt::from_i64(-5);
        let b = BigInt::from_i64(3);
        assert!(a < b);
        assert!(BigInt::from_i64(-10) < BigInt::from_i64(-5));
    }

    #[test]
    fn gcd() {
        let a = BigInt::from_i64(48);
        let b = BigInt::from_i64(18);
        assert_eq!(BigInt::gcd(&a, &b).to_string(), "6");
    }

    #[test]
    fn mod_pow() {
        // 2^10 mod 1000 = 1024 mod 1000 = 24
        let base = BigInt::from_i64(2);
        let exp = BigInt::from_i64(10);
        let modulus = BigInt::from_i64(1000);
        assert_eq!(base.mod_pow(&exp, &modulus).to_string(), "24");
    }

    #[test]
    fn mod_pow_large() {
        // 3^13 mod 50 = 1594323 mod 50 = 23
        let base = BigInt::from_i64(3);
        let exp = BigInt::from_i64(13);
        let modulus = BigInt::from_i64(50);
        assert_eq!(base.mod_pow(&exp, &modulus).to_string(), "23");
    }

    #[test]
    fn factorial_small() {
        assert_eq!(BigInt::factorial(0).to_string(), "1");
        assert_eq!(BigInt::factorial(1).to_string(), "1");
        assert_eq!(BigInt::factorial(5).to_string(), "120");
        assert_eq!(BigInt::factorial(10).to_string(), "3628800");
    }

    #[test]
    fn to_hex() {
        assert_eq!(BigInt::from_i64(255).to_hex(), "0xff");
        assert_eq!(BigInt::from_i64(256).to_hex(), "0x100");
        assert_eq!(BigInt::from_i64(0).to_hex(), "0x0");
    }

    #[test]
    fn from_str_radix10() {
        let n = BigInt::from_str_radix10("12345678901234567890").unwrap();
        assert_eq!(n.to_string(), "12345678901234567890");
    }

    #[test]
    fn from_str_radix16() {
        let n = BigInt::from_str_radix16("0xff").unwrap();
        assert_eq!(n, BigInt::from_i64(255));

        let n2 = BigInt::from_str_radix16("-0xA").unwrap();
        assert_eq!(n2, BigInt::from_i64(-10));
    }

    #[test]
    fn bit_length() {
        assert_eq!(BigInt::from_i64(0).bit_length(), 0);
        assert_eq!(BigInt::from_i64(1).bit_length(), 1);
        assert_eq!(BigInt::from_i64(255).bit_length(), 8);
        assert_eq!(BigInt::from_i64(256).bit_length(), 9);
    }

    #[test]
    fn abs_and_negate() {
        let n = BigInt::from_i64(-42);
        assert_eq!(n.abs().to_string(), "42");
        assert_eq!(n.negate().to_string(), "42");
        assert_eq!(BigInt::zero().negate().to_string(), "0");
    }

    #[test]
    fn parse_roundtrip() {
        let original = "999999999999999999999999999";
        let n = BigInt::from_str_radix10(original).unwrap();
        assert_eq!(n.to_string(), original);
    }
}
