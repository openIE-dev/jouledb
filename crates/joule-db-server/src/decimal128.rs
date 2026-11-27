//! Exact 128-bit Fixed-Point Decimal (`NUMERIC(38, s)` / `DECIMAL(38, s)`)
//!
//! Eliminates IEEE-754 binary floating-point rounding drift (`0.10 + 0.20 != 0.30`)
//! across SQL arithmetic, comparisons, `CAST(... AS NUMERIC(p, s))`, and aggregates.
//! Represents numbers exactly as $m \times 10^{-s}$ where $m \in \texttt{i128}$ (up to
//! 38 decimal digits) and $s \in [0, 18]$.

use std::cmp::Ordering;

pub const MAX_DECIMAL_SCALE: u8 = 18;

const POW10: [i128; 19] = [
    1,
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
    10_000_000_000,
    100_000_000_000,
    1_000_000_000_000,
    10_000_000_000_000,
    100_000_000_000_000,
    1_000_000_000_000_000,
    10_000_000_000_000_000,
    100_000_000_000_000_000,
    1_000_000_000_000_000_000,
];

/// Exact 128-bit scaled fixed-point decimal ($m \times 10^{-\text{scale}}$).
#[derive(Debug, Clone, Copy, Eq)]
pub struct Decimal128 {
    pub mantissa: i128,
    pub scale: u8,
}

impl Decimal128 {
    #[inline]
    pub const fn new(mantissa: i128, scale: u8) -> Option<Self> {
        if scale > MAX_DECIMAL_SCALE {
            None
        } else {
            Some(Self { mantissa, scale })
        }
    }

    #[inline]
    pub const fn from_i64(v: i64) -> Self {
        Self {
            mantissa: v as i128,
            scale: 0,
        }
    }

    /// Parse an exact base-10 string (`[+-]?digits(.digits)?`, no `e`/`E` exponent)
    /// into a `Decimal128` when fractional digits <= 18 and total digits <= 38.
    pub fn parse_str(input: &str) -> Option<Self> {
        let s = input.trim();
        if s.is_empty() || s.contains(['e', 'E']) {
            return None;
        }
        let (neg, rest) = if let Some(r) = s.strip_prefix('-') {
            (true, r)
        } else if let Some(r) = s.strip_prefix('+') {
            (false, r)
        } else {
            (false, s)
        };
        if rest.is_empty() {
            return None;
        }

        let (int_part, frac_part) = match rest.split_once('.') {
            Some((i, f)) => {
                if f.contains('.') {
                    return None;
                }
                (i, Some(f))
            }
            None => (rest, None),
        };

        if int_part.is_empty() && frac_part.is_none_or(|f| f.is_empty()) {
            return None;
        }
        if !int_part.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let frac_str = frac_part.unwrap_or("");
        if !frac_str.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }

        // Trim redundant trailing fractional zeros beyond MAX_DECIMAL_SCALE if present
        let mut frac_trimmed = frac_str;
        while frac_trimmed.len() > MAX_DECIMAL_SCALE as usize && frac_trimmed.ends_with('0') {
            frac_trimmed = &frac_trimmed[..frac_trimmed.len() - 1];
        }
        if frac_trimmed.len() > MAX_DECIMAL_SCALE as usize {
            return None;
        }

        let int_clean = int_part.trim_start_matches('0');
        if int_clean.len() + frac_trimmed.len() > 38 {
            return None;
        }

        let mut acc: i128 = 0;
        for b in int_clean.bytes().chain(frac_trimmed.bytes()) {
            acc = acc.checked_mul(10)?;
            acc = acc.checked_add((b - b'0') as i128)?;
        }
        if neg && acc != 0 {
            acc = acc.checked_neg()?;
        }
        Some(Self {
            mantissa: acc,
            scale: frac_trimmed.len() as u8,
        })
    }

    /// Extract exact `Decimal128` from a `serde_json::Number` (unless scientific notation).
    pub fn from_json_number(n: &serde_json::Number) -> Option<Self> {
        if let Some(i) = n.as_i64() {
            return Some(Self::from_i64(i));
        }
        if let Some(u) = n.as_u64() {
            return Some(Self {
                mantissa: u as i128,
                scale: 0,
            });
        }
        Self::parse_str(&n.to_string())
    }

    /// Extract exact `Decimal128` from a `serde_json::Value` (Number or exact numeric String).
    pub fn from_json_value(v: &serde_json::Value) -> Option<Self> {
        match v {
            serde_json::Value::Number(n) => Self::from_json_number(n),
            serde_json::Value::String(s) => Self::parse_str(s),
            _ => None,
        }
    }

    /// Strip trailing fractional zeros (`12.300` -> `12.3`, `10.00` -> `10`).
    #[inline]
    pub fn normalized(mut self) -> Self {
        while self.scale > 0 && self.mantissa % 10 == 0 {
            self.mantissa /= 10;
            self.scale -= 1;
        }
        if self.mantissa == 0 {
            self.scale = 0;
        }
        self
    }

    /// Align two decimals to `max(self.scale, other.scale)`.
    #[inline]
    pub fn align(self, other: Self) -> Option<(i128, i128, u8)> {
        let target = self.scale.max(other.scale);
        let a_mul = POW10.get((target - self.scale) as usize).copied()?;
        let b_mul = POW10.get((target - other.scale) as usize).copied()?;
        let ma = self.mantissa.checked_mul(a_mul)?;
        let mb = other.mantissa.checked_mul(b_mul)?;
        Some((ma, mb, target))
    }

    #[inline]
    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        let (a, b, sc) = self.align(rhs)?;
        Some(Self {
            mantissa: a.checked_add(b)?,
            scale: sc,
        })
    }

    #[inline]
    pub fn checked_sub(self, rhs: Self) -> Option<Self> {
        let (a, b, sc) = self.align(rhs)?;
        Some(Self {
            mantissa: a.checked_sub(b)?,
            scale: sc,
        })
    }

    #[inline]
    pub fn checked_mul(self, rhs: Self) -> Option<Self> {
        let a = self.normalized();
        let b = rhs.normalized();
        let sc = a.scale.checked_add(b.scale)?;
        if sc > MAX_DECIMAL_SCALE {
            return None;
        }
        Some(Self {
            mantissa: a.mantissa.checked_mul(b.mantissa)?,
            scale: sc,
        })
    }

    /// Exact division rounded half-even (banker's rounding) to `target_scale` decimal places.
    pub fn checked_div_round_even(self, rhs: Self, target_scale: u8) -> Option<Self> {
        if rhs.mantissa == 0 || target_scale > MAX_DECIMAL_SCALE {
            return None;
        }
        // Compute (self.mantissa * 10^(target_scale + rhs.scale - self.scale)) / rhs.mantissa
        let shift: i32 = (target_scale as i32) + (rhs.scale as i32) - (self.scale as i32);
        let (num, den) = if shift >= 0 {
            if shift > 38 {
                return None;
            }
            let mul = pow10_i128(shift as u32)?;
            (self.mantissa.checked_mul(mul)?, rhs.mantissa)
        } else {
            let mul = pow10_i128((-shift) as u32)?;
            (self.mantissa, rhs.mantissa.checked_mul(mul)?)
        };
        let q = div_round_half_even(num, den)?;
        Some(Self {
            mantissa: q,
            scale: target_scale,
        })
    }

    /// Rescale to `target_scale` using half-away-from-zero SQL `ROUND` / `CAST AS NUMERIC(p, s)` semantics.
    pub fn rescale_half_up(self, target_scale: u8) -> Option<Self> {
        if target_scale > MAX_DECIMAL_SCALE {
            return None;
        }
        if target_scale >= self.scale {
            let mul = POW10.get((target_scale - self.scale) as usize).copied()?;
            return Some(Self {
                mantissa: self.mantissa.checked_mul(mul)?,
                scale: target_scale,
            });
        }
        let drop_digits = (self.scale - target_scale) as usize;
        let div = POW10.get(drop_digits).copied()?;
        let q = self.mantissa / div;
        let rem = (self.mantissa % div).abs();
        let half = div / 2;
        let rounded = if rem >= half {
            if self.mantissa >= 0 {
                q.checked_add(1)?
            } else {
                q.checked_sub(1)?
            }
        } else {
            q
        };
        Some(Self {
            mantissa: rounded,
            scale: target_scale,
        })
    }

    /// Format with exact `self.scale` decimal places (e.g. `(30, 2)` -> `"0.30"`).
    pub fn format_fixed(self) -> String {
        if self.scale == 0 {
            return self.mantissa.to_string();
        }
        let neg = self.mantissa < 0;
        let abs = self.mantissa.unsigned_abs();
        let div = POW10[self.scale as usize] as u128;
        let int_part = abs / div;
        let frac_part = abs % div;
        let sign = if neg { "-" } else { "" };
        format!(
            "{}{}.{:0width$}",
            sign,
            int_part,
            frac_part,
            width = self.scale as usize
        )
    }

    /// Format stripped of redundant trailing fractional zeros (`(30, 2)` -> `"0.3"`, `(100, 2)` -> `"1"`).
    pub fn format_normalized(self) -> String {
        self.normalized().format_fixed()
    }

    /// Convert to `serde_json::Value`:
    /// - Exact `i64` JSON integer if `scale == 0` and within `i64`.
    /// - Exact decimal JSON `Number` if significant digits <= 15 (roundtrips losslessly in IEEE-754).
    /// - Exact string if > 15 significant digits so 16..38-digit banking amounts never lose pennies!
    pub fn to_json_number_or_exact(self) -> serde_json::Value {
        let norm = self.normalized();
        if norm.scale == 0 {
            if let Ok(i) = i64::try_from(norm.mantissa) {
                return serde_json::Value::Number(i.into());
            }
        }
        let abs_digits = count_digits_u128(norm.mantissa.unsigned_abs());
        if abs_digits <= 15 {
            let text = norm.format_fixed();
            if let Ok(f) = text.parse::<f64>()
                && let Some(num) = serde_json::Number::from_f64(f)
            {
                return serde_json::Value::Number(num);
            }
        }
        serde_json::Value::String(self.format_fixed())
    }
}

impl PartialEq for Decimal128 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        let a = self.normalized();
        let b = other.normalized();
        a.mantissa == b.mantissa && a.scale == b.scale
    }
}

impl PartialOrd for Decimal128 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Decimal128 {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        if let Some((a, b, _)) = self.align(*other) {
            return a.cmp(&b);
        }
        let an = self.normalized();
        let bn = other.normalized();
        if let Some((a, b, _)) = an.align(bn) {
            return a.cmp(&b);
        }
        // Fallback sign/magnitude comparison when alignment exceeds i128
        match (an.mantissa.signum(), bn.mantissa.signum()) {
            (sa, sb) if sa != sb => sa.cmp(&sb),
            (0, 0) => Ordering::Equal,
            (sign, _) => {
                let da = count_digits_u128(an.mantissa.unsigned_abs()) as i32 - an.scale as i32;
                let db = count_digits_u128(bn.mantissa.unsigned_abs()) as i32 - bn.scale as i32;
                let ord = da.cmp(&db);
                if sign < 0 { ord.reverse() } else { ord }
            }
        }
    }
}

/// Exact `SUM` / `AVG` accumulator over `Decimal128` values.
#[derive(Debug, Clone, Default)]
pub struct Decimal128Accumulator {
    current: Option<Decimal128>,
    count: u64,
}

impl Decimal128Accumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, val: Decimal128) -> bool {
        let next = match self.current {
            None => Some(val),
            Some(cur) => cur
                .checked_add(val)
                .or_else(|| cur.normalized().checked_add(val.normalized())),
        };
        if let Some(v) = next {
            self.current = Some(v);
            self.count += 1;
            true
        } else {
            false
        }
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn sum(&self) -> Option<Decimal128> {
        self.current
    }

    pub fn avg(&self, scale: u8) -> Option<Decimal128> {
        let sum = self.current?;
        if self.count == 0 {
            return None;
        }
        let denom = Decimal128 {
            mantissa: self.count as i128,
            scale: 0,
        };
        sum.checked_div_round_even(denom, scale)
    }
}

fn pow10_i128(exp: u32) -> Option<i128> {
    if (exp as usize) < POW10.len() {
        return Some(POW10[exp as usize]);
    }
    10_i128.checked_pow(exp)
}

fn div_round_half_even(num: i128, den: i128) -> Option<i128> {
    if den == 0 {
        return None;
    }
    let q = num.checked_div(den)?;
    let r = (num % den).unsigned_abs();
    let abs_den = den.unsigned_abs();
    let twice_r = r.checked_mul(2)?;
    let neg = (num < 0) ^ (den < 0);
    let adjust = match twice_r.cmp(&abs_den) {
        Ordering::Less => false,
        Ordering::Greater => true,
        Ordering::Equal => (q.unsigned_abs() & 1) == 1,
    };
    if adjust {
        if neg {
            q.checked_sub(1)
        } else {
            q.checked_add(1)
        }
    } else {
        Some(q)
    }
}

fn count_digits_u128(mut v: u128) -> usize {
    if v == 0 {
        return 1;
    }
    let mut d = 0;
    while v > 0 {
        v /= 10;
        d += 1;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_point_one_plus_point_two_equals_point_three() {
        let a = Decimal128::parse_str("0.10").unwrap();
        let b = Decimal128::parse_str("0.20").unwrap();
        let c = Decimal128::parse_str("0.3").unwrap();
        let sum = a.checked_add(b).unwrap();
        assert_eq!(sum, c);
        assert_eq!(sum.format_fixed(), "0.30");
        assert_eq!(sum.format_normalized(), "0.3");
        assert_eq!(sum.to_json_number_or_exact(), serde_json::json!(0.3));
    }

    #[test]
    fn institutional_30_digit_wire_transfer_preserves_every_cent() {
        let vault = Decimal128::parse_str("999999999999999999999999.99").unwrap();
        let fee = Decimal128::parse_str("0.01").unwrap();
        let after = vault.checked_add(fee).unwrap();
        assert_eq!(after.format_fixed(), "1000000000000000000000000.00");
        let back = after.checked_sub(fee).unwrap();
        assert_eq!(back, vault);
    }

    #[test]
    fn hundred_thousand_penny_additions_zero_drift() {
        let penny = Decimal128::parse_str("0.01").unwrap();
        let mut acc = Decimal128Accumulator::new();
        for _ in 0..100_000 {
            assert!(acc.push(penny));
        }
        let total = acc.sum().unwrap();
        assert_eq!(total, Decimal128::from_i64(1_000));
        assert_eq!(total.to_json_number_or_exact(), serde_json::json!(1000));
    }

    #[test]
    fn bankers_rounding_half_even() {
        let a = Decimal128::parse_str("2.5").unwrap();
        let one = Decimal128::from_i64(1);
        assert_eq!(
            a.checked_div_round_even(one, 0).unwrap(),
            Decimal128::from_i64(2)
        );
        let b = Decimal128::parse_str("3.5").unwrap();
        assert_eq!(
            b.checked_div_round_even(one, 0).unwrap(),
            Decimal128::from_i64(4)
        );
    }
}
