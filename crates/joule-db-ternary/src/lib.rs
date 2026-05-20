//! Packed ternary encoding: 5 trits per byte (1.58 bits/trit).
//!
//! Stores balanced ternary values {-1, 0, +1} with LUT-based decoding
//! and NEON SIMD dot products on aarch64.

pub mod pack;

/// A balanced ternary digit: {-1, 0, +1}.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(i8)]
pub enum Trit {
    /// Negative / inhibitory / false.
    Neg = -1,
    /// Zero / no connection / null.
    Zero = 0,
    /// Positive / excitatory / true.
    Pos = 1,
}

impl Trit {
    /// Convert to i8.
    #[inline]
    pub fn to_i8(self) -> i8 {
        self as i8
    }

    /// Convert from i8. Panics if value is not -1, 0, or 1.
    #[inline]
    pub fn from_i8(v: i8) -> Self {
        match v {
            -1 => Trit::Neg,
            0 => Trit::Zero,
            1 => Trit::Pos,
            _ => panic!("invalid trit value: {v}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trit_roundtrip() {
        for &t in &[Trit::Neg, Trit::Zero, Trit::Pos] {
            assert_eq!(Trit::from_i8(t.to_i8()), t);
        }
    }

    #[test]
    #[should_panic]
    fn trit_invalid() {
        Trit::from_i8(2);
    }
}
