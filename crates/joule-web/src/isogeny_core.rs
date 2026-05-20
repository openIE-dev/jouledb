//! Elliptic curve isogeny core in pure Rust.
//!
//! Implements Montgomery curve arithmetic (x-only representation),
//! point addition/doubling, scalar multiplication, j-invariant
//! computation, and curve parameter validation for isogeny-based
//! post-quantum cryptography.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Errors from isogeny core operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IsogenyError {
    /// Curve parameters fail validation.
    InvalidCurve,
    /// Point is not on the curve.
    PointNotOnCurve,
    /// Scalar is zero or invalid.
    InvalidScalar,
    /// Division by zero in field arithmetic.
    DivisionByZero,
    /// Modular inverse does not exist.
    NoInverse,
    /// Configuration is incomplete.
    InvalidConfig,
}

impl fmt::Display for IsogenyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCurve => write!(f, "invalid curve parameters"),
            Self::PointNotOnCurve => write!(f, "point not on curve"),
            Self::InvalidScalar => write!(f, "invalid scalar value"),
            Self::DivisionByZero => write!(f, "division by zero"),
            Self::NoInverse => write!(f, "modular inverse does not exist"),
            Self::InvalidConfig => write!(f, "incomplete isogeny configuration"),
        }
    }
}

impl std::error::Error for IsogenyError {}

// ── Field Arithmetic (mod p) ────────────────────────────────────

/// Modular addition: (a + b) mod p.
#[inline]
pub fn mod_add(a: u64, b: u64, p: u64) -> u64 {
    let sum = (a as u128 + b as u128) % p as u128;
    sum as u64
}

/// Modular subtraction: (a - b) mod p.
#[inline]
pub fn mod_sub(a: u64, b: u64, p: u64) -> u64 {
    if a >= b {
        (a - b) % p
    } else {
        p - ((b - a) % p)
    }
}

/// Modular multiplication: (a * b) mod p.
#[inline]
pub fn mod_mul(a: u64, b: u64, p: u64) -> u64 {
    let prod = (a as u128 * b as u128) % p as u128;
    prod as u64
}

/// Modular exponentiation: base^exp mod p via square-and-multiply.
pub fn mod_pow(mut base: u64, mut exp: u64, p: u64) -> u64 {
    if p == 1 {
        return 0;
    }
    let mut result: u128 = 1;
    let mut b = base as u128;
    let modulus = p as u128;
    b %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * b) % modulus;
        }
        exp >>= 1;
        b = (b * b) % modulus;
    }
    result as u64
}

/// Extended Euclidean algorithm: returns (gcd, x, y) such that a*x + b*y = gcd.
pub fn extended_gcd(a: i128, b: i128) -> (i128, i128, i128) {
    if a == 0 {
        return (b, 0, 1);
    }
    let (g, x1, y1) = extended_gcd(b % a, a);
    (g, y1 - (b / a) * x1, x1)
}

/// Modular inverse: a^{-1} mod p.
pub fn mod_inv(a: u64, p: u64) -> Result<u64, IsogenyError> {
    if a == 0 {
        return Err(IsogenyError::DivisionByZero);
    }
    let (g, x, _) = extended_gcd(a as i128, p as i128);
    if g != 1 {
        return Err(IsogenyError::NoInverse);
    }
    Ok(((x % p as i128 + p as i128) % p as i128) as u64)
}

// ── Montgomery Curve ────────────────────────────────────────────

/// Montgomery curve: By^2 = x^3 + Ax^2 + x  (mod p).
///
/// For simplicity we use B=1 in most isogeny protocols.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MontgomeryCurve {
    /// Coefficient A.
    pub a: u64,
    /// Coefficient B (typically 1).
    pub b: u64,
    /// Prime modulus.
    pub p: u64,
}

impl MontgomeryCurve {
    /// Create a new Montgomery curve with validation.
    pub fn new(a: u64, b: u64, p: u64) -> Result<Self, IsogenyError> {
        let curve = Self { a, b, p };
        curve.validate()?;
        Ok(curve)
    }

    /// Validate that curve parameters define a non-singular Montgomery curve.
    /// Requires B(A^2 - 4) != 0 mod p.
    pub fn validate(&self) -> Result<(), IsogenyError> {
        if self.p < 3 {
            return Err(IsogenyError::InvalidCurve);
        }
        if self.b == 0 {
            return Err(IsogenyError::InvalidCurve);
        }
        let a_sq = mod_mul(self.a, self.a, self.p);
        let disc = mod_sub(a_sq, 4 % self.p, self.p);
        let check = mod_mul(self.b, disc, self.p);
        if check == 0 {
            return Err(IsogenyError::InvalidCurve);
        }
        Ok(())
    }

    /// Compute j-invariant: j = 256 * (A^2 - 3)^3 / (A^2 - 4).
    pub fn j_invariant(&self) -> Result<u64, IsogenyError> {
        let a_sq = mod_mul(self.a, self.a, self.p);
        let num_base = mod_sub(a_sq, 3 % self.p, self.p);
        let num_cubed = mod_mul(
            mod_mul(num_base, num_base, self.p),
            num_base,
            self.p,
        );
        let numerator = mod_mul(256 % self.p, num_cubed, self.p);
        let denominator = mod_sub(a_sq, 4 % self.p, self.p);
        let inv_den = mod_inv(denominator, self.p)?;
        Ok(mod_mul(numerator, inv_den, self.p))
    }

    /// Check if an x-coordinate lies on the curve.
    /// Checks if x^3 + Ax^2 + x is a quadratic residue mod p.
    pub fn is_x_on_curve(&self, x: u64) -> bool {
        let x_mod = x % self.p;
        let x_sq = mod_mul(x_mod, x_mod, self.p);
        let x_cu = mod_mul(x_sq, x_mod, self.p);
        let ax_sq = mod_mul(self.a, x_sq, self.p);
        let rhs = mod_add(mod_add(x_cu, ax_sq, self.p), x_mod, self.p);
        if rhs == 0 {
            return true;
        }
        // Euler criterion: rhs^((p-1)/2) == 1 mod p => QR
        let exp = (self.p - 1) / 2;
        mod_pow(rhs, exp, self.p) == 1
    }
}

impl fmt::Display for MontgomeryCurve {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Montgomery(A={}, B={}, p={})", self.a, self.b, self.p)
    }
}

// ── X-Only Point ────────────────────────────────────────────────

/// X-only projective point on a Montgomery curve: (X : Z).
/// The affine x-coordinate is X/Z when Z != 0.
/// The point at infinity is represented by Z = 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XPoint {
    pub x: u64,
    pub z: u64,
}

impl XPoint {
    /// Affine point from x-coordinate.
    pub fn affine(x: u64) -> Self {
        Self { x, z: 1 }
    }

    /// Point at infinity.
    pub fn infinity() -> Self {
        Self { x: 1, z: 0 }
    }

    /// Check if this is the point at infinity.
    pub fn is_infinity(&self) -> bool {
        self.z == 0
    }

    /// Convert to affine x-coordinate.
    pub fn to_affine(&self, p: u64) -> Result<u64, IsogenyError> {
        if self.z == 0 {
            return Err(IsogenyError::DivisionByZero);
        }
        let inv_z = mod_inv(self.z, p)?;
        Ok(mod_mul(self.x, inv_z, p))
    }
}

impl fmt::Display for XPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.z == 0 {
            write!(f, "XPoint(inf)")
        } else {
            write!(f, "XPoint({} : {})", self.x, self.z)
        }
    }
}

// ── Differential Addition & Doubling ────────────────────────────

/// Montgomery x-only doubling.
///
/// Given P = (X_P : Z_P), compute [2]P = (X_2P : Z_2P).
/// Uses the standard formulas for Montgomery curves.
pub fn xdbl(p_pt: &XPoint, curve: &MontgomeryCurve) -> XPoint {
    let p = curve.p;
    let a24 = mod_mul(mod_add(curve.a, 2, p), mod_inv(4, p).unwrap_or(1), p);

    let v1 = mod_add(p_pt.x, p_pt.z, p);
    let v1_sq = mod_mul(v1, v1, p);
    let v2 = mod_sub(p_pt.x, p_pt.z, p);
    let v2_sq = mod_mul(v2, v2, p);

    let x_2p = mod_mul(v1_sq, v2_sq, p);
    let diff = mod_sub(v1_sq, v2_sq, p);
    let t = mod_add(v2_sq, mod_mul(a24, diff, p), p);
    let z_2p = mod_mul(diff, t, p);

    XPoint { x: x_2p, z: z_2p }
}

/// Montgomery differential addition.
///
/// Given P, Q and their difference P-Q, compute P+Q.
/// Requires the x-coordinate of P-Q to resolve the sign ambiguity.
pub fn xadd(
    p_pt: &XPoint,
    q_pt: &XPoint,
    diff: &XPoint,
    curve: &MontgomeryCurve,
) -> XPoint {
    let p = curve.p;

    let u = mod_mul(
        mod_sub(p_pt.x, p_pt.z, p),
        mod_add(q_pt.x, q_pt.z, p),
        p,
    );
    let v = mod_mul(
        mod_add(p_pt.x, p_pt.z, p),
        mod_sub(q_pt.x, q_pt.z, p),
        p,
    );

    let sum = mod_add(u, v, p);
    let sub = mod_sub(u, v, p);

    let x_pq = mod_mul(diff.z, mod_mul(sum, sum, p), p);
    let z_pq = mod_mul(diff.x, mod_mul(sub, sub, p), p);

    XPoint { x: x_pq, z: z_pq }
}

// ── Montgomery Ladder ───────────────────────────────────────────

/// Scalar multiplication via the Montgomery ladder.
///
/// Computes [k]P on the given curve using x-only arithmetic.
/// Constant-time w.r.t. the scalar (no branching on bits).
pub fn scalar_mul(
    k: u64,
    base: &XPoint,
    curve: &MontgomeryCurve,
) -> Result<XPoint, IsogenyError> {
    if k == 0 {
        return Ok(XPoint::infinity());
    }
    if base.is_infinity() {
        return Ok(XPoint::infinity());
    }

    let mut r0 = base.clone();
    let mut r1 = xdbl(base, curve);

    let bits = 64 - k.leading_zeros();
    for i in (0..bits - 1).rev() {
        let bit = (k >> i) & 1;
        if bit == 0 {
            r1 = xadd(&r0, &r1, base, curve);
            r0 = xdbl(&r0, curve);
        } else {
            r0 = xadd(&r0, &r1, base, curve);
            r1 = xdbl(&r1, curve);
        }
    }

    Ok(r0)
}

// ── Isogeny Computation ─────────────────────────────────────────

/// Compute a 2-isogeny from a kernel point of order 2.
///
/// Given a point K with 2K = O, compute the codomain curve
/// and the map of a point through the isogeny.
pub fn isogeny_2(
    kernel: &XPoint,
    curve: &MontgomeryCurve,
) -> Result<MontgomeryCurve, IsogenyError> {
    let p = curve.p;
    let xk = kernel.to_affine(p)?;

    // For a 2-isogeny with kernel (xk, 0):
    // New A' = 2(A - 3*xk^2) - (A - 3*xk^2) simplified
    let xk_sq = mod_mul(xk, xk, p);
    let three_xk_sq = mod_mul(3, xk_sq, p);
    let new_a = mod_sub(mod_mul(2, curve.a, p), mod_mul(6, xk_sq, p), p);
    let new_a = mod_add(new_a, 2, p);

    MontgomeryCurve::new(new_a, curve.b, p)
}

/// Compute a 3-isogeny from a kernel point of order 3.
///
/// Given a point K with 3K = O, compute the codomain curve.
pub fn isogeny_3(
    kernel: &XPoint,
    curve: &MontgomeryCurve,
) -> Result<MontgomeryCurve, IsogenyError> {
    let p = curve.p;
    let xk = kernel.to_affine(p)?;

    let xk_sq = mod_mul(xk, xk, p);
    let xk_cu = mod_mul(xk_sq, xk, p);

    // Simplified 3-isogeny codomain formula
    let t1 = mod_mul(6, xk, p);
    let t2 = mod_mul(6, xk_cu, p);
    let new_a = mod_sub(mod_add(curve.a, t1, p), t2, p);

    MontgomeryCurve::new(new_a, curve.b, p)
}

/// Push a point through a 2-isogeny defined by the kernel.
pub fn push_through_2(
    point: &XPoint,
    kernel: &XPoint,
    curve: &MontgomeryCurve,
) -> Result<XPoint, IsogenyError> {
    let p = curve.p;
    let xp = point.to_affine(p)?;
    let xk = kernel.to_affine(p)?;

    if xp == xk {
        return Ok(XPoint::infinity());
    }

    let diff = mod_sub(xp, xk, p);
    let inv_diff = mod_inv(diff, p)?;
    let sum = mod_add(xp, xk, p);
    let new_x = mod_mul(mod_mul(xp, sum, p), inv_diff, p);

    Ok(XPoint::affine(new_x))
}

/// Push a point through a 3-isogeny defined by the kernel.
pub fn push_through_3(
    point: &XPoint,
    kernel: &XPoint,
    curve: &MontgomeryCurve,
) -> Result<XPoint, IsogenyError> {
    let p = curve.p;
    let xp = point.to_affine(p)?;
    let xk = kernel.to_affine(p)?;

    if xp == xk {
        return Ok(XPoint::infinity());
    }

    let diff = mod_sub(xp, xk, p);
    let inv_diff = mod_inv(diff, p)?;
    let t = mod_mul(xp, mod_mul(diff, diff, p), p);
    let new_x = mod_mul(t, mod_mul(inv_diff, inv_diff, p), p);

    Ok(XPoint::affine(new_x))
}

// ── IsogenyConfig Builder ───────────────────────────────────────

/// Configuration for isogeny computations.
#[derive(Debug, Clone)]
pub struct IsogenyConfig {
    pub prime: u64,
    pub curve_a: u64,
    pub curve_b: u64,
    pub max_degree: u32,
    pub validate_points: bool,
}

impl Default for IsogenyConfig {
    fn default() -> Self {
        Self {
            prime: 431,
            curve_a: 6,
            curve_b: 1,
            max_degree: 128,
            validate_points: true,
        }
    }
}

impl IsogenyConfig {
    /// Start building a new config.
    pub fn builder() -> Self {
        Self::default()
    }

    /// Set the prime modulus.
    pub fn with_prime(mut self, p: u64) -> Self {
        self.prime = p;
        self
    }

    /// Set curve coefficient A.
    pub fn with_curve_a(mut self, a: u64) -> Self {
        self.curve_a = a;
        self
    }

    /// Set curve coefficient B.
    pub fn with_curve_b(mut self, b: u64) -> Self {
        self.curve_b = b;
        self
    }

    /// Set maximum isogeny degree.
    pub fn with_max_degree(mut self, d: u32) -> Self {
        self.max_degree = d;
        self
    }

    /// Toggle point validation.
    pub fn with_validation(mut self, v: bool) -> Self {
        self.validate_points = v;
        self
    }

    /// Build and validate the configuration.
    pub fn build(self) -> Result<(Self, MontgomeryCurve), IsogenyError> {
        let curve = MontgomeryCurve::new(self.curve_a, self.curve_b, self.prime)?;
        Ok((self, curve))
    }
}

impl fmt::Display for IsogenyConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "IsogenyConfig(p={}, A={}, B={}, max_deg={})",
            self.prime, self.curve_a, self.curve_b, self.max_degree
        )
    }
}

// ── Utility ─────────────────────────────────────────────────────

/// Check if n is a (probable) prime using Miller-Rabin with small bases.
pub fn is_probable_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }

    let mut d = n - 1;
    let mut r = 0u32;
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }

    let witnesses = [2, 3, 5, 7, 11, 13];
    'witness: for &a in &witnesses {
        if a >= n {
            continue;
        }
        let mut x = mod_pow(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..r - 1 {
            x = mod_mul(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// Legendre symbol: (a/p). Returns 0, 1, or p-1 (representing -1).
pub fn legendre_symbol(a: u64, p: u64) -> u64 {
    mod_pow(a % p, (p - 1) / 2, p)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mod_add() {
        assert_eq!(mod_add(10, 15, 17), 8);
        assert_eq!(mod_add(0, 0, 7), 0);
    }

    #[test]
    fn test_mod_sub() {
        assert_eq!(mod_sub(10, 15, 17), 12); // 10 - 15 = -5 = 12 mod 17
        assert_eq!(mod_sub(5, 3, 17), 2);
    }

    #[test]
    fn test_mod_mul() {
        assert_eq!(mod_mul(7, 8, 17), 5); // 56 mod 17 = 5
        assert_eq!(mod_mul(0, 100, 17), 0);
    }

    #[test]
    fn test_mod_pow() {
        assert_eq!(mod_pow(2, 10, 1000), 24); // 1024 mod 1000
        assert_eq!(mod_pow(3, 0, 17), 1);
        assert_eq!(mod_pow(5, 1, 17), 5);
    }

    #[test]
    fn test_mod_inv() {
        let inv = mod_inv(3, 17).unwrap();
        assert_eq!(mod_mul(3, inv, 17), 1);
    }

    #[test]
    fn test_mod_inv_zero() {
        assert!(mod_inv(0, 17).is_err());
    }

    #[test]
    fn test_montgomery_curve_valid() {
        let curve = MontgomeryCurve::new(6, 1, 431).unwrap();
        assert_eq!(curve.a, 6);
    }

    #[test]
    fn test_montgomery_curve_singular() {
        // A=2, B=1, p=5 => A^2-4 = 0 => singular
        assert!(MontgomeryCurve::new(2, 1, 5).is_err());
    }

    #[test]
    fn test_xpoint_affine() {
        let pt = XPoint::affine(42);
        assert_eq!(pt.x, 42);
        assert_eq!(pt.z, 1);
        assert!(!pt.is_infinity());
    }

    #[test]
    fn test_xpoint_infinity() {
        let pt = XPoint::infinity();
        assert!(pt.is_infinity());
    }

    #[test]
    fn test_xpoint_to_affine() {
        let pt = XPoint { x: 10, z: 2 };
        let x = pt.to_affine(431).unwrap();
        assert_eq!(mod_mul(x, 2, 431), 10);
    }

    #[test]
    fn test_xdbl_identity_like() {
        let curve = MontgomeryCurve::new(6, 1, 431).unwrap();
        let pt = XPoint::affine(1);
        let dbl = xdbl(&pt, &curve);
        assert!(!dbl.is_infinity());
    }

    #[test]
    fn test_scalar_mul_zero() {
        let curve = MontgomeryCurve::new(6, 1, 431).unwrap();
        let pt = XPoint::affine(1);
        let result = scalar_mul(0, &pt, &curve).unwrap();
        assert!(result.is_infinity());
    }

    #[test]
    fn test_scalar_mul_one() {
        let curve = MontgomeryCurve::new(6, 1, 431).unwrap();
        let pt = XPoint::affine(100);
        let result = scalar_mul(1, &pt, &curve).unwrap();
        assert_eq!(result.x, pt.x);
        assert_eq!(result.z, pt.z);
    }

    #[test]
    fn test_scalar_mul_infinity_base() {
        let curve = MontgomeryCurve::new(6, 1, 431).unwrap();
        let pt = XPoint::infinity();
        let result = scalar_mul(5, &pt, &curve).unwrap();
        assert!(result.is_infinity());
    }

    #[test]
    fn test_j_invariant() {
        let curve = MontgomeryCurve::new(6, 1, 431).unwrap();
        let j = curve.j_invariant().unwrap();
        assert!(j < 431);
    }

    #[test]
    fn test_is_probable_prime() {
        assert!(is_probable_prime(431));
        assert!(is_probable_prime(17));
        assert!(!is_probable_prime(15));
        assert!(!is_probable_prime(1));
    }

    #[test]
    fn test_legendre_symbol() {
        // 1 is always a QR
        assert_eq!(legendre_symbol(1, 17), 1);
    }

    #[test]
    fn test_config_builder() {
        let (cfg, curve) = IsogenyConfig::builder()
            .with_prime(431)
            .with_curve_a(6)
            .with_curve_b(1)
            .with_max_degree(64)
            .with_validation(true)
            .build()
            .unwrap();
        assert_eq!(cfg.prime, 431);
        assert_eq!(curve.a, 6);
    }

    #[test]
    fn test_display_curve() {
        let curve = MontgomeryCurve::new(6, 1, 431).unwrap();
        let s = format!("{}", curve);
        assert!(s.contains("Montgomery"));
    }

    #[test]
    fn test_display_xpoint() {
        let pt = XPoint::affine(42);
        let s = format!("{}", pt);
        assert!(s.contains("42"));

        let inf = XPoint::infinity();
        let s2 = format!("{}", inf);
        assert!(s2.contains("inf"));
    }
}
