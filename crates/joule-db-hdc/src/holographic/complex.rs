//! Complex Number Operations
//!
//! Complex number representation and operations for HAM.

use std::ops::{Add, Mul, Sub};

/// Complex number with f32 components
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    /// Real part
    pub real: f32,
    /// Imaginary part
    pub imag: f32,
}

impl Complex {
    /// Create new complex number
    pub fn new(real: f32, imag: f32) -> Self {
        Self { real, imag }
    }

    /// Create from polar coordinates
    pub fn from_polar(magnitude: f32, angle: f32) -> Self {
        Self {
            real: magnitude * angle.cos(),
            imag: magnitude * angle.sin(),
        }
    }

    /// Create a unit complex number at given angle
    pub fn unit(angle: f32) -> Self {
        Self::from_polar(1.0, angle)
    }

    /// Zero complex number
    pub fn zero() -> Self {
        Self {
            real: 0.0,
            imag: 0.0,
        }
    }

    /// Complex conjugate
    pub fn conjugate(&self) -> Complex {
        Complex {
            real: self.real,
            imag: -self.imag,
        }
    }

    /// Magnitude squared (|z|²)
    pub fn magnitude_squared(&self) -> f32 {
        self.real * self.real + self.imag * self.imag
    }

    /// Magnitude (|z|)
    pub fn magnitude(&self) -> f32 {
        self.magnitude_squared().sqrt()
    }

    /// Phase angle in radians
    pub fn phase(&self) -> f32 {
        self.imag.atan2(self.real)
    }

    /// Normalize to unit magnitude
    pub fn normalize(&self) -> Complex {
        let mag = self.magnitude();
        if mag > 0.0 {
            Complex {
                real: self.real / mag,
                imag: self.imag / mag,
            }
        } else {
            Complex::zero()
        }
    }
}

impl Add for Complex {
    type Output = Complex;

    fn add(self, other: Complex) -> Complex {
        Complex {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }
}

impl Sub for Complex {
    type Output = Complex;

    fn sub(self, other: Complex) -> Complex {
        Complex {
            real: self.real - other.real,
            imag: self.imag - other.imag,
        }
    }
}

impl Mul for Complex {
    type Output = Complex;

    fn mul(self, other: Complex) -> Complex {
        Complex {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }
}

impl Add<&Complex> for &Complex {
    type Output = Complex;

    fn add(self, other: &Complex) -> Complex {
        Complex {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }
}

impl Mul<&Complex> for &Complex {
    type Output = Complex;

    fn mul(self, other: &Complex) -> Complex {
        Complex {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }
}

impl Default for Complex {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_new() {
        let c = Complex::new(3.0, 4.0);
        assert_eq!(c.real, 3.0);
        assert_eq!(c.imag, 4.0);
    }

    #[test]
    fn test_complex_magnitude() {
        let c = Complex::new(3.0, 4.0);
        assert!((c.magnitude() - 5.0).abs() < 0.0001);
    }

    #[test]
    fn test_complex_conjugate() {
        let c = Complex::new(3.0, 4.0);
        let conj = c.conjugate();
        assert_eq!(conj.real, 3.0);
        assert_eq!(conj.imag, -4.0);
    }

    #[test]
    fn test_complex_multiply() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let c = a * b;
        // (1 + 2i)(3 + 4i) = 3 + 4i + 6i + 8i² = 3 + 10i - 8 = -5 + 10i
        assert!((c.real - (-5.0)).abs() < 0.0001);
        assert!((c.imag - 10.0).abs() < 0.0001);
    }

    #[test]
    fn test_complex_add() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let c = a + b;
        assert_eq!(c.real, 4.0);
        assert_eq!(c.imag, 6.0);
    }

    #[test]
    fn test_complex_from_polar() {
        let c = Complex::from_polar(2.0, std::f32::consts::PI / 2.0);
        assert!(c.real.abs() < 0.0001);
        assert!((c.imag - 2.0).abs() < 0.0001);
    }

    #[test]
    fn test_complex_normalize() {
        let c = Complex::new(3.0, 4.0);
        let n = c.normalize();
        assert!((n.magnitude() - 1.0).abs() < 0.0001);
    }
}
