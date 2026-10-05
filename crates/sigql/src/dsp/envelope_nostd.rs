//! no_std stubs for envelope / Hilbert (rustfft required).

use super::{DspError, DspResult};
use alloc::string::String;
use alloc::vec::Vec;
use num_complex::Complex64;

pub struct HilbertTransform {
    size: usize,
}

impl HilbertTransform {
    pub fn new(size: usize) -> DspResult<Self> {
        if !size.is_power_of_two() {
            return Err(DspError::InvalidFftSize(size));
        }
        Ok(Self { size })
    }

    pub fn analytic_signal(&self, _signal: &[f64]) -> DspResult<Vec<Complex64>> {
        Err(DspError::InvalidParameter(String::from(
            "HilbertTransform requires the std feature (rustfft)",
        )))
    }

    pub fn envelope(&self, _signal: &[f64]) -> DspResult<Vec<f64>> {
        Err(DspError::InvalidParameter(String::from(
            "HilbertTransform requires the std feature (rustfft)",
        )))
    }

    pub fn instantaneous_phase(&self, _signal: &[f64]) -> DspResult<Vec<f64>> {
        Err(DspError::InvalidParameter(String::from(
            "HilbertTransform requires the std feature (rustfft)",
        )))
    }

    pub fn instantaneous_frequency(
        &self,
        _signal: &[f64],
        _sample_rate: crate::types::SampleRate,
    ) -> DspResult<Vec<f64>> {
        Err(DspError::InvalidParameter(String::from(
            "HilbertTransform requires the std feature (rustfft)",
        )))
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

pub struct EnvelopeExtractor;

impl EnvelopeExtractor {
    pub fn new() -> Self {
        Self
    }

    pub fn extract(&self, _signal: &[f64]) -> DspResult<Vec<f64>> {
        Err(DspError::InvalidParameter(String::from(
            "EnvelopeExtractor requires the std feature (rustfft)",
        )))
    }
}

impl Default for EnvelopeExtractor {
    fn default() -> Self {
        Self::new()
    }
}
