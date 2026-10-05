//! no_std FFT stubs — rustfft requires std. Types exist so the crate typechecks;
//! compute methods return `DspError::InvalidParameter`.

use super::{DspError, DspResult};
use crate::types::SampleRate;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use num_complex::Complex64;

#[derive(Debug, Clone, Copy, Default)]
pub enum WindowType {
    Rectangular,
    #[default]
    Hann,
    Hamming,
    Blackman,
    BlackmanHarris,
    Kaiser { beta: f64 },
    FlatTop,
}

pub struct Fft {
    size: usize,
    pub window: WindowType,
}

impl Fft {
    pub fn new(size: usize) -> DspResult<Self> {
        if !size.is_power_of_two() {
            return Err(DspError::InvalidFftSize(size));
        }
        Ok(Self {
            size,
            window: WindowType::Hann,
        })
    }

    pub fn with_window(size: usize, window: WindowType) -> DspResult<Self> {
        let mut f = Self::new(size)?;
        f.window = window;
        Ok(f)
    }

    pub fn compute(&self, _signal: &[f64]) -> DspResult<Vec<Complex64>> {
        Err(DspError::InvalidParameter(format!(
            "FFT requires the std feature (rustfft); requested size {}",
            self.size
        )))
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

pub struct Ifft {
    size: usize,
}

impl Ifft {
    pub fn new(size: usize) -> DspResult<Self> {
        if !size.is_power_of_two() {
            return Err(DspError::InvalidFftSize(size));
        }
        Ok(Self { size })
    }

    pub fn compute(&self, _spectrum: &[Complex64]) -> DspResult<Vec<f64>> {
        Err(DspError::InvalidParameter(format!(
            "IFFT requires the std feature (rustfft); requested size {}",
            self.size
        )))
    }
}

#[derive(Debug, Clone)]
pub struct StftParams {
    pub fft_size: usize,
    pub hop_size: usize,
    pub window: WindowType,
}

impl Default for StftParams {
    fn default() -> Self {
        Self {
            fft_size: 1024,
            hop_size: 256,
            window: WindowType::Hann,
        }
    }
}

pub struct Stft {
    pub params: StftParams,
    fft: Fft,
}

impl Stft {
    pub fn new(params: StftParams) -> DspResult<Self> {
        let fft = Fft::with_window(params.fft_size, params.window)?;
        Ok(Self { params, fft })
    }

    pub fn compute(&self, _signal: &[f64]) -> DspResult<Vec<Vec<Complex64>>> {
        Err(DspError::InvalidParameter(String::from(
            "STFT requires the std feature (rustfft)",
        )))
    }
}

pub fn next_power_of_two(n: usize) -> usize {
    n.next_power_of_two().max(1)
}

pub fn frequency_for_bin(bin: usize, fft_size: usize, sample_rate: SampleRate) -> f64 {
    bin as f64 * sample_rate.as_u32() as f64 / fft_size as f64
}
