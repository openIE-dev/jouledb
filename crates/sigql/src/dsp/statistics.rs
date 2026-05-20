//! Statistical operations for signal analysis

use super::{DspError, DspResult};
use crate::types::UncertainValue;

/// Compute mean of a signal
pub fn compute_mean(signal: &[f64]) -> DspResult<UncertainValue<f64>> {
    if signal.is_empty() {
        return Err(DspError::SignalTooShort { needed: 1, got: 0 });
    }

    let n = signal.len();
    let mean = signal.iter().sum::<f64>() / n as f64;

    // Standard error of the mean
    let variance = signal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
    let std_error = (variance / n as f64).sqrt();

    Ok(UncertainValue::from_mean_se(mean, std_error, n))
}

/// Compute standard deviation
pub fn compute_std(signal: &[f64]) -> DspResult<UncertainValue<f64>> {
    if signal.len() < 2 {
        return Err(DspError::SignalTooShort {
            needed: 2,
            got: signal.len(),
        });
    }

    let n = signal.len();
    let mean = signal.iter().sum::<f64>() / n as f64;
    let variance = signal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let std = variance.sqrt();

    // Standard error of the standard deviation (approximate)
    let std_error = std / (2.0 * (n - 1) as f64).sqrt();

    Ok(UncertainValue::from_mean_se(std, std_error, n))
}

/// Compute variance
pub fn compute_variance(signal: &[f64]) -> DspResult<UncertainValue<f64>> {
    if signal.len() < 2 {
        return Err(DspError::SignalTooShort {
            needed: 2,
            got: signal.len(),
        });
    }

    let n = signal.len();
    let mean = signal.iter().sum::<f64>() / n as f64;
    let variance = signal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;

    // Standard error of variance
    let m4 = signal.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n as f64;
    let std_error = ((m4 - variance.powi(2)) / n as f64).sqrt();

    Ok(UncertainValue::from_mean_se(variance, std_error, n))
}

/// Compute root mean square
pub fn compute_rms(signal: &[f64]) -> DspResult<UncertainValue<f64>> {
    if signal.is_empty() {
        return Err(DspError::SignalTooShort { needed: 1, got: 0 });
    }

    let n = signal.len();
    let mean_sq = signal.iter().map(|x| x * x).sum::<f64>() / n as f64;
    let rms = mean_sq.sqrt();

    // Bootstrap or approximate error
    let variance_sq = signal
        .iter()
        .map(|x| (x * x - mean_sq).powi(2))
        .sum::<f64>()
        / (n - 1).max(1) as f64;
    let std_error = (variance_sq / n as f64).sqrt() / (2.0 * rms).max(1e-10);

    Ok(UncertainValue::from_mean_se(rms, std_error, n))
}

/// Compute kurtosis (excess kurtosis, 0 for Gaussian)
pub fn compute_kurtosis(signal: &[f64]) -> DspResult<UncertainValue<f64>> {
    if signal.len() < 4 {
        return Err(DspError::SignalTooShort {
            needed: 4,
            got: signal.len(),
        });
    }

    let n = signal.len();
    let mean = signal.iter().sum::<f64>() / n as f64;

    let m2 = signal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let m4 = signal.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n as f64;

    let kurtosis = m4 / (m2 * m2) - 3.0;

    // Standard error (approximate for normal distribution)
    let std_error = (24.0 / n as f64).sqrt();

    Ok(UncertainValue::from_mean_se(kurtosis, std_error, n))
}

/// Compute skewness
pub fn compute_skewness(signal: &[f64]) -> DspResult<UncertainValue<f64>> {
    if signal.len() < 3 {
        return Err(DspError::SignalTooShort {
            needed: 3,
            got: signal.len(),
        });
    }

    let n = signal.len();
    let mean = signal.iter().sum::<f64>() / n as f64;

    let m2 = signal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let m3 = signal.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n as f64;

    let skewness = m3 / m2.powf(1.5);

    // Standard error (approximate for normal distribution)
    let std_error = (6.0 / n as f64).sqrt();

    Ok(UncertainValue::from_mean_se(skewness, std_error, n))
}

/// Compute peak value (maximum)
pub fn compute_peak(signal: &[f64]) -> DspResult<f64> {
    signal
        .iter()
        .cloned()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .ok_or(DspError::SignalTooShort { needed: 1, got: 0 })
}

/// Compute trough value (minimum)
pub fn compute_trough(signal: &[f64]) -> DspResult<f64> {
    signal
        .iter()
        .cloned()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .ok_or(DspError::SignalTooShort { needed: 1, got: 0 })
}

/// Compute peak-to-peak amplitude
pub fn compute_peak_to_peak(signal: &[f64]) -> DspResult<f64> {
    let peak = compute_peak(signal)?;
    let trough = compute_trough(signal)?;
    Ok(peak - trough)
}

/// Count zero crossings
pub fn compute_zero_crossings(signal: &[f64]) -> usize {
    if signal.len() < 2 {
        return 0;
    }

    let mut count = 0;
    for i in 1..signal.len() {
        if (signal[i] >= 0.0) != (signal[i - 1] >= 0.0) {
            count += 1;
        }
    }
    count
}

/// Compute crest factor (peak / RMS)
pub fn compute_crest_factor(signal: &[f64]) -> DspResult<f64> {
    let peak = compute_peak(signal)?
        .abs()
        .max(compute_trough(signal)?.abs());
    let rms = compute_rms(signal)?.value;

    if rms < 1e-10 {
        return Err(DspError::NumericalError(
            "RMS too small for crest factor".to_string(),
        ));
    }

    Ok(peak / rms)
}

/// Compute percentile
pub fn compute_percentile(signal: &[f64], percentile: f64) -> DspResult<f64> {
    if signal.is_empty() {
        return Err(DspError::SignalTooShort { needed: 1, got: 0 });
    }

    if percentile < 0.0 || percentile > 100.0 {
        return Err(DspError::InvalidParameter(format!(
            "Percentile must be 0-100, got {}",
            percentile
        )));
    }

    let mut sorted = signal.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let idx = (percentile / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    Ok(sorted[idx])
}

/// Compute linear trend (slope)
pub fn compute_slope(signal: &[f64]) -> DspResult<UncertainValue<f64>> {
    if signal.len() < 2 {
        return Err(DspError::SignalTooShort {
            needed: 2,
            got: signal.len(),
        });
    }

    let n = signal.len() as f64;
    let x_mean = (n - 1.0) / 2.0;
    let y_mean = signal.iter().sum::<f64>() / n;

    let mut num = 0.0;
    let mut den = 0.0;

    for (i, &y) in signal.iter().enumerate() {
        let x = i as f64;
        num += (x - x_mean) * (y - y_mean);
        den += (x - x_mean).powi(2);
    }

    let slope = num / den;

    // Standard error of slope
    let y_pred: Vec<f64> = (0..signal.len())
        .map(|i| y_mean + slope * (i as f64 - x_mean))
        .collect();

    let residual_var: f64 = signal
        .iter()
        .zip(y_pred.iter())
        .map(|(y, yp)| (y - yp).powi(2))
        .sum::<f64>()
        / (n - 2.0).max(1.0);

    let std_error = (residual_var / den).sqrt();

    Ok(UncertainValue::from_mean_se(slope, std_error, signal.len()))
}

/// Remove linear trend from signal
pub fn detrend(signal: &[f64], order: usize) -> DspResult<Vec<f64>> {
    match order {
        0 => {
            // Remove mean
            let mean = signal.iter().sum::<f64>() / signal.len() as f64;
            Ok(signal.iter().map(|x| x - mean).collect())
        }
        1 => {
            // Remove linear trend
            let n = signal.len() as f64;
            let x_mean = (n - 1.0) / 2.0;
            let y_mean = signal.iter().sum::<f64>() / n;

            let mut num = 0.0;
            let mut den = 0.0;

            for (i, &y) in signal.iter().enumerate() {
                let x = i as f64;
                num += (x - x_mean) * (y - y_mean);
                den += (x - x_mean).powi(2);
            }

            let slope = num / den;
            let intercept = y_mean - slope * x_mean;

            Ok(signal
                .iter()
                .enumerate()
                .map(|(i, &y)| y - (intercept + slope * i as f64))
                .collect())
        }
        _ => {
            // Higher order: use polynomial fit (simplified)
            // For now, just do linear
            detrend(signal, 1)
        }
    }
}

/// Z-score normalization
pub fn zscore(signal: &[f64]) -> DspResult<Vec<f64>> {
    if signal.len() < 2 {
        return Err(DspError::SignalTooShort {
            needed: 2,
            got: signal.len(),
        });
    }

    let mean = signal.iter().sum::<f64>() / signal.len() as f64;
    let std =
        (signal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (signal.len() - 1) as f64).sqrt();

    if std < 1e-10 {
        return Ok(vec![0.0; signal.len()]);
    }

    Ok(signal.iter().map(|x| (x - mean) / std).collect())
}

/// Sample entropy (measure of complexity/regularity)
pub fn compute_sample_entropy(signal: &[f64], m: usize, r: f64) -> DspResult<f64> {
    if signal.len() < m + 2 {
        return Err(DspError::SignalTooShort {
            needed: m + 2,
            got: signal.len(),
        });
    }

    let n = signal.len();

    // Count matching templates of length m and m+1
    let mut a = 0; // matches of length m+1
    let mut b = 0; // matches of length m

    for i in 0..n - m {
        for j in (i + 1)..n - m {
            // Check match of length m
            let mut match_m = true;
            for k in 0..m {
                if (signal[i + k] - signal[j + k]).abs() > r {
                    match_m = false;
                    break;
                }
            }

            if match_m {
                b += 1;

                // Check if match extends to m+1
                if (signal[i + m] - signal[j + m]).abs() <= r {
                    a += 1;
                }
            }
        }
    }

    if b == 0 {
        return Ok(0.0);
    }

    Ok(-((a as f64) / (b as f64)).ln())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mean() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = compute_mean(&signal).unwrap();
        assert_relative_eq!(result.value, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_std() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = compute_std(&signal).unwrap();
        assert_relative_eq!(result.value, 1.5811, epsilon = 0.001);
    }

    #[test]
    fn test_rms() {
        let signal = vec![1.0, 1.0, 1.0, 1.0];
        let result = compute_rms(&signal).unwrap();
        assert_relative_eq!(result.value, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_zero_crossings() {
        let signal = vec![-1.0, 1.0, -1.0, 1.0];
        assert_eq!(compute_zero_crossings(&signal), 3);
    }

    #[test]
    fn test_detrend() {
        let signal: Vec<f64> = (0..100).map(|i| i as f64 * 0.1 + 5.0).collect();
        let detrended = detrend(&signal, 1).unwrap();

        // Mean of detrended should be ~0
        let mean: f64 = detrended.iter().sum::<f64>() / detrended.len() as f64;
        assert_relative_eq!(mean, 0.0, epsilon = 0.01);
    }

    #[test]
    fn test_zscore() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = zscore(&signal).unwrap();

        // Mean should be 0, std should be 1
        let mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
        assert_relative_eq!(mean, 0.0, epsilon = 1e-10);
    }
}
