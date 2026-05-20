//! Temporal Encoding/Decoding
//!
//! Convert time-series data to/from spike trains.

use super::neuron::SpikeEvent;

/// Encoding type for converting values to spikes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingType {
    /// Rate coding - higher value = higher firing rate
    Rate,
    /// Temporal coding - value determines spike timing
    Temporal,
    /// Population coding - distribute across multiple neurons
    Population,
}

/// Decoding type for converting spikes to values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodingType {
    /// Rate decoding - count spikes
    Rate,
    /// Temporal decoding - use spike timing
    Temporal,
    /// Population decoding - combine from multiple neurons
    Population,
}

/// Temporal encoder - converts continuous values to spike trains
#[derive(Clone, Debug)]
pub struct TemporalEncoder {
    encoding_type: EncodingType,
    max_rate: f32, // Maximum firing rate in Hz
}

impl TemporalEncoder {
    /// Create new temporal encoder with default settings
    pub fn new() -> Self {
        Self {
            encoding_type: EncodingType::Rate,
            max_rate: 100.0,
        }
    }

    /// Create encoder with specific encoding type
    pub fn with_type(encoding_type: EncodingType) -> Self {
        Self {
            encoding_type,
            max_rate: 100.0,
        }
    }

    /// Set maximum firing rate
    pub fn with_max_rate(mut self, max_rate: f32) -> Self {
        self.max_rate = max_rate;
        self
    }

    /// Encode time-series data as spike train using rate coding
    ///
    /// Higher values produce more spikes.
    pub fn encode_rate(&self, data: &[f32], time_window: f32) -> Vec<SpikeEvent> {
        let mut spikes = Vec::new();
        let dt = time_window / data.len().max(1) as f32;

        for (i, &value) in data.iter().enumerate() {
            // Rate coding: higher value = higher firing rate
            let rate = value.max(0.0).min(1.0) * self.max_rate;
            if rate < 0.1 {
                continue;
            }
            let interval = 1.0 / rate;

            let mut time = i as f32 * dt;
            let end_time = (i + 1) as f32 * dt;

            while time < end_time {
                spikes.push(SpikeEvent::new(i, time, 1.0));
                time += interval;
            }
        }

        spikes
    }

    /// Encode with temporal coding - value determines spike timing
    pub fn encode_temporal(&self, data: &[f32], time_window: f32) -> Vec<SpikeEvent> {
        let mut spikes = Vec::new();
        let dt = time_window / data.len().max(1) as f32;

        for (i, &value) in data.iter().enumerate() {
            // Temporal coding: value determines spike time within window
            let normalized = value.max(0.0).min(1.0);
            let spike_time = i as f32 * dt + normalized * dt;
            spikes.push(SpikeEvent::new(i, spike_time, 1.0));
        }

        spikes
    }

    /// Encode with population coding - distribute across neurons
    pub fn encode_population(&self, value: f32, num_neurons: usize, time: f32) -> Vec<SpikeEvent> {
        let mut spikes = Vec::new();
        let normalized = value.max(0.0).min(1.0);

        // Each neuron has a preferred value
        for i in 0..num_neurons {
            let preferred = i as f32 / (num_neurons - 1).max(1) as f32;
            let response = (-((normalized - preferred).powi(2)) / 0.1).exp();

            if response > 0.5 {
                spikes.push(SpikeEvent::new(i, time, response));
            }
        }

        spikes
    }

    /// Get encoding type
    pub fn encoding_type(&self) -> EncodingType {
        self.encoding_type
    }
}

impl Default for TemporalEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Temporal decoder - converts spike trains to continuous values
#[derive(Clone, Debug)]
pub struct TemporalDecoder {
    decoding_type: DecodingType,
}

impl TemporalDecoder {
    /// Create new temporal decoder
    pub fn new() -> Self {
        Self {
            decoding_type: DecodingType::Rate,
        }
    }

    /// Create decoder with specific decoding type
    pub fn with_type(decoding_type: DecodingType) -> Self {
        Self { decoding_type }
    }

    /// Decode spike train to values using rate decoding
    ///
    /// Counts spikes in each time bin.
    pub fn decode_rate(
        &self,
        spikes: &[SpikeEvent],
        time_window: f32,
        num_bins: usize,
    ) -> Vec<f32> {
        if num_bins == 0 {
            return Vec::new();
        }

        let bin_size = time_window / num_bins as f32;
        let mut counts = vec![0u32; num_bins];

        for spike in spikes {
            let bin = (spike.time / bin_size) as usize;
            if bin < num_bins {
                counts[bin] += 1;
            }
        }

        // Normalize by max count
        let max_count = counts.iter().max().copied().unwrap_or(1).max(1);
        counts
            .iter()
            .map(|&c| c as f32 / max_count as f32)
            .collect()
    }

    /// Decode using spike timing
    pub fn decode_temporal(
        &self,
        spikes: &[SpikeEvent],
        time_window: f32,
        num_bins: usize,
    ) -> Vec<f32> {
        if num_bins == 0 {
            return Vec::new();
        }

        let bin_size = time_window / num_bins as f32;
        let mut values = vec![0.0f32; num_bins];
        let mut counts = vec![0u32; num_bins];

        for spike in spikes {
            let bin = (spike.time / bin_size) as usize;
            if bin < num_bins {
                // Value based on timing within bin
                let within_bin = (spike.time - bin as f32 * bin_size) / bin_size;
                values[bin] += within_bin;
                counts[bin] += 1;
            }
        }

        // Average timing within each bin
        values
            .iter()
            .zip(counts.iter())
            .map(|(&v, &c)| if c > 0 { v / c as f32 } else { 0.0 })
            .collect()
    }

    /// Decode population activity
    pub fn decode_population(&self, spikes: &[SpikeEvent], num_neurons: usize) -> f32 {
        if spikes.is_empty() || num_neurons == 0 {
            return 0.0;
        }

        // Weighted average based on neuron preferred values
        let mut weighted_sum = 0.0f32;
        let mut total_strength = 0.0f32;

        for spike in spikes {
            if spike.neuron_id < num_neurons {
                let preferred = spike.neuron_id as f32 / (num_neurons - 1).max(1) as f32;
                weighted_sum += preferred * spike.strength;
                total_strength += spike.strength;
            }
        }

        if total_strength > 0.0 {
            weighted_sum / total_strength
        } else {
            0.0
        }
    }

    /// Get decoding type
    pub fn decoding_type(&self) -> DecodingType {
        self.decoding_type
    }
}

impl Default for TemporalDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let encoder = TemporalEncoder::new();
        assert_eq!(encoder.encoding_type(), EncodingType::Rate);
    }

    #[test]
    fn test_rate_encoding() {
        let encoder = TemporalEncoder::new();
        let data = vec![0.5, 0.8, 0.2];

        let spikes = encoder.encode_rate(&data, 3.0);

        // Higher values should produce more spikes
        assert!(!spikes.is_empty());
    }

    #[test]
    fn test_temporal_encoding() {
        let encoder = TemporalEncoder::new();
        let data = vec![0.0, 0.5, 1.0];

        let spikes = encoder.encode_temporal(&data, 3.0);

        assert_eq!(spikes.len(), 3);
        // First spike should be earliest, last should be latest
        assert!(spikes[0].time < spikes[2].time);
    }

    #[test]
    fn test_population_encoding() {
        let encoder = TemporalEncoder::new();

        let spikes = encoder.encode_population(0.5, 10, 0.0);

        // Should activate neurons near preferred value 0.5
        assert!(!spikes.is_empty());
        // Middle neurons should be more active
        let has_middle = spikes.iter().any(|s| s.neuron_id == 5);
        assert!(has_middle);
    }

    #[test]
    fn test_decoder_creation() {
        let decoder = TemporalDecoder::new();
        assert_eq!(decoder.decoding_type(), DecodingType::Rate);
    }

    #[test]
    fn test_rate_decoding() {
        let decoder = TemporalDecoder::new();
        let spikes = vec![
            SpikeEvent::new(0, 0.1, 1.0),
            SpikeEvent::new(0, 0.2, 1.0),
            SpikeEvent::new(0, 1.1, 1.0),
        ];

        let values = decoder.decode_rate(&spikes, 2.0, 2);

        assert_eq!(values.len(), 2);
        // First bin has more spikes
        assert!(values[0] > values[1]);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let encoder = TemporalEncoder::new();
        let decoder = TemporalDecoder::new();

        let original = vec![0.8, 0.2, 0.5];
        let spikes = encoder.encode_rate(&original, 3.0);
        let decoded = decoder.decode_rate(&spikes, 3.0, 3);

        // Decoded should preserve relative ordering
        assert_eq!(decoded.len(), 3);
    }

    #[test]
    fn test_population_decode() {
        let decoder = TemporalDecoder::new();

        // Spikes from middle neurons
        let spikes = vec![
            SpikeEvent::new(4, 0.0, 1.0),
            SpikeEvent::new(5, 0.0, 1.0),
            SpikeEvent::new(6, 0.0, 1.0),
        ];

        let value = decoder.decode_population(&spikes, 10);

        // Should decode to approximately 0.5 (middle of 10 neurons)
        assert!((value - 0.5).abs() < 0.2);
    }
}
