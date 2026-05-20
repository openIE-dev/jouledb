//! Visualization Codec
//!
//! High-level API for data embedding in images.

use super::steganography::{LSBDecoder, LSBEncoder, StegoImage};
use super::{InvertibleError, InvertibleResult};

/// Encoding mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingMode {
    /// Low capacity, high invisibility (1 bit/channel)
    Invisible,
    /// Balanced (2 bits/channel)
    Balanced,
    /// High capacity, lower invisibility (4 bits/channel)
    HighCapacity,
}

impl EncodingMode {
    fn bits_per_channel(&self) -> u8 {
        match self {
            EncodingMode::Invisible => 1,
            EncodingMode::Balanced => 2,
            EncodingMode::HighCapacity => 4,
        }
    }
}

/// VisCode: Visualization Codec
///
/// High-level API for embedding and extracting data from images.
#[derive(Clone)]
pub struct VisCode {
    mode: EncodingMode,
}

impl VisCode {
    /// Create new codec with invisible mode
    pub fn new() -> Self {
        Self {
            mode: EncodingMode::Invisible,
        }
    }

    /// Create codec with specific mode
    pub fn with_mode(mode: EncodingMode) -> Self {
        Self { mode }
    }

    /// Get encoding mode
    pub fn mode(&self) -> EncodingMode {
        self.mode
    }

    /// Calculate capacity for given image dimensions
    pub fn capacity(&self, width: u32, height: u32, channels: u8) -> usize {
        let usable_channels = if channels == 4 { 3 } else { channels as usize };
        let pixels = (width * height) as usize;
        let total_bits = pixels * usable_channels * self.mode.bits_per_channel() as usize;
        (total_bits / 8).saturating_sub(12) // Header size
    }

    /// Encode data into image bytes
    ///
    /// # Arguments
    /// * `data` - Data to embed
    /// * `image_data` - Raw image bytes (RGBA)
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    /// Modified image bytes with embedded data
    pub fn encode(
        &self,
        data: &[u8],
        image_data: &[u8],
        width: u32,
        height: u32,
    ) -> InvertibleResult<Vec<u8>> {
        let image = StegoImage::from_rgba(image_data.to_vec(), width, height);
        let encoder = LSBEncoder::with_bits(self.mode.bits_per_channel());
        let encoded = encoder.encode(data, &image)?;
        Ok(encoded.data)
    }

    /// Encode data into RGB image bytes
    pub fn encode_rgb(
        &self,
        data: &[u8],
        image_data: &[u8],
        width: u32,
        height: u32,
    ) -> InvertibleResult<Vec<u8>> {
        let image = StegoImage::from_rgb(image_data.to_vec(), width, height);
        let encoder = LSBEncoder::with_bits(self.mode.bits_per_channel());
        let encoded = encoder.encode(data, &image)?;
        Ok(encoded.data)
    }

    /// Decode data from image bytes
    ///
    /// # Arguments
    /// * `image_data` - Image bytes containing embedded data (RGBA)
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    /// Extracted data
    pub fn decode(&self, image_data: &[u8], width: u32, height: u32) -> InvertibleResult<Vec<u8>> {
        let image = StegoImage::from_rgba(image_data.to_vec(), width, height);
        let decoder = LSBDecoder::with_bits(self.mode.bits_per_channel());
        decoder.decode(&image)
    }

    /// Decode data from RGB image bytes
    pub fn decode_rgb(
        &self,
        image_data: &[u8],
        width: u32,
        height: u32,
    ) -> InvertibleResult<Vec<u8>> {
        let image = StegoImage::from_rgb(image_data.to_vec(), width, height);
        let decoder = LSBDecoder::with_bits(self.mode.bits_per_channel());
        decoder.decode(&image)
    }
}

impl Default for VisCode {
    fn default() -> Self {
        Self::new()
    }
}

/// InvVis: Invertible Visualization
///
/// Higher-level wrapper with image management.
pub struct InvVis {
    /// Codec for encoding/decoding
    codec: VisCode,
    /// Maximum data size for current configuration
    max_data_size: usize,
    /// Image dimensions
    image_size: (u32, u32),
}

impl InvVis {
    /// Create new InvVis for given image size
    pub fn new(width: u32, height: u32) -> Self {
        let codec = VisCode::new();
        let max_data_size = codec.capacity(width, height, 4);
        Self {
            codec,
            max_data_size,
            image_size: (width, height),
        }
    }

    /// Create with specific encoding mode
    pub fn with_mode(width: u32, height: u32, mode: EncodingMode) -> Self {
        let codec = VisCode::with_mode(mode);
        let max_data_size = codec.capacity(width, height, 4);
        Self {
            codec,
            max_data_size,
            image_size: (width, height),
        }
    }

    /// Get maximum data size
    pub fn max_data_size(&self) -> usize {
        self.max_data_size
    }

    /// Get image dimensions
    pub fn image_size(&self) -> (u32, u32) {
        self.image_size
    }

    /// Encode data into image
    pub fn encode(&self, data: &[u8], image_data: &[u8]) -> InvertibleResult<Vec<u8>> {
        if data.len() > self.max_data_size {
            return Err(InvertibleError::CapacityExceeded {
                data_size: data.len(),
                capacity: self.max_data_size,
            });
        }
        self.codec
            .encode(data, image_data, self.image_size.0, self.image_size.1)
    }

    /// Decode data from image
    pub fn decode(&self, image_data: &[u8]) -> InvertibleResult<Vec<u8>> {
        self.codec
            .decode(image_data, self.image_size.0, self.image_size.1)
    }

    /// Create a carrier image filled with a color
    pub fn create_carrier(&self, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let size = (self.image_size.0 * self.image_size.1 * 4) as usize;
        let mut data = Vec::with_capacity(size);
        for _ in 0..(self.image_size.0 * self.image_size.1) {
            data.push(r);
            data.push(g);
            data.push(b);
            data.push(a);
        }
        data
    }

    /// Create a gradient carrier image
    pub fn create_gradient_carrier(&self) -> Vec<u8> {
        let (w, h) = self.image_size;
        let mut data = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w) as u8;
                let g = ((y * 255) / h) as u8;
                let b = 128u8;
                data.push(r);
                data.push(g);
                data.push(b);
                data.push(255);
            }
        }
        data
    }
}

impl Clone for InvVis {
    fn clone(&self) -> Self {
        Self {
            codec: self.codec.clone(),
            max_data_size: self.max_data_size,
            image_size: self.image_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viscode_creation() {
        let codec = VisCode::new();
        assert_eq!(codec.mode(), EncodingMode::Invisible);
    }

    #[test]
    fn test_viscode_modes() {
        let invisible = VisCode::with_mode(EncodingMode::Invisible);
        let balanced = VisCode::with_mode(EncodingMode::Balanced);
        let high = VisCode::with_mode(EncodingMode::HighCapacity);

        let cap_inv = invisible.capacity(100, 100, 4);
        let cap_bal = balanced.capacity(100, 100, 4);
        let cap_high = high.capacity(100, 100, 4);

        assert!(cap_inv < cap_bal);
        assert!(cap_bal < cap_high);
    }

    #[test]
    fn test_viscode_encode_decode() {
        let codec = VisCode::new();
        let image = vec![128u8; 100 * 100 * 4];
        let data = b"Test message";

        let encoded = codec.encode(data, &image, 100, 100).unwrap();
        let decoded = codec.decode(&encoded, 100, 100).unwrap();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_viscode_balanced_mode() {
        let codec = VisCode::with_mode(EncodingMode::Balanced);
        let image = vec![128u8; 50 * 50 * 4];
        let data = b"Balanced mode test";

        let encoded = codec.encode(data, &image, 50, 50).unwrap();
        let decoded = codec.decode(&encoded, 50, 50).unwrap();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_invvis_creation() {
        let inv = InvVis::new(100, 100);
        assert_eq!(inv.image_size(), (100, 100));
        assert!(inv.max_data_size() > 0);
    }

    #[test]
    fn test_invvis_encode_decode() {
        let inv = InvVis::new(100, 100);
        let carrier = inv.create_carrier(128, 128, 128, 255);
        let data = b"Hello InvVis!";

        let encoded = inv.encode(data, &carrier).unwrap();
        let decoded = inv.decode(&encoded).unwrap();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_invvis_gradient_carrier() {
        let inv = InvVis::new(100, 100);
        let carrier = inv.create_gradient_carrier();
        let data = b"Gradient test";

        let encoded = inv.encode(data, &carrier).unwrap();
        let decoded = inv.decode(&encoded).unwrap();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_invvis_high_capacity_mode() {
        let inv = InvVis::with_mode(50, 50, EncodingMode::HighCapacity);
        let carrier = inv.create_carrier(255, 255, 255, 255);

        // Should have more capacity
        assert!(inv.max_data_size() > InvVis::new(50, 50).max_data_size());

        let data = b"High capacity data test with more content";
        let encoded = inv.encode(data, &carrier).unwrap();
        let decoded = inv.decode(&encoded).unwrap();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_invvis_capacity_exceeded() {
        let inv = InvVis::new(10, 10);
        let carrier = inv.create_carrier(0, 0, 0, 255);
        let data = vec![0u8; 1000]; // Too much

        let result = inv.encode(&data, &carrier);
        assert!(result.is_err());
    }

    #[test]
    fn test_viscode_rgb() {
        let codec = VisCode::new();
        let image = vec![128u8; 100 * 100 * 3];
        let data = b"RGB test";

        let encoded = codec.encode_rgb(data, &image, 100, 100).unwrap();
        let decoded = codec.decode_rgb(&encoded, 100, 100).unwrap();

        assert_eq!(decoded, data);
    }
}
