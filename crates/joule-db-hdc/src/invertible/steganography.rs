//! Steganography
//!
//! LSB (Least Significant Bit) encoding for hiding data in images.

use super::{InvertibleError, InvertibleResult};

/// Header size in bytes (magic + length + checksum)
const HEADER_SIZE: usize = 12;
/// Magic bytes to identify embedded data
const MAGIC: [u8; 4] = [0x49, 0x4E, 0x56, 0x56]; // "INVV"

/// Image wrapper for steganography
#[derive(Debug, Clone)]
pub struct StegoImage {
    /// Raw pixel data (RGBA format expected)
    pub data: Vec<u8>,
    /// Image width
    pub width: u32,
    /// Image height  
    pub height: u32,
    /// Bytes per pixel (3 for RGB, 4 for RGBA)
    pub channels: u8,
}

impl StegoImage {
    /// Create new stego image from raw data
    pub fn new(data: Vec<u8>, width: u32, height: u32, channels: u8) -> Self {
        Self {
            data,
            width,
            height,
            channels,
        }
    }

    /// Create from raw RGBA data
    pub fn from_rgba(data: Vec<u8>, width: u32, height: u32) -> Self {
        Self::new(data, width, height, 4)
    }

    /// Create from raw RGB data
    pub fn from_rgb(data: Vec<u8>, width: u32, height: u32) -> Self {
        Self::new(data, width, height, 3)
    }

    /// Get capacity in bytes (how much data can be embedded)
    pub fn capacity(&self) -> usize {
        // Each pixel channel can hold 1 bit
        // We use RGB channels (not alpha for better invisibility)
        let usable_channels = if self.channels == 4 {
            3
        } else {
            self.channels as usize
        };
        let total_bits = (self.data.len() / self.channels as usize) * usable_channels;
        // Subtract header size
        (total_bits / 8).saturating_sub(HEADER_SIZE)
    }

    /// Get dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// LSB Encoder for embedding data
pub struct LSBEncoder {
    /// Number of LSB bits to use (1-4)
    bits_per_channel: u8,
}

impl LSBEncoder {
    /// Create new encoder with 1 bit per channel (most invisible)
    pub fn new() -> Self {
        Self {
            bits_per_channel: 1,
        }
    }

    /// Create encoder with custom bits per channel
    pub fn with_bits(bits: u8) -> Self {
        Self {
            bits_per_channel: bits.clamp(1, 4),
        }
    }

    /// Get effective capacity for an image
    pub fn capacity(&self, image: &StegoImage) -> usize {
        let usable_channels = if image.channels == 4 {
            3
        } else {
            image.channels as usize
        };
        let pixels = image.data.len() / image.channels as usize;
        let total_bits = pixels * usable_channels * self.bits_per_channel as usize;
        (total_bits / 8).saturating_sub(HEADER_SIZE)
    }

    /// Encode data into image
    pub fn encode(&self, data: &[u8], image: &StegoImage) -> InvertibleResult<StegoImage> {
        let capacity = self.capacity(image);
        if data.len() > capacity {
            return Err(InvertibleError::CapacityExceeded {
                data_size: data.len(),
                capacity,
            });
        }

        // Build header
        let checksum = Self::compute_checksum(data);
        let mut header = Vec::with_capacity(HEADER_SIZE);
        header.extend_from_slice(&MAGIC);
        header.extend_from_slice(&(data.len() as u32).to_le_bytes());
        header.extend_from_slice(&checksum.to_le_bytes());

        // Combine header and data
        let mut payload = header;
        payload.extend_from_slice(data);

        // Convert to bits
        let bits = Self::bytes_to_bits(&payload);

        // Embed bits into image
        let mut result = image.data.clone();
        self.embed_bits(&bits, &mut result, image.channels);

        Ok(StegoImage {
            data: result,
            width: image.width,
            height: image.height,
            channels: image.channels,
        })
    }

    /// Embed bits into image data
    fn embed_bits(&self, bits: &[bool], image_data: &mut [u8], channels: u8) {
        let mask = !(1u8 << self.bits_per_channel) + 1; // Clear lower bits
        let mut bit_idx = 0;

        for (i, byte) in image_data.iter_mut().enumerate() {
            // Skip alpha channel
            if channels == 4 && i % 4 == 3 {
                continue;
            }

            if bit_idx >= bits.len() {
                break;
            }

            // Embed bits_per_channel bits
            let mut value = 0u8;
            for b in 0..self.bits_per_channel {
                if bit_idx < bits.len() && bits[bit_idx] {
                    value |= 1 << b;
                }
                bit_idx += 1;
            }

            *byte = (*byte & mask) | value;
        }
    }

    /// Convert bytes to bits
    fn bytes_to_bits(data: &[u8]) -> Vec<bool> {
        let mut bits = Vec::with_capacity(data.len() * 8);
        for byte in data {
            for i in 0..8 {
                bits.push((byte >> i) & 1 == 1);
            }
        }
        bits
    }

    /// Compute CRC32 checksum
    fn compute_checksum(data: &[u8]) -> u32 {
        // Simple checksum (not cryptographic)
        let mut sum: u32 = 0;
        for (i, &byte) in data.iter().enumerate() {
            sum = sum.wrapping_add((byte as u32).wrapping_mul((i as u32).wrapping_add(1)));
            sum = sum.rotate_left(5);
        }
        sum
    }
}

impl Default for LSBEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// LSB Decoder for extracting data
pub struct LSBDecoder {
    /// Number of LSB bits to use
    bits_per_channel: u8,
}

impl LSBDecoder {
    /// Create new decoder with 1 bit per channel
    pub fn new() -> Self {
        Self {
            bits_per_channel: 1,
        }
    }

    /// Create decoder with custom bits per channel
    pub fn with_bits(bits: u8) -> Self {
        Self {
            bits_per_channel: bits.clamp(1, 4),
        }
    }

    /// Decode data from image
    pub fn decode(&self, image: &StegoImage) -> InvertibleResult<Vec<u8>> {
        // Extract bits from image
        let bits = self.extract_bits(&image.data, image.channels);

        // Convert to bytes
        let bytes = Self::bits_to_bytes(&bits);

        if bytes.len() < HEADER_SIZE {
            return Err(InvertibleError::NoDataFound);
        }

        // Parse header
        let magic = &bytes[0..4];
        if magic != MAGIC {
            return Err(InvertibleError::NoDataFound);
        }

        let length = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let stored_checksum = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);

        if bytes.len() < HEADER_SIZE + length {
            return Err(InvertibleError::DecodingError(format!(
                "incomplete data: expected {} bytes, got {}",
                length,
                bytes.len() - HEADER_SIZE
            )));
        }

        let data = &bytes[HEADER_SIZE..HEADER_SIZE + length];

        // Verify checksum
        let computed_checksum = LSBEncoder::compute_checksum(data);
        if computed_checksum != stored_checksum {
            return Err(InvertibleError::ChecksumMismatch);
        }

        Ok(data.to_vec())
    }

    /// Extract bits from image data
    fn extract_bits(&self, image_data: &[u8], channels: u8) -> Vec<bool> {
        let mut bits = Vec::new();
        let mask = (1u8 << self.bits_per_channel) - 1;

        for (i, &byte) in image_data.iter().enumerate() {
            // Skip alpha channel
            if channels == 4 && i % 4 == 3 {
                continue;
            }

            let value = byte & mask;
            for b in 0..self.bits_per_channel {
                bits.push((value >> b) & 1 == 1);
            }
        }

        bits
    }

    /// Convert bits to bytes
    fn bits_to_bytes(bits: &[bool]) -> Vec<u8> {
        bits.chunks(8)
            .map(|chunk| {
                let mut byte = 0u8;
                for (i, &bit) in chunk.iter().enumerate() {
                    if bit {
                        byte |= 1 << i;
                    }
                }
                byte
            })
            .collect()
    }
}

impl Default for LSBDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_image(width: u32, height: u32) -> StegoImage {
        let size = (width * height * 4) as usize;
        let data = vec![128u8; size]; // Mid-gray RGBA
        StegoImage::from_rgba(data, width, height)
    }

    #[test]
    fn test_stego_image_capacity() {
        let image = create_test_image(100, 100);
        let capacity = image.capacity();
        // 100x100 = 10000 pixels, 3 channels usable = 30000 bits = 3750 bytes - header
        assert!(capacity > 3700);
    }

    #[test]
    fn test_encode_decode_small() {
        let image = create_test_image(100, 100);
        let encoder = LSBEncoder::new();
        let decoder = LSBDecoder::new();

        let data = b"Hello, World!";
        let encoded = encoder.encode(data, &image).unwrap();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_encode_decode_large() {
        let image = create_test_image(200, 200);
        let encoder = LSBEncoder::new();
        let decoder = LSBDecoder::new();

        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let encoded = encoder.encode(&data, &image).unwrap();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_capacity_exceeded() {
        let image = create_test_image(10, 10);
        let encoder = LSBEncoder::new();

        let data = vec![0u8; 1000]; // Too much data
        let result = encoder.encode(&data, &image);

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(InvertibleError::CapacityExceeded { .. })
        ));
    }

    #[test]
    fn test_no_data_found() {
        let image = create_test_image(100, 100);
        let decoder = LSBDecoder::new();

        let result = decoder.decode(&image);
        assert!(result.is_err());
        assert!(matches!(result, Err(InvertibleError::NoDataFound)));
    }

    #[test]
    fn test_multi_bit_encoding() {
        let image = create_test_image(50, 50);
        let encoder = LSBEncoder::with_bits(2);
        let decoder = LSBDecoder::with_bits(2);

        let data = b"Multi-bit encoding test!";
        let encoded = encoder.encode(data, &image).unwrap();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_checksum_validation() {
        let image = create_test_image(100, 100);
        let encoder = LSBEncoder::new();

        let data = b"Test data";
        let mut encoded = encoder.encode(data, &image).unwrap();

        // Corrupt the data
        if encoded.data.len() > 100 {
            encoded.data[100] ^= 0xFF;
        }

        let decoder = LSBDecoder::new();
        let result = decoder.decode(&encoded);

        // Should either fail checksum or produce wrong data
        assert!(result.is_err() || result.unwrap() != data);
    }

    #[test]
    fn test_rgb_image() {
        let size = (100 * 100 * 3) as usize;
        let data = vec![128u8; size];
        let image = StegoImage::from_rgb(data, 100, 100);

        let encoder = LSBEncoder::new();
        let decoder = LSBDecoder::new();

        let payload = b"RGB test";
        let encoded = encoder.encode(payload, &image).unwrap();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, payload);
    }

    #[test]
    fn test_empty_data() {
        let image = create_test_image(100, 100);
        let encoder = LSBEncoder::new();
        let decoder = LSBDecoder::new();

        let data = b"";
        let encoded = encoder.encode(data, &image).unwrap();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_binary_data() {
        let image = create_test_image(100, 100);
        let encoder = LSBEncoder::new();
        let decoder = LSBDecoder::new();

        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let encoded = encoder.encode(&data, &image).unwrap();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, data);
    }
}
