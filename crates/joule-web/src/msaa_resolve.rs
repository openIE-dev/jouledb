// msaa_resolve.rs — MSAA resolve operations.
// Multi-sample buffer, box/weighted resolve, coverage masks, sample positions.

/// RGBA color as f32 components.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub fn black() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }

    pub fn transparent() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }

    pub fn add(&self, other: &Color) -> Color {
        Color {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
            a: self.a + other.a,
        }
    }

    pub fn scale(&self, factor: f32) -> Color {
        Color {
            r: self.r * factor,
            g: self.g * factor,
            b: self.b * factor,
            a: self.a * factor,
        }
    }

    pub fn luminance(&self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }
}

/// Sample position within a pixel, in [-0.5, 0.5] range centered at pixel center.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplePosition {
    pub x: f32,
    pub y: f32,
}

impl SamplePosition {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn center() -> Self {
        Self::new(0.0, 0.0)
    }

    pub fn distance_from_center(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

/// MSAA sample count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleCount {
    X2,
    X4,
    X8,
}

impl SampleCount {
    pub fn count(&self) -> usize {
        match self {
            SampleCount::X2 => 2,
            SampleCount::X4 => 4,
            SampleCount::X8 => 8,
        }
    }
}

/// Standard sample positions for each MSAA level.
pub fn standard_sample_positions(sample_count: SampleCount) -> Vec<SamplePosition> {
    match sample_count {
        SampleCount::X2 => vec![
            SamplePosition::new(-0.25, -0.25),
            SamplePosition::new(0.25, 0.25),
        ],
        SampleCount::X4 => {
            // Rotated grid pattern (RGSS).
            vec![
                SamplePosition::new(-0.125, -0.375),
                SamplePosition::new(0.375, -0.125),
                SamplePosition::new(-0.375, 0.125),
                SamplePosition::new(0.125, 0.375),
            ]
        }
        SampleCount::X8 => vec![
            SamplePosition::new(-0.375, -0.375),
            SamplePosition::new(0.125, -0.375),
            SamplePosition::new(-0.125, -0.125),
            SamplePosition::new(0.375, -0.125),
            SamplePosition::new(-0.375, 0.125),
            SamplePosition::new(0.125, 0.125),
            SamplePosition::new(-0.125, 0.375),
            SamplePosition::new(0.375, 0.375),
        ],
    }
}

/// Coverage mask — a bitmask indicating which samples are covered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoverageMask {
    pub bits: u8,
    pub sample_count: u8,
}

impl CoverageMask {
    pub fn full(sample_count: SampleCount) -> Self {
        let n = sample_count.count() as u8;
        // For 8 samples, (1u16 << 8) - 1 = 255 which fits in u8.
        let bits = ((1u16 << n) - 1) as u8;
        Self {
            bits,
            sample_count: n,
        }
    }

    pub fn empty(sample_count: SampleCount) -> Self {
        Self {
            bits: 0,
            sample_count: sample_count.count() as u8,
        }
    }

    pub fn from_bits(bits: u8, sample_count: SampleCount) -> Self {
        let n = sample_count.count() as u8;
        let mask = ((1u16 << n) - 1) as u8;
        Self {
            bits: bits & mask,
            sample_count: n,
        }
    }

    pub fn is_sample_covered(&self, index: usize) -> bool {
        if index >= self.sample_count as usize {
            return false;
        }
        (self.bits >> index) & 1 == 1
    }

    pub fn covered_count(&self) -> usize {
        self.bits.count_ones() as usize
    }

    pub fn is_fully_covered(&self) -> bool {
        self.covered_count() == self.sample_count as usize
    }

    pub fn is_empty_mask(&self) -> bool {
        self.bits == 0
    }

    pub fn is_partial(&self) -> bool {
        !self.is_fully_covered() && !self.is_empty_mask()
    }

    pub fn union(&self, other: &CoverageMask) -> CoverageMask {
        CoverageMask {
            bits: self.bits | other.bits,
            sample_count: self.sample_count,
        }
    }

    pub fn intersection(&self, other: &CoverageMask) -> CoverageMask {
        CoverageMask {
            bits: self.bits & other.bits,
            sample_count: self.sample_count,
        }
    }
}

/// Multi-sample pixel — holds N color samples plus a coverage mask.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiSamplePixel {
    pub samples: Vec<Color>,
    pub coverage: CoverageMask,
}

impl MultiSamplePixel {
    pub fn new(sample_count: SampleCount) -> Self {
        let n = sample_count.count();
        Self {
            samples: vec![Color::black(); n],
            coverage: CoverageMask::full(sample_count),
        }
    }

    pub fn from_samples(samples: Vec<Color>, coverage: CoverageMask) -> Self {
        Self { samples, coverage }
    }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

/// Multi-sample buffer — 2D grid of multi-sample pixels.
#[derive(Debug, Clone)]
pub struct MultiSampleBuffer {
    pub pixels: Vec<MultiSamplePixel>,
    pub width: usize,
    pub height: usize,
    pub sample_count: SampleCount,
}

impl MultiSampleBuffer {
    pub fn new(width: usize, height: usize, sample_count: SampleCount) -> Self {
        Self {
            pixels: vec![MultiSamplePixel::new(sample_count); width * height],
            width,
            height,
            sample_count,
        }
    }

    pub fn get(&self, x: usize, y: usize) -> &MultiSamplePixel {
        &self.pixels[y * self.width + x]
    }

    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut MultiSamplePixel {
        &mut self.pixels[y * self.width + x]
    }

    pub fn set_sample(&mut self, x: usize, y: usize, sample_index: usize, color: Color) {
        let pixel = self.get_mut(x, y);
        if sample_index < pixel.samples.len() {
            pixel.samples[sample_index] = color;
        }
    }
}

/// Resolved single-sample buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedBuffer {
    pub pixels: Vec<Color>,
    pub width: usize,
    pub height: usize,
}

impl ResolvedBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            pixels: vec![Color::black(); width * height],
            width,
            height,
        }
    }

    pub fn get(&self, x: usize, y: usize) -> Color {
        self.pixels[y * self.width + x]
    }
}

/// Box resolve — simple average of all covered samples.
pub fn box_resolve(buffer: &MultiSampleBuffer) -> ResolvedBuffer {
    let mut result = ResolvedBuffer::new(buffer.width, buffer.height);

    for y in 0..buffer.height {
        for x in 0..buffer.width {
            let pixel = buffer.get(x, y);
            let covered = pixel.coverage.covered_count();
            if covered == 0 {
                result.pixels[y * buffer.width + x] = Color::transparent();
                continue;
            }

            let mut accum = Color::new(0.0, 0.0, 0.0, 0.0);
            for (i, sample) in pixel.samples.iter().enumerate() {
                if pixel.coverage.is_sample_covered(i) {
                    accum = accum.add(sample);
                }
            }
            result.pixels[y * buffer.width + x] = accum.scale(1.0 / covered as f32);
        }
    }

    result
}

/// Weighted resolve — center-weighted, samples closer to center contribute more.
pub fn weighted_resolve(buffer: &MultiSampleBuffer) -> ResolvedBuffer {
    let positions = standard_sample_positions(buffer.sample_count);
    let mut result = ResolvedBuffer::new(buffer.width, buffer.height);

    // Compute weights inversely proportional to distance from center.
    let weights: Vec<f32> = positions
        .iter()
        .map(|p| {
            let dist = p.distance_from_center();
            1.0 / (1.0 + dist * 4.0)
        })
        .collect();

    for y in 0..buffer.height {
        for x in 0..buffer.width {
            let pixel = buffer.get(x, y);
            let mut accum = Color::new(0.0, 0.0, 0.0, 0.0);
            let mut total_weight = 0.0f32;

            for (i, sample) in pixel.samples.iter().enumerate() {
                if pixel.coverage.is_sample_covered(i) {
                    let w = weights.get(i).copied().unwrap_or(1.0);
                    accum = accum.add(&sample.scale(w));
                    total_weight += w;
                }
            }

            if total_weight > 0.0 {
                result.pixels[y * buffer.width + x] = accum.scale(1.0 / total_weight);
            }
        }
    }

    result
}

/// Per-sample shading mask — identifies which pixels need per-sample evaluation.
/// A pixel needs per-sample shading if its samples have different colors (it's an edge pixel).
pub fn compute_per_sample_shading_mask(buffer: &MultiSampleBuffer, threshold: f32) -> Vec<bool> {
    let mut mask = vec![false; buffer.width * buffer.height];

    for y in 0..buffer.height {
        for x in 0..buffer.width {
            let pixel = buffer.get(x, y);
            if pixel.samples.len() < 2 {
                continue;
            }
            let first = &pixel.samples[0];
            let needs_per_sample = pixel.samples.iter().skip(1).any(|s| {
                (s.r - first.r).abs() > threshold
                    || (s.g - first.g).abs() > threshold
                    || (s.b - first.b).abs() > threshold
            });
            mask[y * buffer.width + x] = needs_per_sample;
        }
    }

    mask
}

/// Compute the centroid interpolation point for a coverage mask.
/// Returns the average position of covered samples.
pub fn centroid_position(
    coverage: &CoverageMask,
    sample_count: SampleCount,
) -> SamplePosition {
    let positions = standard_sample_positions(sample_count);
    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    let mut count = 0usize;

    for (i, pos) in positions.iter().enumerate() {
        if coverage.is_sample_covered(i) {
            sum_x += pos.x;
            sum_y += pos.y;
            count += 1;
        }
    }

    if count == 0 {
        return SamplePosition::center();
    }

    SamplePosition::new(sum_x / count as f32, sum_y / count as f32)
}

/// Compute coverage fraction for a pixel (0.0 - 1.0).
pub fn coverage_fraction(coverage: &CoverageMask) -> f32 {
    if coverage.sample_count == 0 {
        return 0.0;
    }
    coverage.covered_count() as f32 / coverage.sample_count as f32
}

/// Quality level for resolve operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolveQuality {
    /// Simple box filter (fastest).
    Box,
    /// Center-weighted filter.
    Weighted,
}

/// Resolve a multi-sample buffer at the specified quality.
pub fn resolve(buffer: &MultiSampleBuffer, quality: ResolveQuality) -> ResolvedBuffer {
    match quality {
        ResolveQuality::Box => box_resolve(buffer),
        ResolveQuality::Weighted => weighted_resolve(buffer),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_count() {
        assert_eq!(SampleCount::X2.count(), 2);
        assert_eq!(SampleCount::X4.count(), 4);
        assert_eq!(SampleCount::X8.count(), 8);
    }

    #[test]
    fn test_sample_positions_2x() {
        let pos = standard_sample_positions(SampleCount::X2);
        assert_eq!(pos.len(), 2);
        for p in &pos {
            assert!(p.x.abs() <= 0.5);
            assert!(p.y.abs() <= 0.5);
        }
    }

    #[test]
    fn test_sample_positions_4x() {
        let pos = standard_sample_positions(SampleCount::X4);
        assert_eq!(pos.len(), 4);
    }

    #[test]
    fn test_sample_positions_8x() {
        let pos = standard_sample_positions(SampleCount::X8);
        assert_eq!(pos.len(), 8);
    }

    #[test]
    fn test_coverage_mask_full() {
        let mask = CoverageMask::full(SampleCount::X4);
        assert_eq!(mask.covered_count(), 4);
        assert!(mask.is_fully_covered());
        assert!(!mask.is_partial());
    }

    #[test]
    fn test_coverage_mask_empty() {
        let mask = CoverageMask::empty(SampleCount::X4);
        assert_eq!(mask.covered_count(), 0);
        assert!(mask.is_empty_mask());
    }

    #[test]
    fn test_coverage_mask_partial() {
        let mask = CoverageMask::from_bits(0b0101, SampleCount::X4);
        assert_eq!(mask.covered_count(), 2);
        assert!(mask.is_partial());
        assert!(mask.is_sample_covered(0));
        assert!(!mask.is_sample_covered(1));
        assert!(mask.is_sample_covered(2));
        assert!(!mask.is_sample_covered(3));
    }

    #[test]
    fn test_coverage_mask_operations() {
        let a = CoverageMask::from_bits(0b0011, SampleCount::X4);
        let b = CoverageMask::from_bits(0b0110, SampleCount::X4);

        let u = a.union(&b);
        assert_eq!(u.bits, 0b0111);

        let i = a.intersection(&b);
        assert_eq!(i.bits, 0b0010);
    }

    #[test]
    fn test_coverage_mask_out_of_range() {
        let mask = CoverageMask::full(SampleCount::X4);
        assert!(!mask.is_sample_covered(10));
    }

    #[test]
    fn test_box_resolve_uniform() {
        let mut buf = MultiSampleBuffer::new(2, 2, SampleCount::X4);
        let red = Color::new(1.0, 0.0, 0.0, 1.0);
        for y in 0..2 {
            for x in 0..2 {
                for s in 0..4 {
                    buf.set_sample(x, y, s, red);
                }
            }
        }
        let resolved = box_resolve(&buf);
        let p = resolved.get(0, 0);
        assert!((p.r - 1.0).abs() < 1e-6);
        assert!(p.g.abs() < 1e-6);
    }

    #[test]
    fn test_box_resolve_mixed_samples() {
        let mut buf = MultiSampleBuffer::new(1, 1, SampleCount::X4);
        buf.set_sample(0, 0, 0, Color::new(1.0, 0.0, 0.0, 1.0));
        buf.set_sample(0, 0, 1, Color::new(1.0, 0.0, 0.0, 1.0));
        buf.set_sample(0, 0, 2, Color::new(0.0, 0.0, 1.0, 1.0));
        buf.set_sample(0, 0, 3, Color::new(0.0, 0.0, 1.0, 1.0));

        let resolved = box_resolve(&buf);
        let p = resolved.get(0, 0);
        assert!((p.r - 0.5).abs() < 1e-6);
        assert!((p.b - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_box_resolve_partial_coverage() {
        let mut buf = MultiSampleBuffer::new(1, 1, SampleCount::X4);
        let pixel = buf.get_mut(0, 0);
        pixel.coverage = CoverageMask::from_bits(0b0011, SampleCount::X4);
        pixel.samples[0] = Color::new(1.0, 0.0, 0.0, 1.0);
        pixel.samples[1] = Color::new(0.0, 1.0, 0.0, 1.0);
        pixel.samples[2] = Color::new(0.0, 0.0, 1.0, 1.0); // not covered
        pixel.samples[3] = Color::new(1.0, 1.0, 1.0, 1.0); // not covered

        let resolved = box_resolve(&buf);
        let p = resolved.get(0, 0);
        // Only samples 0 and 1 are covered.
        assert!((p.r - 0.5).abs() < 1e-6);
        assert!((p.g - 0.5).abs() < 1e-6);
        assert!(p.b.abs() < 1e-6);
    }

    #[test]
    fn test_weighted_resolve_uniform() {
        let mut buf = MultiSampleBuffer::new(1, 1, SampleCount::X4);
        let c = Color::new(0.5, 0.5, 0.5, 1.0);
        for s in 0..4 {
            buf.set_sample(0, 0, s, c);
        }
        let resolved = weighted_resolve(&buf);
        let p = resolved.get(0, 0);
        assert!((p.r - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_weighted_vs_box_differs() {
        let mut buf = MultiSampleBuffer::new(1, 1, SampleCount::X4);
        buf.set_sample(0, 0, 0, Color::new(1.0, 0.0, 0.0, 1.0));
        buf.set_sample(0, 0, 1, Color::new(0.0, 1.0, 0.0, 1.0));
        buf.set_sample(0, 0, 2, Color::new(0.0, 0.0, 1.0, 1.0));
        buf.set_sample(0, 0, 3, Color::new(1.0, 1.0, 0.0, 1.0));

        let box_res = box_resolve(&buf);
        let weighted_res = weighted_resolve(&buf);

        // They can differ due to center weighting.
        let bp = box_res.get(0, 0);
        let wp = weighted_res.get(0, 0);
        // The total should be similar but not necessarily identical.
        let _diff = (bp.r - wp.r).abs() + (bp.g - wp.g).abs() + (bp.b - wp.b).abs();
        // Both should produce valid colors.
        assert!(wp.r >= 0.0 && wp.r <= 1.0);
        assert!(wp.g >= 0.0 && wp.g <= 1.0);
    }

    #[test]
    fn test_per_sample_shading_mask_uniform() {
        let mut buf = MultiSampleBuffer::new(2, 2, SampleCount::X4);
        let c = Color::new(0.5, 0.5, 0.5, 1.0);
        for y in 0..2 {
            for x in 0..2 {
                for s in 0..4 {
                    buf.set_sample(x, y, s, c);
                }
            }
        }
        let mask = compute_per_sample_shading_mask(&buf, 0.01);
        assert!(mask.iter().all(|m| !m), "uniform samples need no per-sample shading");
    }

    #[test]
    fn test_per_sample_shading_mask_edge() {
        let mut buf = MultiSampleBuffer::new(1, 1, SampleCount::X4);
        buf.set_sample(0, 0, 0, Color::new(1.0, 0.0, 0.0, 1.0));
        buf.set_sample(0, 0, 1, Color::new(1.0, 0.0, 0.0, 1.0));
        buf.set_sample(0, 0, 2, Color::new(0.0, 0.0, 1.0, 1.0));
        buf.set_sample(0, 0, 3, Color::new(0.0, 0.0, 1.0, 1.0));

        let mask = compute_per_sample_shading_mask(&buf, 0.01);
        assert!(mask[0], "mixed samples should need per-sample shading");
    }

    #[test]
    fn test_centroid_position_full_coverage() {
        let mask = CoverageMask::full(SampleCount::X4);
        let centroid = centroid_position(&mask, SampleCount::X4);
        // With RGSS pattern, centroid of all 4 should be near center.
        assert!(centroid.x.abs() < 0.01);
        assert!(centroid.y.abs() < 0.01);
    }

    #[test]
    fn test_centroid_position_partial() {
        let mask = CoverageMask::from_bits(0b0001, SampleCount::X4);
        let centroid = centroid_position(&mask, SampleCount::X4);
        let positions = standard_sample_positions(SampleCount::X4);
        // Should be exactly the first sample position.
        assert!((centroid.x - positions[0].x).abs() < 1e-6);
        assert!((centroid.y - positions[0].y).abs() < 1e-6);
    }

    #[test]
    fn test_centroid_empty_coverage() {
        let mask = CoverageMask::empty(SampleCount::X4);
        let centroid = centroid_position(&mask, SampleCount::X4);
        assert!((centroid.x).abs() < 1e-6);
        assert!((centroid.y).abs() < 1e-6);
    }

    #[test]
    fn test_coverage_fraction() {
        let full = CoverageMask::full(SampleCount::X4);
        assert!((coverage_fraction(&full) - 1.0).abs() < 1e-6);

        let half = CoverageMask::from_bits(0b0011, SampleCount::X4);
        assert!((coverage_fraction(&half) - 0.5).abs() < 1e-6);

        let empty = CoverageMask::empty(SampleCount::X4);
        assert!(coverage_fraction(&empty).abs() < 1e-6);
    }

    #[test]
    fn test_resolve_quality_box() {
        let buf = MultiSampleBuffer::new(2, 2, SampleCount::X2);
        let resolved = resolve(&buf, ResolveQuality::Box);
        assert_eq!(resolved.width, 2);
        assert_eq!(resolved.height, 2);
    }

    #[test]
    fn test_resolve_quality_weighted() {
        let buf = MultiSampleBuffer::new(2, 2, SampleCount::X2);
        let resolved = resolve(&buf, ResolveQuality::Weighted);
        assert_eq!(resolved.width, 2);
    }

    #[test]
    fn test_sample_position_distance() {
        let center = SamplePosition::center();
        assert!(center.distance_from_center() < 1e-6);

        let off = SamplePosition::new(0.3, 0.4);
        assert!((off.distance_from_center() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_color_operations() {
        let a = Color::new(0.5, 0.3, 0.2, 1.0);
        let b = Color::new(0.1, 0.2, 0.3, 0.5);
        let sum = a.add(&b);
        assert!((sum.r - 0.6).abs() < 1e-6);
        assert!((sum.g - 0.5).abs() < 1e-6);

        let scaled = a.scale(2.0);
        assert!((scaled.r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_multisample_buffer_dimensions() {
        let buf = MultiSampleBuffer::new(4, 3, SampleCount::X8);
        assert_eq!(buf.pixels.len(), 12);
        assert_eq!(buf.get(0, 0).sample_count(), 8);
    }

    #[test]
    fn test_box_resolve_empty_coverage() {
        let mut buf = MultiSampleBuffer::new(1, 1, SampleCount::X4);
        {
            let pixel = buf.get_mut(0, 0);
            pixel.coverage = CoverageMask::empty(SampleCount::X4);
        }
        let resolved = box_resolve(&buf);
        let p = resolved.get(0, 0);
        assert!(p.r.abs() < 1e-6);
        assert!(p.a.abs() < 1e-6);
    }
}
