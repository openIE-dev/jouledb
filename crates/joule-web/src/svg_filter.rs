//! SVG filter pipeline: filter primitives (blur, color-matrix, displacement-map,
//! turbulence, lighting, merge, composite), filter chain composition, filter region
//! calculation, and named result references (in/in2/result).
//!
//! Pure math — no browser dependency. Operates on abstract RGBA pixel buffers.

use std::collections::HashMap;
use std::fmt;

// ── Color / Pixel ──────────────────────────────────────────────

/// RGBA pixel, channels in 0.0–1.0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pixel {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

impl Pixel {
    pub fn new(r: f64, g: f64, b: f64, a: f64) -> Self {
        Self { r, g, b, a }
    }

    pub fn transparent() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }

    pub fn clamp(self) -> Self {
        Self {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
            a: self.a.clamp(0.0, 1.0),
        }
    }
}

// ── Image Buffer ───────────────────────────────────────────────

/// 2D image buffer of RGBA pixels.
#[derive(Debug, Clone)]
pub struct ImageBuffer {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<Pixel>,
}

impl ImageBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![Pixel::transparent(); width * height],
        }
    }

    pub fn filled(width: usize, height: usize, pixel: Pixel) -> Self {
        Self {
            width,
            height,
            pixels: vec![pixel; width * height],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> Pixel {
        if x < self.width && y < self.height {
            self.pixels[y * self.width + x]
        } else {
            Pixel::transparent()
        }
    }

    pub fn set(&mut self, x: usize, y: usize, px: Pixel) {
        if x < self.width && y < self.height {
            self.pixels[y * self.width + x] = px;
        }
    }

    fn get_clamped(&self, x: i64, y: i64) -> Pixel {
        let cx = x.clamp(0, self.width as i64 - 1) as usize;
        let cy = y.clamp(0, self.height as i64 - 1) as usize;
        self.pixels[cy * self.width + cx]
    }
}

// ── Filter Region ──────────────────────────────────────────────

/// Bounding box for a filter effect region.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilterRegion {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl FilterRegion {
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self { x, y, width, height }
    }

    /// Union of two regions.
    pub fn union(&self, other: &FilterRegion) -> FilterRegion {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        let x2 = (self.x + self.width).max(other.x + other.width);
        let y2 = (self.y + self.height).max(other.y + other.height);
        FilterRegion::new(x, y, x2 - x, y2 - y)
    }

    /// Intersection of two regions.
    pub fn intersect(&self, other: &FilterRegion) -> Option<FilterRegion> {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);
        if x2 > x && y2 > y {
            Some(FilterRegion::new(x, y, x2 - x, y2 - y))
        } else {
            None
        }
    }

    pub fn area(&self) -> f64 {
        self.width * self.height
    }

    /// Expand the region by `margin` on all sides.
    pub fn expand(&self, margin: f64) -> FilterRegion {
        FilterRegion::new(
            self.x - margin,
            self.y - margin,
            self.width + 2.0 * margin,
            self.height + 2.0 * margin,
        )
    }
}

// ── Input Reference ────────────────────────────────────────────

/// Reference to a filter input (SVG `in` / `in2` attributes).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FilterInput {
    /// The original source graphic.
    SourceGraphic,
    /// The source alpha channel.
    SourceAlpha,
    /// A named result from a previous primitive.
    Named(String),
}

impl fmt::Display for FilterInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FilterInput::SourceGraphic => write!(f, "SourceGraphic"),
            FilterInput::SourceAlpha => write!(f, "SourceAlpha"),
            FilterInput::Named(n) => write!(f, "{n}"),
        }
    }
}

// ── Composite Operator ─────────────────────────────────────────

/// SVG feComposite operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeOp {
    Over,
    In,
    Out,
    Atop,
    Xor,
    Arithmetic,
}

/// Coefficients for arithmetic composite: k1*i1*i2 + k2*i1 + k3*i2 + k4.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArithmeticCoeffs {
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
    pub k4: f64,
}

impl Default for ArithmeticCoeffs {
    fn default() -> Self {
        Self { k1: 0.0, k2: 1.0, k3: 0.0, k4: 0.0 }
    }
}

// ── Color Matrix Type ──────────────────────────────────────────

/// Type of feColorMatrix operation.
#[derive(Debug, Clone, PartialEq)]
pub enum ColorMatrixType {
    /// 5×4 matrix (20 values, row-major).
    Matrix([f64; 20]),
    /// Saturate by factor 0..1.
    Saturate(f64),
    /// Hue rotation in degrees.
    HueRotate(f64),
    /// Strip color, keep alpha.
    LuminanceToAlpha,
}

// ── Turbulence ─────────────────────────────────────────────────

/// Type of turbulence noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurbulenceType {
    FractalNoise,
    Turbulence,
}

/// Parameters for feTurbulence.
#[derive(Debug, Clone, PartialEq)]
pub struct TurbulenceParams {
    pub base_frequency_x: f64,
    pub base_frequency_y: f64,
    pub num_octaves: u32,
    pub seed: u32,
    pub stitch_tiles: bool,
    pub turbulence_type: TurbulenceType,
}

impl Default for TurbulenceParams {
    fn default() -> Self {
        Self {
            base_frequency_x: 0.05,
            base_frequency_y: 0.05,
            num_octaves: 1,
            seed: 0,
            stitch_tiles: false,
            turbulence_type: TurbulenceType::Turbulence,
        }
    }
}

// ── Lighting ───────────────────────────────────────────────────

/// Light source for feDiffuseLighting / feSpecularLighting.
#[derive(Debug, Clone, PartialEq)]
pub enum LightSource {
    Distant { azimuth: f64, elevation: f64 },
    Point { x: f64, y: f64, z: f64 },
    Spot { x: f64, y: f64, z: f64, px: f64, py: f64, pz: f64, specular_exponent: f64 },
}

/// Lighting type.
#[derive(Debug, Clone, PartialEq)]
pub enum LightingType {
    Diffuse { surface_scale: f64, diffuse_constant: f64 },
    Specular { surface_scale: f64, specular_constant: f64, specular_exponent: f64 },
}

// ── Filter Primitive ───────────────────────────────────────────

/// A single SVG filter primitive.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterPrimitive {
    GaussianBlur {
        input: FilterInput,
        std_dev_x: f64,
        std_dev_y: f64,
        result: Option<String>,
    },
    ColorMatrix {
        input: FilterInput,
        matrix_type: ColorMatrixType,
        result: Option<String>,
    },
    Composite {
        input1: FilterInput,
        input2: FilterInput,
        operator: CompositeOp,
        coeffs: ArithmeticCoeffs,
        result: Option<String>,
    },
    Merge {
        inputs: Vec<FilterInput>,
        result: Option<String>,
    },
    Turbulence {
        params: TurbulenceParams,
        result: Option<String>,
    },
    DisplacementMap {
        input1: FilterInput,
        input2: FilterInput,
        scale: f64,
        x_channel: Channel,
        y_channel: Channel,
        result: Option<String>,
    },
    Lighting {
        input: FilterInput,
        light: LightSource,
        lighting_type: LightingType,
        result: Option<String>,
    },
    Offset {
        input: FilterInput,
        dx: f64,
        dy: f64,
        result: Option<String>,
    },
    Flood {
        color: Pixel,
        result: Option<String>,
    },
}

impl FilterPrimitive {
    /// The result name for this primitive (if any).
    pub fn result_name(&self) -> Option<&str> {
        match self {
            FilterPrimitive::GaussianBlur { result, .. }
            | FilterPrimitive::ColorMatrix { result, .. }
            | FilterPrimitive::Composite { result, .. }
            | FilterPrimitive::Merge { result, .. }
            | FilterPrimitive::Turbulence { result, .. }
            | FilterPrimitive::DisplacementMap { result, .. }
            | FilterPrimitive::Lighting { result, .. }
            | FilterPrimitive::Offset { result, .. }
            | FilterPrimitive::Flood { result, .. } => result.as_deref(),
        }
    }
}

/// RGBA channel selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    R,
    G,
    B,
    A,
}

// ── Filter Chain ───────────────────────────────────────────────

/// A complete SVG filter: a sequence of primitives + a filter region.
#[derive(Debug, Clone)]
pub struct FilterChain {
    pub primitives: Vec<FilterPrimitive>,
    pub region: FilterRegion,
}

impl FilterChain {
    pub fn new(region: FilterRegion) -> Self {
        Self {
            primitives: Vec::new(),
            region,
        }
    }

    pub fn add(&mut self, primitive: FilterPrimitive) {
        self.primitives.push(primitive);
    }

    pub fn len(&self) -> usize {
        self.primitives.len()
    }

    pub fn is_empty(&self) -> bool {
        self.primitives.is_empty()
    }

    /// Execute the filter chain on a source image, returning the final output.
    pub fn apply(&self, source: &ImageBuffer) -> ImageBuffer {
        let mut named: HashMap<String, ImageBuffer> = HashMap::new();
        let mut last_output = source.clone();

        // Pre-compute source-alpha
        let source_alpha = {
            let mut sa = source.clone();
            for px in sa.pixels.iter_mut() {
                px.r = 0.0;
                px.g = 0.0;
                px.b = 0.0;
            }
            sa
        };

        for prim in &self.primitives {
            let output = self.apply_primitive(prim, source, &source_alpha, &named, &last_output);
            if let Some(name) = prim.result_name() {
                named.insert(name.to_string(), output.clone());
            }
            last_output = output;
        }

        last_output
    }

    fn resolve_input<'a>(
        input: &FilterInput,
        source: &'a ImageBuffer,
        source_alpha: &'a ImageBuffer,
        named: &'a HashMap<String, ImageBuffer>,
        last: &'a ImageBuffer,
    ) -> &'a ImageBuffer {
        match input {
            FilterInput::SourceGraphic => source,
            FilterInput::SourceAlpha => source_alpha,
            FilterInput::Named(n) => named.get(n).unwrap_or(last),
        }
    }

    fn apply_primitive(
        &self,
        prim: &FilterPrimitive,
        source: &ImageBuffer,
        source_alpha: &ImageBuffer,
        named: &HashMap<String, ImageBuffer>,
        last: &ImageBuffer,
    ) -> ImageBuffer {
        match prim {
            FilterPrimitive::GaussianBlur { input, std_dev_x, std_dev_y, .. } => {
                let src = Self::resolve_input(input, source, source_alpha, named, last);
                apply_gaussian_blur(src, *std_dev_x, *std_dev_y)
            }
            FilterPrimitive::ColorMatrix { input, matrix_type, .. } => {
                let src = Self::resolve_input(input, source, source_alpha, named, last);
                apply_color_matrix(src, matrix_type)
            }
            FilterPrimitive::Composite { input1, input2, operator, coeffs, .. } => {
                let a = Self::resolve_input(input1, source, source_alpha, named, last);
                let b = Self::resolve_input(input2, source, source_alpha, named, last);
                apply_composite(a, b, *operator, coeffs)
            }
            FilterPrimitive::Merge { inputs, .. } => {
                let layers: Vec<&ImageBuffer> = inputs
                    .iter()
                    .map(|i| Self::resolve_input(i, source, source_alpha, named, last))
                    .collect();
                apply_merge(&layers)
            }
            FilterPrimitive::Turbulence { params, .. } => {
                generate_turbulence(source.width, source.height, params)
            }
            FilterPrimitive::DisplacementMap { input1, input2, scale, x_channel, y_channel, .. } => {
                let src = Self::resolve_input(input1, source, source_alpha, named, last);
                let map = Self::resolve_input(input2, source, source_alpha, named, last);
                apply_displacement_map(src, map, *scale, *x_channel, *y_channel)
            }
            FilterPrimitive::Lighting { input, light, lighting_type, .. } => {
                let src = Self::resolve_input(input, source, source_alpha, named, last);
                apply_lighting(src, light, lighting_type)
            }
            FilterPrimitive::Offset { input, dx, dy, .. } => {
                let src = Self::resolve_input(input, source, source_alpha, named, last);
                apply_offset(src, *dx, *dy)
            }
            FilterPrimitive::Flood { color, .. } => {
                ImageBuffer::filled(source.width, source.height, *color)
            }
        }
    }
}

// ── Primitive Implementations ──────────────────────────────────

fn apply_gaussian_blur(src: &ImageBuffer, std_x: f64, std_y: f64) -> ImageBuffer {
    // Box blur approximation (3-pass) for Gaussian.
    let mut out = src.clone();
    if std_x > 0.0 {
        let radius = (std_x * 1.5).ceil() as usize;
        horizontal_box_blur(&mut out, radius);
    }
    if std_y > 0.0 {
        let radius = (std_y * 1.5).ceil() as usize;
        vertical_box_blur(&mut out, radius);
    }
    out
}

fn horizontal_box_blur(buf: &mut ImageBuffer, radius: usize) {
    if radius == 0 || buf.width == 0 {
        return;
    }
    let w = buf.width;
    let h = buf.height;
    let diam = (2 * radius + 1) as f64;
    let mut row = vec![Pixel::transparent(); w];
    for y in 0..h {
        // accumulate
        let mut sr = 0.0;
        let mut sg = 0.0;
        let mut sb = 0.0;
        let mut sa = 0.0;
        for dx in 0..=radius.min(w - 1) {
            let p = buf.get(dx, y);
            sr += p.r; sg += p.g; sb += p.b; sa += p.a;
        }
        for x in 0..w {
            let count = {
                let lo = if x > radius { x - radius } else { 0 };
                let hi = (x + radius).min(w - 1);
                (hi - lo + 1) as f64
            };
            row[x] = Pixel::new(sr / count, sg / count, sb / count, sa / count);
            // slide window
            let add_x = x + radius + 1;
            let rem_x = if x >= radius { x - radius } else { usize::MAX };
            if add_x < w {
                let p = buf.get(add_x, y);
                sr += p.r; sg += p.g; sb += p.b; sa += p.a;
            }
            if rem_x < w {
                let p = buf.get(rem_x, y);
                sr -= p.r; sg -= p.g; sb -= p.b; sa -= p.a;
            }
        }
        for x in 0..w {
            buf.set(x, y, row[x]);
        }
    }
}

fn vertical_box_blur(buf: &mut ImageBuffer, radius: usize) {
    if radius == 0 || buf.height == 0 {
        return;
    }
    let w = buf.width;
    let h = buf.height;
    let mut col = vec![Pixel::transparent(); h];
    for x in 0..w {
        let mut sr = 0.0;
        let mut sg = 0.0;
        let mut sb = 0.0;
        let mut sa = 0.0;
        for dy in 0..=radius.min(h - 1) {
            let p = buf.get(x, dy);
            sr += p.r; sg += p.g; sb += p.b; sa += p.a;
        }
        for y in 0..h {
            let count = {
                let lo = if y > radius { y - radius } else { 0 };
                let hi = (y + radius).min(h - 1);
                (hi - lo + 1) as f64
            };
            col[y] = Pixel::new(sr / count, sg / count, sb / count, sa / count);
            let add_y = y + radius + 1;
            let rem_y = if y >= radius { y - radius } else { usize::MAX };
            if add_y < h {
                let p = buf.get(x, add_y);
                sr += p.r; sg += p.g; sb += p.b; sa += p.a;
            }
            if rem_y < h {
                let p = buf.get(x, rem_y);
                sr -= p.r; sg -= p.g; sb -= p.b; sa -= p.a;
            }
        }
        for y in 0..h {
            buf.set(x, y, col[y]);
        }
    }
}

fn apply_color_matrix(src: &ImageBuffer, mat: &ColorMatrixType) -> ImageBuffer {
    let mut out = src.clone();
    let matrix = match mat {
        ColorMatrixType::Matrix(m) => *m,
        ColorMatrixType::Saturate(s) => saturate_matrix(*s),
        ColorMatrixType::HueRotate(deg) => hue_rotate_matrix(*deg),
        ColorMatrixType::LuminanceToAlpha => luminance_to_alpha_matrix(),
    };
    for px in out.pixels.iter_mut() {
        let (r, g, b, a) = (px.r, px.g, px.b, px.a);
        px.r = (matrix[0] * r + matrix[1] * g + matrix[2] * b + matrix[3] * a + matrix[4]).clamp(0.0, 1.0);
        px.g = (matrix[5] * r + matrix[6] * g + matrix[7] * b + matrix[8] * a + matrix[9]).clamp(0.0, 1.0);
        px.b = (matrix[10] * r + matrix[11] * g + matrix[12] * b + matrix[13] * a + matrix[14]).clamp(0.0, 1.0);
        px.a = (matrix[15] * r + matrix[16] * g + matrix[17] * b + matrix[18] * a + matrix[19]).clamp(0.0, 1.0);
    }
    out
}

fn saturate_matrix(s: f64) -> [f64; 20] {
    let mut m = [0.0f64; 20];
    m[0] = 0.213 + 0.787 * s;
    m[1] = 0.715 - 0.715 * s;
    m[2] = 0.072 - 0.072 * s;
    m[5] = 0.213 - 0.213 * s;
    m[6] = 0.715 + 0.285 * s;
    m[7] = 0.072 - 0.072 * s;
    m[10] = 0.213 - 0.213 * s;
    m[11] = 0.715 - 0.715 * s;
    m[12] = 0.072 + 0.928 * s;
    m[18] = 1.0;
    m
}

fn hue_rotate_matrix(deg: f64) -> [f64; 20] {
    let rad = deg * std::f64::consts::PI / 180.0;
    let cos = rad.cos();
    let sin = rad.sin();
    let mut m = [0.0f64; 20];
    m[0] = 0.213 + cos * 0.787 - sin * 0.213;
    m[1] = 0.715 - cos * 0.715 - sin * 0.715;
    m[2] = 0.072 - cos * 0.072 + sin * 0.928;
    m[5] = 0.213 - cos * 0.213 + sin * 0.143;
    m[6] = 0.715 + cos * 0.285 + sin * 0.140;
    m[7] = 0.072 - cos * 0.072 - sin * 0.283;
    m[10] = 0.213 - cos * 0.213 - sin * 0.787;
    m[11] = 0.715 - cos * 0.715 + sin * 0.715;
    m[12] = 0.072 + cos * 0.928 + sin * 0.072;
    m[18] = 1.0;
    m
}

fn luminance_to_alpha_matrix() -> [f64; 20] {
    let mut m = [0.0f64; 20];
    m[15] = 0.2126;
    m[16] = 0.7152;
    m[17] = 0.0722;
    m
}

fn apply_composite(a: &ImageBuffer, b: &ImageBuffer, op: CompositeOp, coeffs: &ArithmeticCoeffs) -> ImageBuffer {
    let w = a.width.min(b.width);
    let h = a.height.min(b.height);
    let mut out = ImageBuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let pa = a.get(x, y);
            let pb = b.get(x, y);
            let result = match op {
                CompositeOp::Over => composite_over(pa, pb),
                CompositeOp::In => composite_in(pa, pb),
                CompositeOp::Out => composite_out(pa, pb),
                CompositeOp::Atop => composite_atop(pa, pb),
                CompositeOp::Xor => composite_xor(pa, pb),
                CompositeOp::Arithmetic => composite_arithmetic(pa, pb, coeffs),
            };
            out.set(x, y, result.clamp());
        }
    }
    out
}

fn composite_over(a: Pixel, b: Pixel) -> Pixel {
    let oa = a.a + b.a * (1.0 - a.a);
    if oa < 1e-10 { return Pixel::transparent(); }
    Pixel::new(
        (a.r * a.a + b.r * b.a * (1.0 - a.a)) / oa,
        (a.g * a.a + b.g * b.a * (1.0 - a.a)) / oa,
        (a.b * a.a + b.b * b.a * (1.0 - a.a)) / oa,
        oa,
    )
}

fn composite_in(a: Pixel, b: Pixel) -> Pixel {
    Pixel::new(a.r, a.g, a.b, a.a * b.a)
}

fn composite_out(a: Pixel, b: Pixel) -> Pixel {
    Pixel::new(a.r, a.g, a.b, a.a * (1.0 - b.a))
}

fn composite_atop(a: Pixel, b: Pixel) -> Pixel {
    let oa = b.a;
    Pixel::new(
        a.r * a.a + b.r * (1.0 - a.a),
        a.g * a.a + b.g * (1.0 - a.a),
        a.b * a.a + b.b * (1.0 - a.a),
        oa,
    )
}

fn composite_xor(a: Pixel, b: Pixel) -> Pixel {
    let oa = a.a * (1.0 - b.a) + b.a * (1.0 - a.a);
    if oa < 1e-10 { return Pixel::transparent(); }
    Pixel::new(
        (a.r * a.a * (1.0 - b.a) + b.r * b.a * (1.0 - a.a)) / oa,
        (a.g * a.a * (1.0 - b.a) + b.g * b.a * (1.0 - a.a)) / oa,
        (a.b * a.a * (1.0 - b.a) + b.b * b.a * (1.0 - a.a)) / oa,
        oa,
    )
}

fn composite_arithmetic(a: Pixel, b: Pixel, c: &ArithmeticCoeffs) -> Pixel {
    Pixel::new(
        c.k1 * a.r * b.r + c.k2 * a.r + c.k3 * b.r + c.k4,
        c.k1 * a.g * b.g + c.k2 * a.g + c.k3 * b.g + c.k4,
        c.k1 * a.b * b.b + c.k2 * a.b + c.k3 * b.b + c.k4,
        c.k1 * a.a * b.a + c.k2 * a.a + c.k3 * b.a + c.k4,
    )
}

fn apply_merge(layers: &[&ImageBuffer]) -> ImageBuffer {
    if layers.is_empty() {
        return ImageBuffer::new(1, 1);
    }
    let w = layers.iter().map(|l| l.width).max().unwrap_or(1);
    let h = layers.iter().map(|l| l.height).max().unwrap_or(1);
    let mut out = ImageBuffer::new(w, h);
    for layer in layers {
        for y in 0..h.min(layer.height) {
            for x in 0..w.min(layer.width) {
                let dst = out.get(x, y);
                let src = layer.get(x, y);
                out.set(x, y, composite_over(src, dst));
            }
        }
    }
    out
}

fn generate_turbulence(width: usize, height: usize, params: &TurbulenceParams) -> ImageBuffer {
    let mut out = ImageBuffer::new(width, height);
    // Simplified Perlin-like noise based on seed + frequency.
    let seed = params.seed as f64;
    for y in 0..height {
        for x in 0..width {
            let mut val = 0.0;
            let mut amp = 1.0;
            let mut freq_x = params.base_frequency_x;
            let mut freq_y = params.base_frequency_y;
            for _oct in 0..params.num_octaves {
                let nx = x as f64 * freq_x + seed;
                let ny = y as f64 * freq_y + seed * 1.7;
                let n = simple_noise(nx, ny);
                val += n * amp;
                amp *= 0.5;
                freq_x *= 2.0;
                freq_y *= 2.0;
            }
            let v = match params.turbulence_type {
                TurbulenceType::FractalNoise => (val * 0.5 + 0.5).clamp(0.0, 1.0),
                TurbulenceType::Turbulence => val.abs().clamp(0.0, 1.0),
            };
            out.set(x, y, Pixel::new(v, v, v, 1.0));
        }
    }
    out
}

fn simple_noise(x: f64, y: f64) -> f64 {
    // Hash-based pseudo-noise.
    let ix = x.floor() as i64;
    let iy = y.floor() as i64;
    let fx = x - x.floor();
    let fy = y - y.floor();
    let u = fx * fx * (3.0 - 2.0 * fx);
    let v = fy * fy * (3.0 - 2.0 * fy);
    let n00 = hash_noise(ix, iy);
    let n10 = hash_noise(ix + 1, iy);
    let n01 = hash_noise(ix, iy + 1);
    let n11 = hash_noise(ix + 1, iy + 1);
    let nx0 = n00 + (n10 - n00) * u;
    let nx1 = n01 + (n11 - n01) * u;
    nx0 + (nx1 - nx0) * v
}

fn hash_noise(x: i64, y: i64) -> f64 {
    let mut h = (x.wrapping_mul(374761393) + y.wrapping_mul(668265263)) as u64;
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    h = h ^ (h >> 16);
    (h & 0xFFFF) as f64 / 65535.0 * 2.0 - 1.0
}

fn apply_displacement_map(
    src: &ImageBuffer,
    map: &ImageBuffer,
    scale: f64,
    x_ch: Channel,
    y_ch: Channel,
) -> ImageBuffer {
    let w = src.width;
    let h = src.height;
    let mut out = ImageBuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mp = map.get(x, y);
            let dx = (channel_value(mp, x_ch) - 0.5) * scale;
            let dy = (channel_value(mp, y_ch) - 0.5) * scale;
            let sx = x as f64 + dx;
            let sy = y as f64 + dy;
            out.set(x, y, src.get_clamped(sx as i64, sy as i64));
        }
    }
    out
}

fn channel_value(p: Pixel, ch: Channel) -> f64 {
    match ch {
        Channel::R => p.r,
        Channel::G => p.g,
        Channel::B => p.b,
        Channel::A => p.a,
    }
}

fn apply_lighting(src: &ImageBuffer, light: &LightSource, lt: &LightingType) -> ImageBuffer {
    let w = src.width;
    let h = src.height;
    let mut out = ImageBuffer::new(w, h);
    let (surface_scale, is_specular) = match lt {
        LightingType::Diffuse { surface_scale, .. } => (*surface_scale, false),
        LightingType::Specular { surface_scale, .. } => (*surface_scale, true),
    };

    for y in 0..h {
        for x in 0..w {
            // Normal from alpha channel (Sobel-like).
            let left = src.get(x.saturating_sub(1), y).a;
            let right = src.get((x + 1).min(w - 1), y).a;
            let up = src.get(x, y.saturating_sub(1)).a;
            let down = src.get(x, (y + 1).min(h - 1)).a;
            let nx = -(right - left) * surface_scale;
            let ny = -(down - up) * surface_scale;
            let nz = 1.0;
            let nl = (nx * nx + ny * ny + nz * nz).sqrt();
            let (nx, ny, nz) = (nx / nl, ny / nl, nz / nl);

            let (lx, ly, lz) = light_direction(light, x as f64, y as f64);
            let dot = (nx * lx + ny * ly + nz * lz).max(0.0);

            let intensity = if is_specular {
                let spec_exp = match lt {
                    LightingType::Specular { specular_exponent, .. } => *specular_exponent,
                    _ => 1.0,
                };
                let sc = match lt {
                    LightingType::Specular { specular_constant, .. } => *specular_constant,
                    _ => 1.0,
                };
                sc * dot.powf(spec_exp)
            } else {
                let dc = match lt {
                    LightingType::Diffuse { diffuse_constant, .. } => *diffuse_constant,
                    _ => 1.0,
                };
                dc * dot
            };

            let v = intensity.clamp(0.0, 1.0);
            out.set(x, y, Pixel::new(v, v, v, 1.0));
        }
    }
    out
}

fn light_direction(light: &LightSource, _px: f64, _py: f64) -> (f64, f64, f64) {
    match light {
        LightSource::Distant { azimuth, elevation } => {
            let az = azimuth * std::f64::consts::PI / 180.0;
            let el = elevation * std::f64::consts::PI / 180.0;
            (az.cos() * el.cos(), az.sin() * el.cos(), el.sin())
        }
        LightSource::Point { x, y, z } => {
            let dx = x - _px;
            let dy = y - _py;
            let dz = *z;
            let l = (dx * dx + dy * dy + dz * dz).sqrt();
            if l < 1e-10 { (0.0, 0.0, 1.0) } else { (dx / l, dy / l, dz / l) }
        }
        LightSource::Spot { x, y, z, .. } => {
            let dx = x - _px;
            let dy = y - _py;
            let dz = *z;
            let l = (dx * dx + dy * dy + dz * dz).sqrt();
            if l < 1e-10 { (0.0, 0.0, 1.0) } else { (dx / l, dy / l, dz / l) }
        }
    }
}

fn apply_offset(src: &ImageBuffer, dx: f64, dy: f64) -> ImageBuffer {
    let w = src.width;
    let h = src.height;
    let mut out = ImageBuffer::new(w, h);
    let idx = dx.round() as i64;
    let idy = dy.round() as i64;
    for y in 0..h {
        for x in 0..w {
            let sx = x as i64 - idx;
            let sy = y as i64 - idy;
            if sx >= 0 && sx < w as i64 && sy >= 0 && sy < h as i64 {
                out.set(x, y, src.get(sx as usize, sy as usize));
            }
        }
    }
    out
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn white_image(w: usize, h: usize) -> ImageBuffer {
        ImageBuffer::filled(w, h, Pixel::new(1.0, 1.0, 1.0, 1.0))
    }

    fn red_image(w: usize, h: usize) -> ImageBuffer {
        ImageBuffer::filled(w, h, Pixel::new(1.0, 0.0, 0.0, 1.0))
    }

    #[test]
    fn test_pixel_clamp() {
        let p = Pixel::new(1.5, -0.2, 0.5, 2.0).clamp();
        assert_eq!(p.r, 1.0);
        assert_eq!(p.g, 0.0);
        assert_eq!(p.b, 0.5);
        assert_eq!(p.a, 1.0);
    }

    #[test]
    fn test_filter_region_union() {
        let a = FilterRegion::new(0.0, 0.0, 10.0, 10.0);
        let b = FilterRegion::new(5.0, 5.0, 10.0, 10.0);
        let u = a.union(&b);
        assert_eq!(u.x, 0.0);
        assert_eq!(u.y, 0.0);
        assert_eq!(u.width, 15.0);
        assert_eq!(u.height, 15.0);
    }

    #[test]
    fn test_filter_region_intersect() {
        let a = FilterRegion::new(0.0, 0.0, 10.0, 10.0);
        let b = FilterRegion::new(5.0, 5.0, 10.0, 10.0);
        let i = a.intersect(&b).unwrap();
        assert_eq!(i.x, 5.0);
        assert_eq!(i.width, 5.0);
    }

    #[test]
    fn test_filter_region_no_intersect() {
        let a = FilterRegion::new(0.0, 0.0, 5.0, 5.0);
        let b = FilterRegion::new(10.0, 10.0, 5.0, 5.0);
        assert!(a.intersect(&b).is_none());
    }

    #[test]
    fn test_flood_primitive() {
        let chain = FilterChain {
            primitives: vec![FilterPrimitive::Flood {
                color: Pixel::new(0.0, 0.5, 1.0, 1.0),
                result: None,
            }],
            region: FilterRegion::new(0.0, 0.0, 4.0, 4.0),
        };
        let src = ImageBuffer::new(4, 4);
        let out = chain.apply(&src);
        assert!((out.get(0, 0).g - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_offset_primitive() {
        let mut src = ImageBuffer::new(4, 4);
        src.set(0, 0, Pixel::new(1.0, 0.0, 0.0, 1.0));
        let out = apply_offset(&src, 2.0, 1.0);
        assert!(out.get(0, 0).a < 1e-10); // original spot is empty
        assert!((out.get(2, 1).r - 1.0).abs() < 1e-10); // shifted
    }

    #[test]
    fn test_composite_over() {
        let a = Pixel::new(1.0, 0.0, 0.0, 0.5);
        let b = Pixel::new(0.0, 0.0, 1.0, 1.0);
        let r = composite_over(a, b);
        assert!(r.a > 0.9);
        assert!(r.r > 0.3); // red shows through
    }

    #[test]
    fn test_composite_in_op() {
        let a = Pixel::new(1.0, 0.0, 0.0, 1.0);
        let b = Pixel::new(0.0, 1.0, 0.0, 0.5);
        let r = composite_in(a, b);
        assert!((r.a - 0.5).abs() < 1e-10);
        assert!((r.r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_blur_preserves_size() {
        let src = red_image(8, 8);
        let out = apply_gaussian_blur(&src, 2.0, 2.0);
        assert_eq!(out.width, 8);
        assert_eq!(out.height, 8);
    }

    #[test]
    fn test_color_matrix_identity() {
        let mut identity = [0.0f64; 20];
        identity[0] = 1.0;
        identity[6] = 1.0;
        identity[12] = 1.0;
        identity[18] = 1.0;
        let src = red_image(2, 2);
        let out = apply_color_matrix(&src, &ColorMatrixType::Matrix(identity));
        let p = out.get(0, 0);
        assert!((p.r - 1.0).abs() < 1e-10);
        assert!(p.g.abs() < 1e-10);
    }

    #[test]
    fn test_turbulence_generates_pixels() {
        let params = TurbulenceParams::default();
        let out = generate_turbulence(8, 8, &params);
        assert_eq!(out.width, 8);
        // All pixels should have a > 0
        for px in &out.pixels {
            assert!((px.a - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_chain_with_named_results() {
        let mut chain = FilterChain::new(FilterRegion::new(0.0, 0.0, 4.0, 4.0));
        chain.add(FilterPrimitive::Flood {
            color: Pixel::new(0.0, 1.0, 0.0, 1.0),
            result: Some("green".into()),
        });
        chain.add(FilterPrimitive::Flood {
            color: Pixel::new(1.0, 0.0, 0.0, 1.0),
            result: Some("red".into()),
        });
        chain.add(FilterPrimitive::Merge {
            inputs: vec![
                FilterInput::Named("green".into()),
                FilterInput::Named("red".into()),
            ],
            result: None,
        });
        let src = ImageBuffer::new(4, 4);
        let out = chain.apply(&src);
        // Red is on top via source-over onto green
        let p = out.get(0, 0);
        assert!((p.r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_filter_region_expand() {
        let r = FilterRegion::new(10.0, 10.0, 20.0, 20.0);
        let e = r.expand(5.0);
        assert_eq!(e.x, 5.0);
        assert_eq!(e.width, 30.0);
    }

    #[test]
    fn test_composite_arithmetic() {
        let a = Pixel::new(0.5, 0.5, 0.5, 1.0);
        let b = Pixel::new(0.5, 0.5, 0.5, 1.0);
        let coeffs = ArithmeticCoeffs { k1: 1.0, k2: 0.0, k3: 0.0, k4: 0.0 };
        let r = composite_arithmetic(a, b, &coeffs);
        assert!((r.r - 0.25).abs() < 1e-10); // 0.5 * 0.5
    }

    #[test]
    fn test_displacement_map_zero_scale() {
        let src = red_image(4, 4);
        let map = white_image(4, 4);
        let out = apply_displacement_map(&src, &map, 0.0, Channel::R, Channel::G);
        // Zero scale = no displacement, output should match source.
        let p = out.get(1, 1);
        assert!((p.r - 1.0).abs() < 1e-10);
    }
}
