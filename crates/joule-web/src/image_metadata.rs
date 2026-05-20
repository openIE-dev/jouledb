//! Image metadata extraction — EXIF-like metadata model, orientation,
//! dimensions, color depth, GPS coordinates, camera info, creation date,
//! thumbnail presence, metadata stripping, and JSON export.
//!
//! Pure-Rust replacement for exifreader, sharp metadata, piexifjs,
//! and similar image metadata libraries.

use std::collections::HashMap;
use std::fmt;
use std::fmt::Write as FmtWrite;

// ── Errors ──────────────────────────────────────────────────────

/// Errors from image metadata operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageMetaError {
    InvalidFormat(String),
    MissingField(String),
    InvalidGps(String),
}

impl fmt::Display for ImageMetaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFormat(msg) => write!(f, "invalid image format: {msg}"),
            Self::MissingField(msg) => write!(f, "missing metadata field: {msg}"),
            Self::InvalidGps(msg) => write!(f, "invalid GPS data: {msg}"),
        }
    }
}

impl std::error::Error for ImageMetaError {}

// ── Orientation ─────────────────────────────────────────────────

/// EXIF orientation values (1-8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    /// Normal (no transformation).
    Normal,
    /// Flipped horizontally.
    FlipHorizontal,
    /// Rotated 180 degrees.
    Rotate180,
    /// Flipped vertically.
    FlipVertical,
    /// Transposed (flipped horizontally + rotated 270 CW).
    Transpose,
    /// Rotated 90 degrees clockwise.
    Rotate90Cw,
    /// Transversed (flipped horizontally + rotated 90 CW).
    Transverse,
    /// Rotated 270 degrees clockwise (90 CCW).
    Rotate270Cw,
}

impl Orientation {
    /// Create from EXIF orientation tag value (1-8).
    pub fn from_exif_value(val: u16) -> Self {
        match val {
            1 => Self::Normal,
            2 => Self::FlipHorizontal,
            3 => Self::Rotate180,
            4 => Self::FlipVertical,
            5 => Self::Transpose,
            6 => Self::Rotate90Cw,
            7 => Self::Transverse,
            8 => Self::Rotate270Cw,
            _ => Self::Normal,
        }
    }

    /// Convert to EXIF orientation tag value.
    pub fn to_exif_value(&self) -> u16 {
        match self {
            Self::Normal => 1,
            Self::FlipHorizontal => 2,
            Self::Rotate180 => 3,
            Self::FlipVertical => 4,
            Self::Transpose => 5,
            Self::Rotate90Cw => 6,
            Self::Transverse => 7,
            Self::Rotate270Cw => 8,
        }
    }

    /// Whether the image needs rotation to display correctly.
    pub fn needs_rotation(&self) -> bool {
        !matches!(self, Self::Normal | Self::FlipHorizontal | Self::FlipVertical)
    }

    /// Whether the image dimensions are swapped (width/height swapped).
    pub fn dimensions_swapped(&self) -> bool {
        matches!(
            self,
            Self::Transpose | Self::Rotate90Cw | Self::Transverse | Self::Rotate270Cw
        )
    }

    /// Description of the orientation.
    pub fn description(&self) -> &str {
        match self {
            Self::Normal => "Normal",
            Self::FlipHorizontal => "Flipped horizontally",
            Self::Rotate180 => "Rotated 180 degrees",
            Self::FlipVertical => "Flipped vertically",
            Self::Transpose => "Transposed",
            Self::Rotate90Cw => "Rotated 90 CW",
            Self::Transverse => "Transversed",
            Self::Rotate270Cw => "Rotated 270 CW",
        }
    }
}

// ── Color Space ─────────────────────────────────────────────────

/// Color space of the image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    Srgb,
    AdobeRgb,
    ProPhotoRgb,
    DisplayP3,
    Grayscale,
    Cmyk,
    Unknown,
}

impl ColorSpace {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Srgb => "sRGB",
            Self::AdobeRgb => "Adobe RGB",
            Self::ProPhotoRgb => "ProPhoto RGB",
            Self::DisplayP3 => "Display P3",
            Self::Grayscale => "Grayscale",
            Self::Cmyk => "CMYK",
            Self::Unknown => "Unknown",
        }
    }
}

impl fmt::Display for ColorSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── Image Format ────────────────────────────────────────────────

/// Image file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Jpeg,
    Png,
    Gif,
    Webp,
    Tiff,
    Bmp,
    Avif,
    Heic,
    Svg,
    Unknown,
}

impl ImageFormat {
    /// Detect format from file extension.
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "jpg" | "jpeg" => Self::Jpeg,
            "png" => Self::Png,
            "gif" => Self::Gif,
            "webp" => Self::Webp,
            "tiff" | "tif" => Self::Tiff,
            "bmp" => Self::Bmp,
            "avif" => Self::Avif,
            "heic" | "heif" => Self::Heic,
            "svg" => Self::Svg,
            _ => Self::Unknown,
        }
    }

    /// Detect format from magic bytes.
    pub fn from_magic_bytes(bytes: &[u8]) -> Self {
        if bytes.len() < 4 {
            return Self::Unknown;
        }
        if bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF {
            Self::Jpeg
        } else if bytes[0..4] == [0x89, 0x50, 0x4E, 0x47] {
            Self::Png
        } else if bytes[0..4] == [0x47, 0x49, 0x46, 0x38] {
            Self::Gif
        } else if bytes.len() >= 12 && bytes[8..12] == [0x57, 0x45, 0x42, 0x50] {
            Self::Webp
        } else if (bytes[0..2] == [0x49, 0x49] || bytes[0..2] == [0x4D, 0x4D])
            && bytes[2] == 0x00
            && bytes[3] == 0x2A
        {
            Self::Tiff
        } else if bytes[0..2] == [0x42, 0x4D] {
            Self::Bmp
        } else {
            Self::Unknown
        }
    }

    /// MIME type for this format.
    pub fn mime_type(&self) -> &str {
        match self {
            Self::Jpeg => "image/jpeg",
            Self::Png => "image/png",
            Self::Gif => "image/gif",
            Self::Webp => "image/webp",
            Self::Tiff => "image/tiff",
            Self::Bmp => "image/bmp",
            Self::Avif => "image/avif",
            Self::Heic => "image/heic",
            Self::Svg => "image/svg+xml",
            Self::Unknown => "application/octet-stream",
        }
    }

    /// Whether the format supports EXIF metadata.
    pub fn supports_exif(&self) -> bool {
        matches!(self, Self::Jpeg | Self::Tiff | Self::Heic | Self::Webp)
    }
}

// ── GPS Coordinates ─────────────────────────────────────────────

/// GPS coordinates in decimal degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpsCoordinates {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: Option<f64>,
}

impl GpsCoordinates {
    pub fn new(latitude: f64, longitude: f64) -> Result<Self, ImageMetaError> {
        if !(-90.0..=90.0).contains(&latitude) {
            return Err(ImageMetaError::InvalidGps(format!(
                "latitude {latitude} out of range [-90, 90]"
            )));
        }
        if !(-180.0..=180.0).contains(&longitude) {
            return Err(ImageMetaError::InvalidGps(format!(
                "longitude {longitude} out of range [-180, 180]"
            )));
        }
        Ok(Self {
            latitude,
            longitude,
            altitude: None,
        })
    }

    pub fn with_altitude(mut self, alt: f64) -> Self {
        self.altitude = Some(alt);
        self
    }

    /// Convert from DMS (degrees, minutes, seconds) to decimal degrees.
    pub fn from_dms(
        lat_deg: f64,
        lat_min: f64,
        lat_sec: f64,
        lat_ref: char,
        lon_deg: f64,
        lon_min: f64,
        lon_sec: f64,
        lon_ref: char,
    ) -> Result<Self, ImageMetaError> {
        let mut lat = lat_deg + lat_min / 60.0 + lat_sec / 3600.0;
        if lat_ref == 'S' || lat_ref == 's' {
            lat = -lat;
        }

        let mut lon = lon_deg + lon_min / 60.0 + lon_sec / 3600.0;
        if lon_ref == 'W' || lon_ref == 'w' {
            lon = -lon;
        }

        Self::new(lat, lon)
    }

    /// Convert to DMS string format.
    pub fn to_dms_string(&self) -> String {
        let lat_ref = if self.latitude >= 0.0 { "N" } else { "S" };
        let lon_ref = if self.longitude >= 0.0 { "E" } else { "W" };

        let lat = self.latitude.abs();
        let lat_d = lat as u32;
        let lat_m = ((lat - lat_d as f64) * 60.0) as u32;
        let lat_s = (lat - lat_d as f64 - lat_m as f64 / 60.0) * 3600.0;

        let lon = self.longitude.abs();
        let lon_d = lon as u32;
        let lon_m = ((lon - lon_d as f64) * 60.0) as u32;
        let lon_s = (lon - lon_d as f64 - lon_m as f64 / 60.0) * 3600.0;

        format!(
            "{}\u{00B0}{}'{:.1}\"{} {}\u{00B0}{}'{:.1}\"{}",
            lat_d, lat_m, lat_s, lat_ref, lon_d, lon_m, lon_s, lon_ref
        )
    }
}

impl fmt::Display for GpsCoordinates {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6}, {:.6}", self.latitude, self.longitude)?;
        if let Some(alt) = self.altitude {
            write!(f, " ({:.1}m)", alt)?;
        }
        Ok(())
    }
}

// ── Camera Info ─────────────────────────────────────────────────

/// Camera and lens information.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CameraInfo {
    pub make: Option<String>,
    pub model: Option<String>,
    pub lens: Option<String>,
    pub focal_length_mm: Option<f64>,
    pub aperture: Option<f64>,
    pub shutter_speed: Option<String>,
    pub iso: Option<u32>,
    pub flash_fired: Option<bool>,
    pub exposure_mode: Option<String>,
    pub white_balance: Option<String>,
}

impl CameraInfo {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_make(mut self, make: impl Into<String>) -> Self {
        self.make = Some(make.into());
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_lens(mut self, lens: impl Into<String>) -> Self {
        self.lens = Some(lens.into());
        self
    }

    pub fn with_focal_length(mut self, mm: f64) -> Self {
        self.focal_length_mm = Some(mm);
        self
    }

    pub fn with_aperture(mut self, f_stop: f64) -> Self {
        self.aperture = Some(f_stop);
        self
    }

    pub fn with_shutter_speed(mut self, speed: impl Into<String>) -> Self {
        self.shutter_speed = Some(speed.into());
        self
    }

    pub fn with_iso(mut self, iso: u32) -> Self {
        self.iso = Some(iso);
        self
    }

    pub fn with_flash(mut self, fired: bool) -> Self {
        self.flash_fired = Some(fired);
        self
    }

    /// Summary string like "Canon EOS R5, f/2.8, 1/250s, ISO 400".
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if let Some(model) = &self.model {
            parts.push(model.clone());
        }
        if let Some(aperture) = self.aperture {
            parts.push(format!("f/{:.1}", aperture));
        }
        if let Some(speed) = &self.shutter_speed {
            parts.push(speed.clone());
        }
        if let Some(iso) = self.iso {
            parts.push(format!("ISO {}", iso));
        }
        parts.join(", ")
    }
}

// ── Image Dimensions ────────────────────────────────────────────

/// Image dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dimensions {
    pub width: u32,
    pub height: u32,
}

impl Dimensions {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Total number of pixels.
    pub fn pixel_count(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Megapixels.
    pub fn megapixels(&self) -> f64 {
        self.pixel_count() as f64 / 1_000_000.0
    }

    /// Aspect ratio as a float (width / height).
    pub fn aspect_ratio(&self) -> f64 {
        if self.height == 0 {
            return 0.0;
        }
        self.width as f64 / self.height as f64
    }

    /// Aspect ratio as a simplified string (e.g., "16:9").
    pub fn aspect_ratio_string(&self) -> String {
        if self.width == 0 || self.height == 0 {
            return "0:0".into();
        }
        let g = gcd(self.width, self.height);
        format!("{}:{}", self.width / g, self.height / g)
    }

    /// Display dimensions after applying orientation.
    pub fn display_dimensions(&self, orientation: Orientation) -> Dimensions {
        if orientation.dimensions_swapped() {
            Dimensions::new(self.height, self.width)
        } else {
            *self
        }
    }
}

impl fmt::Display for Dimensions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

// ── Image Metadata ──────────────────────────────────────────────

/// Complete image metadata.
#[derive(Debug, Clone)]
pub struct ImageMetadata {
    pub format: ImageFormat,
    pub dimensions: Dimensions,
    pub color_depth: u8,
    pub color_space: ColorSpace,
    pub orientation: Orientation,
    pub camera: CameraInfo,
    pub gps: Option<GpsCoordinates>,
    pub creation_date: Option<String>,
    pub modification_date: Option<String>,
    pub software: Option<String>,
    pub description: Option<String>,
    pub copyright: Option<String>,
    pub has_thumbnail: bool,
    pub file_size_bytes: Option<u64>,
    pub custom_tags: HashMap<String, String>,
}

impl ImageMetadata {
    /// Create metadata with required fields.
    pub fn new(format: ImageFormat, dimensions: Dimensions) -> Self {
        Self {
            format,
            dimensions,
            color_depth: 8,
            color_space: ColorSpace::Srgb,
            orientation: Orientation::Normal,
            camera: CameraInfo::new(),
            gps: None,
            creation_date: None,
            modification_date: None,
            software: None,
            description: None,
            copyright: None,
            has_thumbnail: false,
            file_size_bytes: None,
            custom_tags: HashMap::new(),
        }
    }

    pub fn with_color_depth(mut self, depth: u8) -> Self {
        self.color_depth = depth;
        self
    }

    pub fn with_color_space(mut self, space: ColorSpace) -> Self {
        self.color_space = space;
        self
    }

    pub fn with_orientation(mut self, orientation: Orientation) -> Self {
        self.orientation = orientation;
        self
    }

    pub fn with_camera(mut self, camera: CameraInfo) -> Self {
        self.camera = camera;
        self
    }

    pub fn with_gps(mut self, gps: GpsCoordinates) -> Self {
        self.gps = Some(gps);
        self
    }

    pub fn with_creation_date(mut self, date: impl Into<String>) -> Self {
        self.creation_date = Some(date.into());
        self
    }

    pub fn with_software(mut self, sw: impl Into<String>) -> Self {
        self.software = Some(sw.into());
        self
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn with_copyright(mut self, copyright: impl Into<String>) -> Self {
        self.copyright = Some(copyright.into());
        self
    }

    pub fn with_thumbnail(mut self, has: bool) -> Self {
        self.has_thumbnail = has;
        self
    }

    pub fn with_file_size(mut self, size: u64) -> Self {
        self.file_size_bytes = Some(size);
        self
    }

    pub fn set_tag(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.custom_tags.insert(key.into(), value.into());
    }

    pub fn get_tag(&self, key: &str) -> Option<&str> {
        self.custom_tags.get(key).map(|s| s.as_str())
    }

    /// Display dimensions after accounting for orientation.
    pub fn display_dimensions(&self) -> Dimensions {
        self.dimensions.display_dimensions(self.orientation)
    }

    /// Strip all metadata, returning a new metadata with only basic info.
    pub fn stripped(&self) -> Self {
        Self::new(self.format, self.dimensions)
    }

    /// Export metadata to a JSON-compatible string.
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\n");
        let _ = write!(out, "  \"format\": \"{:?}\",\n", self.format);
        let _ = write!(
            out,
            "  \"dimensions\": {{ \"width\": {}, \"height\": {} }},\n",
            self.dimensions.width, self.dimensions.height
        );
        let _ = write!(out, "  \"colorDepth\": {},\n", self.color_depth);
        let _ = write!(out, "  \"colorSpace\": \"{}\",\n", self.color_space);
        let _ = write!(
            out,
            "  \"orientation\": {},\n",
            self.orientation.to_exif_value()
        );
        let _ = write!(out, "  \"hasThumbnail\": {},\n", self.has_thumbnail);
        let _ = write!(
            out,
            "  \"megapixels\": {:.2}",
            self.dimensions.megapixels()
        );

        if let Some(date) = &self.creation_date {
            let _ = write!(out, ",\n  \"creationDate\": \"{}\"", json_escape(date));
        }

        if let Some(gps) = &self.gps {
            let _ = write!(
                out,
                ",\n  \"gps\": {{ \"latitude\": {:.6}, \"longitude\": {:.6}",
                gps.latitude, gps.longitude
            );
            if let Some(alt) = gps.altitude {
                let _ = write!(out, ", \"altitude\": {:.1}", alt);
            }
            out.push_str(" }");
        }

        if let Some(model) = &self.camera.model {
            let _ = write!(out, ",\n  \"camera\": \"{}\"", json_escape(model));
        }

        if let Some(desc) = &self.description {
            let _ = write!(out, ",\n  \"description\": \"{}\"", json_escape(desc));
        }

        if let Some(cr) = &self.copyright {
            let _ = write!(out, ",\n  \"copyright\": \"{}\"", json_escape(cr));
        }

        if let Some(size) = self.file_size_bytes {
            let _ = write!(out, ",\n  \"fileSizeBytes\": {}", size);
        }

        out.push_str("\n}");
        out
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orientation_from_exif() {
        assert_eq!(Orientation::from_exif_value(1), Orientation::Normal);
        assert_eq!(Orientation::from_exif_value(6), Orientation::Rotate90Cw);
        assert_eq!(Orientation::from_exif_value(99), Orientation::Normal);
    }

    #[test]
    fn test_orientation_roundtrip() {
        for val in 1..=8 {
            let orientation = Orientation::from_exif_value(val);
            assert_eq!(orientation.to_exif_value(), val);
        }
    }

    #[test]
    fn test_orientation_needs_rotation() {
        assert!(!Orientation::Normal.needs_rotation());
        assert!(Orientation::Rotate90Cw.needs_rotation());
        assert!(Orientation::Rotate180.needs_rotation());
        assert!(Orientation::Rotate270Cw.needs_rotation());
    }

    #[test]
    fn test_orientation_dimensions_swapped() {
        assert!(!Orientation::Normal.dimensions_swapped());
        assert!(Orientation::Rotate90Cw.dimensions_swapped());
        assert!(!Orientation::Rotate180.dimensions_swapped());
    }

    #[test]
    fn test_dimensions_basic() {
        let dim = Dimensions::new(1920, 1080);
        assert_eq!(dim.pixel_count(), 2_073_600);
        assert!((dim.megapixels() - 2.07).abs() < 0.1);
        assert!((dim.aspect_ratio() - 1.777).abs() < 0.01);
    }

    #[test]
    fn test_dimensions_aspect_ratio_string() {
        let dim = Dimensions::new(1920, 1080);
        assert_eq!(dim.aspect_ratio_string(), "16:9");

        let dim2 = Dimensions::new(4000, 3000);
        assert_eq!(dim2.aspect_ratio_string(), "4:3");
    }

    #[test]
    fn test_dimensions_display_orientation() {
        let dim = Dimensions::new(1920, 1080);
        let swapped = dim.display_dimensions(Orientation::Rotate90Cw);
        assert_eq!(swapped.width, 1080);
        assert_eq!(swapped.height, 1920);
    }

    #[test]
    fn test_gps_coordinates_valid() {
        let gps = GpsCoordinates::new(40.7128, -74.0060).unwrap();
        assert!((gps.latitude - 40.7128).abs() < 0.0001);
        assert!((gps.longitude + 74.0060).abs() < 0.0001);
    }

    #[test]
    fn test_gps_coordinates_invalid_latitude() {
        assert!(GpsCoordinates::new(91.0, 0.0).is_err());
        assert!(GpsCoordinates::new(-91.0, 0.0).is_err());
    }

    #[test]
    fn test_gps_coordinates_invalid_longitude() {
        assert!(GpsCoordinates::new(0.0, 181.0).is_err());
        assert!(GpsCoordinates::new(0.0, -181.0).is_err());
    }

    #[test]
    fn test_gps_from_dms() {
        let gps = GpsCoordinates::from_dms(40.0, 42.0, 46.08, 'N', 74.0, 0.0, 21.6, 'W')
            .unwrap();
        assert!((gps.latitude - 40.7128).abs() < 0.001);
        assert!((gps.longitude + 74.006).abs() < 0.001);
    }

    #[test]
    fn test_gps_display() {
        let gps = GpsCoordinates::new(40.7128, -74.0060)
            .unwrap()
            .with_altitude(10.0);
        let s = format!("{}", gps);
        assert!(s.contains("40.712800"));
        assert!(s.contains("-74.006000"));
        assert!(s.contains("10.0m"));
    }

    #[test]
    fn test_gps_dms_string() {
        let gps = GpsCoordinates::new(40.7128, -74.0060).unwrap();
        let dms = gps.to_dms_string();
        assert!(dms.contains("N"));
        assert!(dms.contains("W"));
    }

    #[test]
    fn test_camera_info_summary() {
        let camera = CameraInfo::new()
            .with_model("Canon EOS R5")
            .with_aperture(2.8)
            .with_shutter_speed("1/250s")
            .with_iso(400);
        let summary = camera.summary();
        assert!(summary.contains("Canon EOS R5"));
        assert!(summary.contains("f/2.8"));
        assert!(summary.contains("ISO 400"));
    }

    #[test]
    fn test_image_format_from_extension() {
        assert_eq!(ImageFormat::from_extension("jpg"), ImageFormat::Jpeg);
        assert_eq!(ImageFormat::from_extension("PNG"), ImageFormat::Png);
        assert_eq!(ImageFormat::from_extension("webp"), ImageFormat::Webp);
        assert_eq!(ImageFormat::from_extension("xyz"), ImageFormat::Unknown);
    }

    #[test]
    fn test_image_format_magic_bytes_jpeg() {
        let jpeg_bytes = [0xFF, 0xD8, 0xFF, 0xE0];
        assert_eq!(ImageFormat::from_magic_bytes(&jpeg_bytes), ImageFormat::Jpeg);
    }

    #[test]
    fn test_image_format_magic_bytes_png() {
        let png_bytes = [0x89, 0x50, 0x4E, 0x47];
        assert_eq!(ImageFormat::from_magic_bytes(&png_bytes), ImageFormat::Png);
    }

    #[test]
    fn test_image_format_mime_type() {
        assert_eq!(ImageFormat::Jpeg.mime_type(), "image/jpeg");
        assert_eq!(ImageFormat::Png.mime_type(), "image/png");
        assert_eq!(ImageFormat::Svg.mime_type(), "image/svg+xml");
    }

    #[test]
    fn test_image_format_supports_exif() {
        assert!(ImageFormat::Jpeg.supports_exif());
        assert!(ImageFormat::Tiff.supports_exif());
        assert!(!ImageFormat::Png.supports_exif());
        assert!(!ImageFormat::Gif.supports_exif());
    }

    #[test]
    fn test_metadata_to_json() {
        let meta = ImageMetadata::new(ImageFormat::Jpeg, Dimensions::new(4000, 3000))
            .with_creation_date("2026-03-09")
            .with_camera(CameraInfo::new().with_model("Test Camera"))
            .with_gps(GpsCoordinates::new(40.7128, -74.0060).unwrap());
        let json = meta.to_json();
        assert!(json.contains("\"format\": \"Jpeg\""));
        assert!(json.contains("\"width\": 4000"));
        assert!(json.contains("\"creationDate\": \"2026-03-09\""));
        assert!(json.contains("\"latitude\":"));
    }

    #[test]
    fn test_metadata_stripped() {
        let meta = ImageMetadata::new(ImageFormat::Jpeg, Dimensions::new(100, 100))
            .with_description("Photo of cat")
            .with_gps(GpsCoordinates::new(0.0, 0.0).unwrap());
        let stripped = meta.stripped();
        assert!(stripped.description.is_none());
        assert!(stripped.gps.is_none());
        assert_eq!(stripped.dimensions.width, 100);
    }

    #[test]
    fn test_metadata_custom_tags() {
        let mut meta = ImageMetadata::new(ImageFormat::Png, Dimensions::new(100, 100));
        meta.set_tag("Artist", "Alice");
        assert_eq!(meta.get_tag("Artist"), Some("Alice"));
        assert_eq!(meta.get_tag("Missing"), None);
    }

    #[test]
    fn test_color_space_display() {
        assert_eq!(ColorSpace::Srgb.to_string(), "sRGB");
        assert_eq!(ColorSpace::Cmyk.to_string(), "CMYK");
    }

    #[test]
    fn test_dimensions_zero() {
        let dim = Dimensions::new(0, 0);
        assert_eq!(dim.pixel_count(), 0);
        assert_eq!(dim.aspect_ratio_string(), "0:0");
    }

    #[test]
    fn test_display_dimensions_with_metadata() {
        let meta = ImageMetadata::new(ImageFormat::Jpeg, Dimensions::new(1920, 1080))
            .with_orientation(Orientation::Rotate90Cw);
        let display = meta.display_dimensions();
        assert_eq!(display.width, 1080);
        assert_eq!(display.height, 1920);
    }
}
