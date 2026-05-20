//! Asset processing pipeline — fingerprinting, inlining, and manifest generation.
//!
//! Replaces Webpack asset modules / Vite asset handling with a pure Rust pipeline.
//! Processors chain together to transform assets before they reach the output.

use std::collections::HashMap;
use std::fmt;

// ── Asset ───────────────────────────────────────────────────────

/// An asset flowing through the pipeline.
#[derive(Debug, Clone)]
pub struct Asset {
    pub path: String,
    pub content_type: String,
    pub content: Vec<u8>,
    pub fingerprint: Option<String>,
}

impl Asset {
    pub fn new(
        path: impl Into<String>,
        content_type: impl Into<String>,
        content: Vec<u8>,
    ) -> Self {
        Self {
            path: path.into(),
            content_type: content_type.into(),
            content,
            fingerprint: None,
        }
    }

    /// Compute content hash (FNV-1a) and store as fingerprint.
    pub fn compute_fingerprint(&mut self) {
        let hash = fnv1a(&self.content);
        self.fingerprint = Some(format!("{:016x}", hash));
    }

    /// Generate fingerprinted filename: `name.HASH.ext`.
    pub fn fingerprinted_path(&self) -> String {
        match &self.fingerprint {
            Some(fp) => {
                if let Some(dot) = self.path.rfind('.') {
                    format!("{}.{}{}", &self.path[..dot], fp, &self.path[dot..])
                } else {
                    format!("{}.{}", self.path, fp)
                }
            }
            None => self.path.clone(),
        }
    }

    /// Size in bytes.
    pub fn size(&self) -> usize {
        self.content.len()
    }

    /// Encode as a data URI (for inlining small assets).
    pub fn as_data_uri(&self) -> String {
        let b64 = base64_encode(&self.content);
        format!("data:{};base64,{}", self.content_type, b64)
    }
}

fn fnv1a(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    let mut i = 0;
    while i + 2 < data.len() {
        let n = (u32::from(data[i]) << 16) | (u32::from(data[i + 1]) << 8) | u32::from(data[i + 2]);
        out.push(char::from(CHARS[(n >> 18 & 0x3f) as usize]));
        out.push(char::from(CHARS[(n >> 12 & 0x3f) as usize]));
        out.push(char::from(CHARS[(n >> 6 & 0x3f) as usize]));
        out.push(char::from(CHARS[(n & 0x3f) as usize]));
        i += 3;
    }
    let rem = data.len() - i;
    if rem == 1 {
        let n = u32::from(data[i]) << 16;
        out.push(char::from(CHARS[(n >> 18 & 0x3f) as usize]));
        out.push(char::from(CHARS[(n >> 12 & 0x3f) as usize]));
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let n = (u32::from(data[i]) << 16) | (u32::from(data[i + 1]) << 8);
        out.push(char::from(CHARS[(n >> 18 & 0x3f) as usize]));
        out.push(char::from(CHARS[(n >> 12 & 0x3f) as usize]));
        out.push(char::from(CHARS[(n >> 6 & 0x3f) as usize]));
        out.push('=');
    }
    out
}

// ── AssetProcessor trait ────────────────────────────────────────

/// A processor that transforms an asset.
pub trait AssetProcessor: fmt::Debug {
    /// Human-readable name.
    fn name(&self) -> &str;

    /// Process the asset, returning a (possibly modified) asset.
    fn process(&self, asset: Asset) -> Asset;
}

// ── Built-in Processors ─────────────────────────────────────────

/// Adds a content-hash fingerprint to the asset.
#[derive(Debug)]
pub struct FingerprintProcessor;

impl AssetProcessor for FingerprintProcessor {
    fn name(&self) -> &str {
        "fingerprint"
    }

    fn process(&self, mut asset: Asset) -> Asset {
        asset.compute_fingerprint();
        asset
    }
}

/// Placeholder for image optimization (marks content_type, doesn't alter bytes).
#[derive(Debug)]
pub struct ImageOptimizer {
    pub quality: u8,
}

impl AssetProcessor for ImageOptimizer {
    fn name(&self) -> &str {
        "image-optimizer"
    }

    fn process(&self, mut asset: Asset) -> Asset {
        // In a real implementation this would re-encode the image.
        // Here we annotate the content type to indicate optimization was applied.
        if asset.content_type.starts_with("image/") {
            // Simulate: append quality marker to path metadata
            asset.content_type = format!("{}; q={}", asset.content_type, self.quality);
        }
        asset
    }
}

/// Placeholder for font subsetting.
#[derive(Debug)]
pub struct FontSubsetter {
    pub charset: String,
}

impl AssetProcessor for FontSubsetter {
    fn name(&self) -> &str {
        "font-subsetter"
    }

    fn process(&self, asset: Asset) -> Asset {
        // In a real implementation this would subset the font to the given charset.
        asset
    }
}

/// Inlines small assets as data URIs if below the threshold.
#[derive(Debug)]
pub struct InlineSmallAssets {
    pub threshold_bytes: usize,
}

impl AssetProcessor for InlineSmallAssets {
    fn name(&self) -> &str {
        "inline-small"
    }

    fn process(&self, mut asset: Asset) -> Asset {
        if asset.content.len() <= self.threshold_bytes {
            let data_uri = asset.as_data_uri();
            // Replace content with the data URI bytes and mark path
            asset.content = data_uri.into_bytes();
            asset.path = format!("inline:{}", asset.path);
        }
        asset
    }
}

// ── Pipeline ────────────────────────────────────────────────────

/// Chains multiple processors.
#[derive(Debug, Default)]
pub struct Pipeline {
    processors: Vec<Box<dyn AssetProcessor>>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_processor(&mut self, processor: Box<dyn AssetProcessor>) {
        self.processors.push(processor);
    }

    /// Process a single asset through all processors.
    pub fn process(&self, asset: Asset) -> Asset {
        let mut current = asset;
        for p in &self.processors {
            current = p.process(current);
        }
        current
    }

    /// Process multiple assets and generate a manifest.
    pub fn process_all(&self, assets: Vec<Asset>) -> (Vec<Asset>, AssetManifest) {
        let mut processed = Vec::with_capacity(assets.len());
        let mut manifest = AssetManifest::new();

        for asset in assets {
            let original_path = asset.path.clone();
            let result = self.process(asset);
            let output_path = result.fingerprinted_path();
            manifest.entries.insert(original_path, output_path);
            processed.push(result);
        }

        (processed, manifest)
    }

    pub fn processor_count(&self) -> usize {
        self.processors.len()
    }
}

// ── Asset Manifest ──────────────────────────────────────────────

/// Maps original asset paths to their fingerprinted output paths.
#[derive(Debug, Clone, Default)]
pub struct AssetManifest {
    pub entries: HashMap<String, String>,
}

impl AssetManifest {
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up the output path for an original path.
    pub fn resolve(&self, original: &str) -> Option<&str> {
        self.entries.get(original).map(|s| s.as_str())
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\n");
        let mut entries: Vec<(&String, &String)> = self.entries.iter().collect();
        entries.sort_by_key(|(k, _)| k.as_str());
        for (i, (k, v)) in entries.iter().enumerate() {
            out.push_str(&format!("  \"{k}\": \"{v}\""));
            if i + 1 < entries.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push('}');
        out
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_fingerprint_deterministic() {
        let mut a = Asset::new("style.css", "text/css", b"body{}".to_vec());
        a.compute_fingerprint();
        let fp1 = a.fingerprint.clone().unwrap();
        a.compute_fingerprint();
        let fp2 = a.fingerprint.unwrap();
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn fingerprinted_path_format() {
        let mut a = Asset::new("app.js", "application/javascript", b"code".to_vec());
        a.compute_fingerprint();
        let path = a.fingerprinted_path();
        assert!(path.starts_with("app."));
        assert!(path.ends_with(".js"));
        assert!(path.len() > "app.js".len());
    }

    #[test]
    fn fingerprinted_path_no_extension() {
        let mut a = Asset::new("LICENSE", "text/plain", b"MIT".to_vec());
        a.compute_fingerprint();
        let path = a.fingerprinted_path();
        assert!(path.starts_with("LICENSE."));
    }

    #[test]
    fn data_uri_generation() {
        let a = Asset::new("icon.png", "image/png", vec![0x89, 0x50, 0x4e, 0x47]);
        let uri = a.as_data_uri();
        assert!(uri.starts_with("data:image/png;base64,"));
    }

    #[test]
    fn inline_small_assets_below_threshold() {
        let p = InlineSmallAssets {
            threshold_bytes: 100,
        };
        let a = Asset::new("tiny.svg", "image/svg+xml", b"<svg/>".to_vec());
        let result = p.process(a);
        assert!(result.path.starts_with("inline:"));
    }

    #[test]
    fn inline_large_assets_untouched() {
        let p = InlineSmallAssets {
            threshold_bytes: 2,
        };
        let a = Asset::new("big.png", "image/png", vec![0; 1000]);
        let result = p.process(a);
        assert_eq!(result.path, "big.png");
    }

    #[test]
    fn pipeline_chains_processors() {
        let mut pipe = Pipeline::new();
        pipe.add_processor(Box::new(FingerprintProcessor));
        assert_eq!(pipe.processor_count(), 1);
        let a = Asset::new("main.js", "application/javascript", b"hello()".to_vec());
        let result = pipe.process(a);
        assert!(result.fingerprint.is_some());
    }

    #[test]
    fn manifest_generation() {
        let mut pipe = Pipeline::new();
        pipe.add_processor(Box::new(FingerprintProcessor));
        let assets = vec![
            Asset::new("a.js", "application/javascript", b"aaa".to_vec()),
            Asset::new("b.css", "text/css", b"bbb".to_vec()),
        ];
        let (_, manifest) = pipe.process_all(assets);
        assert_eq!(manifest.entries.len(), 2);
        assert!(manifest.resolve("a.js").is_some());
        assert!(manifest.resolve("b.css").is_some());
    }

    #[test]
    fn manifest_to_json() {
        let mut manifest = AssetManifest::new();
        manifest
            .entries
            .insert("app.js".into(), "app.abc123.js".into());
        let json = manifest.to_json();
        assert!(json.contains("app.js"));
        assert!(json.contains("app.abc123.js"));
    }

    #[test]
    fn image_optimizer_annotates() {
        let p = ImageOptimizer { quality: 80 };
        let a = Asset::new("photo.jpg", "image/jpeg", vec![0xff, 0xd8]);
        let result = p.process(a);
        assert!(result.content_type.contains("q=80"));
    }

    #[test]
    fn font_subsetter_passthrough() {
        let p = FontSubsetter {
            charset: "latin".into(),
        };
        let a = Asset::new("font.woff2", "font/woff2", vec![0; 50]);
        let result = p.process(a);
        assert_eq!(result.content.len(), 50);
    }

    #[test]
    fn base64_encode_empty() {
        assert_eq!(base64_encode(&[]), "");
    }

    #[test]
    fn base64_roundtrip_simple() {
        let encoded = base64_encode(b"Hello");
        assert_eq!(encoded, "SGVsbG8=");
    }
}
