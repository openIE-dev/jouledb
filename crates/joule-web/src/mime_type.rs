//! MIME type detection and handling.
//!
//! Extension-to-MIME mapping (200+ types), MIME-to-extension reverse lookup,
//! content type parsing (with params), MIME matching/negotiation (Accept header),
//! and magic byte detection for common formats.

use std::fmt;

// ── Content Type ────────────────────────────────────────────────

/// A parsed MIME content type with optional parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentType {
    /// The type portion (e.g. "text").
    pub media_type: String,
    /// The subtype portion (e.g. "html").
    pub subtype: String,
    /// Parameters (e.g. charset=utf-8).
    pub params: Vec<(String, String)>,
}

impl ContentType {
    /// Parse a content type string like `text/html; charset=utf-8`.
    pub fn parse(input: &str) -> Option<Self> {
        let input = input.trim();
        let (main, params_str) = if let Some(semi_pos) = input.find(';') {
            (&input[..semi_pos], Some(&input[semi_pos + 1..]))
        } else {
            (input, None)
        };

        let slash_pos = main.find('/')?;
        let media_type = main[..slash_pos].trim().to_lowercase();
        let subtype = main[slash_pos + 1..].trim().to_lowercase();

        let params = if let Some(ps) = params_str {
            ps.split(';').filter_map(|p| {
                let p = p.trim();
                let eq = p.find('=')?;
                let key = p[..eq].trim().to_lowercase();
                let val = p[eq + 1..].trim().trim_matches('"').to_string();
                Some((key, val))
            }).collect()
        } else {
            Vec::new()
        };

        Some(Self { media_type, subtype, params })
    }

    /// Get the full MIME type as `type/subtype`.
    pub fn mime(&self) -> String {
        format!("{}/{}", self.media_type, self.subtype)
    }

    /// Get a parameter value by key.
    pub fn param(&self, key: &str) -> Option<&str> {
        let key_lower = key.to_lowercase();
        self.params.iter()
            .find(|(k, _)| *k == key_lower)
            .map(|(_, v)| v.as_str())
    }

    /// Check if this content type matches a pattern (supports wildcards).
    pub fn matches(&self, pattern: &str) -> bool {
        let pat = pattern.trim().to_lowercase();
        if pat == "*/*" { return true; }
        if let Some(slash_pos) = pat.find('/') {
            let pat_type = &pat[..slash_pos];
            let pat_sub = &pat[slash_pos + 1..];
            if pat_type == "*" || pat_type == self.media_type {
                if pat_sub == "*" || pat_sub == self.subtype {
                    return true;
                }
            }
        }
        false
    }

    /// Is this a text type?
    pub fn is_text(&self) -> bool {
        self.media_type == "text"
    }

    /// Is this an image type?
    pub fn is_image(&self) -> bool {
        self.media_type == "image"
    }

    /// Is this an audio type?
    pub fn is_audio(&self) -> bool {
        self.media_type == "audio"
    }

    /// Is this a video type?
    pub fn is_video(&self) -> bool {
        self.media_type == "video"
    }

    /// Is this a JSON type?
    pub fn is_json(&self) -> bool {
        self.subtype == "json" || self.subtype.ends_with("+json")
    }

    /// Is this an XML type?
    pub fn is_xml(&self) -> bool {
        self.subtype == "xml" || self.subtype.ends_with("+xml")
    }
}

impl fmt::Display for ContentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.media_type, self.subtype)?;
        for (key, val) in &self.params {
            write!(f, "; {}={}", key, val)?;
        }
        Ok(())
    }
}

// ── Extension Database ──────────────────────────────────────────

/// Get the MIME type for a file extension.
pub fn from_extension(ext: &str) -> Option<&'static str> {
    let ext = ext.trim_start_matches('.').to_lowercase();
    EXTENSION_MAP.iter()
        .find(|(e, _)| *e == ext.as_str())
        .map(|(_, m)| *m)
}

/// Get the file extension for a MIME type.
pub fn to_extension(mime: &str) -> Option<&'static str> {
    let mime_lower = mime.to_lowercase();
    EXTENSION_MAP.iter()
        .find(|(_, m)| *m == mime_lower.as_str())
        .map(|(e, _)| *e)
}

/// Get all matching extensions for a MIME type.
pub fn to_extensions(mime: &str) -> Vec<&'static str> {
    let mime_lower = mime.to_lowercase();
    EXTENSION_MAP.iter()
        .filter(|(_, m)| *m == mime_lower.as_str())
        .map(|(e, _)| *e)
        .collect()
}

/// Get the MIME type from a file path (by extension).
pub fn from_path(path: &str) -> Option<&'static str> {
    let dot_pos = path.rfind('.')?;
    let ext = &path[dot_pos + 1..];
    from_extension(ext)
}

// ── Magic Byte Detection ────────────────────────────────────────

/// Detect MIME type from file content using magic bytes.
pub fn from_bytes(data: &[u8]) -> Option<&'static str> {
    if data.len() < 2 { return None; }

    // PNG
    if data.len() >= 8 && data[..8] == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] {
        return Some("image/png");
    }
    // JPEG
    if data.len() >= 3 && data[..3] == [0xFF, 0xD8, 0xFF] {
        return Some("image/jpeg");
    }
    // GIF87a / GIF89a
    if data.len() >= 6 && (data[..6] == *b"GIF87a" || data[..6] == *b"GIF89a") {
        return Some("image/gif");
    }
    // BMP
    if data.len() >= 2 && data[..2] == *b"BM" {
        return Some("image/bmp");
    }
    // WebP
    if data.len() >= 12 && data[..4] == *b"RIFF" && data[8..12] == *b"WEBP" {
        return Some("image/webp");
    }
    // TIFF (little-endian or big-endian)
    if data.len() >= 4 && (data[..4] == [0x49, 0x49, 0x2A, 0x00] || data[..4] == [0x4D, 0x4D, 0x00, 0x2A]) {
        return Some("image/tiff");
    }
    // ICO
    if data.len() >= 4 && data[..4] == [0x00, 0x00, 0x01, 0x00] {
        return Some("image/x-icon");
    }
    // PDF
    if data.len() >= 5 && data[..5] == *b"%PDF-" {
        return Some("application/pdf");
    }
    // ZIP (also docx, xlsx, jar, etc.)
    if data.len() >= 4 && data[..4] == [0x50, 0x4B, 0x03, 0x04] {
        return Some("application/zip");
    }
    // GZIP
    if data.len() >= 2 && data[..2] == [0x1F, 0x8B] {
        return Some("application/gzip");
    }
    // BZIP2
    if data.len() >= 3 && data[..3] == *b"BZh" {
        return Some("application/x-bzip2");
    }
    // 7z
    if data.len() >= 6 && data[..6] == [0x37, 0x7A, 0xBC, 0xAF, 0x27, 0x1C] {
        return Some("application/x-7z-compressed");
    }
    // RAR
    if data.len() >= 7 && data[..7] == *b"Rar!\x1a\x07\x00" {
        return Some("application/x-rar-compressed");
    }
    // OGG
    if data.len() >= 4 && data[..4] == *b"OggS" {
        return Some("audio/ogg");
    }
    // FLAC
    if data.len() >= 4 && data[..4] == *b"fLaC" {
        return Some("audio/flac");
    }
    // WAVE
    if data.len() >= 12 && data[..4] == *b"RIFF" && data[8..12] == *b"WAVE" {
        return Some("audio/wav");
    }
    // MP3 (ID3 tag or frame sync)
    if data.len() >= 3 && data[..3] == *b"ID3" {
        return Some("audio/mpeg");
    }
    if data.len() >= 2 && data[0] == 0xFF && (data[1] & 0xE0) == 0xE0 {
        return Some("audio/mpeg");
    }
    // MP4 / M4A / MOV — ftyp box
    if data.len() >= 8 && data[4..8] == *b"ftyp" {
        return Some("video/mp4");
    }
    // AVI
    if data.len() >= 12 && data[..4] == *b"RIFF" && data[8..12] == *b"AVI " {
        return Some("video/x-msvideo");
    }
    // MKV / WebM (EBML)
    if data.len() >= 4 && data[..4] == [0x1A, 0x45, 0xDF, 0xA3] {
        return Some("video/x-matroska");
    }
    // WASM
    if data.len() >= 4 && data[..4] == [0x00, 0x61, 0x73, 0x6D] {
        return Some("application/wasm");
    }
    // ELF
    if data.len() >= 4 && data[..4] == [0x7F, 0x45, 0x4C, 0x46] {
        return Some("application/x-elf");
    }
    // SQLite
    if data.len() >= 16 && data[..16] == *b"SQLite format 3\x00" {
        return Some("application/x-sqlite3");
    }
    // AVIF
    if data.len() >= 12 && data[4..8] == *b"ftyp" && (data[8..12] == *b"avif" || data[8..12] == *b"avis") {
        return Some("image/avif");
    }

    // Text heuristic — check if mostly ASCII/UTF-8
    let text_chars = data.iter().take(512).filter(|b| {
        b.is_ascii_graphic() || b.is_ascii_whitespace()
    }).count();
    if text_chars > data.len().min(512) * 9 / 10 {
        // Probably text — try to guess
        let start: String = data.iter().take(256).filter_map(|b| {
            if b.is_ascii() { Some(*b as char) } else { None }
        }).collect();
        if start.trim_start().starts_with("<?xml") || start.trim_start().starts_with("<") {
            return Some("text/xml");
        }
        if start.trim_start().starts_with('{') || start.trim_start().starts_with('[') {
            return Some("application/json");
        }
        return Some("text/plain");
    }

    None
}

// ── Accept Header Negotiation ───────────────────────────────────

/// A parsed Accept header entry with quality.
#[derive(Debug, Clone)]
pub struct AcceptEntry {
    pub content_type: ContentType,
    pub quality: f32,
}

/// Parse an Accept header into entries sorted by quality descending.
pub fn parse_accept(header: &str) -> Vec<AcceptEntry> {
    let mut entries: Vec<AcceptEntry> = header.split(',').filter_map(|part| {
        let part = part.trim();
        let (mime_part, quality) = if let Some(q_pos) = part.find(";q=") {
            let q_str = &part[q_pos + 3..];
            let q: f32 = q_str.trim().parse().unwrap_or(1.0);
            (&part[..q_pos], q)
        } else {
            (part, 1.0)
        };
        let ct = ContentType::parse(mime_part)?;
        Some(AcceptEntry { content_type: ct, quality })
    }).collect();

    entries.sort_by(|a, b| b.quality.partial_cmp(&a.quality).unwrap_or(std::cmp::Ordering::Equal));
    entries
}

/// Negotiate the best MIME type from a list of available types
/// and a parsed Accept header.
pub fn negotiate(available: &[&str], accept_header: &str) -> Option<String> {
    let entries = parse_accept(accept_header);
    for entry in &entries {
        for avail in available {
            if let Some(ct) = ContentType::parse(avail) {
                if entry.content_type.matches(&ct.mime()) || ct.matches(&entry.content_type.mime()) {
                    return Some(ct.mime());
                }
            }
        }
    }
    None
}

// ── Extension to MIME database (200+ entries) ───────────────────

static EXTENSION_MAP: &[(&str, &str)] = &[
    // Text
    ("html", "text/html"),
    ("htm", "text/html"),
    ("css", "text/css"),
    ("csv", "text/csv"),
    ("txt", "text/plain"),
    ("text", "text/plain"),
    ("log", "text/plain"),
    ("md", "text/markdown"),
    ("markdown", "text/markdown"),
    ("rtf", "text/rtf"),
    ("xml", "text/xml"),
    ("yaml", "text/yaml"),
    ("yml", "text/yaml"),
    ("ini", "text/plain"),
    ("cfg", "text/plain"),
    ("conf", "text/plain"),
    ("tsv", "text/tab-separated-values"),
    ("ics", "text/calendar"),
    ("vcf", "text/vcard"),
    // JavaScript / TypeScript
    ("js", "text/javascript"),
    ("mjs", "text/javascript"),
    ("cjs", "text/javascript"),
    ("ts", "text/typescript"),
    ("tsx", "text/typescript"),
    ("jsx", "text/javascript"),
    // JSON
    ("json", "application/json"),
    ("jsonld", "application/ld+json"),
    ("geojson", "application/geo+json"),
    ("topojson", "application/json"),
    ("map", "application/json"),
    // Application
    ("pdf", "application/pdf"),
    ("zip", "application/zip"),
    ("gz", "application/gzip"),
    ("gzip", "application/gzip"),
    ("bz2", "application/x-bzip2"),
    ("xz", "application/x-xz"),
    ("7z", "application/x-7z-compressed"),
    ("rar", "application/x-rar-compressed"),
    ("tar", "application/x-tar"),
    ("tgz", "application/gzip"),
    ("jar", "application/java-archive"),
    ("war", "application/java-archive"),
    ("ear", "application/java-archive"),
    ("wasm", "application/wasm"),
    ("doc", "application/msword"),
    ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ("xls", "application/vnd.ms-excel"),
    ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ("ppt", "application/vnd.ms-powerpoint"),
    ("pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
    ("odt", "application/vnd.oasis.opendocument.text"),
    ("ods", "application/vnd.oasis.opendocument.spreadsheet"),
    ("odp", "application/vnd.oasis.opendocument.presentation"),
    ("epub", "application/epub+zip"),
    ("swf", "application/x-shockwave-flash"),
    ("exe", "application/x-msdownload"),
    ("msi", "application/x-msdownload"),
    ("dll", "application/x-msdownload"),
    ("dmg", "application/x-apple-diskimage"),
    ("deb", "application/x-debian-package"),
    ("rpm", "application/x-rpm"),
    ("iso", "application/x-iso9660-image"),
    ("apk", "application/vnd.android.package-archive"),
    ("ipa", "application/octet-stream"),
    ("bin", "application/octet-stream"),
    ("dat", "application/octet-stream"),
    ("pem", "application/x-pem-file"),
    ("p12", "application/x-pkcs12"),
    ("pfx", "application/x-pkcs12"),
    ("crt", "application/x-x509-ca-cert"),
    ("cer", "application/x-x509-ca-cert"),
    ("der", "application/x-x509-ca-cert"),
    ("key", "application/x-pem-file"),
    ("atom", "application/atom+xml"),
    ("rss", "application/rss+xml"),
    ("woff", "font/woff"),
    ("woff2", "font/woff2"),
    ("ttf", "font/ttf"),
    ("otf", "font/otf"),
    ("eot", "application/vnd.ms-fontobject"),
    ("graphql", "application/graphql"),
    ("sql", "application/sql"),
    ("toml", "application/toml"),
    ("proto", "application/protobuf"),
    ("msgpack", "application/msgpack"),
    ("cbor", "application/cbor"),
    ("avro", "application/avro"),
    ("parquet", "application/parquet"),
    // Images
    ("png", "image/png"),
    ("jpg", "image/jpeg"),
    ("jpeg", "image/jpeg"),
    ("jpe", "image/jpeg"),
    ("gif", "image/gif"),
    ("bmp", "image/bmp"),
    ("ico", "image/x-icon"),
    ("svg", "image/svg+xml"),
    ("svgz", "image/svg+xml"),
    ("tif", "image/tiff"),
    ("tiff", "image/tiff"),
    ("webp", "image/webp"),
    ("avif", "image/avif"),
    ("heic", "image/heic"),
    ("heif", "image/heif"),
    ("jxl", "image/jxl"),
    ("raw", "image/x-raw"),
    ("cr2", "image/x-canon-cr2"),
    ("nef", "image/x-nikon-nef"),
    ("psd", "image/vnd.adobe.photoshop"),
    ("ai", "application/postscript"),
    ("eps", "application/postscript"),
    ("ps", "application/postscript"),
    ("xcf", "image/x-xcf"),
    ("cur", "image/x-icon"),
    // Audio
    ("mp3", "audio/mpeg"),
    ("wav", "audio/wav"),
    ("ogg", "audio/ogg"),
    ("oga", "audio/ogg"),
    ("flac", "audio/flac"),
    ("aac", "audio/aac"),
    ("m4a", "audio/mp4"),
    ("wma", "audio/x-ms-wma"),
    ("opus", "audio/opus"),
    ("mid", "audio/midi"),
    ("midi", "audio/midi"),
    ("aiff", "audio/aiff"),
    ("aif", "audio/aiff"),
    ("ra", "audio/x-realaudio"),
    ("amr", "audio/amr"),
    ("weba", "audio/webm"),
    ("ac3", "audio/ac3"),
    ("spx", "audio/ogg"),
    // Video
    ("mp4", "video/mp4"),
    ("m4v", "video/mp4"),
    ("avi", "video/x-msvideo"),
    ("mkv", "video/x-matroska"),
    ("mov", "video/quicktime"),
    ("qt", "video/quicktime"),
    ("wmv", "video/x-ms-wmv"),
    ("flv", "video/x-flv"),
    ("webm", "video/webm"),
    ("mpg", "video/mpeg"),
    ("mpeg", "video/mpeg"),
    ("ogv", "video/ogg"),
    ("3gp", "video/3gpp"),
    ("3g2", "video/3gpp2"),
    ("ts", "video/mp2t"),
    ("m3u8", "application/vnd.apple.mpegurl"),
    // Programming languages
    ("rs", "text/x-rust"),
    ("py", "text/x-python"),
    ("rb", "text/x-ruby"),
    ("java", "text/x-java"),
    ("c", "text/x-c"),
    ("h", "text/x-c"),
    ("cpp", "text/x-c++"),
    ("cxx", "text/x-c++"),
    ("cc", "text/x-c++"),
    ("hpp", "text/x-c++"),
    ("hxx", "text/x-c++"),
    ("cs", "text/x-csharp"),
    ("go", "text/x-go"),
    ("swift", "text/x-swift"),
    ("kt", "text/x-kotlin"),
    ("kts", "text/x-kotlin"),
    ("scala", "text/x-scala"),
    ("r", "text/x-r"),
    ("lua", "text/x-lua"),
    ("php", "text/x-php"),
    ("pl", "text/x-perl"),
    ("pm", "text/x-perl"),
    ("sh", "text/x-shellscript"),
    ("bash", "text/x-shellscript"),
    ("zsh", "text/x-shellscript"),
    ("fish", "text/x-shellscript"),
    ("ps1", "text/x-powershell"),
    ("bat", "text/x-bat"),
    ("cmd", "text/x-bat"),
    ("zig", "text/x-zig"),
    ("nim", "text/x-nim"),
    ("dart", "text/x-dart"),
    ("elm", "text/x-elm"),
    ("ex", "text/x-elixir"),
    ("exs", "text/x-elixir"),
    ("erl", "text/x-erlang"),
    ("hrl", "text/x-erlang"),
    ("hs", "text/x-haskell"),
    ("lhs", "text/x-haskell"),
    ("ml", "text/x-ocaml"),
    ("mli", "text/x-ocaml"),
    ("fs", "text/x-fsharp"),
    ("fsx", "text/x-fsharp"),
    ("clj", "text/x-clojure"),
    ("cljs", "text/x-clojure"),
    ("v", "text/x-v"),
    ("vhdl", "text/x-vhdl"),
    ("vhd", "text/x-vhdl"),
    ("sv", "text/x-systemverilog"),
    ("verilog", "text/x-verilog"),
    // Config & Data
    ("dockerfile", "text/x-dockerfile"),
    ("makefile", "text/x-makefile"),
    ("cmake", "text/x-cmake"),
    ("gradle", "text/x-gradle"),
    ("sbt", "text/x-sbt"),
    ("tf", "text/x-terraform"),
    ("tfvars", "text/x-terraform"),
    ("hcl", "text/x-hcl"),
    ("nix", "text/x-nix"),
    ("dhall", "text/x-dhall"),
    // 3D / CAD
    ("stl", "model/stl"),
    ("obj", "model/obj"),
    ("gltf", "model/gltf+json"),
    ("glb", "model/gltf-binary"),
    ("fbx", "application/octet-stream"),
    ("usdz", "model/vnd.usdz+zip"),
    // Misc
    ("wsdl", "application/wsdl+xml"),
    ("xsd", "application/xml"),
    ("dtd", "application/xml-dtd"),
    ("xsl", "application/xslt+xml"),
    ("xslt", "application/xslt+xml"),
];

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_extension_html() {
        assert_eq!(from_extension("html"), Some("text/html"));
    }

    #[test]
    fn test_from_extension_with_dot() {
        assert_eq!(from_extension(".png"), Some("image/png"));
    }

    #[test]
    fn test_from_extension_case_insensitive() {
        assert_eq!(from_extension("JSON"), Some("application/json"));
    }

    #[test]
    fn test_from_extension_unknown() {
        assert_eq!(from_extension("xyzabc123"), None);
    }

    #[test]
    fn test_to_extension() {
        assert_eq!(to_extension("application/pdf"), Some("pdf"));
    }

    #[test]
    fn test_to_extensions() {
        let exts = to_extensions("text/html");
        assert!(exts.contains(&"html"));
        assert!(exts.contains(&"htm"));
    }

    #[test]
    fn test_from_path() {
        assert_eq!(from_path("document.pdf"), Some("application/pdf"));
        assert_eq!(from_path("/path/to/file.rs"), Some("text/x-rust"));
    }

    #[test]
    fn test_content_type_parse() {
        let ct = ContentType::parse("text/html; charset=utf-8").unwrap();
        assert_eq!(ct.media_type, "text");
        assert_eq!(ct.subtype, "html");
        assert_eq!(ct.param("charset"), Some("utf-8"));
    }

    #[test]
    fn test_content_type_no_params() {
        let ct = ContentType::parse("image/png").unwrap();
        assert_eq!(ct.mime(), "image/png");
        assert!(ct.params.is_empty());
    }

    #[test]
    fn test_content_type_matches() {
        let ct = ContentType::parse("text/html").unwrap();
        assert!(ct.matches("text/html"));
        assert!(ct.matches("text/*"));
        assert!(ct.matches("*/*"));
        assert!(!ct.matches("image/png"));
    }

    #[test]
    fn test_is_text() {
        let ct = ContentType::parse("text/plain").unwrap();
        assert!(ct.is_text());
        assert!(!ct.is_image());
    }

    #[test]
    fn test_is_json() {
        let ct = ContentType::parse("application/json").unwrap();
        assert!(ct.is_json());
        let ct2 = ContentType::parse("application/vnd.api+json").unwrap();
        assert!(ct2.is_json());
    }

    #[test]
    fn test_is_xml() {
        let ct = ContentType::parse("application/xml").unwrap();
        assert!(ct.is_xml());
        let ct2 = ContentType::parse("application/atom+xml").unwrap();
        assert!(ct2.is_xml());
    }

    #[test]
    fn test_content_type_display() {
        let ct = ContentType::parse("text/html; charset=utf-8").unwrap();
        let s = ct.to_string();
        assert!(s.contains("text/html"));
        assert!(s.contains("charset=utf-8"));
    }

    #[test]
    fn test_magic_bytes_png() {
        let data = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00];
        assert_eq!(from_bytes(&data), Some("image/png"));
    }

    #[test]
    fn test_magic_bytes_jpeg() {
        let data = [0xFF, 0xD8, 0xFF, 0xE0, 0x00];
        assert_eq!(from_bytes(&data), Some("image/jpeg"));
    }

    #[test]
    fn test_magic_bytes_gif() {
        assert_eq!(from_bytes(b"GIF89aXX"), Some("image/gif"));
        assert_eq!(from_bytes(b"GIF87aXX"), Some("image/gif"));
    }

    #[test]
    fn test_magic_bytes_pdf() {
        assert_eq!(from_bytes(b"%PDF-1.7 extra"), Some("application/pdf"));
    }

    #[test]
    fn test_magic_bytes_zip() {
        let data = [0x50, 0x4B, 0x03, 0x04, 0x00];
        assert_eq!(from_bytes(&data), Some("application/zip"));
    }

    #[test]
    fn test_magic_bytes_gzip() {
        let data = [0x1F, 0x8B, 0x08, 0x00];
        assert_eq!(from_bytes(&data), Some("application/gzip"));
    }

    #[test]
    fn test_magic_bytes_wasm() {
        let data = [0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00];
        assert_eq!(from_bytes(&data), Some("application/wasm"));
    }

    #[test]
    fn test_magic_bytes_text() {
        let data = b"Hello, this is plain text content.";
        assert_eq!(from_bytes(data), Some("text/plain"));
    }

    #[test]
    fn test_magic_bytes_too_short() {
        assert_eq!(from_bytes(&[0x00]), None);
    }

    #[test]
    fn test_accept_parse() {
        let entries = parse_accept("text/html, application/json;q=0.9, */*;q=0.1");
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].content_type.mime(), "text/html");
        assert_eq!(entries[0].quality, 1.0);
        assert_eq!(entries[1].quality, 0.9);
    }

    #[test]
    fn test_negotiate() {
        let result = negotiate(
            &["application/json", "text/html"],
            "text/html, application/json;q=0.9",
        );
        assert_eq!(result, Some("text/html".to_string()));
    }

    #[test]
    fn test_negotiate_wildcard() {
        let result = negotiate(
            &["application/json"],
            "*/*",
        );
        assert_eq!(result, Some("application/json".to_string()));
    }

    #[test]
    fn test_negotiate_no_match() {
        let result = negotiate(
            &["image/png"],
            "text/html",
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_extension_count() {
        // Verify we have 200+ entries
        assert!(EXTENSION_MAP.len() >= 200);
    }

    #[test]
    fn test_common_extensions() {
        let common = ["html", "css", "js", "json", "png", "jpg", "gif", "svg",
                       "pdf", "zip", "mp3", "mp4", "wasm", "rs", "py", "go"];
        for ext in common {
            assert!(from_extension(ext).is_some(), "missing extension: {}", ext);
        }
    }
}
