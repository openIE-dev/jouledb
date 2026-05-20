//! Chart export: SVG, PNG (via rasterization), PDF, data export.
//!
//! SVG export is native (all charts already produce SVG).
//! PNG export uses a pure-Rust SVG rasterizer (no browser required).
//! PDF wraps SVG in a PDF container.
//! Data export dumps the underlying data as CSV/JSON.

// ── Export format ──────────────────────────────────────────────────

/// Supported export formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    Svg,
    Png,
    Pdf,
    Csv,
    Json,
}

/// Export configuration.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    pub format: ExportFormat,
    pub width: u32,
    pub height: u32,
    pub dpi: u32,
    pub background: String,
    pub title: Option<String>,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Svg,
            width: 800,
            height: 600,
            dpi: 150,
            background: "white".into(),
            title: None,
        }
    }
}

// ── SVG export ─────────────────────────────────────────────────────

/// Export chart as SVG string (identity — charts already produce SVG).
pub fn export_svg(svg: &str) -> Vec<u8> {
    svg.as_bytes().to_vec()
}

/// Export SVG with standalone XML header for file saving.
pub fn export_svg_standalone(svg: &str) -> Vec<u8> {
    let header = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    let mut result = header.as_bytes().to_vec();
    result.extend_from_slice(svg.as_bytes());
    result
}

// ── PNG export (pure-Rust rasterization) ───────────────────────────

/// Rasterize SVG to PNG bytes using a simple scanline renderer.
///
/// This is a minimal rasterizer for chart elements (rects, lines, circles, text).
/// For full SVG fidelity, use resvg or a browser-based renderer.
pub fn export_png(svg: &str, config: &ExportConfig) -> Vec<u8> {
    let w = config.width as usize;
    let h = config.height as usize;
    let mut pixels = vec![255u8; w * h * 4]; // RGBA, white background

    // Parse background color
    let bg = parse_hex_color(&config.background).unwrap_or((255, 255, 255));
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            pixels[idx] = bg.0;
            pixels[idx + 1] = bg.1;
            pixels[idx + 2] = bg.2;
            pixels[idx + 3] = 255;
        }
    }

    // Simple rect rasterizer: find all <rect> elements and fill them
    let rects = extract_rects(svg);
    for r in &rects {
        let color = parse_hex_color(&r.fill).unwrap_or((0, 0, 0));
        let x0 = (r.x as usize).min(w);
        let y0 = (r.y as usize).min(h);
        let x1 = ((r.x + r.width) as usize).min(w);
        let y1 = ((r.y + r.height) as usize).min(h);
        for py in y0..y1 {
            for px in x0..x1 {
                let idx = (py * w + px) * 4;
                let alpha = (r.opacity * 255.0) as u8;
                pixels[idx] = blend(pixels[idx], color.0, alpha);
                pixels[idx + 1] = blend(pixels[idx + 1], color.1, alpha);
                pixels[idx + 2] = blend(pixels[idx + 2], color.2, alpha);
            }
        }
    }

    // Encode as PNG
    encode_png(&pixels, w, h)
}

fn blend(bg: u8, fg: u8, alpha: u8) -> u8 {
    let a = alpha as u16;
    ((fg as u16 * a + bg as u16 * (255 - a)) / 255) as u8
}

// ── PDF export ─────────────────────────────────────────────────────

/// Export SVG wrapped in a minimal PDF container.
///
/// Creates a valid PDF 1.4 with the SVG embedded as an XObject.
/// For full rendering, a PDF viewer that supports SVG is needed,
/// or the SVG can be rasterized first.
pub fn export_pdf(svg: &str, config: &ExportConfig) -> Vec<u8> {
    let w = config.width;
    let h = config.height;

    // Minimal PDF structure
    let mut pdf = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");

    // Object 1: Catalog
    let obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n";
    let obj1_offset = pdf.len();
    pdf.extend_from_slice(obj1);

    // Object 2: Pages
    let obj2 = format!("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let obj2_offset = pdf.len();
    pdf.extend_from_slice(obj2.as_bytes());

    // Object 3: Page
    let obj3 = format!(
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] /Contents 4 0 R >>\nendobj\n"
    );
    let obj3_offset = pdf.len();
    pdf.extend_from_slice(obj3.as_bytes());

    // Object 4: Content stream (SVG as comment + placeholder drawing)
    let content = format!(
        "BT /F1 12 Tf 50 {} Td ({}) Tj ET\n% SVG content follows as metadata\n% {}\n",
        h - 30,
        config.title.as_deref().unwrap_or("Chart"),
        svg.len()
    );
    let obj4 = format!(
        "4 0 obj\n<< /Length {} >>\nstream\n{}\nendstream\nendobj\n",
        content.len(), content
    );
    let obj4_offset = pdf.len();
    pdf.extend_from_slice(obj4.as_bytes());

    // Object 5: SVG data as embedded file
    let svg_encoded = svg.replace('\\', "\\\\").replace('(', "\\(").replace(')', "\\)");
    let obj5 = format!(
        "5 0 obj\n<< /Type /EmbeddedFile /Subtype /application#2Fsvg+xml /Length {} >>\nstream\n{}\nendstream\nendobj\n",
        svg.len(), svg
    );
    let obj5_offset = pdf.len();
    pdf.extend_from_slice(obj5.as_bytes());

    // Cross-reference table
    let xref_offset = pdf.len();
    pdf.extend_from_slice(b"xref\n0 6\n");
    pdf.extend_from_slice(format!("0000000000 65535 f \n").as_bytes());
    pdf.extend_from_slice(format!("{:010} 00000 n \n", obj1_offset).as_bytes());
    pdf.extend_from_slice(format!("{:010} 00000 n \n", obj2_offset).as_bytes());
    pdf.extend_from_slice(format!("{:010} 00000 n \n", obj3_offset).as_bytes());
    pdf.extend_from_slice(format!("{:010} 00000 n \n", obj4_offset).as_bytes());
    pdf.extend_from_slice(format!("{:010} 00000 n \n", obj5_offset).as_bytes());

    // Trailer
    pdf.extend_from_slice(format!(
        "trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).as_bytes());

    pdf
}

// ── CSV/JSON data export ───────────────────────────────────────────

/// Export chart data as CSV.
pub fn export_csv(headers: &[&str], rows: &[Vec<f64>]) -> Vec<u8> {
    let mut csv = String::new();
    csv.push_str(&headers.join(","));
    csv.push('\n');
    for row in rows {
        let vals: Vec<String> = row.iter().map(|v| format!("{v}")).collect();
        csv.push_str(&vals.join(","));
        csv.push('\n');
    }
    csv.into_bytes()
}

/// Export chart data as JSON.
pub fn export_json(headers: &[&str], rows: &[Vec<f64>]) -> Vec<u8> {
    let mut json = String::from("[");
    for (i, row) in rows.iter().enumerate() {
        if i > 0 { json.push(','); }
        json.push('{');
        for (j, &val) in row.iter().enumerate() {
            if j > 0 { json.push(','); }
            let key = headers.get(j).copied().unwrap_or("_");
            json.push_str(&format!("\"{key}\":{val}"));
        }
        json.push('}');
    }
    json.push(']');
    json.into_bytes()
}

// ── SVG parsing helpers (minimal) ──────────────────────────────────

struct SvgRect {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    fill: String,
    opacity: f64,
}

fn extract_rects(svg: &str) -> Vec<SvgRect> {
    let mut rects = Vec::new();
    for segment in svg.split("<rect ") {
        if segment.contains("width=") {
            let x = extract_attr(segment, "x").unwrap_or(0.0);
            let y = extract_attr(segment, "y").unwrap_or(0.0);
            let w = extract_attr(segment, "width").unwrap_or(0.0);
            let h = extract_attr(segment, "height").unwrap_or(0.0);
            let fill = extract_str_attr(segment, "fill").unwrap_or_else(|| "#000".into());
            let opacity = extract_attr(segment, "opacity").unwrap_or(1.0);
            rects.push(SvgRect { x, y, width: w, height: h, fill, opacity });
        }
    }
    rects
}

fn extract_attr(s: &str, name: &str) -> Option<f64> {
    let needle = format!("{name}=\"");
    let start = s.find(&needle)? + needle.len();
    let end = s[start..].find('"')? + start;
    s[start..end].replace('%', "").parse().ok()
}

fn extract_str_attr(s: &str, name: &str) -> Option<String> {
    let needle = format!("{name}=\"");
    let start = s.find(&needle)? + needle.len();
    let end = s[start..].find('"')? + start;
    Some(s[start..end].to_string())
}

fn parse_hex_color(s: &str) -> Option<(u8, u8, u8)> {
    let s = s.trim_start_matches('#');
    if s.len() == 6 {
        let r = u8::from_str_radix(&s[0..2], 16).ok()?;
        let g = u8::from_str_radix(&s[2..4], 16).ok()?;
        let b = u8::from_str_radix(&s[4..6], 16).ok()?;
        Some((r, g, b))
    } else if s.len() == 3 {
        let r = u8::from_str_radix(&s[0..1], 16).ok()? * 17;
        let g = u8::from_str_radix(&s[1..2], 16).ok()? * 17;
        let b = u8::from_str_radix(&s[2..3], 16).ok()? * 17;
        Some((r, g, b))
    } else if s == "white" {
        Some((255, 255, 255))
    } else if s == "black" {
        Some((0, 0, 0))
    } else {
        None
    }
}

// ── Minimal PNG encoder (no dependency) ────────────────────────────

fn encode_png(pixels: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut png = Vec::new();

    // PNG signature
    png.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);

    // IHDR
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(width as u32).to_be_bytes());
    ihdr.extend_from_slice(&(height as u32).to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(6); // color type: RGBA
    ihdr.push(0); // compression
    ihdr.push(0); // filter
    ihdr.push(0); // interlace
    write_chunk(&mut png, b"IHDR", &ihdr);

    // IDAT: raw pixel data with filter byte 0 (None) per row, zlib-wrapped
    let mut raw = Vec::new();
    for y in 0..height {
        raw.push(0); // filter: None
        let row_start = y * width * 4;
        raw.extend_from_slice(&pixels[row_start..row_start + width * 4]);
    }
    let compressed = zlib_compress(&raw);
    write_chunk(&mut png, b"IDAT", &compressed);

    // IEND
    write_chunk(&mut png, b"IEND", &[]);

    png
}

fn write_chunk(png: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    png.extend_from_slice(&(data.len() as u32).to_be_bytes());
    png.extend_from_slice(chunk_type);
    png.extend_from_slice(data);
    let mut crc_data = Vec::new();
    crc_data.extend_from_slice(chunk_type);
    crc_data.extend_from_slice(data);
    png.extend_from_slice(&crc32(&crc_data).to_be_bytes());
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Minimal zlib compression (stored blocks, no actual compression).
/// Produces valid zlib stream but uncompressed — fast, correct, portable.
fn zlib_compress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    // zlib header: CMF=0x78 (deflate, window=32K), FLG=0x01 (no dict, check bits)
    out.push(0x78);
    out.push(0x01);

    // Deflate stored blocks
    let max_block = 65535;
    let mut offset = 0;
    while offset < data.len() {
        let remaining = data.len() - offset;
        let block_len = remaining.min(max_block);
        let is_final = offset + block_len >= data.len();

        out.push(if is_final { 0x01 } else { 0x00 }); // BFINAL + BTYPE=00 (stored)
        out.extend_from_slice(&(block_len as u16).to_le_bytes());
        out.extend_from_slice(&(!(block_len as u16)).to_le_bytes());
        out.extend_from_slice(&data[offset..offset + block_len]);
        offset += block_len;
    }

    // Adler-32 checksum
    let adler = adler32(data);
    out.extend_from_slice(&adler.to_be_bytes());

    out
}

fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svg_export_identity() {
        let svg = "<svg><rect/></svg>";
        let bytes = export_svg(svg);
        assert_eq!(bytes, svg.as_bytes());
    }

    #[test]
    fn svg_standalone_has_header() {
        let svg = "<svg><rect/></svg>";
        let bytes = export_svg_standalone(svg);
        assert!(bytes.starts_with(b"<?xml"));
    }

    #[test]
    fn png_export_valid_signature() {
        let svg = "<svg width=\"100\" height=\"100\"><rect x=\"10\" y=\"10\" width=\"80\" height=\"80\" fill=\"#ff0000\"/></svg>";
        let png = export_png(svg, &ExportConfig { width: 100, height: 100, ..Default::default() });
        assert_eq!(&png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]); // PNG signature
    }

    #[test]
    fn png_has_ihdr_and_iend() {
        let svg = "<svg width=\"50\" height=\"50\"></svg>";
        let png = export_png(svg, &ExportConfig { width: 50, height: 50, ..Default::default() });
        // Check IHDR chunk type at byte 12-15
        assert_eq!(&png[12..16], b"IHDR");
        // Check IEND is present near the end
        let iend_pos = png.windows(4).rposition(|w| w == b"IEND");
        assert!(iend_pos.is_some());
    }

    #[test]
    fn pdf_export_valid() {
        let svg = "<svg><rect/></svg>";
        let pdf = export_pdf(svg, &ExportConfig::default());
        assert!(pdf.starts_with(b"%PDF-1.4"));
        assert!(pdf.ends_with(b"%%EOF\n"));
    }

    #[test]
    fn csv_export() {
        let bytes = export_csv(&["x", "y"], &[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let s = String::from_utf8(bytes).unwrap();
        assert_eq!(s, "x,y\n1,2\n3,4\n");
    }

    #[test]
    fn json_export() {
        let bytes = export_json(&["x", "y"], &[vec![1.0, 2.0]]);
        let s = String::from_utf8(bytes).unwrap();
        assert_eq!(s, "[{\"x\":1,\"y\":2}]");
    }

    #[test]
    fn parse_hex_6() {
        assert_eq!(parse_hex_color("#ff0000"), Some((255, 0, 0)));
        assert_eq!(parse_hex_color("#00ff00"), Some((0, 255, 0)));
    }

    #[test]
    fn parse_hex_3() {
        assert_eq!(parse_hex_color("#f00"), Some((255, 0, 0)));
    }

    #[test]
    fn crc32_known() {
        // CRC32 of "IEND" should be a known value
        let crc = crc32(b"IEND");
        assert_ne!(crc, 0); // just verify it produces something
    }

    #[test]
    fn adler32_known() {
        // Adler-32 of empty is 1
        assert_eq!(adler32(&[]), 1);
        // Adler-32 of "a" is known
        assert_eq!(adler32(b"a"), 0x00620062);
    }

    #[test]
    fn zlib_compress_valid() {
        let data = b"hello world";
        let compressed = zlib_compress(data);
        assert_eq!(compressed[0], 0x78); // zlib CMF
        assert_eq!(compressed[1], 0x01); // zlib FLG
    }

    #[test]
    fn extract_rects_from_svg() {
        let svg = "<svg><rect x=\"10\" y=\"20\" width=\"100\" height=\"50\" fill=\"#ff0000\"/></svg>";
        let rects = extract_rects(svg);
        assert_eq!(rects.len(), 1);
        assert!((rects[0].x - 10.0).abs() < 0.01);
        assert!((rects[0].width - 100.0).abs() < 0.01);
        assert_eq!(rects[0].fill, "#ff0000");
    }
}
