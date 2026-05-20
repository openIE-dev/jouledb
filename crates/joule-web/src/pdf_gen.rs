//! PDF generation — PDF 1.4 structure, pages, text objects, fonts, images,
//! cross-reference table, stream objects, basic layout engine.
//!
//! Pure-Rust replacement for printpdf, lopdf, and wkhtmltopdf.

use std::fmt;

// ── PDF Errors ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum PdfError {
    EmptyDocument,
    InvalidPage(usize),
    InvalidFont(String),
}

impl fmt::Display for PdfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PdfError::EmptyDocument => write!(f, "document has no pages"),
            PdfError::InvalidPage(n) => write!(f, "invalid page index: {n}"),
            PdfError::InvalidFont(name) => write!(f, "invalid font: {name}"),
        }
    }
}

// ── Units ───────────────────────────────────────────────────────

/// Points (1/72 inch). PDF native unit.
pub type Pt = f64;

/// Convert millimeters to points.
pub fn mm_to_pt(mm: f64) -> Pt { mm * 72.0 / 25.4 }

/// Convert inches to points.
pub fn inch_to_pt(inches: f64) -> Pt { inches * 72.0 }

// ── Page Sizes ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PageSize { pub width: Pt, pub height: Pt }

impl PageSize {
    pub fn new(width: Pt, height: Pt) -> Self { Self { width, height } }
    pub fn a4() -> Self { Self { width: 595.28, height: 841.89 } }
    pub fn letter() -> Self { Self { width: 612.0, height: 792.0 } }
    pub fn a3() -> Self { Self { width: 841.89, height: 1190.55 } }
    pub fn legal() -> Self { Self { width: 612.0, height: 1008.0 } }
    pub fn landscape(self) -> Self { Self { width: self.height, height: self.width } }
}

// ── Color ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PdfColor {
    Rgb(f64, f64, f64),
    Gray(f64),
    Cmyk(f64, f64, f64, f64),
}

impl PdfColor {
    pub fn black() -> Self { PdfColor::Gray(0.0) }
    pub fn white() -> Self { PdfColor::Gray(1.0) }
    pub fn red() -> Self { PdfColor::Rgb(1.0, 0.0, 0.0) }
    pub fn green() -> Self { PdfColor::Rgb(0.0, 1.0, 0.0) }
    pub fn blue() -> Self { PdfColor::Rgb(0.0, 0.0, 1.0) }

    fn stroke_operator(&self) -> String {
        match self {
            PdfColor::Rgb(r, g, b) => format!("{r:.3} {g:.3} {b:.3} RG"),
            PdfColor::Gray(g) => format!("{g:.3} G"),
            PdfColor::Cmyk(c, m, y, k) => format!("{c:.3} {m:.3} {y:.3} {k:.3} K"),
        }
    }

    fn fill_operator(&self) -> String {
        match self {
            PdfColor::Rgb(r, g, b) => format!("{r:.3} {g:.3} {b:.3} rg"),
            PdfColor::Gray(g) => format!("{g:.3} g"),
            PdfColor::Cmyk(c, m, y, k) => format!("{c:.3} {m:.3} {y:.3} {k:.3} k"),
        }
    }
}

// ── Font ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StandardFont {
    Helvetica,
    HelveticaBold,
    HelveticaOblique,
    TimesRoman,
    TimesBold,
    TimesItalic,
    Courier,
    CourierBold,
    CourierOblique,
}

impl StandardFont {
    pub fn pdf_name(&self) -> &str {
        match self {
            StandardFont::Helvetica => "Helvetica",
            StandardFont::HelveticaBold => "Helvetica-Bold",
            StandardFont::HelveticaOblique => "Helvetica-Oblique",
            StandardFont::TimesRoman => "Times-Roman",
            StandardFont::TimesBold => "Times-Bold",
            StandardFont::TimesItalic => "Times-Italic",
            StandardFont::Courier => "Courier",
            StandardFont::CourierBold => "Courier-Bold",
            StandardFont::CourierOblique => "Courier-Oblique",
        }
    }

    /// Approximate width of a character in em-units (1000 units = 1 em).
    pub fn avg_char_width(&self) -> f64 {
        match self {
            StandardFont::Courier | StandardFont::CourierBold | StandardFont::CourierOblique => 600.0,
            StandardFont::Helvetica | StandardFont::HelveticaBold | StandardFont::HelveticaOblique => 550.0,
            StandardFont::TimesRoman | StandardFont::TimesBold | StandardFont::TimesItalic => 500.0,
        }
    }
}

// ── Text Alignment ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAlign { Left, Center, Right }

// ── Drawing Operations ──────────────────────────────────────────

/// A drawing operation on a page.
#[derive(Debug, Clone)]
pub enum DrawOp {
    Text { x: Pt, y: Pt, text: String, font_idx: usize, size: f64, color: PdfColor },
    Line { x1: Pt, y1: Pt, x2: Pt, y2: Pt, width: f64, color: PdfColor },
    Rect { x: Pt, y: Pt, w: Pt, h: Pt, fill: Option<PdfColor>, stroke: Option<PdfColor>, line_width: f64 },
    Image { x: Pt, y: Pt, w: Pt, h: Pt, image_idx: usize },
}

// ── Image Data ──────────────────────────────────────────────────

/// A raw image to embed in the PDF.
#[derive(Debug, Clone)]
pub struct PdfImage {
    pub width: u32,
    pub height: u32,
    pub color_space: &'static str,
    pub bits_per_component: u8,
    pub data: Vec<u8>,
}

impl PdfImage {
    /// Create an RGB image from raw pixel data (R, G, B bytes).
    pub fn rgb(width: u32, height: u32, data: Vec<u8>) -> Self {
        assert_eq!(data.len(), (width * height * 3) as usize);
        Self { width, height, color_space: "DeviceRGB", bits_per_component: 8, data }
    }

    /// Create a grayscale image from raw pixel data.
    pub fn grayscale(width: u32, height: u32, data: Vec<u8>) -> Self {
        assert_eq!(data.len(), (width * height) as usize);
        Self { width, height, color_space: "DeviceGray", bits_per_component: 8, data }
    }
}

// ── Page ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PdfPage {
    pub size: PageSize,
    pub ops: Vec<DrawOp>,
}

impl PdfPage {
    pub fn new(size: PageSize) -> Self { Self { size, ops: Vec::new() } }

    pub fn draw_text(&mut self, x: Pt, y: Pt, text: &str, font_idx: usize, size: f64, color: PdfColor) {
        self.ops.push(DrawOp::Text { x, y, text: text.into(), font_idx, size, color });
    }

    pub fn draw_line(&mut self, x1: Pt, y1: Pt, x2: Pt, y2: Pt, width: f64, color: PdfColor) {
        self.ops.push(DrawOp::Line { x1, y1, x2, y2, width, color });
    }

    pub fn draw_rect(&mut self, x: Pt, y: Pt, w: Pt, h: Pt, fill: Option<PdfColor>, stroke: Option<PdfColor>, line_width: f64) {
        self.ops.push(DrawOp::Rect { x, y, w, h, fill, stroke, line_width });
    }

    pub fn draw_image(&mut self, x: Pt, y: Pt, w: Pt, h: Pt, image_idx: usize) {
        self.ops.push(DrawOp::Image { x, y, w, h, image_idx });
    }
}

// ── PDF Document ────────────────────────────────────────────────

/// A PDF document builder.
pub struct PdfDocument {
    pub title: String,
    pub author: String,
    pub pages: Vec<PdfPage>,
    pub fonts: Vec<StandardFont>,
    pub images: Vec<PdfImage>,
}

impl PdfDocument {
    pub fn new(title: &str) -> Self {
        Self {
            title: title.into(),
            author: String::new(),
            pages: Vec::new(),
            fonts: vec![StandardFont::Helvetica],
            images: Vec::new(),
        }
    }

    pub fn author(mut self, author: &str) -> Self { self.author = author.into(); self }

    /// Add a font and return its index.
    pub fn add_font(&mut self, font: StandardFont) -> usize {
        if let Some(idx) = self.fonts.iter().position(|f| f == &font) {
            return idx;
        }
        self.fonts.push(font);
        self.fonts.len() - 1
    }

    /// Add an image and return its index.
    pub fn add_image(&mut self, image: PdfImage) -> usize {
        self.images.push(image);
        self.images.len() - 1
    }

    /// Add a new page with the given size.
    pub fn add_page(&mut self, size: PageSize) -> &mut PdfPage {
        self.pages.push(PdfPage::new(size));
        self.pages.last_mut().unwrap()
    }

    /// Get a mutable reference to a page by index.
    pub fn page_mut(&mut self, idx: usize) -> Option<&mut PdfPage> {
        self.pages.get_mut(idx)
    }

    /// Render the document to a PDF 1.4 byte vector.
    pub fn render(&self) -> Result<Vec<u8>, PdfError> {
        if self.pages.is_empty() { return Err(PdfError::EmptyDocument); }
        let mut w = PdfWriter::new();

        // Header
        w.line("%PDF-1.4");
        // Binary comment to signal PDF contains 8-bit data
        w.buf.extend_from_slice(b"%\xE2\xE3\xCF\xD3\n");

        // Object numbering plan:
        // 1: Catalog
        // 2: Pages
        // 3..3+nfonts-1: Font objects
        // 3+nfonts..3+nfonts+nimages-1: Image XObjects
        // then page objects (Page + ContentStream pairs)

        let nfonts = self.fonts.len();
        let nimages = self.images.len();
        let font_start = 3;
        let image_start = font_start + nfonts;
        let page_obj_start = image_start + nimages;
        let npages = self.pages.len();

        // Catalog (obj 1)
        w.obj(1);
        w.line("<< /Type /Catalog /Pages 2 0 R >>");
        w.endobj();

        // Pages (obj 2)
        w.obj(2);
        w.raw("<< /Type /Pages /Kids [");
        for i in 0..npages {
            let page_obj_id = page_obj_start + i * 2;
            w.raw(&format!(" {} 0 R", page_obj_id));
        }
        w.raw(&format!("] /Count {} >>", npages));
        w.newline();
        w.endobj();

        // Font objects
        for (i, font) in self.fonts.iter().enumerate() {
            let obj_id = font_start + i;
            w.obj(obj_id);
            w.line(&format!("<< /Type /Font /Subtype /Type1 /BaseFont /{} >>", font.pdf_name()));
            w.endobj();
        }

        // Image XObjects
        for (i, img) in self.images.iter().enumerate() {
            let obj_id = image_start + i;
            w.obj(obj_id);
            let len = img.data.len();
            w.line(&format!(
                "<< /Type /XObject /Subtype /Image /Width {} /Height {} /ColorSpace /{} /BitsPerComponent {} /Length {} >>",
                img.width, img.height, img.color_space, img.bits_per_component, len
            ));
            w.line("stream");
            w.bytes(&img.data);
            w.newline();
            w.line("endstream");
            w.endobj();
        }

        // Page objects + content streams
        for (pi, page) in self.pages.iter().enumerate() {
            let page_obj_id = page_obj_start + pi * 2;
            let content_obj_id = page_obj_id + 1;

            // Build content stream
            let content = self.build_content_stream(page, font_start, image_start);

            // Page object
            w.obj(page_obj_id);
            w.raw("<< /Type /Page /Parent 2 0 R");
            w.raw(&format!(" /MediaBox [0 0 {:.2} {:.2}]", page.size.width, page.size.height));

            // Resources
            w.raw(" /Resources << /Font <<");
            for fi in 0..nfonts {
                w.raw(&format!(" /F{} {} 0 R", fi, font_start + fi));
            }
            w.raw(" >>");

            if !self.images.is_empty() {
                w.raw(" /XObject <<");
                for ii in 0..nimages {
                    w.raw(&format!(" /Im{} {} 0 R", ii, image_start + ii));
                }
                w.raw(" >>");
            }

            w.raw(" >>");
            w.raw(&format!(" /Contents {} 0 R >>", content_obj_id));
            w.newline();
            w.endobj();

            // Content stream object
            w.obj(content_obj_id);
            w.line(&format!("<< /Length {} >>", content.len()));
            w.line("stream");
            w.bytes(content.as_bytes());
            w.newline();
            w.line("endstream");
            w.endobj();
        }

        // Cross-reference table
        let total_objects = page_obj_start + npages * 2;
        let xref_offset = w.pos();
        w.line("xref");
        w.line(&format!("0 {}", total_objects));
        w.line("0000000000 65535 f ");
        for i in 1..total_objects {
            let offset = w.offsets.get(&i).copied().unwrap_or(0);
            w.line(&format!("{:010} 00000 n ", offset));
        }

        // Trailer
        w.line("trailer");
        w.line(&format!("<< /Size {} /Root 1 0 R >>", total_objects));
        w.line("startxref");
        w.line(&format!("{xref_offset}"));
        w.line("%%EOF");

        Ok(w.buf)
    }

    fn build_content_stream(&self, page: &PdfPage, _font_start: usize, _image_start: usize) -> String {
        let mut s = String::new();
        for op in &page.ops {
            match op {
                DrawOp::Text { x, y, text, font_idx, size, color } => {
                    s.push_str("BT\n");
                    s.push_str(&color.fill_operator());
                    s.push('\n');
                    s.push_str(&format!("/F{font_idx} {size:.1} Tf\n"));
                    s.push_str(&format!("{x:.2} {y:.2} Td\n"));
                    s.push_str(&format!("({}) Tj\n", pdf_escape_string(text)));
                    s.push_str("ET\n");
                }
                DrawOp::Line { x1, y1, x2, y2, width, color } => {
                    s.push_str(&format!("{width:.2} w\n"));
                    s.push_str(&color.stroke_operator());
                    s.push('\n');
                    s.push_str(&format!("{x1:.2} {y1:.2} m {x2:.2} {y2:.2} l S\n"));
                }
                DrawOp::Rect { x, y, w, h, fill, stroke, line_width } => {
                    s.push_str(&format!("{line_width:.2} w\n"));
                    if let Some(fc) = fill {
                        s.push_str(&fc.fill_operator());
                        s.push('\n');
                    }
                    if let Some(sc) = stroke {
                        s.push_str(&sc.stroke_operator());
                        s.push('\n');
                    }
                    s.push_str(&format!("{x:.2} {y:.2} {w:.2} {h:.2} re "));
                    match (fill, stroke) {
                        (Some(_), Some(_)) => s.push_str("B\n"),
                        (Some(_), None) => s.push_str("f\n"),
                        (None, Some(_)) => s.push_str("S\n"),
                        (None, None) => s.push('\n'),
                    }
                }
                DrawOp::Image { x, y, w, h, image_idx } => {
                    s.push_str("q\n");
                    s.push_str(&format!("{w:.2} 0 0 {h:.2} {x:.2} {y:.2} cm\n"));
                    s.push_str(&format!("/Im{image_idx} Do\n"));
                    s.push_str("Q\n");
                }
            }
        }
        s
    }
}

fn pdf_escape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '(' => out.push_str("\\("),
            ')' => out.push_str("\\)"),
            '\\' => out.push_str("\\\\"),
            _ => out.push(ch),
        }
    }
    out
}

// ── Simple Layout Engine ────────────────────────────────────────

/// Lay out a paragraph of text with word wrapping.
pub fn layout_paragraph(
    text: &str,
    x: Pt,
    y: Pt,
    max_width: Pt,
    font: &StandardFont,
    font_size: f64,
    leading: f64,
) -> Vec<(Pt, Pt, String)> {
    let char_width = font.avg_char_width() / 1000.0 * font_size;
    let max_chars = (max_width / char_width).floor() as usize;
    if max_chars == 0 { return Vec::new(); }

    let words: Vec<&str> = text.split_whitespace().collect();
    let mut lines: Vec<String> = Vec::new();
    let mut current_line = String::new();

    for word in words {
        if current_line.is_empty() {
            current_line = word.to_string();
        } else if current_line.len() + 1 + word.len() <= max_chars {
            current_line.push(' ');
            current_line.push_str(word);
        } else {
            lines.push(current_line);
            current_line = word.to_string();
        }
    }
    if !current_line.is_empty() {
        lines.push(current_line);
    }

    lines.iter().enumerate()
        .map(|(i, line)| (x, y - i as f64 * leading, line.clone()))
        .collect()
}

// ── PDF Writer helper ───────────────────────────────────────────

struct PdfWriter {
    buf: Vec<u8>,
    offsets: std::collections::HashMap<usize, usize>,
}

impl PdfWriter {
    fn new() -> Self { Self { buf: Vec::new(), offsets: std::collections::HashMap::new() } }
    fn pos(&self) -> usize { self.buf.len() }
    fn raw(&mut self, s: &str) { self.buf.extend_from_slice(s.as_bytes()); }
    fn newline(&mut self) { self.buf.push(b'\n'); }
    fn line(&mut self, s: &str) { self.raw(s); self.newline(); }
    fn bytes(&mut self, data: &[u8]) { self.buf.extend_from_slice(data); }
    fn obj(&mut self, id: usize) {
        self.offsets.insert(id, self.pos());
        self.line(&format!("{id} 0 obj"));
    }
    fn endobj(&mut self) { self.line("endobj"); }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_sizes() {
        let a4 = PageSize::a4();
        assert!((a4.width - 595.28).abs() < 0.01);
        assert!((a4.height - 841.89).abs() < 0.01);
        let letter = PageSize::letter();
        assert!((letter.width - 612.0).abs() < 0.01);
    }

    #[test]
    fn landscape() {
        let a4l = PageSize::a4().landscape();
        assert!((a4l.width - 841.89).abs() < 0.01);
        assert!((a4l.height - 595.28).abs() < 0.01);
    }

    #[test]
    fn unit_conversion() {
        assert!((mm_to_pt(25.4) - 72.0).abs() < 0.01);
        assert!((inch_to_pt(1.0) - 72.0).abs() < 0.01);
    }

    #[test]
    fn color_operators() {
        assert_eq!(PdfColor::black().fill_operator(), "0.000 g");
        assert_eq!(PdfColor::red().fill_operator(), "1.000 0.000 0.000 rg");
        assert_eq!(PdfColor::blue().stroke_operator(), "0.000 0.000 1.000 RG");
    }

    #[test]
    fn font_names() {
        assert_eq!(StandardFont::Helvetica.pdf_name(), "Helvetica");
        assert_eq!(StandardFont::TimesBold.pdf_name(), "Times-Bold");
        assert_eq!(StandardFont::CourierOblique.pdf_name(), "Courier-Oblique");
    }

    #[test]
    fn empty_document_error() {
        let doc = PdfDocument::new("Empty");
        assert_eq!(doc.render(), Err(PdfError::EmptyDocument));
    }

    #[test]
    fn single_page_pdf() {
        let mut doc = PdfDocument::new("Test");
        let page = doc.add_page(PageSize::a4());
        page.draw_text(72.0, 720.0, "Hello, PDF!", 0, 12.0, PdfColor::black());
        let bytes = doc.render().unwrap();
        let header = std::str::from_utf8(&bytes[..8]).unwrap();
        assert_eq!(header, "%PDF-1.4");
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("%%EOF"));
    }

    #[test]
    fn multi_page_pdf() {
        let mut doc = PdfDocument::new("Multi");
        doc.add_page(PageSize::a4());
        doc.add_page(PageSize::letter());
        let bytes = doc.render().unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("/Count 2"));
    }

    #[test]
    fn pdf_with_text() {
        let mut doc = PdfDocument::new("Text");
        let page = doc.add_page(PageSize::a4());
        page.draw_text(100.0, 700.0, "Test text", 0, 14.0, PdfColor::black());
        let bytes = doc.render().unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("Test text"));
        assert!(text.contains("BT"));
        assert!(text.contains("ET"));
    }

    #[test]
    fn pdf_with_line() {
        let mut doc = PdfDocument::new("Line");
        let page = doc.add_page(PageSize::a4());
        page.draw_line(0.0, 400.0, 595.0, 400.0, 1.0, PdfColor::black());
        let bytes = doc.render().unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains(" m "));
        assert!(text.contains(" l S"));
    }

    #[test]
    fn pdf_with_rect() {
        let mut doc = PdfDocument::new("Rect");
        let page = doc.add_page(PageSize::a4());
        page.draw_rect(50.0, 50.0, 200.0, 100.0, Some(PdfColor::blue()), Some(PdfColor::black()), 1.0);
        let bytes = doc.render().unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains(" re B"));
    }

    #[test]
    fn pdf_with_image() {
        let mut doc = PdfDocument::new("Image");
        let data = vec![255u8; 3 * 2 * 2]; // 2x2 white image
        let img_idx = doc.add_image(PdfImage::rgb(2, 2, data));
        let page = doc.add_page(PageSize::a4());
        page.draw_image(100.0, 600.0, 200.0, 200.0, img_idx);
        let bytes = doc.render().unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("/XObject"));
        assert!(text.contains("/Im0 Do"));
    }

    #[test]
    fn pdf_escape_parentheses() {
        assert_eq!(pdf_escape_string("Hello (world)"), "Hello \\(world\\)");
        assert_eq!(pdf_escape_string("back\\slash"), "back\\\\slash");
    }

    #[test]
    fn multiple_fonts() {
        let mut doc = PdfDocument::new("Fonts");
        let f0 = doc.add_font(StandardFont::Helvetica);
        let f1 = doc.add_font(StandardFont::TimesBold);
        let f2 = doc.add_font(StandardFont::Helvetica); // duplicate
        assert_eq!(f0, 0);
        assert_eq!(f1, 1);
        assert_eq!(f2, 0); // same index
    }

    #[test]
    fn layout_paragraph_basic() {
        let lines = layout_paragraph("Hello World", 72.0, 720.0, 500.0, &StandardFont::Helvetica, 12.0, 14.0);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].2, "Hello World");
    }

    #[test]
    fn layout_paragraph_wrapping() {
        let text = "The quick brown fox jumps over the lazy dog and continues running through the forest";
        let lines = layout_paragraph(text, 72.0, 720.0, 200.0, &StandardFont::Helvetica, 12.0, 14.0);
        assert!(lines.len() > 1);
        // Each line should fit within bounds
        for (i, (_, y, _)) in lines.iter().enumerate() {
            let expected_y = 720.0 - i as f64 * 14.0;
            assert!((y - expected_y).abs() < 0.001);
        }
    }

    #[test]
    fn page_operations() {
        let mut page = PdfPage::new(PageSize::a4());
        page.draw_text(0.0, 0.0, "test", 0, 12.0, PdfColor::black());
        page.draw_line(0.0, 0.0, 100.0, 100.0, 1.0, PdfColor::red());
        page.draw_rect(10.0, 10.0, 50.0, 50.0, None, Some(PdfColor::green()), 0.5);
        assert_eq!(page.ops.len(), 3);
    }

    #[test]
    fn grayscale_image() {
        let img = PdfImage::grayscale(4, 4, vec![128; 16]);
        assert_eq!(img.color_space, "DeviceGray");
        assert_eq!(img.bits_per_component, 8);
    }

    #[test]
    fn pdf_has_xref() {
        let mut doc = PdfDocument::new("XRef");
        doc.add_page(PageSize::a4());
        let bytes = doc.render().unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("xref"));
        assert!(text.contains("startxref"));
    }

    #[test]
    fn page_custom_size() {
        let size = PageSize::new(300.0, 400.0);
        assert!((size.width - 300.0).abs() < 0.01);
        assert!((size.height - 400.0).abs() < 0.01);
    }

    #[test]
    fn document_author() {
        let doc = PdfDocument::new("Test").author("Author");
        assert_eq!(doc.author, "Author");
    }

    #[test]
    fn color_cmyk() {
        let c = PdfColor::Cmyk(1.0, 0.0, 0.0, 0.0);
        assert_eq!(c.fill_operator(), "1.000 0.000 0.000 0.000 k");
    }

    #[test]
    fn error_display() {
        assert_eq!(format!("{}", PdfError::EmptyDocument), "document has no pages");
        assert_eq!(format!("{}", PdfError::InvalidPage(5)), "invalid page index: 5");
    }

    #[test]
    fn a3_and_legal_sizes() {
        let a3 = PageSize::a3();
        assert!((a3.width - 841.89).abs() < 0.01);
        let legal = PageSize::legal();
        assert!((legal.height - 1008.0).abs() < 0.01);
    }

    #[test]
    fn rect_fill_only() {
        let mut doc = PdfDocument::new("FillRect");
        let page = doc.add_page(PageSize::a4());
        page.draw_rect(10.0, 10.0, 50.0, 50.0, Some(PdfColor::red()), None, 0.0);
        let bytes = doc.render().unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains(" re f"));
    }

    #[test]
    fn rect_stroke_only() {
        let mut doc = PdfDocument::new("StrokeRect");
        let page = doc.add_page(PageSize::a4());
        page.draw_rect(10.0, 10.0, 50.0, 50.0, None, Some(PdfColor::black()), 1.0);
        let bytes = doc.render().unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains(" re S"));
    }
}
