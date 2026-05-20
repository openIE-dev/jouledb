//! PDF 1.4 generation.
//!
//! Replaces jsPDF and pdfkit with a pure Rust PDF builder that emits
//! valid PDF 1.4 binary output with cross-reference tables.

use std::fmt::Write as FmtWrite;

// ── Page sizes ─────────────────────────────────────────────────────

/// A4 page dimensions in points (595.28 x 841.89).
pub const A4: (f64, f64) = (595.28, 841.89);

/// US Letter page dimensions in points (612 x 792).
pub const LETTER: (f64, f64) = (612.0, 792.0);

/// US Legal page dimensions in points (612 x 1008).
pub const LEGAL: (f64, f64) = (612.0, 1008.0);

// ── Types ──────────────────────────────────────────────────────────

/// Drawable content element on a PDF page.
#[derive(Debug, Clone)]
pub enum PdfContent {
    /// Text rendered at (x, y) with given font size.
    Text {
        x: f64,
        y: f64,
        text: String,
        font_size: f64,
    },
    /// Line from (x1, y1) to (x2, y2) with given stroke width.
    Line {
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        width: f64,
    },
    /// Rectangle at (x, y) with dimensions (w, h).
    Rect {
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        fill: bool,
    },
    /// Circle centred at (cx, cy) with radius r.
    Circle {
        cx: f64,
        cy: f64,
        r: f64,
        fill: bool,
    },
}

/// Metadata embedded in the PDF document info dictionary.
#[derive(Debug, Clone, Default)]
pub struct PdfMetadata {
    pub title: Option<String>,
    pub author: Option<String>,
    pub subject: Option<String>,
    pub creator: String,
}

/// A single page in the PDF document.
#[derive(Debug, Clone)]
pub struct PdfPage {
    pub width: f64,
    pub height: f64,
    pub content: Vec<PdfContent>,
}

impl PdfPage {
    /// Add a text element.
    pub fn add_text(&mut self, x: f64, y: f64, text: &str, size: f64) {
        self.content.push(PdfContent::Text {
            x,
            y,
            text: text.to_string(),
            font_size: size,
        });
    }

    /// Add a line element.
    pub fn add_line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, width: f64) {
        self.content.push(PdfContent::Line {
            x1,
            y1,
            x2,
            y2,
            width,
        });
    }

    /// Add a rectangle element.
    pub fn add_rect(&mut self, x: f64, y: f64, w: f64, h: f64, fill: bool) {
        self.content.push(PdfContent::Rect { x, y, w, h, fill });
    }

    /// Add a circle element.
    pub fn add_circle(&mut self, cx: f64, cy: f64, r: f64, fill: bool) {
        self.content.push(PdfContent::Circle { cx, cy, r, fill });
    }
}

/// A PDF document consisting of pages and metadata.
#[derive(Debug, Clone)]
pub struct PdfDocument {
    pub pages: Vec<PdfPage>,
    pub metadata: PdfMetadata,
}

impl PdfDocument {
    /// Create an empty document.
    pub fn new() -> Self {
        Self {
            pages: Vec::new(),
            metadata: PdfMetadata {
                creator: "joule-web".to_string(),
                ..Default::default()
            },
        }
    }

    /// Add a page with given dimensions (in points) and return a mutable ref.
    pub fn add_page(&mut self, width: f64, height: f64) -> &mut PdfPage {
        self.pages.push(PdfPage {
            width,
            height,
            content: Vec::new(),
        });
        self.pages.last_mut().unwrap()
    }

    /// Set document metadata.
    pub fn set_metadata(&mut self, meta: PdfMetadata) {
        self.metadata = meta;
    }

    /// Generate valid PDF 1.4 binary output.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let mut offsets: Vec<usize> = Vec::new();

        // Header
        out.extend_from_slice(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n");

        // Object numbering plan:
        //   1 = Catalog
        //   2 = Pages
        //   3 = Font (Helvetica)
        //   4 = Info dict
        //   5..5+N-1 = Page objects
        //   5+N..5+2N-1 = Page content streams
        let page_obj_start = 5;
        let content_obj_start = page_obj_start + self.pages.len();
        let total_objects = content_obj_start + self.pages.len();

        // Helper: write object and record offset
        macro_rules! obj {
            ($num:expr, $body:expr) => {
                offsets.push(out.len());
                let _ = write_obj(&mut out, $num, $body);
            };
        }

        // obj 1 — Catalog
        obj!(1, &format!("<< /Type /Catalog /Pages 2 0 R >>"));

        // obj 2 — Pages
        let kids: Vec<String> = (0..self.pages.len())
            .map(|i| format!("{} 0 R", page_obj_start + i))
            .collect();
        obj!(
            2,
            &format!(
                "<< /Type /Pages /Kids [{}] /Count {} >>",
                kids.join(" "),
                self.pages.len()
            )
        );

        // obj 3 — Font
        obj!(
            3,
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
        );

        // obj 4 — Info
        let mut info = String::from("<< ");
        if let Some(ref t) = self.metadata.title {
            let _ = write!(info, "/Title ({}) ", pdf_escape(t));
        }
        if let Some(ref a) = self.metadata.author {
            let _ = write!(info, "/Author ({}) ", pdf_escape(a));
        }
        if let Some(ref s) = self.metadata.subject {
            let _ = write!(info, "/Subject ({}) ", pdf_escape(s));
        }
        let _ = write!(info, "/Creator ({}) ", pdf_escape(&self.metadata.creator));
        info.push_str(">>");
        obj!(4, &info);

        // Page objects + content streams
        for (i, page) in self.pages.iter().enumerate() {
            let page_num = page_obj_start + i;
            let content_num = content_obj_start + i;

            // Build content stream
            let stream = build_page_stream(page);
            let stream_bytes = stream.as_bytes();

            // Page object
            obj!(
                page_num,
                &format!(
                    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {:.2} {:.2}] /Contents {} 0 R /Resources << /Font << /F1 3 0 R >> >> >>",
                    page.width, page.height, content_num
                )
            );

            // Content stream object
            offsets.push(out.len());
            let header = format!(
                "{} 0 obj\n<< /Length {} >>\nstream\n",
                content_num,
                stream_bytes.len()
            );
            out.extend_from_slice(header.as_bytes());
            out.extend_from_slice(stream_bytes);
            out.extend_from_slice(b"\nendstream\nendobj\n");
        }

        // Cross-reference table
        let xref_offset = out.len();
        let _ = write!(
            &mut out as &mut dyn std::io::Write,
            "xref\n0 {}\n",
            total_objects + 1
        );
        // Entry for object 0 (free)
        out.extend_from_slice(b"0000000000 65535 f \n");

        // We need offsets in object-number order (1..total_objects)
        // offsets[0] = obj 1, offsets[1] = obj 2, etc.
        for off in &offsets {
            let entry = format!("{:010} 00000 n \n", off);
            out.extend_from_slice(entry.as_bytes());
        }

        // Trailer
        let trailer = format!(
            "trailer\n<< /Size {} /Root 1 0 R /Info 4 0 R >>\nstartxref\n{}\n%%EOF\n",
            total_objects + 1,
            xref_offset
        );
        out.extend_from_slice(trailer.as_bytes());

        out
    }
}

impl Default for PdfDocument {
    fn default() -> Self {
        Self::new()
    }
}

// ── Internal helpers ───────────────────────────────────────────────

fn write_obj(out: &mut Vec<u8>, num: usize, body: &str) -> std::fmt::Result {
    let s = format!("{num} 0 obj\n{body}\nendobj\n");
    out.extend_from_slice(s.as_bytes());
    Ok(())
}

fn pdf_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('(', "\\(")
        .replace(')', "\\)")
}

fn build_page_stream(page: &PdfPage) -> String {
    let mut s = String::new();
    for item in &page.content {
        match item {
            PdfContent::Text {
                x,
                y,
                text,
                font_size,
            } => {
                let _ = write!(
                    s,
                    "BT /F1 {:.2} Tf {:.2} {:.2} Td ({}) Tj ET\n",
                    font_size,
                    x,
                    y,
                    pdf_escape(text)
                );
            }
            PdfContent::Line {
                x1,
                y1,
                x2,
                y2,
                width,
            } => {
                let _ = write!(
                    s,
                    "{:.2} w {:.2} {:.2} m {:.2} {:.2} l S\n",
                    width, x1, y1, x2, y2
                );
            }
            PdfContent::Rect { x, y, w, h, fill } => {
                let op = if *fill { "f" } else { "S" };
                let _ = write!(s, "{:.2} {:.2} {:.2} {:.2} re {}\n", x, y, w, h, op);
            }
            PdfContent::Circle { cx, cy, r, fill } => {
                // Approximate circle with four Bézier curves
                let k = 0.5522847498; // magic number for circle approx
                let kr = k * r;
                let op = if *fill { "f" } else { "S" };
                let _ = write!(
                    s,
                    "{:.2} {:.2} m {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} c {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} c {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} c {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} c {}\n",
                    cx + r, *cy,
                    cx + r, cy + kr, cx + kr, cy + r, *cx, cy + r,
                    cx - kr, cy + r, cx - r, cy + kr, cx - r, *cy,
                    cx - r, cy - kr, cx - kr, cy - r, *cx, cy - r,
                    cx + kr, cy - r, cx + r, cy - kr, cx + r, *cy,
                    op
                );
            }
        }
    }
    s
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_doc_has_valid_header() {
        let doc = PdfDocument::new();
        let bytes = doc.to_bytes();
        assert!(bytes.starts_with(b"%PDF-1.4"));
    }

    #[test]
    fn empty_doc_has_eof() {
        let doc = PdfDocument::new();
        let bytes = doc.to_bytes();
        let s = String::from_utf8_lossy(&bytes);
        assert!(s.contains("%%EOF"));
    }

    #[test]
    fn empty_doc_has_xref() {
        let doc = PdfDocument::new();
        let bytes = doc.to_bytes();
        let s = String::from_utf8_lossy(&bytes);
        assert!(s.contains("xref"));
        assert!(s.contains("startxref"));
    }

    #[test]
    fn page_with_text() {
        let mut doc = PdfDocument::new();
        let page = doc.add_page(A4.0, A4.1);
        page.add_text(72.0, 720.0, "Hello PDF", 12.0);
        let bytes = doc.to_bytes();
        let s = String::from_utf8_lossy(&bytes);
        assert!(s.contains("Hello PDF"));
    }

    #[test]
    fn metadata_included() {
        let mut doc = PdfDocument::new();
        doc.set_metadata(PdfMetadata {
            title: Some("Test Title".to_string()),
            author: Some("Author Name".to_string()),
            subject: None,
            creator: "joule-web-test".to_string(),
        });
        doc.add_page(LETTER.0, LETTER.1);
        let bytes = doc.to_bytes();
        let s = String::from_utf8_lossy(&bytes);
        assert!(s.contains("Test Title"));
        assert!(s.contains("Author Name"));
    }

    #[test]
    fn page_count() {
        let mut doc = PdfDocument::new();
        doc.add_page(A4.0, A4.1);
        doc.add_page(A4.0, A4.1);
        doc.add_page(A4.0, A4.1);
        assert_eq!(doc.pages.len(), 3);
        let bytes = doc.to_bytes();
        let s = String::from_utf8_lossy(&bytes);
        assert!(s.contains("/Count 3"));
    }

    #[test]
    fn a4_dimensions_in_output() {
        let mut doc = PdfDocument::new();
        doc.add_page(A4.0, A4.1);
        let bytes = doc.to_bytes();
        let s = String::from_utf8_lossy(&bytes);
        assert!(s.contains("595.28"));
        assert!(s.contains("841.89"));
    }

    #[test]
    fn multi_page_with_content() {
        let mut doc = PdfDocument::new();
        {
            let p = doc.add_page(LETTER.0, LETTER.1);
            p.add_text(10.0, 10.0, "Page 1", 14.0);
            p.add_line(0.0, 0.0, 612.0, 0.0, 1.0);
        }
        {
            let p = doc.add_page(LETTER.0, LETTER.1);
            p.add_rect(50.0, 50.0, 200.0, 100.0, true);
            p.add_circle(300.0, 400.0, 50.0, false);
        }
        let bytes = doc.to_bytes();
        let s = String::from_utf8_lossy(&bytes);
        assert!(s.contains("Page 1"));
        assert!(s.contains("/Count 2"));
    }

    #[test]
    fn binary_starts_with_pdf() {
        let mut doc = PdfDocument::new();
        doc.add_page(LEGAL.0, LEGAL.1);
        let bytes = doc.to_bytes();
        assert_eq!(&bytes[..5], b"%PDF-");
    }

    #[test]
    fn legal_dimensions() {
        assert_eq!(LEGAL, (612.0, 1008.0));
    }
}
