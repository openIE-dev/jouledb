//! PDF document layout engine — page model, text blocks, paragraphs,
//! tables, lists, headers/footers, and PDF-like content stream output.
//!
//! Pure-Rust replacement for pdfkit, jsPDF, and pdfmake. Generates
//! valid PDF 1.4 content streams with page layout, font metrics, and
//! automatic page breaks.

use std::fmt;
use std::fmt::Write as FmtWrite;

// ── Constants ───────────────────────────────────────────────────

/// A4 page dimensions in points.
pub const A4_WIDTH: f64 = 595.28;
pub const A4_HEIGHT: f64 = 841.89;

/// US Letter page dimensions in points.
pub const LETTER_WIDTH: f64 = 612.0;
pub const LETTER_HEIGHT: f64 = 792.0;

/// Default margin in points.
pub const DEFAULT_MARGIN: f64 = 72.0;

/// Default line height multiplier.
pub const DEFAULT_LINE_HEIGHT: f64 = 1.2;

// ── Page size ───────────────────────────────────────────────────

/// Predefined page sizes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PageSize {
    A4,
    Letter,
    Legal,
    A3,
    Custom { width: f64, height: f64 },
}

impl PageSize {
    pub fn dimensions(&self) -> (f64, f64) {
        match self {
            Self::A4 => (595.28, 841.89),
            Self::Letter => (612.0, 792.0),
            Self::Legal => (612.0, 1008.0),
            Self::A3 => (841.89, 1190.55),
            Self::Custom { width, height } => (*width, *height),
        }
    }
}

/// Page orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    Portrait,
    Landscape,
}

// ── Margins ─────────────────────────────────────────────────────

/// Page margins in points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Margins {
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
    pub left: f64,
}

impl Margins {
    pub fn new(top: f64, right: f64, bottom: f64, left: f64) -> Self {
        Self {
            top,
            right,
            bottom,
            left,
        }
    }

    pub fn uniform(m: f64) -> Self {
        Self::new(m, m, m, m)
    }
}

impl Default for Margins {
    fn default() -> Self {
        Self::uniform(DEFAULT_MARGIN)
    }
}

// ── Font metrics ────────────────────────────────────────────────

/// Simplified font metrics for standard PDF base 14 fonts.
#[derive(Debug, Clone)]
pub struct FontMetrics {
    pub name: String,
    /// Average character width as a fraction of font size.
    pub avg_char_width: f64,
    /// Ascender height as a fraction of font size.
    pub ascender: f64,
    /// Descender depth as a fraction of font size (negative).
    pub descender: f64,
    /// Cap height as a fraction of font size.
    pub cap_height: f64,
}

impl FontMetrics {
    /// Helvetica-like metrics.
    pub fn helvetica() -> Self {
        Self {
            name: "Helvetica".to_string(),
            avg_char_width: 0.52,
            ascender: 0.72,
            descender: -0.28,
            cap_height: 0.72,
        }
    }

    /// Times-like metrics.
    pub fn times() -> Self {
        Self {
            name: "Times-Roman".to_string(),
            avg_char_width: 0.48,
            ascender: 0.68,
            descender: -0.32,
            cap_height: 0.66,
        }
    }

    /// Courier (monospace) metrics.
    pub fn courier() -> Self {
        Self {
            name: "Courier".to_string(),
            avg_char_width: 0.60,
            ascender: 0.63,
            descender: -0.37,
            cap_height: 0.56,
        }
    }

    /// Estimate the width of a string in points.
    pub fn string_width(&self, text: &str, font_size: f64) -> f64 {
        text.len() as f64 * self.avg_char_width * font_size
    }

    /// Line height for a given font size and multiplier.
    pub fn line_height(&self, font_size: f64, multiplier: f64) -> f64 {
        font_size * multiplier
    }
}

impl Default for FontMetrics {
    fn default() -> Self {
        Self::helvetica()
    }
}

// ── Text style ──────────────────────────────────────────────────

/// Text alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAlign {
    Left,
    Center,
    Right,
    Justify,
}

impl Default for TextAlign {
    fn default() -> Self {
        Self::Left
    }
}

/// Style for text rendering.
#[derive(Debug, Clone)]
pub struct TextStyle {
    pub font: FontMetrics,
    pub font_size: f64,
    pub line_height: f64,
    pub align: TextAlign,
    pub bold: bool,
    pub italic: bool,
    pub color: Color,
}

impl Default for TextStyle {
    fn default() -> Self {
        Self {
            font: FontMetrics::helvetica(),
            font_size: 12.0,
            line_height: DEFAULT_LINE_HEIGHT,
            align: TextAlign::Left,
            bold: false,
            italic: false,
            color: Color::black(),
        }
    }
}

/// RGB color (0.0 to 1.0 per channel).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl Color {
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    pub fn black() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    pub fn white() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }

    pub fn red() -> Self {
        Self::new(1.0, 0.0, 0.0)
    }

    pub fn blue() -> Self {
        Self::new(0.0, 0.0, 1.0)
    }

    pub fn gray(v: f64) -> Self {
        Self::new(v, v, v)
    }

    /// Convert from 0-255 RGB values.
    pub fn from_rgb8(r: u8, g: u8, b: u8) -> Self {
        Self::new(r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0)
    }
}

// ── Content elements ────────────────────────────────────────────

/// A content element in the document.
#[derive(Debug, Clone)]
pub enum ContentElement {
    /// A heading with level (1-6).
    Heading { level: u8, text: String },
    /// A paragraph of text.
    Paragraph { text: String, style: TextStyle },
    /// An unordered list.
    UnorderedList { items: Vec<String> },
    /// An ordered list.
    OrderedList { items: Vec<String> },
    /// A table with headers and rows.
    Table {
        headers: Vec<String>,
        rows: Vec<Vec<String>>,
        column_widths: Option<Vec<f64>>,
    },
    /// A horizontal rule.
    HorizontalRule,
    /// Raw PDF content stream commands.
    RawStream(String),
    /// A rectangle.
    Rect {
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        fill: Option<Color>,
        stroke: Option<Color>,
        stroke_width: f64,
    },
    /// A line.
    Line {
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        color: Color,
        width: f64,
    },
    /// A circle (approximated with bezier curves).
    Circle {
        cx: f64,
        cy: f64,
        radius: f64,
        fill: Option<Color>,
        stroke: Option<Color>,
    },
    /// Vertical spacing.
    Spacer(f64),
    /// Page break.
    PageBreak,
}

/// Header/footer callback content.
#[derive(Debug, Clone)]
pub struct HeaderFooter {
    pub left: Option<String>,
    pub center: Option<String>,
    pub right: Option<String>,
    pub font_size: f64,
}

impl Default for HeaderFooter {
    fn default() -> Self {
        Self {
            left: None,
            center: None,
            right: None,
            font_size: 10.0,
        }
    }
}

// ── Layout engine ───────────────────────────────────────────────

/// A rendered page with content stream.
#[derive(Debug, Clone)]
pub struct RenderedPage {
    pub width: f64,
    pub height: f64,
    pub content_stream: String,
    pub page_number: usize,
}

/// The PDF layout engine.
pub struct PdfLayout {
    page_size: PageSize,
    orientation: Orientation,
    margins: Margins,
    default_style: TextStyle,
    elements: Vec<ContentElement>,
    header: Option<HeaderFooter>,
    footer: Option<HeaderFooter>,
}

impl PdfLayout {
    pub fn new() -> Self {
        Self {
            page_size: PageSize::A4,
            orientation: Orientation::Portrait,
            margins: Margins::default(),
            default_style: TextStyle::default(),
            elements: Vec::new(),
            header: None,
            footer: None,
        }
    }

    pub fn page_size(mut self, size: PageSize) -> Self {
        self.page_size = size;
        self
    }

    pub fn orientation(mut self, orientation: Orientation) -> Self {
        self.orientation = orientation;
        self
    }

    pub fn margins(mut self, margins: Margins) -> Self {
        self.margins = margins;
        self
    }

    pub fn default_style(mut self, style: TextStyle) -> Self {
        self.default_style = style;
        self
    }

    pub fn header(mut self, header: HeaderFooter) -> Self {
        self.header = Some(header);
        self
    }

    pub fn footer(mut self, footer: HeaderFooter) -> Self {
        self.footer = Some(footer);
        self
    }

    /// Add a content element.
    pub fn add(&mut self, element: ContentElement) {
        self.elements.push(element);
    }

    /// Add a heading.
    pub fn heading(&mut self, level: u8, text: impl Into<String>) {
        self.add(ContentElement::Heading {
            level,
            text: text.into(),
        });
    }

    /// Add a paragraph.
    pub fn paragraph(&mut self, text: impl Into<String>) {
        self.add(ContentElement::Paragraph {
            text: text.into(),
            style: self.default_style.clone(),
        });
    }

    /// Add a paragraph with custom style.
    pub fn styled_paragraph(&mut self, text: impl Into<String>, style: TextStyle) {
        self.add(ContentElement::Paragraph {
            text: text.into(),
            style,
        });
    }

    /// Add an unordered list.
    pub fn unordered_list(&mut self, items: Vec<String>) {
        self.add(ContentElement::UnorderedList { items });
    }

    /// Add an ordered list.
    pub fn ordered_list(&mut self, items: Vec<String>) {
        self.add(ContentElement::OrderedList { items });
    }

    /// Add a table.
    pub fn table(
        &mut self,
        headers: Vec<String>,
        rows: Vec<Vec<String>>,
        column_widths: Option<Vec<f64>>,
    ) {
        self.add(ContentElement::Table {
            headers,
            rows,
            column_widths,
        });
    }

    /// Add a horizontal rule.
    pub fn horizontal_rule(&mut self) {
        self.add(ContentElement::HorizontalRule);
    }

    /// Add a page break.
    pub fn page_break(&mut self) {
        self.add(ContentElement::PageBreak);
    }

    /// Add vertical spacing.
    pub fn spacer(&mut self, points: f64) {
        self.add(ContentElement::Spacer(points));
    }

    /// Add a rectangle.
    pub fn rect(
        &mut self,
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        fill: Option<Color>,
        stroke: Option<Color>,
    ) {
        self.add(ContentElement::Rect {
            x,
            y,
            width,
            height,
            fill,
            stroke,
            stroke_width: 1.0,
        });
    }

    /// Add a line.
    pub fn line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, color: Color, width: f64) {
        self.add(ContentElement::Line {
            x1,
            y1,
            x2,
            y2,
            color,
            width,
        });
    }

    /// Render all elements into pages.
    pub fn render(&self) -> Vec<RenderedPage> {
        let (raw_w, raw_h) = self.page_size.dimensions();
        let (page_w, page_h) = match self.orientation {
            Orientation::Portrait => (raw_w, raw_h),
            Orientation::Landscape => (raw_h, raw_w),
        };

        let content_width = page_w - self.margins.left - self.margins.right;
        let content_top = page_h - self.margins.top;
        let content_bottom = self.margins.bottom;

        let mut pages: Vec<String> = Vec::new();
        let mut current_stream = String::new();
        let mut cursor_y = content_top;

        // Clone header/footer for use in rendering.
        let hdr = self.header.clone();
        let ftr = self.footer.clone();

        for element in &self.elements {
            match element {
                ContentElement::PageBreak => {
                    pages.push(current_stream.clone());
                    current_stream = String::new();
                    cursor_y = content_top;
                }
                ContentElement::Spacer(pts) => {
                    cursor_y -= pts;
                    if cursor_y < content_bottom {
                        pages.push(current_stream.clone());
                        current_stream = String::new();
                        cursor_y = content_top - pts;
                    }
                }
                ContentElement::Heading { level, text } => {
                    let font_size = heading_font_size(*level);
                    let lh = font_size * DEFAULT_LINE_HEIGHT;

                    if cursor_y - lh < content_bottom {
                        pages.push(current_stream.clone());
                        current_stream = String::new();
                        cursor_y = content_top;
                    }

                    let _ = writeln!(
                        current_stream,
                        "BT /F1 {font_size:.1} Tf {:.2} {cursor_y:.2} Td ({}) Tj ET",
                        self.margins.left,
                        pdf_escape(text)
                    );
                    cursor_y -= lh + 4.0;
                }
                ContentElement::Paragraph { text, style } => {
                    let lh = style.font_size * style.line_height;
                    let wrapped = wrap_text(text, content_width, &style.font, style.font_size);

                    for line in &wrapped {
                        if cursor_y - lh < content_bottom {
                            pages.push(current_stream.clone());
                            current_stream = String::new();
                            cursor_y = content_top;
                        }

                        let x = match style.align {
                            TextAlign::Left => self.margins.left,
                            TextAlign::Center => {
                                let w = style.font.string_width(line, style.font_size);
                                self.margins.left + (content_width - w) / 2.0
                            }
                            TextAlign::Right => {
                                let w = style.font.string_width(line, style.font_size);
                                self.margins.left + content_width - w
                            }
                            TextAlign::Justify => self.margins.left,
                        };

                        // Set color.
                        let _ = write!(
                            current_stream,
                            "{:.3} {:.3} {:.3} rg ",
                            style.color.r, style.color.g, style.color.b
                        );
                        let _ = writeln!(
                            current_stream,
                            "BT /F1 {:.1} Tf {x:.2} {cursor_y:.2} Td ({}) Tj ET",
                            style.font_size,
                            pdf_escape(line)
                        );
                        cursor_y -= lh;
                    }
                    cursor_y -= 4.0; // Paragraph spacing.
                }
                ContentElement::UnorderedList { items } => {
                    let lh = self.default_style.font_size * DEFAULT_LINE_HEIGHT;
                    let indent = 20.0;

                    for item in items {
                        if cursor_y - lh < content_bottom {
                            pages.push(current_stream.clone());
                            current_stream = String::new();
                            cursor_y = content_top;
                        }

                        // Bullet.
                        let _ = writeln!(
                            current_stream,
                            "BT /F1 {:.1} Tf {:.2} {cursor_y:.2} Td (\\225) Tj ET",
                            self.default_style.font_size,
                            self.margins.left
                        );
                        // Text.
                        let _ = writeln!(
                            current_stream,
                            "BT /F1 {:.1} Tf {:.2} {cursor_y:.2} Td ({}) Tj ET",
                            self.default_style.font_size,
                            self.margins.left + indent,
                            pdf_escape(item)
                        );
                        cursor_y -= lh;
                    }
                    cursor_y -= 4.0;
                }
                ContentElement::OrderedList { items } => {
                    let lh = self.default_style.font_size * DEFAULT_LINE_HEIGHT;
                    let indent = 24.0;

                    for (i, item) in items.iter().enumerate() {
                        if cursor_y - lh < content_bottom {
                            pages.push(current_stream.clone());
                            current_stream = String::new();
                            cursor_y = content_top;
                        }

                        let number = format!("{}.", i + 1);
                        let _ = writeln!(
                            current_stream,
                            "BT /F1 {:.1} Tf {:.2} {cursor_y:.2} Td ({}) Tj ET",
                            self.default_style.font_size,
                            self.margins.left,
                            pdf_escape(&number)
                        );
                        let _ = writeln!(
                            current_stream,
                            "BT /F1 {:.1} Tf {:.2} {cursor_y:.2} Td ({}) Tj ET",
                            self.default_style.font_size,
                            self.margins.left + indent,
                            pdf_escape(item)
                        );
                        cursor_y -= lh;
                    }
                    cursor_y -= 4.0;
                }
                ContentElement::Table {
                    headers,
                    rows,
                    column_widths,
                } => {
                    let num_cols = headers.len();
                    let col_w = if let Some(widths) = column_widths {
                        widths.clone()
                    } else {
                        let w = content_width / num_cols as f64;
                        vec![w; num_cols]
                    };

                    let row_height = self.default_style.font_size * DEFAULT_LINE_HEIGHT + 8.0;

                    // Draw header row.
                    if cursor_y - row_height < content_bottom {
                        pages.push(current_stream.clone());
                        current_stream = String::new();
                        cursor_y = content_top;
                    }

                    // Header background.
                    let _ = writeln!(
                        current_stream,
                        "0.9 0.9 0.9 rg {:.2} {:.2} {:.2} {:.2} re f",
                        self.margins.left,
                        cursor_y - row_height,
                        content_width,
                        row_height
                    );

                    let mut x = self.margins.left;
                    for (j, hdr_text) in headers.iter().enumerate() {
                        let cw = col_w.get(j).copied().unwrap_or(50.0);
                        let _ = writeln!(
                            current_stream,
                            "BT /F1 {:.1} Tf {:.2} {:.2} Td ({}) Tj ET",
                            self.default_style.font_size,
                            x + 4.0,
                            cursor_y - row_height + 4.0,
                            pdf_escape(hdr_text)
                        );
                        x += cw;
                    }
                    cursor_y -= row_height;

                    // Data rows.
                    for row in rows {
                        if cursor_y - row_height < content_bottom {
                            pages.push(current_stream.clone());
                            current_stream = String::new();
                            cursor_y = content_top;
                        }

                        // Row border.
                        let _ = writeln!(
                            current_stream,
                            "0.8 0.8 0.8 RG {:.2} {:.2} {:.2} {:.2} re S",
                            self.margins.left,
                            cursor_y - row_height,
                            content_width,
                            row_height
                        );

                        let mut x_pos = self.margins.left;
                        for (j, cell) in row.iter().enumerate() {
                            let cw = col_w.get(j).copied().unwrap_or(50.0);
                            let _ = writeln!(
                                current_stream,
                                "BT /F1 {:.1} Tf {:.2} {:.2} Td ({}) Tj ET",
                                self.default_style.font_size,
                                x_pos + 4.0,
                                cursor_y - row_height + 4.0,
                                pdf_escape(cell)
                            );
                            x_pos += cw;
                        }
                        cursor_y -= row_height;
                    }
                    cursor_y -= 4.0;
                }
                ContentElement::HorizontalRule => {
                    if cursor_y - 10.0 < content_bottom {
                        pages.push(current_stream.clone());
                        current_stream = String::new();
                        cursor_y = content_top;
                    }
                    let _ = writeln!(
                        current_stream,
                        "0.5 0.5 0.5 RG 1 w {:.2} {:.2} m {:.2} {:.2} l S",
                        self.margins.left,
                        cursor_y - 5.0,
                        self.margins.left + content_width,
                        cursor_y - 5.0
                    );
                    cursor_y -= 10.0;
                }
                ContentElement::RawStream(s) => {
                    current_stream.push_str(s);
                    current_stream.push('\n');
                }
                ContentElement::Rect {
                    x,
                    y,
                    width,
                    height,
                    fill,
                    stroke,
                    stroke_width,
                } => {
                    if let Some(c) = fill {
                        let _ = writeln!(
                            current_stream,
                            "{:.3} {:.3} {:.3} rg {x:.2} {y:.2} {width:.2} {height:.2} re f",
                            c.r, c.g, c.b
                        );
                    }
                    if let Some(c) = stroke {
                        let _ = writeln!(
                            current_stream,
                            "{:.3} {:.3} {:.3} RG {stroke_width} w {x:.2} {y:.2} {width:.2} {height:.2} re S",
                            c.r, c.g, c.b
                        );
                    }
                }
                ContentElement::Line {
                    x1,
                    y1,
                    x2,
                    y2,
                    color,
                    width,
                } => {
                    let _ = writeln!(
                        current_stream,
                        "{:.3} {:.3} {:.3} RG {width} w {x1:.2} {y1:.2} m {x2:.2} {y2:.2} l S",
                        color.r, color.g, color.b
                    );
                }
                ContentElement::Circle {
                    cx,
                    cy,
                    radius,
                    fill,
                    stroke,
                } => {
                    // Approximate circle with 4 bezier curves.
                    let k = 0.5522847498;
                    let kr = k * radius;
                    let stream = format!(
                        "{:.2} {:.2} m {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} c {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} c {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} c {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} c",
                        cx + radius, *cy,
                        cx + radius, cy + kr, cx + kr, cy + radius, *cx, cy + radius,
                        cx - kr, cy + radius, cx - radius, cy + kr, cx - radius, *cy,
                        cx - radius, cy - kr, cx - kr, cy - radius, *cx, cy - radius,
                        cx + kr, cy - radius, cx + radius, cy - kr, cx + radius, *cy,
                    );
                    if let Some(c) = fill {
                        let _ = writeln!(
                            current_stream,
                            "{:.3} {:.3} {:.3} rg {stream} f",
                            c.r, c.g, c.b
                        );
                    }
                    if let Some(c) = stroke {
                        let _ = writeln!(
                            current_stream,
                            "{:.3} {:.3} {:.3} RG 1 w {stream} S",
                            c.r, c.g, c.b
                        );
                    }
                }
            }
        }

        // Push the last page.
        if !current_stream.is_empty() || pages.is_empty() {
            pages.push(current_stream);
        }

        let total_pages = pages.len();

        // Build rendered pages with headers/footers.
        pages
            .into_iter()
            .enumerate()
            .map(|(i, mut stream)| {
                let page_num = i + 1;

                // Add header.
                if let Some(h) = &hdr {
                    let hy = page_h - self.margins.top / 2.0;
                    if let Some(left) = &h.left {
                        let text = replace_page_vars(left, page_num, total_pages);
                        let _ = writeln!(
                            stream,
                            "BT /F1 {:.1} Tf {:.2} {hy:.2} Td ({}) Tj ET",
                            h.font_size,
                            self.margins.left,
                            pdf_escape(&text)
                        );
                    }
                    if let Some(center) = &h.center {
                        let text = replace_page_vars(center, page_num, total_pages);
                        let w = FontMetrics::helvetica().string_width(&text, h.font_size);
                        let cx = page_w / 2.0 - w / 2.0;
                        let _ = writeln!(
                            stream,
                            "BT /F1 {:.1} Tf {cx:.2} {hy:.2} Td ({}) Tj ET",
                            h.font_size,
                            pdf_escape(&text)
                        );
                    }
                    if let Some(right) = &h.right {
                        let text = replace_page_vars(right, page_num, total_pages);
                        let w = FontMetrics::helvetica().string_width(&text, h.font_size);
                        let rx = page_w - self.margins.right - w;
                        let _ = writeln!(
                            stream,
                            "BT /F1 {:.1} Tf {rx:.2} {hy:.2} Td ({}) Tj ET",
                            h.font_size,
                            pdf_escape(&text)
                        );
                    }
                }

                // Add footer.
                if let Some(ft) = &ftr {
                    let fy = self.margins.bottom / 2.0;
                    if let Some(left) = &ft.left {
                        let text = replace_page_vars(left, page_num, total_pages);
                        let _ = writeln!(
                            stream,
                            "BT /F1 {:.1} Tf {:.2} {fy:.2} Td ({}) Tj ET",
                            ft.font_size,
                            self.margins.left,
                            pdf_escape(&text)
                        );
                    }
                    if let Some(center) = &ft.center {
                        let text = replace_page_vars(center, page_num, total_pages);
                        let w = FontMetrics::helvetica().string_width(&text, ft.font_size);
                        let cx = page_w / 2.0 - w / 2.0;
                        let _ = writeln!(
                            stream,
                            "BT /F1 {:.1} Tf {cx:.2} {fy:.2} Td ({}) Tj ET",
                            ft.font_size,
                            pdf_escape(&text)
                        );
                    }
                    if let Some(right) = &ft.right {
                        let text = replace_page_vars(right, page_num, total_pages);
                        let w = FontMetrics::helvetica().string_width(&text, ft.font_size);
                        let rx = page_w - self.margins.right - w;
                        let _ = writeln!(
                            stream,
                            "BT /F1 {:.1} Tf {rx:.2} {fy:.2} Td ({}) Tj ET",
                            ft.font_size,
                            pdf_escape(&text)
                        );
                    }
                }

                RenderedPage {
                    width: page_w,
                    height: page_h,
                    content_stream: stream,
                    page_number: page_num,
                }
            })
            .collect()
    }

    /// Render the document as a complete PDF 1.4 binary.
    pub fn to_pdf(&self) -> Vec<u8> {
        let rendered = self.render();
        build_pdf(&rendered)
    }
}

impl Default for PdfLayout {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ─────────────────────────────────────────────────────

fn heading_font_size(level: u8) -> f64 {
    match level {
        1 => 24.0,
        2 => 20.0,
        3 => 16.0,
        4 => 14.0,
        5 => 12.0,
        _ => 11.0,
    }
}

fn pdf_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '(' => out.push_str("\\("),
            ')' => out.push_str("\\)"),
            '\\' => out.push_str("\\\\"),
            _ => out.push(ch),
        }
    }
    out
}

fn replace_page_vars(template: &str, page: usize, total: usize) -> String {
    template
        .replace("{{page}}", &page.to_string())
        .replace("{{total}}", &total.to_string())
}

fn wrap_text(text: &str, max_width: f64, font: &FontMetrics, font_size: f64) -> Vec<String> {
    let mut lines = Vec::new();
    let words: Vec<&str> = text.split_whitespace().collect();

    if words.is_empty() {
        return vec![String::new()];
    }

    let mut current_line = String::new();

    for word in words {
        let test = if current_line.is_empty() {
            word.to_string()
        } else {
            format!("{current_line} {word}")
        };

        if font.string_width(&test, font_size) <= max_width {
            current_line = test;
        } else {
            if !current_line.is_empty() {
                lines.push(current_line);
            }
            current_line = word.to_string();
        }
    }

    if !current_line.is_empty() {
        lines.push(current_line);
    }

    lines
}

/// Build a minimal valid PDF 1.4 file from rendered pages.
fn build_pdf(pages: &[RenderedPage]) -> Vec<u8> {
    let mut out = String::new();
    let mut offsets: Vec<usize> = Vec::new();

    out.push_str("%PDF-1.4\n");

    // Object 1: Catalog.
    offsets.push(out.len());
    out.push_str("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    // Object 2: Pages.
    offsets.push(out.len());
    let page_refs: Vec<String> = (0..pages.len())
        .map(|i| format!("{} 0 R", 3 + i * 2))
        .collect();
    let _ = writeln!(
        out,
        "2 0 obj\n<< /Type /Pages /Kids [{}] /Count {} >>\nendobj",
        page_refs.join(" "),
        pages.len()
    );

    // Objects for each page.
    let mut next_obj = 3;
    for page in pages {
        let page_obj = next_obj;
        let stream_obj = next_obj + 1;
        next_obj += 2;

        offsets.push(out.len());
        let _ = writeln!(
            out,
            "{page_obj} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {:.2} {:.2}] /Contents {stream_obj} 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>\nendobj",
            page.width, page.height
        );

        offsets.push(out.len());
        let stream_bytes = page.content_stream.len();
        let _ = writeln!(
            out,
            "{stream_obj} 0 obj\n<< /Length {stream_bytes} >>\nstream\n{}endstream\nendobj",
            page.content_stream
        );
    }

    // Xref table.
    let xref_offset = out.len();
    let _ = writeln!(out, "xref\n0 {}", offsets.len() + 1);
    out.push_str("0000000000 65535 f \n");
    for off in &offsets {
        let _ = writeln!(out, "{off:010} 00000 n ");
    }

    let _ = writeln!(
        out,
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF",
        offsets.len() + 1
    );

    out.into_bytes()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_size_a4() {
        let (w, h) = PageSize::A4.dimensions();
        assert!((w - 595.28).abs() < 0.01);
        assert!((h - 841.89).abs() < 0.01);
    }

    #[test]
    fn test_page_size_letter() {
        let (w, h) = PageSize::Letter.dimensions();
        assert!((w - 612.0).abs() < 0.01);
        assert!((h - 792.0).abs() < 0.01);
    }

    #[test]
    fn test_font_metrics_string_width() {
        let font = FontMetrics::helvetica();
        let w = font.string_width("Hello", 12.0);
        assert!(w > 0.0);
    }

    #[test]
    fn test_font_metrics_line_height() {
        let font = FontMetrics::helvetica();
        let lh = font.line_height(12.0, 1.2);
        assert!((lh - 14.4).abs() < 0.01);
    }

    #[test]
    fn test_margins_uniform() {
        let m = Margins::uniform(50.0);
        assert_eq!(m.top, 50.0);
        assert_eq!(m.right, 50.0);
        assert_eq!(m.bottom, 50.0);
        assert_eq!(m.left, 50.0);
    }

    #[test]
    fn test_color_from_rgb8() {
        let c = Color::from_rgb8(255, 0, 128);
        assert!((c.r - 1.0).abs() < 0.01);
        assert!((c.g).abs() < 0.01);
        assert!((c.b - 0.502).abs() < 0.01);
    }

    #[test]
    fn test_simple_document() {
        let mut doc = PdfLayout::new();
        doc.heading(1, "Title");
        doc.paragraph("Hello, World!");
        let pages = doc.render();
        assert_eq!(pages.len(), 1);
        assert!(pages[0].content_stream.contains("Title"));
    }

    #[test]
    fn test_page_break() {
        let mut doc = PdfLayout::new();
        doc.paragraph("Page 1 content");
        doc.page_break();
        doc.paragraph("Page 2 content");
        let pages = doc.render();
        assert_eq!(pages.len(), 2);
    }

    #[test]
    fn test_unordered_list() {
        let mut doc = PdfLayout::new();
        doc.unordered_list(vec!["Item 1".to_string(), "Item 2".to_string()]);
        let pages = doc.render();
        assert!(pages[0].content_stream.contains("Item 1"));
        assert!(pages[0].content_stream.contains("Item 2"));
    }

    #[test]
    fn test_ordered_list() {
        let mut doc = PdfLayout::new();
        doc.ordered_list(vec!["First".to_string(), "Second".to_string()]);
        let pages = doc.render();
        assert!(pages[0].content_stream.contains("1."));
        assert!(pages[0].content_stream.contains("First"));
    }

    #[test]
    fn test_table() {
        let mut doc = PdfLayout::new();
        doc.table(
            vec!["Name".to_string(), "Age".to_string()],
            vec![
                vec!["Alice".to_string(), "30".to_string()],
                vec!["Bob".to_string(), "25".to_string()],
            ],
            None,
        );
        let pages = doc.render();
        assert!(pages[0].content_stream.contains("Alice"));
    }

    #[test]
    fn test_horizontal_rule() {
        let mut doc = PdfLayout::new();
        doc.horizontal_rule();
        let pages = doc.render();
        assert!(pages[0].content_stream.contains(" l S"));
    }

    #[test]
    fn test_header_footer() {
        let doc = PdfLayout::new()
            .header(HeaderFooter {
                center: Some("Page {{page}} of {{total}}".to_string()),
                ..Default::default()
            })
            .footer(HeaderFooter {
                center: Some("Footer".to_string()),
                ..Default::default()
            });
        let mut d = doc;
        d.paragraph("Content");
        let pages = d.render();
        assert!(pages[0].content_stream.contains("Page 1 of 1"));
    }

    #[test]
    fn test_landscape() {
        let doc = PdfLayout::new()
            .page_size(PageSize::A4)
            .orientation(Orientation::Landscape);
        let mut d = doc;
        d.paragraph("Landscape");
        let pages = d.render();
        assert!(pages[0].width > pages[0].height);
    }

    #[test]
    fn test_pdf_escape() {
        assert_eq!(pdf_escape("a(b)c"), "a\\(b\\)c");
        assert_eq!(pdf_escape("a\\b"), "a\\\\b");
    }

    #[test]
    fn test_wrap_text() {
        let font = FontMetrics::helvetica();
        let lines = wrap_text("This is a test of word wrapping in the PDF layout engine", 100.0, &font, 12.0);
        assert!(lines.len() > 1);
    }

    #[test]
    fn test_to_pdf() {
        let mut doc = PdfLayout::new();
        doc.heading(1, "Test");
        doc.paragraph("Hello");
        let bytes = doc.to_pdf();
        assert!(bytes.starts_with(b"%PDF-1.4"));
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("%%EOF"));
    }

    #[test]
    fn test_rect_element() {
        let mut doc = PdfLayout::new();
        doc.rect(100.0, 100.0, 200.0, 50.0, Some(Color::red()), None);
        let pages = doc.render();
        assert!(pages[0].content_stream.contains("re f"));
    }

    #[test]
    fn test_line_element() {
        let mut doc = PdfLayout::new();
        doc.line(0.0, 0.0, 100.0, 100.0, Color::black(), 2.0);
        let pages = doc.render();
        assert!(pages[0].content_stream.contains("l S"));
    }

    #[test]
    fn test_custom_page_size() {
        let size = PageSize::Custom {
            width: 400.0,
            height: 300.0,
        };
        let (w, h) = size.dimensions();
        assert_eq!(w, 400.0);
        assert_eq!(h, 300.0);
    }

    #[test]
    fn test_spacer() {
        let mut doc = PdfLayout::new();
        doc.paragraph("Before");
        doc.spacer(50.0);
        doc.paragraph("After");
        let pages = doc.render();
        assert_eq!(pages.len(), 1);
    }

    #[test]
    fn test_courier_metrics() {
        let font = FontMetrics::courier();
        assert_eq!(font.name, "Courier");
        let w1 = font.string_width("iii", 12.0);
        let w2 = font.string_width("mmm", 12.0);
        // Monospace: same width for different chars.
        assert!((w1 - w2).abs() < 0.001);
    }

    #[test]
    fn test_text_alignment_center() {
        let mut doc = PdfLayout::new();
        let style = TextStyle {
            align: TextAlign::Center,
            ..Default::default()
        };
        doc.styled_paragraph("Centered text", style);
        let pages = doc.render();
        assert!(pages[0].content_stream.contains("Centered text"));
    }
}
