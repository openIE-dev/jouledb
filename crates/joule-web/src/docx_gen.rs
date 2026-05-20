//! Document builder — simplified OOXML document generation.
//!
//! Replaces docx.js with a pure Rust document model. Supports paragraphs,
//! runs (text spans with formatting), headings, lists, tables, images,
//! page breaks, and structured XML export.

use std::fmt;

// ── Text Formatting ────────────────────────────────────────────

/// Formatting applied to a run of text.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RunFormat {
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub strikethrough: bool,
    pub font_size: Option<u16>,
    pub font_family: Option<String>,
    pub color: Option<String>,
}

/// A run is a contiguous span of text with uniform formatting.
#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    pub text: String,
    pub format: RunFormat,
}

impl Run {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            format: RunFormat::default(),
        }
    }

    pub fn bold(mut self) -> Self {
        self.format.bold = true;
        self
    }

    pub fn italic(mut self) -> Self {
        self.format.italic = true;
        self
    }

    pub fn underline(mut self) -> Self {
        self.format.underline = true;
        self
    }

    pub fn strikethrough(mut self) -> Self {
        self.format.strikethrough = true;
        self
    }

    pub fn font_size(mut self, size: u16) -> Self {
        self.format.font_size = Some(size);
        self
    }

    pub fn font_family(mut self, family: impl Into<String>) -> Self {
        self.format.font_family = Some(family.into());
        self
    }

    pub fn color(mut self, color: impl Into<String>) -> Self {
        self.format.color = Some(color.into());
        self
    }

    fn to_xml(&self) -> String {
        let mut rpr = String::new();
        if self.format.bold {
            rpr.push_str("<w:b/>");
        }
        if self.format.italic {
            rpr.push_str("<w:i/>");
        }
        if self.format.underline {
            rpr.push_str("<w:u w:val=\"single\"/>");
        }
        if self.format.strikethrough {
            rpr.push_str("<w:strike/>");
        }
        if let Some(size) = self.format.font_size {
            // OOXML font size is in half-points.
            let half_pt = size * 2;
            rpr.push_str(&format!("<w:sz w:val=\"{half_pt}\"/>"));
        }
        if let Some(fam) = &self.format.font_family {
            rpr.push_str(&format!(
                "<w:rFonts w:ascii=\"{fam}\" w:hAnsi=\"{fam}\"/>"
            ));
        }
        if let Some(col) = &self.format.color {
            rpr.push_str(&format!("<w:color w:val=\"{col}\"/>"));
        }

        let mut xml = String::from("<w:r>");
        if !rpr.is_empty() {
            xml.push_str("<w:rPr>");
            xml.push_str(&rpr);
            xml.push_str("</w:rPr>");
        }
        xml.push_str("<w:t xml:space=\"preserve\">");
        xml.push_str(&xml_escape(&self.text));
        xml.push_str("</w:t></w:r>");
        xml
    }
}

// ── List Style ─────────────────────────────────────────────────

/// List numbering style.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListStyle {
    Bullet,
    Numbered,
}

// ── Paragraph ──────────────────────────────────────────────────

/// A paragraph element in the document.
#[derive(Debug, Clone, PartialEq)]
pub struct Paragraph {
    pub runs: Vec<Run>,
    pub heading_level: Option<u8>,
    pub list_style: Option<ListStyle>,
    pub alignment: Option<Alignment>,
}

/// Text alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alignment {
    Left,
    Center,
    Right,
    Justify,
}

impl fmt::Display for Alignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Left => write!(f, "left"),
            Self::Center => write!(f, "center"),
            Self::Right => write!(f, "right"),
            Self::Justify => write!(f, "both"),
        }
    }
}

impl Paragraph {
    pub fn new() -> Self {
        Self {
            runs: Vec::new(),
            heading_level: None,
            list_style: None,
            alignment: None,
        }
    }

    pub fn add_run(mut self, run: Run) -> Self {
        self.runs.push(run);
        self
    }

    pub fn heading(mut self, level: u8) -> Self {
        assert!((1..=6).contains(&level), "heading level must be 1-6");
        self.heading_level = Some(level);
        self
    }

    pub fn bullet(mut self) -> Self {
        self.list_style = Some(ListStyle::Bullet);
        self
    }

    pub fn numbered(mut self) -> Self {
        self.list_style = Some(ListStyle::Numbered);
        self
    }

    pub fn align(mut self, alignment: Alignment) -> Self {
        self.alignment = Some(alignment);
        self
    }

    fn to_xml(&self) -> String {
        let mut ppr = String::new();
        if let Some(level) = self.heading_level {
            ppr.push_str(&format!(
                "<w:pStyle w:val=\"Heading{level}\"/>"
            ));
        }
        if let Some(ls) = &self.list_style {
            let num_id = match ls {
                ListStyle::Bullet => 1,
                ListStyle::Numbered => 2,
            };
            ppr.push_str(&format!(
                "<w:numPr><w:ilvl w:val=\"0\"/><w:numId w:val=\"{num_id}\"/></w:numPr>"
            ));
        }
        if let Some(align) = &self.alignment {
            ppr.push_str(&format!("<w:jc w:val=\"{align}\"/>"));
        }

        let mut xml = String::from("<w:p>");
        if !ppr.is_empty() {
            xml.push_str("<w:pPr>");
            xml.push_str(&ppr);
            xml.push_str("</w:pPr>");
        }
        for run in &self.runs {
            xml.push_str(&run.to_xml());
        }
        xml.push_str("</w:p>");
        xml
    }
}

impl Default for Paragraph {
    fn default() -> Self {
        Self::new()
    }
}

// ── Table ──────────────────────────────────────────────────────

/// Merge info for a table cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeType {
    /// Start of a horizontal merge.
    HMergeStart,
    /// Continuation of a horizontal merge.
    HMergeContinue,
    /// Start of a vertical merge.
    VMergeStart,
    /// Continuation of a vertical merge.
    VMergeContinue,
}

/// A single cell in a table.
#[derive(Debug, Clone)]
pub struct TableCell {
    pub content: Vec<Paragraph>,
    pub merge: Option<MergeType>,
}

impl TableCell {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            content: vec![Paragraph::new().add_run(Run::new(text))],
            merge: None,
        }
    }

    pub fn empty() -> Self {
        Self {
            content: vec![Paragraph::new()],
            merge: None,
        }
    }

    pub fn with_merge(mut self, merge: MergeType) -> Self {
        self.merge = Some(merge);
        self
    }

    fn to_xml(&self) -> String {
        let mut xml = String::from("<w:tc>");
        if let Some(merge) = &self.merge {
            xml.push_str("<w:tcPr>");
            match merge {
                MergeType::HMergeStart => {
                    xml.push_str("<w:hMerge w:val=\"restart\"/>");
                }
                MergeType::HMergeContinue => {
                    xml.push_str("<w:hMerge/>");
                }
                MergeType::VMergeStart => {
                    xml.push_str("<w:vMerge w:val=\"restart\"/>");
                }
                MergeType::VMergeContinue => {
                    xml.push_str("<w:vMerge/>");
                }
            }
            xml.push_str("</w:tcPr>");
        }
        for para in &self.content {
            xml.push_str(&para.to_xml());
        }
        xml.push_str("</w:tc>");
        xml
    }
}

/// A row in a table.
#[derive(Debug, Clone)]
pub struct TableRow {
    pub cells: Vec<TableCell>,
}

impl TableRow {
    pub fn new(cells: Vec<TableCell>) -> Self {
        Self { cells }
    }

    fn to_xml(&self) -> String {
        let mut xml = String::from("<w:tr>");
        for cell in &self.cells {
            xml.push_str(&cell.to_xml());
        }
        xml.push_str("</w:tr>");
        xml
    }
}

/// A table in the document.
#[derive(Debug, Clone)]
pub struct Table {
    pub rows: Vec<TableRow>,
}

impl Table {
    pub fn new(rows: Vec<TableRow>) -> Self {
        Self { rows }
    }

    /// Create a simple table from a 2D grid of strings.
    pub fn from_grid(grid: &[&[&str]]) -> Self {
        let rows = grid
            .iter()
            .map(|row| {
                let cells = row.iter().map(|text| TableCell::new(*text)).collect();
                TableRow::new(cells)
            })
            .collect();
        Self { rows }
    }

    fn to_xml(&self) -> String {
        let mut xml = String::from("<w:tbl><w:tblPr><w:tblBorders>");
        for border in &["top", "left", "bottom", "right", "insideH", "insideV"] {
            xml.push_str(&format!(
                "<w:{border} w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/>"
            ));
        }
        xml.push_str("</w:tblBorders></w:tblPr>");
        for row in &self.rows {
            xml.push_str(&row.to_xml());
        }
        xml.push_str("</w:tbl>");
        xml
    }
}

// ── Image Placeholder ──────────────────────────────────────────

/// An image placeholder (no actual binary — stores path/dimensions).
#[derive(Debug, Clone, PartialEq)]
pub struct ImagePlaceholder {
    pub source: String,
    pub width_px: u32,
    pub height_px: u32,
    pub alt_text: String,
}

impl ImagePlaceholder {
    pub fn new(source: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            source: source.into(),
            width_px: width,
            height_px: height,
            alt_text: String::new(),
        }
    }

    pub fn alt(mut self, text: impl Into<String>) -> Self {
        self.alt_text = text.into();
        self
    }

    fn to_xml(&self) -> String {
        // EMU = English Metric Units. 1 pixel ≈ 9525 EMU at 96 DPI.
        let cx = self.width_px as u64 * 9525;
        let cy = self.height_px as u64 * 9525;
        format!(
            "<w:p><w:r><w:drawing><wp:inline distT=\"0\" distB=\"0\" distL=\"0\" distR=\"0\">\
             <wp:extent cx=\"{cx}\" cy=\"{cy}\"/>\
             <wp:docPr name=\"{alt}\" descr=\"{alt}\"/>\
             <a:graphic><a:graphicData uri=\"{src}\"/></a:graphic>\
             </wp:inline></w:drawing></w:r></w:p>",
            alt = xml_escape(&self.alt_text),
            src = xml_escape(&self.source),
        )
    }
}

// ── Document Element ───────────────────────────────────────────

/// An element in the document body.
#[derive(Debug, Clone)]
pub enum DocElement {
    Paragraph(Paragraph),
    Table(Table),
    Image(ImagePlaceholder),
    PageBreak,
}

// ── Document ───────────────────────────────────────────────────

/// A complete document.
#[derive(Debug, Clone)]
pub struct DocxDocument {
    pub elements: Vec<DocElement>,
    pub title: Option<String>,
}

impl DocxDocument {
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            title: None,
        }
    }

    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn add_paragraph(mut self, para: Paragraph) -> Self {
        self.elements.push(DocElement::Paragraph(para));
        self
    }

    pub fn add_table(mut self, table: Table) -> Self {
        self.elements.push(DocElement::Table(table));
        self
    }

    pub fn add_image(mut self, image: ImagePlaceholder) -> Self {
        self.elements.push(DocElement::Image(image));
        self
    }

    pub fn add_page_break(mut self) -> Self {
        self.elements.push(DocElement::PageBreak);
        self
    }

    /// Export as simplified OOXML string.
    pub fn to_xml(&self) -> String {
        let mut xml = String::from(
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\
             <w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" \
             xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" \
             xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\">\
             <w:body>",
        );

        for element in &self.elements {
            match element {
                DocElement::Paragraph(p) => xml.push_str(&p.to_xml()),
                DocElement::Table(t) => xml.push_str(&t.to_xml()),
                DocElement::Image(img) => xml.push_str(&img.to_xml()),
                DocElement::PageBreak => {
                    xml.push_str(
                        "<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>",
                    );
                }
            }
        }

        xml.push_str("</w:body></w:document>");
        xml
    }

    /// Count elements in the document.
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }
}

impl Default for DocxDocument {
    fn default() -> Self {
        Self::new()
    }
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_plain() {
        let run = Run::new("Hello");
        let xml = run.to_xml();
        assert!(xml.contains("<w:t xml:space=\"preserve\">Hello</w:t>"));
        assert!(!xml.contains("<w:rPr>"));
    }

    #[test]
    fn run_bold_italic() {
        let run = Run::new("styled").bold().italic();
        let xml = run.to_xml();
        assert!(xml.contains("<w:b/>"));
        assert!(xml.contains("<w:i/>"));
    }

    #[test]
    fn run_font_size_color() {
        let run = Run::new("big").font_size(24).color("FF0000");
        let xml = run.to_xml();
        assert!(xml.contains("<w:sz w:val=\"48\"/>"));
        assert!(xml.contains("<w:color w:val=\"FF0000\"/>"));
    }

    #[test]
    fn paragraph_heading() {
        let p = Paragraph::new()
            .heading(2)
            .add_run(Run::new("Section Title"));
        let xml = p.to_xml();
        assert!(xml.contains("<w:pStyle w:val=\"Heading2\"/>"));
    }

    #[test]
    fn paragraph_bullet_list() {
        let p = Paragraph::new()
            .bullet()
            .add_run(Run::new("Item one"));
        let xml = p.to_xml();
        assert!(xml.contains("<w:numId w:val=\"1\"/>"));
    }

    #[test]
    fn paragraph_numbered_list() {
        let p = Paragraph::new()
            .numbered()
            .add_run(Run::new("Step 1"));
        let xml = p.to_xml();
        assert!(xml.contains("<w:numId w:val=\"2\"/>"));
    }

    #[test]
    fn table_from_grid() {
        let table = Table::from_grid(&[
            &["Name", "Age"],
            &["Alice", "30"],
        ]);
        let xml = table.to_xml();
        assert!(xml.contains("<w:tbl>"));
        assert!(xml.contains("Alice"));
        assert!(xml.contains("Age"));
    }

    #[test]
    fn table_cell_merge() {
        let cell = TableCell::new("Merged")
            .with_merge(MergeType::HMergeStart);
        let xml = cell.to_xml();
        assert!(xml.contains("<w:hMerge w:val=\"restart\"/>"));
    }

    #[test]
    fn image_placeholder() {
        let img = ImagePlaceholder::new("logo.png", 100, 50).alt("Logo");
        let xml = img.to_xml();
        assert!(xml.contains("logo.png"));
        assert!(xml.contains("Logo"));
        // 100 * 9525 = 952500
        assert!(xml.contains("952500"));
    }

    #[test]
    fn page_break_xml() {
        let doc = DocxDocument::new().add_page_break();
        let xml = doc.to_xml();
        assert!(xml.contains("<w:br w:type=\"page\"/>"));
    }

    #[test]
    fn full_document() {
        let doc = DocxDocument::new()
            .title("Test Doc")
            .add_paragraph(
                Paragraph::new()
                    .heading(1)
                    .add_run(Run::new("Title")),
            )
            .add_paragraph(
                Paragraph::new()
                    .add_run(Run::new("Body text")),
            )
            .add_page_break()
            .add_table(Table::from_grid(&[&["A", "B"]]))
            .add_image(ImagePlaceholder::new("fig.png", 200, 100));

        assert_eq!(doc.element_count(), 5);
        let xml = doc.to_xml();
        assert!(xml.starts_with("<?xml version="));
        assert!(xml.contains("</w:document>"));
    }

    #[test]
    fn xml_escaping() {
        let run = Run::new("<script>alert('xss')</script>");
        let xml = run.to_xml();
        assert!(!xml.contains("<script>"));
        assert!(xml.contains("&lt;script&gt;"));
    }

    #[test]
    fn alignment() {
        let p = Paragraph::new()
            .align(Alignment::Center)
            .add_run(Run::new("centered"));
        let xml = p.to_xml();
        assert!(xml.contains("<w:jc w:val=\"center\"/>"));
    }
}
