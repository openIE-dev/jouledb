//! Markdown to HTML converter.
//!
//! Supports headers, paragraphs, bold, italic, code, code blocks, links,
//! images, ordered/unordered lists, blockquotes, horizontal rules, GFM tables,
//! and task lists.

use std::fmt;

// ── AST ─────────────────────────────────────────────────────────

/// Markdown AST node.
#[derive(Debug, Clone, PartialEq)]
pub enum MdAstNode {
    Document(Vec<MdAstNode>),
    Heading { level: u8, children: Vec<MdAstNode> },
    Paragraph(Vec<MdAstNode>),
    BlockQuote(Vec<MdAstNode>),
    CodeBlock { language: Option<String>, code: String },
    UnorderedList(Vec<MdAstNode>),
    OrderedList { start: u32, items: Vec<MdAstNode> },
    ListItem { checked: Option<bool>, children: Vec<MdAstNode> },
    ThematicBreak,
    Table { headers: Vec<MdAstNode>, alignments: Vec<Align>, rows: Vec<Vec<MdAstNode>> },
    // Inline
    Text(String),
    Bold(Vec<MdAstNode>),
    Italic(Vec<MdAstNode>),
    Code(String),
    Link { url: String, title: Option<String>, children: Vec<MdAstNode> },
    Image { url: String, alt: String },
    LineBreak,
}

/// Table column alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Align {
    Left,
    Center,
    Right,
    None,
}

// ── Parser ──────────────────────────────────────────────────────

/// Parse markdown text into an AST.
pub fn parse(input: &str) -> MdAstNode {
    let lines: Vec<&str> = input.lines().collect();
    let blocks = parse_blocks(&lines);
    MdAstNode::Document(blocks)
}

fn parse_blocks(lines: &[&str]) -> Vec<MdAstNode> {
    let mut nodes = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();

        // Empty line
        if trimmed.is_empty() {
            i += 1;
            continue;
        }

        // Thematic break
        if is_thematic_break(trimmed) {
            nodes.push(MdAstNode::ThematicBreak);
            i += 1;
            continue;
        }

        // ATX heading
        if let Some(heading) = parse_atx_heading(trimmed) {
            nodes.push(heading);
            i += 1;
            continue;
        }

        // Fenced code block
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            let fence_char = if trimmed.starts_with("```") { '`' } else { '~' };
            let lang = trimmed.trim_start_matches(fence_char).trim();
            let language = if lang.is_empty() { None } else { Some(lang.to_string()) };
            let mut code = String::new();
            i += 1;
            while i < lines.len() {
                let cl = lines[i];
                if cl.trim().starts_with(fence_char) && cl.trim().chars().all(|c| c == fence_char) && cl.trim().len() >= 3 {
                    i += 1;
                    break;
                }
                if !code.is_empty() { code.push('\n'); }
                code.push_str(cl);
                i += 1;
            }
            nodes.push(MdAstNode::CodeBlock { language, code });
            continue;
        }

        // Blockquote
        if trimmed.starts_with('>') {
            let mut bq_lines = Vec::new();
            while i < lines.len() {
                let bl = lines[i].trim();
                if bl.starts_with('>') {
                    let content = bl[1..].strip_prefix(' ').unwrap_or(&bl[1..]);
                    bq_lines.push(content);
                    i += 1;
                } else if bl.is_empty() {
                    break;
                } else {
                    break;
                }
            }
            let children = parse_blocks(&bq_lines);
            nodes.push(MdAstNode::BlockQuote(children));
            continue;
        }

        // Unordered list
        if is_unordered_list_item(trimmed) {
            let mut items = Vec::new();
            while i < lines.len() {
                let il = lines[i].trim();
                if is_unordered_list_item(il) {
                    let content = &il[2..];
                    let (checked, text) = parse_task_checkbox(content);
                    items.push(MdAstNode::ListItem {
                        checked,
                        children: vec![MdAstNode::Paragraph(parse_inline(text))],
                    });
                    i += 1;
                } else if il.is_empty() {
                    break;
                } else {
                    break;
                }
            }
            nodes.push(MdAstNode::UnorderedList(items));
            continue;
        }

        // Ordered list
        if let Some(start_num) = parse_ordered_list_marker(trimmed) {
            let mut items = Vec::new();
            let mut first = true;
            let mut start = start_num;
            while i < lines.len() {
                let il = lines[i].trim();
                if let Some(num) = parse_ordered_list_marker(il) {
                    if first {
                        start = num;
                        first = false;
                    }
                    let after_dot = il.find('.').map(|p| &il[p + 1..]).unwrap_or("").trim_start();
                    items.push(MdAstNode::ListItem {
                        checked: None,
                        children: vec![MdAstNode::Paragraph(parse_inline(after_dot))],
                    });
                    i += 1;
                } else if il.is_empty() {
                    break;
                } else {
                    break;
                }
            }
            nodes.push(MdAstNode::OrderedList { start, items });
            continue;
        }

        // Table
        if i + 1 < lines.len() && is_table_separator(lines[i + 1].trim()) {
            let header_cells = parse_table_row(trimmed);
            let alignments = parse_alignments(lines[i + 1].trim());
            let mut rows = Vec::new();
            i += 2;
            while i < lines.len() {
                let rl = lines[i].trim();
                if rl.is_empty() || !rl.contains('|') { break; }
                rows.push(parse_table_row(rl).into_iter().map(|c| {
                    MdAstNode::Paragraph(parse_inline(&c))
                }).collect());
                i += 1;
            }
            let header_nodes: Vec<MdAstNode> = header_cells.into_iter().map(|c| {
                MdAstNode::Paragraph(parse_inline(&c))
            }).collect();
            nodes.push(MdAstNode::Table { headers: header_nodes, alignments, rows });
            continue;
        }

        // Paragraph (default)
        let mut para_lines = Vec::new();
        while i < lines.len() {
            let pl = lines[i].trim();
            if pl.is_empty() { break; }
            if pl.starts_with('#') || is_thematic_break(pl) || pl.starts_with("```") || pl.starts_with("~~~") { break; }
            if is_unordered_list_item(pl) { break; }
            if parse_ordered_list_marker(pl).is_some() { break; }
            if pl.starts_with('>') { break; }
            para_lines.push(pl);
            i += 1;
        }
        let text = para_lines.join(" ");
        nodes.push(MdAstNode::Paragraph(parse_inline(&text)));
    }

    nodes
}

fn is_thematic_break(s: &str) -> bool {
    let chars: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();
    if chars.len() < 3 { return false; }
    let first = chars[0];
    (first == '-' || first == '*' || first == '_') && chars.iter().all(|c| *c == first)
}

fn parse_atx_heading(s: &str) -> Option<MdAstNode> {
    if !s.starts_with('#') { return None; }
    let level = s.chars().take_while(|c| *c == '#').count();
    if level > 6 { return None; }
    let rest = s[level..].trim();
    // Remove trailing #s
    let text = rest.trim_end_matches('#').trim();
    Some(MdAstNode::Heading {
        level: level as u8,
        children: parse_inline(text),
    })
}

fn is_unordered_list_item(s: &str) -> bool {
    (s.starts_with("- ") || s.starts_with("* ") || s.starts_with("+ ")) && s.len() >= 2
}

fn parse_ordered_list_marker(s: &str) -> Option<u32> {
    let dot_pos = s.find('.')?;
    if dot_pos == 0 { return None; }
    let num_str = &s[..dot_pos];
    if !num_str.chars().all(|c| c.is_ascii_digit()) { return None; }
    if dot_pos + 1 >= s.len() || s.as_bytes()[dot_pos + 1] != b' ' { return None; }
    num_str.parse().ok()
}

fn parse_task_checkbox(s: &str) -> (Option<bool>, &str) {
    let trimmed = s.trim_start();
    if trimmed.starts_with("[x] ") || trimmed.starts_with("[X] ") {
        (Some(true), &trimmed[4..])
    } else if trimmed.starts_with("[ ] ") {
        (Some(false), &trimmed[4..])
    } else {
        (None, s)
    }
}

fn is_table_separator(s: &str) -> bool {
    let s = s.trim().trim_start_matches('|').trim_end_matches('|');
    if s.is_empty() { return false; }
    s.split('|').all(|cell| {
        let c = cell.trim();
        if c.is_empty() { return true; }
        let c = c.trim_start_matches(':').trim_end_matches(':');
        !c.is_empty() && c.chars().all(|ch| ch == '-')
    })
}

fn parse_alignments(s: &str) -> Vec<Align> {
    let s = s.trim().trim_start_matches('|').trim_end_matches('|');
    s.split('|').map(|cell| {
        let c = cell.trim();
        let left = c.starts_with(':');
        let right = c.ends_with(':');
        match (left, right) {
            (true, true) => Align::Center,
            (true, false) => Align::Left,
            (false, true) => Align::Right,
            (false, false) => Align::None,
        }
    }).collect()
}

fn parse_table_row(s: &str) -> Vec<String> {
    let s = s.trim().trim_start_matches('|').trim_end_matches('|');
    s.split('|').map(|c| c.trim().to_string()).collect()
}

// ── Inline Parser ───────────────────────────────────────────────

fn parse_inline(input: &str) -> Vec<MdAstNode> {
    let mut nodes = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    let mut text_buf = String::new();

    while i < chars.len() {
        // Escaped character
        if chars[i] == '\\' && i + 1 < chars.len() {
            text_buf.push(chars[i + 1]);
            i += 2;
            continue;
        }

        // Bold (**...**) or (__...__)
        if i + 1 < chars.len() && ((chars[i] == '*' && chars[i + 1] == '*') || (chars[i] == '_' && chars[i + 1] == '_')) {
            let marker = chars[i];
            if let Some(end) = find_closing_double(&chars, i + 2, marker) {
                flush_text(&mut text_buf, &mut nodes);
                let inner: String = chars[i + 2..end].iter().collect();
                nodes.push(MdAstNode::Bold(parse_inline(&inner)));
                i = end + 2;
                continue;
            }
        }

        // Italic (*...* or _..._)
        if (chars[i] == '*' || chars[i] == '_') && i + 1 < chars.len() && chars[i + 1] != chars[i] {
            let marker = chars[i];
            if let Some(end) = find_closing_single(&chars, i + 1, marker) {
                flush_text(&mut text_buf, &mut nodes);
                let inner: String = chars[i + 1..end].iter().collect();
                nodes.push(MdAstNode::Italic(parse_inline(&inner)));
                i = end + 1;
                continue;
            }
        }

        // Inline code
        if chars[i] == '`' {
            if let Some(end) = find_char(&chars, i + 1, '`') {
                flush_text(&mut text_buf, &mut nodes);
                let code: String = chars[i + 1..end].iter().collect();
                nodes.push(MdAstNode::Code(code));
                i = end + 1;
                continue;
            }
        }

        // Image ![alt](url)
        if chars[i] == '!' && i + 1 < chars.len() && chars[i + 1] == '[' {
            if let Some((alt, url, end)) = parse_link_or_image(&chars, i + 1) {
                flush_text(&mut text_buf, &mut nodes);
                nodes.push(MdAstNode::Image { alt, url });
                i = end;
                continue;
            }
        }

        // Link [text](url)
        if chars[i] == '[' {
            if let Some((text, url, end)) = parse_link_or_image(&chars, i) {
                flush_text(&mut text_buf, &mut nodes);
                nodes.push(MdAstNode::Link {
                    url,
                    title: None,
                    children: vec![MdAstNode::Text(text)],
                });
                i = end;
                continue;
            }
        }

        text_buf.push(chars[i]);
        i += 1;
    }

    flush_text(&mut text_buf, &mut nodes);
    nodes
}

fn flush_text(buf: &mut String, nodes: &mut Vec<MdAstNode>) {
    if !buf.is_empty() {
        nodes.push(MdAstNode::Text(std::mem::take(buf)));
    }
}

fn find_closing_double(chars: &[char], start: usize, marker: char) -> Option<usize> {
    let mut last = None;
    let mut i = start;
    while i + 1 < chars.len() {
        if chars[i] == '\\' { i += 2; continue; }
        if chars[i] == marker && chars[i + 1] == marker {
            last = Some(i);
        }
        i += 1;
    }
    last
}

fn find_closing_single(chars: &[char], start: usize, marker: char) -> Option<usize> {
    let mut i = start;
    while i < chars.len() {
        if chars[i] == '\\' { i += 2; continue; }
        if chars[i] == marker {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn find_char(chars: &[char], start: usize, ch: char) -> Option<usize> {
    (start..chars.len()).find(|j| chars[*j] == ch)
}

fn parse_link_or_image(chars: &[char], start: usize) -> Option<(String, String, usize)> {
    // start points to '['
    let bracket_end = find_char(chars, start + 1, ']')?;
    let text: String = chars[start + 1..bracket_end].iter().collect();
    let paren_start = bracket_end + 1;
    if paren_start >= chars.len() || chars[paren_start] != '(' { return None; }
    let paren_end = find_char(chars, paren_start + 1, ')')?;
    let url: String = chars[paren_start + 1..paren_end].iter().collect();
    Some((text, url, paren_end + 1))
}

// ── HTML Renderer ───────────────────────────────────────────────

/// Convert a markdown AST to HTML.
pub fn to_html(node: &MdAstNode) -> String {
    let mut out = String::new();
    render_html(node, &mut out);
    out
}

/// Parse markdown and convert directly to HTML.
pub fn markdown_to_html(input: &str) -> String {
    let ast = parse(input);
    to_html(&ast)
}

fn render_html(node: &MdAstNode, out: &mut String) {
    match node {
        MdAstNode::Document(children) => {
            for child in children {
                render_html(child, out);
            }
        }
        MdAstNode::Heading { level, children } => {
            out.push_str(&format!("<h{}>", level));
            for child in children {
                render_html(child, out);
            }
            out.push_str(&format!("</h{}>", level));
        }
        MdAstNode::Paragraph(children) => {
            out.push_str("<p>");
            for child in children {
                render_html(child, out);
            }
            out.push_str("</p>");
        }
        MdAstNode::BlockQuote(children) => {
            out.push_str("<blockquote>");
            for child in children {
                render_html(child, out);
            }
            out.push_str("</blockquote>");
        }
        MdAstNode::CodeBlock { language, code } => {
            if let Some(lang) = language {
                out.push_str(&format!("<pre><code class=\"language-{}\">", escape_html_text(lang)));
            } else {
                out.push_str("<pre><code>");
            }
            out.push_str(&escape_html_text(code));
            out.push_str("</code></pre>");
        }
        MdAstNode::UnorderedList(items) => {
            out.push_str("<ul>");
            for item in items {
                render_html(item, out);
            }
            out.push_str("</ul>");
        }
        MdAstNode::OrderedList { start, items } => {
            if *start != 1 {
                out.push_str(&format!("<ol start=\"{}\">", start));
            } else {
                out.push_str("<ol>");
            }
            for item in items {
                render_html(item, out);
            }
            out.push_str("</ol>");
        }
        MdAstNode::ListItem { checked, children } => {
            out.push_str("<li>");
            if let Some(checked_val) = checked {
                if *checked_val {
                    out.push_str("<input type=\"checkbox\" checked disabled> ");
                } else {
                    out.push_str("<input type=\"checkbox\" disabled> ");
                }
            }
            for child in children {
                render_html(child, out);
            }
            out.push_str("</li>");
        }
        MdAstNode::ThematicBreak => {
            out.push_str("<hr>");
        }
        MdAstNode::Table { headers, alignments, rows } => {
            out.push_str("<table><thead><tr>");
            for (i, header) in headers.iter().enumerate() {
                let align = alignments.get(i).copied().unwrap_or(Align::None);
                let style = align_style(align);
                out.push_str(&format!("<th{}>", style));
                render_html(header, out);
                out.push_str("</th>");
            }
            out.push_str("</tr></thead><tbody>");
            for row in rows {
                out.push_str("<tr>");
                for (i, cell) in row.iter().enumerate() {
                    let align = alignments.get(i).copied().unwrap_or(Align::None);
                    let style = align_style(align);
                    out.push_str(&format!("<td{}>", style));
                    render_html(cell, out);
                    out.push_str("</td>");
                }
                out.push_str("</tr>");
            }
            out.push_str("</tbody></table>");
        }
        MdAstNode::Text(t) => {
            out.push_str(&escape_html_text(t));
        }
        MdAstNode::Bold(children) => {
            out.push_str("<strong>");
            for child in children {
                render_html(child, out);
            }
            out.push_str("</strong>");
        }
        MdAstNode::Italic(children) => {
            out.push_str("<em>");
            for child in children {
                render_html(child, out);
            }
            out.push_str("</em>");
        }
        MdAstNode::Code(code) => {
            out.push_str("<code>");
            out.push_str(&escape_html_text(code));
            out.push_str("</code>");
        }
        MdAstNode::Link { url, children, .. } => {
            out.push_str(&format!("<a href=\"{}\">", escape_html_text(url)));
            for child in children {
                render_html(child, out);
            }
            out.push_str("</a>");
        }
        MdAstNode::Image { url, alt } => {
            out.push_str(&format!("<img src=\"{}\" alt=\"{}\">", escape_html_text(url), escape_html_text(alt)));
        }
        MdAstNode::LineBreak => {
            out.push_str("<br>");
        }
    }
}

fn align_style(align: Align) -> String {
    match align {
        Align::Left => " style=\"text-align:left\"".to_string(),
        Align::Center => " style=\"text-align:center\"".to_string(),
        Align::Right => " style=\"text-align:right\"".to_string(),
        Align::None => String::new(),
    }
}

fn escape_html_text(s: &str) -> String {
    s.replace('&', "&amp;")
     .replace('<', "&lt;")
     .replace('>', "&gt;")
     .replace('"', "&quot;")
}

impl fmt::Display for MdAstNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", to_html(self))
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heading() {
        let html = markdown_to_html("# Hello");
        assert!(html.contains("<h1>Hello</h1>"));
    }

    #[test]
    fn test_heading_levels() {
        for level in 1..=6 {
            let md = format!("{} Heading", "#".repeat(level));
            let html = markdown_to_html(&md);
            assert!(html.contains(&format!("<h{}>Heading</h{}>", level, level)));
        }
    }

    #[test]
    fn test_paragraph() {
        let html = markdown_to_html("Hello world");
        assert!(html.contains("<p>Hello world</p>"));
    }

    #[test]
    fn test_bold() {
        let html = markdown_to_html("This is **bold** text");
        assert!(html.contains("<strong>bold</strong>"));
    }

    #[test]
    fn test_italic() {
        let html = markdown_to_html("This is *italic* text");
        assert!(html.contains("<em>italic</em>"));
    }

    #[test]
    fn test_inline_code() {
        let html = markdown_to_html("Use `code` here");
        assert!(html.contains("<code>code</code>"));
    }

    #[test]
    fn test_code_block() {
        let md = "```rust\nfn main() {}\n```";
        let html = markdown_to_html(md);
        assert!(html.contains("<pre><code class=\"language-rust\">"));
        assert!(html.contains("fn main() {}"));
    }

    #[test]
    fn test_code_block_no_lang() {
        let md = "```\nplain code\n```";
        let html = markdown_to_html(md);
        assert!(html.contains("<pre><code>"));
    }

    #[test]
    fn test_link() {
        let html = markdown_to_html("[click](http://example.com)");
        assert!(html.contains("<a href=\"http://example.com\">click</a>"));
    }

    #[test]
    fn test_image() {
        let html = markdown_to_html("![alt text](img.png)");
        assert!(html.contains("<img src=\"img.png\" alt=\"alt text\">"));
    }

    #[test]
    fn test_unordered_list() {
        let md = "- one\n- two\n- three";
        let html = markdown_to_html(md);
        assert!(html.contains("<ul>"));
        assert!(html.contains("<li>"));
        assert!(html.contains("one"));
    }

    #[test]
    fn test_ordered_list() {
        let md = "1. first\n2. second\n3. third";
        let html = markdown_to_html(md);
        assert!(html.contains("<ol>"));
        assert!(html.contains("<li>"));
        assert!(html.contains("first"));
    }

    #[test]
    fn test_ordered_list_custom_start() {
        let md = "5. fifth\n6. sixth";
        let html = markdown_to_html(md);
        assert!(html.contains("<ol start=\"5\">"));
    }

    #[test]
    fn test_blockquote() {
        let md = "> quoted text";
        let html = markdown_to_html(md);
        assert!(html.contains("<blockquote>"));
        assert!(html.contains("quoted text"));
    }

    #[test]
    fn test_horizontal_rule() {
        let html = markdown_to_html("---");
        assert!(html.contains("<hr>"));
    }

    #[test]
    fn test_horizontal_rule_stars() {
        let html = markdown_to_html("***");
        assert!(html.contains("<hr>"));
    }

    #[test]
    fn test_table() {
        let md = "| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |";
        let html = markdown_to_html(md);
        assert!(html.contains("<table>"));
        assert!(html.contains("<th>"));
        assert!(html.contains("<td>"));
        assert!(html.contains("Alice"));
    }

    #[test]
    fn test_table_alignment() {
        let md = "| L | C | R |\n|:--|:--:|--:|\n| a | b | c |";
        let html = markdown_to_html(md);
        assert!(html.contains("text-align:left"));
        assert!(html.contains("text-align:center"));
        assert!(html.contains("text-align:right"));
    }

    #[test]
    fn test_task_list() {
        let md = "- [x] done\n- [ ] todo";
        let html = markdown_to_html(md);
        assert!(html.contains("checked"));
        assert!(html.contains("disabled"));
    }

    #[test]
    fn test_nested_formatting() {
        let html = markdown_to_html("**bold and *italic***");
        assert!(html.contains("<strong>"));
        assert!(html.contains("<em>"));
    }

    #[test]
    fn test_escaped_chars() {
        let html = markdown_to_html("\\*not italic\\*");
        assert!(!html.contains("<em>"));
        assert!(html.contains("*not italic*"));
    }

    #[test]
    fn test_html_escaping() {
        let html = markdown_to_html("a < b & c > d");
        assert!(html.contains("&lt;"));
        assert!(html.contains("&amp;"));
        assert!(html.contains("&gt;"));
    }

    #[test]
    fn test_empty_input() {
        let ast = parse("");
        match ast {
            MdAstNode::Document(children) => assert!(children.is_empty()),
            _ => panic!("expected document"),
        }
    }
}
