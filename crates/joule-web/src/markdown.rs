//! Markdown parser and renderer (CommonMark subset + GFM tables).
//!
//! Replaces marked, remark, and markdown-it with a pure-Rust parser
//! that produces an `MdNode` AST, renderable to HTML or plain text.

use std::fmt;

// ── AST Types ───────────────────────────────────────────────────

/// Column alignment in a GFM table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alignment {
    Left,
    Center,
    Right,
    None_,
}

/// A Markdown AST node.
#[derive(Debug, Clone, PartialEq)]
pub enum MdNode {
    Document(Vec<MdNode>),
    Heading { level: u8, children: Vec<MdNode> },
    Paragraph(Vec<MdNode>),
    BlockQuote(Vec<MdNode>),
    CodeBlock { language: Option<String>, code: String },
    UnorderedList(Vec<MdNode>),
    OrderedList { start: u32, items: Vec<MdNode> },
    ListItem(Vec<MdNode>),
    ThematicBreak,
    Table {
        headers: Vec<MdNode>,
        rows: Vec<Vec<MdNode>>,
        alignments: Vec<Alignment>,
    },
    // Inline nodes
    Text(String),
    Bold(Vec<MdNode>),
    Italic(Vec<MdNode>),
    StrikeThrough(Vec<MdNode>),
    Code(String),
    Link { url: String, title: Option<String>, children: Vec<MdNode> },
    Image { url: String, alt: String, title: Option<String> },
    LineBreak,
    HtmlInline(String),
}

impl fmt::Display for MdNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", render_html(self))
    }
}

// ── Block Parser ────────────────────────────────────────────────

/// Parse a Markdown string into an `MdNode::Document`.
pub fn parse_markdown(input: &str) -> MdNode {
    let lines: Vec<&str> = input.lines().collect();
    let blocks = parse_blocks(&lines);
    MdNode::Document(blocks)
}

fn parse_blocks(lines: &[&str]) -> Vec<MdNode> {
    let mut nodes = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];

        // Blank line — skip
        if line.trim().is_empty() {
            i += 1;
            continue;
        }

        // Thematic break: --- or *** or ___ (at least 3, optional spaces)
        let trimmed = line.trim();
        if is_thematic_break(trimmed) {
            nodes.push(MdNode::ThematicBreak);
            i += 1;
            continue;
        }

        // ATX Heading
        if let Some(heading) = parse_atx_heading(line) {
            nodes.push(heading);
            i += 1;
            continue;
        }

        // Fenced code block
        if trimmed.starts_with("```") {
            let (block, consumed) = parse_fenced_code(&lines[i..]);
            nodes.push(block);
            i += consumed;
            continue;
        }

        // Block quote
        if trimmed.starts_with("> ") || trimmed == ">" {
            let (block, consumed) = parse_block_quote(&lines[i..]);
            nodes.push(block);
            i += consumed;
            continue;
        }

        // Unordered list
        if is_unordered_list_marker(trimmed) {
            let (block, consumed) = parse_unordered_list(&lines[i..]);
            nodes.push(block);
            i += consumed;
            continue;
        }

        // Ordered list
        if is_ordered_list_marker(trimmed) {
            let (block, consumed) = parse_ordered_list(&lines[i..]);
            nodes.push(block);
            i += consumed;
            continue;
        }

        // Indented code block (4+ spaces)
        if line.starts_with("    ") || line.starts_with('\t') {
            let (block, consumed) = parse_indented_code(&lines[i..]);
            nodes.push(block);
            i += consumed;
            continue;
        }

        // Table — check if next line is a separator row
        if i + 1 < lines.len() && is_table_separator(lines[i + 1]) {
            let (table, consumed) = parse_table(&lines[i..]);
            if let Some(t) = table {
                nodes.push(t);
                i += consumed;
                continue;
            }
        }

        // Paragraph (collect lines until blank line or block-level construct)
        let (para, consumed) = parse_paragraph(&lines[i..]);
        nodes.push(para);
        i += consumed;
    }

    nodes
}

fn is_thematic_break(trimmed: &str) -> bool {
    let chars: Vec<char> = trimmed.chars().filter(|c| !c.is_whitespace()).collect();
    if chars.len() < 3 {
        return false;
    }
    let first = chars[0];
    if first != '-' && first != '*' && first != '_' {
        return false;
    }
    chars.iter().all(|c| *c == first)
}

fn parse_atx_heading(line: &str) -> Option<MdNode> {
    let trimmed = line.trim();
    if !trimmed.starts_with('#') {
        return None;
    }
    let level = trimmed.chars().take_while(|c| *c == '#').count();
    if level == 0 || level > 6 {
        return None;
    }
    let rest = &trimmed[level..];
    // Must be followed by space or be empty
    if !rest.is_empty() && !rest.starts_with(' ') {
        return None;
    }
    let content = rest.trim();
    // Remove trailing # if present
    let content = content.trim_end_matches('#').trim();
    let children = parse_inline(content);
    Some(MdNode::Heading {
        level: level as u8,
        children,
    })
}

fn parse_fenced_code(lines: &[&str]) -> (MdNode, usize) {
    let first = lines[0].trim();
    let lang_part = first.trim_start_matches('`');
    let language = if lang_part.is_empty() {
        None
    } else {
        Some(lang_part.trim().to_string())
    };

    let mut code_lines = Vec::new();
    let mut i = 1;
    while i < lines.len() {
        let trimmed = lines[i].trim();
        if trimmed.starts_with("```") {
            i += 1;
            break;
        }
        code_lines.push(lines[i]);
        i += 1;
    }

    let code = code_lines.join("\n");
    (MdNode::CodeBlock { language, code }, i)
}

fn parse_block_quote(lines: &[&str]) -> (MdNode, usize) {
    let mut inner_lines = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let trimmed = lines[i].trim();
        if trimmed.starts_with("> ") {
            inner_lines.push(&trimmed[2..]);
        } else if trimmed == ">" {
            inner_lines.push("");
        } else if trimmed.is_empty() {
            break;
        } else {
            break;
        }
        i += 1;
    }
    let inner_strs: Vec<&str> = inner_lines.iter().map(|s| *s).collect();
    let children = parse_blocks(&inner_strs);
    (MdNode::BlockQuote(children), i)
}

fn is_unordered_list_marker(trimmed: &str) -> bool {
    trimmed.starts_with("- ")
        || trimmed.starts_with("* ")
        || trimmed.starts_with("+ ")
}

fn parse_unordered_list(lines: &[&str]) -> (MdNode, usize) {
    let mut items = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let trimmed = lines[i].trim();
        if !is_unordered_list_marker(trimmed) {
            if trimmed.is_empty() {
                i += 1;
                // Check if next line continues the list
                if i < lines.len() && is_unordered_list_marker(lines[i].trim()) {
                    continue;
                }
                break;
            }
            break;
        }
        let content = &trimmed[2..];
        let children = parse_inline(content);
        items.push(MdNode::ListItem(children));
        i += 1;
    }
    (MdNode::UnorderedList(items), i)
}

fn is_ordered_list_marker(trimmed: &str) -> bool {
    if let Some(dot_pos) = trimmed.find(". ") {
        trimmed[..dot_pos].chars().all(|c| c.is_ascii_digit()) && dot_pos > 0
    } else {
        false
    }
}

fn parse_ordered_list(lines: &[&str]) -> (MdNode, usize) {
    let mut items = Vec::new();
    let mut start = 1u32;
    let mut i = 0;
    while i < lines.len() {
        let trimmed = lines[i].trim();
        if !is_ordered_list_marker(trimmed) {
            if trimmed.is_empty() {
                i += 1;
                if i < lines.len() && is_ordered_list_marker(lines[i].trim()) {
                    continue;
                }
                break;
            }
            break;
        }
        let dot_pos = trimmed.find(". ").unwrap();
        let num: u32 = trimmed[..dot_pos].parse().unwrap_or(1);
        if items.is_empty() {
            start = num;
        }
        let content = &trimmed[dot_pos + 2..];
        let children = parse_inline(content);
        items.push(MdNode::ListItem(children));
        i += 1;
    }
    (
        MdNode::OrderedList {
            start,
            items,
        },
        i,
    )
}

fn parse_indented_code(lines: &[&str]) -> (MdNode, usize) {
    let mut code_lines = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if line.starts_with("    ") {
            code_lines.push(&line[4..]);
        } else if line.starts_with('\t') {
            code_lines.push(&line[1..]);
        } else if line.trim().is_empty() {
            code_lines.push("");
        } else {
            break;
        }
        i += 1;
    }
    // Remove trailing blank lines
    while code_lines.last().is_some_and(|l| l.is_empty()) {
        code_lines.pop();
    }
    let code = code_lines.join("\n");
    (MdNode::CodeBlock { language: None, code }, i)
}

fn is_table_separator(line: &str) -> bool {
    let trimmed = line.trim();
    if !trimmed.contains('|') {
        return false;
    }
    let cells: Vec<&str> = trimmed
        .trim_start_matches('|')
        .trim_end_matches('|')
        .split('|')
        .collect();
    cells.iter().all(|cell| {
        let c = cell.trim();
        if c.is_empty() {
            return true;
        }
        c.chars().all(|ch| ch == '-' || ch == ':' || ch == ' ')
            && c.contains('-')
    })
}

fn parse_alignment(cell: &str) -> Alignment {
    let c = cell.trim();
    let left = c.starts_with(':');
    let right = c.ends_with(':');
    match (left, right) {
        (true, true) => Alignment::Center,
        (false, true) => Alignment::Right,
        (true, false) => Alignment::Left,
        (false, false) => Alignment::None_,
    }
}

fn parse_table(lines: &[&str]) -> (Option<MdNode>, usize) {
    if lines.len() < 2 {
        return (None, 0);
    }

    let header_line = lines[0].trim();
    let sep_line = lines[1].trim();

    let header_cells = split_table_row(header_line);
    let sep_cells = split_table_row(sep_line);

    let alignments: Vec<Alignment> = sep_cells.iter().map(|c| parse_alignment(c)).collect();

    let headers: Vec<MdNode> = header_cells
        .iter()
        .map(|c| {
            let children = parse_inline(c.trim());
            if children.len() == 1 {
                children.into_iter().next().unwrap()
            } else {
                MdNode::Paragraph(children)
            }
        })
        .collect();

    let mut rows = Vec::new();
    let mut i = 2;
    while i < lines.len() {
        let trimmed = lines[i].trim();
        if trimmed.is_empty() || !trimmed.contains('|') {
            break;
        }
        let cells = split_table_row(trimmed);
        let row: Vec<MdNode> = cells
            .iter()
            .map(|c| {
                let children = parse_inline(c.trim());
                if children.len() == 1 {
                    children.into_iter().next().unwrap()
                } else {
                    MdNode::Paragraph(children)
                }
            })
            .collect();
        rows.push(row);
        i += 1;
    }

    (
        Some(MdNode::Table {
            headers,
            rows,
            alignments,
        }),
        i,
    )
}

fn split_table_row(line: &str) -> Vec<String> {
    let trimmed = line.trim().trim_start_matches('|').trim_end_matches('|');
    trimmed.split('|').map(|s| s.trim().to_string()).collect()
}

fn parse_paragraph(lines: &[&str]) -> (MdNode, usize) {
    let mut text_lines = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();
        if trimmed.is_empty() {
            i += 1;
            break;
        }
        // Stop at block-level constructs
        if trimmed.starts_with('#')
            || trimmed.starts_with("```")
            || trimmed.starts_with("> ")
            || is_thematic_break(trimmed)
        {
            break;
        }
        text_lines.push(line.trim());
        i += 1;
    }
    let combined = text_lines.join(" ");
    let children = parse_inline(&combined);
    (MdNode::Paragraph(children), i)
}

// ── Inline Parser ───────────────────────────────────────────────

fn parse_inline(input: &str) -> Vec<MdNode> {
    let mut nodes = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    let mut text_buf = String::new();

    while i < chars.len() {
        // Hard line break: two trailing spaces before end or backslash
        if chars[i] == '\\' && i + 1 < chars.len() && chars[i + 1] == '\n' {
            if !text_buf.is_empty() {
                nodes.push(MdNode::Text(std::mem::take(&mut text_buf)));
            }
            nodes.push(MdNode::LineBreak);
            i += 2;
            continue;
        }

        // Inline code: `code`
        if chars[i] == '`' {
            if let Some((code, end)) = scan_inline_code(&chars, i) {
                if !text_buf.is_empty() {
                    nodes.push(MdNode::Text(std::mem::take(&mut text_buf)));
                }
                nodes.push(MdNode::Code(code));
                i = end;
                continue;
            }
        }

        // Image: ![alt](url "title")
        if chars[i] == '!' && i + 1 < chars.len() && chars[i + 1] == '[' {
            if let Some((img, end)) = scan_image(&chars, i) {
                if !text_buf.is_empty() {
                    nodes.push(MdNode::Text(std::mem::take(&mut text_buf)));
                }
                nodes.push(img);
                i = end;
                continue;
            }
        }

        // Link: [text](url "title")
        if chars[i] == '[' {
            if let Some((link, end)) = scan_link(&chars, i) {
                if !text_buf.is_empty() {
                    nodes.push(MdNode::Text(std::mem::take(&mut text_buf)));
                }
                nodes.push(link);
                i = end;
                continue;
            }
        }

        // Strikethrough: ~~text~~
        if chars[i] == '~' && i + 1 < chars.len() && chars[i + 1] == '~' {
            if let Some((node, end)) = scan_delimited(&chars, i, "~~", "~~") {
                if !text_buf.is_empty() {
                    nodes.push(MdNode::Text(std::mem::take(&mut text_buf)));
                }
                let children = parse_inline(&node);
                nodes.push(MdNode::StrikeThrough(children));
                i = end;
                continue;
            }
        }

        // Bold: **text** or __text__
        if (chars[i] == '*' && i + 1 < chars.len() && chars[i + 1] == '*')
            || (chars[i] == '_' && i + 1 < chars.len() && chars[i + 1] == '_')
        {
            let delim: String = [chars[i], chars[i]].iter().collect();
            if let Some((content, end)) = scan_delimited(&chars, i, &delim, &delim) {
                if !text_buf.is_empty() {
                    nodes.push(MdNode::Text(std::mem::take(&mut text_buf)));
                }
                let children = parse_inline(&content);
                nodes.push(MdNode::Bold(children));
                i = end;
                continue;
            }
        }

        // Italic: *text* or _text_
        if chars[i] == '*' || chars[i] == '_' {
            let delim: String = [chars[i]].iter().collect();
            if let Some((content, end)) = scan_delimited(&chars, i, &delim, &delim) {
                if !text_buf.is_empty() {
                    nodes.push(MdNode::Text(std::mem::take(&mut text_buf)));
                }
                let children = parse_inline(&content);
                nodes.push(MdNode::Italic(children));
                i = end;
                continue;
            }
        }

        // HTML inline: <tag>
        if chars[i] == '<' {
            if let Some((html, end)) = scan_html_inline(&chars, i) {
                if !text_buf.is_empty() {
                    nodes.push(MdNode::Text(std::mem::take(&mut text_buf)));
                }
                nodes.push(MdNode::HtmlInline(html));
                i = end;
                continue;
            }
        }

        text_buf.push(chars[i]);
        i += 1;
    }

    if !text_buf.is_empty() {
        nodes.push(MdNode::Text(text_buf));
    }

    nodes
}

fn scan_inline_code(chars: &[char], start: usize) -> Option<(String, usize)> {
    let mut i = start + 1;
    let mut code = String::new();
    while i < chars.len() {
        if chars[i] == '`' {
            return Some((code, i + 1));
        }
        code.push(chars[i]);
        i += 1;
    }
    None
}

fn scan_delimited(
    chars: &[char],
    start: usize,
    open: &str,
    close: &str,
) -> Option<(String, usize)> {
    let open_chars: Vec<char> = open.chars().collect();
    let close_chars: Vec<char> = close.chars().collect();

    // Verify opening delimiter
    for (j, &oc) in open_chars.iter().enumerate() {
        if start + j >= chars.len() || chars[start + j] != oc {
            return None;
        }
    }

    let content_start = start + open_chars.len();
    let mut i = content_start;
    let mut content = String::new();

    while i < chars.len() {
        // Check for closing delimiter
        let mut matches = true;
        for (j, &cc) in close_chars.iter().enumerate() {
            if i + j >= chars.len() || chars[i + j] != cc {
                matches = false;
                break;
            }
        }
        if matches && !content.is_empty() {
            // For single-char delimiters like * or _, don't match at a
            // position where the delimiter is doubled (e.g. ** inside *...*).
            if close_chars.len() == 1 {
                let cc = close_chars[0];
                let next = i + 1;
                let prev = if i > 0 { Some(chars[i - 1]) } else { None };
                if (next < chars.len() && chars[next] == cc)
                    || (prev == Some(cc))
                {
                    content.push(chars[i]);
                    i += 1;
                    continue;
                }
            }
            return Some((content, i + close_chars.len()));
        }
        content.push(chars[i]);
        i += 1;
    }
    None
}

fn scan_link(chars: &[char], start: usize) -> Option<(MdNode, usize)> {
    // [text](url "title")
    if chars[start] != '[' {
        return None;
    }
    let mut i = start + 1;
    let mut text = String::new();
    while i < chars.len() && chars[i] != ']' {
        text.push(chars[i]);
        i += 1;
    }
    if i >= chars.len() {
        return None;
    }
    i += 1; // skip ]
    if i >= chars.len() || chars[i] != '(' {
        return None;
    }
    i += 1; // skip (
    let (url, title, end) = scan_url_title(chars, i)?;
    let children = parse_inline(&text);
    Some((
        MdNode::Link {
            url,
            title,
            children,
        },
        end,
    ))
}

fn scan_image(chars: &[char], start: usize) -> Option<(MdNode, usize)> {
    // ![alt](url "title")
    if chars[start] != '!' || start + 1 >= chars.len() || chars[start + 1] != '[' {
        return None;
    }
    let mut i = start + 2;
    let mut alt = String::new();
    while i < chars.len() && chars[i] != ']' {
        alt.push(chars[i]);
        i += 1;
    }
    if i >= chars.len() {
        return None;
    }
    i += 1; // skip ]
    if i >= chars.len() || chars[i] != '(' {
        return None;
    }
    i += 1; // skip (
    let (url, title, end) = scan_url_title(chars, i)?;
    Some((MdNode::Image { url, alt, title }, end))
}

fn scan_url_title(chars: &[char], start: usize) -> Option<(String, Option<String>, usize)> {
    let mut i = start;
    let mut url = String::new();

    // Skip leading whitespace
    while i < chars.len() && chars[i] == ' ' {
        i += 1;
    }

    // Read URL
    while i < chars.len() && chars[i] != ')' && chars[i] != ' ' && chars[i] != '"' {
        url.push(chars[i]);
        i += 1;
    }

    // Optional title
    let mut title = None;
    while i < chars.len() && chars[i] == ' ' {
        i += 1;
    }
    if i < chars.len() && chars[i] == '"' {
        i += 1;
        let mut t = String::new();
        while i < chars.len() && chars[i] != '"' {
            t.push(chars[i]);
            i += 1;
        }
        if i < chars.len() {
            i += 1; // skip closing "
        }
        title = Some(t);
    }

    // Skip to closing )
    while i < chars.len() && chars[i] != ')' {
        i += 1;
    }
    if i >= chars.len() {
        return None;
    }
    Some((url, title, i + 1))
}

fn scan_html_inline(chars: &[char], start: usize) -> Option<(String, usize)> {
    if chars[start] != '<' {
        return None;
    }
    // Must start with < followed by a letter or /
    if start + 1 >= chars.len() {
        return None;
    }
    let next = chars[start + 1];
    if !next.is_ascii_alphabetic() && next != '/' {
        return None;
    }
    let mut i = start + 1;
    let mut html = String::from('<');
    while i < chars.len() {
        html.push(chars[i]);
        if chars[i] == '>' {
            return Some((html, i + 1));
        }
        i += 1;
    }
    None
}

// ── HTML Renderer ───────────────────────────────────────────────

/// Render an `MdNode` tree to an HTML string.
pub fn render_html(node: &MdNode) -> String {
    let mut out = String::new();
    render_html_inner(node, &mut out);
    out
}

fn render_html_inner(node: &MdNode, out: &mut String) {
    match node {
        MdNode::Document(children) => {
            for child in children {
                render_html_inner(child, out);
            }
        }
        MdNode::Heading { level, children } => {
            out.push_str(&format!("<h{}>", level));
            for c in children {
                render_html_inner(c, out);
            }
            out.push_str(&format!("</h{}>", level));
        }
        MdNode::Paragraph(children) => {
            out.push_str("<p>");
            for c in children {
                render_html_inner(c, out);
            }
            out.push_str("</p>");
        }
        MdNode::BlockQuote(children) => {
            out.push_str("<blockquote>");
            for c in children {
                render_html_inner(c, out);
            }
            out.push_str("</blockquote>");
        }
        MdNode::CodeBlock { language, code } => {
            if let Some(lang) = language {
                out.push_str(&format!("<pre><code class=\"language-{}\">", lang));
            } else {
                out.push_str("<pre><code>");
            }
            out.push_str(&html_escape(code));
            out.push_str("</code></pre>");
        }
        MdNode::UnorderedList(items) => {
            out.push_str("<ul>");
            for item in items {
                render_html_inner(item, out);
            }
            out.push_str("</ul>");
        }
        MdNode::OrderedList { start, items } => {
            if *start != 1 {
                out.push_str(&format!("<ol start=\"{}\">", start));
            } else {
                out.push_str("<ol>");
            }
            for item in items {
                render_html_inner(item, out);
            }
            out.push_str("</ol>");
        }
        MdNode::ListItem(children) => {
            out.push_str("<li>");
            for c in children {
                render_html_inner(c, out);
            }
            out.push_str("</li>");
        }
        MdNode::ThematicBreak => {
            out.push_str("<hr>");
        }
        MdNode::Table {
            headers,
            rows,
            alignments,
        } => {
            out.push_str("<table><thead><tr>");
            for (i, h) in headers.iter().enumerate() {
                let align = alignments.get(i).copied().unwrap_or(Alignment::None_);
                if align != Alignment::None_ {
                    out.push_str(&format!("<th style=\"text-align: {}\">", align_str(align)));
                } else {
                    out.push_str("<th>");
                }
                render_html_inner(h, out);
                out.push_str("</th>");
            }
            out.push_str("</tr></thead><tbody>");
            for row in rows {
                out.push_str("<tr>");
                for (i, cell) in row.iter().enumerate() {
                    let align = alignments.get(i).copied().unwrap_or(Alignment::None_);
                    if align != Alignment::None_ {
                        out.push_str(&format!(
                            "<td style=\"text-align: {}\">",
                            align_str(align)
                        ));
                    } else {
                        out.push_str("<td>");
                    }
                    render_html_inner(cell, out);
                    out.push_str("</td>");
                }
                out.push_str("</tr>");
            }
            out.push_str("</tbody></table>");
        }
        MdNode::Text(t) => out.push_str(&html_escape(t)),
        MdNode::Bold(children) => {
            out.push_str("<strong>");
            for c in children {
                render_html_inner(c, out);
            }
            out.push_str("</strong>");
        }
        MdNode::Italic(children) => {
            out.push_str("<em>");
            for c in children {
                render_html_inner(c, out);
            }
            out.push_str("</em>");
        }
        MdNode::StrikeThrough(children) => {
            out.push_str("<del>");
            for c in children {
                render_html_inner(c, out);
            }
            out.push_str("</del>");
        }
        MdNode::Code(code) => {
            out.push_str("<code>");
            out.push_str(&html_escape(code));
            out.push_str("</code>");
        }
        MdNode::Link {
            url,
            title,
            children,
        } => {
            out.push_str(&format!("<a href=\"{}\"", html_escape(url)));
            if let Some(t) = title {
                out.push_str(&format!(" title=\"{}\"", html_escape(t)));
            }
            out.push('>');
            for c in children {
                render_html_inner(c, out);
            }
            out.push_str("</a>");
        }
        MdNode::Image { url, alt, title } => {
            out.push_str(&format!(
                "<img src=\"{}\" alt=\"{}\"",
                html_escape(url),
                html_escape(alt)
            ));
            if let Some(t) = title {
                out.push_str(&format!(" title=\"{}\"", html_escape(t)));
            }
            out.push_str(">");
        }
        MdNode::LineBreak => out.push_str("<br>"),
        MdNode::HtmlInline(html) => out.push_str(html),
    }
}

fn align_str(a: Alignment) -> &'static str {
    match a {
        Alignment::Left => "left",
        Alignment::Center => "center",
        Alignment::Right => "right",
        Alignment::None_ => "",
    }
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ── Plain Text Renderer ─────────────────────────────────────────

/// Render an `MdNode` tree to plain text (strip all formatting).
pub fn render_plain_text(node: &MdNode) -> String {
    let mut out = String::new();
    render_plain_inner(node, &mut out);
    out.trim().to_string()
}

fn render_plain_inner(node: &MdNode, out: &mut String) {
    match node {
        MdNode::Document(children) | MdNode::Paragraph(children) | MdNode::BlockQuote(children) => {
            for c in children {
                render_plain_inner(c, out);
            }
            out.push('\n');
        }
        MdNode::Heading { children, .. } => {
            for c in children {
                render_plain_inner(c, out);
            }
            out.push('\n');
        }
        MdNode::CodeBlock { code, .. } => {
            out.push_str(code);
            out.push('\n');
        }
        MdNode::UnorderedList(items) => {
            for item in items {
                render_plain_inner(item, out);
            }
        }
        MdNode::OrderedList { items, .. } => {
            for item in items {
                render_plain_inner(item, out);
            }
        }
        MdNode::ListItem(children) => {
            for c in children {
                render_plain_inner(c, out);
            }
            out.push('\n');
        }
        MdNode::ThematicBreak => out.push('\n'),
        MdNode::Table {
            headers, rows, ..
        } => {
            for h in headers {
                render_plain_inner(h, out);
                out.push('\t');
            }
            out.push('\n');
            for row in rows {
                for cell in row {
                    render_plain_inner(cell, out);
                    out.push('\t');
                }
                out.push('\n');
            }
        }
        MdNode::Text(t) => out.push_str(t),
        MdNode::Bold(children) | MdNode::Italic(children) | MdNode::StrikeThrough(children) => {
            for c in children {
                render_plain_inner(c, out);
            }
        }
        MdNode::Code(code) => out.push_str(code),
        MdNode::Link { children, .. } => {
            for c in children {
                render_plain_inner(c, out);
            }
        }
        MdNode::Image { alt, .. } => out.push_str(alt),
        MdNode::LineBreak => out.push('\n'),
        MdNode::HtmlInline(_) => {}
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heading_levels_1_through_6() {
        for level in 1u8..=6 {
            let hashes = "#".repeat(level as usize);
            let input = format!("{} Heading {}", hashes, level);
            let doc = parse_markdown(&input);
            match &doc {
                MdNode::Document(nodes) => {
                    assert_eq!(nodes.len(), 1);
                    match &nodes[0] {
                        MdNode::Heading { level: l, children } => {
                            assert_eq!(*l, level);
                            assert_eq!(children.len(), 1);
                        }
                        other => panic!("Expected Heading, got {:?}", other),
                    }
                }
                _ => panic!("Expected Document"),
            }
        }
    }

    #[test]
    fn paragraph() {
        let doc = parse_markdown("Hello world.\n\nSecond paragraph.");
        match doc {
            MdNode::Document(nodes) => {
                assert_eq!(nodes.len(), 2);
                assert!(matches!(&nodes[0], MdNode::Paragraph(_)));
                assert!(matches!(&nodes[1], MdNode::Paragraph(_)));
            }
            _ => panic!("Expected Document"),
        }
    }

    #[test]
    fn bold_inline() {
        let doc = parse_markdown("Hello **world**.");
        let html = render_html(&doc);
        assert!(html.contains("<strong>world</strong>"));
    }

    #[test]
    fn italic_inline() {
        let doc = parse_markdown("Hello *world*.");
        let html = render_html(&doc);
        assert!(html.contains("<em>world</em>"));
    }

    #[test]
    fn code_inline() {
        let doc = parse_markdown("Use `println!` to print.");
        let html = render_html(&doc);
        assert!(html.contains("<code>println!</code>"));
    }

    #[test]
    fn nested_bold_in_italic() {
        let doc = parse_markdown("*hello **bold** world*");
        let html = render_html(&doc);
        assert!(html.contains("<em>"));
        assert!(html.contains("<strong>bold</strong>"));
    }

    #[test]
    fn link_with_title() {
        let doc = parse_markdown("[Rust](https://rust-lang.org \"The Rust Language\")");
        let html = render_html(&doc);
        assert!(html.contains("href=\"https://rust-lang.org\""));
        assert!(html.contains("title=\"The Rust Language\""));
        assert!(html.contains(">Rust</a>"));
    }

    #[test]
    fn image() {
        let doc = parse_markdown("![logo](logo.png \"Logo\")");
        let html = render_html(&doc);
        assert!(html.contains("src=\"logo.png\""));
        assert!(html.contains("alt=\"logo\""));
        assert!(html.contains("title=\"Logo\""));
    }

    #[test]
    fn code_block_with_language() {
        let input = "```rust\nfn main() {}\n```";
        let doc = parse_markdown(input);
        let html = render_html(&doc);
        assert!(html.contains("class=\"language-rust\""));
        assert!(html.contains("fn main() {}"));
    }

    #[test]
    fn block_quote() {
        let input = "> This is quoted\n> text.";
        let doc = parse_markdown(input);
        let html = render_html(&doc);
        assert!(html.contains("<blockquote>"));
    }

    #[test]
    fn unordered_list() {
        let input = "- Item 1\n- Item 2\n- Item 3";
        let doc = parse_markdown(input);
        match doc {
            MdNode::Document(nodes) => {
                assert_eq!(nodes.len(), 1);
                match &nodes[0] {
                    MdNode::UnorderedList(items) => assert_eq!(items.len(), 3),
                    other => panic!("Expected UnorderedList, got {:?}", other),
                }
            }
            _ => panic!("Expected Document"),
        }
    }

    #[test]
    fn ordered_list() {
        let input = "1. First\n2. Second";
        let doc = parse_markdown(input);
        let html = render_html(&doc);
        assert!(html.contains("<ol>"));
        assert!(html.contains("<li>First</li>"));
        assert!(html.contains("<li>Second</li>"));
    }

    #[test]
    fn table_with_alignment() {
        let input = "| Left | Center | Right |\n|:-----|:------:|------:|\n| a | b | c |";
        let doc = parse_markdown(input);
        match doc {
            MdNode::Document(nodes) => {
                assert_eq!(nodes.len(), 1);
                match &nodes[0] {
                    MdNode::Table {
                        headers,
                        rows,
                        alignments,
                    } => {
                        assert_eq!(headers.len(), 3);
                        assert_eq!(rows.len(), 1);
                        assert_eq!(alignments[0], Alignment::Left);
                        assert_eq!(alignments[1], Alignment::Center);
                        assert_eq!(alignments[2], Alignment::Right);
                    }
                    other => panic!("Expected Table, got {:?}", other),
                }
            }
            _ => panic!("Expected Document"),
        }
    }

    #[test]
    fn thematic_break() {
        let doc = parse_markdown("---");
        match doc {
            MdNode::Document(nodes) => {
                assert_eq!(nodes.len(), 1);
                assert!(matches!(&nodes[0], MdNode::ThematicBreak));
            }
            _ => panic!("Expected Document"),
        }
    }

    #[test]
    fn render_html_produces_tags() {
        let doc = parse_markdown("# Hello\n\nWorld.");
        let html = render_html(&doc);
        assert!(html.contains("<h1>Hello</h1>"));
        assert!(html.contains("<p>World.</p>"));
    }

    #[test]
    fn render_plain_text_strips_markup() {
        let doc = parse_markdown("# Hello\n\n**Bold** and *italic*.");
        let text = render_plain_text(&doc);
        assert!(text.contains("Hello"));
        assert!(text.contains("Bold"));
        assert!(text.contains("italic"));
        assert!(!text.contains('#'));
        assert!(!text.contains('*'));
    }

    #[test]
    fn empty_input() {
        let doc = parse_markdown("");
        match doc {
            MdNode::Document(nodes) => assert!(nodes.is_empty()),
            _ => panic!("Expected empty Document"),
        }
    }

    #[test]
    fn complex_document() {
        let input = "# Title\n\nSome **bold** and *italic* text.\n\n- Item 1\n- Item 2\n\n```rust\nlet x = 1;\n```\n\n---\n\n> A quote";
        let doc = parse_markdown(input);
        match doc {
            MdNode::Document(nodes) => {
                assert!(nodes.len() >= 5);
            }
            _ => panic!("Expected Document"),
        }
    }

    #[test]
    fn strikethrough() {
        let doc = parse_markdown("~~deleted~~");
        let html = render_html(&doc);
        assert!(html.contains("<del>deleted</del>"));
    }
}
