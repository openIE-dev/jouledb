//! Code minification — JS, CSS, and HTML minifiers in pure Rust.
//!
//! Replaces Terser / cssnano / html-minifier with lightweight string
//! transforms. No AST parsing — operates on source text directly.

// ── JS Minifier ─────────────────────────────────────────────────

/// Minify JavaScript source code.
pub fn minify_js(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        // Single-line comment
        if i + 1 < len && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            // Skip to end of line
            while i < len && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        // Multi-line comment
        if i + 1 < len && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < len && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            if i + 1 < len {
                i += 2; // skip */
            }
            continue;
        }

        // String literals — preserve as-is
        if bytes[i] == b'"' || bytes[i] == b'\'' || bytes[i] == b'`' {
            let quote = bytes[i];
            out.push(char::from(quote));
            i += 1;
            while i < len && bytes[i] != quote {
                if bytes[i] == b'\\' && i + 1 < len {
                    out.push(char::from(bytes[i]));
                    out.push(char::from(bytes[i + 1]));
                    i += 2;
                } else {
                    out.push(char::from(bytes[i]));
                    i += 1;
                }
            }
            if i < len {
                out.push(char::from(bytes[i]));
                i += 1;
            }
            continue;
        }

        // Whitespace collapse
        if bytes[i].is_ascii_whitespace() {
            // Emit at most one space if needed between tokens
            while i < len && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            // Only emit a space if the previous and next chars need separation
            let prev = out.as_bytes().last().copied().unwrap_or(0);
            let next = if i < len { bytes[i] } else { 0 };
            if needs_space(prev, next) {
                out.push(' ');
            }
            continue;
        }

        out.push(char::from(bytes[i]));
        i += 1;
    }

    // Shorten true/false
    let out = out.replace("true", "!0").replace("false", "!1");

    // Remove trailing semicolons before }
    let out = out.replace(";}", "}");

    out
}

fn needs_space(prev: u8, next: u8) -> bool {
    if prev == 0 || next == 0 {
        return false;
    }
    let p_alnum = prev.is_ascii_alphanumeric() || prev == b'_' || prev == b'$';
    let n_alnum = next.is_ascii_alphanumeric() || next == b'_' || next == b'$';
    p_alnum && n_alnum
}

// ── CSS Minifier ────────────────────────────────────────────────

/// Minify CSS source code.
pub fn minify_css(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        // Multi-line comment
        if i + 1 < len && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < len && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            if i + 1 < len {
                i += 2;
            }
            continue;
        }

        // String literals in CSS
        if bytes[i] == b'"' || bytes[i] == b'\'' {
            let quote = bytes[i];
            out.push(char::from(quote));
            i += 1;
            while i < len && bytes[i] != quote {
                if bytes[i] == b'\\' && i + 1 < len {
                    out.push(char::from(bytes[i]));
                    out.push(char::from(bytes[i + 1]));
                    i += 2;
                } else {
                    out.push(char::from(bytes[i]));
                    i += 1;
                }
            }
            if i < len {
                out.push(char::from(bytes[i]));
                i += 1;
            }
            continue;
        }

        // Whitespace collapse
        if bytes[i].is_ascii_whitespace() {
            while i < len && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            let prev = out.as_bytes().last().copied().unwrap_or(0);
            let next = if i < len { bytes[i] } else { 0 };
            if needs_space(prev, next) {
                out.push(' ');
            }
            continue;
        }

        out.push(char::from(bytes[i]));
        i += 1;
    }

    // Remove trailing semicolons before }
    let out = out.replace(";}", "}");

    // Shorten 6-digit hex colors where pairs match: #aabbcc -> #abc
    shorten_hex_colors(&out)
}

fn shorten_hex_colors(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        if bytes[i] == b'#'
            && i + 6 < len
            && bytes[i + 1..i + 7].iter().all(|b| b.is_ascii_hexdigit())
        {
            // Check if next char after the 6 hex digits is NOT a hex digit (boundary)
            let boundary = i + 7 >= len || !bytes[i + 7].is_ascii_hexdigit();
            if boundary
                && bytes[i + 1] == bytes[i + 2]
                && bytes[i + 3] == bytes[i + 4]
                && bytes[i + 5] == bytes[i + 6]
            {
                out.push('#');
                out.push(char::from(bytes[i + 1]).to_ascii_lowercase());
                out.push(char::from(bytes[i + 3]).to_ascii_lowercase());
                out.push(char::from(bytes[i + 5]).to_ascii_lowercase());
                i += 7;
                continue;
            }
        }
        out.push(char::from(bytes[i]));
        i += 1;
    }
    out
}

// ── HTML Minifier ───────────────────────────────────────────────

/// Optional closing tags that browsers insert automatically.
const OPTIONAL_CLOSE_TAGS: &[&str] = &[
    "</li>", "</dt>", "</dd>", "</p>", "</tr>", "</td>", "</th>",
    "</thead>", "</tbody>", "</tfoot>", "</option>", "</colgroup>",
];

/// Minify HTML source code.
pub fn minify_html(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        // HTML comment <!-- ... -->
        if i + 3 < len && &bytes[i..i + 4] == b"<!--" {
            i += 4;
            while i + 2 < len && &bytes[i..i + 3] != b"-->" {
                i += 1;
            }
            if i + 2 < len {
                i += 3;
            }
            continue;
        }

        // Collapse whitespace between tags: >   < becomes ><
        if bytes[i] == b'>' {
            out.push('>');
            i += 1;
            // Skip whitespace between > and <
            let ws_start = i;
            while i < len && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < len && bytes[i] == b'<' {
                // Remove inter-tag whitespace
                continue;
            }
            // Not followed by <, so collapse whitespace to single space
            if i > ws_start {
                out.push(' ');
            }
            continue;
        }

        out.push(char::from(bytes[i]));
        i += 1;
    }

    // Remove optional closing tags
    let mut result = out;
    for tag in OPTIONAL_CLOSE_TAGS {
        result = result.replace(tag, "");
    }

    result
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn js_remove_single_line_comment() {
        let input = "var x = 1; // comment\nvar y = 2;";
        let out = minify_js(input);
        assert!(!out.contains("comment"));
        assert!(out.contains("var"));
    }

    #[test]
    fn js_remove_multiline_comment() {
        let input = "var x = 1; /* multi\nline */ var y = 2;";
        let out = minify_js(input);
        assert!(!out.contains("multi"));
        assert!(out.contains("var x"));
    }

    #[test]
    fn js_collapse_whitespace() {
        let input = "var   x   =   1  ;";
        let out = minify_js(input);
        assert!(!out.contains("  "));
    }

    #[test]
    fn js_shorten_booleans() {
        let input = "var a = true; var b = false;";
        let out = minify_js(input);
        assert!(out.contains("!0"));
        assert!(out.contains("!1"));
    }

    #[test]
    fn js_preserve_strings() {
        let input = r#"var x = "hello   world";"#;
        let out = minify_js(input);
        assert!(out.contains("hello   world"));
    }

    #[test]
    fn js_remove_semicolons_before_brace() {
        let input = "function f() { return 1; }";
        let out = minify_js(input);
        assert!(out.contains("return 1}"));
    }

    #[test]
    fn css_remove_comments() {
        let input = "body { /* color */ color: red; }";
        let out = minify_css(input);
        assert!(!out.contains("/* color */"));
        assert!(out.contains("color:red"));
    }

    #[test]
    fn css_shorten_hex() {
        let input = "body{color:#ffffff}";
        let out = minify_css(input);
        assert!(out.contains("#fff"));
        assert!(!out.contains("#ffffff"));
    }

    #[test]
    fn css_trailing_semicolons() {
        let input = "body { color: red; }";
        let out = minify_css(input);
        assert!(!out.contains(";}"));
    }

    #[test]
    fn html_remove_comments() {
        let input = "<div><!-- comment --><p>hi</p></div>";
        let out = minify_html(input);
        assert!(!out.contains("comment"));
        assert!(out.contains("<div>"));
    }

    #[test]
    fn html_collapse_inter_tag_whitespace() {
        let input = "<div>   <span>hi</span>   </div>";
        let out = minify_html(input);
        assert!(out.contains("<div><span>"));
    }

    #[test]
    fn html_remove_optional_close_tags() {
        let input = "<ul><li>a</li><li>b</li></ul>";
        let out = minify_html(input);
        assert!(!out.contains("</li>"));
    }

    #[test]
    fn css_no_shorten_short_hex() {
        // Already short or not matching pattern
        let input = "body{color:#abc}";
        let out = minify_css(input);
        assert!(out.contains("#abc"));
    }
}
