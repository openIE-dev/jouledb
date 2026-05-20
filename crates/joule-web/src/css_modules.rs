//! CSS Modules — scoped class names with deterministic hashing.
//!
//! Replaces `css-modules` / `postcss-modules` with a pure Rust implementation.
//! Parses CSS, generates unique scoped names, handles `composes`, and outputs
//! a rewritten stylesheet plus an export map.

use std::collections::HashMap;

// ── Hashing ─────────────────────────────────────────────────────

fn fnv1a(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Generate a scoped class name: `filename_classname_hash`.
fn scoped_name(filename: &str, class_name: &str) -> String {
    let stem = filename
        .rsplit('/')
        .next()
        .unwrap_or(filename)
        .trim_end_matches(".module.css")
        .trim_end_matches(".css");
    let hash_input = format!("{filename}:{class_name}");
    let hash = fnv1a(hash_input.as_bytes());
    let short_hash = format!("{:x}", hash & 0xfffff); // 5 hex chars
    format!("{stem}_{class_name}_{short_hash}")
}

// ── Scope ───────────────────────────────────────────────────────

/// The scope of a CSS selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    Local,
    Global,
}

// ── Parsed class ────────────────────────────────────────────────

/// A class extracted from CSS Modules source.
#[derive(Debug, Clone)]
pub struct ParsedClass {
    pub original_name: String,
    pub scoped_name: String,
    pub scope: Scope,
    /// Classes this composes from (scoped names).
    pub composes: Vec<String>,
}

// ── CSS Module ──────────────────────────────────────────────────

/// Result of processing a CSS module file.
#[derive(Debug, Clone)]
pub struct CssModule {
    /// The filename this module was parsed from.
    pub filename: String,
    /// Parsed class definitions.
    pub classes: Vec<ParsedClass>,
    /// Export map: original class name -> scoped class name(s).
    pub exports: HashMap<String, String>,
    /// Rewritten CSS with scoped class names.
    pub output_css: String,
}

impl CssModule {
    /// Look up the scoped name for an original class.
    pub fn scoped(&self, original: &str) -> Option<&str> {
        self.exports.get(original).map(|s| s.as_str())
    }
}

// ── Parser ──────────────────────────────────────────────────────

/// Process a CSS module file, generating scoped class names.
pub fn process_css_module(filename: &str, css: &str) -> CssModule {
    let mut classes = Vec::new();
    let mut exports = HashMap::new();
    let mut output_css = String::with_capacity(css.len());

    // External compose references: class_name -> [(source_file, source_class)]
    let mut compose_refs: HashMap<String, Vec<(String, String)>> = HashMap::new();

    // Simple parser: find selectors (.className) and rule blocks
    let bytes = css.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        // Skip comments
        if i + 1 < len && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            output_css.push('/');
            output_css.push('*');
            i += 2;
            while i + 1 < len && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                output_css.push(char::from(bytes[i]));
                i += 1;
            }
            if i + 1 < len {
                output_css.push('*');
                output_css.push('/');
                i += 2;
            }
            continue;
        }

        // Detect :global(.className) — pass through unscoped
        if i + 8 < len && &css[i..i + 8] == ":global(" {
            let start = i;
            i += 8;
            let paren_start = i;
            let mut depth = 1;
            while i < len && depth > 0 {
                if bytes[i] == b'(' {
                    depth += 1;
                } else if bytes[i] == b')' {
                    depth -= 1;
                }
                if depth > 0 {
                    i += 1;
                }
            }
            let inner = &css[paren_start..i];
            // Extract class names from inner
            for class in extract_class_names(inner) {
                let pc = ParsedClass {
                    original_name: class.clone(),
                    scoped_name: class.clone(),
                    scope: Scope::Global,
                    composes: Vec::new(),
                };
                exports.insert(class.clone(), class.clone());
                classes.push(pc);
            }
            output_css.push_str(inner);
            if i < len {
                i += 1; // skip )
            }
            continue;
        }

        // Detect .className selector
        if bytes[i] == b'.' && (i == 0 || !bytes[i - 1].is_ascii_alphanumeric()) {
            i += 1;
            let name_start = i;
            while i < len
                && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'-')
            {
                i += 1;
            }
            let class_name = &css[name_start..i];
            if class_name.is_empty() {
                output_css.push('.');
                continue;
            }

            let scoped = scoped_name(filename, class_name);

            // Check if this class already registered
            if !exports.contains_key(class_name) {
                let pc = ParsedClass {
                    original_name: class_name.to_string(),
                    scoped_name: scoped.clone(),
                    scope: Scope::Local,
                    composes: Vec::new(),
                };
                classes.push(pc);
                exports.insert(class_name.to_string(), scoped.clone());
            }

            output_css.push('.');
            output_css.push_str(&scoped);

            // Look ahead for { ... } block to find composes declarations
            let mut j = i;
            while j < len && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j < len && bytes[j] == b'{' {
                // Copy whitespace
                output_css.push_str(&css[i..j]);
                output_css.push('{');
                j += 1;
                let block_start = j;
                let mut depth = 1;
                let mut block_end = j;
                while j < len && depth > 0 {
                    if bytes[j] == b'{' {
                        depth += 1;
                    } else if bytes[j] == b'}' {
                        depth -= 1;
                        if depth == 0 {
                            block_end = j;
                        }
                    }
                    j += 1;
                }
                let block = &css[block_start..block_end];

                // Parse composes declarations
                let mut filtered_block = String::new();
                for decl in block.split(';') {
                    let trimmed = decl.trim();
                    if trimmed.starts_with("composes:") {
                        let value = trimmed["composes:".len()..].trim();
                        // composes: className from "file.css"
                        if let Some(from_pos) = value.find(" from ") {
                            let composed_class = value[..from_pos].trim();
                            let source = value[from_pos + 6..]
                                .trim()
                                .trim_matches('"')
                                .trim_matches('\'');
                            compose_refs
                                .entry(class_name.to_string())
                                .or_default()
                                .push((source.to_string(), composed_class.to_string()));
                        } else {
                            // composes: localClassName (same file)
                            let composed_class = value.trim();
                            if let Some(composed_scoped) = exports.get(composed_class).cloned() {
                                // Add to composes list for this class
                                if let Some(pc) =
                                    classes.iter_mut().find(|c| c.original_name == class_name)
                                {
                                    pc.composes.push(composed_scoped.clone());
                                }
                                // Update export to include composed class
                                if let Some(exp) = exports.get_mut(class_name) {
                                    *exp = format!("{} {}", exp, composed_scoped);
                                }
                            }
                        }
                    } else if !trimmed.is_empty() {
                        filtered_block.push_str(decl);
                        filtered_block.push(';');
                    }
                }

                output_css.push_str(&filtered_block);
                output_css.push('}');
                i = j;
            }
            continue;
        }

        output_css.push(char::from(bytes[i]));
        i += 1;
    }

    CssModule {
        filename: filename.to_string(),
        classes,
        exports,
        output_css,
    }
}

fn extract_class_names(selector: &str) -> Vec<String> {
    let mut names = Vec::new();
    let bytes = selector.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        if bytes[i] == b'.' {
            i += 1;
            let start = i;
            while i < len
                && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'-')
            {
                i += 1;
            }
            if i > start {
                names.push(selector[start..i].to_string());
            }
        } else {
            i += 1;
        }
    }
    names
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_name_deterministic() {
        let a = scoped_name("button.module.css", "primary");
        let b = scoped_name("button.module.css", "primary");
        assert_eq!(a, b);
    }

    #[test]
    fn scoped_name_format() {
        let name = scoped_name("components/button.module.css", "primary");
        assert!(name.starts_with("button_primary_"));
        assert!(name.len() > "button_primary_".len());
    }

    #[test]
    fn different_files_different_hashes() {
        let a = scoped_name("a.css", "btn");
        let b = scoped_name("b.css", "btn");
        assert_ne!(a, b);
    }

    #[test]
    fn process_basic_class() {
        let css = ".title { color: red; }";
        let module = process_css_module("app.module.css", css);
        assert!(module.exports.contains_key("title"));
        let scoped = module.scoped("title").unwrap();
        assert!(scoped.starts_with("app_title_"));
        assert!(module.output_css.contains(scoped));
    }

    #[test]
    fn process_multiple_classes() {
        let css = ".header { color: blue; }\n.footer { color: green; }";
        let module = process_css_module("layout.module.css", css);
        assert_eq!(module.exports.len(), 2);
        assert!(module.exports.contains_key("header"));
        assert!(module.exports.contains_key("footer"));
    }

    #[test]
    fn global_scope_not_scoped() {
        let css = ":global(.body) { margin: 0; }";
        let module = process_css_module("reset.module.css", css);
        assert_eq!(module.scoped("body"), Some("body"));
    }

    #[test]
    fn composes_same_file() {
        let css = ".base { font-size: 14px; }\n.title { composes: base; color: red; }";
        let module = process_css_module("app.module.css", css);
        let title_export = module.scoped("title").unwrap();
        // Should contain both scoped names
        assert!(title_export.contains("app_base_"));
    }

    #[test]
    fn output_css_contains_scoped_selectors() {
        let css = ".btn { padding: 8px; }";
        let module = process_css_module("ui.module.css", css);
        // Original class name should NOT appear in output CSS as a selector
        assert!(!module.output_css.contains(".btn "));
        assert!(!module.output_css.contains(".btn{"));
        // Scoped name should appear
        let scoped = module.scoped("btn").unwrap();
        assert!(module.output_css.contains(scoped));
    }

    #[test]
    fn export_map_correct_count() {
        let css = ".a { color: red; }\n.b { color: blue; }\n.c { color: green; }";
        let module = process_css_module("test.module.css", css);
        assert_eq!(module.exports.len(), 3);
    }

    #[test]
    fn nested_selector_handled() {
        let css = ".parent .child { color: red; }";
        let module = process_css_module("nest.module.css", css);
        assert!(module.exports.contains_key("parent"));
        assert!(module.exports.contains_key("child"));
    }

    #[test]
    fn css_module_filename() {
        let module = process_css_module("my-file.module.css", ".x { color: red; }");
        assert_eq!(module.filename, "my-file.module.css");
    }

    #[test]
    fn fnv1a_hash_consistency() {
        let h1 = fnv1a(b"test");
        let h2 = fnv1a(b"test");
        assert_eq!(h1, h2);
        assert_ne!(fnv1a(b"test"), fnv1a(b"other"));
    }
}
