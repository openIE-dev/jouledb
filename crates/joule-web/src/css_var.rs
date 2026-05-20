//! CSS custom properties (variables): scoped registries, fallbacks,
//! circular reference detection, and property value resolution.

use std::collections::{HashMap, HashSet};

// ── Variable Registry ───────────────────────────────────────────

/// A scoped registry of CSS custom properties.
#[derive(Debug, Clone)]
pub struct VarRegistry {
    /// Own variables defined in this scope.
    vars: HashMap<String, VarEntry>,
    /// Optional parent scope for inheritance.
    parent: Option<Box<VarRegistry>>,
}

#[derive(Debug, Clone)]
struct VarEntry {
    value: String,
    inherited: bool,
}

impl VarRegistry {
    /// Create a root registry with no parent.
    pub fn new() -> Self {
        Self { vars: HashMap::new(), parent: None }
    }

    /// Create a child scope that inherits from this registry.
    pub fn child(&self) -> Self {
        Self {
            vars: HashMap::new(),
            parent: Some(Box::new(self.clone())),
        }
    }

    /// Define a custom property (inherited by default).
    pub fn define(&mut self, name: &str, value: &str) {
        self.vars.insert(name.to_string(), VarEntry {
            value: value.to_string(),
            inherited: true,
        });
    }

    /// Define a non-inherited custom property.
    pub fn define_non_inherited(&mut self, name: &str, value: &str) {
        self.vars.insert(name.to_string(), VarEntry {
            value: value.to_string(),
            inherited: false,
        });
    }

    /// Resolve a variable by name, checking this scope then ancestors.
    /// Only inherited vars propagate up.
    pub fn resolve(&self, name: &str) -> Option<&str> {
        if let Some(entry) = self.vars.get(name) {
            return Some(&entry.value);
        }
        // Walk parent chain — only inherited vars propagate
        if let Some(parent) = &self.parent {
            if let Some(entry) = parent.find_entry(name) {
                if entry.inherited {
                    return Some(&entry.value);
                }
            }
        }
        None
    }

    /// Resolve with a fallback value.
    pub fn resolve_with_fallback<'a>(&'a self, name: &str, fallback: &'a str) -> &'a str {
        self.resolve(name).unwrap_or(fallback)
    }

    fn find_entry(&self, name: &str) -> Option<&VarEntry> {
        if let Some(entry) = self.vars.get(name) {
            return Some(entry);
        }
        if let Some(parent) = &self.parent {
            return parent.find_entry(name);
        }
        None
    }
}

impl Default for VarRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── var() Parsing ───────────────────────────────────────────────

/// Parsed result of a `var(--name, fallback)` expression.
#[derive(Debug, Clone, PartialEq)]
pub struct VarCall {
    pub name: String,
    pub fallback: Option<String>,
}

/// Parse a `var(--name)` or `var(--name, fallback)` expression.
pub fn parse_var_call(input: &str) -> Option<VarCall> {
    let trimmed = input.trim();
    if !trimmed.starts_with("var(") || !trimmed.ends_with(')') {
        return None;
    }
    let inner = &trimmed[4..trimmed.len() - 1];
    if let Some(comma_pos) = inner.find(',') {
        let name = inner[..comma_pos].trim().to_string();
        let fallback = inner[comma_pos + 1..].trim().to_string();
        Some(VarCall {
            name,
            fallback: if fallback.is_empty() { None } else { Some(fallback) },
        })
    } else {
        Some(VarCall {
            name: inner.trim().to_string(),
            fallback: None,
        })
    }
}

// ── Circular Reference Detection ────────────────────────────────

/// Check whether resolving variables in the registry would create a cycle.
/// Returns the names of variables involved in cycles.
pub fn detect_cycles(registry: &VarRegistry) -> Vec<String> {
    let mut cyclic = Vec::new();
    for name in registry.vars.keys() {
        let mut visited = HashSet::new();
        if has_cycle(registry, name, &mut visited) {
            cyclic.push(name.clone());
        }
    }
    cyclic.sort();
    cyclic
}

fn has_cycle(registry: &VarRegistry, name: &str, visited: &mut HashSet<String>) -> bool {
    if visited.contains(name) {
        return true;
    }
    visited.insert(name.to_string());
    if let Some(value) = registry.resolve(name) {
        let value = value.to_string();
        // Find all var() references in the value
        let refs = extract_var_refs(&value);
        for var_ref in refs {
            if has_cycle(registry, &var_ref, visited) {
                return true;
            }
        }
    }
    visited.remove(name);
    false
}

fn extract_var_refs(value: &str) -> Vec<String> {
    let mut refs = Vec::new();
    let mut search_from = 0;
    while let Some(start) = value[search_from..].find("var(") {
        let abs_start = search_from + start;
        // Find matching close paren
        let inner_start = abs_start + 4;
        if let Some(end) = find_matching_paren(&value[inner_start..]) {
            let inner = &value[inner_start..inner_start + end];
            // Extract just the name (before comma)
            let name = if let Some(comma) = inner.find(',') {
                inner[..comma].trim()
            } else {
                inner.trim()
            };
            refs.push(name.to_string());
            search_from = inner_start + end + 1;
        } else {
            break;
        }
    }
    refs
}

fn find_matching_paren(s: &str) -> Option<usize> {
    let mut depth = 0u32;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                if depth == 0 {
                    return Some(i);
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    None
}

// ── Full Resolution ─────────────────────────────────────────────

/// Resolve all `var()` references in a property value string.
/// Returns `None` if a circular reference is detected.
pub fn resolve_value(registry: &VarRegistry, value: &str) -> Option<String> {
    resolve_value_inner(registry, value, &mut HashSet::new())
}

fn resolve_value_inner(
    registry: &VarRegistry,
    value: &str,
    resolving: &mut HashSet<String>,
) -> Option<String> {
    let mut result = String::with_capacity(value.len());
    let mut pos = 0;
    let bytes = value.as_bytes();

    while pos < value.len() {
        if value[pos..].starts_with("var(") {
            let inner_start = pos + 4;
            if let Some(end) = find_matching_paren(&value[inner_start..]) {
                let inner = &value[inner_start..inner_start + end];
                let (var_name, fallback) = if let Some(comma) = inner.find(',') {
                    (inner[..comma].trim(), Some(inner[comma + 1..].trim()))
                } else {
                    (inner.trim(), None)
                };

                if resolving.contains(var_name) {
                    return None; // Circular!
                }

                resolving.insert(var_name.to_string());
                let resolved = if let Some(val) = registry.resolve(var_name) {
                    resolve_value_inner(registry, val, resolving)
                } else if let Some(fb) = fallback {
                    resolve_value_inner(registry, fb, resolving)
                } else {
                    None
                };
                resolving.remove(var_name);

                match resolved {
                    Some(v) => result.push_str(&v),
                    None => {
                        if let Some(fb) = fallback {
                            result.push_str(fb);
                        } else {
                            return None;
                        }
                    }
                }

                pos = inner_start + end + 1;
                continue;
            }
        }
        result.push(bytes[pos] as char);
        pos += 1;
    }

    Some(result)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn define_and_resolve() {
        let mut reg = VarRegistry::new();
        reg.define("--primary", "#007bff");
        assert_eq!(reg.resolve("--primary"), Some("#007bff"));
    }

    #[test]
    fn resolve_missing_returns_none() {
        let reg = VarRegistry::new();
        assert_eq!(reg.resolve("--missing"), None);
    }

    #[test]
    fn resolve_with_fallback_value() {
        let reg = VarRegistry::new();
        assert_eq!(reg.resolve_with_fallback("--missing", "red"), "red");
    }

    #[test]
    fn child_inherits_parent() {
        let mut parent = VarRegistry::new();
        parent.define("--color", "blue");
        let child = parent.child();
        assert_eq!(child.resolve("--color"), Some("blue"));
    }

    #[test]
    fn child_overrides_parent() {
        let mut parent = VarRegistry::new();
        parent.define("--color", "blue");
        let mut child = parent.child();
        child.define("--color", "red");
        assert_eq!(child.resolve("--color"), Some("red"));
    }

    #[test]
    fn non_inherited_does_not_propagate() {
        let mut parent = VarRegistry::new();
        parent.define_non_inherited("--local", "secret");
        let child = parent.child();
        assert_eq!(child.resolve("--local"), None);
    }

    #[test]
    fn parse_var_call_simple() {
        let result = parse_var_call("var(--color)").unwrap();
        assert_eq!(result.name, "--color");
        assert_eq!(result.fallback, None);
    }

    #[test]
    fn parse_var_call_with_fallback() {
        let result = parse_var_call("var(--color, red)").unwrap();
        assert_eq!(result.name, "--color");
        assert_eq!(result.fallback, Some("red".to_string()));
    }

    #[test]
    fn parse_var_call_invalid() {
        assert!(parse_var_call("not-a-var").is_none());
    }

    #[test]
    fn resolve_value_simple() {
        let mut reg = VarRegistry::new();
        reg.define("--bg", "white");
        let result = resolve_value(&reg, "background: var(--bg)").unwrap();
        assert_eq!(result, "background: white");
    }

    #[test]
    fn resolve_value_nested() {
        let mut reg = VarRegistry::new();
        reg.define("--base", "10px");
        reg.define("--spacing", "var(--base)");
        let result = resolve_value(&reg, "margin: var(--spacing)").unwrap();
        assert_eq!(result, "margin: 10px");
    }

    #[test]
    fn resolve_value_with_fallback() {
        let reg = VarRegistry::new();
        let result = resolve_value(&reg, "color: var(--missing, blue)").unwrap();
        assert_eq!(result, "color: blue");
    }

    #[test]
    fn detect_circular_reference() {
        let mut reg = VarRegistry::new();
        reg.define("--a", "var(--b)");
        reg.define("--b", "var(--a)");
        let cycles = detect_cycles(&reg);
        assert!(cycles.contains(&"--a".to_string()));
        assert!(cycles.contains(&"--b".to_string()));
    }

    #[test]
    fn resolve_circular_returns_none() {
        let mut reg = VarRegistry::new();
        reg.define("--a", "var(--b)");
        reg.define("--b", "var(--a)");
        assert!(resolve_value(&reg, "var(--a)").is_none());
    }

    #[test]
    fn multiple_vars_in_one_value() {
        let mut reg = VarRegistry::new();
        reg.define("--x", "10px");
        reg.define("--y", "20px");
        let result = resolve_value(&reg, "var(--x) var(--y)").unwrap();
        assert_eq!(result, "10px 20px");
    }
}
