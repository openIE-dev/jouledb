//! Theming system — design tokens, color schemes, and theme switching.
//!
//! Replaces `styled-system`, `theme-ui`, and manual CSS-variable management
//! with a typed Rust API.

use std::collections::HashMap;

// ── Theme mode ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ThemeMode {
    Light,
    Dark,
    System,
    Custom(String),
}

// ── Design tokens ───────────────────────────────────────────────────────────

/// A flat key-value map of design tokens.
#[derive(Debug, Clone, Default)]
pub struct DesignTokens {
    tokens: HashMap<String, String>,
}

impl DesignTokens {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, key: &str, value: &str) -> &mut Self {
        self.tokens.insert(key.to_string(), value.to_string());
        self
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.tokens.get(key).map(|s| s.as_str())
    }

    pub fn merge(&mut self, other: &DesignTokens) {
        for (k, v) in &other.tokens {
            self.tokens.insert(k.clone(), v.clone());
        }
    }

    /// Render tokens as CSS custom property declarations.
    pub fn to_css_variables(&self) -> String {
        let mut lines: Vec<String> = self
            .tokens
            .iter()
            .map(|(k, v)| format!("--{k}: {v};"))
            .collect();
        lines.sort(); // deterministic output
        lines.join("\n")
    }

    /// Wrap in a `:root` block.
    pub fn to_css_root(&self) -> String {
        format!(":root {{\n{}\n}}", self.to_css_variables())
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

// ── Semantic structs ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ColorScheme {
    pub primary: String,
    pub secondary: String,
    pub accent: String,
    pub background: String,
    pub surface: String,
    pub text: String,
    pub text_secondary: String,
    pub error: String,
    pub warning: String,
    pub success: String,
    pub info: String,
    pub border: String,
    pub shadow: String,
}

#[derive(Debug, Clone)]
pub struct Spacing {
    pub xs: String,
    pub sm: String,
    pub md: String,
    pub lg: String,
    pub xl: String,
    pub xxl: String,
}

#[derive(Debug, Clone)]
pub struct Typography {
    pub font_family: String,
    pub font_family_mono: String,
    pub size_xs: String,
    pub size_sm: String,
    pub size_base: String,
    pub size_lg: String,
    pub size_xl: String,
    pub size_xxl: String,
    pub weight_normal: String,
    pub weight_medium: String,
    pub weight_bold: String,
    pub line_height_tight: String,
    pub line_height_normal: String,
    pub line_height_relaxed: String,
}

#[derive(Debug, Clone)]
pub struct Breakpoints {
    pub sm: f64,
    pub md: f64,
    pub lg: f64,
    pub xl: f64,
    pub xxl: f64,
}

// ── Theme ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Theme {
    pub name: String,
    pub mode: ThemeMode,
    pub colors: ColorScheme,
    pub spacing: Spacing,
    pub typography: Typography,
    pub breakpoints: Breakpoints,
    pub tokens: DesignTokens,
    pub border_radius: String,
    pub shadow_sm: String,
    pub shadow_md: String,
    pub shadow_lg: String,
    pub transition_fast: String,
    pub transition_normal: String,
    pub transition_slow: String,
}

impl Theme {
    /// A sensible default light theme.
    pub fn light() -> Self {
        Self {
            name: "light".into(),
            mode: ThemeMode::Light,
            colors: ColorScheme {
                primary: "#3b82f6".into(),
                secondary: "#6366f1".into(),
                accent: "#f59e0b".into(),
                background: "#ffffff".into(),
                surface: "#f9fafb".into(),
                text: "#111827".into(),
                text_secondary: "#6b7280".into(),
                error: "#ef4444".into(),
                warning: "#f59e0b".into(),
                success: "#10b981".into(),
                info: "#3b82f6".into(),
                border: "#e5e7eb".into(),
                shadow: "rgba(0,0,0,0.1)".into(),
            },
            spacing: default_spacing(),
            typography: default_typography(),
            breakpoints: default_breakpoints(),
            tokens: DesignTokens::new(),
            border_radius: "0.375rem".into(),
            shadow_sm: "0 1px 2px 0 rgba(0,0,0,0.05)".into(),
            shadow_md: "0 4px 6px -1px rgba(0,0,0,0.1)".into(),
            shadow_lg: "0 10px 15px -3px rgba(0,0,0,0.1)".into(),
            transition_fast: "150ms ease".into(),
            transition_normal: "300ms ease".into(),
            transition_slow: "500ms ease".into(),
        }
    }

    /// A sensible default dark theme.
    pub fn dark() -> Self {
        Self {
            name: "dark".into(),
            mode: ThemeMode::Dark,
            colors: ColorScheme {
                primary: "#60a5fa".into(),
                secondary: "#818cf8".into(),
                accent: "#fbbf24".into(),
                background: "#111827".into(),
                surface: "#1f2937".into(),
                text: "#f9fafb".into(),
                text_secondary: "#9ca3af".into(),
                error: "#f87171".into(),
                warning: "#fbbf24".into(),
                success: "#34d399".into(),
                info: "#60a5fa".into(),
                border: "#374151".into(),
                shadow: "rgba(0,0,0,0.3)".into(),
            },
            spacing: default_spacing(),
            typography: default_typography(),
            breakpoints: default_breakpoints(),
            tokens: DesignTokens::new(),
            border_radius: "0.375rem".into(),
            shadow_sm: "0 1px 2px 0 rgba(0,0,0,0.2)".into(),
            shadow_md: "0 4px 6px -1px rgba(0,0,0,0.3)".into(),
            shadow_lg: "0 10px 15px -3px rgba(0,0,0,0.3)".into(),
            transition_fast: "150ms ease".into(),
            transition_normal: "300ms ease".into(),
            transition_slow: "500ms ease".into(),
        }
    }

    /// Render all theme values as CSS custom properties.
    pub fn to_css_variables(&self) -> String {
        let mut lines = Vec::new();
        let c = &self.colors;
        lines.push(format!("--color-primary: {};", c.primary));
        lines.push(format!("--color-secondary: {};", c.secondary));
        lines.push(format!("--color-accent: {};", c.accent));
        lines.push(format!("--color-background: {};", c.background));
        lines.push(format!("--color-surface: {};", c.surface));
        lines.push(format!("--color-text: {};", c.text));
        lines.push(format!("--color-text-secondary: {};", c.text_secondary));
        lines.push(format!("--color-error: {};", c.error));
        lines.push(format!("--color-warning: {};", c.warning));
        lines.push(format!("--color-success: {};", c.success));
        lines.push(format!("--color-info: {};", c.info));
        lines.push(format!("--color-border: {};", c.border));
        lines.push(format!("--color-shadow: {};", c.shadow));

        let s = &self.spacing;
        lines.push(format!("--spacing-xs: {};", s.xs));
        lines.push(format!("--spacing-sm: {};", s.sm));
        lines.push(format!("--spacing-md: {};", s.md));
        lines.push(format!("--spacing-lg: {};", s.lg));
        lines.push(format!("--spacing-xl: {};", s.xl));
        lines.push(format!("--spacing-xxl: {};", s.xxl));

        let t = &self.typography;
        lines.push(format!("--font-family: {};", t.font_family));
        lines.push(format!("--font-family-mono: {};", t.font_family_mono));
        lines.push(format!("--font-size-xs: {};", t.size_xs));
        lines.push(format!("--font-size-sm: {};", t.size_sm));
        lines.push(format!("--font-size-base: {};", t.size_base));
        lines.push(format!("--font-size-lg: {};", t.size_lg));
        lines.push(format!("--font-size-xl: {};", t.size_xl));
        lines.push(format!("--font-size-xxl: {};", t.size_xxl));
        lines.push(format!("--font-weight-normal: {};", t.weight_normal));
        lines.push(format!("--font-weight-medium: {};", t.weight_medium));
        lines.push(format!("--font-weight-bold: {};", t.weight_bold));
        lines.push(format!("--line-height-tight: {};", t.line_height_tight));
        lines.push(format!("--line-height-normal: {};", t.line_height_normal));
        lines.push(format!("--line-height-relaxed: {};", t.line_height_relaxed));

        lines.push(format!("--border-radius: {};", self.border_radius));
        lines.push(format!("--shadow-sm: {};", self.shadow_sm));
        lines.push(format!("--shadow-md: {};", self.shadow_md));
        lines.push(format!("--shadow-lg: {};", self.shadow_lg));
        lines.push(format!("--transition-fast: {};", self.transition_fast));
        lines.push(format!("--transition-normal: {};", self.transition_normal));
        lines.push(format!("--transition-slow: {};", self.transition_slow));

        // Append extra tokens.
        if !self.tokens.is_empty() {
            lines.push(self.tokens.to_css_variables());
        }

        lines.join("\n")
    }

    /// Full CSS stylesheet with `:root` variables and a
    /// `prefers-color-scheme: dark` media query section.
    pub fn to_stylesheet(&self) -> String {
        let vars = self.to_css_variables();
        let dark = Theme::dark().to_css_variables();
        format!(
            ":root {{\n{vars}\n}}\n\n@media (prefers-color-scheme: dark) {{\n:root {{\n{dark}\n}}\n}}"
        )
    }
}

fn default_spacing() -> Spacing {
    Spacing {
        xs: "0.25rem".into(),
        sm: "0.5rem".into(),
        md: "1rem".into(),
        lg: "1.5rem".into(),
        xl: "2rem".into(),
        xxl: "3rem".into(),
    }
}

fn default_typography() -> Typography {
    Typography {
        font_family: "system-ui, -apple-system, sans-serif".into(),
        font_family_mono: "ui-monospace, monospace".into(),
        size_xs: "0.75rem".into(),
        size_sm: "0.875rem".into(),
        size_base: "1rem".into(),
        size_lg: "1.125rem".into(),
        size_xl: "1.25rem".into(),
        size_xxl: "1.5rem".into(),
        weight_normal: "400".into(),
        weight_medium: "500".into(),
        weight_bold: "700".into(),
        line_height_tight: "1.25".into(),
        line_height_normal: "1.5".into(),
        line_height_relaxed: "1.75".into(),
    }
}

fn default_breakpoints() -> Breakpoints {
    Breakpoints {
        sm: 640.0,
        md: 768.0,
        lg: 1024.0,
        xl: 1280.0,
        xxl: 1536.0,
    }
}

// ── ThemeManager ────────────────────────────────────────────────────────────

pub struct ThemeManager {
    themes: HashMap<String, Theme>,
    current: String,
    pub mode: ThemeMode,
    overrides: DesignTokens,
}

impl ThemeManager {
    pub fn new(default_theme: Theme) -> Self {
        let name = default_theme.name.clone();
        let mode = default_theme.mode.clone();
        let mut themes = HashMap::new();
        themes.insert(name.clone(), default_theme);
        Self {
            themes,
            current: name,
            mode,
            overrides: DesignTokens::new(),
        }
    }

    pub fn add_theme(&mut self, name: &str, theme: Theme) {
        self.themes.insert(name.to_string(), theme);
    }

    pub fn set_theme(&mut self, name: &str) -> bool {
        if self.themes.contains_key(name) {
            self.current = name.to_string();
            true
        } else {
            false
        }
    }

    pub fn set_mode(&mut self, mode: ThemeMode) {
        self.mode = mode;
    }

    pub fn current_theme(&self) -> &Theme {
        self.themes.get(&self.current).expect("current theme must exist")
    }

    pub fn override_token(&mut self, key: &str, value: &str) {
        self.overrides.set(key, value);
    }

    pub fn clear_overrides(&mut self) {
        self.overrides = DesignTokens::new();
    }

    /// Look up a token: overrides take precedence, then the current theme's
    /// token map.
    pub fn resolved_token(&self, key: &str) -> Option<&str> {
        self.overrides
            .get(key)
            .or_else(|| self.current_theme().tokens.get(key))
    }

    pub fn theme_names(&self) -> Vec<&str> {
        self.themes.keys().map(|s| s.as_str()).collect()
    }

    pub fn export_css(&self) -> String {
        self.current_theme().to_css_variables()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn light_theme_has_sensible_defaults() {
        let t = Theme::light();
        assert_eq!(t.name, "light");
        assert!(!t.colors.primary.is_empty());
        assert_eq!(t.colors.background, "#ffffff");
    }

    #[test]
    fn dark_theme_has_dark_colors() {
        let t = Theme::dark();
        assert_eq!(t.colors.background, "#111827");
        assert_eq!(t.mode, ThemeMode::Dark);
    }

    #[test]
    fn to_css_variables_output() {
        let t = Theme::light();
        let css = t.to_css_variables();
        assert!(css.contains("--color-primary:"));
        assert!(css.contains("--spacing-md:"));
        assert!(css.contains("--font-family:"));
    }

    #[test]
    fn theme_switch_changes_current() {
        let mut mgr = ThemeManager::new(Theme::light());
        mgr.add_theme("dark", Theme::dark());
        assert!(mgr.set_theme("dark"));
        assert_eq!(mgr.current_theme().name, "dark");
        assert!(!mgr.set_theme("nonexistent"));
    }

    #[test]
    fn design_tokens_merge() {
        let mut a = DesignTokens::new();
        a.set("x", "1");
        let mut b = DesignTokens::new();
        b.set("x", "2");
        b.set("y", "3");

        a.merge(&b);
        assert_eq!(a.get("x"), Some("2"));
        assert_eq!(a.get("y"), Some("3"));
    }

    #[test]
    fn override_token_takes_precedence() {
        let mut t = Theme::light();
        t.tokens.set("brand", "blue");
        let mut mgr = ThemeManager::new(t);
        mgr.override_token("brand", "red");
        assert_eq!(mgr.resolved_token("brand"), Some("red"));
    }

    #[test]
    fn clear_overrides_reverts() {
        let mut t = Theme::light();
        t.tokens.set("brand", "blue");
        let mut mgr = ThemeManager::new(t);
        mgr.override_token("brand", "red");
        mgr.clear_overrides();
        assert_eq!(mgr.resolved_token("brand"), Some("blue"));
    }

    #[test]
    fn resolved_token_checks_overrides_first() {
        let mut t = Theme::light();
        t.tokens.set("key", "theme-val");
        let mut mgr = ThemeManager::new(t);
        assert_eq!(mgr.resolved_token("key"), Some("theme-val"));

        mgr.override_token("key", "override-val");
        assert_eq!(mgr.resolved_token("key"), Some("override-val"));
    }

    #[test]
    fn to_stylesheet_includes_dark_mode() {
        let t = Theme::light();
        let css = t.to_stylesheet();
        assert!(css.contains(":root {"));
        assert!(css.contains("prefers-color-scheme: dark"));
    }

    #[test]
    fn breakpoints_values() {
        let bp = default_breakpoints();
        assert_eq!(bp.sm, 640.0);
        assert_eq!(bp.xxl, 1536.0);
    }

    #[test]
    fn typography_defaults() {
        let t = default_typography();
        assert!(t.font_family.contains("system-ui"));
        assert_eq!(t.size_base, "1rem");
        assert_eq!(t.weight_bold, "700");
    }

    #[test]
    fn spacing_defaults() {
        let s = default_spacing();
        assert_eq!(s.xs, "0.25rem");
        assert_eq!(s.xxl, "3rem");
    }

    #[test]
    fn theme_names() {
        let mut mgr = ThemeManager::new(Theme::light());
        mgr.add_theme("dark", Theme::dark());
        let mut names = mgr.theme_names();
        names.sort();
        assert_eq!(names, vec!["dark", "light"]);
    }

    #[test]
    fn design_tokens_css_root() {
        let mut dt = DesignTokens::new();
        dt.set("brand", "#ff0000");
        let css = dt.to_css_root();
        assert!(css.contains(":root {"));
        assert!(css.contains("--brand: #ff0000;"));
    }
}
