//! Container queries: container registration with name/type, size-based
//! conditions (min-width/max-width/min-height), style-based conditions,
//! container context stack, query evaluation, and CSS generation.
//!
//! Models the CSS Container Queries specification in pure Rust.

use std::fmt;

// ── Container Type ──────────────────────────────────────────────

/// The type of containment applied to a container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerType {
    /// `container-type: size` — queries on both inline and block axes.
    Size,
    /// `container-type: inline-size` — queries on inline axis only.
    InlineSize,
    /// `container-type: normal` — no size containment, only style queries.
    Normal,
}

impl fmt::Display for ContainerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContainerType::Size => write!(f, "size"),
            ContainerType::InlineSize => write!(f, "inline-size"),
            ContainerType::Normal => write!(f, "normal"),
        }
    }
}

// ── Container Registration ──────────────────────────────────────

/// A registered container element.
#[derive(Debug, Clone)]
pub struct Container {
    /// The container name (used in `@container <name>`).
    pub name: String,
    /// The container type.
    pub container_type: ContainerType,
}

impl Container {
    pub fn new(name: impl Into<String>, container_type: ContainerType) -> Self {
        Self {
            name: name.into(),
            container_type,
        }
    }

    pub fn inline_size(name: impl Into<String>) -> Self {
        Self::new(name, ContainerType::InlineSize)
    }

    pub fn size(name: impl Into<String>) -> Self {
        Self::new(name, ContainerType::Size)
    }

    /// CSS to register this container on a selector.
    pub fn to_css(&self, selector: &str) -> String {
        format!(
            "{selector} {{\n  container-name: {};\n  container-type: {};\n}}\n",
            self.name, self.container_type
        )
    }
}

// ── Size Conditions ─────────────────────────────────────────────

/// A single size-based condition.
#[derive(Debug, Clone, PartialEq)]
pub enum SizeCondition {
    MinWidth(f64),
    MaxWidth(f64),
    MinHeight(f64),
    MaxHeight(f64),
    WidthRange(f64, f64), // min, max
}

impl SizeCondition {
    /// Evaluate against actual container dimensions.
    pub fn matches(&self, width: f64, height: f64) -> bool {
        match self {
            SizeCondition::MinWidth(min) => width >= *min,
            SizeCondition::MaxWidth(max) => width <= *max,
            SizeCondition::MinHeight(min) => height >= *min,
            SizeCondition::MaxHeight(max) => height <= *max,
            SizeCondition::WidthRange(min, max) => width >= *min && width <= *max,
        }
    }

    /// CSS condition fragment.
    pub fn to_css(&self) -> String {
        match self {
            SizeCondition::MinWidth(v) => format!("(min-width: {v}px)"),
            SizeCondition::MaxWidth(v) => format!("(max-width: {v}px)"),
            SizeCondition::MinHeight(v) => format!("(min-height: {v}px)"),
            SizeCondition::MaxHeight(v) => format!("(max-height: {v}px)"),
            SizeCondition::WidthRange(min, max) => {
                format!("(min-width: {min}px) and (max-width: {max}px)")
            }
        }
    }
}

// ── Style Conditions ────────────────────────────────────────────

/// A style-based container query condition.
#[derive(Debug, Clone, PartialEq)]
pub struct StyleCondition {
    /// CSS custom property name (without `--`).
    pub property: String,
    /// Expected value.
    pub value: String,
}

impl StyleCondition {
    pub fn new(property: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            property: property.into(),
            value: value.into(),
        }
    }

    /// CSS `style()` query fragment.
    pub fn to_css(&self) -> String {
        format!("style(--{}: {})", self.property, self.value)
    }

    /// Evaluate against a set of custom property values.
    pub fn matches(&self, properties: &[(String, String)]) -> bool {
        properties
            .iter()
            .any(|(k, v)| *k == self.property && *v == self.value)
    }
}

// ── Query Combinator ────────────────────────────────────────────

/// How multiple conditions are combined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Combinator {
    And,
    Or,
    Not,
}

// ── Container Query ─────────────────────────────────────────────

/// A complete container query with conditions and rules.
#[derive(Debug, Clone)]
pub struct ContainerQuery {
    /// Which container to query (None = nearest ancestor).
    pub container_name: Option<String>,
    pub conditions: Vec<QueryCondition>,
    pub combinator: Combinator,
    /// CSS rules inside the query.
    pub rules: Vec<(String, Vec<(String, String)>)>, // (selector, [(property, value)])
}

/// A single condition in a query.
#[derive(Debug, Clone)]
pub enum QueryCondition {
    Size(SizeCondition),
    Style(StyleCondition),
}

impl QueryCondition {
    pub fn to_css(&self) -> String {
        match self {
            QueryCondition::Size(s) => s.to_css(),
            QueryCondition::Style(s) => s.to_css(),
        }
    }
}

impl ContainerQuery {
    pub fn new(container_name: Option<impl Into<String>>) -> Self {
        Self {
            container_name: container_name.map(|n| n.into()),
            conditions: Vec::new(),
            combinator: Combinator::And,
            rules: Vec::new(),
        }
    }

    pub fn with_combinator(mut self, combinator: Combinator) -> Self {
        self.combinator = combinator;
        self
    }

    pub fn add_size_condition(mut self, cond: SizeCondition) -> Self {
        self.conditions.push(QueryCondition::Size(cond));
        self
    }

    pub fn add_style_condition(mut self, cond: StyleCondition) -> Self {
        self.conditions.push(QueryCondition::Style(cond));
        self
    }

    pub fn add_rule(
        mut self,
        selector: impl Into<String>,
        declarations: Vec<(impl Into<String>, impl Into<String>)>,
    ) -> Self {
        self.rules.push((
            selector.into(),
            declarations
                .into_iter()
                .map(|(p, v)| (p.into(), v.into()))
                .collect(),
        ));
        self
    }

    /// Evaluate all conditions against container dimensions and styles.
    pub fn evaluate(
        &self,
        width: f64,
        height: f64,
        styles: &[(String, String)],
    ) -> bool {
        if self.conditions.is_empty() {
            return true;
        }

        let results: Vec<bool> = self
            .conditions
            .iter()
            .map(|c| match c {
                QueryCondition::Size(s) => s.matches(width, height),
                QueryCondition::Style(s) => s.matches(styles),
            })
            .collect();

        match self.combinator {
            Combinator::And => results.iter().all(|r| *r),
            Combinator::Or => results.iter().any(|r| *r),
            Combinator::Not => !results[0],
        }
    }

    /// Generate CSS for this container query.
    pub fn to_css(&self) -> String {
        let condition_css = match self.combinator {
            Combinator::And => self
                .conditions
                .iter()
                .map(|c| c.to_css())
                .collect::<Vec<_>>()
                .join(" and "),
            Combinator::Or => self
                .conditions
                .iter()
                .map(|c| c.to_css())
                .collect::<Vec<_>>()
                .join(" or "),
            Combinator::Not => {
                if let Some(first) = self.conditions.first() {
                    format!("not {}", first.to_css())
                } else {
                    String::new()
                }
            }
        };

        let container_part = self
            .container_name
            .as_deref()
            .map(|n| format!("{n} "))
            .unwrap_or_default();

        let mut css = format!("@container {container_part}{condition_css} {{\n");
        for (selector, declarations) in &self.rules {
            css.push_str(&format!("  {selector} {{\n"));
            for (prop, val) in declarations {
                css.push_str(&format!("    {prop}: {val};\n"));
            }
            css.push_str("  }\n");
        }
        css.push_str("}\n");
        css
    }
}

// ── Container Context Stack ─────────────────────────────────────

/// Runtime state: the current container dimensions and styles.
#[derive(Debug, Clone)]
pub struct ContainerState {
    pub name: String,
    pub width: f64,
    pub height: f64,
    pub styles: Vec<(String, String)>,
}

impl ContainerState {
    pub fn new(name: impl Into<String>, width: f64, height: f64) -> Self {
        Self {
            name: name.into(),
            width,
            height,
            styles: Vec::new(),
        }
    }

    pub fn with_style(mut self, property: impl Into<String>, value: impl Into<String>) -> Self {
        self.styles.push((property.into(), value.into()));
        self
    }
}

/// A stack of nested container contexts for query resolution.
#[derive(Debug, Clone, Default)]
pub struct ContainerContextStack {
    stack: Vec<ContainerState>,
}

impl ContainerContextStack {
    pub fn new() -> Self {
        Self { stack: Vec::new() }
    }

    pub fn push(&mut self, state: ContainerState) {
        self.stack.push(state);
    }

    pub fn pop(&mut self) -> Option<ContainerState> {
        self.stack.pop()
    }

    /// Find the nearest container matching a name, or the innermost if None.
    pub fn find(&self, name: Option<&str>) -> Option<&ContainerState> {
        match name {
            Some(n) => self.stack.iter().rev().find(|s| s.name == n),
            None => self.stack.last(),
        }
    }

    /// Evaluate a query against the context stack.
    pub fn evaluate(&self, query: &ContainerQuery) -> bool {
        let state = self.find(query.container_name.as_deref());
        match state {
            Some(s) => query.evaluate(s.width, s.height, &s.styles),
            None => false,
        }
    }

    pub fn depth(&self) -> usize {
        self.stack.len()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_container_registration_css() {
        let c = Container::inline_size("sidebar");
        let css = c.to_css(".sidebar");
        assert!(css.contains("container-name: sidebar"));
        assert!(css.contains("container-type: inline-size"));
    }

    #[test]
    fn test_container_type_display() {
        assert_eq!(ContainerType::Size.to_string(), "size");
        assert_eq!(ContainerType::InlineSize.to_string(), "inline-size");
        assert_eq!(ContainerType::Normal.to_string(), "normal");
    }

    #[test]
    fn test_size_condition_min_width() {
        let c = SizeCondition::MinWidth(400.0);
        assert!(c.matches(500.0, 300.0));
        assert!(!c.matches(300.0, 300.0));
        assert_eq!(c.to_css(), "(min-width: 400px)");
    }

    #[test]
    fn test_size_condition_range() {
        let c = SizeCondition::WidthRange(300.0, 600.0);
        assert!(c.matches(400.0, 0.0));
        assert!(!c.matches(700.0, 0.0));
        let css = c.to_css();
        assert!(css.contains("min-width: 300px"));
        assert!(css.contains("max-width: 600px"));
    }

    #[test]
    fn test_style_condition() {
        let c = StyleCondition::new("theme", "dark");
        let styles = vec![("theme".to_owned(), "dark".to_owned())];
        assert!(c.matches(&styles));
        assert_eq!(c.to_css(), "style(--theme: dark)");
    }

    #[test]
    fn test_style_condition_no_match() {
        let c = StyleCondition::new("theme", "dark");
        let styles = vec![("theme".to_owned(), "light".to_owned())];
        assert!(!c.matches(&styles));
    }

    #[test]
    fn test_query_evaluate_and() {
        let q = ContainerQuery::new(None::<String>)
            .add_size_condition(SizeCondition::MinWidth(400.0))
            .add_size_condition(SizeCondition::MaxWidth(800.0));
        assert!(q.evaluate(500.0, 300.0, &[]));
        assert!(!q.evaluate(300.0, 300.0, &[]));
        assert!(!q.evaluate(900.0, 300.0, &[]));
    }

    #[test]
    fn test_query_evaluate_or() {
        let q = ContainerQuery::new(None::<String>)
            .with_combinator(Combinator::Or)
            .add_size_condition(SizeCondition::MinWidth(800.0))
            .add_size_condition(SizeCondition::MaxWidth(200.0));
        assert!(q.evaluate(900.0, 0.0, &[]));
        assert!(q.evaluate(100.0, 0.0, &[]));
        assert!(!q.evaluate(500.0, 0.0, &[]));
    }

    #[test]
    fn test_query_evaluate_not() {
        let q = ContainerQuery::new(None::<String>)
            .with_combinator(Combinator::Not)
            .add_size_condition(SizeCondition::MinWidth(600.0));
        assert!(q.evaluate(400.0, 0.0, &[]));
        assert!(!q.evaluate(700.0, 0.0, &[]));
    }

    #[test]
    fn test_query_css_generation() {
        let q = ContainerQuery::new(Some("sidebar"))
            .add_size_condition(SizeCondition::MinWidth(400.0))
            .add_rule(".card", vec![("font-size", "1.2rem"), ("padding", "2rem")]);
        let css = q.to_css();
        assert!(css.contains("@container sidebar (min-width: 400px)"));
        assert!(css.contains("font-size: 1.2rem"));
        assert!(css.contains("padding: 2rem"));
    }

    #[test]
    fn test_query_unnamed_container() {
        let q = ContainerQuery::new(None::<String>)
            .add_size_condition(SizeCondition::MinWidth(300.0))
            .add_rule(".item", vec![("display", "flex")]);
        let css = q.to_css();
        assert!(css.starts_with("@container (min-width: 300px)"));
    }

    #[test]
    fn test_context_stack_find() {
        let mut stack = ContainerContextStack::new();
        stack.push(ContainerState::new("page", 1200.0, 800.0));
        stack.push(ContainerState::new("sidebar", 300.0, 800.0));

        assert_eq!(stack.find(Some("sidebar")).unwrap().width, 300.0);
        assert_eq!(stack.find(Some("page")).unwrap().width, 1200.0);
        // Nearest = last pushed.
        assert_eq!(stack.find(None).unwrap().name, "sidebar");
    }

    #[test]
    fn test_context_stack_evaluate() {
        let mut stack = ContainerContextStack::new();
        stack.push(ContainerState::new("main", 800.0, 600.0));

        let q = ContainerQuery::new(Some("main"))
            .add_size_condition(SizeCondition::MinWidth(600.0));
        assert!(stack.evaluate(&q));

        let q2 = ContainerQuery::new(Some("main"))
            .add_size_condition(SizeCondition::MinWidth(1000.0));
        assert!(!stack.evaluate(&q2));
    }

    #[test]
    fn test_context_stack_missing_container() {
        let stack = ContainerContextStack::new();
        let q = ContainerQuery::new(Some("missing"))
            .add_size_condition(SizeCondition::MinWidth(100.0));
        assert!(!stack.evaluate(&q));
    }

    #[test]
    fn test_context_stack_push_pop() {
        let mut stack = ContainerContextStack::new();
        stack.push(ContainerState::new("a", 100.0, 100.0));
        stack.push(ContainerState::new("b", 200.0, 200.0));
        assert_eq!(stack.depth(), 2);
        let popped = stack.pop().unwrap();
        assert_eq!(popped.name, "b");
        assert_eq!(stack.depth(), 1);
    }

    #[test]
    fn test_container_state_with_style() {
        let state = ContainerState::new("box", 500.0, 400.0)
            .with_style("theme", "dark")
            .with_style("density", "compact");
        assert_eq!(state.styles.len(), 2);
    }
}
