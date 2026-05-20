//! CSS selector parser and matcher: type, class, id, attribute, pseudo-class,
//! pseudo-element selectors with descendant/child/adjacent/sibling combinators.

use std::collections::HashMap;

// ── Selector AST ────────────────────────────────────────────────

/// A parsed CSS selector.
#[derive(Debug, Clone, PartialEq)]
pub enum Selector {
    /// Universal selector `*`
    Universal,
    /// Type selector `div`
    Type(String),
    /// Class selector `.foo`
    Class(String),
    /// ID selector `#bar`
    Id(String),
    /// Attribute selector `[href]` or `[href="value"]`
    Attribute(AttrSelector),
    /// Pseudo-class `:hover`, `:first-child`, `:nth-child(n)`
    PseudoClass(PseudoClass),
    /// Pseudo-element `::before`, `::after`
    PseudoElement(String),
    /// Compound selector — all must match the same element
    Compound(Vec<Selector>),
    /// Combinator — relates two selectors
    Combinator(Box<Selector>, CombinatorKind, Box<Selector>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttrSelector {
    pub name: String,
    pub op: Option<AttrOp>,
    pub value: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttrOp {
    Equals,       // =
    Contains,     // *=
    StartsWith,   // ^=
    EndsWith,     // $=
    DashMatch,    // |=
    Includes,     // ~=
}

#[derive(Debug, Clone, PartialEq)]
pub enum PseudoClass {
    Hover,
    Focus,
    Active,
    Visited,
    FirstChild,
    LastChild,
    NthChild(NthExpr),
    Not(Box<Selector>),
}

/// `an+b` expression for `:nth-child`.
#[derive(Debug, Clone, PartialEq)]
pub struct NthExpr {
    pub a: i32,
    pub b: i32,
}

impl NthExpr {
    /// Check if a 1-based index matches `an+b`.
    pub fn matches(&self, index: i32) -> bool {
        if self.a == 0 {
            return index == self.b;
        }
        let diff = index - self.b;
        if self.a > 0 {
            diff >= 0 && diff % self.a == 0
        } else {
            diff <= 0 && diff % self.a == 0
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CombinatorKind {
    Descendant,   // space
    Child,        // >
    Adjacent,     // +
    Sibling,      // ~
}

// ── Simple DOM Node ─────────────────────────────────────────────

/// Minimal DOM node for selector matching.
#[derive(Debug, Clone)]
pub struct DomNode {
    pub tag: String,
    pub id: Option<String>,
    pub classes: Vec<String>,
    pub attributes: HashMap<String, String>,
    pub pseudo_states: Vec<String>,
    pub parent: Option<Box<DomNode>>,
    /// 1-based index among siblings.
    pub child_index: usize,
    /// Total number of siblings (including self).
    pub sibling_count: usize,
    /// Previous siblings (for adjacent/general sibling combinators).
    pub prev_siblings: Vec<Box<DomNode>>,
}

impl DomNode {
    pub fn new(tag: &str) -> Self {
        Self {
            tag: tag.to_string(),
            id: None,
            classes: vec![],
            attributes: HashMap::new(),
            pseudo_states: vec![],
            parent: None,
            child_index: 1,
            sibling_count: 1,
            prev_siblings: vec![],
        }
    }

    pub fn with_id(mut self, id: &str) -> Self {
        self.id = Some(id.to_string());
        self
    }

    pub fn with_class(mut self, class: &str) -> Self {
        self.classes.push(class.to_string());
        self
    }

    pub fn with_attr(mut self, name: &str, value: &str) -> Self {
        self.attributes.insert(name.to_string(), value.to_string());
        self
    }

    pub fn with_pseudo(mut self, state: &str) -> Self {
        self.pseudo_states.push(state.to_string());
        self
    }

    pub fn with_parent(mut self, parent: DomNode) -> Self {
        self.parent = Some(Box::new(parent));
        self
    }

    pub fn with_child_index(mut self, index: usize, total: usize) -> Self {
        self.child_index = index;
        self.sibling_count = total;
        self
    }

    pub fn with_prev_sibling(mut self, sibling: DomNode) -> Self {
        self.prev_siblings.push(Box::new(sibling));
        self
    }
}

// ── Parser ──────────────────────────────────────────────────────

/// Parse a CSS selector string into an AST.
pub fn parse(input: &str) -> Result<Selector, String> {
    let input = input.trim();
    if input.is_empty() {
        return Err("Empty selector".to_string());
    }

    parse_combinator_chain(input)
}

fn parse_combinator_chain(input: &str) -> Result<Selector, String> {
    let tokens = tokenize_combinators(input)?;

    if tokens.len() == 1 {
        return parse_compound(&tokens[0].0);
    }

    // Build left-associative combinator chain
    let mut iter = tokens.into_iter();
    let first = iter.next().unwrap();
    let mut left = parse_compound(&first.0)?;

    while let Some((sel_str, comb)) = iter.next() {
        let combinator = comb.unwrap_or(CombinatorKind::Descendant);
        let right = parse_compound(&sel_str)?;
        left = Selector::Combinator(Box::new(left), combinator, Box::new(right));
    }

    Ok(left)
}

fn tokenize_combinators(
    input: &str,
) -> Result<Vec<(String, Option<CombinatorKind>)>, String> {
    let mut tokens: Vec<(String, Option<CombinatorKind>)> = Vec::new();
    let mut current = String::new();
    let mut chars = input.chars().peekable();
    let mut last_combinator: Option<CombinatorKind> = None;

    while let Some(ch) = chars.next() {
        match ch {
            '>' | '+' | '~' => {
                if !current.trim().is_empty() {
                    tokens.push((current.trim().to_string(), last_combinator.take()));
                    current.clear();
                }
                last_combinator = Some(match ch {
                    '>' => CombinatorKind::Child,
                    '+' => CombinatorKind::Adjacent,
                    '~' => CombinatorKind::Sibling,
                    _ => unreachable!(),
                });
            }
            ' ' => {
                if !current.trim().is_empty() && last_combinator.is_none() {
                    // Could be descendant combinator or just spacing around > + ~
                    // Peek ahead to see if there's an explicit combinator
                    let rest: String = chars.clone().collect();
                    let rest_trimmed = rest.trim_start();
                    if rest_trimmed.starts_with('>') || rest_trimmed.starts_with('+') || rest_trimmed.starts_with('~') {
                        // Spacing before explicit combinator — not a descendant
                        tokens.push((current.trim().to_string(), last_combinator.take()));
                        current.clear();
                    } else if !rest_trimmed.is_empty() {
                        tokens.push((current.trim().to_string(), last_combinator.take()));
                        current.clear();
                        last_combinator = Some(CombinatorKind::Descendant);
                    } else {
                        current.push(ch);
                    }
                } else {
                    // Skip whitespace
                }
            }
            '[' => {
                current.push(ch);
                // Read until matching ]
                for c in chars.by_ref() {
                    current.push(c);
                    if c == ']' {
                        break;
                    }
                }
            }
            '(' => {
                current.push(ch);
                let mut depth = 1u32;
                for c in chars.by_ref() {
                    current.push(c);
                    match c {
                        '(' => depth += 1,
                        ')' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => current.push(ch),
        }
    }

    if !current.trim().is_empty() {
        tokens.push((current.trim().to_string(), last_combinator.take()));
    }

    Ok(tokens)
}

fn parse_compound(input: &str) -> Result<Selector, String> {
    let input = input.trim();
    if input == "*" {
        return Ok(Selector::Universal);
    }

    let mut parts: Vec<Selector> = Vec::new();
    let mut chars = input.chars().peekable();

    while chars.peek().is_some() {
        match chars.peek().copied() {
            Some('#') => {
                chars.next();
                let name = collect_ident(&mut chars);
                parts.push(Selector::Id(name));
            }
            Some('.') => {
                chars.next();
                let name = collect_ident(&mut chars);
                parts.push(Selector::Class(name));
            }
            Some('[') => {
                chars.next();
                let attr = parse_attr_selector(&mut chars)?;
                parts.push(Selector::Attribute(attr));
            }
            Some(':') => {
                chars.next();
                if chars.peek() == Some(&':') {
                    chars.next();
                    let name = collect_ident(&mut chars);
                    parts.push(Selector::PseudoElement(name));
                } else {
                    let pseudo = parse_pseudo_class(&mut chars)?;
                    parts.push(Selector::PseudoClass(pseudo));
                }
            }
            Some(c) if c.is_alphanumeric() || c == '-' || c == '_' || c == '*' => {
                let name = collect_ident(&mut chars);
                if name == "*" {
                    parts.push(Selector::Universal);
                } else {
                    parts.push(Selector::Type(name));
                }
            }
            Some(c) => return Err(format!("Unexpected character: {c}")),
            None => break,
        }
    }

    match parts.len() {
        0 => Err("Empty compound selector".to_string()),
        1 => Ok(parts.into_iter().next().unwrap()),
        _ => Ok(Selector::Compound(parts)),
    }
}

fn collect_ident(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    let mut s = String::new();
    while let Some(&ch) = chars.peek() {
        if ch.is_alphanumeric() || ch == '-' || ch == '_' {
            s.push(ch);
            chars.next();
        } else {
            break;
        }
    }
    s
}

fn parse_attr_selector(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
) -> Result<AttrSelector, String> {
    let name = collect_ident(chars);

    // Check for operator
    let op = match chars.peek() {
        Some(']') => {
            chars.next();
            return Ok(AttrSelector { name, op: None, value: None });
        }
        Some('=') => {
            chars.next();
            Some(AttrOp::Equals)
        }
        Some('*') => {
            chars.next();
            if chars.next() != Some('=') {
                return Err("Expected = after *".to_string());
            }
            Some(AttrOp::Contains)
        }
        Some('^') => {
            chars.next();
            if chars.next() != Some('=') {
                return Err("Expected = after ^".to_string());
            }
            Some(AttrOp::StartsWith)
        }
        Some('$') => {
            chars.next();
            if chars.next() != Some('=') {
                return Err("Expected = after $".to_string());
            }
            Some(AttrOp::EndsWith)
        }
        Some('|') => {
            chars.next();
            if chars.next() != Some('=') {
                return Err("Expected = after |".to_string());
            }
            Some(AttrOp::DashMatch)
        }
        Some('~') => {
            chars.next();
            if chars.next() != Some('=') {
                return Err("Expected = after ~".to_string());
            }
            Some(AttrOp::Includes)
        }
        _ => None,
    };

    // Read value (possibly quoted)
    let value = if op.is_some() {
        let mut val = String::new();
        let quoted = matches!(chars.peek(), Some('"') | Some('\''));
        if quoted {
            let quote = chars.next().unwrap();
            for ch in chars.by_ref() {
                if ch == quote {
                    break;
                }
                val.push(ch);
            }
        } else {
            while let Some(&ch) = chars.peek() {
                if ch == ']' {
                    break;
                }
                val.push(ch);
                chars.next();
            }
        }
        Some(val)
    } else {
        None
    };

    // Consume closing ]
    if chars.peek() == Some(&']') {
        chars.next();
    }

    Ok(AttrSelector { name, op, value })
}

fn parse_pseudo_class(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
) -> Result<PseudoClass, String> {
    let name = collect_ident(chars);
    match name.as_str() {
        "hover" => Ok(PseudoClass::Hover),
        "focus" => Ok(PseudoClass::Focus),
        "active" => Ok(PseudoClass::Active),
        "visited" => Ok(PseudoClass::Visited),
        "first-child" => Ok(PseudoClass::FirstChild),
        "last-child" => Ok(PseudoClass::LastChild),
        "nth-child" => {
            if chars.next() != Some('(') {
                return Err("Expected ( after nth-child".to_string());
            }
            let mut expr_str = String::new();
            let mut depth = 1u32;
            for ch in chars.by_ref() {
                match ch {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
                expr_str.push(ch);
            }
            let nth = parse_nth_expr(expr_str.trim())?;
            Ok(PseudoClass::NthChild(nth))
        }
        "not" => {
            if chars.next() != Some('(') {
                return Err("Expected ( after :not".to_string());
            }
            let mut inner = String::new();
            let mut depth = 1u32;
            for ch in chars.by_ref() {
                match ch {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
                inner.push(ch);
            }
            let sel = parse_compound(inner.trim())?;
            Ok(PseudoClass::Not(Box::new(sel)))
        }
        _ => Err(format!("Unknown pseudo-class: {name}")),
    }
}

fn parse_nth_expr(s: &str) -> Result<NthExpr, String> {
    let s = s.trim().to_lowercase();
    if s == "odd" {
        return Ok(NthExpr { a: 2, b: 1 });
    }
    if s == "even" {
        return Ok(NthExpr { a: 2, b: 0 });
    }

    if let Some(n_pos) = s.find('n') {
        let a_str = &s[..n_pos].trim().replace('+', "");
        let a = if a_str.is_empty() || a_str == &"" {
            1
        } else if a_str == &"-" {
            -1
        } else {
            a_str.parse::<i32>().map_err(|e| e.to_string())?
        };

        let after = s[n_pos + 1..].trim().to_string();
        let b = if after.is_empty() {
            0
        } else {
            after.replace('+', "").replace(' ', "").parse::<i32>().map_err(|e| e.to_string())?
        };

        Ok(NthExpr { a, b })
    } else {
        let b = s.parse::<i32>().map_err(|e| e.to_string())?;
        Ok(NthExpr { a: 0, b })
    }
}

// ── Matcher ─────────────────────────────────────────────────────

/// Check if a selector matches a DOM node.
pub fn matches(selector: &Selector, node: &DomNode) -> bool {
    match selector {
        Selector::Universal => true,
        Selector::Type(tag) => node.tag == *tag,
        Selector::Class(cls) => node.classes.contains(cls),
        Selector::Id(id) => node.id.as_deref() == Some(id.as_str()),
        Selector::Attribute(attr) => match_attr(attr, node),
        Selector::PseudoClass(pc) => match_pseudo_class(pc, node),
        Selector::PseudoElement(_) => true, // Pseudo-elements always "match" structurally
        Selector::Compound(parts) => parts.iter().all(|p| matches(p, node)),
        Selector::Combinator(left, kind, right) => {
            if !matches(right, node) {
                return false;
            }
            match kind {
                CombinatorKind::Child => {
                    node.parent.as_ref().is_some_and(|p| matches(left, p))
                }
                CombinatorKind::Descendant => {
                    let mut ancestor = node.parent.as_ref();
                    while let Some(anc) = ancestor {
                        if matches(left, anc) {
                            return true;
                        }
                        ancestor = anc.parent.as_ref();
                    }
                    false
                }
                CombinatorKind::Adjacent => {
                    node.prev_siblings.last().is_some_and(|s| matches(left, s))
                }
                CombinatorKind::Sibling => {
                    node.prev_siblings.iter().any(|s| matches(left, s))
                }
            }
        }
    }
}

fn match_attr(attr: &AttrSelector, node: &DomNode) -> bool {
    let node_val = match node.attributes.get(&attr.name) {
        Some(v) => v,
        None => return false,
    };

    match (&attr.op, &attr.value) {
        (None, _) => true, // Just [attr] — existence check
        (Some(op), Some(val)) => match op {
            AttrOp::Equals => node_val == val,
            AttrOp::Contains => node_val.contains(val.as_str()),
            AttrOp::StartsWith => node_val.starts_with(val.as_str()),
            AttrOp::EndsWith => node_val.ends_with(val.as_str()),
            AttrOp::DashMatch => node_val == val || node_val.starts_with(&format!("{val}-")),
            AttrOp::Includes => node_val.split_whitespace().any(|w| w == val),
        },
        _ => false,
    }
}

fn match_pseudo_class(pc: &PseudoClass, node: &DomNode) -> bool {
    match pc {
        PseudoClass::Hover => node.pseudo_states.iter().any(|s| s == "hover"),
        PseudoClass::Focus => node.pseudo_states.iter().any(|s| s == "focus"),
        PseudoClass::Active => node.pseudo_states.iter().any(|s| s == "active"),
        PseudoClass::Visited => node.pseudo_states.iter().any(|s| s == "visited"),
        PseudoClass::FirstChild => node.child_index == 1,
        PseudoClass::LastChild => node.child_index == node.sibling_count,
        PseudoClass::NthChild(expr) => expr.matches(node.child_index as i32),
        PseudoClass::Not(sel) => !matches(sel, node),
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_type_selector() {
        let sel = parse("div").unwrap();
        assert_eq!(sel, Selector::Type("div".to_string()));
    }

    #[test]
    fn parse_class_selector() {
        let sel = parse(".active").unwrap();
        assert_eq!(sel, Selector::Class("active".to_string()));
    }

    #[test]
    fn parse_id_selector() {
        let sel = parse("#main").unwrap();
        assert_eq!(sel, Selector::Id("main".to_string()));
    }

    #[test]
    fn parse_compound_selector() {
        let sel = parse("div.active#hero").unwrap();
        match sel {
            Selector::Compound(parts) => {
                assert_eq!(parts.len(), 3);
                assert_eq!(parts[0], Selector::Type("div".to_string()));
                assert_eq!(parts[1], Selector::Class("active".to_string()));
                assert_eq!(parts[2], Selector::Id("hero".to_string()));
            }
            _ => panic!("Expected compound selector"),
        }
    }

    #[test]
    fn parse_descendant_combinator() {
        let sel = parse("div p").unwrap();
        match sel {
            Selector::Combinator(_, CombinatorKind::Descendant, _) => {}
            _ => panic!("Expected descendant combinator"),
        }
    }

    #[test]
    fn parse_child_combinator() {
        let sel = parse("div > p").unwrap();
        match sel {
            Selector::Combinator(_, CombinatorKind::Child, _) => {}
            _ => panic!("Expected child combinator"),
        }
    }

    #[test]
    fn parse_attribute_selector() {
        let sel = parse("[href]").unwrap();
        match sel {
            Selector::Attribute(attr) => {
                assert_eq!(attr.name, "href");
                assert!(attr.op.is_none());
            }
            _ => panic!("Expected attribute selector"),
        }
    }

    #[test]
    fn parse_attribute_equals() {
        let sel = parse("[type=\"text\"]").unwrap();
        match sel {
            Selector::Attribute(attr) => {
                assert_eq!(attr.name, "type");
                assert_eq!(attr.op, Some(AttrOp::Equals));
                assert_eq!(attr.value, Some("text".to_string()));
            }
            _ => panic!("Expected attribute selector"),
        }
    }

    #[test]
    fn parse_pseudo_class_hover() {
        let sel = parse(":hover").unwrap();
        assert_eq!(sel, Selector::PseudoClass(PseudoClass::Hover));
    }

    #[test]
    fn parse_pseudo_element() {
        let sel = parse("::before").unwrap();
        assert_eq!(sel, Selector::PseudoElement("before".to_string()));
    }

    #[test]
    fn parse_nth_child() {
        let sel = parse(":nth-child(2n+1)").unwrap();
        match sel {
            Selector::PseudoClass(PseudoClass::NthChild(expr)) => {
                assert_eq!(expr.a, 2);
                assert_eq!(expr.b, 1);
                assert!(expr.matches(1));
                assert!(!expr.matches(2));
                assert!(expr.matches(3));
            }
            _ => panic!("Expected nth-child"),
        }
    }

    #[test]
    fn match_type_selector() {
        let sel = parse("div").unwrap();
        let node = DomNode::new("div");
        assert!(matches(&sel, &node));
        let other = DomNode::new("span");
        assert!(!matches(&sel, &other));
    }

    #[test]
    fn match_class_selector() {
        let sel = parse(".active").unwrap();
        let node = DomNode::new("div").with_class("active");
        assert!(matches(&sel, &node));
    }

    #[test]
    fn match_child_combinator() {
        let sel = parse("ul > li").unwrap();
        let parent = DomNode::new("ul");
        let child = DomNode::new("li").with_parent(parent);
        assert!(matches(&sel, &child));

        let wrong_parent = DomNode::new("div");
        let wrong = DomNode::new("li").with_parent(wrong_parent);
        assert!(!matches(&sel, &wrong));
    }

    #[test]
    fn match_descendant_combinator() {
        let sel = parse("div p").unwrap();
        let grandparent = DomNode::new("div");
        let parent = DomNode::new("section").with_parent(grandparent);
        let node = DomNode::new("p").with_parent(parent);
        assert!(matches(&sel, &node));
    }

    #[test]
    fn match_adjacent_sibling() {
        let sel = parse("h1 + p").unwrap();
        let prev = DomNode::new("h1");
        let node = DomNode::new("p").with_prev_sibling(prev);
        assert!(matches(&sel, &node));
    }

    #[test]
    fn match_pseudo_class_first_child() {
        let sel = parse(":first-child").unwrap();
        let first = DomNode::new("li").with_child_index(1, 3);
        let second = DomNode::new("li").with_child_index(2, 3);
        assert!(matches(&sel, &first));
        assert!(!matches(&sel, &second));
    }

    #[test]
    fn match_not_pseudo_class() {
        let sel = parse(":not(.hidden)").unwrap();
        let visible = DomNode::new("div");
        let hidden = DomNode::new("div").with_class("hidden");
        assert!(matches(&sel, &visible));
        assert!(!matches(&sel, &hidden));
    }
}
