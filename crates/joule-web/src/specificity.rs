//! CSS specificity calculator: compute, compare, and sort declarations
//! by cascade precedence (origin, importance, specificity, source order).

use crate::selector_engine::{Selector, PseudoClass};

// ── Specificity ─────────────────────────────────────────────────

/// CSS specificity as a 4-component tuple: (inline, id, class/attr/pseudo-class, type/pseudo-element).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Specificity {
    /// Inline style: 1 if inline, 0 otherwise.
    pub inline: u32,
    /// Count of ID selectors.
    pub a: u32,
    /// Count of class, attribute, and pseudo-class selectors.
    pub b: u32,
    /// Count of type and pseudo-element selectors.
    pub c: u32,
}

impl Specificity {
    pub fn new(a: u32, b: u32, c: u32) -> Self {
        Self { inline: 0, a, b, c }
    }

    pub fn inline_style() -> Self {
        Self { inline: 1, a: 0, b: 0, c: 0 }
    }

    pub fn zero() -> Self {
        Self { inline: 0, a: 0, b: 0, c: 0 }
    }

    /// Compare two specificities. Returns Ordering.
    pub fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.inline
            .cmp(&other.inline)
            .then(self.a.cmp(&other.a))
            .then(self.b.cmp(&other.b))
            .then(self.c.cmp(&other.c))
    }

    /// Add two specificities (for compound selectors).
    fn add(self, other: Self) -> Self {
        Self {
            inline: self.inline + other.inline,
            a: self.a + other.a,
            b: self.b + other.b,
            c: self.c + other.c,
        }
    }
}

impl PartialOrd for Specificity {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Specificity {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        Specificity::cmp(self, other)
    }
}

// ── Compute from Selector ───────────────────────────────────────

/// Compute the specificity of a selector AST.
pub fn compute(selector: &Selector) -> Specificity {
    match selector {
        Selector::Universal => Specificity::zero(),
        Selector::Type(_) => Specificity::new(0, 0, 1),
        Selector::Class(_) => Specificity::new(0, 1, 0),
        Selector::Id(_) => Specificity::new(1, 0, 0),
        Selector::Attribute(_) => Specificity::new(0, 1, 0),
        Selector::PseudoClass(pc) => match pc {
            PseudoClass::Not(inner) => compute(inner),
            _ => Specificity::new(0, 1, 0),
        },
        Selector::PseudoElement(_) => Specificity::new(0, 0, 1),
        Selector::Compound(parts) => {
            parts.iter().fold(Specificity::zero(), |acc, s| acc.add(compute(s)))
        }
        Selector::Combinator(left, _, right) => {
            compute(left).add(compute(right))
        }
    }
}

// ── Cascade Origin ──────────────────────────────────────────────

/// CSS cascade origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Origin {
    UserAgent,
    User,
    Author,
}

// ── Declaration ─────────────────────────────────────────────────

/// A CSS declaration with cascade metadata.
#[derive(Debug, Clone)]
pub struct Declaration {
    pub property: String,
    pub value: String,
    pub important: bool,
    pub origin: Origin,
    pub specificity: Specificity,
    pub source_order: usize,
}

impl Declaration {
    pub fn new(
        property: &str,
        value: &str,
        important: bool,
        origin: Origin,
        specificity: Specificity,
        source_order: usize,
    ) -> Self {
        Self {
            property: property.to_string(),
            value: value.to_string(),
            important,
            origin,
            specificity,
            source_order,
        }
    }
}

// ── Cascade Sorting ─────────────────────────────────────────────

/// Sort declarations by CSS cascade precedence:
/// 1. Origin + importance
/// 2. Specificity
/// 3. Source order
pub fn sort_by_cascade(declarations: &mut [Declaration]) {
    declarations.sort_by(|a, b| cascade_order(a, b));
}

fn cascade_order(a: &Declaration, b: &Declaration) -> std::cmp::Ordering {
    // Step 1: Origin + importance
    let a_priority = origin_priority(a.origin, a.important);
    let b_priority = origin_priority(b.origin, b.important);
    a_priority
        .cmp(&b_priority)
        .then_with(|| a.specificity.cmp(&b.specificity))
        .then_with(|| a.source_order.cmp(&b.source_order))
}

/// Higher = wins. CSS cascade origin priority:
/// Normal: user-agent < user < author
/// Important: author !important < user !important < user-agent !important
fn origin_priority(origin: Origin, important: bool) -> u32 {
    if important {
        match origin {
            Origin::Author => 3,
            Origin::User => 4,
            Origin::UserAgent => 5,
        }
    } else {
        match origin {
            Origin::UserAgent => 0,
            Origin::User => 1,
            Origin::Author => 2,
        }
    }
}

/// Given a list of declarations for the same property, return the winning one.
pub fn cascade_winner(declarations: &[Declaration]) -> Option<&Declaration> {
    if declarations.is_empty() {
        return None;
    }
    let mut sorted: Vec<&Declaration> = declarations.iter().collect();
    sorted.sort_by(|a, b| cascade_order(a, b));
    sorted.last().copied()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selector_engine::{self, AttrSelector, NthExpr};

    #[test]
    fn specificity_type_selector() {
        let sel = selector_engine::parse("div").unwrap();
        let spec = compute(&sel);
        assert_eq!(spec, Specificity::new(0, 0, 1));
    }

    #[test]
    fn specificity_class_selector() {
        let sel = selector_engine::parse(".active").unwrap();
        let spec = compute(&sel);
        assert_eq!(spec, Specificity::new(0, 1, 0));
    }

    #[test]
    fn specificity_id_selector() {
        let sel = selector_engine::parse("#main").unwrap();
        let spec = compute(&sel);
        assert_eq!(spec, Specificity::new(1, 0, 0));
    }

    #[test]
    fn specificity_compound() {
        let sel = selector_engine::parse("div.active#hero").unwrap();
        let spec = compute(&sel);
        // 1 id + 1 class + 1 type = (1, 1, 1)
        assert_eq!(spec, Specificity::new(1, 1, 1));
    }

    #[test]
    fn specificity_descendant() {
        let sel = selector_engine::parse("div p").unwrap();
        let spec = compute(&sel);
        assert_eq!(spec, Specificity::new(0, 0, 2));
    }

    #[test]
    fn specificity_pseudo_class() {
        let sel = selector_engine::parse(":hover").unwrap();
        let spec = compute(&sel);
        assert_eq!(spec, Specificity::new(0, 1, 0));
    }

    #[test]
    fn specificity_pseudo_element() {
        let sel = selector_engine::parse("::before").unwrap();
        let spec = compute(&sel);
        assert_eq!(spec, Specificity::new(0, 0, 1));
    }

    #[test]
    fn specificity_inline_style() {
        let spec = Specificity::inline_style();
        let id_spec = Specificity::new(1, 0, 0);
        assert!(spec > id_spec);
    }

    #[test]
    fn compare_specificities() {
        let a = Specificity::new(0, 1, 0); // .class
        let b = Specificity::new(0, 0, 2); // div span
        assert!(a > b);
    }

    #[test]
    fn cascade_sort_by_specificity() {
        let mut decls = vec![
            Declaration::new("color", "red", false, Origin::Author, Specificity::new(0, 0, 1), 0),
            Declaration::new("color", "blue", false, Origin::Author, Specificity::new(0, 1, 0), 1),
        ];
        sort_by_cascade(&mut decls);
        assert_eq!(decls.last().unwrap().value, "blue");
    }

    #[test]
    fn cascade_important_overrides() {
        let mut decls = vec![
            Declaration::new("color", "blue", false, Origin::Author, Specificity::new(1, 0, 0), 0),
            Declaration::new("color", "red", true, Origin::Author, Specificity::new(0, 0, 1), 1),
        ];
        sort_by_cascade(&mut decls);
        // !important author beats non-important author even with lower specificity
        assert_eq!(decls.last().unwrap().value, "red");
    }

    #[test]
    fn cascade_source_order_tiebreak() {
        let decls = vec![
            Declaration::new("color", "red", false, Origin::Author, Specificity::new(0, 1, 0), 0),
            Declaration::new("color", "blue", false, Origin::Author, Specificity::new(0, 1, 0), 1),
        ];
        let winner = cascade_winner(&decls).unwrap();
        assert_eq!(winner.value, "blue"); // later source order wins
    }

    #[test]
    fn cascade_user_agent_lowest() {
        let decls = vec![
            Declaration::new("display", "block", false, Origin::UserAgent, Specificity::new(0, 0, 1), 0),
            Declaration::new("display", "flex", false, Origin::Author, Specificity::new(0, 0, 1), 1),
        ];
        let winner = cascade_winner(&decls).unwrap();
        assert_eq!(winner.value, "flex");
    }

    #[test]
    fn specificity_not_pseudo_inherits_inner() {
        // :not(.foo) has specificity of .foo = (0,1,0)
        let sel = selector_engine::parse(":not(.foo)").unwrap();
        let spec = compute(&sel);
        assert_eq!(spec, Specificity::new(0, 1, 0));
    }
}
