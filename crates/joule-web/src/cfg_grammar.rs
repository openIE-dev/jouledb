//! Context-free grammar tools — grammar definition (productions), FIRST/FOLLOW
//! sets, LL(1) parse table, LR(0) item sets, nullable detection, left recursion
//! elimination, grammar validation.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt;

// ── Symbols ─────────────────────────────────────────────────────────────────

/// A grammar symbol.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Symbol {
    /// A non-terminal (identified by name).
    NonTerminal(String),
    /// A terminal / token.
    Terminal(String),
    /// End-of-input marker ($).
    Eof,
    /// Epsilon (empty string).
    Epsilon,
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonTerminal(s) => write!(f, "{s}"),
            Self::Terminal(s) => write!(f, "'{s}'"),
            Self::Eof => write!(f, "$"),
            Self::Epsilon => write!(f, "epsilon"),
        }
    }
}

impl Symbol {
    pub fn nt(name: &str) -> Self {
        Self::NonTerminal(name.into())
    }

    pub fn t(name: &str) -> Self {
        Self::Terminal(name.into())
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Terminal(_) | Self::Eof)
    }

    pub fn is_non_terminal(&self) -> bool {
        matches!(self, Self::NonTerminal(_))
    }
}

// ── Production ──────────────────────────────────────────────────────────────

/// A single production rule: head -> body.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Production {
    pub head: String,
    pub body: Vec<Symbol>,
}

impl Production {
    pub fn new(head: &str, body: Vec<Symbol>) -> Self {
        Self {
            head: head.into(),
            body,
        }
    }

    /// True if the body is epsilon (or empty).
    pub fn is_epsilon(&self) -> bool {
        self.body.is_empty() || (self.body.len() == 1 && self.body[0] == Symbol::Epsilon)
    }
}

impl fmt::Display for Production {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ->", self.head)?;
        if self.body.is_empty() {
            write!(f, " epsilon")?;
        } else {
            for s in &self.body {
                write!(f, " {s}")?;
            }
        }
        Ok(())
    }
}

// ── Grammar ─────────────────────────────────────────────────────────────────

/// Errors for grammar operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GrammarError {
    /// A production references a non-terminal that has no productions.
    UndefinedNonTerminal(String),
    /// The grammar has unreachable non-terminals.
    Unreachable(Vec<String>),
    /// Left recursion detected.
    LeftRecursion(String),
    /// LL(1) conflict.
    Ll1Conflict(String, String),
    /// No start symbol.
    NoStart,
    /// Other.
    Other(String),
}

impl fmt::Display for GrammarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UndefinedNonTerminal(s) => write!(f, "undefined non-terminal: {s}"),
            Self::Unreachable(nts) => write!(f, "unreachable: {}", nts.join(", ")),
            Self::LeftRecursion(nt) => write!(f, "left recursion on: {nt}"),
            Self::Ll1Conflict(nt, t) => write!(f, "LL(1) conflict: {nt} on '{t}'"),
            Self::NoStart => write!(f, "no start symbol"),
            Self::Other(msg) => write!(f, "{msg}"),
        }
    }
}

/// A context-free grammar.
#[derive(Debug, Clone)]
pub struct Grammar {
    pub productions: Vec<Production>,
    pub start: String,
}

impl Grammar {
    /// Create a new grammar with the given start symbol.
    pub fn new(start: &str) -> Self {
        Self {
            productions: Vec::new(),
            start: start.into(),
        }
    }

    /// Add a production rule.
    pub fn add_production(&mut self, head: &str, body: Vec<Symbol>) {
        self.productions.push(Production::new(head, body));
    }

    /// Add an epsilon production.
    pub fn add_epsilon(&mut self, head: &str) {
        self.productions.push(Production::new(head, vec![]));
    }

    /// Get all non-terminals that have productions.
    pub fn non_terminals(&self) -> BTreeSet<String> {
        self.productions.iter().map(|p| p.head.clone()).collect()
    }

    /// Get all terminal symbols.
    pub fn terminals(&self) -> BTreeSet<String> {
        let mut terms = BTreeSet::new();
        for p in &self.productions {
            for s in &p.body {
                if let Symbol::Terminal(t) = s {
                    terms.insert(t.clone());
                }
            }
        }
        terms
    }

    /// Get productions for a given non-terminal.
    pub fn productions_for(&self, nt: &str) -> Vec<&Production> {
        self.productions.iter().filter(|p| p.head == nt).collect()
    }

    // ── Nullable ────────────────────────────────────────────────────────

    /// Compute the set of nullable non-terminals.
    pub fn nullable_set(&self) -> BTreeSet<String> {
        let mut nullable = BTreeSet::new();
        let mut changed = true;
        while changed {
            changed = false;
            for p in &self.productions {
                if nullable.contains(&p.head) {
                    continue;
                }
                let body_nullable = p.body.is_empty()
                    || p.body.iter().all(|s| match s {
                        Symbol::Epsilon => true,
                        Symbol::NonTerminal(nt) => nullable.contains(nt),
                        _ => false,
                    });
                if body_nullable {
                    nullable.insert(p.head.clone());
                    changed = true;
                }
            }
        }
        nullable
    }

    /// True if the non-terminal can derive epsilon.
    pub fn is_nullable(&self, nt: &str) -> bool {
        self.nullable_set().contains(nt)
    }

    // ── FIRST sets ──────────────────────────────────────────────────────

    /// Compute FIRST sets for all non-terminals.
    pub fn first_sets(&self) -> BTreeMap<String, BTreeSet<Symbol>> {
        let nullable = self.nullable_set();
        let nts = self.non_terminals();
        let mut first: BTreeMap<String, BTreeSet<Symbol>> = BTreeMap::new();
        for nt in &nts {
            first.insert(nt.clone(), BTreeSet::new());
        }

        let mut changed = true;
        while changed {
            changed = false;
            for p in &self.productions {
                let new_firsts = self.first_of_sequence(&p.body, &first, &nullable);
                let entry = first.get_mut(&p.head).unwrap();
                for s in new_firsts {
                    if entry.insert(s) {
                        changed = true;
                    }
                }
            }
        }
        first
    }

    /// Compute FIRST of a sequence of symbols.
    fn first_of_sequence(
        &self,
        seq: &[Symbol],
        first: &BTreeMap<String, BTreeSet<Symbol>>,
        nullable: &BTreeSet<String>,
    ) -> BTreeSet<Symbol> {
        let mut result = BTreeSet::new();
        if seq.is_empty() {
            result.insert(Symbol::Epsilon);
            return result;
        }
        for sym in seq {
            match sym {
                Symbol::Terminal(_) | Symbol::Eof => {
                    result.insert(sym.clone());
                    return result;
                }
                Symbol::Epsilon => {
                    continue;
                }
                Symbol::NonTerminal(nt) => {
                    if let Some(f) = first.get(nt) {
                        for s in f {
                            if *s != Symbol::Epsilon {
                                result.insert(s.clone());
                            }
                        }
                    }
                    if !nullable.contains(nt) {
                        return result;
                    }
                }
            }
        }
        result.insert(Symbol::Epsilon);
        result
    }

    // ── FOLLOW sets ─────────────────────────────────────────────────────

    /// Compute FOLLOW sets for all non-terminals.
    pub fn follow_sets(&self) -> BTreeMap<String, BTreeSet<Symbol>> {
        let first = self.first_sets();
        let nullable = self.nullable_set();
        let nts = self.non_terminals();
        let mut follow: BTreeMap<String, BTreeSet<Symbol>> = BTreeMap::new();
        for nt in &nts {
            follow.insert(nt.clone(), BTreeSet::new());
        }
        // $ in FOLLOW(start)
        follow.get_mut(&self.start).unwrap().insert(Symbol::Eof);

        let mut changed = true;
        while changed {
            changed = false;
            for p in &self.productions {
                for (i, sym) in p.body.iter().enumerate() {
                    if let Symbol::NonTerminal(b) = sym {
                        let rest = &p.body[i + 1..];
                        let first_rest = self.first_of_sequence(rest, &first, &nullable);
                        let follow_b = follow.get_mut(b).unwrap();
                        for s in &first_rest {
                            if *s != Symbol::Epsilon && follow_b.insert(s.clone()) {
                                changed = true;
                            }
                        }
                        let rest_nullable = rest.is_empty()
                            || rest.iter().all(|s| match s {
                                Symbol::Epsilon => true,
                                Symbol::NonTerminal(nt) => nullable.contains(nt),
                                _ => false,
                            });
                        if rest_nullable {
                            let head_follow: BTreeSet<Symbol> =
                                follow.get(&p.head).cloned().unwrap_or_default();
                            let follow_b = follow.get_mut(b).unwrap();
                            for s in &head_follow {
                                if follow_b.insert(s.clone()) {
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        follow
    }

    // ── LL(1) parse table ───────────────────────────────────────────────

    /// Build an LL(1) parse table. Returns a map: (non-terminal, terminal) -> production index.
    pub fn ll1_table(&self) -> Result<HashMap<(String, Symbol), usize>, GrammarError> {
        let first = self.first_sets();
        let follow = self.follow_sets();
        let nullable = self.nullable_set();
        let mut table: HashMap<(String, Symbol), usize> = HashMap::new();

        for (idx, p) in self.productions.iter().enumerate() {
            let body_first = self.first_of_sequence(&p.body, &first, &nullable);
            for s in &body_first {
                if *s == Symbol::Epsilon {
                    continue;
                }
                let key = (p.head.clone(), s.clone());
                if table.contains_key(&key) {
                    return Err(GrammarError::Ll1Conflict(p.head.clone(), format!("{s}")));
                }
                table.insert(key, idx);
            }
            if body_first.contains(&Symbol::Epsilon) {
                if let Some(follow_set) = follow.get(&p.head) {
                    for s in follow_set {
                        let key = (p.head.clone(), s.clone());
                        if table.contains_key(&key) {
                            return Err(GrammarError::Ll1Conflict(
                                p.head.clone(),
                                format!("{s}"),
                            ));
                        }
                        table.insert(key, idx);
                    }
                }
            }
        }
        Ok(table)
    }

    // ── LR(0) items ─────────────────────────────────────────────────────

    /// Compute the closure of a set of LR(0) items.
    pub fn lr0_closure(&self, items: &BTreeSet<Lr0Item>) -> BTreeSet<Lr0Item> {
        lr0_closure_impl(items, &self.productions)
    }

    /// Compute all LR(0) item sets (canonical collection).
    pub fn lr0_item_sets(&self) -> Vec<BTreeSet<Lr0Item>> {
        lr0_item_sets_impl(&self.productions, &self.start)
    }

    // ── Left recursion detection & elimination ──────────────────────────

    /// Check if any non-terminal has direct left recursion.
    pub fn has_left_recursion(&self) -> Option<String> {
        for p in &self.productions {
            if let Some(Symbol::NonTerminal(nt)) = p.body.first() {
                if *nt == p.head {
                    return Some(p.head.clone());
                }
            }
        }
        None
    }

    /// Eliminate direct left recursion. Returns a new grammar.
    pub fn eliminate_left_recursion(&self) -> Grammar {
        let mut new_prods = Vec::new();
        let nts = self.non_terminals();

        for nt in &nts {
            let prods: Vec<&Production> = self.productions_for(nt);
            let mut left_recursive = Vec::new();
            let mut non_recursive = Vec::new();

            for p in prods {
                if let Some(Symbol::NonTerminal(first)) = p.body.first() {
                    if first == nt {
                        left_recursive.push(p);
                    } else {
                        non_recursive.push(p);
                    }
                } else {
                    non_recursive.push(p);
                }
            }

            if left_recursive.is_empty() {
                for p in non_recursive {
                    new_prods.push(p.clone());
                }
            } else {
                let new_nt = format!("{nt}'");
                // For each non-recursive alternative: A -> beta A'
                for p in &non_recursive {
                    let mut body = p.body.clone();
                    body.push(Symbol::NonTerminal(new_nt.clone()));
                    new_prods.push(Production::new(nt, body));
                }
                if non_recursive.is_empty() {
                    // A -> A'
                    new_prods.push(Production::new(
                        nt,
                        vec![Symbol::NonTerminal(new_nt.clone())],
                    ));
                }
                // For each left-recursive alternative: A' -> alpha A'
                for p in &left_recursive {
                    let mut body: Vec<Symbol> = p.body[1..].to_vec();
                    body.push(Symbol::NonTerminal(new_nt.clone()));
                    new_prods.push(Production::new(&new_nt, body));
                }
                // A' -> epsilon
                new_prods.push(Production::new(&new_nt, vec![]));
            }
        }

        Grammar {
            productions: new_prods,
            start: self.start.clone(),
        }
    }

    // ── Validation ──────────────────────────────────────────────────────

    /// Validate that all referenced non-terminals have at least one production.
    pub fn validate(&self) -> Vec<GrammarError> {
        let mut errors = Vec::new();
        let defined = self.non_terminals();

        for p in &self.productions {
            for s in &p.body {
                if let Symbol::NonTerminal(nt) = s {
                    if !defined.contains(nt) {
                        errors.push(GrammarError::UndefinedNonTerminal(nt.clone()));
                    }
                }
            }
        }

        // Check reachability from start
        let mut reachable = BTreeSet::new();
        let mut worklist = VecDeque::new();
        worklist.push_back(self.start.clone());
        while let Some(nt) = worklist.pop_front() {
            if reachable.contains(&nt) {
                continue;
            }
            reachable.insert(nt.clone());
            for p in self.productions_for(&nt) {
                for s in &p.body {
                    if let Symbol::NonTerminal(child) = s {
                        if !reachable.contains(child) {
                            worklist.push_back(child.clone());
                        }
                    }
                }
            }
        }
        let unreachable: Vec<String> = defined.difference(&reachable).cloned().collect();
        if !unreachable.is_empty() {
            errors.push(GrammarError::Unreachable(unreachable));
        }

        errors
    }
}

// ── LR(0) free functions (to avoid GAT issues) ─────────────────────────────

/// An LR(0) item.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Lr0Item {
    pub prod_index: usize,
    pub dot: usize,
}

impl fmt::Display for Lr0Item {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[prod:{}, dot:{}]", self.prod_index, self.dot)
    }
}

fn lr0_closure_impl(items: &BTreeSet<Lr0Item>, productions: &[Production]) -> BTreeSet<Lr0Item> {
    let mut closure = items.clone();
    let mut worklist: VecDeque<Lr0Item> = items.iter().cloned().collect();
    while let Some(item) = worklist.pop_front() {
        if item.prod_index >= productions.len() {
            continue;
        }
        let prod = &productions[item.prod_index];
        if item.dot < prod.body.len() {
            if let Symbol::NonTerminal(ref nt) = prod.body[item.dot] {
                for (i, p) in productions.iter().enumerate() {
                    if p.head == *nt {
                        let new_item = Lr0Item { prod_index: i, dot: 0 };
                        if closure.insert(new_item.clone()) {
                            worklist.push_back(new_item);
                        }
                    }
                }
            }
        }
    }
    closure
}

fn lr0_goto(
    items: &BTreeSet<Lr0Item>,
    sym: &Symbol,
    productions: &[Production],
) -> BTreeSet<Lr0Item> {
    let mut moved = BTreeSet::new();
    for item in items {
        if item.prod_index >= productions.len() {
            continue;
        }
        let prod = &productions[item.prod_index];
        if item.dot < prod.body.len() && prod.body[item.dot] == *sym {
            moved.insert(Lr0Item {
                prod_index: item.prod_index,
                dot: item.dot + 1,
            });
        }
    }
    lr0_closure_impl(&moved, productions)
}

fn lr0_item_sets_impl(productions: &[Production], start: &str) -> Vec<BTreeSet<Lr0Item>> {
    let mut sets: Vec<BTreeSet<Lr0Item>> = Vec::new();
    let mut set_map: HashMap<BTreeSet<Lr0Item>, usize> = HashMap::new();

    // Initial item set: closure of all S -> . body
    let mut init = BTreeSet::new();
    for (i, p) in productions.iter().enumerate() {
        if p.head == start {
            init.insert(Lr0Item { prod_index: i, dot: 0 });
        }
    }
    let init = lr0_closure_impl(&init, productions);
    set_map.insert(init.clone(), 0);
    sets.push(init.clone());

    let mut worklist: VecDeque<usize> = VecDeque::new();
    worklist.push_back(0);

    while let Some(idx) = worklist.pop_front() {
        // Collect all symbols after dots
        let current = sets[idx].clone();
        let mut symbols: BTreeSet<Symbol> = BTreeSet::new();
        for item in &current {
            if item.prod_index < productions.len() {
                let prod = &productions[item.prod_index];
                if item.dot < prod.body.len() {
                    let sym = &prod.body[item.dot];
                    if *sym != Symbol::Epsilon {
                        symbols.insert(sym.clone());
                    }
                }
            }
        }

        for sym in symbols {
            let goto_set = lr0_goto(&current, &sym, productions);
            if goto_set.is_empty() {
                continue;
            }
            if !set_map.contains_key(&goto_set) {
                let id = sets.len();
                set_map.insert(goto_set.clone(), id);
                sets.push(goto_set);
                worklist.push_back(id);
            }
        }
    }

    sets
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_grammar() -> Grammar {
        // E -> E + T | T
        // T -> T * F | F
        // F -> ( E ) | id
        let mut g = Grammar::new("E");
        g.add_production("E", vec![Symbol::nt("E"), Symbol::t("+"), Symbol::nt("T")]);
        g.add_production("E", vec![Symbol::nt("T")]);
        g.add_production("T", vec![Symbol::nt("T"), Symbol::t("*"), Symbol::nt("F")]);
        g.add_production("T", vec![Symbol::nt("F")]);
        g.add_production("F", vec![Symbol::t("("), Symbol::nt("E"), Symbol::t(")")]);
        g.add_production("F", vec![Symbol::t("id")]);
        g
    }

    fn ll1_grammar() -> Grammar {
        // S -> a B | b
        // B -> c | d
        let mut g = Grammar::new("S");
        g.add_production("S", vec![Symbol::t("a"), Symbol::nt("B")]);
        g.add_production("S", vec![Symbol::t("b")]);
        g.add_production("B", vec![Symbol::t("c")]);
        g.add_production("B", vec![Symbol::t("d")]);
        g
    }

    #[test]
    fn test_non_terminals() {
        let g = simple_grammar();
        let nts = g.non_terminals();
        assert!(nts.contains("E"));
        assert!(nts.contains("T"));
        assert!(nts.contains("F"));
    }

    #[test]
    fn test_terminals() {
        let g = simple_grammar();
        let ts = g.terminals();
        assert!(ts.contains("+"));
        assert!(ts.contains("*"));
        assert!(ts.contains("id"));
    }

    #[test]
    fn test_nullable_empty_prod() {
        let mut g = Grammar::new("S");
        g.add_production("S", vec![Symbol::nt("A")]);
        g.add_epsilon("A");
        assert!(g.is_nullable("A"));
        assert!(g.is_nullable("S"));
    }

    #[test]
    fn test_not_nullable() {
        let g = simple_grammar();
        assert!(!g.is_nullable("E"));
        assert!(!g.is_nullable("F"));
    }

    #[test]
    fn test_first_sets() {
        let g = simple_grammar();
        let first = g.first_sets();
        let e_first = &first["E"];
        assert!(e_first.contains(&Symbol::t("(")));
        assert!(e_first.contains(&Symbol::t("id")));
    }

    #[test]
    fn test_follow_sets() {
        let g = simple_grammar();
        let follow = g.follow_sets();
        let e_follow = &follow["E"];
        assert!(e_follow.contains(&Symbol::Eof));
        assert!(e_follow.contains(&Symbol::t("+")));
        assert!(e_follow.contains(&Symbol::t(")")));
    }

    #[test]
    fn test_ll1_table_simple() {
        let g = ll1_grammar();
        let table = g.ll1_table().unwrap();
        assert!(table.contains_key(&("S".into(), Symbol::t("a"))));
        assert!(table.contains_key(&("S".into(), Symbol::t("b"))));
        assert!(table.contains_key(&("B".into(), Symbol::t("c"))));
        assert!(table.contains_key(&("B".into(), Symbol::t("d"))));
    }

    #[test]
    fn test_ll1_conflict() {
        // Ambiguous grammar: S -> a | a
        let mut g = Grammar::new("S");
        g.add_production("S", vec![Symbol::t("a")]);
        g.add_production("S", vec![Symbol::t("a")]);
        let err = g.ll1_table().unwrap_err();
        assert!(matches!(err, GrammarError::Ll1Conflict(..)));
    }

    #[test]
    fn test_left_recursion_detection() {
        let g = simple_grammar();
        let lr = g.has_left_recursion();
        assert!(lr.is_some()); // E -> E + T is left recursive
    }

    #[test]
    fn test_no_left_recursion() {
        let g = ll1_grammar();
        assert!(g.has_left_recursion().is_none());
    }

    #[test]
    fn test_eliminate_left_recursion() {
        let g = simple_grammar();
        let g2 = g.eliminate_left_recursion();
        assert!(g2.has_left_recursion().is_none());
        // The new grammar should have more productions due to primed non-terminals
        assert!(g2.productions.len() > g.productions.len());
    }

    #[test]
    fn test_lr0_item_sets() {
        let g = ll1_grammar();
        let sets = g.lr0_item_sets();
        assert!(!sets.is_empty());
        // The initial set should include items for S productions
        let initial = &sets[0];
        assert!(initial.iter().any(|item| item.prod_index == 0 && item.dot == 0));
    }

    #[test]
    fn test_lr0_closure() {
        let g = ll1_grammar();
        let mut items = BTreeSet::new();
        items.insert(Lr0Item { prod_index: 0, dot: 0 });
        let closure = g.lr0_closure(&items);
        assert!(closure.len() >= 1);
    }

    #[test]
    fn test_validation_ok() {
        let g = ll1_grammar();
        let errors = g.validate();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_validation_undefined() {
        let mut g = Grammar::new("S");
        g.add_production("S", vec![Symbol::nt("X")]); // X is undefined
        let errors = g.validate();
        assert!(errors.iter().any(|e| matches!(e, GrammarError::UndefinedNonTerminal(s) if s == "X")));
    }

    #[test]
    fn test_validation_unreachable() {
        let mut g = Grammar::new("S");
        g.add_production("S", vec![Symbol::t("a")]);
        g.add_production("X", vec![Symbol::t("b")]); // X is unreachable
        let errors = g.validate();
        assert!(errors.iter().any(|e| matches!(e, GrammarError::Unreachable(..))));
    }

    #[test]
    fn test_production_display() {
        let p = Production::new("S", vec![Symbol::t("a"), Symbol::nt("B")]);
        let s = format!("{p}");
        assert!(s.contains("S ->"));
        assert!(s.contains("'a'"));
        assert!(s.contains("B"));
    }

    #[test]
    fn test_epsilon_production() {
        let p = Production::new("A", vec![]);
        assert!(p.is_epsilon());
    }

    #[test]
    fn test_productions_for() {
        let g = simple_grammar();
        let e_prods = g.productions_for("E");
        assert_eq!(e_prods.len(), 2);
    }

    #[test]
    fn test_symbol_predicates() {
        assert!(Symbol::t("a").is_terminal());
        assert!(!Symbol::t("a").is_non_terminal());
        assert!(Symbol::nt("S").is_non_terminal());
        assert!(!Symbol::nt("S").is_terminal());
    }

    #[test]
    fn test_nullable_chain() {
        let mut g = Grammar::new("S");
        g.add_production("S", vec![Symbol::nt("A"), Symbol::nt("B")]);
        g.add_epsilon("A");
        g.add_epsilon("B");
        assert!(g.is_nullable("S"));
    }

    #[test]
    fn test_first_with_nullable() {
        let mut g = Grammar::new("S");
        g.add_production("S", vec![Symbol::nt("A"), Symbol::t("b")]);
        g.add_epsilon("A");
        g.add_production("A", vec![Symbol::t("a")]);
        let first = g.first_sets();
        let s_first = &first["S"];
        assert!(s_first.contains(&Symbol::t("a")));
        assert!(s_first.contains(&Symbol::t("b")));
    }
}
