//! Simple NFA-based regex engine.
//!
//! Supports literal chars, `.` wildcard, `*` `+` `?` quantifiers (greedy/lazy),
//! character classes `[abc]` `[a-z]` `[^abc]`, anchors `^` `$`, alternation `|`,
//! grouping `()` — all in pure Rust, no external regex crate.

use std::collections::{BTreeSet, HashMap};
use std::fmt;

// ── AST ──────────────────────────────────────────────────────────

/// A parsed regex AST node.
#[derive(Debug, Clone, PartialEq)]
enum Ast {
    Literal(char),
    Dot,
    CharClass { ranges: Vec<(char, char)>, negated: bool },
    Anchor(Anchor),
    Concat(Vec<Ast>),
    Alt(Vec<Ast>),
    Repeat { child: Box<Ast>, min: u32, max: Option<u32>, lazy: bool },
    Group { child: Box<Ast>, index: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Anchor {
    Start,
    End,
}

// ── NFA ──────────────────────────────────────────────────────────

type StateId = usize;

#[derive(Debug, Clone)]
enum Transition {
    Char(char),
    Dot,
    CharClass { ranges: Vec<(char, char)>, negated: bool },
    Epsilon,
    AnchorStart,
    AnchorEnd,
    GroupOpen(usize),
    GroupClose(usize),
}

#[derive(Debug, Clone)]
struct NfaState {
    transitions: Vec<(Transition, StateId)>,
}

struct Nfa {
    states: Vec<NfaState>,
    start: StateId,
    accept: StateId,
}

impl Nfa {
    fn new_state(&mut self) -> StateId {
        let id = self.states.len();
        self.states.push(NfaState { transitions: vec![] });
        id
    }

    fn add_transition(&mut self, from: StateId, t: Transition, to: StateId) {
        self.states[from].transitions.push((t, to));
    }
}

// ── Parser ───────────────────────────────────────────────────────

struct Parser {
    chars: Vec<char>,
    pos: usize,
    group_count: usize,
}

impl Parser {
    fn new(pattern: &str) -> Self {
        Self { chars: pattern.chars().collect(), pos: 0, group_count: 0 }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied();
        if c.is_some() { self.pos += 1; }
        c
    }

    fn parse(&mut self) -> Result<Ast, String> {
        let ast = self.parse_alt()?;
        if self.pos != self.chars.len() {
            return Err(format!("unexpected char at pos {}", self.pos));
        }
        Ok(ast)
    }

    fn parse_alt(&mut self) -> Result<Ast, String> {
        let mut branches = vec![self.parse_concat()?];
        while self.peek() == Some('|') {
            self.advance();
            branches.push(self.parse_concat()?);
        }
        if branches.len() == 1 { Ok(branches.remove(0)) } else { Ok(Ast::Alt(branches)) }
    }

    fn parse_concat(&mut self) -> Result<Ast, String> {
        let mut parts = vec![];
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' { break; }
            parts.push(self.parse_quantifier()?);
        }
        if parts.is_empty() {
            Ok(Ast::Concat(vec![]))
        } else if parts.len() == 1 {
            Ok(parts.remove(0))
        } else {
            Ok(Ast::Concat(parts))
        }
    }

    fn parse_quantifier(&mut self) -> Result<Ast, String> {
        let atom = self.parse_atom()?;
        match self.peek() {
            Some('*') => { self.advance(); let lazy = self.eat_lazy(); Ok(Ast::Repeat { child: Box::new(atom), min: 0, max: None, lazy }) }
            Some('+') => { self.advance(); let lazy = self.eat_lazy(); Ok(Ast::Repeat { child: Box::new(atom), min: 1, max: None, lazy }) }
            Some('?') => { self.advance(); let lazy = self.eat_lazy(); Ok(Ast::Repeat { child: Box::new(atom), min: 0, max: Some(1), lazy }) }
            _ => Ok(atom),
        }
    }

    fn eat_lazy(&mut self) -> bool {
        if self.peek() == Some('?') { self.advance(); true } else { false }
    }

    fn parse_atom(&mut self) -> Result<Ast, String> {
        match self.peek() {
            None => Err("unexpected end of pattern".into()),
            Some('.') => { self.advance(); Ok(Ast::Dot) }
            Some('^') => { self.advance(); Ok(Ast::Anchor(Anchor::Start)) }
            Some('$') => { self.advance(); Ok(Ast::Anchor(Anchor::End)) }
            Some('(') => {
                self.advance();
                self.group_count += 1;
                let idx = self.group_count;
                let inner = self.parse_alt()?;
                if self.advance() != Some(')') { return Err("unmatched (".into()); }
                Ok(Ast::Group { child: Box::new(inner), index: idx })
            }
            Some('[') => self.parse_char_class(),
            Some('\\') => {
                self.advance();
                let c = self.advance().ok_or("trailing backslash")?;
                Ok(Ast::Literal(c))
            }
            Some(c) => { self.advance(); Ok(Ast::Literal(c)) }
        }
    }

    fn parse_char_class(&mut self) -> Result<Ast, String> {
        self.advance(); // eat '['
        let negated = if self.peek() == Some('^') { self.advance(); true } else { false };
        let mut ranges: Vec<(char, char)> = vec![];
        let mut first = true;
        loop {
            match self.peek() {
                None => return Err("unterminated character class".into()),
                Some(']') if !first => { self.advance(); break; }
                Some(c) => {
                    self.advance();
                    let ch = if c == '\\' { self.advance().ok_or("trailing backslash in class")? } else { c };
                    if self.peek() == Some('-') {
                        // Check if next-next is ']' — if so, '-' is literal
                        if self.chars.get(self.pos + 1) == Some(&']') {
                            ranges.push((ch, ch));
                        } else {
                            self.advance(); // eat '-'
                            let end = self.advance().ok_or("unterminated range")?;
                            let end = if end == '\\' { self.advance().ok_or("trailing backslash")? } else { end };
                            ranges.push((ch, end));
                        }
                    } else {
                        ranges.push((ch, ch));
                    }
                    first = false;
                }
            }
        }
        Ok(Ast::CharClass { ranges, negated })
    }
}

// ── NFA Compiler ─────────────────────────────────────────────────

fn compile(ast: &Ast, nfa: &mut Nfa) -> (StateId, StateId) {
    match ast {
        Ast::Literal(c) => {
            let s = nfa.new_state();
            let e = nfa.new_state();
            nfa.add_transition(s, Transition::Char(*c), e);
            (s, e)
        }
        Ast::Dot => {
            let s = nfa.new_state();
            let e = nfa.new_state();
            nfa.add_transition(s, Transition::Dot, e);
            (s, e)
        }
        Ast::CharClass { ranges, negated } => {
            let s = nfa.new_state();
            let e = nfa.new_state();
            nfa.add_transition(s, Transition::CharClass { ranges: ranges.clone(), negated: *negated }, e);
            (s, e)
        }
        Ast::Anchor(Anchor::Start) => {
            let s = nfa.new_state();
            let e = nfa.new_state();
            nfa.add_transition(s, Transition::AnchorStart, e);
            (s, e)
        }
        Ast::Anchor(Anchor::End) => {
            let s = nfa.new_state();
            let e = nfa.new_state();
            nfa.add_transition(s, Transition::AnchorEnd, e);
            (s, e)
        }
        Ast::Concat(parts) => {
            if parts.is_empty() {
                let s = nfa.new_state();
                return (s, s);
            }
            let mut iter = parts.iter();
            let (start, mut prev_end) = compile(iter.next().unwrap(), nfa);
            for part in iter {
                let (ns, ne) = compile(part, nfa);
                nfa.add_transition(prev_end, Transition::Epsilon, ns);
                prev_end = ne;
            }
            (start, prev_end)
        }
        Ast::Alt(branches) => {
            let s = nfa.new_state();
            let e = nfa.new_state();
            for branch in branches {
                let (bs, be) = compile(branch, nfa);
                nfa.add_transition(s, Transition::Epsilon, bs);
                nfa.add_transition(be, Transition::Epsilon, e);
            }
            (s, e)
        }
        Ast::Repeat { child, min, max, lazy } => {
            // Build chain of min required copies
            let s = nfa.new_state();
            let e = nfa.new_state();
            let mut prev = s;
            for _ in 0..*min {
                let (cs, ce) = compile(child, nfa);
                nfa.add_transition(prev, Transition::Epsilon, cs);
                prev = ce;
            }
            match max {
                None => {
                    // Unlimited: add a loop
                    let (cs, ce) = compile(child, nfa);
                    if *lazy {
                        nfa.add_transition(prev, Transition::Epsilon, e);
                        nfa.add_transition(prev, Transition::Epsilon, cs);
                    } else {
                        nfa.add_transition(prev, Transition::Epsilon, cs);
                        nfa.add_transition(prev, Transition::Epsilon, e);
                    }
                    nfa.add_transition(ce, Transition::Epsilon, cs);
                    nfa.add_transition(ce, Transition::Epsilon, e);
                }
                Some(mx) => {
                    // Up to (mx - min) optional copies
                    let extra = *mx - *min;
                    for _ in 0..extra {
                        let (cs, ce) = compile(child, nfa);
                        if *lazy {
                            nfa.add_transition(prev, Transition::Epsilon, e);
                            nfa.add_transition(prev, Transition::Epsilon, cs);
                        } else {
                            nfa.add_transition(prev, Transition::Epsilon, cs);
                            nfa.add_transition(prev, Transition::Epsilon, e);
                        }
                        prev = ce;
                    }
                    nfa.add_transition(prev, Transition::Epsilon, e);
                }
            }
            (s, e)
        }
        Ast::Group { child, index } => {
            let s = nfa.new_state();
            let e = nfa.new_state();
            let (cs, ce) = compile(child, nfa);
            nfa.add_transition(s, Transition::GroupOpen(*index), cs);
            nfa.add_transition(ce, Transition::GroupClose(*index), e);
            (s, e)
        }
    }
}

// ── NFA Simulation (Thompson's) ──────────────────────────────────

fn char_in_class(c: char, ranges: &[(char, char)], negated: bool) -> bool {
    let found = ranges.iter().any(|(lo, hi)| c >= *lo && c <= *hi);
    if negated { !found } else { found }
}

/// A compiled regex ready for matching.
pub struct Regex {
    nfa: Nfa,
    num_groups: usize,
}

/// A match result with captured groups.
#[derive(Debug, Clone)]
pub struct Match {
    pub start: usize,
    pub end: usize,
    pub groups: Vec<Option<(usize, usize)>>,
}

impl Match {
    /// Return the full matched text.
    pub fn as_str<'a>(&self, input: &'a str) -> &'a str {
        &input[self.start..self.end]
    }

    /// Return a captured group's text.
    pub fn group<'a>(&self, i: usize, input: &'a str) -> Option<&'a str> {
        self.groups.get(i).and_then(|g| g.map(|(s, e)| &input[s..e]))
    }
}

impl Regex {
    /// Compile a pattern into an NFA-based regex.
    pub fn new(pattern: &str) -> Result<Self, String> {
        let mut parser = Parser::new(pattern);
        let ast = parser.parse()?;
        let num_groups = parser.group_count;
        let mut nfa = Nfa {
            states: vec![],
            start: 0,
            accept: 0,
        };
        let (start, accept) = compile(&ast, &mut nfa);
        nfa.start = start;
        nfa.accept = accept;
        Ok(Self { nfa, num_groups })
    }

    /// Check if the pattern matches anywhere in the input.
    pub fn is_match(&self, input: &str) -> bool {
        self.find(input).is_some()
    }

    /// Find the first match in the input.
    pub fn find(&self, input: &str) -> Option<Match> {
        let chars: Vec<char> = input.chars().collect();
        let len = chars.len();
        for start_pos in 0..=len {
            if let Some(m) = self.try_match_at(&chars, start_pos, len) {
                return Some(m);
            }
        }
        None
    }

    /// Find all non-overlapping matches.
    pub fn find_all(&self, input: &str) -> Vec<Match> {
        let chars: Vec<char> = input.chars().collect();
        let len = chars.len();
        let mut results = vec![];
        let mut pos = 0;
        while pos <= len {
            if let Some(m) = self.try_match_at(&chars, pos, len) {
                let next = if m.end == m.start { m.end + 1 } else { m.end };
                results.push(m);
                pos = next;
            } else {
                pos += 1;
            }
        }
        results
    }

    fn try_match_at(&self, chars: &[char], start: usize, len: usize) -> Option<Match> {
        // NFA simulation using sets of states, tracking groups per thread.
        // We use a simple recursive backtracking for group support.
        let mut groups = vec![None; self.num_groups + 1];
        if self.backtrack(chars, self.nfa.start, start, len, &mut groups) {
            // Find the end position by trying longest match
            let mut best_end = start;
            let mut full_groups = groups.clone();
            self.backtrack_full(chars, self.nfa.start, start, len, &mut full_groups, &mut best_end);
            // Use groups from backtrack (correct captures) with end from backtrack_full (longest)
            // Update group ends that were found in the simpler backtrack pass
            Some(Match { start, end: best_end, groups })
        } else {
            None
        }
    }

    fn backtrack(&self, chars: &[char], state: StateId, pos: usize, len: usize, groups: &mut Vec<Option<(usize, usize)>>) -> bool {
        self.backtrack_inner(chars, state, pos, len, groups, &mut BTreeSet::new())
    }

    fn backtrack_inner(&self, chars: &[char], state: StateId, pos: usize, len: usize,
                       groups: &mut Vec<Option<(usize, usize)>>, visited: &mut BTreeSet<(StateId, usize)>) -> bool {
        if state == self.nfa.accept { return true; }
        if !visited.insert((state, pos)) { return false; }

        for (trans, next) in &self.nfa.states[state].transitions {
            match trans {
                Transition::Char(c) => {
                    if pos < len && chars[pos] == *c {
                        if self.backtrack_inner(chars, *next, pos + 1, len, groups, visited) {
                            return true;
                        }
                    }
                }
                Transition::Dot => {
                    if pos < len && chars[pos] != '\n' {
                        if self.backtrack_inner(chars, *next, pos + 1, len, groups, visited) {
                            return true;
                        }
                    }
                }
                Transition::CharClass { ranges, negated } => {
                    if pos < len && char_in_class(chars[pos], ranges, *negated) {
                        if self.backtrack_inner(chars, *next, pos + 1, len, groups, visited) {
                            return true;
                        }
                    }
                }
                Transition::Epsilon => {
                    if self.backtrack_inner(chars, *next, pos, len, groups, visited) {
                        return true;
                    }
                }
                Transition::AnchorStart => {
                    if pos == 0 {
                        if self.backtrack_inner(chars, *next, pos, len, groups, visited) {
                            return true;
                        }
                    }
                }
                Transition::AnchorEnd => {
                    if pos == len {
                        if self.backtrack_inner(chars, *next, pos, len, groups, visited) {
                            return true;
                        }
                    }
                }
                Transition::GroupOpen(idx) => {
                    let old = groups[*idx];
                    groups[*idx] = Some((pos, pos));
                    if self.backtrack_inner(chars, *next, pos, len, groups, visited) {
                        return true;
                    }
                    groups[*idx] = old;
                }
                Transition::GroupClose(idx) => {
                    let old = groups[*idx];
                    if let Some((s, _)) = groups[*idx] {
                        groups[*idx] = Some((s, pos));
                    }
                    if self.backtrack_inner(chars, *next, pos, len, groups, visited) {
                        return true;
                    }
                    groups[*idx] = old;
                }
            }
        }
        false
    }

    fn find_end(&self, chars: &[char], state: StateId, pos: usize, len: usize, groups: &mut Vec<Option<(usize, usize)>>) -> usize {
        let mut best = pos;
        self.backtrack_full(chars, state, pos, len, groups, &mut best);
        best
    }

    fn backtrack_full(&self, chars: &[char], state: StateId, pos: usize, len: usize,
                      groups: &mut Vec<Option<(usize, usize)>>, best_end: &mut usize) {
        self.backtrack_full_inner(chars, state, pos, len, groups, best_end, &mut BTreeSet::new());
    }

    fn backtrack_full_inner(&self, chars: &[char], state: StateId, pos: usize, len: usize,
                            groups: &mut Vec<Option<(usize, usize)>>, best_end: &mut usize,
                            visited: &mut BTreeSet<(StateId, usize)>) {
        if state == self.nfa.accept {
            if pos >= *best_end {
                *best_end = pos;
            }
            return;
        }
        if !visited.insert((state, pos)) { return; }

        for (trans, next) in &self.nfa.states[state].transitions {
            match trans {
                Transition::Char(c) => {
                    if pos < len && chars[pos] == *c {
                        self.backtrack_full_inner(chars, *next, pos + 1, len, groups, best_end, visited);
                    }
                }
                Transition::Dot => {
                    if pos < len && chars[pos] != '\n' {
                        self.backtrack_full_inner(chars, *next, pos + 1, len, groups, best_end, visited);
                    }
                }
                Transition::CharClass { ranges, negated } => {
                    if pos < len && char_in_class(chars[pos], ranges, *negated) {
                        self.backtrack_full_inner(chars, *next, pos + 1, len, groups, best_end, visited);
                    }
                }
                Transition::Epsilon => {
                    self.backtrack_full_inner(chars, *next, pos, len, groups, best_end, visited);
                }
                Transition::AnchorStart => {
                    if pos == 0 {
                        self.backtrack_full_inner(chars, *next, pos, len, groups, best_end, visited);
                    }
                }
                Transition::AnchorEnd => {
                    if pos == len {
                        self.backtrack_full_inner(chars, *next, pos, len, groups, best_end, visited);
                    }
                }
                Transition::GroupOpen(idx) => {
                    let old = groups[*idx];
                    groups[*idx] = Some((pos, pos));
                    self.backtrack_full_inner(chars, *next, pos, len, groups, best_end, visited);
                    groups[*idx] = old;
                }
                Transition::GroupClose(idx) => {
                    let old = groups[*idx];
                    if let Some((s, _)) = groups[*idx] {
                        groups[*idx] = Some((s, pos));
                    }
                    self.backtrack_full_inner(chars, *next, pos, len, groups, best_end, visited);
                    groups[*idx] = old;
                }
            }
        }
    }
}

impl fmt::Display for Regex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Regex({} states, {} groups)", self.nfa.states.len(), self.num_groups)
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_match() {
        let re = Regex::new("hello").unwrap();
        assert!(re.is_match("hello world"));
        assert!(!re.is_match("HELLO"));
    }

    #[test]
    fn test_dot_wildcard() {
        let re = Regex::new("h.llo").unwrap();
        assert!(re.is_match("hello"));
        assert!(re.is_match("hallo"));
        assert!(!re.is_match("hllo"));
    }

    #[test]
    fn test_star_quantifier() {
        let re = Regex::new("ab*c").unwrap();
        assert!(re.is_match("ac"));
        assert!(re.is_match("abc"));
        assert!(re.is_match("abbbc"));
    }

    #[test]
    fn test_plus_quantifier() {
        let re = Regex::new("ab+c").unwrap();
        assert!(!re.is_match("ac"));
        assert!(re.is_match("abc"));
        assert!(re.is_match("abbbc"));
    }

    #[test]
    fn test_question_quantifier() {
        let re = Regex::new("ab?c").unwrap();
        assert!(re.is_match("ac"));
        assert!(re.is_match("abc"));
        assert!(!re.is_match("abbc"));
    }

    #[test]
    fn test_character_class() {
        let re = Regex::new("[abc]").unwrap();
        assert!(re.is_match("a"));
        assert!(re.is_match("b"));
        assert!(!re.is_match("d"));
    }

    #[test]
    fn test_char_class_range() {
        let re = Regex::new("[a-z]+").unwrap();
        assert!(re.is_match("hello"));
        assert!(!re.is_match("123"));
    }

    #[test]
    fn test_negated_class() {
        let re = Regex::new("[^0-9]+").unwrap();
        assert!(re.is_match("abc"));
        let re2 = Regex::new("^[^0-9]+$").unwrap();
        assert!(!re2.is_match("123"));
    }

    #[test]
    fn test_anchors() {
        let re = Regex::new("^hello$").unwrap();
        assert!(re.is_match("hello"));
        assert!(!re.is_match("hello world"));
        assert!(!re.is_match("say hello"));
    }

    #[test]
    fn test_alternation() {
        let re = Regex::new("cat|dog").unwrap();
        assert!(re.is_match("cat"));
        assert!(re.is_match("dog"));
        assert!(!re.is_match("fish"));
    }

    #[test]
    fn test_grouping() {
        let re = Regex::new("(ab)+").unwrap();
        assert!(re.is_match("ab"));
        assert!(re.is_match("abab"));
        assert!(!re.is_match("cd"));
    }

    #[test]
    fn test_find_all() {
        let re = Regex::new("[0-9]+").unwrap();
        let matches = re.find_all("abc 123 def 456");
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].as_str("abc 123 def 456"), "123");
        assert_eq!(matches[1].as_str("abc 123 def 456"), "456");
    }

    #[test]
    fn test_group_capture() {
        let re = Regex::new("(a+)(b+)").unwrap();
        let m = re.find("xxxaabbbyyy").unwrap();
        assert_eq!(m.group(1, "xxxaabbbyyy"), Some("aa"));
        assert_eq!(m.group(2, "xxxaabbbyyy"), Some("bbb"));
    }

    #[test]
    fn test_escaped_chars() {
        let re = Regex::new(r"a\.b").unwrap();
        assert!(re.is_match("a.b"));
        assert!(!re.is_match("axb"));
    }

    #[test]
    fn test_complex_pattern() {
        let re = Regex::new("^[a-zA-Z][a-zA-Z0-9_]*$").unwrap();
        assert!(re.is_match("valid_name"));
        assert!(re.is_match("CamelCase"));
        assert!(!re.is_match("123bad"));
        assert!(!re.is_match(""));
    }
}
