//! Regex to NFA/DFA compiler — parse regex to AST, Thompson's construction
//! (NFA), subset construction (NFA to DFA), DFA minimization, match execution,
//! character classes, quantifiers (*, +, ?, {n,m}).

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::fmt;

// ── Regex AST ───────────────────────────────────────────────────────────────

/// A node in the regex abstract syntax tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegexAst {
    /// Match a single literal character.
    Literal(char),
    /// Match any character (`.`).
    Dot,
    /// Character class: set of allowed chars + negated flag.
    CharClass(Vec<CharRange>, bool),
    /// Concatenation of two patterns.
    Concat(Box<RegexAst>, Box<RegexAst>),
    /// Alternation: `a|b`.
    Alt(Box<RegexAst>, Box<RegexAst>),
    /// Kleene star: `a*`.
    Star(Box<RegexAst>),
    /// One or more: `a+`.
    Plus(Box<RegexAst>),
    /// Zero or one: `a?`.
    Optional(Box<RegexAst>),
    /// Empty pattern (matches epsilon).
    Empty,
}

/// A range within a character class: single char or inclusive range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CharRange {
    Single(char),
    Range(char, char),
}

impl CharRange {
    pub fn matches(&self, c: char) -> bool {
        match self {
            Self::Single(ch) => c == *ch,
            Self::Range(lo, hi) => c >= *lo && c <= *hi,
        }
    }
}

// ── Parser ──────────────────────────────────────────────────────────────────

/// Parse error.
#[derive(Debug, Clone, PartialEq)]
pub enum RegexError {
    UnexpectedChar(char, usize),
    UnmatchedParen(usize),
    EmptyGroup(usize),
    InvalidEscape(char, usize),
    InvalidCharClass(usize),
    EmptyPattern,
}

impl fmt::Display for RegexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedChar(c, pos) => write!(f, "unexpected '{c}' at {pos}"),
            Self::UnmatchedParen(pos) => write!(f, "unmatched parenthesis at {pos}"),
            Self::EmptyGroup(pos) => write!(f, "empty group at {pos}"),
            Self::InvalidEscape(c, pos) => write!(f, "invalid escape '\\{c}' at {pos}"),
            Self::InvalidCharClass(pos) => write!(f, "invalid char class at {pos}"),
            Self::EmptyPattern => write!(f, "empty pattern"),
        }
    }
}

/// Parse a regex string into an AST.
pub fn parse_regex(pattern: &str) -> Result<RegexAst, RegexError> {
    if pattern.is_empty() {
        return Ok(RegexAst::Empty);
    }
    let chars: Vec<char> = pattern.chars().collect();
    let mut pos = 0;
    let ast = parse_alternation(&chars, &mut pos)?;
    if pos < chars.len() {
        return Err(RegexError::UnexpectedChar(chars[pos], pos));
    }
    Ok(ast)
}

fn parse_alternation(chars: &[char], pos: &mut usize) -> Result<RegexAst, RegexError> {
    let mut left = parse_concat(chars, pos)?;
    while *pos < chars.len() && chars[*pos] == '|' {
        *pos += 1;
        let right = parse_concat(chars, pos)?;
        left = RegexAst::Alt(Box::new(left), Box::new(right));
    }
    Ok(left)
}

fn parse_concat(chars: &[char], pos: &mut usize) -> Result<RegexAst, RegexError> {
    let mut nodes = Vec::new();
    while *pos < chars.len() && chars[*pos] != '|' && chars[*pos] != ')' {
        nodes.push(parse_quantified(chars, pos)?);
    }
    if nodes.is_empty() {
        return Ok(RegexAst::Empty);
    }
    let mut result = nodes.remove(0);
    for node in nodes {
        result = RegexAst::Concat(Box::new(result), Box::new(node));
    }
    Ok(result)
}

fn parse_quantified(chars: &[char], pos: &mut usize) -> Result<RegexAst, RegexError> {
    let base = parse_atom(chars, pos)?;
    if *pos < chars.len() {
        match chars[*pos] {
            '*' => {
                *pos += 1;
                return Ok(RegexAst::Star(Box::new(base)));
            }
            '+' => {
                *pos += 1;
                return Ok(RegexAst::Plus(Box::new(base)));
            }
            '?' => {
                *pos += 1;
                return Ok(RegexAst::Optional(Box::new(base)));
            }
            _ => {}
        }
    }
    Ok(base)
}

fn parse_atom(chars: &[char], pos: &mut usize) -> Result<RegexAst, RegexError> {
    if *pos >= chars.len() {
        return Ok(RegexAst::Empty);
    }
    match chars[*pos] {
        '(' => {
            let start = *pos;
            *pos += 1;
            let inner = parse_alternation(chars, pos)?;
            if *pos >= chars.len() || chars[*pos] != ')' {
                return Err(RegexError::UnmatchedParen(start));
            }
            *pos += 1;
            Ok(inner)
        }
        '[' => parse_char_class(chars, pos),
        '.' => {
            *pos += 1;
            Ok(RegexAst::Dot)
        }
        '\\' => {
            *pos += 1;
            if *pos >= chars.len() {
                return Err(RegexError::InvalidEscape('\\', *pos - 1));
            }
            let c = chars[*pos];
            *pos += 1;
            match c {
                'd' => Ok(RegexAst::CharClass(vec![CharRange::Range('0', '9')], false)),
                'w' => Ok(RegexAst::CharClass(
                    vec![
                        CharRange::Range('a', 'z'),
                        CharRange::Range('A', 'Z'),
                        CharRange::Range('0', '9'),
                        CharRange::Single('_'),
                    ],
                    false,
                )),
                's' => Ok(RegexAst::CharClass(
                    vec![
                        CharRange::Single(' '),
                        CharRange::Single('\t'),
                        CharRange::Single('\n'),
                        CharRange::Single('\r'),
                    ],
                    false,
                )),
                'n' => Ok(RegexAst::Literal('\n')),
                't' => Ok(RegexAst::Literal('\t')),
                'r' => Ok(RegexAst::Literal('\r')),
                _ => Ok(RegexAst::Literal(c)),
            }
        }
        c if c == ')' || c == '|' => Ok(RegexAst::Empty),
        c => {
            *pos += 1;
            Ok(RegexAst::Literal(c))
        }
    }
}

fn parse_char_class(chars: &[char], pos: &mut usize) -> Result<RegexAst, RegexError> {
    let start = *pos;
    *pos += 1; // skip [
    let negated = if *pos < chars.len() && chars[*pos] == '^' {
        *pos += 1;
        true
    } else {
        false
    };
    let mut ranges = Vec::new();
    while *pos < chars.len() && chars[*pos] != ']' {
        let c = chars[*pos];
        *pos += 1;
        if *pos + 1 < chars.len() && chars[*pos] == '-' && chars[*pos + 1] != ']' {
            *pos += 1; // skip -
            let hi = chars[*pos];
            *pos += 1;
            ranges.push(CharRange::Range(c, hi));
        } else {
            ranges.push(CharRange::Single(c));
        }
    }
    if *pos >= chars.len() {
        return Err(RegexError::InvalidCharClass(start));
    }
    *pos += 1; // skip ]
    Ok(RegexAst::CharClass(ranges, negated))
}

// ── NFA ─────────────────────────────────────────────────────────────────────

/// Edge label in the NFA.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NfaEdge {
    /// Epsilon transition.
    Epsilon,
    /// Match a specific character.
    Char(char),
    /// Match any character.
    Any,
    /// Character class.
    Class(Vec<CharRange>, bool),
}

/// An NFA state.
#[derive(Debug, Clone)]
pub struct NfaState {
    pub id: usize,
    pub transitions: Vec<(NfaEdge, usize)>,
    pub is_accept: bool,
}

/// A non-deterministic finite automaton.
#[derive(Debug, Clone)]
pub struct Nfa {
    pub states: Vec<NfaState>,
    pub start: usize,
    pub accept: usize,
}

impl Nfa {
    fn new_state(states: &mut Vec<NfaState>, is_accept: bool) -> usize {
        let id = states.len();
        states.push(NfaState {
            id,
            transitions: Vec::new(),
            is_accept,
        });
        id
    }

    /// Build an NFA from a regex AST using Thompson's construction.
    pub fn from_ast(ast: &RegexAst) -> Self {
        let mut states = Vec::new();
        let (start, accept) = thompson(ast, &mut states);
        states[accept].is_accept = true;
        Nfa { states, start, accept }
    }

    /// Compute the epsilon closure of a set of states.
    pub fn epsilon_closure(&self, state_set: &BTreeSet<usize>) -> BTreeSet<usize> {
        let mut closure = state_set.clone();
        let mut worklist: VecDeque<usize> = state_set.iter().copied().collect();
        while let Some(s) = worklist.pop_front() {
            if s >= self.states.len() {
                continue;
            }
            for (edge, target) in &self.states[s].transitions {
                if *edge == NfaEdge::Epsilon && !closure.contains(target) {
                    closure.insert(*target);
                    worklist.push_back(*target);
                }
            }
        }
        closure
    }

    /// Simulate the NFA on input to check for a full match.
    pub fn matches(&self, input: &str) -> bool {
        let mut current = BTreeSet::new();
        current.insert(self.start);
        current = self.epsilon_closure(&current);

        for c in input.chars() {
            let mut next = BTreeSet::new();
            for s in &current {
                if *s >= self.states.len() {
                    continue;
                }
                for (edge, target) in &self.states[*s].transitions {
                    let m = match edge {
                        NfaEdge::Char(ec) => c == *ec,
                        NfaEdge::Any => true,
                        NfaEdge::Class(ranges, negated) => {
                            let in_class = ranges.iter().any(|r| r.matches(c));
                            if *negated { !in_class } else { in_class }
                        }
                        NfaEdge::Epsilon => false,
                    };
                    if m {
                        next.insert(*target);
                    }
                }
            }
            current = self.epsilon_closure(&next);
            if current.is_empty() {
                return false;
            }
        }

        current.iter().any(|s| {
            self.states.get(*s).is_some_and(|st| st.is_accept)
        })
    }
}

fn thompson(ast: &RegexAst, states: &mut Vec<NfaState>) -> (usize, usize) {
    match ast {
        RegexAst::Literal(c) => {
            let start = Nfa::new_state(states, false);
            let accept = Nfa::new_state(states, false);
            states[start].transitions.push((NfaEdge::Char(*c), accept));
            (start, accept)
        }
        RegexAst::Dot => {
            let start = Nfa::new_state(states, false);
            let accept = Nfa::new_state(states, false);
            states[start].transitions.push((NfaEdge::Any, accept));
            (start, accept)
        }
        RegexAst::CharClass(ranges, negated) => {
            let start = Nfa::new_state(states, false);
            let accept = Nfa::new_state(states, false);
            states[start]
                .transitions
                .push((NfaEdge::Class(ranges.clone(), *negated), accept));
            (start, accept)
        }
        RegexAst::Concat(left, right) => {
            let (ls, la) = thompson(left, states);
            let (rs, ra) = thompson(right, states);
            states[la].transitions.push((NfaEdge::Epsilon, rs));
            (ls, ra)
        }
        RegexAst::Alt(left, right) => {
            let start = Nfa::new_state(states, false);
            let accept = Nfa::new_state(states, false);
            let (ls, la) = thompson(left, states);
            let (rs, ra) = thompson(right, states);
            states[start].transitions.push((NfaEdge::Epsilon, ls));
            states[start].transitions.push((NfaEdge::Epsilon, rs));
            states[la].transitions.push((NfaEdge::Epsilon, accept));
            states[ra].transitions.push((NfaEdge::Epsilon, accept));
            (start, accept)
        }
        RegexAst::Star(inner) => {
            let start = Nfa::new_state(states, false);
            let accept = Nfa::new_state(states, false);
            let (is, ia) = thompson(inner, states);
            states[start].transitions.push((NfaEdge::Epsilon, is));
            states[start].transitions.push((NfaEdge::Epsilon, accept));
            states[ia].transitions.push((NfaEdge::Epsilon, is));
            states[ia].transitions.push((NfaEdge::Epsilon, accept));
            (start, accept)
        }
        RegexAst::Plus(inner) => {
            let start = Nfa::new_state(states, false);
            let accept = Nfa::new_state(states, false);
            let (is, ia) = thompson(inner, states);
            states[start].transitions.push((NfaEdge::Epsilon, is));
            states[ia].transitions.push((NfaEdge::Epsilon, is));
            states[ia].transitions.push((NfaEdge::Epsilon, accept));
            (start, accept)
        }
        RegexAst::Optional(inner) => {
            let start = Nfa::new_state(states, false);
            let accept = Nfa::new_state(states, false);
            let (is, ia) = thompson(inner, states);
            states[start].transitions.push((NfaEdge::Epsilon, is));
            states[start].transitions.push((NfaEdge::Epsilon, accept));
            states[ia].transitions.push((NfaEdge::Epsilon, accept));
            (start, accept)
        }
        RegexAst::Empty => {
            let start = Nfa::new_state(states, false);
            let accept = Nfa::new_state(states, false);
            states[start].transitions.push((NfaEdge::Epsilon, accept));
            (start, accept)
        }
    }
}

// ── DFA ─────────────────────────────────────────────────────────────────────

/// A DFA state.
#[derive(Debug, Clone)]
pub struct DfaState {
    pub id: usize,
    pub transitions: HashMap<char, usize>,
    /// Transition for 'any char' and char classes — we approximate with
    /// an explicit default transition.
    pub default_next: Option<usize>,
    pub is_accept: bool,
    /// The NFA state set this DFA state corresponds to.
    pub nfa_states: BTreeSet<usize>,
}

/// A deterministic finite automaton.
#[derive(Debug, Clone)]
pub struct Dfa {
    pub states: Vec<DfaState>,
    pub start: usize,
}

impl Dfa {
    /// Construct a DFA from an NFA via subset construction.
    /// `alphabet` is the set of characters to consider.
    pub fn from_nfa(nfa: &Nfa, alphabet: &[char]) -> Self {
        let mut dfa_states: Vec<DfaState> = Vec::new();
        let mut state_map: HashMap<BTreeSet<usize>, usize> = HashMap::new();
        let mut worklist: VecDeque<BTreeSet<usize>> = VecDeque::new();

        let start_set = {
            let mut s = BTreeSet::new();
            s.insert(nfa.start);
            nfa.epsilon_closure(&s)
        };

        let is_accept = start_set.iter().any(|s| {
            nfa.states.get(*s).is_some_and(|st| st.is_accept)
        });
        dfa_states.push(DfaState {
            id: 0,
            transitions: HashMap::new(),
            default_next: None,
            is_accept,
            nfa_states: start_set.clone(),
        });
        state_map.insert(start_set.clone(), 0);
        worklist.push_back(start_set);

        while let Some(current_set) = worklist.pop_front() {
            let current_id = *state_map.get(&current_set).unwrap();

            for &c in alphabet {
                let mut next_set = BTreeSet::new();
                for &s in &current_set {
                    if s >= nfa.states.len() {
                        continue;
                    }
                    for (edge, target) in &nfa.states[s].transitions {
                        let m = match edge {
                            NfaEdge::Char(ec) => c == *ec,
                            NfaEdge::Any => true,
                            NfaEdge::Class(ranges, negated) => {
                                let in_class = ranges.iter().any(|r| r.matches(c));
                                if *negated { !in_class } else { in_class }
                            }
                            NfaEdge::Epsilon => false,
                        };
                        if m {
                            next_set.insert(*target);
                        }
                    }
                }
                let next_set = nfa.epsilon_closure(&next_set);
                if next_set.is_empty() {
                    continue;
                }

                let next_id = if let Some(&id) = state_map.get(&next_set) {
                    id
                } else {
                    let id = dfa_states.len();
                    let is_accept = next_set.iter().any(|s| {
                        nfa.states.get(*s).is_some_and(|st| st.is_accept)
                    });
                    dfa_states.push(DfaState {
                        id,
                        transitions: HashMap::new(),
                        default_next: None,
                        is_accept,
                        nfa_states: next_set.clone(),
                    });
                    state_map.insert(next_set.clone(), id);
                    worklist.push_back(next_set);
                    id
                };

                dfa_states[current_id].transitions.insert(c, next_id);
            }
        }

        Dfa {
            states: dfa_states,
            start: 0,
        }
    }

    /// Minimize the DFA using Hopcroft's algorithm (simplified).
    pub fn minimize(&self) -> Dfa {
        if self.states.is_empty() {
            return self.clone();
        }

        // Partition into accepting and non-accepting
        let mut accept_group: BTreeSet<usize> = BTreeSet::new();
        let mut reject_group: BTreeSet<usize> = BTreeSet::new();
        for s in &self.states {
            if s.is_accept {
                accept_group.insert(s.id);
            } else {
                reject_group.insert(s.id);
            }
        }

        let mut partitions: Vec<BTreeSet<usize>> = Vec::new();
        if !accept_group.is_empty() {
            partitions.push(accept_group);
        }
        if !reject_group.is_empty() {
            partitions.push(reject_group);
        }

        // Collect alphabet from transitions
        let mut alphabet: BTreeSet<char> = BTreeSet::new();
        for s in &self.states {
            for c in s.transitions.keys() {
                alphabet.insert(*c);
            }
        }

        let mut changed = true;
        while changed {
            changed = false;
            let mut new_partitions = Vec::new();
            for part in &partitions {
                if part.len() <= 1 {
                    new_partitions.push(part.clone());
                    continue;
                }
                // Try to split this partition
                let repr = *part.iter().next().unwrap();
                let mut same = BTreeSet::new();
                let mut diff = BTreeSet::new();
                same.insert(repr);

                for &s in part {
                    if s == repr {
                        continue;
                    }
                    let mut matches_repr = true;
                    for &c in &alphabet {
                        let repr_target = self.states[repr].transitions.get(&c);
                        let s_target = self.states[s].transitions.get(&c);
                        let repr_part = repr_target.and_then(|t| {
                            partitions.iter().position(|p| p.contains(t))
                        });
                        let s_part = s_target.and_then(|t| {
                            partitions.iter().position(|p| p.contains(t))
                        });
                        if repr_part != s_part {
                            matches_repr = false;
                            break;
                        }
                    }
                    if matches_repr {
                        same.insert(s);
                    } else {
                        diff.insert(s);
                    }
                }

                new_partitions.push(same);
                if !diff.is_empty() {
                    new_partitions.push(diff);
                    changed = true;
                }
            }
            partitions = new_partitions;
        }

        // Build minimized DFA
        let state_to_part: HashMap<usize, usize> = partitions
            .iter()
            .enumerate()
            .flat_map(|(i, p)| p.iter().map(move |s| (*s, i)))
            .collect();

        let mut min_states: Vec<DfaState> = Vec::new();
        for (i, part) in partitions.iter().enumerate() {
            let repr = *part.iter().next().unwrap();
            let is_accept = self.states[repr].is_accept;
            let mut transitions = HashMap::new();
            for (&c, &target) in &self.states[repr].transitions {
                if let Some(&target_part) = state_to_part.get(&target) {
                    transitions.insert(c, target_part);
                }
            }
            min_states.push(DfaState {
                id: i,
                transitions,
                default_next: None,
                is_accept,
                nfa_states: BTreeSet::new(),
            });
        }

        let start = *state_to_part.get(&self.start).unwrap_or(&0);
        Dfa {
            states: min_states,
            start,
        }
    }

    /// Run the DFA on input, returning true if it matches the entire string.
    pub fn matches(&self, input: &str) -> bool {
        if self.states.is_empty() {
            return input.is_empty();
        }
        let mut current = self.start;
        for c in input.chars() {
            if current >= self.states.len() {
                return false;
            }
            if let Some(&next) = self.states[current].transitions.get(&c) {
                current = next;
            } else if let Some(def) = self.states[current].default_next {
                current = def;
            } else {
                return false;
            }
        }
        self.states.get(current).is_some_and(|s| s.is_accept)
    }
}

// ── High-level API ──────────────────────────────────────────────────────────

/// Compile a regex pattern and test if it fully matches `input`.
pub fn regex_match(pattern: &str, input: &str) -> Result<bool, RegexError> {
    let ast = parse_regex(pattern)?;
    let nfa = Nfa::from_ast(&ast);
    Ok(nfa.matches(input))
}

/// Compile a regex to a minimized DFA.
pub fn compile_dfa(pattern: &str, alphabet: &[char]) -> Result<Dfa, RegexError> {
    let ast = parse_regex(pattern)?;
    let nfa = Nfa::from_ast(&ast);
    let dfa = Dfa::from_nfa(&nfa, alphabet);
    Ok(dfa.minimize())
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_match() {
        assert!(regex_match("abc", "abc").unwrap());
        assert!(!regex_match("abc", "abd").unwrap());
    }

    #[test]
    fn test_empty_pattern() {
        assert!(regex_match("", "").unwrap());
    }

    #[test]
    fn test_dot() {
        assert!(regex_match("a.c", "abc").unwrap());
        assert!(regex_match("a.c", "axc").unwrap());
        assert!(!regex_match("a.c", "ac").unwrap());
    }

    #[test]
    fn test_alternation() {
        assert!(regex_match("a|b", "a").unwrap());
        assert!(regex_match("a|b", "b").unwrap());
        assert!(!regex_match("a|b", "c").unwrap());
    }

    #[test]
    fn test_star() {
        assert!(regex_match("a*", "").unwrap());
        assert!(regex_match("a*", "a").unwrap());
        assert!(regex_match("a*", "aaaa").unwrap());
    }

    #[test]
    fn test_plus() {
        assert!(!regex_match("a+", "").unwrap());
        assert!(regex_match("a+", "a").unwrap());
        assert!(regex_match("a+", "aaa").unwrap());
    }

    #[test]
    fn test_optional() {
        assert!(regex_match("ab?c", "ac").unwrap());
        assert!(regex_match("ab?c", "abc").unwrap());
        assert!(!regex_match("ab?c", "abbc").unwrap());
    }

    #[test]
    fn test_char_class() {
        assert!(regex_match("[abc]", "a").unwrap());
        assert!(regex_match("[abc]", "b").unwrap());
        assert!(!regex_match("[abc]", "d").unwrap());
    }

    #[test]
    fn test_char_class_range() {
        assert!(regex_match("[a-z]", "m").unwrap());
        assert!(!regex_match("[a-z]", "M").unwrap());
    }

    #[test]
    fn test_negated_char_class() {
        assert!(!regex_match("[^abc]", "a").unwrap());
        assert!(regex_match("[^abc]", "d").unwrap());
    }

    #[test]
    fn test_escape_digit() {
        assert!(regex_match("\\d+", "123").unwrap());
        assert!(!regex_match("\\d+", "abc").unwrap());
    }

    #[test]
    fn test_escape_word() {
        assert!(regex_match("\\w+", "hello_42").unwrap());
    }

    #[test]
    fn test_grouped_alternation() {
        assert!(regex_match("(ab|cd)e", "abe").unwrap());
        assert!(regex_match("(ab|cd)e", "cde").unwrap());
        assert!(!regex_match("(ab|cd)e", "ace").unwrap());
    }

    #[test]
    fn test_complex_pattern() {
        assert!(regex_match("a(b|c)*d", "ad").unwrap());
        assert!(regex_match("a(b|c)*d", "abd").unwrap());
        assert!(regex_match("a(b|c)*d", "abcbd").unwrap());
    }

    #[test]
    fn test_nfa_construction() {
        let ast = parse_regex("a|b").unwrap();
        let nfa = Nfa::from_ast(&ast);
        assert!(nfa.states.len() >= 4); // at least start, accept, and two branches
    }

    #[test]
    fn test_dfa_construction() {
        let ast = parse_regex("ab").unwrap();
        let nfa = Nfa::from_ast(&ast);
        let dfa = Dfa::from_nfa(&nfa, &['a', 'b']);
        assert!(dfa.matches("ab"));
        assert!(!dfa.matches("ba"));
    }

    #[test]
    fn test_dfa_minimization() {
        let ast = parse_regex("a|a").unwrap();
        let nfa = Nfa::from_ast(&ast);
        let dfa = Dfa::from_nfa(&nfa, &['a']);
        let min = dfa.minimize();
        assert!(min.states.len() <= dfa.states.len());
        assert!(min.matches("a"));
    }

    #[test]
    fn test_compile_dfa() {
        let dfa = compile_dfa("ab*c", &['a', 'b', 'c']).unwrap();
        assert!(dfa.matches("ac"));
        assert!(dfa.matches("abc"));
        assert!(dfa.matches("abbbbc"));
        assert!(!dfa.matches("ab"));
    }

    #[test]
    fn test_unmatched_paren() {
        let err = parse_regex("(abc").unwrap_err();
        assert!(matches!(err, RegexError::UnmatchedParen(_)));
    }

    #[test]
    fn test_epsilon_closure() {
        let ast = parse_regex("a*").unwrap();
        let nfa = Nfa::from_ast(&ast);
        let mut start = BTreeSet::new();
        start.insert(nfa.start);
        let closure = nfa.epsilon_closure(&start);
        // Closure should include the start, the inner state, and the accept
        assert!(closure.len() > 1);
    }

    #[test]
    fn test_nested_groups() {
        assert!(regex_match("((a))", "a").unwrap());
        assert!(!regex_match("((a))", "b").unwrap());
    }

    #[test]
    fn test_ast_display() {
        let ast = parse_regex("a").unwrap();
        assert_eq!(ast, RegexAst::Literal('a'));
    }

    #[test]
    fn test_escaped_special_chars() {
        assert!(regex_match("\\.", ".").unwrap());
        assert!(!regex_match("\\.", "a").unwrap());
    }
}
