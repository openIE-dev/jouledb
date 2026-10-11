//! SQL LIKE / ILIKE matching without regex.
//!
//! The executors used to build and compile a `regex::Regex` for every row a
//! LIKE predicate touched (2.6 s for 10k rows of `url LIKE '%/search%'`). A
//! pattern is parsed once into `%`-separated segments, cached per thread, and
//! matched with substring search (`str::find` uses SIMD two-way search).
//! A segment containing `_` is matched char by char. `\` escapes `%`, `_`,
//! and `\` (the Postgres default ESCAPE).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Lit(char),
    Any1,
}

#[derive(Debug)]
struct Seg {
    toks: Vec<Tok>,
    /// Literal text when the segment has no `_`
    plain: Option<String>,
}

/// A parsed LIKE pattern.
#[derive(Debug)]
pub struct LikePattern {
    segs: Vec<Seg>,
    case_insensitive: bool,
}

impl LikePattern {
    pub fn new(pattern: &str, case_insensitive: bool) -> Self {
        let pat = if case_insensitive { pattern.to_lowercase() } else { pattern.to_string() };
        let mut segs = vec![Vec::new()];
        let mut it = pat.chars();
        while let Some(c) = it.next() {
            match c {
                '\\' => segs.last_mut().unwrap().push(Tok::Lit(it.next().unwrap_or('\\'))),
                '%' => segs.push(Vec::new()),
                '_' => segs.last_mut().unwrap().push(Tok::Any1),
                c => segs.last_mut().unwrap().push(Tok::Lit(c)),
            }
        }
        let segs = segs
            .into_iter()
            .map(|toks| {
                let plain = toks
                    .iter()
                    .map(|t| match t {
                        Tok::Lit(c) => Some(*c),
                        Tok::Any1 => None,
                    })
                    .collect::<Option<String>>();
                Seg { toks, plain }
            })
            .collect();
        Self { segs, case_insensitive }
    }

    /// End byte offset if `seg` matches `text` starting at byte `pos`.
    fn match_at(text: &str, pos: usize, seg: &Seg) -> Option<usize> {
        if let Some(p) = &seg.plain {
            return text[pos..].starts_with(p.as_str()).then(|| pos + p.len());
        }
        let mut chars = text[pos..].char_indices();
        let mut end = pos;
        for t in &seg.toks {
            let (i, c) = chars.next()?;
            if let Tok::Lit(l) = t {
                if *l != c {
                    return None;
                }
            }
            end = pos + i + c.len_utf8();
        }
        Some(end)
    }

    /// Leftmost (start, end) match of `seg` at or after byte `from`.
    fn find(text: &str, from: usize, seg: &Seg) -> Option<usize> {
        if let Some(p) = &seg.plain {
            return text[from..].find(p.as_str()).map(|i| from + i + p.len());
        }
        let mut pos = from;
        loop {
            if let Some(end) = Self::match_at(text, pos, seg) {
                return Some(end);
            }
            let c = text[pos..].chars().next()?;
            pos += c.len_utf8();
        }
    }

    pub fn matches(&self, text: &str) -> bool {
        let lowered;
        let text = if self.case_insensitive {
            lowered = text.to_lowercase();
            lowered.as_str()
        } else {
            text
        };
        let n = self.segs.len();
        if n == 1 {
            return Self::match_at(text, 0, &self.segs[0]) == Some(text.len());
        }
        let Some(mut pos) = Self::match_at(text, 0, &self.segs[0]) else {
            return false;
        };
        for seg in &self.segs[1..n - 1] {
            if seg.toks.is_empty() {
                continue;
            }
            match Self::find(text, pos, seg) {
                Some(end) => pos = end,
                None => return false,
            }
        }
        // the last segment is anchored at the end of the text
        let last = &self.segs[n - 1];
        let k = last.toks.len();
        let mut start = text.len();
        let mut back = text[pos..].char_indices().rev();
        for _ in 0..k {
            match back.next() {
                Some((i, _)) => start = pos + i,
                None => return false,
            }
        }
        Self::match_at(text, start, last) == Some(text.len())
    }
}

const CACHE_CAP: usize = 256;

thread_local! {
    static LIKE_CACHE: RefCell<HashMap<(String, bool), Rc<LikePattern>>> = RefCell::new(HashMap::new());
    static REGEX_CACHE: RefCell<HashMap<String, Option<Rc<regex::Regex>>>> = RefCell::new(HashMap::new());
}

/// `text LIKE pattern` (or ILIKE), with the parsed pattern cached per thread.
pub fn like(text: &str, pattern: &str, case_insensitive: bool) -> bool {
    let p = LIKE_CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if let Some(p) = c.get(&(pattern.to_string(), case_insensitive)) {
            return p.clone();
        }
        if c.len() >= CACHE_CAP {
            c.clear();
        }
        let p = Rc::new(LikePattern::new(pattern, case_insensitive));
        c.insert((pattern.to_string(), case_insensitive), p.clone());
        p
    });
    p.matches(text)
}

/// `regex.is_match(text)` with the compiled regex cached per thread.
/// An invalid pattern matches nothing (the executors' previous behaviour).
pub fn regex_is_match(pattern: &str, text: &str) -> bool {
    let re = REGEX_CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if let Some(r) = c.get(pattern) {
            return r.clone();
        }
        if c.len() >= CACHE_CAP {
            c.clear();
        }
        let r = regex::Regex::new(pattern).ok().map(Rc::new);
        c.insert(pattern.to_string(), r.clone());
        r
    });
    re.is_some_and(|r| r.is_match(text))
}

#[cfg(test)]
mod tests {
    use super::like;

    #[test]
    fn like_semantics() {
        let cases = [
            ("hello", "hello", true),
            ("hello", "hell", false),
            ("hello", "h%", true),
            ("hello", "%o", true),
            ("hello", "%ll%", true),
            ("hello", "%x%", false),
            ("hello", "h_llo", true),
            ("hello", "h_l%", true),
            ("hello", "_____", true),
            ("hello", "______", false),
            ("", "%", true),
            ("", "", true),
            ("a", "", false),
            ("abcabc", "%abc", true),
            ("abcab", "a%b%b", true),
            ("aab", "%a_b", true),
            ("ab", "a%%b", true),
            ("https://x.io/search?q=1", "%/search%", true),
            ("50%", "50\\%", true),
            ("505", "50\\%", false),
            ("a_b", "a\\_b", true),
            ("axb", "a\\_b", false),
            ("héllo 世界", "h_llo %界", true),
            ("世界", "_界", true),
            ("ab", "%b_", false),
            ("a.c", "a.c", true),
            ("abc", "a.c", false),
        ];
        for (t, p, want) in cases {
            assert_eq!(like(t, p, false), want, "{t:?} LIKE {p:?}");
        }
        assert!(like("HeLLo", "%ll%", true));
        assert!(!like("HeLLo", "%ll%", false));
    }
}
