//! Suffix array — efficient string indexing for pattern search, longest
//! repeated substring, and distinct substring counting. Includes LCP array
//! construction via Kasai's algorithm and binary-search-based pattern matching.
//!
//! Replaces JS suffix-array libraries with a pure-Rust O(n log^2 n)
//! construction and O(m log n) pattern search.

use std::fmt;

// ── SuffixArray ────────────────────────────────────────────────────────────

/// A suffix array with LCP (Longest Common Prefix) array for string queries.
pub struct SuffixArray {
    /// The original text as bytes.
    text: Vec<u8>,
    /// Sorted array of suffix starting positions.
    sa: Vec<usize>,
    /// LCP array: lcp[i] = length of longest common prefix between sa[i-1] and sa[i].
    lcp: Vec<usize>,
}

impl fmt::Debug for SuffixArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SuffixArray")
            .field("text_len", &self.text.len())
            .field("suffixes", &self.sa.len())
            .finish()
    }
}

impl SuffixArray {
    /// Build a suffix array from a string using O(n log^2 n) construction.
    pub fn from_str(text: &str) -> Self {
        let bytes = text.as_bytes().to_vec();
        let sa = build_suffix_array(&bytes);
        let lcp = build_lcp_array(&bytes, &sa);
        Self {
            text: bytes,
            sa,
            lcp,
        }
    }

    /// Build a suffix array from raw bytes.
    pub fn from_bytes(text: &[u8]) -> Self {
        let bytes = text.to_vec();
        let sa = build_suffix_array(&bytes);
        let lcp = build_lcp_array(&bytes, &sa);
        Self {
            text: bytes,
            sa,
            lcp,
        }
    }

    /// Length of the original text.
    pub fn text_len(&self) -> usize {
        self.text.len()
    }

    /// The suffix array (sorted suffix positions).
    pub fn suffix_array(&self) -> &[usize] {
        &self.sa
    }

    /// The LCP array.
    pub fn lcp_array(&self) -> &[usize] {
        &self.lcp
    }

    /// The original text as bytes.
    pub fn text(&self) -> &[u8] {
        &self.text
    }

    /// Search for a pattern. Returns all starting positions where the pattern occurs.
    pub fn search(&self, pattern: &str) -> Vec<usize> {
        self.search_bytes(pattern.as_bytes())
    }

    /// Search for a byte pattern.
    pub fn search_bytes(&self, pattern: &[u8]) -> Vec<usize> {
        if pattern.is_empty() || self.sa.is_empty() {
            return Vec::new();
        }

        // Binary search for the leftmost match
        let left = self.lower_bound(pattern);
        let right = self.upper_bound(pattern);

        if left > right {
            return Vec::new();
        }

        let mut positions: Vec<usize> = self.sa[left..=right].to_vec();
        positions.sort_unstable();
        positions
    }

    /// Count occurrences of a pattern.
    pub fn count(&self, pattern: &str) -> usize {
        self.count_bytes(pattern.as_bytes())
    }

    /// Count occurrences of a byte pattern.
    pub fn count_bytes(&self, pattern: &[u8]) -> usize {
        if pattern.is_empty() || self.sa.is_empty() {
            return 0;
        }
        let left = self.lower_bound(pattern);
        let right = self.upper_bound(pattern);
        if left > right {
            0
        } else {
            right - left + 1
        }
    }

    /// Find the longest repeated substring. Returns (start, length).
    /// If no repetition exists, returns (0, 0).
    pub fn longest_repeated_substring(&self) -> (usize, usize) {
        if self.lcp.is_empty() {
            return (0, 0);
        }
        let mut max_lcp = 0;
        let mut max_idx = 0;
        for i in 1..self.lcp.len() {
            if self.lcp[i] > max_lcp {
                max_lcp = self.lcp[i];
                max_idx = i;
            }
        }
        if max_lcp == 0 {
            (0, 0)
        } else {
            (self.sa[max_idx], max_lcp)
        }
    }

    /// The longest repeated substring as a string (assuming UTF-8 text).
    pub fn longest_repeated_substring_str(&self) -> &str {
        let (start, len) = self.longest_repeated_substring();
        if len == 0 {
            return "";
        }
        std::str::from_utf8(&self.text[start..start + len]).unwrap_or("")
    }

    /// Count the number of distinct non-empty substrings.
    /// Uses the formula: n*(n+1)/2 - sum(lcp).
    pub fn distinct_substrings(&self) -> usize {
        let n = self.text.len();
        if n == 0 {
            return 0;
        }
        let total = n * (n + 1) / 2;
        let lcp_sum: usize = self.lcp.iter().sum();
        total - lcp_sum
    }

    /// Get the suffix starting at position `pos` as a string slice.
    pub fn suffix_at(&self, pos: usize) -> &str {
        if pos >= self.text.len() {
            return "";
        }
        std::str::from_utf8(&self.text[pos..]).unwrap_or("")
    }

    // ── Internal binary search helpers ──

    fn lower_bound(&self, pattern: &[u8]) -> usize {
        let n = self.sa.len();
        let mut lo: usize = 0;
        let mut hi = n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let suffix = &self.text[self.sa[mid]..];
            let cmp_len = pattern.len().min(suffix.len());
            if suffix[..cmp_len] < *pattern {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    fn upper_bound(&self, pattern: &[u8]) -> usize {
        let n = self.sa.len();
        let mut lo: usize = 0;
        let mut hi = n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let suffix = &self.text[self.sa[mid]..];
            let cmp_len = pattern.len().min(suffix.len());
            if suffix[..cmp_len] <= *pattern {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // lo is one past the last match
        if lo == 0 { 0 } else { lo - 1 }
    }
}

// ── Construction ───────────────────────────────────────────────────────────

/// Build suffix array using O(n log^2 n) prefix-doubling.
fn build_suffix_array(text: &[u8]) -> Vec<usize> {
    let n = text.len();
    if n == 0 {
        return Vec::new();
    }

    let mut sa: Vec<usize> = (0..n).collect();
    let mut rank: Vec<i64> = text.iter().map(|b| *b as i64).collect();
    let mut new_rank = vec![0i64; n];
    let mut k = 1;

    loop {
        // Sort by (rank[i], rank[i+k])
        let rank_ref = &rank;
        let kk = k;
        sa.sort_by(|a, b| {
            let ra = rank_ref[*a];
            let rb = rank_ref[*b];
            if ra != rb {
                return ra.cmp(&rb);
            }
            let ra2 = if *a + kk < n { rank_ref[*a + kk] } else { -1 };
            let rb2 = if *b + kk < n { rank_ref[*b + kk] } else { -1 };
            ra2.cmp(&rb2)
        });

        // Compute new ranks
        new_rank[sa[0]] = 0;
        for i in 1..n {
            let prev = sa[i - 1];
            let curr = sa[i];
            let same_first = rank[prev] == rank[curr];
            let prev_second = if prev + k < n { rank[prev + k] } else { -1 };
            let curr_second = if curr + k < n { rank[curr + k] } else { -1 };
            let same_second = prev_second == curr_second;
            new_rank[curr] = new_rank[prev] + if same_first && same_second { 0 } else { 1 };
        }

        std::mem::swap(&mut rank, &mut new_rank);

        // If all ranks are unique, we're done
        if rank[sa[n - 1]] as usize == n - 1 {
            break;
        }
        k *= 2;
    }

    sa
}

/// Build LCP array using Kasai's algorithm in O(n).
fn build_lcp_array(text: &[u8], sa: &[usize]) -> Vec<usize> {
    let n = sa.len();
    if n == 0 {
        return Vec::new();
    }

    let mut rank = vec![0usize; n];
    for i in 0..n {
        rank[sa[i]] = i;
    }

    let mut lcp = vec![0usize; n];
    let mut h: usize = 0;

    for i in 0..n {
        if rank[i] > 0 {
            let j = sa[rank[i] - 1];
            while i + h < n && j + h < n && text[i + h] == text[j + h] {
                h += 1;
            }
            lcp[rank[i]] = h;
            if h > 0 {
                h -= 1;
            }
        } else {
            h = 0;
        }
    }

    lcp
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_simple() {
        let sa = SuffixArray::from_str("banana");
        assert_eq!(sa.text_len(), 6);
        assert_eq!(sa.suffix_array().len(), 6);
    }

    #[test]
    fn test_suffix_array_sorted() {
        let sa = SuffixArray::from_str("banana");
        // Suffixes sorted: a, ana, anana, banana, na, nana
        // Positions:       5, 3, 1, 0, 4, 2
        let positions = sa.suffix_array();
        assert_eq!(positions, &[5, 3, 1, 0, 4, 2]);
    }

    #[test]
    fn test_lcp_banana() {
        let sa = SuffixArray::from_str("banana");
        // LCP: [0, 1, 3, 0, 0, 2]
        let lcp = sa.lcp_array();
        assert_eq!(lcp, &[0, 1, 3, 0, 0, 2]);
    }

    #[test]
    fn test_search_found() {
        let sa = SuffixArray::from_str("banana");
        let mut results = sa.search("ana");
        results.sort();
        assert_eq!(results, vec![1, 3]);
    }

    #[test]
    fn test_search_not_found() {
        let sa = SuffixArray::from_str("banana");
        assert!(sa.search("xyz").is_empty());
    }

    #[test]
    fn test_search_single_char() {
        let sa = SuffixArray::from_str("banana");
        let mut results = sa.search("a");
        results.sort();
        assert_eq!(results, vec![1, 3, 5]);
    }

    #[test]
    fn test_count() {
        let sa = SuffixArray::from_str("abcabc");
        assert_eq!(sa.count("abc"), 2);
        assert_eq!(sa.count("bc"), 2);
        assert_eq!(sa.count("xyz"), 0);
    }

    #[test]
    fn test_longest_repeated() {
        let sa = SuffixArray::from_str("banana");
        let (start, len) = sa.longest_repeated_substring();
        let substr = std::str::from_utf8(&sa.text()[start..start + len]).unwrap();
        assert_eq!(substr, "ana");
        assert_eq!(len, 3);
    }

    #[test]
    fn test_longest_repeated_str() {
        let sa = SuffixArray::from_str("abcabcabc");
        assert_eq!(sa.longest_repeated_substring_str(), "abcabc");
    }

    #[test]
    fn test_no_repeated() {
        let sa = SuffixArray::from_str("abcd");
        let (_, len) = sa.longest_repeated_substring();
        // Some characters repeat in pairs but no multi-char repetition beyond that
        // For "abcd", no character repeats, so LRS length depends on actual LCP
        assert!(len <= 1); // No repeated substring of length > 0 expected for unique chars
    }

    #[test]
    fn test_distinct_substrings() {
        let sa = SuffixArray::from_str("abc");
        // Substrings: a, ab, abc, b, bc, c = 6
        assert_eq!(sa.distinct_substrings(), 6);
    }

    #[test]
    fn test_distinct_substrings_repeated() {
        let sa = SuffixArray::from_str("aab");
        // Total possible: 6. LCP sum removes duplicates.
        // Substrings: a, aa, aab, a, ab, b => unique: a, aa, aab, ab, b = 5
        assert_eq!(sa.distinct_substrings(), 5);
    }

    #[test]
    fn test_empty_string() {
        let sa = SuffixArray::from_str("");
        assert_eq!(sa.text_len(), 0);
        assert!(sa.suffix_array().is_empty());
        assert!(sa.search("anything").is_empty());
        assert_eq!(sa.distinct_substrings(), 0);
    }

    #[test]
    fn test_single_char() {
        let sa = SuffixArray::from_str("a");
        assert_eq!(sa.suffix_array(), &[0]);
        assert_eq!(sa.search("a"), vec![0]);
        assert_eq!(sa.distinct_substrings(), 1);
    }

    #[test]
    fn test_all_same_chars() {
        let sa = SuffixArray::from_str("aaaa");
        assert_eq!(sa.count("a"), 4);
        assert_eq!(sa.count("aa"), 3);
        assert_eq!(sa.count("aaa"), 2);
        assert_eq!(sa.count("aaaa"), 1);
    }

    #[test]
    fn test_suffix_at() {
        let sa = SuffixArray::from_str("hello");
        assert_eq!(sa.suffix_at(0), "hello");
        assert_eq!(sa.suffix_at(2), "llo");
        assert_eq!(sa.suffix_at(4), "o");
    }

    #[test]
    fn test_from_bytes() {
        let data = b"test";
        let sa = SuffixArray::from_bytes(data);
        assert_eq!(sa.text_len(), 4);
        assert_eq!(sa.count_bytes(b"t"), 2);
    }

    #[test]
    fn test_longer_text() {
        let text = "the quick brown fox jumps over the lazy dog";
        let sa = SuffixArray::from_str(text);
        let results = sa.search("the");
        assert_eq!(results.len(), 2);
    }
}
