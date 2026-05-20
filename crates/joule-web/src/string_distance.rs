//! String distance and similarity algorithms.
//!
//! Pure Rust implementations of Levenshtein, Damerau-Levenshtein, Hamming,
//! Jaro, Jaro-Winkler, longest common subsequence, longest common substring,
//! and normalized distance (0..1).

use std::cmp;

// ── Levenshtein ──────────────────────────────────────────────────

/// Compute the Levenshtein edit distance between two strings.
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 { return n; }
    if n == 0 { return m; }

    let mut prev = (0..=n).collect::<Vec<usize>>();
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
            curr[j] = cmp::min(
                cmp::min(prev[j] + 1, curr[j - 1] + 1),
                prev[j - 1] + cost,
            );
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Normalized Levenshtein distance in range [0.0, 1.0].
/// 0.0 = identical, 1.0 = completely different.
pub fn levenshtein_normalized(a: &str, b: &str) -> f64 {
    let max_len = cmp::max(a.chars().count(), b.chars().count());
    if max_len == 0 { return 0.0; }
    levenshtein(a, b) as f64 / max_len as f64
}

// ── Damerau-Levenshtein ──────────────────────────────────────────

/// Compute the Damerau-Levenshtein distance (includes transpositions).
pub fn damerau_levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 { return n; }
    if n == 0 { return m; }

    let mut matrix = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m { matrix[i][0] = i; }
    for j in 0..=n { matrix[0][j] = j; }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
            matrix[i][j] = cmp::min(
                cmp::min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1),
                matrix[i - 1][j - 1] + cost,
            );
            if i > 1 && j > 1 && a_chars[i - 1] == b_chars[j - 2] && a_chars[i - 2] == b_chars[j - 1] {
                matrix[i][j] = cmp::min(matrix[i][j], matrix[i - 2][j - 2] + cost);
            }
        }
    }
    matrix[m][n]
}

// ── Hamming ──────────────────────────────────────────────────────

/// Compute Hamming distance. Returns None if lengths differ.
pub fn hamming(a: &str, b: &str) -> Option<usize> {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    if a_chars.len() != b_chars.len() { return None; }
    Some(a_chars.iter().zip(b_chars.iter()).filter(|(x, y)| x != y).count())
}

// ── Jaro ─────────────────────────────────────────────────────────

/// Compute Jaro similarity in range [0.0, 1.0]. 1.0 = identical.
pub fn jaro(a: &str, b: &str) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();

    if a_len == 0 && b_len == 0 { return 1.0; }
    if a_len == 0 || b_len == 0 { return 0.0; }

    let match_distance = cmp::max(a_len, b_len) / 2;
    let match_distance = if match_distance > 0 { match_distance - 1 } else { 0 };

    let mut a_matched = vec![false; a_len];
    let mut b_matched = vec![false; b_len];
    let mut matches = 0usize;
    let mut transpositions = 0usize;

    for i in 0..a_len {
        let start = if i > match_distance { i - match_distance } else { 0 };
        let end = cmp::min(i + match_distance + 1, b_len);
        for j in start..end {
            if b_matched[j] || a_chars[i] != b_chars[j] { continue; }
            a_matched[i] = true;
            b_matched[j] = true;
            matches += 1;
            break;
        }
    }

    if matches == 0 { return 0.0; }

    let mut k = 0;
    for i in 0..a_len {
        if !a_matched[i] { continue; }
        while !b_matched[k] { k += 1; }
        if a_chars[i] != b_chars[k] { transpositions += 1; }
        k += 1;
    }

    let m = matches as f64;
    (m / a_len as f64 + m / b_len as f64 + (m - transpositions as f64 / 2.0) / m) / 3.0
}

// ── Jaro-Winkler ─────────────────────────────────────────────────

/// Jaro-Winkler similarity. Gives bonus for common prefix (up to 4 chars).
pub fn jaro_winkler(a: &str, b: &str) -> f64 {
    let jaro_sim = jaro(a, b);
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let prefix_len = a_chars.iter().zip(b_chars.iter())
        .take(4)
        .take_while(|(x, y)| x == y)
        .count();
    let p = 0.1;
    jaro_sim + prefix_len as f64 * p * (1.0 - jaro_sim)
}

// ── Longest Common Subsequence ───────────────────────────────────

/// Length of the longest common subsequence.
pub fn lcs_length(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        for j in 1..=n {
            if a_chars[i - 1] == b_chars[j - 1] {
                curr[j] = prev[j - 1] + 1;
            } else {
                curr[j] = cmp::max(prev[j], curr[j - 1]);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }
    prev[n]
}

/// Return the actual longest common subsequence string.
pub fn lcs(a: &str, b: &str) -> String {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            if a_chars[i - 1] == b_chars[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = cmp::max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }

    let mut result = vec![];
    let mut i = m;
    let mut j = n;
    while i > 0 && j > 0 {
        if a_chars[i - 1] == b_chars[j - 1] {
            result.push(a_chars[i - 1]);
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] > dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    result.reverse();
    result.into_iter().collect()
}

// ── Longest Common Substring ─────────────────────────────────────

/// Return the longest common substring.
pub fn longest_common_substring(a: &str, b: &str) -> String {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];
    let mut max_len = 0;
    let mut end_idx = 0; // end index in a_chars

    for i in 1..=m {
        for j in 1..=n {
            if a_chars[i - 1] == b_chars[j - 1] {
                curr[j] = prev[j - 1] + 1;
                if curr[j] > max_len {
                    max_len = curr[j];
                    end_idx = i;
                }
            } else {
                curr[j] = 0;
            }
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }

    if max_len == 0 { return String::new(); }
    a_chars[end_idx - max_len..end_idx].iter().collect()
}

/// Normalized string distance: 1.0 - (LCS length / max length).
pub fn normalized_distance(a: &str, b: &str) -> f64 {
    let max_len = cmp::max(a.chars().count(), b.chars().count());
    if max_len == 0 { return 0.0; }
    1.0 - lcs_length(a, b) as f64 / max_len as f64
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein("hello", "hello"), 0);
    }

    #[test]
    fn test_levenshtein_basic() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", ""), 3);
    }

    #[test]
    fn test_levenshtein_normalized() {
        assert!((levenshtein_normalized("hello", "hello") - 0.0).abs() < 1e-10);
        assert!((levenshtein_normalized("", "") - 0.0).abs() < 1e-10);
        assert!(levenshtein_normalized("abc", "xyz") > 0.0);
    }

    #[test]
    fn test_damerau_levenshtein() {
        // Transposition: "ab" -> "ba" is distance 1 with DL, but 2 with plain Levenshtein
        assert_eq!(damerau_levenshtein("ab", "ba"), 1);
        assert_eq!(damerau_levenshtein("abc", "abc"), 0);
        assert_eq!(damerau_levenshtein("ca", "abc"), 3);
    }

    #[test]
    fn test_hamming() {
        assert_eq!(hamming("karolin", "kathrin"), Some(3));
        assert_eq!(hamming("abc", "abc"), Some(0));
        assert_eq!(hamming("ab", "abc"), None);
    }

    #[test]
    fn test_jaro() {
        let sim = jaro("martha", "marhta");
        assert!(sim > 0.94 && sim < 0.95, "jaro = {}", sim);
        assert!((jaro("", "") - 1.0).abs() < 1e-10);
        assert!((jaro("abc", "") - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaro_winkler() {
        let sim = jaro_winkler("martha", "marhta");
        assert!(sim > 0.96, "jaro_winkler = {}", sim);
        // Common prefix bonus should make it higher than plain Jaro
        assert!(jaro_winkler("martha", "marhta") >= jaro("martha", "marhta"));
    }

    #[test]
    fn test_lcs_length() {
        assert_eq!(lcs_length("ABCBDAB", "BDCABA"), 4);
        assert_eq!(lcs_length("", "abc"), 0);
    }

    #[test]
    fn test_lcs_string() {
        let result = lcs("ABCBDAB", "BDCABA");
        assert_eq!(result.len(), 4);
        // One valid LCS is "BCBA"
        assert!(result == "BDAB" || result == "BCBA" || result == "BCAB",
                "got lcs = {}", result);
    }

    #[test]
    fn test_longest_common_substring() {
        assert_eq!(longest_common_substring("abcdef", "zbcdf"), "bcd");
        assert_eq!(longest_common_substring("abc", "xyz"), "");
    }

    #[test]
    fn test_normalized_distance() {
        assert!((normalized_distance("abc", "abc") - 0.0).abs() < 1e-10);
        assert!(normalized_distance("abc", "xyz") > 0.0);
        assert!((normalized_distance("", "") - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_levenshtein_single_char() {
        assert_eq!(levenshtein("a", "b"), 1);
        assert_eq!(levenshtein("a", "a"), 0);
        assert_eq!(levenshtein("a", ""), 1);
    }

    #[test]
    fn test_hamming_binary() {
        assert_eq!(hamming("1011101", "1001001"), Some(2));
    }
}
