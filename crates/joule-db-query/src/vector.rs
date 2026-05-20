//! Vector computation support for JouleDB SQL.
//!
//! Provides vector parsing, distance metrics, normalization, and utility functions.
//! Vectors are stored as JSON-style array strings like `'[1.0, 2.0, 3.0]'` in TEXT columns
//! and parsed on-the-fly per function call (same pattern as WKT for spatial).

/// Parse a vector literal string like `'[1.0, 2.0, 3.0]'` into a Vec<f64>.
/// Accepts JSON array format and handles whitespace gracefully.
pub fn parse_vector(text: &str) -> Option<Vec<f64>> {
    let trimmed = text.trim();
    let inner = trimmed.strip_prefix('[')?.strip_suffix(']')?;
    if inner.trim().is_empty() {
        return Some(Vec::new());
    }
    inner
        .split(',')
        .map(|s| s.trim().parse::<f64>().ok())
        .collect()
}

/// Convert a vector back to text format: `[1.0, 2.0, 3.0]`.
pub fn vector_to_string(v: &[f64]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{}", x)).collect();
    format!("[{}]", parts.join(", "))
}

/// Euclidean (L2) distance: sqrt(sum((a_i - b_i)^2)).
/// Returns None if dimensions mismatch.
pub fn l2_distance(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    let sum: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    Some(sum.sqrt())
}

/// Cosine similarity: dot(a,b) / (|a| * |b|).
/// Returns None if dimensions mismatch or either vector is zero-norm.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return None;
    }
    Some(dot / (norm_a * norm_b))
}

/// Cosine distance: 1.0 - cosine_similarity.
pub fn cosine_distance(a: &[f64], b: &[f64]) -> Option<f64> {
    cosine_similarity(a, b).map(|sim| 1.0 - sim)
}

/// Inner (dot) product: sum(a_i * b_i).
/// Returns None if dimensions mismatch.
pub fn inner_product(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    Some(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
}

/// Manhattan (L1) distance: sum(|a_i - b_i|).
/// Returns None if dimensions mismatch.
pub fn manhattan_distance(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    Some(a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum())
}

/// Dispatch to appropriate distance metric by name (case-insensitive).
/// Supported: "euclidean"/"l2", "cosine", "manhattan"/"l1", "inner_product"/"dot".
pub fn vector_distance(a: &[f64], b: &[f64], metric: &str) -> Option<f64> {
    match metric.to_lowercase().as_str() {
        "euclidean" | "l2" => l2_distance(a, b),
        "cosine" => cosine_distance(a, b),
        "manhattan" | "l1" => manhattan_distance(a, b),
        "inner_product" | "dot" => inner_product(a, b),
        _ => None,
    }
}

/// L2 norm: sqrt(sum(v_i^2)).
pub fn vector_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Normalize to unit vector: v / |v|.
/// Returns a copy if zero-norm (avoids division by zero).
pub fn vector_normalize(v: &[f64]) -> Vec<f64> {
    let norm = vector_norm(v);
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// Number of dimensions in a vector.
pub fn vector_dims(v: &[f64]) -> usize {
    v.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_vector_basic() {
        let v = parse_vector("[1.0, 2.0, 3.0]").unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parse_vector_integers() {
        let v = parse_vector("[1, 2, 3]").unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parse_vector_spaces() {
        let v = parse_vector("[ 1.0 , 2.0 , 3.0 ]").unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parse_vector_empty() {
        let v = parse_vector("[]").unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn test_parse_vector_invalid() {
        assert!(parse_vector("not a vector").is_none());
        assert!(parse_vector("[a, b, c]").is_none());
    }

    #[test]
    fn test_vector_to_string_roundtrip() {
        let original = vec![1.5, 2.5, 3.5];
        let text = vector_to_string(&original);
        let parsed = parse_vector(&text).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_l2_distance_3_4() {
        let d = l2_distance(&[0.0, 0.0], &[3.0, 4.0]).unwrap();
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_l2_distance_same() {
        let d = l2_distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).unwrap();
        assert!(d.abs() < 1e-10);
    }

    #[test]
    fn test_l2_distance_dimension_mismatch() {
        assert!(l2_distance(&[1.0, 2.0], &[1.0, 2.0, 3.0]).is_none());
    }

    #[test]
    fn test_cosine_similarity_parallel() {
        let sim = cosine_similarity(&[1.0, 0.0], &[2.0, 0.0]).unwrap();
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let sim = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let sim = cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]).unwrap();
        assert!((sim - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_distance() {
        let d = cosine_distance(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
        assert!((d - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inner_product() {
        let d = inner_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
        assert!((d - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_manhattan_distance() {
        let d = manhattan_distance(&[1.0, 2.0, 3.0], &[4.0, 6.0, 3.0]).unwrap();
        assert!((d - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_distance_dispatch() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert!((vector_distance(&a, &b, "euclidean").unwrap() - 5.0).abs() < 1e-10);
        assert!((vector_distance(&a, &b, "l2").unwrap() - 5.0).abs() < 1e-10);
        assert!((vector_distance(&a, &b, "manhattan").unwrap() - 7.0).abs() < 1e-10);
        assert!(vector_distance(&a, &b, "unknown").is_none());
    }

    #[test]
    fn test_vector_norm() {
        assert!((vector_norm(&[3.0, 4.0]) - 5.0).abs() < 1e-10);
        assert!((vector_norm(&[0.0, 0.0]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_normalize() {
        let n = vector_normalize(&[3.0, 4.0]);
        assert!((n[0] - 0.6).abs() < 1e-10);
        assert!((n[1] - 0.8).abs() < 1e-10);
        assert!((vector_norm(&n) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_normalize_zero() {
        let n = vector_normalize(&[0.0, 0.0]);
        assert_eq!(n, vec![0.0, 0.0]);
    }

    #[test]
    fn test_vector_dims() {
        assert_eq!(vector_dims(&[1.0, 2.0, 3.0]), 3);
        assert_eq!(vector_dims(&[]), 0);
    }
}
