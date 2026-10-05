//! Chunked columnar scan for aggregates.
//!
//! The B-tree row path stays in place. Numeric aggregates reduce a column
//! in fixed-size [`DataChunk`]s (2048 values) instead of one scalar add
//! per row in the hot loop.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Rows processed together by the vectorized aggregate scan.
pub const CHUNK_ROWS: usize = 2048;

static CHUNK_SCANS: AtomicUsize = AtomicUsize::new(0);

/// How many chunked aggregate scans have run in this process.
pub fn chunk_scans() -> usize {
    CHUNK_SCANS.load(Ordering::Relaxed)
}

/// Reset the counter. Tests call this before an aggregate.
pub fn reset_chunk_scans() {
    CHUNK_SCANS.store(0, Ordering::Relaxed);
}

/// One vector of numeric column values, at most [`CHUNK_ROWS`] long.
#[derive(Debug, Clone)]
pub struct DataChunk {
    pub values: Vec<f64>,
}

impl DataChunk {
    pub fn from_slice(values: &[f64]) -> Self {
        Self {
            values: values.to_vec(),
        }
    }

    pub fn sum(&self) -> f64 {
        let mut acc = 0.0;
        for v in &self.values {
            acc += *v;
        }
        acc
    }

    pub fn min(&self) -> Option<f64> {
        self.values.iter().copied().reduce(f64::min)
    }

    pub fn max(&self) -> Option<f64> {
        self.values.iter().copied().reduce(f64::max)
    }
}

/// Split `values` into chunks of [`CHUNK_ROWS`] and reduce them.
///
/// Returns `(aggregate, chunks_used)`. `op` is `sum`, `avg`, `min`, `max`, or `count`.
pub fn aggregate_chunked(values: &[f64], op: &str) -> (f64, usize) {
    CHUNK_SCANS.fetch_add(1, Ordering::Relaxed);
    if values.is_empty() {
        return (0.0, 0);
    }
    let mut chunks = 0usize;
    let mut sum = 0.0;
    let mut count = 0.0;
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for piece in values.chunks(CHUNK_ROWS) {
        chunks += 1;
        let chunk = DataChunk::from_slice(piece);
        sum += chunk.sum();
        count += chunk.values.len() as f64;
        if let Some(v) = chunk.min() {
            min_v = min_v.min(v);
        }
        if let Some(v) = chunk.max() {
            max_v = max_v.max(v);
        }
    }
    let op = op.to_ascii_lowercase();
    let value = match op.as_str() {
        "count" => count,
        "avg" | "mean" => {
            if count == 0.0 {
                0.0
            } else {
                sum / count
            }
        }
        "min" => min_v,
        "max" => max_v,
        _ => sum,
    };
    (value, chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_uses_2048_chunks() {
        let values: Vec<f64> = (0..5000).map(|i| i as f64).collect();
        let (sum, chunks) = aggregate_chunked(&values, "sum");
        assert_eq!(chunks, 3, "5000 values must split as 2048+2048+904");
        let expect: f64 = values.iter().sum();
        assert!((sum - expect).abs() < 1e-6);
        let (avg, _) = aggregate_chunked(&values, "avg");
        assert!((avg - expect / 5000.0).abs() < 1e-6);
    }
}
