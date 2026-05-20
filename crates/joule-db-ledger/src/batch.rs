use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A batch of energy receipts with their Merkle commitment.
///
/// The `batch_id` is the hex-encoded Merkle root, making it content-addressable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptBatch {
    /// Unique batch identifier (hex Merkle root).
    pub batch_id: String,
    /// Hex-encoded Merkle root hash.
    pub merkle_root: String,
    /// Number of receipts in this batch.
    pub receipt_count: usize,
    /// Timestamp of the earliest receipt in the batch.
    pub time_start: DateTime<Utc>,
    /// Timestamp of the latest receipt in the batch.
    pub time_end: DateTime<Utc>,
    /// Aggregate kilowatt-hours across all receipts.
    pub aggregate_kwh: f64,
    /// Aggregate kgCO2e across all receipts.
    pub aggregate_kg_co2e: f64,
    /// Issuer node identifier.
    pub issuer: String,
    /// Receipt IDs in order (positional correspondence to Merkle leaves).
    pub receipt_ids: Vec<String>,
    /// Tenant this batch belongs to (only set in tenant-isolated mode).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,
}

/// On-chain (or backend-specific) commitment reference.
///
/// Returned by `LedgerBackend::commit_batch()` after successful attestation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchCommitment {
    /// Batch ID (same as ReceiptBatch.batch_id).
    pub batch_id: String,
    /// Backend-specific transaction reference (tx hash, file offset, sequence number).
    pub tx_ref: String,
    /// Backend name (e.g., "memory", "file", "ethereum", "solana").
    pub backend_name: String,
    /// Timestamp when commitment was recorded.
    pub committed_at: DateTime<Utc>,
    /// Block number (for blockchain backends, None for file/memory).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_number: Option<u64>,
    /// Energy overhead of the commit operation itself (joules).
    /// Reported by the backend. None if the backend doesn't measure this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain_overhead_joules: Option<f64>,
    /// Amortized overhead per receipt (chain_overhead_joules / receipt_count).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub amortized_overhead_joules: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn batch_serde_roundtrip() {
        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        let batch = ReceiptBatch {
            batch_id: "abc123".to_string(),
            merkle_root: "abc123".to_string(),
            receipt_count: 5,
            time_start: ts,
            time_end: ts + chrono::Duration::seconds(60),
            aggregate_kwh: 0.001,
            aggregate_kg_co2e: 0.0004,
            issuer: "node-1".to_string(),
            receipt_ids: vec!["r1".into(), "r2".into()],
            tenant_id: None,
        };
        let json = serde_json::to_string(&batch).unwrap();
        let batch2: ReceiptBatch = serde_json::from_str(&json).unwrap();
        assert_eq!(batch.batch_id, batch2.batch_id);
        assert_eq!(batch.receipt_count, batch2.receipt_count);
    }

    #[test]
    fn commitment_serde_roundtrip() {
        let commitment = BatchCommitment {
            batch_id: "abc123".to_string(),
            tx_ref: "0xdeadbeef".to_string(),
            backend_name: "ethereum".to_string(),
            committed_at: Utc::now(),
            block_number: Some(12345),
            chain_overhead_joules: Some(0.05),
            amortized_overhead_joules: Some(0.005),
        };
        let json = serde_json::to_string(&commitment).unwrap();
        let c2: BatchCommitment = serde_json::from_str(&json).unwrap();
        assert_eq!(commitment.batch_id, c2.batch_id);
        assert_eq!(commitment.block_number, Some(12345));
    }
}
