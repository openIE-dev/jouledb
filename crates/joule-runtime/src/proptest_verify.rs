//! Property-based tests for joule-runtime security invariants.
//!
//! Verifies cryptographic attestation and energy enforcer correctness:
//! - Attestation key transport roundtrip
//! - Sign→verify roundtrip for arbitrary energy values
//! - Tampered receipt detection (energy, carbon, sequence, nonce)
//! - Replay detection (sequence ordering)
//! - Enforcer state arithmetic (utilization, remaining, saturation)

use proptest::prelude::*;

use crate::attestation::{AttestationKey, ReceiptSigner, ReceiptVerifier};
use crate::energy_enforcer::{EnergyEnforcer, EnergyEnforcerConfig};

// ═══════════════════════════════════════════════════════════════════════════
// Attestation Key Properties
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    /// K1: Attestation key transport roundtrip preserves all fields.
    #[test]
    fn prop_key_transport_roundtrip(instance_id in "[a-z0-9-]{3,20}") {
        let key = AttestationKey::generate(&instance_id);
        let bytes = key.to_bytes();
        let restored = AttestationKey::from_transport_bytes(&bytes).unwrap();
        prop_assert!(restored.instance_id() == key.instance_id());
        prop_assert!(restored.session_nonce_hex() == key.session_nonce_hex());
    }

    /// K2: Short key material is always rejected.
    #[test]
    fn prop_short_key_rejected(data in prop::collection::vec(any::<u8>(), 0..52)) {
        prop_assert!(AttestationKey::from_transport_bytes(&data).is_err());
    }

    /// K3: Truncated instance ID is rejected.
    #[test]
    fn prop_truncated_id_rejected(instance_id in "[a-z]{3,10}") {
        let key = AttestationKey::generate(&instance_id);
        let bytes = key.to_bytes();
        // Truncate by 1 byte — should fail
        if bytes.len() > 52 {
            prop_assert!(AttestationKey::from_transport_bytes(&bytes[..bytes.len() - 1]).is_err());
        }
    }

    /// K4: Session nonce is always 32 hex chars (16 bytes).
    #[test]
    fn prop_nonce_length(instance_id in "[a-z]{3,10}") {
        let key = AttestationKey::generate(&instance_id);
        prop_assert!(key.session_nonce_hex().len() == 32);
        prop_assert!(key.session_nonce_hex().chars().all(|c| c.is_ascii_hexdigit()));
    }

    /// K5: Different key generations produce different nonces.
    #[test]
    fn prop_nonce_uniqueness(_ in 0..50u32) {
        let k1 = AttestationKey::generate("test");
        let k2 = AttestationKey::generate("test");
        // Same instance_id, different sessions → different nonces
        prop_assert!(k1.session_nonce_hex() != k2.session_nonce_hex());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Receipt Signing Properties
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    /// S1: Sign→verify roundtrip succeeds for arbitrary energy values.
    #[test]
    fn prop_sign_verify_roundtrip(
        energy_uj in any::<u64>(),
        carbon_ugco2eq in any::<u64>()
    ) {
        let key = AttestationKey::generate("rt-test");
        let mut signer = ReceiptSigner::new(key.clone());
        let mut verifier = ReceiptVerifier::new(key);

        let receipt = signer.sign(energy_uj, carbon_ugco2eq);
        prop_assert!(verifier.verify(&receipt).is_ok());
    }

    /// S2: Signed receipt preserves energy and carbon values.
    #[test]
    fn prop_receipt_preserves_values(
        energy_uj in any::<u64>(),
        carbon_ugco2eq in any::<u64>()
    ) {
        let key = AttestationKey::generate("preserve-test");
        let mut signer = ReceiptSigner::new(key);
        let receipt = signer.sign(energy_uj, carbon_ugco2eq);
        prop_assert!(receipt.energy_uj == energy_uj);
        prop_assert!(receipt.carbon_ugco2eq == carbon_ugco2eq);
    }

    /// S3: Sequence numbers are monotonically increasing.
    #[test]
    fn prop_sequence_monotonic(count in 1usize..20) {
        let key = AttestationKey::generate("seq-test");
        let mut signer = ReceiptSigner::new(key);
        let mut prev_seq = 0u64;
        for _ in 0..count {
            let receipt = signer.sign(1000, 100);
            prop_assert!(receipt.sequence > prev_seq);
            prev_seq = receipt.sequence;
        }
    }

    /// S4: Tampered energy value fails verification.
    #[test]
    fn prop_tampered_energy_detected(
        energy_uj in any::<u64>(),
        delta in 1u64..1000
    ) {
        let key = AttestationKey::generate("tamper-e");
        let mut signer = ReceiptSigner::new(key.clone());
        let mut receipt = signer.sign(energy_uj, 100);
        receipt.energy_uj = energy_uj.wrapping_add(delta);
        if receipt.energy_uj != energy_uj {
            prop_assert!(receipt.verify(&key).is_err());
        }
    }

    /// S5: Tampered carbon value fails verification.
    #[test]
    fn prop_tampered_carbon_detected(
        carbon in any::<u64>(),
        delta in 1u64..1000
    ) {
        let key = AttestationKey::generate("tamper-c");
        let mut signer = ReceiptSigner::new(key.clone());
        let mut receipt = signer.sign(1000, carbon);
        receipt.carbon_ugco2eq = carbon.wrapping_add(delta);
        if receipt.carbon_ugco2eq != carbon {
            prop_assert!(receipt.verify(&key).is_err());
        }
    }

    /// S6: Tampered sequence number fails verification.
    #[test]
    fn prop_tampered_sequence_detected(_ in 0..50u32) {
        let key = AttestationKey::generate("tamper-s");
        let mut signer = ReceiptSigner::new(key.clone());
        let mut receipt = signer.sign(1000, 100);
        receipt.sequence += 1;
        prop_assert!(receipt.verify(&key).is_err());
    }

    /// S7: Wrong key fails verification.
    #[test]
    fn prop_wrong_key_fails(_ in 0..50u32) {
        let key1 = AttestationKey::generate("key-1");
        let key2 = AttestationKey::generate("key-1");
        let mut signer = ReceiptSigner::new(key1);
        let receipt = signer.sign(1000, 100);
        prop_assert!(receipt.verify(&key2).is_err());
    }

    /// S8: Different instance ID fails verification.
    #[test]
    fn prop_instance_mismatch_fails(_ in 0..50u32) {
        let key = AttestationKey::generate("instance-a");
        let wrong_key = AttestationKey::from_bytes(
            [0u8; 32],
            "instance-b".to_string(),
            [0u8; 16],
        );
        let mut signer = ReceiptSigner::new(key);
        let receipt = signer.sign(1000, 100);
        prop_assert!(receipt.verify(&wrong_key).is_err());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Replay Detection Properties
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    /// R1: Replaying a receipt is always detected.
    #[test]
    fn prop_replay_detected(count in 2usize..10) {
        let key = AttestationKey::generate("replay-test");
        let mut signer = ReceiptSigner::new(key.clone());
        let mut verifier = ReceiptVerifier::new(key);

        let mut receipts = Vec::new();
        for _ in 0..count {
            let r = signer.sign(1000, 100);
            verifier.verify(&r).unwrap();
            receipts.push(r);
        }

        // Replay any previous receipt — must fail
        for r in &receipts {
            prop_assert!(verifier.verify(r).is_err());
        }
    }

    /// R2: Out-of-order delivery is rejected.
    #[test]
    fn prop_out_of_order_rejected(_ in 0..50u32) {
        let key = AttestationKey::generate("order-test");
        let mut signer = ReceiptSigner::new(key.clone());
        let mut verifier = ReceiptVerifier::new(key);

        let r1 = signer.sign(1000, 100);
        let r2 = signer.sign(2000, 200);

        // Accept r2 first
        verifier.verify(&r2).unwrap();
        // r1 has lower sequence → reject
        prop_assert!(verifier.verify(&r1).is_err());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Energy Enforcer State Properties
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    /// E1: remaining_uj is always <= budget_uj (saturating subtraction).
    #[test]
    fn prop_remaining_bounded(budget in 1u64..10_000_000_000) {
        let config = EnergyEnforcerConfig { max_energy_uj: budget, ..Default::default() };
        let enforcer = EnergyEnforcer::new(config);
        let state = enforcer.state();
        // Fresh state: remaining == budget
        prop_assert!(state.remaining_uj() <= budget);
        prop_assert!(state.remaining_uj() == budget);
    }

    /// E2: utilization is non-negative.
    #[test]
    fn prop_utilization_nonneg(budget in any::<u64>()) {
        let config = EnergyEnforcerConfig { max_energy_uj: budget, ..Default::default() };
        let enforcer = EnergyEnforcer::new(config);
        let state = enforcer.state();
        prop_assert!(state.utilization() >= 0.0);
    }

    /// E3: Zero budget → zero utilization (no division by zero).
    #[test]
    fn prop_zero_budget_utilization(_ in 0..50u32) {
        let config = EnergyEnforcerConfig { max_energy_uj: 0, ..Default::default() };
        let enforcer = EnergyEnforcer::new(config);
        let state = enforcer.state();
        prop_assert!(state.utilization() == 0.0);
    }

    /// E4: Update budget changes the budget atomically.
    #[test]
    fn prop_update_budget(initial in any::<u64>(), updated in any::<u64>()) {
        let config = EnergyEnforcerConfig { max_energy_uj: initial, ..Default::default() };
        let enforcer = EnergyEnforcer::new(config);
        let state = enforcer.state();
        prop_assert!(state.budget_uj() == initial);
        enforcer.update_budget(updated);
        prop_assert!(state.budget_uj() == updated);
    }

    /// E5: budget_joules == budget_uj / 1_000_000.
    #[test]
    fn prop_joules_conversion(budget in 0u64..1_000_000_000_000) {
        let config = EnergyEnforcerConfig { max_energy_uj: budget, ..Default::default() };
        let enforcer = EnergyEnforcer::new(config);
        let state = enforcer.state();
        let expected = budget as f64 / 1_000_000.0;
        prop_assert!((state.budget_joules() - expected).abs() < 1e-6);
    }

    /// E6: Fresh enforcer is not running, not exceeded.
    #[test]
    fn prop_fresh_state(budget in any::<u64>()) {
        let config = EnergyEnforcerConfig { max_energy_uj: budget, ..Default::default() };
        let enforcer = EnergyEnforcer::new(config);
        let state = enforcer.state();
        prop_assert!(!state.is_running());
        prop_assert!(!state.is_exceeded());
        prop_assert!(state.consumed_uj() == 0);
    }
}
