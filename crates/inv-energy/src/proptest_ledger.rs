//! Property-based tests for the energy ledger and receipt system.
//!
//! Verifies accounting invariants that must hold for ALL inputs:
//! - Total energy = sum of all layers
//! - Unit conversions are consistent
//! - Carbon accounting is bounded and consistent
//! - Receipts have valid audit hashes

use proptest::prelude::*;

use crate::ledger::{EnergyLedger, OperationalLayer};
use crate::receipt::{EnergyReceipt, MemoryTier, SiliconType, MeasurementSource};

// ═══════════════════════════════════════════════════════════════════════════
// Strategies
// ═══════════════════════════════════════════════════════════════════════════

fn arb_layer() -> impl Strategy<Value = OperationalLayer> {
    prop::sample::select(OperationalLayer::ALL)
}

// ═══════════════════════════════════════════════════════════════════════════
// Ledger Properties
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    /// L1: After recording N values, total_uj == sum of all recorded values.
    #[test]
    fn prop_total_equals_sum(
        recordings in prop::collection::vec(
            (arb_layer(), 0u64..1_000_000),
            1..20
        )
    ) {
        let ledger = EnergyLedger::new();
        let mut expected_total = 0u64;

        for &(layer, uj) in &recordings {
            ledger.record(layer, uj);
            expected_total += uj;
        }

        prop_assert!(
            ledger.total_uj() == expected_total,
            "total mismatch: {} vs {}", ledger.total_uj(), expected_total
        );
    }

    /// L2: total_ops == count of record() calls.
    #[test]
    fn prop_ops_count(
        recordings in prop::collection::vec(
            (arb_layer(), 0u64..1_000_000),
            1..20
        )
    ) {
        let ledger = EnergyLedger::new();
        for &(layer, uj) in &recordings {
            ledger.record(layer, uj);
        }
        prop_assert!(
            ledger.total_ops() == recordings.len() as u64,
            "ops mismatch: {} vs {}", ledger.total_ops(), recordings.len()
        );
    }

    /// L3: total_uwh == total_uj / 3600 (integer division).
    #[test]
    fn prop_uwh_conversion(
        recordings in prop::collection::vec(
            (arb_layer(), 0u64..1_000_000),
            1..20
        )
    ) {
        let ledger = EnergyLedger::new();
        for &(layer, uj) in &recordings {
            ledger.record(layer, uj);
        }
        prop_assert!(
            ledger.total_uwh() == ledger.total_uj() / 3600,
            "uwh conversion wrong"
        );
    }

    /// L4: total_joules == total_uj / 1_000_000.0.
    #[test]
    fn prop_joules_conversion(
        recordings in prop::collection::vec(
            (arb_layer(), 0u64..1_000_000),
            1..10
        )
    ) {
        let ledger = EnergyLedger::new();
        for &(layer, uj) in &recordings {
            ledger.record(layer, uj);
        }
        let expected = ledger.total_uj() as f64 / 1_000_000.0;
        let actual = ledger.total_joules();
        prop_assert!(
            (actual - expected).abs() < 1e-10,
            "joules conversion off: {} vs {}", actual, expected
        );
    }

    /// L5: Per-layer sum equals total.
    #[test]
    fn prop_layer_sum_equals_total(
        recordings in prop::collection::vec(
            (arb_layer(), 0u64..1_000_000),
            1..30
        )
    ) {
        let ledger = EnergyLedger::new();
        for &(layer, uj) in &recordings {
            ledger.record(layer, uj);
        }
        let layer_sum: u64 = OperationalLayer::ALL.iter()
            .map(|l| ledger.layer_uj(*l))
            .sum();
        prop_assert!(
            layer_sum == ledger.total_uj(),
            "layer sum {} != total {}", layer_sum, ledger.total_uj()
        );
    }

    /// L6: Carbon emissions are non-negative when intensity is non-negative.
    #[test]
    fn prop_carbon_nonnegative(
        uj in 0u64..10_000_000,
        intensity in 0.0f64..1000.0
    ) {
        let ledger = EnergyLedger::new();
        ledger.record(OperationalLayer::WasmExecution, uj);
        ledger.set_carbon_intensity(intensity);
        prop_assert!(
            ledger.total_carbon_gco2e() >= 0.0,
            "carbon should be non-negative"
        );
    }

    /// L7: Zero energy → zero carbon.
    #[test]
    fn prop_zero_energy_zero_carbon(intensity in 0.0f64..1000.0) {
        let ledger = EnergyLedger::new();
        ledger.set_carbon_intensity(intensity);
        prop_assert!(
            ledger.total_carbon_gco2e() == 0.0,
            "zero energy should mean zero carbon"
        );
    }

    /// L8: record_batch equivalent to N individual records.
    #[test]
    fn prop_batch_equivalence(
        layer in arb_layer(),
        uj in 0u64..1_000_000,
        count in 1u64..10
    ) {
        let single_ledger = EnergyLedger::new();
        for _ in 0..count {
            single_ledger.record(layer, uj);
        }

        let batch_ledger = EnergyLedger::new();
        batch_ledger.record_batch(layer, uj * count, count);

        prop_assert!(
            single_ledger.total_uj() == batch_ledger.total_uj(),
            "batch energy mismatch"
        );
        prop_assert!(
            single_ledger.total_ops() == batch_ledger.total_ops(),
            "batch ops mismatch"
        );
    }

    /// L9: reset() zeroes all counters.
    #[test]
    fn prop_reset_zeroes(
        recordings in prop::collection::vec(
            (arb_layer(), 1u64..1_000_000),
            1..10
        )
    ) {
        let ledger = EnergyLedger::new();
        for &(layer, uj) in &recordings {
            ledger.record(layer, uj);
        }
        ledger.reset();
        prop_assert!(ledger.total_uj() == 0, "total not zero after reset");
        prop_assert!(ledger.total_ops() == 0, "ops not zero after reset");
        for layer in OperationalLayer::ALL {
            prop_assert!(ledger.layer_uj(*layer) == 0, "layer not zero after reset");
        }
    }

    /// L10: snapshot total matches live total.
    #[test]
    fn prop_snapshot_consistent(
        recordings in prop::collection::vec(
            (arb_layer(), 0u64..1_000_000),
            1..10
        )
    ) {
        let ledger = EnergyLedger::new();
        for &(layer, uj) in &recordings {
            ledger.record(layer, uj);
        }
        let snap = ledger.snapshot();
        prop_assert!(
            snap.total_uj == ledger.total_uj(),
            "snapshot total mismatch"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Receipt Properties
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    /// R1: Receipt audit hash verifies after creation.
    #[test]
    fn prop_receipt_verifies(energy in 0.0f64..1e9) {
        let receipt = EnergyReceipt::new(
            energy,
            energy * 0.233,
            SiliconType::Cpu,
            MemoryTier::Dram,
            MeasurementSource::TdpModel,
            "node-1".to_string(),
            "us-east-1".to_string(),
        );
        prop_assert!(receipt.verify(), "fresh receipt should verify");
    }

    /// R2: Merged receipts have combined energy.
    #[test]
    fn prop_receipt_merge_energy(e1 in 0.0f64..1e6, e2 in 0.0f64..1e6) {
        let r1 = EnergyReceipt::estimate(e1, "n1", "r1");
        let r2 = EnergyReceipt::estimate(e2, "n1", "r1");
        let merged = r1.merge(&r2);
        let expected = e1 + e2;
        prop_assert!(
            (merged.energy_joules - expected).abs() < 1e-6,
            "merge energy: {} vs {}", merged.energy_joules, expected
        );
    }

    /// R3: Zero receipt has zero energy.
    #[test]
    fn prop_zero_receipt(_ in Just(())) {
        let r = EnergyReceipt::zero("n1", "r1");
        prop_assert!(r.energy_joules == 0.0);
        prop_assert!(r.verify());
    }
}
