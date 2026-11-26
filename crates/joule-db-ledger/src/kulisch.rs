//! Horizon `H33`/`H34` (`Ring R8` Port) — Zero-Panic Stack-Resident `4,352`-Bit Kulisch Exact
//! `binary64` Superaccumulator for `JouleDB` (`joule-db-ledger`).
//!
//! Eliminates two subtle floating-point hazards in thermodynamic & carbon receipt accounting:
//! 1. **Non-associative parallel / `HashMap` reassociation drift:** `HashMap::values().sum()` and
//!    shuffled arrival orders in distributed batch committers produce `1..N ULP` differences under
//!    naive IEEE-754 `f64` addition.
//! 2. **Multi-scale catastrophic cancellation:** Summing room-temperature Landauer bit-erasure
//!    bounds (`2.8709267e-21 J`) alongside megajoule/planetary cluster totals (`10^6..10^120 J`) or
//!    credit adjustments can wipe low-magnitude terms — or even hallucinate negative joules (`-1.0 J`)!
//!
//! Because every finite IEEE-754 `f64` is an exact dyadic rational $m \cdot 2^e$, [`ExactF64Superaccumulator`]
//! accumulates arbitrary finite `f64` sums and `f64 * f64` products into `68` two's-complement `u64`
//! stack words (`4,352` bits = `544` bytes, zero heap allocation) with **zero intermediate rounding**,
//! rounding **once** (`0.5 ULP` Round-to-Nearest-Even) at readout.

use crate::receipt::LedgerEnergyReceipt;
use serde::{Deserialize, Serialize};

/// Number of `64`-bit two's-complement limbs in [`ExactF64Superaccumulator`].
/// Bit `0` of `words[0]` represents $2^{-2148}$ (`(f64::MIN_POSITIVE_SUBNORMAL)^2`),
/// spanning `[-2148 .. +2203]` (`4,352` bits = `544` bytes) so every finite `f64` value and
/// `f64 * f64` product accumulates with zero rounding and zero overflow.
pub const KULISCH_F64_WORDS: usize = 68;
const KULISCH_MIN_EXP: i32 = -2148;

/// Stack-resident (`544`-byte, zero-heap-allocation) exact Kulisch `binary64` superaccumulator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExactF64Superaccumulator {
    words: [u64; KULISCH_F64_WORDS],
    nan_seen: bool,
    pos_inf_seen: bool,
    neg_inf_seen: bool,
}

impl Default for ExactF64Superaccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl ExactF64Superaccumulator {
    /// Creates a zeroed `4,352`-bit superaccumulator on the stack (`0` heap allocations).
    #[inline]
    pub const fn new() -> Self {
        Self {
            words: [0u64; KULISCH_F64_WORDS],
            nan_seen: false,
            pos_inf_seen: false,
            neg_inf_seen: false,
        }
    }

    #[inline]
    fn is_negative(&self) -> bool {
        (self.words[KULISCH_F64_WORDS - 1] >> 63) != 0
    }

    #[inline]
    fn accumulate_mag_u128(&mut self, neg: bool, mag: u128, bit_pos: usize) {
        if mag == 0 {
            return;
        }
        let word_idx = bit_pos >> 6;
        let bit_off = (bit_pos & 63) as u32;
        if word_idx >= KULISCH_F64_WORDS {
            return;
        }

        let lo_u64 = mag as u64;
        let hi_u64 = (mag >> 64) as u64;
        let w0 = lo_u64 << bit_off;
        let w1 = if bit_off == 0 {
            hi_u64
        } else {
            (lo_u64 >> (64 - bit_off)) | (hi_u64 << bit_off)
        };
        let w2 = if bit_off == 0 {
            0u64
        } else {
            hi_u64 >> (64 - bit_off)
        };
        let limbs = [w0, w1, w2];

        if !neg {
            let mut carry = 0u64;
            for (k, &add_v) in limbs.iter().enumerate() {
                let idx = word_idx + k;
                let Some(slot) = self.words.get_mut(idx) else {
                    break;
                };
                let (s1, c1) = (*slot).overflowing_add(add_v);
                let (s2, c2) = s1.overflowing_add(carry);
                *slot = s2;
                carry = u64::from(c1) + u64::from(c2);
            }
            let mut idx = word_idx + limbs.len();
            while carry != 0 {
                let Some(slot) = self.words.get_mut(idx) else {
                    break;
                };
                let (s, c) = (*slot).overflowing_add(carry);
                *slot = s;
                carry = u64::from(c);
                idx += 1;
            }
        } else {
            let mut borrow = 0u64;
            for (k, &sub_v) in limbs.iter().enumerate() {
                let idx = word_idx + k;
                let Some(slot) = self.words.get_mut(idx) else {
                    break;
                };
                let (d1, b1) = (*slot).overflowing_sub(sub_v);
                let (d2, b2) = d1.overflowing_sub(borrow);
                *slot = d2;
                borrow = u64::from(b1) + u64::from(b2);
            }
            let mut idx = word_idx + limbs.len();
            while borrow != 0 {
                let Some(slot) = self.words.get_mut(idx) else {
                    break;
                };
                let (d, b) = (*slot).overflowing_sub(borrow);
                *slot = d;
                borrow = u64::from(b);
                idx += 1;
            }
        }
    }

    #[inline]
    fn decode_finite_f64(x: f64) -> Option<(bool, u64, i32)> {
        let bits = x.to_bits();
        let neg = (bits >> 63) != 0;
        let exp_raw = ((bits >> 52) & 0x7FF) as i32;
        let frac = bits & 0x000F_FFFF_FFFF_FFFF;
        if exp_raw == 0x7FF {
            return None;
        }
        if exp_raw == 0 {
            if frac == 0 {
                Some((neg, 0, 0))
            } else {
                Some((neg, frac, -1074))
            }
        } else {
            Some((neg, (1u64 << 52) | frac, exp_raw - 1075))
        }
    }

    /// Accumulates an IEEE-754 `f64` with **zero intermediate rounding error**.
    #[inline]
    pub fn add_f64(&mut self, x: f64) {
        if x.is_nan() {
            self.nan_seen = true;
            return;
        }
        if x.is_infinite() {
            if x.is_sign_positive() {
                self.pos_inf_seen = true;
            } else {
                self.neg_inf_seen = true;
            }
            return;
        }
        if let Some((neg, sig, exp2)) = Self::decode_finite_f64(x)
            && sig != 0
        {
            let bit_pos = (exp2 - KULISCH_MIN_EXP) as usize;
            self.accumulate_mag_u128(neg, u128::from(sig), bit_pos);
        }
    }

    /// Accumulates the exact `106`-bit product `a * b` with **zero intermediate rounding**.
    #[inline]
    pub fn add_product_f64(&mut self, a: f64, b: f64) {
        if a.is_nan() || b.is_nan() {
            self.nan_seen = true;
            return;
        }
        if a.is_infinite() || b.is_infinite() {
            if a == 0.0 || b == 0.0 {
                self.nan_seen = true;
            } else if a.is_sign_positive() == b.is_sign_positive() {
                self.pos_inf_seen = true;
            } else {
                self.neg_inf_seen = true;
            }
            return;
        }
        if let (Some((na, sa, ea)), Some((nb, sb, eb))) =
            (Self::decode_finite_f64(a), Self::decode_finite_f64(b))
            && sa != 0
            && sb != 0
        {
            let prod: u128 = u128::from(sa) * u128::from(sb);
            let bit_pos = ((ea + eb) - KULISCH_MIN_EXP) as usize;
            self.accumulate_mag_u128(na ^ nb, prod, bit_pos);
        }
    }

    /// Associatively merges another `ExactF64Superaccumulator` into `self` for parallel shard reduction.
    #[inline]
    pub fn merge(&mut self, other: &Self) {
        self.nan_seen |= other.nan_seen;
        self.pos_inf_seen |= other.pos_inf_seen;
        self.neg_inf_seen |= other.neg_inf_seen;
        let mut carry = 0u64;
        for (dst, &src) in self.words.iter_mut().zip(other.words.iter()) {
            let (s1, c1) = (*dst).overflowing_add(src);
            let (s2, c2) = s1.overflowing_add(carry);
            *dst = s2;
            carry = u64::from(c1) + u64::from(c2);
        }
    }

    /// Rounds the exact superaccumulator **once** to the nearest IEEE-754 `f64` (`0.5 ULP` RNE,
    /// canonical quiet `NaN` `0x7FF8_0000_0000_0000` and canonical `+0.0`).
    pub fn to_f64_rne(&self) -> f64 {
        if self.nan_seen || (self.pos_inf_seen && self.neg_inf_seen) {
            return f64::from_bits(0x7FF8_0000_0000_0000);
        }
        if self.pos_inf_seen {
            return f64::INFINITY;
        }
        if self.neg_inf_seen {
            return f64::NEG_INFINITY;
        }

        let neg = self.is_negative();
        let mut mag = [0u64; KULISCH_F64_WORDS];
        if !neg {
            mag.copy_from_slice(&self.words);
        } else {
            let mut carry = 1u64;
            for (dst, &w) in mag.iter_mut().zip(self.words.iter()) {
                let (s, c) = (!w).overflowing_add(carry);
                *dst = s;
                carry = u64::from(c);
            }
        }

        let mut top_word = None;
        for (i, &w) in mag.iter().enumerate().rev() {
            if w != 0 {
                top_word = Some(i);
                break;
            }
        }
        let Some((tw, &tw_val)) = top_word.and_then(|i| mag.get(i).map(|v| (i, v))) else {
            return 0.0;
        };

        let msb_in_word = 63 - tw_val.leading_zeros() as usize;
        let msb_pos = tw * 64 + msb_in_word;
        let unbiased_exp = (msb_pos as i32) + KULISCH_MIN_EXP;

        let sign_mask = if neg { 1u64 << 63 } else { 0u64 };
        if unbiased_exp > 1023 {
            return f64::from_bits(sign_mask | (0x7FFu64 << 52));
        }

        let get_bit = |pos: isize| -> u64 {
            if pos < 0 {
                return 0;
            }
            let pu = pos as usize;
            let wi = pu >> 6;
            match mag.get(wi) {
                Some(&word) => (word >> (pu & 63)) & 1,
                None => 0,
            }
        };

        let any_nonzero_below = |pos: isize| -> bool {
            if pos < 0 {
                return false;
            }
            let pu = pos as usize;
            let wi = pu >> 6;
            let bi = pu & 63;
            let mask = if bi == 63 {
                u64::MAX
            } else {
                (1u64 << (bi + 1)) - 1
            };
            if let Some(&word) = mag.get(wi)
                && (word & mask) != 0
            {
                return true;
            }
            for &w in mag.iter().take(wi.min(KULISCH_F64_WORDS)) {
                if w != 0 {
                    return true;
                }
            }
            false
        };

        if unbiased_exp >= -1022 {
            let lsb_pos = (msb_pos as isize) - 52;
            let mut sig53 = 0u64;
            for offset in 0..=52isize {
                sig53 |= get_bit(lsb_pos + offset) << (offset as u32);
            }
            let guard = get_bit(lsb_pos - 1);
            let sticky = any_nonzero_below(lsb_pos - 2);
            if guard == 1 && (sticky || (sig53 & 1) == 1) {
                sig53 += 1;
            }
            let mut exp_field = (unbiased_exp + 1023) as u64;
            if sig53 >= (1u64 << 53) {
                sig53 >>= 1;
                exp_field += 1;
                if exp_field >= 0x7FF {
                    return f64::from_bits(sign_mask | (0x7FFu64 << 52));
                }
            }
            let frac = sig53 & ((1u64 << 52) - 1);
            f64::from_bits(sign_mask | (exp_field << 52) | frac)
        } else {
            let sub_lsb_pos: isize = (-1074 - KULISCH_MIN_EXP) as isize;
            let mut frac52 = 0u64;
            for offset in 0..52isize {
                frac52 |= get_bit(sub_lsb_pos + offset) << (offset as u32);
            }
            let guard = get_bit(sub_lsb_pos - 1);
            let sticky = any_nonzero_below(sub_lsb_pos - 2);
            if guard == 1 && (sticky || (frac52 & 1) == 1) {
                frac52 += 1;
            }
            if frac52 >= (1u64 << 52) {
                f64::from_bits(sign_mask | (1u64 << 52))
            } else if frac52 == 0 {
                0.0
            } else {
                f64::from_bits(sign_mask | frac52)
            }
        }
    }

    /// Exact Kulisch `f64` slice sum (`0.5 ULP` exact, 100% permutation-invariant).
    #[inline]
    pub fn sum_slice_exact(values: &[f64]) -> f64 {
        Self::sum_iter_exact(values.iter().copied())
    }

    /// Exact Kulisch `f64` iterator sum (`0.5 ULP` exact, immune to `HashMap` randomized iteration order).
    #[inline]
    pub fn sum_iter_exact<I: IntoIterator<Item = f64>>(iter: I) -> f64 {
        let mut acc = Self::new();
        for v in iter {
            acc.add_f64(v);
        }
        acc.to_f64_rne()
    }

    /// Exact Kulisch `f64` dot product over `min(a.len(), b.len())` pairs (`0` panics).
    #[inline]
    pub fn dot_slice_exact(a: &[f64], b: &[f64]) -> f64 {
        let mut acc = Self::new();
        for (&x, &y) in a.iter().zip(b.iter()) {
            acc.add_product_f64(x, y);
        }
        acc.to_f64_rne()
    }
}

/// Exact Kulisch-accumulated thermodynamic & carbon batch totals over a slice of [`LedgerEnergyReceipt`]s.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BatchThermodynamicSummary {
    pub receipt_count: usize,
    pub silicon_joules_total: f64,
    pub kwh_total: f64,
    pub kg_co2e_total: f64,
    pub landauer_floor_joules_total: f64,
    pub modeled_remote_joules_total: f64,
    pub external_cost_usd_total: f64,
}

/// Computes `(silicon_joules_total, landauer_floor_joules_total)` across `receipts` with `0.5 ULP`
/// Kulisch exactness and strict permutation invariance (matching `waios-jouledb`).
pub fn sum_receipt_joules_exact_kulisch(receipts: &[LedgerEnergyReceipt]) -> (f64, f64) {
    let summary = summarize_receipt_batch_exact(receipts);
    (
        summary.silicon_joules_total,
        summary.landauer_floor_joules_total,
    )
}

/// Summarizes all local silicon, grid carbon, Landauer lower-bound, and segregated external-API
/// metrics across `receipts` using stack-resident `ExactF64Superaccumulator`s (`0` heap allocations).
pub fn summarize_receipt_batch_exact(receipts: &[LedgerEnergyReceipt]) -> BatchThermodynamicSummary {
    let mut silicon_acc = ExactF64Superaccumulator::new();
    let mut kwh_acc = ExactF64Superaccumulator::new();
    let mut co2_acc = ExactF64Superaccumulator::new();
    let mut landauer_acc = ExactF64Superaccumulator::new();
    let mut remote_j_acc = ExactF64Superaccumulator::new();
    let mut usd_acc = ExactF64Superaccumulator::new();

    for r in receipts {
        silicon_acc.add_f64(r.energy_joules_total);
        kwh_acc.add_f64(r.kwh);
        co2_acc.add_f64(r.kg_co2e);
        if let Some(ref prov) = r.provenance {
            if let Some(ref ch) = prov.channels
                && let Some(lf) = ch.landauer_floor_joules
            {
                landauer_acc.add_f64(lf);
            }
            if let Some(ref ext) = prov.external_api {
                usd_acc.add_f64(ext.cost_usd);
                if let Some(rj) = ext.modeled_remote_joules {
                    remote_j_acc.add_f64(rj);
                }
            }
        }
    }

    BatchThermodynamicSummary {
        receipt_count: receipts.len(),
        silicon_joules_total: silicon_acc.to_f64_rne(),
        kwh_total: kwh_acc.to_f64_rne(),
        kg_co2e_total: co2_acc.to_f64_rne(),
        landauer_floor_joules_total: landauer_acc.to_f64_rne(),
        modeled_remote_joules_total: remote_j_acc.to_f64_rne(),
        external_cost_usd_total: usd_acc.to_f64_rne(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::receipt::{AcquisitionTier, EnergyProvenance, HardwareChannelJoules};
    use chrono::{TimeZone, Utc};
    use std::collections::HashMap;

    #[test]
    fn kulisch_prevents_negative_joule_cancellation_and_order_drift() {
        let landauer = 2.870_926_7e-21_f64;
        let fwd = [1.0e150, 1.0, landauer, -1.0e150, -1.0];
        let rev = [-1.0, -1.0e150, landauer, 1.0, 1.0e150];
        let shuf = [1.0e150, -1.0, landauer, -1.0e150, 1.0];

        // Naive IEEE-754 left-to-right sum hallucinates `-1.0 J` for strictly positive `+2.87e-21 J`:
        let naive: f64 = fwd.iter().copied().sum();
        assert_eq!(naive, -1.0);

        assert_eq!(
            ExactF64Superaccumulator::sum_slice_exact(&fwd).to_bits(),
            landauer.to_bits()
        );
        assert_eq!(
            ExactF64Superaccumulator::sum_slice_exact(&rev).to_bits(),
            landauer.to_bits()
        );
        assert_eq!(
            ExactF64Superaccumulator::sum_slice_exact(&shuf).to_bits(),
            landauer.to_bits()
        );

        let mut shard_a = ExactF64Superaccumulator::new();
        let mut shard_b = ExactF64Superaccumulator::new();
        shard_a.add_f64(fwd[0]);
        shard_a.add_f64(fwd[1]);
        shard_b.add_f64(fwd[2]);
        shard_b.add_f64(fwd[3]);
        shard_b.add_f64(fwd[4]);
        shard_a.merge(&shard_b);
        assert_eq!(shard_a.to_f64_rne().to_bits(), landauer.to_bits());
    }

    #[test]
    fn receipt_batch_summary_is_bit_exact_across_arrival_permutations() {
        let ts = Utc.with_ymd_and_hms(2026, 11, 26, 12, 0, 0).unwrap();
        let mk = |qid: &str, j: f64, lf: f64| LedgerEnergyReceipt {
            receipt_id: LedgerEnergyReceipt::compute_id(qid, "t1", &ts),
            qid: qid.to_string(),
            tenant_id: "t1".to_string(),
            workload_tag: None,
            branch_id: None,
            energy_joules_total: j,
            energy_joules_by_stage: HashMap::new(),
            kwh: j / 3_600_000.0,
            kg_co2e: (j / 3_600_000.0) * 0.35,
            grid_region: "US-NE-ISO".to_string(),
            grid_factor_source: "test".to_string(),
            timestamp_start: ts,
            timestamp_end: ts,
            device_target: "cpu".to_string(),
            algorithm_type: "hdc".to_string(),
            provenance: Some(EnergyProvenance {
                acquisition: AcquisitionTier::Measured,
                cascade_tier: None,
                channels: Some(HardwareChannelJoules {
                    cpu_joules: Some(j),
                    gpu_joules: None,
                    npu_joules: None,
                    dram_joules: None,
                    bits_erased: Some(1),
                    landauer_floor_joules: Some(lf),
                }),
                external_api: None,
                attestation: None,
            }),
        };

        let r_macro = mk("q-macro", 1.0e8, 2.870_926_7e-21);
        let r_micro = mk("q-micro", 1.0e-10, 2.870_926_7e-21);
        let r_mid = mk("q-mid", 3.141592653589793, 2.870_926_7e-21);

        let s1 = summarize_receipt_batch_exact(&[r_macro.clone(), r_micro.clone(), r_mid.clone()]);
        let s2 = summarize_receipt_batch_exact(&[r_micro, r_mid, r_macro]);
        assert_eq!(s1.silicon_joules_total.to_bits(), s2.silicon_joules_total.to_bits());
        assert_eq!(s1.kwh_total.to_bits(), s2.kwh_total.to_bits());
        assert_eq!(s1.kg_co2e_total.to_bits(), s2.kg_co2e_total.to_bits());
        assert_eq!(
            s1.landauer_floor_joules_total.to_bits(),
            s2.landauer_floor_joules_total.to_bits()
        );
    }
}
