//! Property-based tests for GridMix carbon intensity calculations.
//!
//! Verifies that GridMix.carbon_intensity() and GridMix.renewable_pct()
//! satisfy their algebraic contracts for all possible grid compositions.

use proptest::prelude::*;

use crate::carbon_aggregator::GridMix;

/// Strategy to generate a valid GridMix (all non-negative percentages).
fn arb_grid_mix() -> impl Strategy<Value = GridMix> {
    // Generate 9 non-negative values
    prop::array::uniform9(0.0f64..100.0).prop_map(|arr| GridMix {
        coal: arr[0],
        gas: arr[1],
        oil: arr[2],
        nuclear: arr[3],
        hydro: arr[4],
        wind: arr[5],
        solar: arr[6],
        biomass: arr[7],
        geothermal: arr[8],
    })
}

/// Strategy to generate a normalized GridMix (sums to 100).
fn arb_normalized_grid_mix() -> impl Strategy<Value = GridMix> {
    prop::array::uniform9(0.0f64..100.0).prop_map(|arr| {
        let total: f64 = arr.iter().sum();
        if total == 0.0 {
            // All zeros — use equal distribution
            return GridMix {
                coal: 100.0 / 9.0,
                gas: 100.0 / 9.0,
                oil: 100.0 / 9.0,
                nuclear: 100.0 / 9.0,
                hydro: 100.0 / 9.0,
                wind: 100.0 / 9.0,
                solar: 100.0 / 9.0,
                biomass: 100.0 / 9.0,
                geothermal: 100.0 / 9.0,
            };
        }
        let scale = 100.0 / total;
        GridMix {
            coal: arr[0] * scale,
            gas: arr[1] * scale,
            oil: arr[2] * scale,
            nuclear: arr[3] * scale,
            hydro: arr[4] * scale,
            wind: arr[5] * scale,
            solar: arr[6] * scale,
            biomass: arr[7] * scale,
            geothermal: arr[8] * scale,
        }
    })
}

proptest! {
    /// **Property 1: carbon_intensity is bounded by IPCC extremes.**
    ///
    /// For any normalized grid mix, intensity must be between 11.0 (wind) and 820.0 (coal).
    #[test]
    fn prop_carbon_intensity_bounded(mix in arb_normalized_grid_mix()) {
        let ci = mix.carbon_intensity();
        prop_assert!(
            ci >= 11.0 - 0.01 && ci <= 820.0 + 0.01,
            "carbon_intensity = {ci} must be in [11.0, 820.0]"
        );
    }

    /// **Property 2: carbon_intensity never panics for any input.**
    #[test]
    fn prop_carbon_intensity_no_panic(mix in arb_grid_mix()) {
        let _ci = mix.carbon_intensity();
    }

    /// **Property 3: Pure coal grid has highest intensity.**
    #[test]
    fn prop_pure_coal_max(_dummy in Just(())) {
        let coal_grid = GridMix {
            coal: 100.0, gas: 0.0, oil: 0.0, nuclear: 0.0,
            hydro: 0.0, wind: 0.0, solar: 0.0, biomass: 0.0, geothermal: 0.0,
        };
        let ci = coal_grid.carbon_intensity();
        prop_assert!((ci - 820.0).abs() < 0.01, "pure coal = {ci}, expected 820.0");
    }

    /// **Property 4: Pure wind grid has lowest intensity.**
    #[test]
    fn prop_pure_wind_min(_dummy in Just(())) {
        let wind_grid = GridMix {
            coal: 0.0, gas: 0.0, oil: 0.0, nuclear: 0.0,
            hydro: 0.0, wind: 100.0, solar: 0.0, biomass: 0.0, geothermal: 0.0,
        };
        let ci = wind_grid.carbon_intensity();
        prop_assert!((ci - 11.0).abs() < 0.01, "pure wind = {ci}, expected 11.0");
    }

    /// **Property 5: renewable_pct + fossil_pct + nuclear = total.**
    #[test]
    fn prop_pct_partition(mix in arb_grid_mix()) {
        let renewable = mix.renewable_pct();
        let fossil = mix.fossil_pct();
        let total = mix.coal + mix.gas + mix.oil + mix.nuclear
            + mix.hydro + mix.wind + mix.solar + mix.biomass + mix.geothermal;
        let sum = renewable + fossil + mix.nuclear;
        prop_assert!(
            (sum - total).abs() < 1e-10,
            "renewable({renewable}) + fossil({fossil}) + nuclear({}) = {sum}, expected {total}",
            mix.nuclear
        );
    }

    /// **Property 6: renewable_pct is non-negative.**
    #[test]
    fn prop_renewable_pct_nonneg(mix in arb_grid_mix()) {
        let pct = mix.renewable_pct();
        prop_assert!(pct >= 0.0, "renewable_pct must be >= 0, got {pct}");
    }

    /// **Property 7: fossil_pct is non-negative.**
    #[test]
    fn prop_fossil_pct_nonneg(mix in arb_grid_mix()) {
        let pct = mix.fossil_pct();
        prop_assert!(pct >= 0.0, "fossil_pct must be >= 0, got {pct}");
    }

    /// **Property 8: Adding more renewables decreases carbon intensity.**
    ///
    /// Shifting 10% from coal to wind must decrease (or hold) carbon intensity.
    #[test]
    fn prop_more_renewables_lower_carbon(mix in arb_normalized_grid_mix()) {
        if mix.coal >= 10.0 {
            let greener = GridMix {
                coal: mix.coal - 10.0,
                wind: mix.wind + 10.0,
                ..mix
            };
            let ci_before = mix.carbon_intensity();
            let ci_after = greener.carbon_intensity();
            prop_assert!(
                ci_after <= ci_before + 0.01,
                "shifting coal->wind must decrease intensity: {ci_before} -> {ci_after}"
            );
        }
    }

    /// **Property 9: Zero grid returns fallback (400.0).**
    #[test]
    fn prop_zero_grid_fallback(_dummy in Just(())) {
        let zero = GridMix {
            coal: 0.0, gas: 0.0, oil: 0.0, nuclear: 0.0,
            hydro: 0.0, wind: 0.0, solar: 0.0, biomass: 0.0, geothermal: 0.0,
        };
        prop_assert_eq!(zero.carbon_intensity(), 400.0);
    }
}
