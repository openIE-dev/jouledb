//! Live energy rate provider — bridges mining's energy knowledge to the scheduler.
//!
//! The mining system has the most sophisticated energy model in the mesh:
//! real $/kWh rates, time-of-day curves (wind, solar, nordic, metro peak),
//! and marginal power tracking. This module exposes that knowledge so the
//! scheduler and VDC placement can prefer cheap-energy nodes for ALL workloads,
//! not just mining.
//!
//! # Energy Zones
//!
//! Nodes are classified into energy zones based on their power source and
//! pricing pattern. The zone determines the time-of-day pricing curve:
//!
//! - **HydroFlat**: Hydro/geothermal — constant rate, always cheap
//! - **WindDominant**: Wind farms — cheapest overnight, expensive afternoon
//! - **SolarDominant**: Solar arrays — cheapest midday, expensive at night
//! - **NordicMix**: Hydro base + wind overlay — moderate variance
//! - **GridPeak**: City grid — business-hours premium, cheaper overnight
//! - **FlaredGas**: Stranded gas at wellheads — near-zero marginal cost
//!
//! # Design
//!
//! `NodeEnergyRate` is a per-node snapshot produced each epoch. The scheduler
//! normalizes rates across candidates to populate `CostInputs.energy` with
//! real data instead of hard-coded class defaults.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// EnergyZone — what kind of power site is this?
// ---------------------------------------------------------------------------

/// Energy zone classification — determines time-of-day pricing behavior.
///
/// Nodes advertise their zone via gossip. The zone is typically stable
/// (a datacenter doesn't change its power source hourly) but the *rate*
/// within the zone changes based on the time-of-day curve.
///
/// Zones cover both wealthy-nation patterns (hydro, wind, solar, nordic)
/// and developing-nation patterns (diesel islands, coal-heavy grids,
/// seasonal hydro, subsidized gas, off-grid microgrids).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnergyZone {
    /// Hydroelectric or geothermal — flat pricing, always cheap.
    HydroFlat,
    /// Wind-dominated — cheapest overnight (0.15×), expensive evening (1.4×).
    WindDominant,
    /// Solar-dominated — cheapest midday (0.2×), expensive at night (1.3×).
    SolarDominant,
    /// Nordic hydro + wind mix — moderate variance.
    NordicMix,
    /// City/metro grid — business-hours premium (1.2×), cheap overnight (0.6×).
    GridPeak,
    /// Flared gas at oil/gas wellheads — near-zero marginal cost.
    FlaredGas,
    /// Diesel generator island — constant high cost, common in Caribbean,
    /// Pacific islands, and remote African locations. Flat rate (expensive).
    DieselIsland,
    /// Coal-heavy grid — cheap but high-carbon. India, South Africa,
    /// Indonesia, parts of SE Asia. Slight day/night variance.
    CoalHeavy,
    /// Seasonal hydro — East Africa (Kenya, Ethiopia, Uganda), Central
    /// America, Nepal. Cheap in wet season, expensive in dry season.
    /// Time-of-day multiplier models the *average* annual pattern;
    /// actual rates swing ±40% between wet/dry months.
    SeasonalHydro,
    /// State-subsidized natural gas — Middle East, North Africa, Central
    /// Asia. Cheap and flat because governments absorb price volatility.
    GasSubsidized,
    /// Off-grid solar + battery microgrid — rural Africa, South Asia,
    /// Pacific islands. Solar curve buffered by battery storage.
    MiniGrid,
    /// Unknown or unclassified energy source.
    #[default]
    Unknown,
}

impl EnergyZone {
    /// Time-of-day rate multiplier for this zone.
    /// Base rate × multiplier = actual $/kWh at the given hour (0–23 UTC).
    pub fn multiplier(&self, hour: u8) -> f64 {
        let h = hour.min(23);
        match self {
            Self::HydroFlat | Self::FlaredGas => 1.0,
            Self::WindDominant => match h {
                0..=5 => 0.15,  // overnight wind peak
                6..=8 => 0.6,   // morning ramp
                9..=15 => 1.1,  // daytime
                16..=20 => 1.4, // evening peak
                _ => 0.4,       // late evening
            },
            Self::SolarDominant => match h {
                9..=15 => 0.2,  // peak sun — near-zero marginal
                7..=8 => 0.5,   // morning ramp
                16..=18 => 0.7, // afternoon decline
                _ => 1.3,       // nighttime
            },
            Self::NordicMix => match h {
                1..=5 => 0.5, // deep night hydro
                6..=7 => 0.8,
                8..=16 => 1.0,
                17..=20 => 1.3, // evening peak
                _ => 0.7,
            },
            Self::GridPeak => match h {
                8..=21 => 1.2, // business + evening
                _ => 0.6,      // overnight
            },
            // -- Developing-nation zones --
            Self::DieselIsland => 1.0, // flat — diesel cost doesn't vary with time
            Self::CoalHeavy => match h {
                0..=5 => 0.85,   // slight overnight dip (less industrial load)
                6..=9 => 1.0,    // morning ramp
                10..=17 => 1.1,  // daytime industrial peak
                18..=21 => 1.15, // evening residential peak
                _ => 0.9,        // late evening
            },
            Self::SeasonalHydro => match h {
                // Models annual-average pattern; actual rates swing ±40%
                // between wet (cheap) and dry (expensive) months.
                0..=5 => 0.7,   // overnight baseload hydro
                6..=8 => 0.9,   // morning ramp
                9..=16 => 1.0,  // daytime
                17..=21 => 1.3, // evening peak (hydro + thermal backup)
                _ => 0.8,       // late evening
            },
            Self::GasSubsidized => 1.0, // flat — state absorbs volatility
            Self::MiniGrid => match h {
                // Solar + battery: cheap midday, moderate overnight (battery),
                // expensive at dawn/dusk when battery is depleted/charging.
                10..=15 => 0.3, // peak solar generation
                7..=9 => 0.8,   // morning (battery depleting)
                16..=18 => 0.7, // afternoon (still some solar + battery)
                19..=22 => 1.2, // evening (battery only)
                _ => 1.1,       // overnight (battery draining)
            },
            Self::Unknown => 1.0,
        }
    }

    /// Average 24-hour multiplier (for estimating daily cost from base rate).
    pub fn daily_average_multiplier(&self) -> f64 {
        let sum: f64 = (0..24).map(|h| self.multiplier(h)).sum();
        sum / 24.0
    }
}

impl std::fmt::Display for EnergyZone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HydroFlat => write!(f, "hydro-flat"),
            Self::WindDominant => write!(f, "wind-dominant"),
            Self::SolarDominant => write!(f, "solar-dominant"),
            Self::NordicMix => write!(f, "nordic-mix"),
            Self::GridPeak => write!(f, "grid-peak"),
            Self::FlaredGas => write!(f, "flared-gas"),
            Self::DieselIsland => write!(f, "diesel-island"),
            Self::CoalHeavy => write!(f, "coal-heavy"),
            Self::SeasonalHydro => write!(f, "seasonal-hydro"),
            Self::GasSubsidized => write!(f, "gas-subsidized"),
            Self::MiniGrid => write!(f, "mini-grid"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

// ---------------------------------------------------------------------------
// NodeEnergyRate — per-node live snapshot
// ---------------------------------------------------------------------------

/// Live energy rate snapshot for a single node.
///
/// Produced each epoch by the energy scanner (or mining's SubstrateScanner).
/// The scheduler uses this to compute real `CostInputs.energy` scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeEnergyRate {
    /// Node identifier.
    pub node_id: String,
    /// Energy zone classification.
    pub zone: EnergyZone,
    /// Base electricity rate ($/kWh) before time-of-day adjustment.
    pub base_rate_usd_kwh: f64,
    /// Current effective rate ($/kWh) = base × zone multiplier at current hour.
    pub effective_rate_usd_kwh: f64,
    /// Current hour (UTC, 0–23) used for the multiplier.
    pub hour_utc: u8,
    /// Energy source label (for transparency).
    pub energy_source: String,
    /// Marginal watts — the additional power if a workload is added to this node.
    /// Lower marginal watts = node already running → cheaper to add work.
    pub marginal_watts: Option<f64>,
}

impl NodeEnergyRate {
    /// Compute from base rate, zone, and current hour.
    pub fn new(
        node_id: String,
        zone: EnergyZone,
        base_rate: f64,
        hour_utc: u8,
        source: String,
    ) -> Self {
        let effective = base_rate * zone.multiplier(hour_utc);
        Self {
            node_id,
            zone,
            base_rate_usd_kwh: base_rate,
            effective_rate_usd_kwh: effective,
            hour_utc,
            energy_source: source,
            marginal_watts: None,
        }
    }

    /// Normalize this rate to a [0.0, 1.0] score for the scheduler.
    /// 0.0 = cheapest possible, 1.0 = most expensive.
    /// `min_rate` and `max_rate` are the fleet-wide range for normalization.
    pub fn to_cost_score(&self, min_rate: f64, max_rate: f64) -> f64 {
        if (max_rate - min_rate).abs() < f64::EPSILON {
            return 0.0; // all nodes same rate — no preference
        }
        ((self.effective_rate_usd_kwh - min_rate) / (max_rate - min_rate)).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hydro_flat_is_constant() {
        let zone = EnergyZone::HydroFlat;
        for h in 0..24 {
            assert!((zone.multiplier(h) - 1.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn wind_cheapest_overnight() {
        let zone = EnergyZone::WindDominant;
        let overnight = zone.multiplier(2); // 0.15
        let peak = zone.multiplier(18); // 1.4
        assert!(overnight < 0.2);
        assert!(peak > 1.3);
        assert!(overnight < peak);
    }

    #[test]
    fn solar_cheapest_midday() {
        let zone = EnergyZone::SolarDominant;
        let midday = zone.multiplier(12); // 0.2
        let night = zone.multiplier(23); // 1.3
        assert!(midday < 0.3);
        assert!(night > 1.2);
    }

    #[test]
    fn daily_average_multiplier_reasonable() {
        // Flat should average to exactly 1.0
        assert!((EnergyZone::HydroFlat.daily_average_multiplier() - 1.0).abs() < f64::EPSILON);
        // Others should average somewhere around 0.5–1.0
        let wind_avg = EnergyZone::WindDominant.daily_average_multiplier();
        assert!(wind_avg > 0.4 && wind_avg < 1.2, "wind avg = {wind_avg}");
    }

    #[test]
    fn node_energy_rate_effective() {
        let rate = NodeEnergyRate::new(
            "solar-node-1".into(),
            EnergyZone::SolarDominant,
            0.10, // $0.10/kWh base
            12,   // noon UTC
            "solar".into(),
        );
        // Midday solar: 0.10 × 0.2 = $0.02/kWh
        assert!((rate.effective_rate_usd_kwh - 0.02).abs() < 1e-10);
    }

    #[test]
    fn cost_score_normalization() {
        let cheap = NodeEnergyRate::new(
            "cheap".into(),
            EnergyZone::HydroFlat,
            0.01,
            12,
            "hydro".into(),
        );
        let mid = NodeEnergyRate::new("mid".into(), EnergyZone::GridPeak, 0.10, 12, "grid".into());
        let expensive = NodeEnergyRate::new(
            "expensive".into(),
            EnergyZone::GridPeak,
            0.30,
            14,
            "grid".into(),
        );

        let min = cheap.effective_rate_usd_kwh;
        let max = expensive.effective_rate_usd_kwh;

        assert!((cheap.to_cost_score(min, max) - 0.0).abs() < 0.01);
        assert!(mid.to_cost_score(min, max) > 0.2);
        assert!((expensive.to_cost_score(min, max) - 1.0).abs() < 0.01);
    }

    #[test]
    fn flared_gas_near_zero() {
        let rate = NodeEnergyRate::new(
            "permian-1".into(),
            EnergyZone::FlaredGas,
            0.008, // $0.008/kWh flared gas
            0,     // any hour
            "flared-gas".into(),
        );
        // Flat multiplier, so effective = base
        assert!((rate.effective_rate_usd_kwh - 0.008).abs() < f64::EPSILON);
    }

    #[test]
    fn zone_display() {
        assert_eq!(format!("{}", EnergyZone::HydroFlat), "hydro-flat");
        assert_eq!(format!("{}", EnergyZone::WindDominant), "wind-dominant");
        assert_eq!(format!("{}", EnergyZone::SolarDominant), "solar-dominant");
        assert_eq!(format!("{}", EnergyZone::FlaredGas), "flared-gas");
        assert_eq!(format!("{}", EnergyZone::DieselIsland), "diesel-island");
        assert_eq!(format!("{}", EnergyZone::CoalHeavy), "coal-heavy");
        assert_eq!(format!("{}", EnergyZone::SeasonalHydro), "seasonal-hydro");
        assert_eq!(format!("{}", EnergyZone::GasSubsidized), "gas-subsidized");
        assert_eq!(format!("{}", EnergyZone::MiniGrid), "mini-grid");
    }

    // -- Developing-nation zone tests --

    #[test]
    fn diesel_island_flat() {
        let zone = EnergyZone::DieselIsland;
        for h in 0..24 {
            assert!((zone.multiplier(h) - 1.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn coal_heavy_slight_variance() {
        let zone = EnergyZone::CoalHeavy;
        let overnight = zone.multiplier(2); // 0.85
        let peak = zone.multiplier(19); // 1.15
        assert!(overnight < 0.9);
        assert!(peak > 1.1);
        // Coal grids have less variance than wind/solar
        assert!((peak - overnight) < 0.5);
    }

    #[test]
    fn seasonal_hydro_evening_peak() {
        let zone = EnergyZone::SeasonalHydro;
        let overnight = zone.multiplier(3); // 0.7
        let peak = zone.multiplier(19); // 1.3
        assert!(overnight < 0.8);
        assert!(peak > 1.2);
    }

    #[test]
    fn gas_subsidized_flat() {
        let zone = EnergyZone::GasSubsidized;
        for h in 0..24 {
            assert!((zone.multiplier(h) - 1.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn mini_grid_solar_peak() {
        let zone = EnergyZone::MiniGrid;
        let midday = zone.multiplier(12); // 0.3 — peak solar
        let night = zone.multiplier(20); // 1.2 — battery only
        assert!(midday < 0.4);
        assert!(night > 1.0);
    }

    #[test]
    fn developing_zone_daily_averages() {
        // All developing-nation zones should average between 0.5 and 1.2
        for zone in [
            EnergyZone::DieselIsland,
            EnergyZone::CoalHeavy,
            EnergyZone::SeasonalHydro,
            EnergyZone::GasSubsidized,
            EnergyZone::MiniGrid,
        ] {
            let avg = zone.daily_average_multiplier();
            assert!(avg > 0.5 && avg < 1.2, "{zone} daily avg = {avg}");
        }
    }
}
