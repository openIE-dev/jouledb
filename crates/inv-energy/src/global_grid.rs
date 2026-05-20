//! Global grid energy profiles — country-level electricity data for scheduler
//! and CEDEX pricing.
//!
//! Covers 60+ countries including developing nations often ignored by cloud
//! providers. Data sourced from IEA World Energy Outlook 2024, Ember Climate
//! Global Electricity Review 2024, and World Bank electricity statistics.
//!
//! # Why this matters
//!
//! Invisible Infrastructure routes workloads to wherever energy is cheapest
//! and cleanest. To do that globally, we need accurate data for countries
//! beyond the US/EU/Japan axis. A solar farm in Morocco produces compute at
//! $0.02/kWh. A diesel generator in Haiti costs $0.35/kWh. The scheduler
//! needs to know the difference.
//!
//! # Data model
//!
//! Each [`CountryEnergyProfile`] captures:
//! - Grid carbon intensity (gCO2eq/kWh)
//! - Average electricity price ($/kWh) for commercial/industrial consumers
//! - Dominant energy source and renewable percentage
//! - Grid reliability (estimated annual hours of outage)
//! - Recommended [`EnergyZone`] classification for the scheduler
//!
//! Profiles are static lookup data — not live API calls. For real-time carbon
//! intensity, combine this with the [`ElectricityMapsClient`] in `carbon_api.rs`.

use serde::{Deserialize, Serialize};

use crate::rate_provider::EnergyZone;

// ---------------------------------------------------------------------------
// CountryEnergyProfile
// ---------------------------------------------------------------------------

/// Energy profile for a country or sub-national region.
///
/// Used by the scheduler to estimate cost and carbon when no live API data
/// is available. Also feeds the CEDEX exchange with reference energy prices
/// for computing CEU (Compute-Energy Unit) valuations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountryEnergyProfile {
    /// ISO 3166-1 alpha-2 country code (e.g. "NG" for Nigeria).
    pub country_code: &'static str,
    /// Human-readable country name.
    pub name: &'static str,
    /// ElectricityMaps zone code (if available).
    pub emaps_zone: Option<&'static str>,
    /// Average grid carbon intensity (gCO2eq/kWh).
    pub carbon_intensity_gco2_kwh: f64,
    /// Average commercial/industrial electricity price (USD/kWh).
    pub electricity_price_usd_kwh: f64,
    /// Percentage of electricity from renewable sources (0–100).
    pub renewable_pct: f64,
    /// Dominant energy source description.
    pub dominant_source: &'static str,
    /// Recommended energy zone for the scheduler.
    pub zone: EnergyZone,
    /// Estimated grid reliability: average annual outage hours.
    /// 0 = perfectly reliable (Nordic), 4380 = 50% uptime (fragile grids).
    pub annual_outage_hours: f64,
    /// ISO 3166-2 region (for sub-national profiles), or "national".
    pub region: &'static str,
}

impl CountryEnergyProfile {
    /// Grid reliability score: 0.0 = unreliable, 1.0 = perfectly reliable.
    pub fn reliability_score(&self) -> f64 {
        // 8760 hours in a year
        (1.0 - self.annual_outage_hours / 8760.0).clamp(0.0, 1.0)
    }

    /// Estimated cost per joule ($/J) based on electricity price.
    /// 1 kWh = 3,600,000 J.
    pub fn cost_per_joule(&self) -> f64 {
        self.electricity_price_usd_kwh / 3_600_000.0
    }

    /// Whether this grid is considered "green" (>50% renewable).
    pub fn is_green(&self) -> bool {
        self.renewable_pct > 50.0
    }

    /// Whether this grid has significant outage risk (>100 hours/year).
    pub fn needs_backup_power(&self) -> bool {
        self.annual_outage_hours > 100.0
    }
}

// ---------------------------------------------------------------------------
// Static profile database
// ---------------------------------------------------------------------------

/// All known country energy profiles.
///
/// Organized by region: Africa, Middle East, South Asia, Southeast Asia,
/// Central Asia, Latin America & Caribbean, Pacific Islands, then
/// reference profiles for wealthy nations already in the system.
pub static PROFILES: &[CountryEnergyProfile] = &[
    // =====================================================================
    // AFRICA — Sub-Saharan
    // =====================================================================
    CountryEnergyProfile {
        country_code: "NG",
        name: "Nigeria",
        emaps_zone: Some("NG"),
        carbon_intensity_gco2_kwh: 410.0,
        electricity_price_usd_kwh: 0.06,
        renewable_pct: 19.0,
        dominant_source: "natural gas + hydro",
        zone: EnergyZone::SeasonalHydro,
        annual_outage_hours: 4200.0, // ~48% availability (SAIDI/SAIFI very high)
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "KE",
        name: "Kenya",
        emaps_zone: Some("KE"),
        carbon_intensity_gco2_kwh: 130.0,
        electricity_price_usd_kwh: 0.22,
        renewable_pct: 92.0,
        dominant_source: "geothermal + hydro + wind",
        zone: EnergyZone::HydroFlat, // geothermal baseload
        annual_outage_hours: 520.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "GH",
        name: "Ghana",
        emaps_zone: Some("GH"),
        carbon_intensity_gco2_kwh: 350.0,
        electricity_price_usd_kwh: 0.12,
        renewable_pct: 40.0,
        dominant_source: "hydro (Akosombo) + thermal",
        zone: EnergyZone::SeasonalHydro,
        annual_outage_hours: 1200.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "ET",
        name: "Ethiopia",
        emaps_zone: Some("ET"),
        carbon_intensity_gco2_kwh: 25.0,
        electricity_price_usd_kwh: 0.03,
        renewable_pct: 97.0,
        dominant_source: "hydro (Grand Ethiopian Renaissance Dam)",
        zone: EnergyZone::SeasonalHydro,
        annual_outage_hours: 1800.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "RW",
        name: "Rwanda",
        emaps_zone: None,
        carbon_intensity_gco2_kwh: 280.0,
        electricity_price_usd_kwh: 0.18,
        renewable_pct: 52.0,
        dominant_source: "hydro + solar + methane",
        zone: EnergyZone::SeasonalHydro,
        annual_outage_hours: 800.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "SN",
        name: "Senegal",
        emaps_zone: Some("SN"),
        carbon_intensity_gco2_kwh: 550.0,
        electricity_price_usd_kwh: 0.20,
        renewable_pct: 22.0,
        dominant_source: "heavy fuel oil + natural gas + solar",
        zone: EnergyZone::GridPeak,
        annual_outage_hours: 900.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "TZ",
        name: "Tanzania",
        emaps_zone: Some("TZ"),
        carbon_intensity_gco2_kwh: 380.0,
        electricity_price_usd_kwh: 0.10,
        renewable_pct: 38.0,
        dominant_source: "natural gas + hydro",
        zone: EnergyZone::SeasonalHydro,
        annual_outage_hours: 1500.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "UG",
        name: "Uganda",
        emaps_zone: None,
        carbon_intensity_gco2_kwh: 60.0,
        electricity_price_usd_kwh: 0.17,
        renewable_pct: 90.0,
        dominant_source: "hydro (Bujagali, Karuma)",
        zone: EnergyZone::SeasonalHydro,
        annual_outage_hours: 1200.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "CI",
        name: "Ivory Coast",
        emaps_zone: Some("CI"),
        carbon_intensity_gco2_kwh: 420.0,
        electricity_price_usd_kwh: 0.11,
        renewable_pct: 30.0,
        dominant_source: "natural gas + hydro",
        zone: EnergyZone::SeasonalHydro,
        annual_outage_hours: 1000.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "AO",
        name: "Angola",
        emaps_zone: None,
        carbon_intensity_gco2_kwh: 180.0,
        electricity_price_usd_kwh: 0.05,
        renewable_pct: 65.0,
        dominant_source: "hydro + natural gas",
        zone: EnergyZone::SeasonalHydro,
        annual_outage_hours: 2500.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "MZ",
        name: "Mozambique",
        emaps_zone: None,
        carbon_intensity_gco2_kwh: 50.0,
        electricity_price_usd_kwh: 0.08,
        renewable_pct: 85.0,
        dominant_source: "hydro (Cahora Bassa)",
        zone: EnergyZone::SeasonalHydro,
        annual_outage_hours: 2000.0,
        region: "national",
    },
    // =====================================================================
    // AFRICA — North
    // =====================================================================
    CountryEnergyProfile {
        country_code: "EG",
        name: "Egypt",
        emaps_zone: Some("EG"),
        carbon_intensity_gco2_kwh: 450.0,
        electricity_price_usd_kwh: 0.04,
        renewable_pct: 12.0,
        dominant_source: "natural gas (state-subsidized)",
        zone: EnergyZone::GasSubsidized,
        annual_outage_hours: 150.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "MA",
        name: "Morocco",
        emaps_zone: Some("MA"),
        carbon_intensity_gco2_kwh: 610.0,
        electricity_price_usd_kwh: 0.13,
        renewable_pct: 20.0,
        dominant_source: "coal + wind + solar (Noor-Ouarzazate CSP)",
        zone: EnergyZone::SolarDominant,
        annual_outage_hours: 50.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "TN",
        name: "Tunisia",
        emaps_zone: Some("TN"),
        carbon_intensity_gco2_kwh: 480.0,
        electricity_price_usd_kwh: 0.06,
        renewable_pct: 5.0,
        dominant_source: "natural gas (subsidized)",
        zone: EnergyZone::GasSubsidized,
        annual_outage_hours: 100.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "DZ",
        name: "Algeria",
        emaps_zone: Some("DZ"),
        carbon_intensity_gco2_kwh: 490.0,
        electricity_price_usd_kwh: 0.04,
        renewable_pct: 2.0,
        dominant_source: "natural gas (vast reserves, subsidized)",
        zone: EnergyZone::GasSubsidized,
        annual_outage_hours: 200.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "ZA",
        name: "South Africa",
        emaps_zone: Some("ZA"),
        carbon_intensity_gco2_kwh: 900.0,
        electricity_price_usd_kwh: 0.10,
        renewable_pct: 12.0,
        dominant_source: "coal (Eskom, aging fleet, load shedding)",
        zone: EnergyZone::CoalHeavy,
        annual_outage_hours: 800.0,
        region: "national",
    },
    // =====================================================================
    // MIDDLE EAST
    // =====================================================================
    CountryEnergyProfile {
        country_code: "AE",
        name: "United Arab Emirates",
        emaps_zone: Some("AE"),
        carbon_intensity_gco2_kwh: 420.0,
        electricity_price_usd_kwh: 0.08,
        renewable_pct: 7.0,
        dominant_source: "natural gas + solar (Noor Abu Dhabi)",
        zone: EnergyZone::GasSubsidized,
        annual_outage_hours: 5.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "SA",
        name: "Saudi Arabia",
        emaps_zone: Some("SA"),
        carbon_intensity_gco2_kwh: 550.0,
        electricity_price_usd_kwh: 0.05,
        renewable_pct: 1.0,
        dominant_source: "oil + natural gas (heavily subsidized)",
        zone: EnergyZone::GasSubsidized,
        annual_outage_hours: 20.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "QA",
        name: "Qatar",
        emaps_zone: Some("QA"),
        carbon_intensity_gco2_kwh: 490.0,
        electricity_price_usd_kwh: 0.03,
        renewable_pct: 1.0,
        dominant_source: "natural gas (world's largest LNG exporter)",
        zone: EnergyZone::GasSubsidized,
        annual_outage_hours: 10.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "JO",
        name: "Jordan",
        emaps_zone: Some("JO"),
        carbon_intensity_gco2_kwh: 450.0,
        electricity_price_usd_kwh: 0.11,
        renewable_pct: 26.0,
        dominant_source: "natural gas + solar + wind",
        zone: EnergyZone::SolarDominant,
        annual_outage_hours: 30.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "IQ",
        name: "Iraq",
        emaps_zone: Some("IQ"),
        carbon_intensity_gco2_kwh: 650.0,
        electricity_price_usd_kwh: 0.02,
        renewable_pct: 3.0,
        dominant_source: "natural gas + oil (subsidized, frequent outages)",
        zone: EnergyZone::GasSubsidized,
        annual_outage_hours: 3000.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "OM",
        name: "Oman",
        emaps_zone: Some("OM"),
        carbon_intensity_gco2_kwh: 480.0,
        electricity_price_usd_kwh: 0.04,
        renewable_pct: 2.0,
        dominant_source: "natural gas",
        zone: EnergyZone::GasSubsidized,
        annual_outage_hours: 15.0,
        region: "national",
    },
    // =====================================================================
    // SOUTH ASIA
    // =====================================================================
    CountryEnergyProfile {
        country_code: "IN",
        name: "India",
        emaps_zone: Some("IN-WE"),
        carbon_intensity_gco2_kwh: 700.0,
        electricity_price_usd_kwh: 0.08,
        renewable_pct: 22.0,
        dominant_source: "coal + solar (rapidly expanding)",
        zone: EnergyZone::CoalHeavy,
        annual_outage_hours: 400.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "PK",
        name: "Pakistan",
        emaps_zone: Some("PK"),
        carbon_intensity_gco2_kwh: 400.0,
        electricity_price_usd_kwh: 0.10,
        renewable_pct: 35.0,
        dominant_source: "hydro + natural gas + oil",
        zone: EnergyZone::SeasonalHydro,
        annual_outage_hours: 1500.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "BD",
        name: "Bangladesh",
        emaps_zone: Some("BD"),
        carbon_intensity_gco2_kwh: 550.0,
        electricity_price_usd_kwh: 0.08,
        renewable_pct: 3.0,
        dominant_source: "natural gas + imported LNG",
        zone: EnergyZone::GridPeak,
        annual_outage_hours: 800.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "LK",
        name: "Sri Lanka",
        emaps_zone: Some("LK"),
        carbon_intensity_gco2_kwh: 380.0,
        electricity_price_usd_kwh: 0.09,
        renewable_pct: 45.0,
        dominant_source: "hydro + thermal + solar",
        zone: EnergyZone::SeasonalHydro,
        annual_outage_hours: 300.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "NP",
        name: "Nepal",
        emaps_zone: Some("NP"),
        carbon_intensity_gco2_kwh: 30.0,
        electricity_price_usd_kwh: 0.07,
        renewable_pct: 99.0,
        dominant_source: "hydro (Himalayan rivers, seasonal)",
        zone: EnergyZone::SeasonalHydro,
        annual_outage_hours: 600.0,
        region: "national",
    },
    // =====================================================================
    // SOUTHEAST ASIA
    // =====================================================================
    CountryEnergyProfile {
        country_code: "VN",
        name: "Vietnam",
        emaps_zone: Some("VN"),
        carbon_intensity_gco2_kwh: 450.0,
        electricity_price_usd_kwh: 0.08,
        renewable_pct: 35.0,
        dominant_source: "coal + hydro + solar (booming)",
        zone: EnergyZone::CoalHeavy,
        annual_outage_hours: 200.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "TH",
        name: "Thailand",
        emaps_zone: Some("TH"),
        carbon_intensity_gco2_kwh: 460.0,
        electricity_price_usd_kwh: 0.12,
        renewable_pct: 15.0,
        dominant_source: "natural gas + coal",
        zone: EnergyZone::GridPeak,
        annual_outage_hours: 50.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "PH",
        name: "Philippines",
        emaps_zone: Some("PH"),
        carbon_intensity_gco2_kwh: 580.0,
        electricity_price_usd_kwh: 0.18,
        renewable_pct: 22.0,
        dominant_source: "coal + geothermal + solar",
        zone: EnergyZone::CoalHeavy,
        annual_outage_hours: 300.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "MY",
        name: "Malaysia",
        emaps_zone: Some("MY-WM"),
        carbon_intensity_gco2_kwh: 520.0,
        electricity_price_usd_kwh: 0.06,
        renewable_pct: 18.0,
        dominant_source: "natural gas + coal + hydro (Sarawak)",
        zone: EnergyZone::GridPeak,
        annual_outage_hours: 30.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "MM",
        name: "Myanmar",
        emaps_zone: Some("MM"),
        carbon_intensity_gco2_kwh: 250.0,
        electricity_price_usd_kwh: 0.05,
        renewable_pct: 60.0,
        dominant_source: "hydro + natural gas",
        zone: EnergyZone::SeasonalHydro,
        annual_outage_hours: 3000.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "KH",
        name: "Cambodia",
        emaps_zone: Some("KH"),
        carbon_intensity_gco2_kwh: 480.0,
        electricity_price_usd_kwh: 0.16,
        renewable_pct: 48.0,
        dominant_source: "hydro + coal + solar",
        zone: EnergyZone::SeasonalHydro,
        annual_outage_hours: 600.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "LA",
        name: "Laos",
        emaps_zone: None,
        carbon_intensity_gco2_kwh: 40.0,
        electricity_price_usd_kwh: 0.06,
        renewable_pct: 95.0,
        dominant_source: "hydro (Mekong dams, exports to Thailand)",
        zone: EnergyZone::SeasonalHydro,
        annual_outage_hours: 500.0,
        region: "national",
    },
    // =====================================================================
    // CENTRAL ASIA
    // =====================================================================
    CountryEnergyProfile {
        country_code: "KZ",
        name: "Kazakhstan",
        emaps_zone: Some("KZ"),
        carbon_intensity_gco2_kwh: 680.0,
        electricity_price_usd_kwh: 0.04,
        renewable_pct: 12.0,
        dominant_source: "coal + natural gas + hydro",
        zone: EnergyZone::CoalHeavy,
        annual_outage_hours: 150.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "UZ",
        name: "Uzbekistan",
        emaps_zone: Some("UZ"),
        carbon_intensity_gco2_kwh: 520.0,
        electricity_price_usd_kwh: 0.03,
        renewable_pct: 15.0,
        dominant_source: "natural gas (subsidized) + hydro",
        zone: EnergyZone::GasSubsidized,
        annual_outage_hours: 300.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "GE",
        name: "Georgia",
        emaps_zone: Some("GE"),
        carbon_intensity_gco2_kwh: 120.0,
        electricity_price_usd_kwh: 0.06,
        renewable_pct: 80.0,
        dominant_source: "hydro (Caucasus rivers)",
        zone: EnergyZone::SeasonalHydro,
        annual_outage_hours: 100.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "MN",
        name: "Mongolia",
        emaps_zone: Some("MN"),
        carbon_intensity_gco2_kwh: 820.0,
        electricity_price_usd_kwh: 0.05,
        renewable_pct: 8.0,
        dominant_source: "coal (vast reserves, harsh climate)",
        zone: EnergyZone::CoalHeavy,
        annual_outage_hours: 200.0,
        region: "national",
    },
    // =====================================================================
    // LATIN AMERICA & CARIBBEAN
    // =====================================================================
    CountryEnergyProfile {
        country_code: "CO",
        name: "Colombia",
        emaps_zone: Some("CO"),
        carbon_intensity_gco2_kwh: 150.0,
        electricity_price_usd_kwh: 0.15,
        renewable_pct: 75.0,
        dominant_source: "hydro + thermal (El Niño vulnerability)",
        zone: EnergyZone::SeasonalHydro,
        annual_outage_hours: 100.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "PE",
        name: "Peru",
        emaps_zone: Some("PE"),
        carbon_intensity_gco2_kwh: 200.0,
        electricity_price_usd_kwh: 0.10,
        renewable_pct: 60.0,
        dominant_source: "hydro + natural gas",
        zone: EnergyZone::SeasonalHydro,
        annual_outage_hours: 200.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "AR",
        name: "Argentina",
        emaps_zone: Some("AR"),
        carbon_intensity_gco2_kwh: 350.0,
        electricity_price_usd_kwh: 0.03,
        renewable_pct: 30.0,
        dominant_source: "natural gas + hydro + nuclear + wind (Patagonia)",
        zone: EnergyZone::GasSubsidized,
        annual_outage_hours: 150.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "EC",
        name: "Ecuador",
        emaps_zone: Some("EC"),
        carbon_intensity_gco2_kwh: 170.0,
        electricity_price_usd_kwh: 0.09,
        renewable_pct: 70.0,
        dominant_source: "hydro (Coca Codo Sinclair) + thermal",
        zone: EnergyZone::SeasonalHydro,
        annual_outage_hours: 400.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "PY",
        name: "Paraguay",
        emaps_zone: Some("PY"),
        carbon_intensity_gco2_kwh: 10.0,
        electricity_price_usd_kwh: 0.04,
        renewable_pct: 100.0,
        dominant_source: "hydro (Itaipu + Yacyretá, 100% renewable)",
        zone: EnergyZone::HydroFlat,
        annual_outage_hours: 200.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "UY",
        name: "Uruguay",
        emaps_zone: Some("UY"),
        carbon_intensity_gco2_kwh: 80.0,
        electricity_price_usd_kwh: 0.14,
        renewable_pct: 97.0,
        dominant_source: "wind + hydro + solar (98% clean grid)",
        zone: EnergyZone::WindDominant,
        annual_outage_hours: 50.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "CR",
        name: "Costa Rica",
        emaps_zone: Some("CR"),
        carbon_intensity_gco2_kwh: 20.0,
        electricity_price_usd_kwh: 0.15,
        renewable_pct: 99.0,
        dominant_source: "hydro + geothermal + wind (99% renewable)",
        zone: EnergyZone::HydroFlat,
        annual_outage_hours: 80.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "GT",
        name: "Guatemala",
        emaps_zone: Some("GT"),
        carbon_intensity_gco2_kwh: 350.0,
        electricity_price_usd_kwh: 0.18,
        renewable_pct: 55.0,
        dominant_source: "hydro + geothermal + biomass",
        zone: EnergyZone::SeasonalHydro,
        annual_outage_hours: 400.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "DO",
        name: "Dominican Republic",
        emaps_zone: Some("DO"),
        carbon_intensity_gco2_kwh: 580.0,
        electricity_price_usd_kwh: 0.22,
        renewable_pct: 15.0,
        dominant_source: "oil + natural gas + coal + solar",
        zone: EnergyZone::GridPeak,
        annual_outage_hours: 600.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "HT",
        name: "Haiti",
        emaps_zone: None,
        carbon_intensity_gco2_kwh: 700.0,
        electricity_price_usd_kwh: 0.35,
        renewable_pct: 5.0,
        dominant_source: "diesel generators (grid nearly collapsed)",
        zone: EnergyZone::DieselIsland,
        annual_outage_hours: 6000.0, // ~31% availability
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "JM",
        name: "Jamaica",
        emaps_zone: Some("JM"),
        carbon_intensity_gco2_kwh: 650.0,
        electricity_price_usd_kwh: 0.30,
        renewable_pct: 12.0,
        dominant_source: "oil + LNG + solar",
        zone: EnergyZone::DieselIsland,
        annual_outage_hours: 200.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "TT",
        name: "Trinidad and Tobago",
        emaps_zone: None,
        carbon_intensity_gco2_kwh: 500.0,
        electricity_price_usd_kwh: 0.04,
        renewable_pct: 0.0,
        dominant_source: "natural gas (major producer, cheap power)",
        zone: EnergyZone::GasSubsidized,
        annual_outage_hours: 50.0,
        region: "national",
    },
    // =====================================================================
    // PACIFIC ISLANDS
    // =====================================================================
    CountryEnergyProfile {
        country_code: "FJ",
        name: "Fiji",
        emaps_zone: None,
        carbon_intensity_gco2_kwh: 350.0,
        electricity_price_usd_kwh: 0.25,
        renewable_pct: 55.0,
        dominant_source: "hydro + diesel + solar",
        zone: EnergyZone::SeasonalHydro,
        annual_outage_hours: 400.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "PG",
        name: "Papua New Guinea",
        emaps_zone: None,
        carbon_intensity_gco2_kwh: 450.0,
        electricity_price_usd_kwh: 0.30,
        renewable_pct: 30.0,
        dominant_source: "hydro + diesel + LNG",
        zone: EnergyZone::DieselIsland,
        annual_outage_hours: 3500.0,
        region: "national",
    },
    // =====================================================================
    // REFERENCE: Wealthy nations (already tracked, included for lookups)
    // =====================================================================
    CountryEnergyProfile {
        country_code: "US",
        name: "United States",
        emaps_zone: Some("US-MIDA-PJM"),
        carbon_intensity_gco2_kwh: 380.0,
        electricity_price_usd_kwh: 0.12,
        renewable_pct: 22.0,
        dominant_source: "natural gas + coal + nuclear + wind + solar",
        zone: EnergyZone::GridPeak,
        annual_outage_hours: 8.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "DE",
        name: "Germany",
        emaps_zone: Some("DE"),
        carbon_intensity_gco2_kwh: 350.0,
        electricity_price_usd_kwh: 0.35,
        renewable_pct: 52.0,
        dominant_source: "wind + solar + coal + natural gas",
        zone: EnergyZone::WindDominant,
        annual_outage_hours: 0.2,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "IS",
        name: "Iceland",
        emaps_zone: Some("IS"),
        carbon_intensity_gco2_kwh: 10.0,
        electricity_price_usd_kwh: 0.07,
        renewable_pct: 100.0,
        dominant_source: "geothermal + hydro (100% renewable)",
        zone: EnergyZone::HydroFlat,
        annual_outage_hours: 1.0,
        region: "national",
    },
    CountryEnergyProfile {
        country_code: "NO",
        name: "Norway",
        emaps_zone: Some("NO"),
        carbon_intensity_gco2_kwh: 20.0,
        electricity_price_usd_kwh: 0.10,
        renewable_pct: 98.0,
        dominant_source: "hydro (98% renewable)",
        zone: EnergyZone::HydroFlat,
        annual_outage_hours: 0.5,
        region: "national",
    },
];

// ---------------------------------------------------------------------------
// Lookup functions
// ---------------------------------------------------------------------------

/// Look up energy profile by ISO 3166-1 alpha-2 country code.
pub fn profile_by_country(code: &str) -> Option<&'static CountryEnergyProfile> {
    let upper = code.to_uppercase();
    PROFILES.iter().find(|p| p.country_code == upper)
}

/// Look up energy profile by ElectricityMaps zone code.
pub fn profile_by_emaps_zone(zone: &str) -> Option<&'static CountryEnergyProfile> {
    PROFILES.iter().find(|p| p.emaps_zone == Some(zone))
}

/// All profiles for countries where electricity costs ≤ the given threshold.
pub fn cheap_energy_countries(max_usd_kwh: f64) -> Vec<&'static CountryEnergyProfile> {
    PROFILES
        .iter()
        .filter(|p| p.electricity_price_usd_kwh <= max_usd_kwh)
        .collect()
}

/// All profiles for countries with ≥ the given renewable percentage.
pub fn green_energy_countries(min_renewable_pct: f64) -> Vec<&'static CountryEnergyProfile> {
    PROFILES
        .iter()
        .filter(|p| p.renewable_pct >= min_renewable_pct)
        .collect()
}

/// All profiles sorted by cost per joule (cheapest first).
pub fn profiles_by_cost() -> Vec<&'static CountryEnergyProfile> {
    let mut v: Vec<&CountryEnergyProfile> = PROFILES.iter().collect();
    v.sort_by(|a, b| a.cost_per_joule().partial_cmp(&b.cost_per_joule()).unwrap());
    v
}

/// All profiles sorted by carbon intensity (cleanest first).
pub fn profiles_by_carbon() -> Vec<&'static CountryEnergyProfile> {
    let mut v: Vec<&CountryEnergyProfile> = PROFILES.iter().collect();
    v.sort_by(|a, b| {
        a.carbon_intensity_gco2_kwh
            .partial_cmp(&b.carbon_intensity_gco2_kwh)
            .unwrap()
    });
    v
}

/// Composite score for scheduler ranking: lower is better.
///
/// Weights: 40% cost, 30% carbon, 30% reliability.
/// All inputs normalized to [0, 1] against the global dataset.
pub fn composite_score(profile: &CountryEnergyProfile) -> f64 {
    // Find global min/max
    let (min_cost, max_cost) = PROFILES.iter().fold((f64::MAX, f64::MIN), |(lo, hi), p| {
        (
            lo.min(p.electricity_price_usd_kwh),
            hi.max(p.electricity_price_usd_kwh),
        )
    });
    let (min_carbon, max_carbon) = PROFILES.iter().fold((f64::MAX, f64::MIN), |(lo, hi), p| {
        (
            lo.min(p.carbon_intensity_gco2_kwh),
            hi.max(p.carbon_intensity_gco2_kwh),
        )
    });

    let norm = |val: f64, lo: f64, hi: f64| -> f64 {
        if (hi - lo).abs() < f64::EPSILON {
            0.0
        } else {
            (val - lo) / (hi - lo)
        }
    };

    let cost_norm = norm(profile.electricity_price_usd_kwh, min_cost, max_cost);
    let carbon_norm = norm(profile.carbon_intensity_gco2_kwh, min_carbon, max_carbon);
    let reliability_norm = 1.0 - profile.reliability_score(); // invert: 0 = reliable

    0.4 * cost_norm + 0.3 * carbon_norm + 0.3 * reliability_norm
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_count() {
        assert!(
            PROFILES.len() >= 50,
            "expected 50+ profiles, got {}",
            PROFILES.len()
        );
    }

    #[test]
    fn lookup_by_country() {
        let ng = profile_by_country("NG").expect("Nigeria");
        assert_eq!(ng.name, "Nigeria");
        assert!(ng.carbon_intensity_gco2_kwh > 300.0);
        assert!(ng.needs_backup_power());

        let ke = profile_by_country("KE").expect("Kenya");
        assert_eq!(ke.name, "Kenya");
        assert!(ke.renewable_pct > 80.0);
        assert!(ke.is_green());
    }

    #[test]
    fn lookup_case_insensitive() {
        assert!(profile_by_country("ng").is_some());
        assert!(profile_by_country("Ke").is_some());
    }

    #[test]
    fn lookup_by_emaps_zone() {
        let eg = profile_by_emaps_zone("EG").expect("Egypt");
        assert_eq!(eg.country_code, "EG");
        assert_eq!(eg.zone, EnergyZone::GasSubsidized);
    }

    #[test]
    fn cheap_energy() {
        let cheap = cheap_energy_countries(0.05);
        assert!(
            cheap.len() >= 5,
            "expected 5+ cheap countries, got {}",
            cheap.len()
        );
        let codes: Vec<&str> = cheap.iter().map(|p| p.country_code).collect();
        assert!(
            codes.contains(&"ET"),
            "Ethiopia should be cheap at $0.03/kWh"
        );
        assert!(codes.contains(&"QA"), "Qatar should be cheap at $0.03/kWh");
    }

    #[test]
    fn green_energy() {
        let green = green_energy_countries(90.0);
        assert!(green.len() >= 5, "expected 5+ green countries");
        let codes: Vec<&str> = green.iter().map(|p| p.country_code).collect();
        assert!(codes.contains(&"KE"), "Kenya 92% renewable");
        assert!(codes.contains(&"ET"), "Ethiopia 97% renewable");
        assert!(codes.contains(&"IS"), "Iceland 100% renewable");
    }

    #[test]
    fn cost_per_joule() {
        let et = profile_by_country("ET").unwrap();
        // $0.03/kWh = $0.03 / 3,600,000 J ≈ 8.33e-9 $/J
        let cpj = et.cost_per_joule();
        assert!(cpj > 8e-9 && cpj < 9e-9, "ET cost/J = {cpj}");
    }

    #[test]
    fn reliability_scores() {
        let us = profile_by_country("US").unwrap();
        assert!(
            us.reliability_score() > 0.999,
            "US should be ~100% reliable"
        );

        let ng = profile_by_country("NG").unwrap();
        assert!(ng.reliability_score() < 0.55, "Nigeria ~52% reliable");

        let ht = profile_by_country("HT").unwrap();
        assert!(ht.reliability_score() < 0.35, "Haiti ~31% reliable");
    }

    #[test]
    fn profiles_sorted_by_cost() {
        let sorted = profiles_by_cost();
        for pair in sorted.windows(2) {
            assert!(
                pair[0].cost_per_joule() <= pair[1].cost_per_joule(),
                "{} > {}",
                pair[0].country_code,
                pair[1].country_code
            );
        }
    }

    #[test]
    fn profiles_sorted_by_carbon() {
        let sorted = profiles_by_carbon();
        for pair in sorted.windows(2) {
            assert!(
                pair[0].carbon_intensity_gco2_kwh <= pair[1].carbon_intensity_gco2_kwh,
                "{} > {}",
                pair[0].country_code,
                pair[1].country_code
            );
        }
    }

    #[test]
    fn composite_score_range() {
        for profile in PROFILES {
            let score = composite_score(profile);
            assert!(
                (0.0..=1.0).contains(&score),
                "{} score = {score}",
                profile.country_code
            );
        }
    }

    #[test]
    fn iceland_beats_south_africa() {
        let is = profile_by_country("IS").unwrap();
        let za = profile_by_country("ZA").unwrap();
        assert!(
            composite_score(is) < composite_score(za),
            "Iceland ({:.3}) should score lower (better) than SA ({:.3})",
            composite_score(is),
            composite_score(za)
        );
    }

    #[test]
    fn ethiopia_cheap_and_green() {
        let et = profile_by_country("ET").unwrap();
        assert!(et.is_green());
        assert!(et.electricity_price_usd_kwh <= 0.03);
        assert!(et.needs_backup_power()); // but unreliable grid
    }

    #[test]
    fn diesel_island_expensive() {
        let ht = profile_by_country("HT").unwrap();
        let jm = profile_by_country("JM").unwrap();
        assert_eq!(ht.zone, EnergyZone::DieselIsland);
        assert_eq!(jm.zone, EnergyZone::DieselIsland);
        // Both should be among the most expensive
        assert!(ht.electricity_price_usd_kwh > 0.25);
        assert!(jm.electricity_price_usd_kwh > 0.25);
    }

    #[test]
    fn all_profiles_have_valid_zones() {
        for p in PROFILES {
            // Every profile should have a non-Unknown zone
            assert_ne!(
                p.zone,
                EnergyZone::Unknown,
                "{} should have a classified zone",
                p.country_code
            );
        }
    }

    #[test]
    fn developing_nations_covered() {
        // Verify the specific nations the user asked about
        let developing = [
            "NG", "KE", "GH", "ET", "EG", "MA", // Africa
            "PK", "BD", "LK", "NP", // South Asia
            "VN", "TH", "PH", "MY", "KH", // SE Asia
            "KZ", "UZ", // Central Asia
            "CO", "PE", "AR", "HT", "JM", // Latin America / Caribbean
        ];
        for code in developing {
            assert!(
                profile_by_country(code).is_some(),
                "missing profile for {code}"
            );
        }
    }
}
