//! Software Carbon Intensity (SCI) score calculation.
//!
//! Implements the Green Software Foundation's SCI formula:
//! SCI = ((E * I) + M) / R
//! where E = energy, I = carbon intensity, M = embodied carbon, R = functional units.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// SciConfig
// ---------------------------------------------------------------------------

/// Configuration describing the deployment context for an SCI calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SciConfig {
    /// Cloud region or data-centre location (e.g. "eu-west-1").
    pub region: String,
    /// Hardware model identifier (e.g. "Dell PowerEdge R750").
    pub hardware_model: String,
    /// Expected hardware lifetime in years; used to amortise embodied carbon.
    pub lifetime_years: f64,
    /// Name of the functional unit (e.g. "API request", "transaction").
    pub functional_unit_name: String,
}

// ---------------------------------------------------------------------------
// EmbodiedCarbonEstimate
// ---------------------------------------------------------------------------

/// Estimated embodied carbon broken down by lifecycle stage.
///
/// All values are in grams of CO2 equivalent (gCO2eq).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbodiedCarbonEstimate {
    /// Carbon emitted during hardware manufacturing.
    pub manufacturing_gco2: f64,
    /// Carbon emitted during transport and logistics.
    pub transport_gco2: f64,
    /// Carbon emitted during disposal / end-of-life processing.
    pub end_of_life_gco2: f64,
}

impl EmbodiedCarbonEstimate {
    /// Total embodied carbon across all lifecycle stages.
    pub fn total(&self) -> f64 {
        self.manufacturing_gco2 + self.transport_gco2 + self.end_of_life_gco2
    }
}

// ---------------------------------------------------------------------------
// SciMeasurements
// ---------------------------------------------------------------------------

/// Runtime measurements fed into the SCI formula.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SciMeasurements {
    /// Energy consumed during the measurement period (kWh).
    pub energy_kwh: f64,
    /// Marginal carbon intensity of the grid (gCO2 per kWh).
    pub carbon_intensity_gco2_kwh: f64,
    /// Number of functional units served during the period.
    pub functional_unit_count: f64,
}

// ---------------------------------------------------------------------------
// SciScore
// ---------------------------------------------------------------------------

/// The fully computed SCI score together with its input components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SciScore {
    /// Energy consumed (kWh).
    pub energy_kwh: f64,
    /// Carbon intensity (gCO2 / kWh).
    pub carbon_intensity_gco2_kwh: f64,
    /// Embodied carbon (gCO2).
    pub embodied_carbon_gco2: f64,
    /// Number of functional units.
    pub functional_unit_count: f64,
    /// Operational carbon: E * I (gCO2).
    pub operational_carbon_gco2: f64,
    /// Final SCI value: ((E*I) + M) / R  (gCO2 per functional unit).
    pub sci_value: f64,
    /// Configuration used for this calculation.
    pub config: SciConfig,
}

// ---------------------------------------------------------------------------
// SciError
// ---------------------------------------------------------------------------

/// Validation errors that can occur when computing the SCI score.
#[derive(Debug, thiserror::Error)]
pub enum SciError {
    /// The functional unit count was zero, which would cause division by zero.
    #[error("functional unit count must be greater than zero")]
    ZeroFunctionalUnits,
    /// Energy value was negative.
    #[error("energy must not be negative, got {0}")]
    NegativeEnergy(f64),
    /// Carbon intensity value was negative.
    #[error("carbon intensity must not be negative, got {0}")]
    InvalidCarbonIntensity(f64),
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Computes the SCI score without input validation.
///
/// # Panics
///
/// This function does **not** validate its inputs. If
/// `functional_unit_count` is zero the result will be `inf` or `NaN`.
/// Use [`calculate_sci_checked`] for validated inputs.
pub fn calculate_sci(
    config: SciConfig,
    measurements: &SciMeasurements,
    embodied: &EmbodiedCarbonEstimate,
) -> SciScore {
    let operational = measurements.energy_kwh * measurements.carbon_intensity_gco2_kwh;
    let sci_value = (operational + embodied.total()) / measurements.functional_unit_count;

    SciScore {
        energy_kwh: measurements.energy_kwh,
        carbon_intensity_gco2_kwh: measurements.carbon_intensity_gco2_kwh,
        embodied_carbon_gco2: embodied.total(),
        functional_unit_count: measurements.functional_unit_count,
        operational_carbon_gco2: operational,
        sci_value,
        config,
    }
}

/// Computes the SCI score after validating all inputs.
///
/// Returns [`SciError`] if any input is out of range.
pub fn calculate_sci_checked(
    config: SciConfig,
    measurements: &SciMeasurements,
    embodied: &EmbodiedCarbonEstimate,
) -> Result<SciScore, SciError> {
    if measurements.functional_unit_count == 0.0 {
        return Err(SciError::ZeroFunctionalUnits);
    }
    if measurements.energy_kwh < 0.0 {
        return Err(SciError::NegativeEnergy(measurements.energy_kwh));
    }
    if measurements.carbon_intensity_gco2_kwh < 0.0 {
        return Err(SciError::InvalidCarbonIntensity(
            measurements.carbon_intensity_gco2_kwh,
        ));
    }

    Ok(calculate_sci(config, measurements, embodied))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_config(region: &str) -> SciConfig {
        SciConfig {
            region: region.to_string(),
            hardware_model: "TestServer-1U".to_string(),
            lifetime_years: 4.0,
            functional_unit_name: "request".to_string(),
        }
    }

    fn sample_embodied() -> EmbodiedCarbonEstimate {
        EmbodiedCarbonEstimate {
            manufacturing_gco2: 1000.0,
            transport_gco2: 200.0,
            end_of_life_gco2: 50.0,
        }
    }

    #[test]
    fn basic_sci_calculation() {
        let measurements = SciMeasurements {
            energy_kwh: 100.0,
            carbon_intensity_gco2_kwh: 400.0,
            functional_unit_count: 1000.0,
        };
        let score = calculate_sci(
            sample_config("us-east-1"),
            &measurements,
            &sample_embodied(),
        );

        // operational = 100 * 400 = 40_000
        // embodied total = 1250
        // SCI = (40_000 + 1250) / 1000 = 41.25
        assert!((score.sci_value - 41.25).abs() < 1e-10);
    }

    #[test]
    fn sci_with_zero_embodied() {
        let embodied = EmbodiedCarbonEstimate {
            manufacturing_gco2: 0.0,
            transport_gco2: 0.0,
            end_of_life_gco2: 0.0,
        };
        let measurements = SciMeasurements {
            energy_kwh: 50.0,
            carbon_intensity_gco2_kwh: 200.0,
            functional_unit_count: 100.0,
        };
        let score = calculate_sci(sample_config("eu-west-1"), &measurements, &embodied);

        // SCI = (50 * 200 + 0) / 100 = 100.0
        assert!((score.sci_value - 100.0).abs() < 1e-10);
    }

    #[test]
    fn sci_high_carbon_intensity() {
        let measurements = SciMeasurements {
            energy_kwh: 10.0,
            carbon_intensity_gco2_kwh: 900.0,
            functional_unit_count: 100.0,
        };
        let score = calculate_sci(
            sample_config("ap-south-1"),
            &measurements,
            &sample_embodied(),
        );

        // operational = 10 * 900 = 9000; total = (9000 + 1250) / 100 = 102.5
        assert!((score.sci_value - 102.5).abs() < 1e-10);
    }

    #[test]
    fn sci_low_carbon_intensity() {
        let measurements = SciMeasurements {
            energy_kwh: 10.0,
            carbon_intensity_gco2_kwh: 20.0,
            functional_unit_count: 100.0,
        };
        let score = calculate_sci(
            sample_config("eu-north-1"),
            &measurements,
            &sample_embodied(),
        );

        // operational = 10 * 20 = 200; total = (200 + 1250) / 100 = 14.5
        assert!((score.sci_value - 14.5).abs() < 1e-10);
    }

    #[test]
    fn sci_many_functional_units() {
        let measurements = SciMeasurements {
            energy_kwh: 500.0,
            carbon_intensity_gco2_kwh: 300.0,
            functional_unit_count: 1_000_000.0,
        };
        let score = calculate_sci(
            sample_config("us-west-2"),
            &measurements,
            &sample_embodied(),
        );

        // operational = 150_000; total = (150_000 + 1250) / 1_000_000 = 0.15125
        assert!((score.sci_value - 0.15125).abs() < 1e-10);
    }

    #[test]
    fn embodied_total() {
        let embodied = EmbodiedCarbonEstimate {
            manufacturing_gco2: 500.0,
            transport_gco2: 100.0,
            end_of_life_gco2: 25.0,
        };
        assert!((embodied.total() - 625.0).abs() < 1e-10);
    }

    #[test]
    fn sci_checked_valid() {
        let measurements = SciMeasurements {
            energy_kwh: 100.0,
            carbon_intensity_gco2_kwh: 400.0,
            functional_unit_count: 1000.0,
        };
        let result = calculate_sci_checked(
            sample_config("us-east-1"),
            &measurements,
            &sample_embodied(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn sci_checked_zero_functional_units() {
        let measurements = SciMeasurements {
            energy_kwh: 100.0,
            carbon_intensity_gco2_kwh: 400.0,
            functional_unit_count: 0.0,
        };
        let result = calculate_sci_checked(
            sample_config("us-east-1"),
            &measurements,
            &sample_embodied(),
        );
        assert!(matches!(result, Err(SciError::ZeroFunctionalUnits)));
    }

    #[test]
    fn sci_checked_negative_energy() {
        let measurements = SciMeasurements {
            energy_kwh: -10.0,
            carbon_intensity_gco2_kwh: 400.0,
            functional_unit_count: 100.0,
        };
        let result = calculate_sci_checked(
            sample_config("us-east-1"),
            &measurements,
            &sample_embodied(),
        );
        assert!(matches!(result, Err(SciError::NegativeEnergy(_))));
    }

    #[test]
    fn sci_checked_negative_intensity() {
        let measurements = SciMeasurements {
            energy_kwh: 100.0,
            carbon_intensity_gco2_kwh: -50.0,
            functional_unit_count: 100.0,
        };
        let result = calculate_sci_checked(
            sample_config("us-east-1"),
            &measurements,
            &sample_embodied(),
        );
        assert!(matches!(result, Err(SciError::InvalidCarbonIntensity(_))));
    }

    #[test]
    fn operational_carbon_calculation() {
        let measurements = SciMeasurements {
            energy_kwh: 25.0,
            carbon_intensity_gco2_kwh: 300.0,
            functional_unit_count: 10.0,
        };
        let score = calculate_sci(
            sample_config("eu-west-1"),
            &measurements,
            &sample_embodied(),
        );

        assert!((score.operational_carbon_gco2 - 7500.0).abs() < 1e-10);
    }

    #[test]
    fn config_serialization() {
        let config = sample_config("us-east-1");
        let json = serde_json::to_string(&config).expect("serialize config");
        let deserialized: SciConfig = serde_json::from_str(&json).expect("deserialize config");

        assert_eq!(deserialized.region, "us-east-1");
        assert_eq!(deserialized.hardware_model, "TestServer-1U");
        assert!((deserialized.lifetime_years - 4.0).abs() < 1e-10);
        assert_eq!(deserialized.functional_unit_name, "request");
    }

    #[test]
    fn score_serialization() {
        let measurements = SciMeasurements {
            energy_kwh: 100.0,
            carbon_intensity_gco2_kwh: 400.0,
            functional_unit_count: 1000.0,
        };
        let score = calculate_sci(
            sample_config("us-east-1"),
            &measurements,
            &sample_embodied(),
        );
        let json = serde_json::to_string(&score).expect("serialize score");
        let deserialized: SciScore = serde_json::from_str(&json).expect("deserialize score");

        assert!((deserialized.sci_value - score.sci_value).abs() < 1e-10);
        assert_eq!(deserialized.config.region, "us-east-1");
    }

    #[test]
    fn measurements_serialization() {
        let measurements = SciMeasurements {
            energy_kwh: 42.0,
            carbon_intensity_gco2_kwh: 350.0,
            functional_unit_count: 500.0,
        };
        let json = serde_json::to_string(&measurements).expect("serialize measurements");
        let deserialized: SciMeasurements =
            serde_json::from_str(&json).expect("deserialize measurements");

        assert!((deserialized.energy_kwh - 42.0).abs() < 1e-10);
        assert!((deserialized.carbon_intensity_gco2_kwh - 350.0).abs() < 1e-10);
        assert!((deserialized.functional_unit_count - 500.0).abs() < 1e-10);
    }

    #[test]
    fn embodied_serialization() {
        let embodied = sample_embodied();
        let json = serde_json::to_string(&embodied).expect("serialize embodied");
        let deserialized: EmbodiedCarbonEstimate =
            serde_json::from_str(&json).expect("deserialize embodied");

        assert!((deserialized.total() - embodied.total()).abs() < 1e-10);
    }

    #[test]
    fn different_regions_different_scores() {
        let embodied = sample_embodied();
        let m_high = SciMeasurements {
            energy_kwh: 100.0,
            carbon_intensity_gco2_kwh: 800.0,
            functional_unit_count: 1000.0,
        };
        let m_low = SciMeasurements {
            energy_kwh: 100.0,
            carbon_intensity_gco2_kwh: 50.0,
            functional_unit_count: 1000.0,
        };

        let score_high = calculate_sci(sample_config("coal-region"), &m_high, &embodied);
        let score_low = calculate_sci(sample_config("hydro-region"), &m_low, &embodied);

        assert!(score_high.sci_value > score_low.sci_value);
    }
}
