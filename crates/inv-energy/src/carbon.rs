use inv_core::energy::Joules;
use serde::{Deserialize, Serialize};

/// Grid carbon intensity data for a region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarbonIntensity {
    /// Region identifier (e.g., "us-east", "eu-west").
    pub region: String,
    /// Grams of CO2 equivalent per kilowatt-hour.
    pub gco2_per_kwh: f64,
    /// Unix timestamp when this measurement was taken.
    pub timestamp: u64,
    /// Data source (e.g., "watttime", "electricity-maps", "estimated").
    pub source: String,
}

impl CarbonIntensity {
    /// Convert gCO2e/kWh to gCO2e/joule.
    /// 1 kWh = 3,600,000 joules.
    pub fn gco2_per_joule(&self) -> f64 {
        self.gco2_per_kwh / 3_600_000.0
    }

    /// Calculate the carbon footprint of a given energy amount.
    pub fn carbon_for(&self, joules: Joules) -> f64 {
        joules.as_f64() * self.gco2_per_joule()
    }
}

/// Network transfer energy cost constants (joules per megabyte).
pub struct TransferCost;

impl TransferCost {
    /// Ethernet: ~0.0064 J/MB
    pub const ETHERNET_J_PER_MB: f64 = 0.0064;
    /// WiFi: ~0.037 J/MB
    pub const WIFI_J_PER_MB: f64 = 0.037;
    /// Cellular: ~0.12 J/MB
    pub const CELLULAR_J_PER_MB: f64 = 0.12;

    /// Estimate transfer energy for a given size and connection type.
    pub fn estimate(megabytes: f64, connection: ConnectionType) -> Joules {
        let j_per_mb = match connection {
            ConnectionType::Ethernet => Self::ETHERNET_J_PER_MB,
            ConnectionType::Wifi => Self::WIFI_J_PER_MB,
            ConnectionType::Cellular => Self::CELLULAR_J_PER_MB,
        };
        Joules::new(megabytes * j_per_mb)
    }
}

/// Network connection type for transfer cost estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionType {
    Ethernet,
    Wifi,
    Cellular,
}

/// Default carbon intensity values for common regions (2024 averages).
///
/// Covers wealthy-nation cloud regions and developing-nation grids.
/// Values from IEA World Energy Outlook 2024 and Ember Climate 2024.
pub fn default_carbon_intensity(region: &str) -> CarbonIntensity {
    let gco2_per_kwh = match region {
        // -- Wealthy-nation cloud regions --
        "us-east" | "us-east-1" => 380.0,
        "us-west" | "us-west-2" => 200.0,
        "us-central" | "us-central-1" => 420.0,
        "us-south" | "us-south-1" => 400.0,
        "eu-west" | "eu-west-1" => 270.0,
        "eu-north" | "eu-north-1" => 30.0,
        "eu-central" | "eu-central-1" => 350.0,
        "ap-south" | "ap-south-1" => 700.0,
        "ap-northeast" | "ap-northeast-1" => 450.0,
        "ap-southeast" | "ap-southeast-1" => 400.0,
        "ca-central" | "ca-central-1" => 120.0,
        "sa-east" | "sa-east-1" => 60.0, // Brazil — mostly hydro
        // -- Africa --
        "af-south" | "af-south-1" => 900.0, // South Africa — coal
        "af-west-ng" => 410.0,              // Nigeria — gas + hydro
        "af-east-ke" => 130.0,              // Kenya — geothermal
        "af-west-gh" => 350.0,              // Ghana — hydro + thermal
        "af-east-et" => 25.0,               // Ethiopia — nearly 100% hydro
        "af-north-eg" => 450.0,             // Egypt — gas
        "af-north-ma" => 610.0,             // Morocco — coal + renewables
        "af-north-tn" => 480.0,             // Tunisia — gas
        "af-north-dz" => 490.0,             // Algeria — gas
        "af-east-tz" => 380.0,              // Tanzania — gas + hydro
        "af-east-ug" => 60.0,               // Uganda — hydro
        "af-east-rw" => 280.0,              // Rwanda — mixed
        "af-west-sn" => 550.0,              // Senegal — oil + gas
        "af-west-ci" => 420.0,              // Ivory Coast — gas + hydro
        "af-south-mz" => 50.0,              // Mozambique — hydro
        "af-south-ao" => 180.0,             // Angola — hydro + gas
        // -- Middle East --
        "me-central" | "me-central-1" => 420.0, // UAE
        "me-south" | "me-south-1" => 550.0,     // Saudi Arabia
        "me-west" | "me-west-1" => 450.0,       // Jordan / Israel
        "me-qatar" => 490.0,
        "me-iraq" => 650.0,
        "me-oman" => 480.0,
        // -- South Asia --
        "sa-south-pk" => 400.0, // Pakistan — hydro + gas
        "sa-south-bd" => 550.0, // Bangladesh — gas
        "sa-south-lk" => 380.0, // Sri Lanka — hydro + thermal
        "sa-south-np" => 30.0,  // Nepal — hydro
        // -- Southeast Asia --
        "ap-southeast-vn" => 450.0, // Vietnam — coal + hydro
        "ap-southeast-th" => 460.0, // Thailand — gas + coal
        "ap-southeast-ph" => 580.0, // Philippines — coal
        "ap-southeast-my" => 520.0, // Malaysia — gas + coal
        "ap-southeast-mm" => 250.0, // Myanmar — hydro + gas
        "ap-southeast-kh" => 480.0, // Cambodia — hydro + coal
        "ap-southeast-la" => 40.0,  // Laos — hydro
        // -- Central Asia --
        "ca-kz" => 680.0, // Kazakhstan — coal
        "ca-uz" => 520.0, // Uzbekistan — gas
        "ca-ge" => 120.0, // Georgia — hydro
        "ca-mn" => 820.0, // Mongolia — coal
        // -- Latin America --
        "la-co" => 150.0,                // Colombia — hydro
        "la-pe" => 200.0,                // Peru — hydro + gas
        "la-ar" => 350.0,                // Argentina — gas + hydro
        "la-ec" => 170.0,                // Ecuador — hydro
        "la-py" => 10.0,                 // Paraguay — 100% hydro
        "la-uy" => 80.0,                 // Uruguay — wind + hydro
        "la-cr" => 20.0,                 // Costa Rica — 99% renewable
        "la-gt" => 350.0,                // Guatemala — hydro + geo
        "la-do" => 580.0,                // Dominican Republic — oil
        "la-ht" => 700.0,                // Haiti — diesel
        "la-jm" => 650.0,                // Jamaica — oil
        "la-tt" => 500.0,                // Trinidad — gas
        "la-mx" | "mx-central" => 420.0, // Mexico — gas + oil
        "la-cl" => 350.0,                // Chile — mix
        // -- Pacific --
        "pac-fj" => 350.0, // Fiji — hydro + diesel
        "pac-pg" => 450.0, // Papua New Guinea — diesel + hydro
        // -- Orbital --
        "orbital" => 0.0,
        // -- Fallback --
        _ => 400.0,
    };

    CarbonIntensity {
        region: region.to_string(),
        gco2_per_kwh,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        source: "default-estimate".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carbon_intensity_conversion() {
        let ci = CarbonIntensity {
            region: "us-east".into(),
            gco2_per_kwh: 400.0,
            timestamp: 0,
            source: "test".into(),
        };
        let gco2_per_j = ci.gco2_per_joule();
        assert!((gco2_per_j - 0.000_111_111).abs() < 0.000_001);
    }

    #[test]
    fn carbon_for_energy() {
        let ci = CarbonIntensity {
            region: "us-east".into(),
            gco2_per_kwh: 360.0,
            timestamp: 0,
            source: "test".into(),
        };
        let carbon = ci.carbon_for(Joules::new(10_000.0));
        assert!((carbon - 1.0).abs() < 0.01);
    }

    #[test]
    fn transfer_cost_ethernet() {
        let cost = TransferCost::estimate(100.0, ConnectionType::Ethernet);
        assert!((cost.as_f64() - 0.64).abs() < 0.01);
    }

    #[test]
    fn transfer_cost_ordering() {
        let eth = TransferCost::estimate(1.0, ConnectionType::Ethernet);
        let wifi = TransferCost::estimate(1.0, ConnectionType::Wifi);
        let cell = TransferCost::estimate(1.0, ConnectionType::Cellular);
        assert!(eth.as_f64() < wifi.as_f64());
        assert!(wifi.as_f64() < cell.as_f64());
    }

    #[test]
    fn default_regions() {
        let nordic = default_carbon_intensity("eu-north");
        let india = default_carbon_intensity("ap-south");
        assert!(nordic.gco2_per_kwh < india.gco2_per_kwh);
    }

    #[test]
    fn developing_nation_carbon_intensities() {
        // Africa
        let kenya = default_carbon_intensity("af-east-ke");
        assert!(kenya.gco2_per_kwh < 150.0, "Kenya should be very green");
        let nigeria = default_carbon_intensity("af-west-ng");
        assert!(nigeria.gco2_per_kwh > 300.0, "Nigeria uses gas");
        let sa = default_carbon_intensity("af-south-1");
        assert!(sa.gco2_per_kwh > 800.0, "South Africa is coal-heavy");

        // Latin America
        let paraguay = default_carbon_intensity("la-py");
        assert!(paraguay.gco2_per_kwh < 15.0, "Paraguay is 100% hydro");
        let haiti = default_carbon_intensity("la-ht");
        assert!(haiti.gco2_per_kwh > 600.0, "Haiti runs on diesel");

        // South Asia
        let nepal = default_carbon_intensity("sa-south-np");
        assert!(nepal.gco2_per_kwh < 50.0, "Nepal is nearly 100% hydro");

        // Central Asia
        let mongolia = default_carbon_intensity("ca-mn");
        assert!(mongolia.gco2_per_kwh > 700.0, "Mongolia is coal-heavy");
    }

    #[test]
    fn ethiopia_cheapest_and_cleanest() {
        let et = default_carbon_intensity("af-east-et");
        let us = default_carbon_intensity("us-east");
        assert!(et.gco2_per_kwh < us.gco2_per_kwh / 10.0);
    }
}
