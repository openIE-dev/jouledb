//! RAPL (Running Average Power Limit) energy reader for Linux
//!
//! Reads energy counters from `/sys/class/powercap/intel-rapl/`

use crate::error::{Error, Result};
use std::fs;
use std::path::PathBuf;

/// RAPL energy reader
pub struct RAPLReader {
    package_energy_path: PathBuf,
    dram_energy_path: Option<PathBuf>,
    max_energy_range_uj: f64,
}

impl RAPLReader {
    /// Create a new RAPL reader
    pub fn new() -> Result<Self> {
        let base = PathBuf::from("/sys/class/powercap/intel-rapl/intel-rapl:0");

        if !base.exists() {
            return Err(Error::Unsupported(
                "RAPL interface not found. This may require root access or Intel CPU.".to_string(),
            ));
        }

        // Read max energy range for overflow handling
        let max_energy_path = base.join("max_energy_range_uj");
        let max_energy_str = fs::read_to_string(&max_energy_path).map_err(|_| {
            Error::Permission(
                "Cannot read RAPL max_energy_range. Try running with sudo.".to_string(),
            )
        })?;

        let max_energy_range_uj = max_energy_str
            .trim()
            .parse::<f64>()
            .map_err(|e| Error::Parse(format!("Failed to parse max_energy_range: {}", e)))?;

        // Package energy is always available
        let package_energy_path = base.join("energy_uj");

        // DRAM energy may or may not be available
        let dram_energy_path = base.join("intel-rapl:0:2/energy_uj");
        let dram_energy_path = if dram_energy_path.exists() {
            Some(dram_energy_path)
        } else {
            None
        };

        Ok(Self {
            package_energy_path,
            dram_energy_path,
            max_energy_range_uj,
        })
    }

    /// Read package energy in joules
    pub fn read_package_energy(&self) -> Result<f64> {
        let content = fs::read_to_string(&self.package_energy_path)?;
        let microjoules = content
            .trim()
            .parse::<f64>()
            .map_err(|e| Error::Parse(format!("Failed to parse energy: {}", e)))?;

        Ok(microjoules / 1_000_000.0)
    }

    /// Read DRAM energy in joules (if available)
    pub fn read_dram_energy(&self) -> Result<Option<f64>> {
        match &self.dram_energy_path {
            Some(path) => {
                let content = fs::read_to_string(path)?;
                let microjoules = content
                    .trim()
                    .parse::<f64>()
                    .map_err(|e| Error::Parse(format!("Failed to parse DRAM energy: {}", e)))?;

                Ok(Some(microjoules / 1_000_000.0))
            }
            None => Ok(None),
        }
    }

    /// Read total energy (package + DRAM if available)
    pub fn read_total_energy(&self) -> Result<f64> {
        let package = self.read_package_energy()?;
        let dram = self.read_dram_energy()?.unwrap_or(0.0);
        Ok(package + dram)
    }

    /// Get max energy range (for overflow detection)
    pub fn max_energy_range(&self) -> f64 {
        self.max_energy_range_uj / 1_000_000.0 // Convert to joules
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires Linux with Intel RAPL
    fn test_rapl_read() {
        let reader = RAPLReader::new().unwrap();
        let energy = reader.read_package_energy().unwrap();
        println!("Package energy: {:.6} J", energy);

        if let Ok(Some(dram)) = reader.read_dram_energy() {
            println!("DRAM energy: {:.6} J", dram);
        }

        assert!(energy > 0.0);
    }

    #[test]
    #[ignore] // Requires Linux with Intel RAPL
    fn test_energy_measurement() {
        let reader = RAPLReader::new().unwrap();
        let start = reader.read_total_energy().unwrap();

        // Do some work
        let mut sum = 0u64;
        for i in 0..10_000_000 {
            sum = sum.wrapping_add(i);
        }

        let end = reader.read_total_energy().unwrap();
        let consumed = end - start;

        println!(
            "Energy consumed: {:.6} J = {:.3} mJ",
            consumed,
            consumed * 1000.0
        );
        println!("Sum: {}", sum); // Prevent optimization

        assert!(consumed > 0.0);
        assert!(consumed < 1.0); // Should be less than 1 J
    }
}
