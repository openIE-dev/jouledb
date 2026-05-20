//! Cross-platform energy reader abstraction

use crate::error::{Error, Result};

/// Platform-independent energy reader trait
pub trait EnergyReader: Send + Sync {
    /// Read current energy in joules
    fn read_energy(&self) -> Result<f64>;

    /// Read current power in watts (optional)
    fn read_power(&self) -> Result<f64> {
        Err(Error::Unsupported(
            "Power reading not supported".to_string(),
        ))
    }

    /// Read current temperature in Celsius (optional)
    fn read_temperature(&self) -> Result<f64> {
        Err(Error::Unsupported(
            "Temperature reading not supported".to_string(),
        ))
    }
}

/// Linux RAPL-based energy reader
#[cfg(target_os = "linux")]
mod linux_impl {
    use super::*;
    use crate::rapl::RAPLReader;
    use crate::thermal::ThermalMonitor;

    pub struct LinuxEnergyReader {
        rapl: RAPLReader,
        thermal: Option<ThermalMonitor>,
    }

    impl LinuxEnergyReader {
        pub fn new() -> Result<Self> {
            let rapl = RAPLReader::new()?;
            let thermal = ThermalMonitor::new().ok(); // Thermal is optional

            Ok(Self { rapl, thermal })
        }
    }

    impl EnergyReader for LinuxEnergyReader {
        fn read_energy(&self) -> Result<f64> {
            self.rapl.read_total_energy()
        }

        fn read_temperature(&self) -> Result<f64> {
            match &self.thermal {
                Some(monitor) => monitor.read_temperature(),
                None => Err(Error::Unsupported(
                    "Thermal monitoring not available".to_string(),
                )),
            }
        }
    }
}

/// macOS IOKit-based energy reader for Apple Silicon
#[cfg(target_os = "macos")]
mod macos_impl {
    use super::*;
    use crate::macos::MacOSEnergyMonitor;

    pub struct MacOSEnergyReader {
        monitor: MacOSEnergyMonitor,
    }

    impl MacOSEnergyReader {
        pub fn new() -> Result<Self> {
            let monitor = MacOSEnergyMonitor::new()?;
            Ok(Self { monitor })
        }
    }

    impl EnergyReader for MacOSEnergyReader {
        fn read_energy(&self) -> Result<f64> {
            self.monitor.read_energy()
        }

        fn read_power(&self) -> Result<f64> {
            self.monitor.read_power()
        }

        fn read_temperature(&self) -> Result<f64> {
            self.monitor.read_temperature()
        }
    }
}

/// Windows EMI-based energy reader
#[cfg(target_os = "windows")]
mod windows_impl {
    use super::*;
    use crate::windows::WindowsEnergyMonitor;

    pub struct WindowsEnergyReader {
        monitor: WindowsEnergyMonitor,
    }

    impl WindowsEnergyReader {
        pub fn new() -> Result<Self> {
            let monitor = WindowsEnergyMonitor::new()?;
            Ok(Self { monitor })
        }
    }

    impl EnergyReader for WindowsEnergyReader {
        fn read_energy(&self) -> Result<f64> {
            self.monitor.read_energy()
        }

        fn read_power(&self) -> Result<f64> {
            self.monitor.read_power()
        }

        fn read_temperature(&self) -> Result<f64> {
            self.monitor.read_temperature()
        }
    }
}

/// Create a platform-specific energy reader
pub fn create_reader() -> Result<Box<dyn EnergyReader>> {
    #[cfg(target_os = "linux")]
    {
        Ok(Box::new(linux_impl::LinuxEnergyReader::new()?))
    }

    #[cfg(target_os = "macos")]
    {
        Ok(Box::new(macos_impl::MacOSEnergyReader::new()?))
    }

    #[cfg(target_os = "windows")]
    {
        Ok(Box::new(windows_impl::WindowsEnergyReader::new()?))
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        Err(Error::Unsupported(
            "Platform not supported for energy monitoring".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "linux")]
    #[ignore] // Requires Linux with RAPL
    fn test_create_reader_linux() {
        let reader = create_reader().unwrap();
        let energy = reader.read_energy().unwrap();
        println!("Energy: {:.6} J", energy);
        assert!(energy > 0.0);
    }

    #[test]
    #[cfg(target_os = "macos")]
    #[ignore] // Requires Apple Silicon Mac
    fn test_create_reader_macos() {
        let reader = create_reader().unwrap();
        let energy = reader.read_energy().unwrap();
        println!("Energy: {:.6} J", energy);

        // Wait a bit then read power
        std::thread::sleep(std::time::Duration::from_millis(100));
        let power = reader.read_power().unwrap();
        println!("Power: {:.2} W", power);
        assert!(power > 0.0 && power < 200.0);
    }

    #[test]
    #[cfg(target_os = "windows")]
    #[ignore] // Requires Windows with EMI-compatible hardware
    fn test_create_reader_windows() {
        let reader = create_reader().unwrap();
        let energy = reader.read_energy().unwrap();
        println!("Energy: {:.6} J", energy);

        // Wait a bit then read power
        std::thread::sleep(std::time::Duration::from_millis(100));
        let power = reader.read_power().unwrap();
        println!("Power: {:.2} W", power);
        assert!(power > 0.0 && power < 500.0);
    }
}
