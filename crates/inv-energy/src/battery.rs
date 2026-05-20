use inv_core::energy::EnergySource;

/// Battery state for mobile/laptop devices.
#[derive(Debug, Clone, Copy)]
pub struct BatteryState {
    /// Battery percentage (0-100).
    pub percentage: u8,
    /// Whether the device is charging.
    pub charging: bool,
    /// Energy source.
    pub source: EnergySource,
}

impl BatteryState {
    /// Read the current battery state. Platform-specific.
    pub fn read() -> Option<Self> {
        #[cfg(target_os = "macos")]
        {
            read_macos_battery()
        }
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
    }
}

#[cfg(target_os = "macos")]
fn read_macos_battery() -> Option<BatteryState> {
    let output = std::process::Command::new("pmset")
        .args(["-g", "batt"])
        .output()
        .ok()?;

    let stdout = String::from_utf8_lossy(&output.stdout);

    let percentage = stdout
        .lines()
        .find(|l| l.contains("InternalBattery"))
        .and_then(|l| {
            l.split('\t')
                .nth(1)?
                .split('%')
                .next()?
                .trim()
                .parse::<u8>()
                .ok()
        })?;

    let charging = stdout.contains("AC Power");
    let source = if charging {
        EnergySource::WallPower
    } else {
        EnergySource::Battery
    };

    Some(BatteryState {
        percentage,
        charging,
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn battery_state_read() {
        let state = BatteryState::read();
        if let Some(s) = state {
            assert!(s.percentage <= 100);
        }
    }
}
