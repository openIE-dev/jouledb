//! Energy policy engine.
//!
//! Dynamic quality and performance adjustment based on hardware metrics,
//! thermal state, battery level, and power consumption. Bridges the
//! `joule-db-energy` hardware monitor to application-level decisions.

use crate::hw::monitor::ThermalState;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

const DECISION_HISTORY_SIZE: usize = 10;

/// Hardware profile presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardwareProfile {
    /// Full performance, no throttling.
    Performance,
    /// Balanced power/performance (default).
    Balanced,
    /// Minimize energy usage.
    Efficiency,
    /// Battery saver — aggressive throttling.
    BatterySaver,
}

impl Default for HardwareProfile {
    fn default() -> Self {
        Self::Balanced
    }
}

/// A quality scaling factor (0.0 = minimum, 1.0 = maximum).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityScale {
    /// Render resolution multiplier (1.0 = native).
    pub render_scale: f64,
    /// AI inference quality (1.0 = full model, 0.5 = quantized/distilled).
    pub ai_quality: f64,
    /// Collaboration sync frequency (1.0 = real-time, 0.1 = 10s batched).
    pub sync_frequency: f64,
    /// Background task concurrency multiplier.
    pub background_concurrency: f64,
    /// Animation frame budget multiplier.
    pub animation_budget: f64,
}

impl Default for QualityScale {
    fn default() -> Self {
        Self {
            render_scale: 1.0,
            ai_quality: 1.0,
            sync_frequency: 1.0,
            background_concurrency: 1.0,
            animation_budget: 1.0,
        }
    }
}

/// A recorded policy decision for history/debugging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDecision {
    pub timestamp: u64,
    pub profile: HardwareProfile,
    pub quality: QualityScale,
    pub reason: String,
}

/// Energy policy engine.
///
/// Takes hardware state (thermal, battery, power, utilization) and produces
/// quality scaling decisions. Operates as a pure function with history.
pub struct EnergyPolicy {
    profile: HardwareProfile,
    auto_adapt: bool,
    quality: QualityScale,
    history: VecDeque<PolicyDecision>,
}

impl EnergyPolicy {
    pub fn new(profile: HardwareProfile) -> Self {
        Self {
            profile,
            auto_adapt: true,
            quality: QualityScale::default(),
            history: VecDeque::with_capacity(DECISION_HISTORY_SIZE),
        }
    }

    /// Update the policy based on current hardware state.
    /// Returns the overall quality multiplier (0.0 - 1.0).
    pub fn update(
        &mut self,
        thermal: ThermalState,
        battery_percent: Option<f64>,
        battery_charging: bool,
        gpu_utilization: f64,
        power_watts: f64,
    ) -> f64 {
        if !self.auto_adapt {
            return 1.0;
        }

        let mut reason = String::new();

        // Thermal response — most aggressive throttling.
        let thermal_factor = match thermal {
            ThermalState::Nominal => 1.0,
            ThermalState::Fair => 0.85,
            ThermalState::Serious => 0.6,
            ThermalState::Critical => 0.3,
        };
        if thermal_factor < 1.0 {
            reason.push_str(&format!("thermal:{thermal:?} "));
        }

        // Battery response.
        let battery_factor = match (battery_percent, battery_charging) {
            (_, true) => 1.0,
            (Some(pct), false) if pct > 50.0 => 1.0,
            (Some(pct), false) if pct > 20.0 => 0.8,
            (Some(pct), false) if pct > 10.0 => 0.5,
            (Some(_), false) => 0.3,
            (None, _) => 1.0, // Desktop, no battery.
        };
        if battery_factor < 1.0 {
            reason.push_str(&format!("battery:{:.0}% ", battery_percent.unwrap_or(0.0)));
        }

        // Auto-select profile based on factors.
        if self.auto_adapt {
            let combined = thermal_factor * battery_factor;
            self.profile = if combined >= 0.95 {
                HardwareProfile::Performance
            } else if combined >= 0.7 {
                HardwareProfile::Balanced
            } else if combined >= 0.4 {
                HardwareProfile::Efficiency
            } else {
                HardwareProfile::BatterySaver
            };
        }

        // Apply profile to quality scales.
        let profile_factor = match self.profile {
            HardwareProfile::Performance => 1.0,
            HardwareProfile::Balanced => 0.85,
            HardwareProfile::Efficiency => 0.6,
            HardwareProfile::BatterySaver => 0.35,
        };

        self.quality = QualityScale {
            render_scale: profile_factor,
            ai_quality: (profile_factor * 1.2).min(1.0), // Prefer keeping AI quality higher.
            sync_frequency: profile_factor,
            background_concurrency: (profile_factor * 0.8).max(0.1),
            animation_budget: profile_factor,
        };

        // Record decision.
        let decision = PolicyDecision {
            timestamp: unix_timestamp(),
            profile: self.profile,
            quality: self.quality.clone(),
            reason: if reason.is_empty() {
                "nominal".to_string()
            } else {
                reason.trim().to_string()
            },
        };

        if self.history.len() >= DECISION_HISTORY_SIZE {
            self.history.pop_front();
        }
        self.history.push_back(decision);

        thermal_factor * battery_factor * profile_factor
    }

    /// Update policy directly from a `joule-db-energy` EnergySnapshot.
    pub fn update_from_snapshot(&mut self, snap: &crate::EnergySnapshot) -> f64 {
        self.update(
            snap.thermal_state,
            snap.battery_percent,
            snap.battery_charging,
            snap.gpu_utilization,
            snap.power_watts,
        )
    }

    pub fn current_profile(&self) -> HardwareProfile {
        self.profile
    }

    pub fn force_profile(&mut self, profile: HardwareProfile) {
        self.profile = profile;
        self.auto_adapt = false;
    }

    pub fn set_auto_adapt(&mut self, enabled: bool) {
        self.auto_adapt = enabled;
    }

    pub fn auto_adapt(&self) -> bool {
        self.auto_adapt
    }

    pub fn quality(&self) -> &QualityScale {
        &self.quality
    }

    pub fn history(&self) -> &VecDeque<PolicyDecision> {
        &self.history
    }
}

impl Default for EnergyPolicy {
    fn default() -> Self {
        Self::new(HardwareProfile::Balanced)
    }
}

fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_policy() {
        let policy = EnergyPolicy::default();
        assert_eq!(policy.current_profile(), HardwareProfile::Balanced);
        assert!(policy.auto_adapt());
    }

    #[test]
    fn test_thermal_throttling() {
        let mut policy = EnergyPolicy::default();
        let factor = policy.update(ThermalState::Critical, None, false, 0.0, 15.0);
        assert!(factor < 0.5);
        assert_eq!(policy.current_profile(), HardwareProfile::BatterySaver);
    }

    #[test]
    fn test_battery_throttling() {
        let mut policy = EnergyPolicy::default();
        let factor = policy.update(ThermalState::Nominal, Some(8.0), false, 0.0, 5.0);
        assert!(factor < 0.5);
    }

    #[test]
    fn test_plugged_in_no_throttle() {
        let mut policy = EnergyPolicy::default();
        let factor = policy.update(ThermalState::Nominal, Some(8.0), true, 0.0, 15.0);
        assert!(factor > 0.8);
    }

    #[test]
    fn test_force_profile_disables_auto() {
        let mut policy = EnergyPolicy::default();
        policy.force_profile(HardwareProfile::Performance);
        assert!(!policy.auto_adapt());
        assert_eq!(policy.current_profile(), HardwareProfile::Performance);
    }
}
