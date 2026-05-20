use inv_core::energy::{EnergyBudget, Joules};

use crate::meter::{EnergyMeter, EnergyMeterError};

/// Runtime energy budget enforcer for workloads.
/// Wraps an EnergyMeter and tracks consumption against a budget.
pub struct BudgetEnforcer {
    budget: EnergyBudget,
    baseline_joules: Option<Joules>,
}

impl BudgetEnforcer {
    /// Create a new enforcer with the given budget limit.
    pub fn new(max_joules: Joules) -> Self {
        Self {
            budget: EnergyBudget::new(max_joules),
            baseline_joules: None,
        }
    }

    /// Start tracking: record the current meter reading as baseline.
    pub fn start(&mut self, meter: &dyn EnergyMeter) -> Result<(), EnergyMeterError> {
        let reading = meter.read()?;
        self.baseline_joules = Some(reading.joules);
        Ok(())
    }

    /// Check the current consumption against the budget.
    pub fn check(&mut self, meter: &dyn EnergyMeter) -> Result<&EnergyBudget, BudgetError> {
        let reading = meter.read().map_err(BudgetError::Meter)?;

        if let Some(baseline) = self.baseline_joules {
            let consumed = Joules::new(reading.joules.as_f64() - baseline.as_f64());
            self.budget = EnergyBudget::new(self.budget.max_joules);
            self.budget.consume(consumed);

            if self.budget.is_exceeded() {
                return Err(BudgetError::Exceeded {
                    consumed,
                    limit: self.budget.max_joules,
                });
            }
        }

        Ok(&self.budget)
    }

    /// Remaining energy in the budget.
    pub fn remaining(&self) -> Joules {
        self.budget.remaining()
    }

    /// Current utilization (0.0 to 1.0+).
    pub fn utilization(&self) -> f64 {
        self.budget.utilization()
    }

    /// Whether the budget has been exceeded.
    pub fn is_exceeded(&self) -> bool {
        self.budget.is_exceeded()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BudgetError {
    #[error("energy budget exceeded: consumed {consumed}, limit {limit}")]
    Exceeded { consumed: Joules, limit: Joules },
    #[error("meter error: {0}")]
    Meter(EnergyMeterError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meter::EstimationMeter;

    #[test]
    fn budget_enforcer_creation() {
        let enforcer = BudgetEnforcer::new(Joules::new(10.0));
        assert!(!enforcer.is_exceeded());
        assert!((enforcer.remaining().as_f64() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn budget_enforcer_start_and_check() {
        let meter = EstimationMeter::new(45.0);
        let mut enforcer = BudgetEnforcer::new(Joules::new(1000.0));
        enforcer.start(&meter).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(10));

        let budget = enforcer.check(&meter).unwrap();
        assert!(!budget.is_exceeded());
    }
}
