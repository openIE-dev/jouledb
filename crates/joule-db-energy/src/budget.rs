//! Per-query energy budget enforcement.
//!
//! Tracks cumulative energy consumption for a query/transaction and
//! returns an error if the budget is exceeded. This is the integration
//! point for the Joule programming language's compile-time energy budgets.
//!
//! ```ignore
//! // Future Joule integration:
//! #[energy_budget(max_joules = 0.001)]
//! fn handle_query(db: &JouleDB, query: &str) -> Result<()> { ... }
//! ```

use std::sync::atomic::{AtomicU64, Ordering};

/// Error returned when an energy budget is exceeded.
#[derive(Debug, Clone)]
pub struct EnergyBudgetError {
    pub consumed_joules: f64,
    pub budget_joules: f64,
}

impl std::fmt::Display for EnergyBudgetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "energy budget exceeded: {:.6}J consumed, limit was {:.6}J",
            self.consumed_joules, self.budget_joules
        )
    }
}

impl std::error::Error for EnergyBudgetError {}

/// Per-query energy budget tracker.
///
/// Thread-safe via atomic operations. Check at operation boundaries
/// (not mid-computation) for minimal overhead.
pub struct EnergyBudget {
    max_joules: f64,
    /// Consumed joules stored as u64 bits (using f64::to_bits/from_bits).
    consumed_bits: AtomicU64,
}

impl EnergyBudget {
    /// Create a budget with a maximum energy limit in joules.
    pub fn new(max_joules: f64) -> Self {
        Self {
            max_joules,
            consumed_bits: AtomicU64::new(0.0_f64.to_bits()),
        }
    }

    /// Create an unlimited budget (never exceeds).
    pub fn unlimited() -> Self {
        Self::new(f64::INFINITY)
    }

    /// Record energy consumed. Returns Err if budget is exceeded.
    pub fn record(&self, joules: f64) -> Result<(), EnergyBudgetError> {
        // Atomic CAS loop to update consumed
        loop {
            let current_bits = self.consumed_bits.load(Ordering::Relaxed);
            let current = f64::from_bits(current_bits);
            let new_total = current + joules;

            if new_total > self.max_joules {
                return Err(EnergyBudgetError {
                    consumed_joules: new_total,
                    budget_joules: self.max_joules,
                });
            }

            match self.consumed_bits.compare_exchange_weak(
                current_bits,
                new_total.to_bits(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(()),
                Err(_) => continue, // Retry on contention
            }
        }
    }

    /// Check if an additional amount would exceed the budget (precheck).
    pub fn would_exceed(&self, additional_joules: f64) -> bool {
        self.consumed() + additional_joules > self.max_joules
    }

    /// Get remaining budget in joules.
    pub fn remaining(&self) -> f64 {
        (self.max_joules - self.consumed()).max(0.0)
    }

    /// Get total consumed energy in joules.
    pub fn consumed(&self) -> f64 {
        f64::from_bits(self.consumed_bits.load(Ordering::Relaxed))
    }

    /// Get the maximum budget in joules.
    pub fn max_joules(&self) -> f64 {
        self.max_joules
    }

    /// Check if this is an unlimited budget.
    pub fn is_unlimited(&self) -> bool {
        self.max_joules.is_infinite()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_within_limit() {
        let budget = EnergyBudget::new(1.0);
        assert!(budget.record(0.3).is_ok());
        assert!(budget.record(0.3).is_ok());
        assert!(budget.record(0.3).is_ok());
        assert!((budget.consumed() - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_budget_exceeded() {
        let budget = EnergyBudget::new(1.0);
        assert!(budget.record(0.5).is_ok());
        let err = budget.record(0.6).unwrap_err();
        assert!(err.consumed_joules > 1.0);
        assert_eq!(err.budget_joules, 1.0);
    }

    #[test]
    fn test_budget_remaining() {
        let budget = EnergyBudget::new(1.0);
        budget.record(0.3).unwrap();
        assert!((budget.remaining() - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_budget_would_exceed() {
        let budget = EnergyBudget::new(1.0);
        budget.record(0.8).unwrap();
        assert!(!budget.would_exceed(0.1));
        assert!(budget.would_exceed(0.3));
    }

    #[test]
    fn test_unlimited_budget() {
        let budget = EnergyBudget::unlimited();
        assert!(budget.is_unlimited());
        assert!(budget.record(1_000_000.0).is_ok());
        assert!(budget.record(1_000_000.0).is_ok());
    }

    #[test]
    fn test_budget_error_display() {
        let err = EnergyBudgetError {
            consumed_joules: 1.5,
            budget_joules: 1.0,
        };
        let msg = err.to_string();
        assert!(msg.contains("1.5"));
        assert!(msg.contains("1.0"));
    }

    #[test]
    fn test_concurrent_budget_recording() {
        use std::sync::Arc;
        let budget = Arc::new(EnergyBudget::new(100.0));
        let mut handles = Vec::new();

        for _ in 0..10 {
            let b = budget.clone();
            handles.push(std::thread::spawn(move || {
                for _ in 0..100 {
                    let _ = b.record(0.01);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // 10 threads × 100 records × 0.01J = 10.0J
        assert!((budget.consumed() - 10.0).abs() < 0.1);
    }
}
