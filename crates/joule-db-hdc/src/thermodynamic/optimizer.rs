//! Thermodynamic optimizer implementation

use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use thiserror::Error;

/// Thermodynamic optimizer errors
#[derive(Error, Debug, Clone)]
pub enum ThermoError {
    /// No plans provided
    #[error("No plans to optimize")]
    NoPlans,

    /// Lock error
    #[error("Lock poisoned")]
    LockPoisoned,
}

/// Query execution plan
#[derive(Clone, Debug)]
pub struct QueryPlan {
    /// Estimated selectivity (0.0 to 1.0, lower = more selective)
    selectivity: f64,
    /// Number of join operations
    join_count: usize,
    /// Whether an index is available
    index_available: bool,
    /// Estimated cost (computed)
    estimated_cost: f64,
    /// Plan description
    description: String,
}

impl QueryPlan {
    /// Create new query plan
    pub fn new(
        selectivity: f64,
        join_count: usize,
        index_available: bool,
        description: &str,
    ) -> Self {
        let mut plan = Self {
            selectivity: selectivity.max(0.001).min(1.0),
            join_count,
            index_available,
            estimated_cost: 0.0,
            description: description.to_string(),
        };
        plan.estimated_cost = plan.compute_cost();
        plan
    }

    /// Compute cost based on plan parameters
    fn compute_cost(&self) -> f64 {
        let base_cost = 1.0 / self.selectivity;
        let join_cost = (self.join_count as f64 + 1.0).powi(2);
        let index_factor = if self.index_available { 0.1 } else { 1.0 };

        (base_cost + join_cost) * index_factor
    }

    /// Get selectivity
    pub fn selectivity(&self) -> f64 {
        self.selectivity
    }

    /// Get join count
    pub fn join_count(&self) -> usize {
        self.join_count
    }

    /// Get index availability
    pub fn index_available(&self) -> bool {
        self.index_available
    }

    /// Get estimated cost
    pub fn cost(&self) -> f64 {
        self.estimated_cost
    }

    /// Get description
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Optimizer statistics
#[derive(Debug, Clone)]
pub struct OptimizerStats {
    /// Current temperature
    pub temperature: f64,
    /// Number of optimizations performed
    pub optimizations: usize,
    /// Average energy (cost) seen
    pub average_energy: f64,
    /// Best energy (cost) seen
    pub best_energy: f64,
}

/// Thermodynamic query optimizer using simulated annealing
pub struct ThermodynamicOptimizer {
    /// Current temperature (controls exploration vs exploitation)
    temperature: f64,
    /// Cooling rate
    cooling_rate: f64,
    /// Minimum temperature
    min_temperature: f64,
    /// Energy history for analysis
    energy_history: Arc<RwLock<VecDeque<f64>>>,
    /// Best plan seen
    best_plan: Arc<RwLock<Option<(QueryPlan, f64)>>>,
    /// Random state
    random_state: u64,
    /// Optimization count
    optimization_count: Arc<RwLock<usize>>,
}

impl ThermodynamicOptimizer {
    /// Create new optimizer with default parameters
    pub fn new() -> Self {
        Self::with_params(1.0, 0.95, 0.01)
    }

    /// Create optimizer with custom parameters
    pub fn with_params(initial_temp: f64, cooling_rate: f64, min_temp: f64) -> Self {
        Self {
            temperature: initial_temp,
            cooling_rate,
            min_temperature: min_temp,
            energy_history: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
            best_plan: Arc::new(RwLock::new(None)),
            random_state: 12345,
            optimization_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Get current temperature
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Cool down (decrease temperature)
    pub fn cool_down(&mut self) {
        self.temperature = (self.temperature * self.cooling_rate).max(self.min_temperature);
    }

    /// Heat up (increase temperature)
    pub fn heat_up(&mut self, factor: f64) {
        self.temperature = (self.temperature * factor).min(10.0);
    }

    /// Reset temperature
    pub fn reset_temperature(&mut self) {
        self.temperature = 1.0;
    }

    /// Simple deterministic random
    fn next_random(&mut self) -> f64 {
        self.random_state = self
            .random_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.random_state as f64) / (u64::MAX as f64)
    }

    /// Compute energy (cost) for a query plan
    pub fn query_energy(selectivity: f64, join_count: usize, has_index: bool) -> f64 {
        QueryPlan::new(selectivity, join_count, has_index, "temp").cost()
    }

    /// Accept probability based on Boltzmann distribution
    fn acceptance_probability(&self, current_energy: f64, new_energy: f64) -> f64 {
        if new_energy < current_energy {
            1.0
        } else {
            let delta = new_energy - current_energy;
            (-delta / self.temperature).exp()
        }
    }

    /// Optimize a set of query plans, returning the best one
    pub fn optimize_plans(&mut self, plans: Vec<QueryPlan>) -> QueryPlan {
        if plans.is_empty() {
            return QueryPlan::new(1.0, 0, false, "empty");
        }

        *self.optimization_count.write().unwrap() += 1;

        let mut current = plans[0].clone();
        let mut current_energy = current.cost();

        // Record initial energy
        {
            let mut history = self.energy_history.write().unwrap();
            history.push_back(current_energy);
            if history.len() > 100 {
                history.pop_front();
            }
        }

        // Try each plan using simulated annealing
        for plan in plans.iter().skip(1) {
            let new_energy = plan.cost();

            let accept_prob = self.acceptance_probability(current_energy, new_energy);
            if self.next_random() < accept_prob {
                current = plan.clone();
                current_energy = new_energy;
            }
        }

        // Cool down after optimization
        self.cool_down();

        // Update best plan
        {
            let mut best = self.best_plan.write().unwrap();
            if best.is_none() || current_energy < best.as_ref().unwrap().1 {
                *best = Some((current.clone(), current_energy));
            }
        }

        current
    }

    /// Get the best plan seen so far
    pub fn best_plan(&self) -> Option<QueryPlan> {
        self.best_plan
            .read()
            .unwrap()
            .as_ref()
            .map(|(p, _)| p.clone())
    }

    /// Get statistics
    pub fn stats(&self) -> OptimizerStats {
        let history = self.energy_history.read().unwrap();
        let best = self.best_plan.read().unwrap();
        let count = *self.optimization_count.read().unwrap();

        let average_energy = if history.is_empty() {
            0.0
        } else {
            history.iter().sum::<f64>() / history.len() as f64
        };

        let best_energy = best.as_ref().map(|(_, e)| *e).unwrap_or(f64::INFINITY);

        OptimizerStats {
            temperature: self.temperature,
            optimizations: count,
            average_energy,
            best_energy,
        }
    }

    /// Clear history
    pub fn clear(&mut self) {
        self.energy_history.write().unwrap().clear();
        *self.best_plan.write().unwrap() = None;
        *self.optimization_count.write().unwrap() = 0;
        self.reset_temperature();
    }
}

impl Default for ThermodynamicOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_plan_cost() {
        let indexed = QueryPlan::new(0.1, 0, true, "indexed");
        let non_indexed = QueryPlan::new(0.1, 0, false, "non_indexed");

        // Indexed should be cheaper
        assert!(indexed.cost() < non_indexed.cost());
    }

    #[test]
    fn test_selectivity_affects_cost() {
        // Selectivity = fraction of rows selected
        // Lower selectivity = fewer rows = higher base_cost (1/selectivity)
        // But typically we want lower selectivity to mean fewer rows examined
        let selective = QueryPlan::new(0.1, 0, false, "selective");
        let non_selective = QueryPlan::new(0.9, 0, false, "non_selective");

        // In our cost model: base_cost = 1/selectivity
        // So 0.1 selectivity = 10 base_cost, 0.9 selectivity = 1.11 base_cost
        // This means more selective (fewer rows) actually has HIGHER base cost in our formula
        // This is inverted from typical behavior, but matches the original stdlib
        assert!(selective.cost() > non_selective.cost());
    }

    #[test]
    fn test_joins_affect_cost() {
        let no_joins = QueryPlan::new(0.5, 0, false, "no_joins");
        let many_joins = QueryPlan::new(0.5, 3, false, "many_joins");

        // More joins should be more expensive
        assert!(no_joins.cost() < many_joins.cost());
    }

    #[test]
    fn test_optimizer_creation() {
        let optimizer = ThermodynamicOptimizer::new();
        assert_eq!(optimizer.temperature(), 1.0);
    }

    #[test]
    fn test_cool_down() {
        let mut optimizer = ThermodynamicOptimizer::new();
        let initial_temp = optimizer.temperature();

        optimizer.cool_down();
        assert!(optimizer.temperature() < initial_temp);
    }

    #[test]
    fn test_heat_up() {
        let mut optimizer = ThermodynamicOptimizer::new();
        optimizer.cool_down();
        optimizer.cool_down();
        let cooled_temp = optimizer.temperature();

        optimizer.heat_up(2.0);
        assert!(optimizer.temperature() > cooled_temp);
    }

    #[test]
    fn test_optimize_plans() {
        let mut optimizer = ThermodynamicOptimizer::new();

        let plans = vec![
            QueryPlan::new(0.5, 2, false, "plan_a"),
            QueryPlan::new(0.1, 0, true, "plan_b"), // Best
            QueryPlan::new(0.8, 1, false, "plan_c"),
        ];

        // Run multiple times to let annealing converge
        for _ in 0..10 {
            optimizer.reset_temperature();
            optimizer.optimize_plans(plans.clone());
        }

        // Best plan should be tracked
        let best = optimizer.best_plan().unwrap();
        assert!(best.cost() < 5.0); // plan_b has low cost
    }

    #[test]
    fn test_acceptance_probability() {
        let optimizer = ThermodynamicOptimizer::new();

        // Better (lower) energy always accepted
        assert_eq!(optimizer.acceptance_probability(10.0, 5.0), 1.0);

        // Worse energy has probability < 1
        let prob = optimizer.acceptance_probability(5.0, 10.0);
        assert!(prob > 0.0 && prob < 1.0);
    }

    #[test]
    fn test_stats() {
        let mut optimizer = ThermodynamicOptimizer::new();

        let plans = vec![QueryPlan::new(0.5, 0, false, "plan")];

        optimizer.optimize_plans(plans);

        let stats = optimizer.stats();
        assert_eq!(stats.optimizations, 1);
        assert!(stats.average_energy > 0.0);
    }

    #[test]
    fn test_clear() {
        let mut optimizer = ThermodynamicOptimizer::new();

        let plans = vec![QueryPlan::new(0.5, 0, false, "plan")];

        optimizer.optimize_plans(plans);
        optimizer.cool_down();

        optimizer.clear();

        let stats = optimizer.stats();
        assert_eq!(stats.optimizations, 0);
        assert_eq!(optimizer.temperature(), 1.0);
    }
}
