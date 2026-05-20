//! Thermodynamic Query Optimizer
//!
//! Self-tuning query execution using simulated annealing principles.
//! Queries "flow" to optimal execution paths like heat.
//!
//! # Overview
//!
//! The optimizer uses simulated annealing to explore the space of
//! query execution plans. High "temperature" means more exploration,
//! low temperature means more exploitation of known good plans.
//!
//! # Example
//!
//! ```rust
//! use joule_db_hdc::thermodynamic::{ThermodynamicOptimizer, QueryPlan};
//!
//! let mut optimizer = ThermodynamicOptimizer::new();
//!
//! // Create candidate plans
//! let plans = vec![
//!     QueryPlan::new(0.1, 2, true, "indexed_join"),
//!     QueryPlan::new(0.5, 0, false, "full_scan"),
//! ];
//!
//! // Find optimal plan
//! let best = optimizer.optimize_plans(plans);
//! println!("Best plan: {} with cost {}", best.description(), best.cost());
//! ```

mod optimizer;

pub use optimizer::{OptimizerStats, QueryPlan, ThermoError, ThermodynamicOptimizer};
