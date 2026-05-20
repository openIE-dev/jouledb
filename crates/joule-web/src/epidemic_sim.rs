//! Epidemic simulation — SIR/SEIR compartmental models.
//!
//! Replaces epimodel.js / EpiJS / compartmental-models with pure Rust.
//! Supports SIR and SEIR compartmental models, transmission rate (beta),
//! recovery rate (gamma), incubation rate (sigma), basic reproduction
//! number (R0), daily stepping, peak infection tracking, and herd
//! immunity threshold estimation.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Domain errors for epidemic simulation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpidemicError {
    /// Invalid rate (must be non-negative).
    InvalidRate(String),
    /// Population must be positive.
    ZeroPopulation,
    /// Invalid initial conditions.
    InvalidInitial(String),
    /// Model not configured.
    NotConfigured(String),
}

impl fmt::Display for EpidemicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRate(r) => write!(f, "invalid rate: {r}"),
            Self::ZeroPopulation => write!(f, "population must be > 0"),
            Self::InvalidInitial(msg) => write!(f, "invalid initial conditions: {msg}"),
            Self::NotConfigured(msg) => write!(f, "not configured: {msg}"),
        }
    }
}

impl std::error::Error for EpidemicError {}

// ── Model Type ──────────────────────────────────────────────────

/// Type of compartmental model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// Susceptible-Infected-Recovered.
    SIR,
    /// Susceptible-Exposed-Infected-Recovered.
    SEIR,
}

// ── State Snapshot ──────────────────────────────────────────────

/// A snapshot of the epidemic state at a given time step.
#[derive(Debug, Clone)]
pub struct EpidemicState {
    pub day: u64,
    pub susceptible: f64,
    pub exposed: f64,
    pub infected: f64,
    pub recovered: f64,
    pub new_infections: f64,
    pub new_recoveries: f64,
    pub effective_r: f64,
}

impl EpidemicState {
    /// Total population (should remain constant).
    pub fn total(&self) -> f64 {
        self.susceptible + self.exposed + self.infected + self.recovered
    }

    /// Fraction of population infected.
    pub fn infection_rate(&self) -> f64 {
        let total = self.total();
        if total == 0.0 { 0.0 } else { self.infected / total }
    }

    /// Fraction of population recovered.
    pub fn recovery_rate(&self) -> f64 {
        let total = self.total();
        if total == 0.0 { 0.0 } else { self.recovered / total }
    }
}

// ── Parameters ──────────────────────────────────────────────────

/// Epidemic model parameters.
#[derive(Debug, Clone)]
pub struct EpidemicParams {
    /// Transmission rate (contacts * probability of transmission).
    pub beta: f64,
    /// Recovery rate (1 / duration of infection).
    pub gamma: f64,
    /// Incubation rate (1 / incubation period). Only for SEIR.
    pub sigma: f64,
    /// Total population.
    pub population: f64,
    /// Time step (fraction of a day, typically 1.0).
    pub dt: f64,
}

impl EpidemicParams {
    /// Create SIR parameters.
    pub fn sir(beta: f64, gamma: f64, population: f64) -> Result<Self, EpidemicError> {
        if beta < 0.0 {
            return Err(EpidemicError::InvalidRate(format!("beta={beta}")));
        }
        if gamma < 0.0 {
            return Err(EpidemicError::InvalidRate(format!("gamma={gamma}")));
        }
        if population <= 0.0 {
            return Err(EpidemicError::ZeroPopulation);
        }
        Ok(Self { beta, gamma, sigma: 0.0, population, dt: 1.0 })
    }

    /// Create SEIR parameters.
    pub fn seir(beta: f64, gamma: f64, sigma: f64, population: f64) -> Result<Self, EpidemicError> {
        if beta < 0.0 {
            return Err(EpidemicError::InvalidRate(format!("beta={beta}")));
        }
        if gamma < 0.0 {
            return Err(EpidemicError::InvalidRate(format!("gamma={gamma}")));
        }
        if sigma < 0.0 {
            return Err(EpidemicError::InvalidRate(format!("sigma={sigma}")));
        }
        if population <= 0.0 {
            return Err(EpidemicError::ZeroPopulation);
        }
        Ok(Self { beta, gamma, sigma, population, dt: 1.0 })
    }

    /// Set the time step.
    pub fn with_dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    /// Basic reproduction number R0 = beta / gamma.
    pub fn r0(&self) -> f64 {
        if self.gamma == 0.0 { f64::INFINITY } else { self.beta / self.gamma }
    }

    /// Herd immunity threshold: 1 - 1/R0.
    pub fn herd_immunity_threshold(&self) -> f64 {
        let r0 = self.r0();
        if r0 <= 1.0 { 0.0 } else { 1.0 - 1.0 / r0 }
    }
}

// ── Epidemic Simulator ──────────────────────────────────────────

/// The epidemic simulator.
#[derive(Debug, Clone)]
pub struct EpidemicSim {
    model: ModelType,
    params: EpidemicParams,
    susceptible: f64,
    exposed: f64,
    infected: f64,
    recovered: f64,
    day: u64,
    history: Vec<EpidemicState>,
    peak_infected: f64,
    peak_day: u64,
    cumulative_infections: f64,
}

impl EpidemicSim {
    /// Create a new SIR simulation.
    pub fn sir(params: EpidemicParams, initial_infected: f64) -> Result<Self, EpidemicError> {
        if initial_infected < 0.0 || initial_infected > params.population {
            return Err(EpidemicError::InvalidInitial(
                format!("infected={initial_infected}, pop={}", params.population),
            ));
        }
        let s = params.population - initial_infected;
        Ok(Self {
            model: ModelType::SIR,
            params,
            susceptible: s,
            exposed: 0.0,
            infected: initial_infected,
            recovered: 0.0,
            day: 0,
            history: Vec::new(),
            peak_infected: initial_infected,
            peak_day: 0,
            cumulative_infections: initial_infected,
        })
    }

    /// Create a new SEIR simulation.
    pub fn seir(
        params: EpidemicParams,
        initial_exposed: f64,
        initial_infected: f64,
    ) -> Result<Self, EpidemicError> {
        let total_initial = initial_exposed + initial_infected;
        if total_initial > params.population || initial_exposed < 0.0 || initial_infected < 0.0 {
            return Err(EpidemicError::InvalidInitial(
                format!("exposed={initial_exposed}, infected={initial_infected}, pop={}", params.population),
            ));
        }
        let s = params.population - total_initial;
        Ok(Self {
            model: ModelType::SEIR,
            params,
            susceptible: s,
            exposed: initial_exposed,
            infected: initial_infected,
            recovered: 0.0,
            day: 0,
            history: Vec::new(),
            peak_infected: initial_infected,
            peak_day: 0,
            cumulative_infections: initial_infected,
        })
    }

    /// Model type.
    pub fn model_type(&self) -> ModelType { self.model }

    /// Current day.
    pub fn day(&self) -> u64 { self.day }

    /// Current compartment values.
    pub fn susceptible(&self) -> f64 { self.susceptible }
    pub fn exposed(&self) -> f64 { self.exposed }
    pub fn infected(&self) -> f64 { self.infected }
    pub fn recovered(&self) -> f64 { self.recovered }

    /// Peak number of infected individuals.
    pub fn peak_infected(&self) -> f64 { self.peak_infected }

    /// Day of peak infection.
    pub fn peak_day(&self) -> u64 { self.peak_day }

    /// Cumulative infections.
    pub fn cumulative_infections(&self) -> f64 { self.cumulative_infections }

    /// History of states.
    pub fn history(&self) -> &[EpidemicState] { &self.history }

    /// Current effective reproduction number: R_eff = R0 * (S / N).
    pub fn effective_r(&self) -> f64 {
        let n = self.params.population;
        if n == 0.0 { 0.0 } else { self.params.r0() * (self.susceptible / n) }
    }

    /// R0.
    pub fn r0(&self) -> f64 { self.params.r0() }

    /// Herd immunity threshold.
    pub fn herd_immunity_threshold(&self) -> f64 { self.params.herd_immunity_threshold() }

    /// Record the current state.
    fn record(&mut self, new_infections: f64, new_recoveries: f64) {
        self.history.push(EpidemicState {
            day: self.day,
            susceptible: self.susceptible,
            exposed: self.exposed,
            infected: self.infected,
            recovered: self.recovered,
            new_infections,
            new_recoveries,
            effective_r: self.effective_r(),
        });
    }

    /// Advance one day using Euler's method.
    pub fn step(&mut self) {
        let dt = self.params.dt;
        let beta = self.params.beta;
        let gamma = self.params.gamma;
        let n = self.params.population;

        match self.model {
            ModelType::SIR => {
                let new_infections = beta * self.susceptible * self.infected / n * dt;
                let new_recoveries = gamma * self.infected * dt;

                self.susceptible -= new_infections;
                self.infected += new_infections - new_recoveries;
                self.recovered += new_recoveries;

                // Clamp to non-negative.
                self.susceptible = self.susceptible.max(0.0);
                self.infected = self.infected.max(0.0);
                self.recovered = self.recovered.max(0.0);

                self.cumulative_infections += new_infections;

                if self.infected > self.peak_infected {
                    self.peak_infected = self.infected;
                    self.peak_day = self.day;
                }

                self.day += 1;
                self.record(new_infections, new_recoveries);
            }
            ModelType::SEIR => {
                let sigma = self.params.sigma;
                let new_exposed = beta * self.susceptible * self.infected / n * dt;
                let new_infected = sigma * self.exposed * dt;
                let new_recoveries = gamma * self.infected * dt;

                self.susceptible -= new_exposed;
                self.exposed += new_exposed - new_infected;
                self.infected += new_infected - new_recoveries;
                self.recovered += new_recoveries;

                self.susceptible = self.susceptible.max(0.0);
                self.exposed = self.exposed.max(0.0);
                self.infected = self.infected.max(0.0);
                self.recovered = self.recovered.max(0.0);

                self.cumulative_infections += new_infected;

                if self.infected > self.peak_infected {
                    self.peak_infected = self.infected;
                    self.peak_day = self.day;
                }

                self.day += 1;
                self.record(new_exposed, new_recoveries);
            }
        }
    }

    /// Run for a number of days.
    pub fn run(&mut self, days: u64) {
        for _ in 0..days {
            self.step();
        }
    }

    /// Check if the epidemic is over (infected < threshold).
    pub fn is_over(&self, threshold: f64) -> bool {
        self.infected < threshold
    }

    /// Run until the epidemic is over or max_days reached.
    pub fn run_until_over(&mut self, threshold: f64, max_days: u64) -> u64 {
        let mut days_run = 0u64;
        while !self.is_over(threshold) && days_run < max_days {
            self.step();
            days_run += 1;
        }
        days_run
    }

    /// Attack rate: fraction of population that was ever infected.
    pub fn attack_rate(&self) -> f64 {
        self.cumulative_infections / self.params.population
    }

    /// Current state snapshot.
    pub fn current_state(&self) -> EpidemicState {
        EpidemicState {
            day: self.day,
            susceptible: self.susceptible,
            exposed: self.exposed,
            infected: self.infected,
            recovered: self.recovered,
            new_infections: 0.0,
            new_recoveries: 0.0,
            effective_r: self.effective_r(),
        }
    }

    /// Population conservation check.
    pub fn population_check(&self) -> f64 {
        (self.susceptible + self.exposed + self.infected + self.recovered - self.params.population).abs()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sir_params() -> EpidemicParams {
        EpidemicParams::sir(0.3, 0.1, 10_000.0).unwrap()
    }

    fn seir_params() -> EpidemicParams {
        EpidemicParams::seir(0.3, 0.1, 0.2, 10_000.0).unwrap()
    }

    #[test]
    fn test_sir_creation() {
        let sim = EpidemicSim::sir(sir_params(), 10.0).unwrap();
        assert_eq!(sim.model_type(), ModelType::SIR);
        assert!((sim.susceptible() - 9990.0).abs() < 1e-10);
        assert!((sim.infected() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_seir_creation() {
        let sim = EpidemicSim::seir(seir_params(), 5.0, 10.0).unwrap();
        assert_eq!(sim.model_type(), ModelType::SEIR);
        assert!((sim.exposed() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_invalid_initial() {
        assert!(EpidemicSim::sir(sir_params(), -1.0).is_err());
        assert!(EpidemicSim::sir(sir_params(), 20_000.0).is_err());
    }

    #[test]
    fn test_invalid_rate() {
        assert!(EpidemicParams::sir(-1.0, 0.1, 10_000.0).is_err());
        assert!(EpidemicParams::sir(0.3, -0.1, 10_000.0).is_err());
    }

    #[test]
    fn test_zero_population() {
        assert!(EpidemicParams::sir(0.3, 0.1, 0.0).is_err());
    }

    #[test]
    fn test_r0() {
        let params = sir_params();
        assert!((params.r0() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_herd_immunity() {
        let params = sir_params();
        let hit = params.herd_immunity_threshold();
        // R0 = 3, HIT = 1 - 1/3 = 2/3.
        assert!((hit - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_herd_immunity_low_r0() {
        let params = EpidemicParams::sir(0.05, 0.1, 10_000.0).unwrap();
        assert_eq!(params.herd_immunity_threshold(), 0.0);
    }

    #[test]
    fn test_sir_step() {
        let mut sim = EpidemicSim::sir(sir_params(), 10.0).unwrap();
        sim.step();
        assert_eq!(sim.day(), 1);
        assert!(sim.infected() > 0.0);
    }

    #[test]
    fn test_sir_population_conserved() {
        let mut sim = EpidemicSim::sir(sir_params(), 100.0).unwrap();
        sim.run(100);
        let err = sim.population_check();
        assert!(err < 1.0, "population drift was {err}");
    }

    #[test]
    fn test_seir_population_conserved() {
        let mut sim = EpidemicSim::seir(seir_params(), 10.0, 5.0).unwrap();
        sim.run(100);
        let err = sim.population_check();
        assert!(err < 1.0, "population drift was {err}");
    }

    #[test]
    fn test_epidemic_rises_and_falls() {
        let mut sim = EpidemicSim::sir(sir_params(), 10.0).unwrap();
        sim.run(200);
        // Infections should rise to a peak and then fall.
        assert!(sim.peak_infected() > 10.0);
        assert!(sim.infected() < sim.peak_infected());
    }

    #[test]
    fn test_peak_day_tracked() {
        let mut sim = EpidemicSim::sir(sir_params(), 10.0).unwrap();
        sim.run(200);
        assert!(sim.peak_day() > 0);
        assert!(sim.peak_day() < 200);
    }

    #[test]
    fn test_effective_r_decreases() {
        let mut sim = EpidemicSim::sir(sir_params(), 100.0).unwrap();
        let r_initial = sim.effective_r();
        sim.run(50);
        let r_after = sim.effective_r();
        assert!(r_after < r_initial, "R_eff should decrease: {r_after} vs {r_initial}");
    }

    #[test]
    fn test_history_length() {
        let mut sim = EpidemicSim::sir(sir_params(), 10.0).unwrap();
        sim.run(30);
        assert_eq!(sim.history().len(), 30);
    }

    #[test]
    fn test_is_over() {
        let mut sim = EpidemicSim::sir(sir_params(), 10.0).unwrap();
        sim.run(500);
        assert!(sim.is_over(0.5));
    }

    #[test]
    fn test_run_until_over() {
        let mut sim = EpidemicSim::sir(sir_params(), 100.0).unwrap();
        let days = sim.run_until_over(0.5, 1000);
        assert!(days < 1000);
        assert!(sim.infected() < 0.5);
    }

    #[test]
    fn test_attack_rate() {
        let mut sim = EpidemicSim::sir(sir_params(), 100.0).unwrap();
        sim.run(500);
        let ar = sim.attack_rate();
        assert!(ar > 0.0 && ar <= 1.0, "attack rate was {ar}");
    }

    #[test]
    fn test_cumulative_infections() {
        let mut sim = EpidemicSim::sir(sir_params(), 10.0).unwrap();
        sim.run(100);
        assert!(sim.cumulative_infections() > 10.0);
    }

    #[test]
    fn test_current_state() {
        let mut sim = EpidemicSim::sir(sir_params(), 10.0).unwrap();
        sim.run(5);
        let state = sim.current_state();
        assert_eq!(state.day, 5);
        assert!(state.total() > 0.0);
    }

    #[test]
    fn test_epidemic_state_rates() {
        let state = EpidemicState {
            day: 0,
            susceptible: 9000.0,
            exposed: 0.0,
            infected: 500.0,
            recovered: 500.0,
            new_infections: 0.0,
            new_recoveries: 0.0,
            effective_r: 0.0,
        };
        assert!((state.infection_rate() - 0.05).abs() < 1e-10);
        assert!((state.recovery_rate() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_seir_exposed_transition() {
        let mut sim = EpidemicSim::seir(seir_params(), 100.0, 0.0).unwrap();
        // Initially all exposed, none infected.
        sim.step();
        // Some exposed should move to infected.
        assert!(sim.infected() > 0.0, "exposed should transition to infected");
    }

    #[test]
    fn test_seir_delay() {
        // SEIR should have a delayed peak compared to SIR.
        let sir_p = EpidemicParams::sir(0.3, 0.1, 10_000.0).unwrap();
        let mut sir_sim = EpidemicSim::sir(sir_p, 10.0).unwrap();
        sir_sim.run(300);

        let seir_p = EpidemicParams::seir(0.3, 0.1, 0.2, 10_000.0).unwrap();
        let mut seir_sim = EpidemicSim::seir(seir_p, 0.0, 10.0).unwrap();
        seir_sim.run(300);

        // SEIR peak should occur later.
        assert!(seir_sim.peak_day() >= sir_sim.peak_day(),
            "SEIR peak day {} should be >= SIR peak day {}",
            seir_sim.peak_day(), sir_sim.peak_day());
    }

    #[test]
    fn test_no_outbreak_low_r0() {
        let params = EpidemicParams::sir(0.05, 0.1, 10_000.0).unwrap();
        let mut sim = EpidemicSim::sir(params, 10.0).unwrap();
        sim.run(100);
        // R0 = 0.5 < 1, so epidemic should die out.
        assert!(sim.infected() < 10.0, "infections should decrease with R0 < 1");
    }

    #[test]
    fn test_dt_parameter() {
        let params = EpidemicParams::sir(0.3, 0.1, 10_000.0).unwrap().with_dt(0.5);
        let sim = EpidemicSim::sir(params, 10.0).unwrap();
        assert_eq!(sim.day(), 0);
    }
}
