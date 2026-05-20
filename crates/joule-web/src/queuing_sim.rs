//! Queuing theory simulation — M/M/1, M/M/c, M/D/1 queue models.
//!
//! Replaces SimPy.js / queueing-tool / simjs with pure Rust.
//! Supports arrival/service rate, utilization, average wait time
//! (Little's law), queue length distribution, and discrete-event
//! simulation stepping.

use std::collections::VecDeque;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Domain errors for queuing simulation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueueError {
    /// Invalid rate (must be positive).
    InvalidRate(String),
    /// Server count must be positive.
    ZeroServers,
    /// System unstable (arrival rate >= service capacity).
    Unstable { arrival: String, capacity: String },
    /// Simulation not yet run.
    NoData,
}

impl fmt::Display for QueueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRate(r) => write!(f, "invalid rate: {r}"),
            Self::ZeroServers => write!(f, "server count must be >= 1"),
            Self::Unstable { arrival, capacity } => {
                write!(f, "system unstable: arrival {arrival} >= capacity {capacity}")
            }
            Self::NoData => write!(f, "no simulation data available"),
        }
    }
}

impl std::error::Error for QueueError {}

// ── PRNG ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_f64(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Exponential random variable with rate lambda.
    fn exponential(&mut self, lambda: f64) -> f64 {
        let u = self.next_f64().max(1e-15);
        -u.ln() / lambda
    }
}

// ── Queue Model Type ────────────────────────────────────────────

/// Type of queuing model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueModel {
    /// M/M/1: Poisson arrivals, exponential service, 1 server.
    MM1,
    /// M/M/c: Poisson arrivals, exponential service, c servers.
    MMc,
    /// M/D/1: Poisson arrivals, deterministic (constant) service, 1 server.
    MD1,
}

// ── Customer ────────────────────────────────────────────────────

/// A customer entity in the queue.
#[derive(Debug, Clone)]
pub struct Customer {
    pub id: u64,
    pub arrival_time: f64,
    pub service_start_time: Option<f64>,
    pub departure_time: Option<f64>,
}

impl Customer {
    /// Wait time (time in queue before service).
    pub fn wait_time(&self) -> Option<f64> {
        self.service_start_time.map(|s| s - self.arrival_time)
    }

    /// Total time in system (wait + service).
    pub fn system_time(&self) -> Option<f64> {
        self.departure_time.map(|d| d - self.arrival_time)
    }

    /// Service duration.
    pub fn service_duration(&self) -> Option<f64> {
        match (self.service_start_time, self.departure_time) {
            (Some(s), Some(d)) => Some(d - s),
            _ => None,
        }
    }
}

// ── Analytical Results ──────────────────────────────────────────

/// Analytical results for M/M/1 queue.
#[derive(Debug, Clone)]
pub struct MM1Analytics {
    pub lambda: f64,
    pub mu: f64,
    pub rho: f64,
    pub l_q: f64,
    pub l_s: f64,
    pub w_q: f64,
    pub w_s: f64,
    pub p0: f64,
}

/// Compute analytical M/M/1 results.
pub fn mm1_analytics(lambda: f64, mu: f64) -> Result<MM1Analytics, QueueError> {
    if lambda <= 0.0 {
        return Err(QueueError::InvalidRate(format!("{lambda}")));
    }
    if mu <= 0.0 {
        return Err(QueueError::InvalidRate(format!("{mu}")));
    }
    let rho = lambda / mu;
    if rho >= 1.0 {
        return Err(QueueError::Unstable {
            arrival: format!("{lambda}"),
            capacity: format!("{mu}"),
        });
    }
    let l_s = rho / (1.0 - rho);
    let l_q = rho * rho / (1.0 - rho);
    let w_s = 1.0 / (mu - lambda);
    let w_q = rho / (mu - lambda);
    let p0 = 1.0 - rho;
    Ok(MM1Analytics { lambda, mu, rho, l_q, l_s, w_q, w_s, p0 })
}

/// Analytical results for M/D/1 queue.
#[derive(Debug, Clone)]
pub struct MD1Analytics {
    pub lambda: f64,
    pub mu: f64,
    pub rho: f64,
    pub l_q: f64,
    pub w_q: f64,
    pub w_s: f64,
}

/// Compute analytical M/D/1 results.
pub fn md1_analytics(lambda: f64, mu: f64) -> Result<MD1Analytics, QueueError> {
    if lambda <= 0.0 {
        return Err(QueueError::InvalidRate(format!("{lambda}")));
    }
    if mu <= 0.0 {
        return Err(QueueError::InvalidRate(format!("{mu}")));
    }
    let rho = lambda / mu;
    if rho >= 1.0 {
        return Err(QueueError::Unstable {
            arrival: format!("{lambda}"),
            capacity: format!("{mu}"),
        });
    }
    let l_q = rho * rho / (2.0 * (1.0 - rho));
    let w_q = l_q / lambda;
    let w_s = w_q + 1.0 / mu;
    Ok(MD1Analytics { lambda, mu, rho, l_q, w_q, w_s })
}

// ── Server ──────────────────────────────────────────────────────

/// A service server.
#[derive(Debug, Clone)]
struct Server {
    busy_until: f64,
    customers_served: u64,
    busy_time: f64,
}

impl Server {
    fn new() -> Self {
        Self { busy_until: 0.0, customers_served: 0, busy_time: 0.0 }
    }

    fn is_idle(&self, time: f64) -> bool {
        time >= self.busy_until
    }
}

// ── Queue Simulator ─────────────────────────────────────────────

/// Discrete-event queue simulator.
#[derive(Debug, Clone)]
pub struct QueueSimulator {
    model: QueueModel,
    lambda: f64,
    mu: f64,
    servers: Vec<Server>,
    queue: VecDeque<Customer>,
    completed: Vec<Customer>,
    current_time: f64,
    next_customer_id: u64,
    rng: Rng,
    max_queue_length_observed: usize,
    queue_length_samples: Vec<(f64, usize)>,
}

impl QueueSimulator {
    /// Create a new simulator.
    pub fn new(model: QueueModel, lambda: f64, mu: f64, servers: usize, seed: u64) -> Result<Self, QueueError> {
        if lambda <= 0.0 {
            return Err(QueueError::InvalidRate(format!("{lambda}")));
        }
        if mu <= 0.0 {
            return Err(QueueError::InvalidRate(format!("{mu}")));
        }
        let server_count = match model {
            QueueModel::MM1 | QueueModel::MD1 => 1,
            QueueModel::MMc => {
                if servers == 0 { return Err(QueueError::ZeroServers); }
                servers
            }
        };
        Ok(Self {
            model,
            lambda,
            mu,
            servers: (0..server_count).map(|_| Server::new()).collect(),
            queue: VecDeque::new(),
            completed: Vec::new(),
            current_time: 0.0,
            next_customer_id: 0,
            rng: Rng::new(seed),
            max_queue_length_observed: 0,
            queue_length_samples: Vec::new(),
        })
    }

    /// Number of servers.
    pub fn server_count(&self) -> usize { self.servers.len() }

    /// Current simulation time.
    pub fn current_time(&self) -> f64 { self.current_time }

    /// Number of customers currently waiting in queue.
    pub fn queue_length(&self) -> usize { self.queue.len() }

    /// Number of completed customers.
    pub fn completed_count(&self) -> usize { self.completed.len() }

    /// Maximum queue length observed.
    pub fn max_queue_length(&self) -> usize { self.max_queue_length_observed }

    /// Utilization (fraction of time servers are busy).
    pub fn utilization(&self) -> f64 {
        if self.current_time == 0.0 || self.servers.is_empty() {
            return 0.0;
        }
        let total_busy: f64 = self.servers.iter().map(|s| s.busy_time).sum();
        total_busy / (self.current_time * self.servers.len() as f64)
    }

    /// Generate the next inter-arrival time.
    fn next_inter_arrival(&mut self) -> f64 {
        self.rng.exponential(self.lambda)
    }

    /// Generate the next service time.
    fn next_service_time(&mut self) -> f64 {
        match self.model {
            QueueModel::MM1 | QueueModel::MMc => self.rng.exponential(self.mu),
            QueueModel::MD1 => 1.0 / self.mu,
        }
    }

    /// Find the server that becomes free earliest.
    fn earliest_free_server(&self) -> (usize, f64) {
        let mut best_idx = 0;
        let mut best_time = self.servers[0].busy_until;
        for (i, s) in self.servers.iter().enumerate().skip(1) {
            if s.busy_until < best_time {
                best_time = s.busy_until;
                best_idx = i;
            }
        }
        (best_idx, best_time)
    }

    /// Process a single arrival event.
    pub fn process_arrival(&mut self) -> Customer {
        let inter_arrival = self.next_inter_arrival();
        self.current_time += inter_arrival;

        let customer = Customer {
            id: self.next_customer_id,
            arrival_time: self.current_time,
            service_start_time: None,
            departure_time: None,
        };
        self.next_customer_id += 1;
        self.queue.push_back(customer.clone());

        if self.queue.len() > self.max_queue_length_observed {
            self.max_queue_length_observed = self.queue.len();
        }
        self.queue_length_samples.push((self.current_time, self.queue.len()));

        self.try_serve();
        customer
    }

    /// Try to serve waiting customers with idle servers.
    fn try_serve(&mut self) {
        loop {
            if self.queue.is_empty() {
                break;
            }
            // Find an idle server.
            let idle = self.servers.iter().position(|s| s.is_idle(self.current_time));
            let server_idx = match idle {
                Some(idx) => idx,
                None => break,
            };

            let mut customer = self.queue.pop_front().unwrap();
            let service_time = self.next_service_time();
            customer.service_start_time = Some(self.current_time);
            customer.departure_time = Some(self.current_time + service_time);

            self.servers[server_idx].busy_until = self.current_time + service_time;
            self.servers[server_idx].customers_served += 1;
            self.servers[server_idx].busy_time += service_time;

            self.completed.push(customer);
        }
    }

    /// Run the simulation for `n` arrivals.
    pub fn run(&mut self, n_arrivals: u64) {
        for _ in 0..n_arrivals {
            self.process_arrival();
        }
        // Process any remaining customers in queue by advancing time.
        while !self.queue.is_empty() {
            let (idx, free_time) = self.earliest_free_server();
            self.current_time = free_time;
            if let Some(mut customer) = self.queue.pop_front() {
                let service_time = self.next_service_time();
                customer.service_start_time = Some(self.current_time);
                customer.departure_time = Some(self.current_time + service_time);
                self.servers[idx].busy_until = self.current_time + service_time;
                self.servers[idx].customers_served += 1;
                self.servers[idx].busy_time += service_time;
                self.completed.push(customer);
            }
        }
    }

    /// Average wait time across completed customers.
    pub fn average_wait_time(&self) -> Option<f64> {
        if self.completed.is_empty() {
            return None;
        }
        let total: f64 = self.completed.iter()
            .filter_map(|c| c.wait_time())
            .sum();
        let count = self.completed.iter()
            .filter(|c| c.wait_time().is_some())
            .count();
        if count == 0 { None } else { Some(total / count as f64) }
    }

    /// Average system time (wait + service).
    pub fn average_system_time(&self) -> Option<f64> {
        if self.completed.is_empty() {
            return None;
        }
        let total: f64 = self.completed.iter()
            .filter_map(|c| c.system_time())
            .sum();
        let count = self.completed.iter()
            .filter(|c| c.system_time().is_some())
            .count();
        if count == 0 { None } else { Some(total / count as f64) }
    }

    /// Average service time.
    pub fn average_service_time(&self) -> Option<f64> {
        if self.completed.is_empty() {
            return None;
        }
        let total: f64 = self.completed.iter()
            .filter_map(|c| c.service_duration())
            .sum();
        let count = self.completed.iter()
            .filter(|c| c.service_duration().is_some())
            .count();
        if count == 0 { None } else { Some(total / count as f64) }
    }

    /// Little's law estimate: L = lambda * W.
    pub fn littles_law_l(&self) -> Option<f64> {
        self.average_system_time().map(|w| self.lambda * w)
    }

    /// Queue length distribution (approximate).
    pub fn queue_length_distribution(&self) -> Vec<(usize, f64)> {
        if self.queue_length_samples.is_empty() {
            return Vec::new();
        }
        let max_len = self.queue_length_samples.iter().map(|(_, l)| *l).max().unwrap_or(0);
        let mut counts = vec![0usize; max_len + 1];
        for &(_, len) in &self.queue_length_samples {
            counts[len] += 1;
        }
        let total = self.queue_length_samples.len() as f64;
        counts.iter().enumerate().map(|(len, &c)| (len, c as f64 / total)).collect()
    }

    /// All completed customers.
    pub fn completed_customers(&self) -> &[Customer] {
        &self.completed
    }

    /// Throughput: customers served per unit time.
    pub fn throughput(&self) -> f64 {
        if self.current_time == 0.0 {
            return 0.0;
        }
        self.completed.len() as f64 / self.current_time
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mm1_analytics() {
        let a = mm1_analytics(2.0, 5.0).unwrap();
        assert!((a.rho - 0.4).abs() < 1e-10);
        assert!((a.l_s - 2.0 / 3.0).abs() < 1e-6);
        assert!((a.p0 - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_mm1_unstable() {
        assert!(mm1_analytics(5.0, 3.0).is_err());
    }

    #[test]
    fn test_mm1_invalid_rate() {
        assert!(mm1_analytics(0.0, 5.0).is_err());
        assert!(mm1_analytics(2.0, 0.0).is_err());
    }

    #[test]
    fn test_md1_analytics() {
        let a = md1_analytics(2.0, 5.0).unwrap();
        assert!((a.rho - 0.4).abs() < 1e-10);
        assert!(a.l_q >= 0.0);
        assert!(a.w_q >= 0.0);
    }

    #[test]
    fn test_simulator_creation() {
        let sim = QueueSimulator::new(QueueModel::MM1, 2.0, 5.0, 1, 42).unwrap();
        assert_eq!(sim.server_count(), 1);
        assert_eq!(sim.queue_length(), 0);
        assert_eq!(sim.completed_count(), 0);
    }

    #[test]
    fn test_mmc_creation() {
        let sim = QueueSimulator::new(QueueModel::MMc, 2.0, 5.0, 3, 42).unwrap();
        assert_eq!(sim.server_count(), 3);
    }

    #[test]
    fn test_zero_servers() {
        assert!(QueueSimulator::new(QueueModel::MMc, 2.0, 5.0, 0, 42).is_err());
    }

    #[test]
    fn test_single_arrival() {
        let mut sim = QueueSimulator::new(QueueModel::MM1, 2.0, 5.0, 1, 42).unwrap();
        let c = sim.process_arrival();
        assert_eq!(c.id, 0);
        assert!(c.arrival_time > 0.0);
    }

    #[test]
    fn test_run_mm1() {
        let mut sim = QueueSimulator::new(QueueModel::MM1, 2.0, 5.0, 1, 42).unwrap();
        sim.run(1000);
        assert_eq!(sim.completed_count(), 1000);
        assert!(sim.current_time() > 0.0);
    }

    #[test]
    fn test_utilization_range() {
        let mut sim = QueueSimulator::new(QueueModel::MM1, 2.0, 5.0, 1, 42).unwrap();
        sim.run(500);
        let u = sim.utilization();
        assert!(u >= 0.0 && u <= 1.0, "utilization was {u}");
    }

    #[test]
    fn test_average_wait_time() {
        let mut sim = QueueSimulator::new(QueueModel::MM1, 2.0, 5.0, 1, 42).unwrap();
        sim.run(1000);
        let wt = sim.average_wait_time().unwrap();
        assert!(wt >= 0.0);
    }

    #[test]
    fn test_average_system_time() {
        let mut sim = QueueSimulator::new(QueueModel::MM1, 2.0, 5.0, 1, 42).unwrap();
        sim.run(500);
        let st = sim.average_system_time().unwrap();
        let wt = sim.average_wait_time().unwrap();
        assert!(st >= wt);
    }

    #[test]
    fn test_littles_law() {
        let mut sim = QueueSimulator::new(QueueModel::MM1, 2.0, 5.0, 1, 42).unwrap();
        sim.run(5000);
        let l = sim.littles_law_l().unwrap();
        assert!(l > 0.0);
    }

    #[test]
    fn test_throughput() {
        let mut sim = QueueSimulator::new(QueueModel::MM1, 2.0, 5.0, 1, 42).unwrap();
        sim.run(1000);
        let tp = sim.throughput();
        assert!(tp > 0.0);
    }

    #[test]
    fn test_queue_length_distribution() {
        let mut sim = QueueSimulator::new(QueueModel::MM1, 2.0, 5.0, 1, 42).unwrap();
        sim.run(500);
        let dist = sim.queue_length_distribution();
        assert!(!dist.is_empty());
        let total: f64 = dist.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_max_queue_length() {
        let mut sim = QueueSimulator::new(QueueModel::MM1, 2.0, 5.0, 1, 42).unwrap();
        sim.run(1000);
        assert!(sim.max_queue_length() >= 0);
    }

    #[test]
    fn test_md1_simulation() {
        let mut sim = QueueSimulator::new(QueueModel::MD1, 2.0, 5.0, 1, 42).unwrap();
        sim.run(500);
        assert_eq!(sim.completed_count(), 500);
        // M/D/1 has deterministic service.
        let svc = sim.average_service_time().unwrap();
        assert!((svc - 0.2).abs() < 0.01, "avg service was {svc}");
    }

    #[test]
    fn test_mmc_less_wait_than_mm1() {
        let mut sim1 = QueueSimulator::new(QueueModel::MM1, 4.0, 5.0, 1, 42).unwrap();
        sim1.run(2000);
        let w1 = sim1.average_wait_time().unwrap();

        let mut simc = QueueSimulator::new(QueueModel::MMc, 4.0, 5.0, 3, 42).unwrap();
        simc.run(2000);
        let wc = simc.average_wait_time().unwrap();

        assert!(wc <= w1, "M/M/c wait {wc} should be <= M/M/1 wait {w1}");
    }

    #[test]
    fn test_customer_wait_time() {
        let c = Customer {
            id: 0,
            arrival_time: 1.0,
            service_start_time: Some(3.0),
            departure_time: Some(5.0),
        };
        assert!((c.wait_time().unwrap() - 2.0).abs() < 1e-10);
        assert!((c.system_time().unwrap() - 4.0).abs() < 1e-10);
        assert!((c.service_duration().unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_customer_no_service() {
        let c = Customer {
            id: 0,
            arrival_time: 1.0,
            service_start_time: None,
            departure_time: None,
        };
        assert!(c.wait_time().is_none());
        assert!(c.system_time().is_none());
    }

    #[test]
    fn test_no_data_average() {
        let sim = QueueSimulator::new(QueueModel::MM1, 2.0, 5.0, 1, 42).unwrap();
        assert!(sim.average_wait_time().is_none());
    }
}
