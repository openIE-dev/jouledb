//! System resource metrics model — CPU, memory, disk, and network metrics
//! with collection intervals, timestamped series, threshold alerts,
//! utilization percentages, and moving averages.
//!
//! Replaces `sysinfo`, `systemstat`, and platform-specific metric collectors
//! with a pure-Rust resource metrics model suitable for ingesting, analyzing,
//! and alerting on system telemetry data.

use std::collections::HashMap;
use std::fmt;

// ── Resource Kind ───────────────────────────────────────────

/// Type of system resource being monitored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    Cpu,
    Memory,
    Disk,
    Network,
    Gpu,
    Custom,
}

impl ResourceKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ResourceKind::Cpu => "cpu",
            ResourceKind::Memory => "memory",
            ResourceKind::Disk => "disk",
            ResourceKind::Network => "network",
            ResourceKind::Gpu => "gpu",
            ResourceKind::Custom => "custom",
        }
    }
}

impl fmt::Display for ResourceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── Metric Sample ───────────────────────────────────────────

/// A single timestamped metric sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    /// Timestamp in milliseconds since epoch.
    pub timestamp_ms: u64,
    /// The metric value.
    pub value: f64,
}

impl Sample {
    pub fn new(timestamp_ms: u64, value: f64) -> Self {
        Self {
            timestamp_ms,
            value,
        }
    }
}

// ── Metric Series ───────────────────────────────────────────

/// A time series of metric samples with a name and resource kind.
#[derive(Debug, Clone)]
pub struct MetricSeries {
    /// Metric name (e.g., "cpu.user", "memory.used_bytes").
    pub name: String,
    /// Resource kind.
    pub kind: ResourceKind,
    /// Unit (e.g., "percent", "bytes", "bytes/s").
    pub unit: String,
    /// Tags/labels.
    pub tags: HashMap<String, String>,
    /// Samples in chronological order.
    samples: Vec<Sample>,
    /// Maximum number of samples to retain.
    max_samples: usize,
}

impl MetricSeries {
    pub fn new(name: &str, kind: ResourceKind, unit: &str) -> Self {
        Self {
            name: name.to_string(),
            kind,
            unit: unit.to_string(),
            tags: HashMap::new(),
            samples: Vec::new(),
            max_samples: 10_000,
        }
    }

    pub fn with_max_samples(mut self, max: usize) -> Self {
        self.max_samples = max.max(1);
        self
    }

    pub fn with_tag(mut self, key: &str, value: &str) -> Self {
        self.tags.insert(key.to_string(), value.to_string());
        self
    }

    /// Add a sample. Evicts oldest if at capacity.
    pub fn add(&mut self, sample: Sample) {
        if self.samples.len() >= self.max_samples {
            self.samples.remove(0);
        }
        self.samples.push(sample);
    }

    /// Add by timestamp and value.
    pub fn record(&mut self, timestamp_ms: u64, value: f64) {
        self.add(Sample::new(timestamp_ms, value));
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get all samples.
    pub fn samples(&self) -> &[Sample] {
        &self.samples
    }

    /// Last sample value.
    pub fn latest(&self) -> Option<&Sample> {
        self.samples.last()
    }

    /// Latest value or 0.
    pub fn latest_value(&self) -> f64 {
        self.latest().map(|s| s.value).unwrap_or(0.0)
    }

    /// Range query: samples within [start_ms, end_ms].
    pub fn range(&self, start_ms: u64, end_ms: u64) -> Vec<&Sample> {
        self.samples
            .iter()
            .filter(|s| s.timestamp_ms >= start_ms && s.timestamp_ms <= end_ms)
            .collect()
    }

    /// Minimum value.
    pub fn min_value(&self) -> f64 {
        self.samples
            .iter()
            .map(|s| s.value)
            .fold(f64::INFINITY, f64::min)
    }

    /// Maximum value.
    pub fn max_value(&self) -> f64 {
        self.samples
            .iter()
            .map(|s| s.value)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Mean of all samples.
    pub fn mean(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().map(|s| s.value).sum::<f64>() / self.samples.len() as f64
    }

    /// Simple moving average over the last N samples.
    pub fn moving_average(&self, window: usize) -> f64 {
        if self.samples.is_empty() || window == 0 {
            return 0.0;
        }
        let start = self.samples.len().saturating_sub(window);
        let window_samples = &self.samples[start..];
        window_samples.iter().map(|s| s.value).sum::<f64>() / window_samples.len() as f64
    }

    /// Exponential moving average (EMA) of the entire series.
    pub fn ema(&self, alpha: f64) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let mut ema = self.samples[0].value;
        for s in self.samples.iter().skip(1) {
            ema = alpha * s.value + (1.0 - alpha) * ema;
        }
        ema
    }

    /// Clear all samples.
    pub fn clear(&mut self) {
        self.samples.clear();
    }
}

// ── CPU Metrics ─────────────────────────────────────────────

/// CPU utilization snapshot.
#[derive(Debug, Clone)]
pub struct CpuMetrics {
    /// Per-core utilization percentages [0..100].
    pub core_usage: Vec<f64>,
    /// Overall utilization.
    pub total_usage: f64,
    /// User-space percentage.
    pub user: f64,
    /// System/kernel percentage.
    pub system: f64,
    /// Idle percentage.
    pub idle: f64,
    /// I/O wait percentage.
    pub iowait: f64,
    /// Timestamp.
    pub timestamp_ms: u64,
}

impl CpuMetrics {
    pub fn new(timestamp_ms: u64) -> Self {
        Self {
            core_usage: Vec::new(),
            total_usage: 0.0,
            user: 0.0,
            system: 0.0,
            idle: 100.0,
            iowait: 0.0,
            timestamp_ms,
        }
    }

    /// Number of cores.
    pub fn num_cores(&self) -> usize {
        self.core_usage.len()
    }

    /// Compute total_usage from components.
    pub fn compute_total(&mut self) {
        self.total_usage = self.user + self.system + self.iowait;
        self.idle = (100.0 - self.total_usage).max(0.0);
    }
}

// ── Memory Metrics ──────────────────────────────────────────

/// Memory utilization snapshot.
#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    /// Total physical memory in bytes.
    pub total_bytes: u64,
    /// Used memory in bytes.
    pub used_bytes: u64,
    /// Free (available) memory in bytes.
    pub available_bytes: u64,
    /// Swap total in bytes.
    pub swap_total_bytes: u64,
    /// Swap used in bytes.
    pub swap_used_bytes: u64,
    /// Timestamp.
    pub timestamp_ms: u64,
}

impl MemoryMetrics {
    pub fn new(timestamp_ms: u64) -> Self {
        Self {
            total_bytes: 0,
            used_bytes: 0,
            available_bytes: 0,
            swap_total_bytes: 0,
            swap_used_bytes: 0,
            timestamp_ms,
        }
    }

    /// Memory utilization as a percentage [0..100].
    pub fn utilization_percent(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        self.used_bytes as f64 / self.total_bytes as f64 * 100.0
    }

    /// Swap utilization as a percentage.
    pub fn swap_utilization_percent(&self) -> f64 {
        if self.swap_total_bytes == 0 {
            return 0.0;
        }
        self.swap_used_bytes as f64 / self.swap_total_bytes as f64 * 100.0
    }
}

// ── Disk Metrics ────────────────────────────────────────────

/// Disk utilization snapshot for a single mount/device.
#[derive(Debug, Clone)]
pub struct DiskMetrics {
    /// Device/mount name.
    pub device: String,
    /// Total space in bytes.
    pub total_bytes: u64,
    /// Used space in bytes.
    pub used_bytes: u64,
    /// Available space in bytes.
    pub available_bytes: u64,
    /// Read bytes/s.
    pub read_bytes_per_sec: f64,
    /// Write bytes/s.
    pub write_bytes_per_sec: f64,
    /// IOPS (reads + writes per second).
    pub iops: f64,
    /// Timestamp.
    pub timestamp_ms: u64,
}

impl DiskMetrics {
    pub fn new(device: &str, timestamp_ms: u64) -> Self {
        Self {
            device: device.to_string(),
            total_bytes: 0,
            used_bytes: 0,
            available_bytes: 0,
            read_bytes_per_sec: 0.0,
            write_bytes_per_sec: 0.0,
            iops: 0.0,
            timestamp_ms,
        }
    }

    /// Disk utilization percentage.
    pub fn utilization_percent(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        self.used_bytes as f64 / self.total_bytes as f64 * 100.0
    }

    /// Combined throughput in bytes/s.
    pub fn throughput_bytes_per_sec(&self) -> f64 {
        self.read_bytes_per_sec + self.write_bytes_per_sec
    }
}

// ── Network Metrics ─────────────────────────────────────────

/// Network interface metrics snapshot.
#[derive(Debug, Clone)]
pub struct NetworkMetrics {
    /// Interface name.
    pub interface: String,
    /// Receive bytes/s.
    pub rx_bytes_per_sec: f64,
    /// Transmit bytes/s.
    pub tx_bytes_per_sec: f64,
    /// Receive packets/s.
    pub rx_packets_per_sec: f64,
    /// Transmit packets/s.
    pub tx_packets_per_sec: f64,
    /// Receive errors/s.
    pub rx_errors_per_sec: f64,
    /// Transmit errors/s.
    pub tx_errors_per_sec: f64,
    /// Timestamp.
    pub timestamp_ms: u64,
}

impl NetworkMetrics {
    pub fn new(interface: &str, timestamp_ms: u64) -> Self {
        Self {
            interface: interface.to_string(),
            rx_bytes_per_sec: 0.0,
            tx_bytes_per_sec: 0.0,
            rx_packets_per_sec: 0.0,
            tx_packets_per_sec: 0.0,
            rx_errors_per_sec: 0.0,
            tx_errors_per_sec: 0.0,
            timestamp_ms,
        }
    }

    /// Total bandwidth in bytes/s.
    pub fn total_bandwidth(&self) -> f64 {
        self.rx_bytes_per_sec + self.tx_bytes_per_sec
    }

    /// Error rate (errors / packets).
    pub fn error_rate(&self) -> f64 {
        let total_packets = self.rx_packets_per_sec + self.tx_packets_per_sec;
        if total_packets == 0.0 {
            return 0.0;
        }
        let total_errors = self.rx_errors_per_sec + self.tx_errors_per_sec;
        total_errors / total_packets
    }
}

// ── Threshold Alert ─────────────────────────────────────────

/// Direction of a threshold comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdDirection {
    /// Alert when value > threshold.
    Above,
    /// Alert when value < threshold.
    Below,
}

/// A threshold alert configuration.
#[derive(Debug, Clone)]
pub struct ThresholdAlert {
    pub name: String,
    pub metric_name: String,
    pub threshold: f64,
    pub direction: ThresholdDirection,
    /// Severity: "warning", "critical", "info".
    pub severity: String,
    /// Whether this alert is currently firing.
    pub firing: bool,
    /// Last checked timestamp.
    pub last_check_ms: u64,
}

impl ThresholdAlert {
    pub fn above(name: &str, metric: &str, threshold: f64, severity: &str) -> Self {
        Self {
            name: name.to_string(),
            metric_name: metric.to_string(),
            threshold,
            direction: ThresholdDirection::Above,
            severity: severity.to_string(),
            firing: false,
            last_check_ms: 0,
        }
    }

    pub fn below(name: &str, metric: &str, threshold: f64, severity: &str) -> Self {
        Self {
            name: name.to_string(),
            metric_name: metric.to_string(),
            threshold,
            direction: ThresholdDirection::Below,
            severity: severity.to_string(),
            firing: false,
            last_check_ms: 0,
        }
    }

    /// Check the alert against a value. Updates `firing` state and returns it.
    pub fn check(&mut self, value: f64, timestamp_ms: u64) -> bool {
        self.last_check_ms = timestamp_ms;
        self.firing = match self.direction {
            ThresholdDirection::Above => value > self.threshold,
            ThresholdDirection::Below => value < self.threshold,
        };
        self.firing
    }
}

// ── Resource Collector ──────────────────────────────────────

/// Collects and stores metric series for multiple resources.
#[derive(Debug, Clone)]
pub struct ResourceCollector {
    /// All metric series, keyed by name.
    series: HashMap<String, MetricSeries>,
    /// Configured alerts.
    alerts: Vec<ThresholdAlert>,
    /// Collection interval in milliseconds.
    pub collection_interval_ms: u64,
}

impl ResourceCollector {
    pub fn new(collection_interval_ms: u64) -> Self {
        Self {
            series: HashMap::new(),
            alerts: Vec::new(),
            collection_interval_ms,
        }
    }

    /// Register a metric series.
    pub fn register(&mut self, series: MetricSeries) {
        self.series.insert(series.name.clone(), series);
    }

    /// Record a value for a named series.
    pub fn record(&mut self, name: &str, timestamp_ms: u64, value: f64) {
        if let Some(s) = self.series.get_mut(name) {
            s.record(timestamp_ms, value);
        }
    }

    /// Record CPU metrics.
    pub fn record_cpu(&mut self, cpu: &CpuMetrics) {
        self.record("cpu.total", cpu.timestamp_ms, cpu.total_usage);
        self.record("cpu.user", cpu.timestamp_ms, cpu.user);
        self.record("cpu.system", cpu.timestamp_ms, cpu.system);
        self.record("cpu.idle", cpu.timestamp_ms, cpu.idle);
        self.record("cpu.iowait", cpu.timestamp_ms, cpu.iowait);
    }

    /// Record memory metrics.
    pub fn record_memory(&mut self, mem: &MemoryMetrics) {
        self.record("memory.utilization", mem.timestamp_ms, mem.utilization_percent());
        self.record("memory.used", mem.timestamp_ms, mem.used_bytes as f64);
        self.record("memory.available", mem.timestamp_ms, mem.available_bytes as f64);
    }

    /// Record disk metrics.
    pub fn record_disk(&mut self, disk: &DiskMetrics) {
        let prefix = format!("disk.{}", disk.device);
        let util_name = format!("{}.utilization", prefix);
        let iops_name = format!("{}.iops", prefix);
        if self.series.contains_key(&util_name) {
            self.record(&util_name, disk.timestamp_ms, disk.utilization_percent());
        }
        if self.series.contains_key(&iops_name) {
            self.record(&iops_name, disk.timestamp_ms, disk.iops);
        }
    }

    /// Record network metrics.
    pub fn record_network(&mut self, net: &NetworkMetrics) {
        let prefix = format!("net.{}", net.interface);
        let bw_name = format!("{}.bandwidth", prefix);
        if self.series.contains_key(&bw_name) {
            self.record(&bw_name, net.timestamp_ms, net.total_bandwidth());
        }
    }

    /// Get a series by name.
    pub fn get_series(&self, name: &str) -> Option<&MetricSeries> {
        self.series.get(name)
    }

    /// Number of registered series.
    pub fn series_count(&self) -> usize {
        self.series.len()
    }

    /// Add an alert.
    pub fn add_alert(&mut self, alert: ThresholdAlert) {
        self.alerts.push(alert);
    }

    /// Check all alerts against current latest values. Returns names of firing alerts.
    pub fn check_alerts(&mut self, timestamp_ms: u64) -> Vec<String> {
        let mut firing = Vec::new();
        // Collect latest values first to avoid borrow issues
        let latest_values: Vec<(usize, f64)> = self
            .alerts
            .iter()
            .enumerate()
            .filter_map(|(i, alert)| {
                self.series
                    .get(&alert.metric_name)
                    .and_then(|s| s.latest())
                    .map(|sample| (i, sample.value))
            })
            .collect();

        for (i, value) in latest_values {
            if self.alerts[i].check(value, timestamp_ms) {
                firing.push(self.alerts[i].name.clone());
            }
        }
        firing
    }

    /// Get all alert states.
    pub fn alerts(&self) -> &[ThresholdAlert] {
        &self.alerts
    }

    /// All series names.
    pub fn series_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.series.keys().cloned().collect();
        names.sort();
        names
    }
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_kind() {
        assert_eq!(ResourceKind::Cpu.as_str(), "cpu");
        assert_eq!(format!("{}", ResourceKind::Memory), "memory");
    }

    #[test]
    fn test_sample() {
        let s = Sample::new(1000, 42.5);
        assert_eq!(s.timestamp_ms, 1000);
        assert!((s.value - 42.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metric_series_basic() {
        let mut series = MetricSeries::new("cpu.total", ResourceKind::Cpu, "percent");
        series.record(1000, 45.0);
        series.record(2000, 55.0);
        assert_eq!(series.len(), 2);
        assert!((series.latest_value() - 55.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metric_series_max_samples() {
        let mut series = MetricSeries::new("m", ResourceKind::Cpu, "%")
            .with_max_samples(3);
        for i in 0..5 {
            series.record(i as u64 * 1000, i as f64);
        }
        assert_eq!(series.len(), 3);
        // Oldest should have been evicted; latest = 4.0
        assert!((series.latest_value() - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metric_series_range() {
        let mut series = MetricSeries::new("m", ResourceKind::Cpu, "%");
        for i in 0..10 {
            series.record(i * 100, i as f64);
        }
        let range = series.range(300, 600);
        assert_eq!(range.len(), 4); // 300, 400, 500, 600
    }

    #[test]
    fn test_metric_series_stats() {
        let mut series = MetricSeries::new("m", ResourceKind::Cpu, "%");
        series.record(0, 10.0);
        series.record(1, 20.0);
        series.record(2, 30.0);
        assert!((series.mean() - 20.0).abs() < 0.01);
        assert!((series.min_value() - 10.0).abs() < 0.01);
        assert!((series.max_value() - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_moving_average() {
        let mut series = MetricSeries::new("m", ResourceKind::Cpu, "%");
        for i in 1..=10 {
            series.record(i, i as f64);
        }
        // Last 3 values: 8, 9, 10 -> avg = 9.0
        assert!((series.moving_average(3) - 9.0).abs() < 0.01);
    }

    #[test]
    fn test_ema() {
        let mut series = MetricSeries::new("m", ResourceKind::Cpu, "%");
        for i in 0..10 {
            series.record(i, 50.0);
        }
        // Constant values -> EMA should be 50.0
        assert!((series.ema(0.3) - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_cpu_metrics() {
        let mut cpu = CpuMetrics::new(1000);
        cpu.user = 30.0;
        cpu.system = 10.0;
        cpu.iowait = 5.0;
        cpu.compute_total();
        assert!((cpu.total_usage - 45.0).abs() < 0.01);
        assert!((cpu.idle - 55.0).abs() < 0.01);
    }

    #[test]
    fn test_memory_metrics() {
        let mut mem = MemoryMetrics::new(1000);
        mem.total_bytes = 16_000_000_000;
        mem.used_bytes = 8_000_000_000;
        mem.available_bytes = 8_000_000_000;
        assert!((mem.utilization_percent() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_memory_swap() {
        let mut mem = MemoryMetrics::new(1000);
        mem.swap_total_bytes = 4_000_000_000;
        mem.swap_used_bytes = 1_000_000_000;
        assert!((mem.swap_utilization_percent() - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_disk_metrics() {
        let mut disk = DiskMetrics::new("/dev/sda1", 1000);
        disk.total_bytes = 500_000_000_000;
        disk.used_bytes = 250_000_000_000;
        assert!((disk.utilization_percent() - 50.0).abs() < 0.01);
        disk.read_bytes_per_sec = 100.0;
        disk.write_bytes_per_sec = 50.0;
        assert!((disk.throughput_bytes_per_sec() - 150.0).abs() < 0.01);
    }

    #[test]
    fn test_network_metrics() {
        let mut net = NetworkMetrics::new("eth0", 1000);
        net.rx_bytes_per_sec = 1000.0;
        net.tx_bytes_per_sec = 500.0;
        assert!((net.total_bandwidth() - 1500.0).abs() < 0.01);
    }

    #[test]
    fn test_network_error_rate() {
        let mut net = NetworkMetrics::new("eth0", 1000);
        net.rx_packets_per_sec = 1000.0;
        net.tx_packets_per_sec = 500.0;
        net.rx_errors_per_sec = 10.0;
        net.tx_errors_per_sec = 5.0;
        let rate = net.error_rate();
        assert!((rate - 0.01).abs() < 0.001); // 15/1500
    }

    #[test]
    fn test_threshold_alert_above() {
        let mut alert = ThresholdAlert::above("cpu_high", "cpu.total", 90.0, "critical");
        assert!(!alert.check(85.0, 1000));
        assert!(alert.check(95.0, 2000));
        assert!(alert.firing);
    }

    #[test]
    fn test_threshold_alert_below() {
        let mut alert = ThresholdAlert::below("mem_low", "memory.available", 1000.0, "warning");
        assert!(alert.check(500.0, 1000));
        assert!(!alert.check(2000.0, 2000));
    }

    #[test]
    fn test_resource_collector() {
        let mut collector = ResourceCollector::new(5000);
        collector.register(MetricSeries::new("cpu.total", ResourceKind::Cpu, "percent"));
        collector.register(MetricSeries::new("cpu.user", ResourceKind::Cpu, "percent"));
        collector.register(MetricSeries::new("cpu.system", ResourceKind::Cpu, "percent"));
        collector.register(MetricSeries::new("cpu.idle", ResourceKind::Cpu, "percent"));
        collector.register(MetricSeries::new("cpu.iowait", ResourceKind::Cpu, "percent"));
        assert_eq!(collector.series_count(), 5);

        let mut cpu = CpuMetrics::new(1000);
        cpu.user = 40.0;
        cpu.system = 10.0;
        cpu.compute_total();
        collector.record_cpu(&cpu);

        let s = collector.get_series("cpu.total").unwrap();
        assert!((s.latest_value() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_collector_alerts() {
        let mut collector = ResourceCollector::new(1000);
        collector.register(MetricSeries::new("cpu.total", ResourceKind::Cpu, "percent"));
        collector.add_alert(ThresholdAlert::above("cpu_high", "cpu.total", 80.0, "warning"));
        collector.record("cpu.total", 1000, 90.0);
        let firing = collector.check_alerts(1000);
        assert_eq!(firing.len(), 1);
        assert_eq!(firing[0], "cpu_high");
    }

    #[test]
    fn test_collector_record_memory() {
        let mut collector = ResourceCollector::new(1000);
        collector.register(MetricSeries::new("memory.utilization", ResourceKind::Memory, "percent"));
        collector.register(MetricSeries::new("memory.used", ResourceKind::Memory, "bytes"));
        collector.register(MetricSeries::new("memory.available", ResourceKind::Memory, "bytes"));

        let mut mem = MemoryMetrics::new(1000);
        mem.total_bytes = 16_000_000_000;
        mem.used_bytes = 12_000_000_000;
        mem.available_bytes = 4_000_000_000;
        collector.record_memory(&mem);

        let s = collector.get_series("memory.utilization").unwrap();
        assert!((s.latest_value() - 75.0).abs() < 0.01);
    }

    #[test]
    fn test_series_clear() {
        let mut series = MetricSeries::new("m", ResourceKind::Cpu, "%");
        series.record(0, 1.0);
        series.clear();
        assert!(series.is_empty());
    }

    #[test]
    fn test_series_with_tag() {
        let series = MetricSeries::new("m", ResourceKind::Cpu, "%")
            .with_tag("host", "web-01");
        assert_eq!(series.tags.get("host").unwrap(), "web-01");
    }

    #[test]
    fn test_empty_series_stats() {
        let series = MetricSeries::new("m", ResourceKind::Cpu, "%");
        assert_eq!(series.mean(), 0.0);
        assert_eq!(series.moving_average(5), 0.0);
        assert_eq!(series.ema(0.3), 0.0);
        assert_eq!(series.latest_value(), 0.0);
    }

    #[test]
    fn test_zero_total_memory() {
        let mem = MemoryMetrics::new(0);
        assert_eq!(mem.utilization_percent(), 0.0);
        assert_eq!(mem.swap_utilization_percent(), 0.0);
    }
}
