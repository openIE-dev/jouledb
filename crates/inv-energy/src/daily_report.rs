//! Daily energy report generation.
//!
//! Aggregates per-operation energy data from [`JouleMeter`] into a
//! service-level daily report matching the HTML specification format:
//!
//! ```text
//! ┌────────────────────────┬──────────────┬─────────────┐
//! │ Work done              │ Total energy │ Per unit     │
//! ├────────────────────────┼──────────────┼─────────────┤
//! │ API — 100k requests    │ 34,000 J     │ 0.34 J/req  │
//! │ Database — 50k queries │ 4,000 J      │ 0.08 J/query│
//! └────────────────────────┴──────────────┴─────────────┘
//! ```

use serde::{Deserialize, Serialize};

use crate::joule_meter::JouleMeter;

/// A single row in the daily energy report — one per service category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEnergyRow {
    /// Service category name (e.g., "API", "Database", "AI Inference").
    pub service: String,
    /// Human-readable work unit label (e.g., "requests", "queries", "ops").
    pub work_unit: String,
    /// Total operations (call count) for this service today.
    pub work_count: u64,
    /// Total energy consumed by this service today (joules).
    pub total_joules: f64,
    /// Energy per operation (joules/unit).
    pub per_unit_joules: f64,
}

/// The complete daily energy report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyEnergyReport {
    /// Report date (YYYY-MM-DD).
    pub date: String,
    /// Per-service breakdown rows.
    pub services: Vec<ServiceEnergyRow>,
    /// Grand total energy across all services (joules).
    pub total_energy_joules: f64,
    /// Grand total operations across all services.
    pub total_operations: u64,
    /// Grand total energy per operation (joules/op).
    pub avg_joules_per_operation: f64,
}

/// Known service category with its operation-name prefix and work-unit label.
struct ServiceCategory {
    name: &'static str,
    prefix: &'static str,
    work_unit: &'static str,
}

const SERVICE_CATEGORIES: &[ServiceCategory] = &[
    ServiceCategory {
        name: "API",
        prefix: "api",
        work_unit: "requests",
    },
    ServiceCategory {
        name: "Database",
        prefix: "db",
        work_unit: "queries",
    },
    ServiceCategory {
        name: "Object Store",
        prefix: "object",
        work_unit: "ops",
    },
    ServiceCategory {
        name: "AI Inference",
        prefix: "inference",
        work_unit: "requests",
    },
    ServiceCategory {
        name: "Functions",
        prefix: "function",
        work_unit: "invocations",
    },
    ServiceCategory {
        name: "Cache",
        prefix: "cache",
        work_unit: "ops",
    },
    ServiceCategory {
        name: "Queue",
        prefix: "queue",
        work_unit: "messages",
    },
    ServiceCategory {
        name: "Stream",
        prefix: "stream",
        work_unit: "records",
    },
    ServiceCategory {
        name: "Search",
        prefix: "search",
        work_unit: "queries",
    },
    ServiceCategory {
        name: "PubSub",
        prefix: "pubsub",
        work_unit: "messages",
    },
    ServiceCategory {
        name: "IoT",
        prefix: "iot",
        work_unit: "events",
    },
    ServiceCategory {
        name: "TimeSeries",
        prefix: "ts",
        work_unit: "writes",
    },
    ServiceCategory {
        name: "CDN",
        prefix: "cdn",
        work_unit: "requests",
    },
    ServiceCategory {
        name: "DNS",
        prefix: "dns",
        work_unit: "lookups",
    },
    ServiceCategory {
        name: "Load Balancer",
        prefix: "lb",
        work_unit: "connections",
    },
];

/// Generates a daily energy report from a [`JouleMeter`].
///
/// Operations are grouped by prefix into service categories. Any
/// operations that don't match a known prefix are collected under "Other".
pub fn generate_daily_report(meter: &JouleMeter, date: &str) -> DailyEnergyReport {
    let profiles = meter.list_profiles();
    let mut rows = Vec::new();
    let mut claimed = std::collections::HashSet::new();

    for cat in SERVICE_CATEGORIES {
        let mut total_j = 0.0;
        let mut count = 0u64;

        for p in &profiles {
            if p.op_name.starts_with(cat.prefix)
                || p.op_name.starts_with(&format!("{}_", cat.prefix))
            {
                total_j += p.total_joules;
                count += p.call_count;
                claimed.insert(p.op_name.as_str());
            }
        }

        if count > 0 {
            rows.push(ServiceEnergyRow {
                service: cat.name.to_string(),
                work_unit: cat.work_unit.to_string(),
                work_count: count,
                total_joules: total_j,
                per_unit_joules: if count > 0 {
                    total_j / count as f64
                } else {
                    0.0
                },
            });
        }
    }

    // Collect unmatched operations under "Other"
    let mut other_j = 0.0;
    let mut other_count = 0u64;
    for p in &profiles {
        if !claimed.contains(p.op_name.as_str()) {
            other_j += p.total_joules;
            other_count += p.call_count;
        }
    }
    if other_count > 0 {
        rows.push(ServiceEnergyRow {
            service: "Other".to_string(),
            work_unit: "ops".to_string(),
            work_count: other_count,
            total_joules: other_j,
            per_unit_joules: other_j / other_count as f64,
        });
    }

    // Sort by total_joules descending
    rows.sort_by(|a, b| {
        b.total_joules
            .partial_cmp(&a.total_joules)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let total_energy: f64 = rows.iter().map(|r| r.total_joules).sum();
    let total_ops: u64 = rows.iter().map(|r| r.work_count).sum();

    DailyEnergyReport {
        date: date.to_string(),
        services: rows,
        total_energy_joules: total_energy,
        total_operations: total_ops,
        avg_joules_per_operation: if total_ops > 0 {
            total_energy / total_ops as f64
        } else {
            0.0
        },
    }
}

/// Formats a daily energy report as a pretty-printed ASCII table.
pub fn format_report(report: &DailyEnergyReport) -> String {
    let mut lines = Vec::new();

    lines.push(format!("Daily Energy Report — {}", report.date));
    lines.push("━".repeat(66));
    lines.push(format!(
        "{:<28} {:>14} {:>18}",
        "Work done", "Total energy", "Per unit"
    ));
    lines.push("─".repeat(66));

    for row in &report.services {
        let work = format!(
            "{} — {} {}",
            row.service,
            format_count(row.work_count),
            row.work_unit
        );
        let total = format_joules(row.total_joules);
        let per_unit = format!("{:.4} J/{}", row.per_unit_joules, singular(&row.work_unit));
        lines.push(format!("{:<28} {:>14} {:>18}", work, total, per_unit));
    }

    lines.push("─".repeat(66));
    lines.push(format!(
        "{:<28} {:>14} {:>18}",
        format!("Total — {} ops", format_count(report.total_operations)),
        format_joules(report.total_energy_joules),
        format!("{:.4} J/op", report.avg_joules_per_operation),
    ));

    lines.join("\n")
}

/// Format a count with k/M suffixes for readability.
fn format_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// Format joules with comma separators.
fn format_joules(j: f64) -> String {
    if j >= 1_000_000.0 {
        format!("{:.0} J", j)
    } else if j >= 1_000.0 {
        let whole = j as u64;
        let thousands = whole / 1000;
        let remainder = whole % 1000;
        format!("{},{:03} J", thousands, remainder)
    } else if j >= 1.0 {
        format!("{:.1} J", j)
    } else {
        format!("{:.4} J", j)
    }
}

/// Convert plural work unit to singular for per-unit display.
fn singular(unit: &str) -> &str {
    match unit {
        "requests" => "req",
        "queries" => "query",
        "ops" => "op",
        "invocations" => "call",
        "messages" => "msg",
        "records" => "rec",
        "events" => "event",
        "writes" => "write",
        "lookups" => "lookup",
        "connections" => "conn",
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_meter_produces_empty_report() {
        let meter = JouleMeter::new();
        let report = generate_daily_report(&meter, "2026-03-01");
        assert_eq!(report.date, "2026-03-01");
        assert!(report.services.is_empty());
        assert!((report.total_energy_joules).abs() < 1e-10);
        assert_eq!(report.total_operations, 0);
    }

    #[test]
    fn categorizes_by_prefix() {
        let mut meter = JouleMeter::new();
        meter.record("api_request", 0.34);
        meter.record("api_request", 0.36);
        meter.record("db_query", 0.08);
        meter.record("db_query", 0.07);
        meter.record("db_query", 0.09);

        let report = generate_daily_report(&meter, "2026-03-01");
        assert_eq!(report.services.len(), 2);

        let api = report.services.iter().find(|s| s.service == "API").unwrap();
        assert_eq!(api.work_count, 2);
        assert!((api.total_joules - 0.70).abs() < 1e-10);
        assert!((api.per_unit_joules - 0.35).abs() < 1e-10);
        assert_eq!(api.work_unit, "requests");

        let db = report
            .services
            .iter()
            .find(|s| s.service == "Database")
            .unwrap();
        assert_eq!(db.work_count, 3);
        assert!((db.total_joules - 0.24).abs() < 1e-10);
        assert_eq!(db.work_unit, "queries");
    }

    #[test]
    fn unmatched_ops_go_to_other() {
        let mut meter = JouleMeter::new();
        meter.record("custom_workflow", 1.5);
        meter.record("custom_workflow", 2.5);

        let report = generate_daily_report(&meter, "2026-03-01");
        assert_eq!(report.services.len(), 1);
        assert_eq!(report.services[0].service, "Other");
        assert_eq!(report.services[0].work_count, 2);
        assert!((report.services[0].total_joules - 4.0).abs() < 1e-10);
    }

    #[test]
    fn sorted_by_total_energy_descending() {
        let mut meter = JouleMeter::new();
        // API: 2 ops, 10J total
        meter.record("api_request", 5.0);
        meter.record("api_request", 5.0);
        // DB: 100 ops, 80J total
        for _ in 0..100 {
            meter.record("db_query", 0.8);
        }
        // Functions: 1000 ops, 3J total
        for _ in 0..1000 {
            meter.record("function_invoke", 0.003);
        }

        let report = generate_daily_report(&meter, "2026-03-01");
        assert_eq!(report.services[0].service, "Database");
        assert_eq!(report.services[1].service, "API");
        assert_eq!(report.services[2].service, "Functions");
    }

    #[test]
    fn totals_correct() {
        let mut meter = JouleMeter::new();
        for _ in 0..100_000 {
            meter.record("api_request", 0.34);
        }
        for _ in 0..50_000 {
            meter.record("db_query", 0.08);
        }

        let report = generate_daily_report(&meter, "2026-03-01");
        assert_eq!(report.total_operations, 150_000);
        let expected_energy = 100_000.0 * 0.34 + 50_000.0 * 0.08;
        assert!((report.total_energy_joules - expected_energy).abs() < 1.0);
    }

    #[test]
    fn format_report_output() {
        let mut meter = JouleMeter::new();
        for _ in 0..100 {
            meter.record("api_request", 0.34);
        }
        for _ in 0..50 {
            meter.record("db_query", 0.08);
        }

        let report = generate_daily_report(&meter, "2026-03-01");
        let formatted = format_report(&report);
        assert!(formatted.contains("Daily Energy Report — 2026-03-01"));
        assert!(formatted.contains("API"));
        assert!(formatted.contains("Database"));
        assert!(formatted.contains("J/req"));
        assert!(formatted.contains("J/query"));
    }

    #[test]
    fn format_count_suffixes() {
        assert_eq!(format_count(500), "500");
        assert_eq!(format_count(1_500), "1.5k");
        assert_eq!(format_count(100_000), "100.0k");
        assert_eq!(format_count(1_500_000), "1.5M");
    }

    #[test]
    fn format_joules_ranges() {
        assert_eq!(format_joules(0.003), "0.0030 J");
        assert_eq!(format_joules(5.5), "5.5 J");
        assert_eq!(format_joules(34_000.0), "34,000 J");
    }

    #[test]
    fn multiple_ops_same_service() {
        let mut meter = JouleMeter::new();
        meter.record("api_request", 0.34);
        meter.record("api_health", 0.01);
        meter.record("api_webhook", 0.15);

        let report = generate_daily_report(&meter, "2026-03-01");
        let api = report.services.iter().find(|s| s.service == "API").unwrap();
        assert_eq!(api.work_count, 3);
        assert!((api.total_joules - 0.50).abs() < 1e-10);
    }

    #[test]
    fn all_service_categories_recognized() {
        let mut meter = JouleMeter::new();
        let ops = [
            "api_req",
            "db_query",
            "object_get",
            "inference_run",
            "function_exec",
            "cache_get",
            "queue_send",
            "stream_produce",
            "search_query",
            "pubsub_publish",
            "iot_telemetry",
            "ts_write",
            "cdn_fetch",
            "dns_resolve",
            "lb_route",
        ];
        for op in &ops {
            meter.record(op, 1.0);
        }

        let report = generate_daily_report(&meter, "2026-03-01");
        assert_eq!(report.services.len(), 15);
        assert!(report.services.iter().all(|s| s.service != "Other"));
    }
}
