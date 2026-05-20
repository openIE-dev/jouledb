// debug_toolbar.rs — Debug toolbar: panel system, request timing,
// query log, cache stats, memory, middleware timing, HTML rendering,
// and panel enable/disable.

use std::collections::HashMap;

/// A named panel that can be enabled/disabled and holds structured data.
#[derive(Debug, Clone)]
pub struct Panel {
    pub id: String,
    pub title: String,
    pub enabled: bool,
    pub entries: Vec<PanelEntry>,
}

/// One entry in a panel (key-value pair with optional unit).
#[derive(Debug, Clone)]
pub struct PanelEntry {
    pub label: String,
    pub value: String,
    pub unit: Option<String>,
}

impl PanelEntry {
    pub fn new(label: &str, value: &str) -> Self {
        Self {
            label: label.to_string(),
            value: value.to_string(),
            unit: None,
        }
    }

    pub fn with_unit(mut self, unit: &str) -> Self {
        self.unit = Some(unit.to_string());
        self
    }

    pub fn display_value(&self) -> String {
        match &self.unit {
            Some(u) => format!("{} {u}", self.value),
            None => self.value.clone(),
        }
    }
}

impl Panel {
    pub fn new(id: &str, title: &str) -> Self {
        Self {
            id: id.to_string(),
            title: title.to_string(),
            enabled: true,
            entries: Vec::new(),
        }
    }

    pub fn add_entry(&mut self, entry: PanelEntry) {
        self.entries.push(entry);
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn render_html(&self) -> String {
        if !self.enabled {
            return String::new();
        }
        let mut html = format!(
            "<div class=\"debug-panel\" id=\"panel-{}\">\n  <h3>{}</h3>\n  <table>\n",
            self.id, self.title
        );
        for e in &self.entries {
            html.push_str(&format!(
                "    <tr><td>{}</td><td>{}</td></tr>\n",
                e.label,
                e.display_value()
            ));
        }
        html.push_str("  </table>\n</div>\n");
        html
    }
}

// ---------------------------------------------------------------------------
// Timing panel
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TimingRecord {
    pub label: String,
    pub duration_us: u64,
}

#[derive(Debug, Clone, Default)]
pub struct RequestTimingPanel {
    records: Vec<TimingRecord>,
}

impl RequestTimingPanel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, label: &str, duration_us: u64) {
        self.records.push(TimingRecord {
            label: label.to_string(),
            duration_us,
        });
    }

    pub fn total_us(&self) -> u64 {
        self.records.iter().map(|r| r.duration_us).sum()
    }

    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    pub fn slowest(&self) -> Option<&TimingRecord> {
        self.records.iter().max_by_key(|r| r.duration_us)
    }

    pub fn to_panel(&self) -> Panel {
        let mut panel = Panel::new("timing", "Request Timing");
        for r in &self.records {
            panel.add_entry(
                PanelEntry::new(&r.label, &r.duration_us.to_string()).with_unit("us"),
            );
        }
        panel.add_entry(
            PanelEntry::new("Total", &self.total_us().to_string()).with_unit("us"),
        );
        panel
    }
}

// ---------------------------------------------------------------------------
// Query log panel
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct QueryRecord {
    pub sql: String,
    pub duration_us: u64,
    pub rows_affected: u64,
}

#[derive(Debug, Clone, Default)]
pub struct QueryLogPanel {
    queries: Vec<QueryRecord>,
}

impl QueryLogPanel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn log(&mut self, sql: &str, duration_us: u64, rows: u64) {
        self.queries.push(QueryRecord {
            sql: sql.to_string(),
            duration_us,
            rows_affected: rows,
        });
    }

    pub fn query_count(&self) -> usize {
        self.queries.len()
    }

    pub fn total_duration_us(&self) -> u64 {
        self.queries.iter().map(|q| q.duration_us).sum()
    }

    pub fn slowest(&self) -> Option<&QueryRecord> {
        self.queries.iter().max_by_key(|q| q.duration_us)
    }

    pub fn total_rows(&self) -> u64 {
        self.queries.iter().map(|q| q.rows_affected).sum()
    }

    pub fn to_panel(&self) -> Panel {
        let mut panel = Panel::new("queries", "Query Log");
        for (i, q) in self.queries.iter().enumerate() {
            let label = format!("Q{}", i + 1);
            let val = format!("{} ({}us, {} rows)", q.sql, q.duration_us, q.rows_affected);
            panel.add_entry(PanelEntry::new(&label, &val));
        }
        panel.add_entry(PanelEntry::new(
            "Total Queries",
            &self.query_count().to_string(),
        ));
        panel.add_entry(
            PanelEntry::new("Total Duration", &self.total_duration_us().to_string())
                .with_unit("us"),
        );
        panel
    }
}

// ---------------------------------------------------------------------------
// Cache stats panel
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct CacheStatsPanel {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub size_bytes: u64,
}

impl CacheStatsPanel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_hit(&mut self) {
        self.hits += 1;
    }

    pub fn record_miss(&mut self) {
        self.misses += 1;
    }

    pub fn record_eviction(&mut self) {
        self.evictions += 1;
    }

    pub fn set_size(&mut self, bytes: u64) {
        self.size_bytes = bytes;
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    pub fn total_lookups(&self) -> u64 {
        self.hits + self.misses
    }

    pub fn to_panel(&self) -> Panel {
        let mut panel = Panel::new("cache", "Cache Stats");
        panel.add_entry(PanelEntry::new("Hits", &self.hits.to_string()));
        panel.add_entry(PanelEntry::new("Misses", &self.misses.to_string()));
        panel.add_entry(PanelEntry::new(
            "Hit Rate",
            &format!("{:.1}%", self.hit_rate() * 100.0),
        ));
        panel.add_entry(PanelEntry::new("Evictions", &self.evictions.to_string()));
        panel.add_entry(
            PanelEntry::new("Size", &self.size_bytes.to_string()).with_unit("bytes"),
        );
        panel
    }
}

// ---------------------------------------------------------------------------
// Memory panel
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct MemoryPanel {
    pub heap_bytes: u64,
    pub stack_bytes: u64,
    pub allocations: u64,
    pub deallocations: u64,
}

impl MemoryPanel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn net_allocations(&self) -> i64 {
        self.allocations as i64 - self.deallocations as i64
    }

    pub fn total_bytes(&self) -> u64 {
        self.heap_bytes + self.stack_bytes
    }

    pub fn to_panel(&self) -> Panel {
        let mut panel = Panel::new("memory", "Memory");
        panel.add_entry(
            PanelEntry::new("Heap", &self.heap_bytes.to_string()).with_unit("bytes"),
        );
        panel.add_entry(
            PanelEntry::new("Stack", &self.stack_bytes.to_string()).with_unit("bytes"),
        );
        panel.add_entry(PanelEntry::new(
            "Allocations",
            &self.allocations.to_string(),
        ));
        panel.add_entry(PanelEntry::new(
            "Deallocations",
            &self.deallocations.to_string(),
        ));
        panel.add_entry(PanelEntry::new(
            "Net Live",
            &self.net_allocations().to_string(),
        ));
        panel
    }
}

// ---------------------------------------------------------------------------
// Middleware timing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MiddlewareTiming {
    pub name: String,
    pub duration_us: u64,
}

#[derive(Debug, Clone, Default)]
pub struct MiddlewareTimingPanel {
    timings: Vec<MiddlewareTiming>,
}

impl MiddlewareTimingPanel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, name: &str, duration_us: u64) {
        self.timings.push(MiddlewareTiming {
            name: name.to_string(),
            duration_us,
        });
    }

    pub fn total_us(&self) -> u64 {
        self.timings.iter().map(|t| t.duration_us).sum()
    }

    pub fn count(&self) -> usize {
        self.timings.len()
    }

    pub fn to_panel(&self) -> Panel {
        let mut panel = Panel::new("middleware", "Middleware Timing");
        for t in &self.timings {
            panel.add_entry(
                PanelEntry::new(&t.name, &t.duration_us.to_string()).with_unit("us"),
            );
        }
        panel.add_entry(
            PanelEntry::new("Total", &self.total_us().to_string()).with_unit("us"),
        );
        panel
    }
}

// ---------------------------------------------------------------------------
// Debug toolbar (aggregates panels)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct DebugToolbar {
    panels: Vec<Panel>,
    enabled_ids: HashMap<String, bool>,
}

impl DebugToolbar {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_panel(&mut self, panel: Panel) {
        self.enabled_ids
            .insert(panel.id.clone(), panel.enabled);
        self.panels.push(panel);
    }

    pub fn enable_panel(&mut self, id: &str) {
        self.enabled_ids.insert(id.to_string(), true);
        for p in &mut self.panels {
            if p.id == id {
                p.enabled = true;
            }
        }
    }

    pub fn disable_panel(&mut self, id: &str) {
        self.enabled_ids.insert(id.to_string(), false);
        for p in &mut self.panels {
            if p.id == id {
                p.enabled = false;
            }
        }
    }

    pub fn is_panel_enabled(&self, id: &str) -> bool {
        self.enabled_ids.get(id).copied().unwrap_or(false)
    }

    pub fn panel_count(&self) -> usize {
        self.panels.len()
    }

    pub fn enabled_panel_count(&self) -> usize {
        self.panels.iter().filter(|p| p.enabled).count()
    }

    pub fn get_panel(&self, id: &str) -> Option<&Panel> {
        self.panels.iter().find(|p| p.id == id)
    }

    /// Render the full toolbar as an HTML string.
    pub fn render_html(&self) -> String {
        let mut html = String::from(
            "<div id=\"debug-toolbar\" style=\"position:fixed;bottom:0;width:100%;background:#222;color:#eee;font-family:monospace;font-size:12px;z-index:9999;\">\n",
        );
        html.push_str("  <div style=\"padding:4px 8px;font-weight:bold;\">Debug Toolbar</div>\n");
        for panel in &self.panels {
            html.push_str(&panel.render_html());
        }
        html.push_str("</div>\n");
        html
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_panel_entry_display() {
        let e = PanelEntry::new("Time", "42").with_unit("ms");
        assert_eq!(e.display_value(), "42 ms");
    }

    #[test]
    fn test_panel_entry_no_unit() {
        let e = PanelEntry::new("Count", "7");
        assert_eq!(e.display_value(), "7");
    }

    #[test]
    fn test_panel_add_and_count() {
        let mut p = Panel::new("test", "Test Panel");
        p.add_entry(PanelEntry::new("a", "1"));
        p.add_entry(PanelEntry::new("b", "2"));
        assert_eq!(p.entry_count(), 2);
    }

    #[test]
    fn test_panel_clear() {
        let mut p = Panel::new("test", "Test");
        p.add_entry(PanelEntry::new("x", "1"));
        p.clear();
        assert_eq!(p.entry_count(), 0);
    }

    #[test]
    fn test_panel_render_html() {
        let mut p = Panel::new("test", "Test");
        p.add_entry(PanelEntry::new("Key", "Val"));
        let html = p.render_html();
        assert!(html.contains("panel-test"));
        assert!(html.contains("<h3>Test</h3>"));
        assert!(html.contains("Key"));
        assert!(html.contains("Val"));
    }

    #[test]
    fn test_panel_disabled_render_empty() {
        let mut p = Panel::new("off", "Off");
        p.enabled = false;
        assert_eq!(p.render_html(), "");
    }

    #[test]
    fn test_request_timing_panel() {
        let mut t = RequestTimingPanel::new();
        t.record("parse", 100);
        t.record("db", 500);
        t.record("render", 200);
        assert_eq!(t.total_us(), 800);
        assert_eq!(t.record_count(), 3);
        assert_eq!(t.slowest().unwrap().label, "db");

        let panel = t.to_panel();
        assert_eq!(panel.id, "timing");
        // 3 records + 1 total entry.
        assert_eq!(panel.entry_count(), 4);
    }

    #[test]
    fn test_request_timing_empty() {
        let t = RequestTimingPanel::new();
        assert_eq!(t.total_us(), 0);
        assert!(t.slowest().is_none());
    }

    #[test]
    fn test_query_log_panel() {
        let mut ql = QueryLogPanel::new();
        ql.log("SELECT 1", 50, 1);
        ql.log("INSERT INTO t VALUES (1)", 200, 1);
        assert_eq!(ql.query_count(), 2);
        assert_eq!(ql.total_duration_us(), 250);
        assert_eq!(ql.total_rows(), 2);
        assert_eq!(ql.slowest().unwrap().sql, "INSERT INTO t VALUES (1)");

        let panel = ql.to_panel();
        assert_eq!(panel.id, "queries");
    }

    #[test]
    fn test_cache_stats_panel() {
        let mut cs = CacheStatsPanel::new();
        cs.record_hit(); cs.record_hit(); cs.record_miss(); cs.record_eviction(); cs.set_size(4096);
        assert_eq!(cs.total_lookups(), 3);
        assert!((cs.hit_rate() - 2.0 / 3.0).abs() < 0.001);
        assert_eq!(cs.to_panel().id, "cache");
        assert!((CacheStatsPanel::new().hit_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_memory_panel() {
        let mp = MemoryPanel { heap_bytes: 1024, stack_bytes: 256, allocations: 100, deallocations: 30 };
        assert_eq!(mp.total_bytes(), 1280);
        assert_eq!(mp.net_allocations(), 70);
        assert_eq!(mp.to_panel().id, "memory");
    }

    #[test]
    fn test_middleware_timing_panel() {
        let mut mw = MiddlewareTimingPanel::new();
        mw.record("auth", 50); mw.record("cors", 10); mw.record("logging", 30);
        assert_eq!(mw.count(), 3);
        assert_eq!(mw.total_us(), 90);
        assert_eq!(mw.to_panel().id, "middleware");
    }

    #[test]
    fn test_toolbar_enable_disable_render() {
        let mut tb = DebugToolbar::new();
        tb.add_panel(Panel::new("a", "A"));
        tb.add_panel(Panel::new("b", "B"));
        assert_eq!(tb.panel_count(), 2);
        assert_eq!(tb.enabled_panel_count(), 2);
        tb.disable_panel("a");
        assert!(!tb.is_panel_enabled("a"));
        assert_eq!(tb.enabled_panel_count(), 1);
        tb.enable_panel("a");
        assert!(tb.is_panel_enabled("a"));
        assert!(tb.get_panel("a").is_some());
        assert!(tb.get_panel("z").is_none());
        let mut p = Panel::new("demo", "Demo");
        p.add_entry(PanelEntry::new("Item", "Value"));
        tb.add_panel(p);
        let html = tb.render_html();
        assert!(html.contains("debug-toolbar") && html.contains("Demo"));
        tb.disable_panel("demo");
        assert!(!tb.render_html().contains("panel-demo"));
    }

    #[test]
    fn test_full_workflow() {
        let mut timing = RequestTimingPanel::new();
        timing.record("handler", 1500);
        let mut queries = QueryLogPanel::new();
        queries.log("SELECT * FROM users", 300, 10);
        let mut cache = CacheStatsPanel::new();
        cache.record_hit();
        let mem = MemoryPanel { heap_bytes: 8192, stack_bytes: 512, allocations: 50, deallocations: 10 };
        let mut mw = MiddlewareTimingPanel::new();
        mw.record("auth", 80);
        let mut tb = DebugToolbar::new();
        tb.add_panel(timing.to_panel());
        tb.add_panel(queries.to_panel());
        tb.add_panel(cache.to_panel());
        tb.add_panel(mem.to_panel());
        tb.add_panel(mw.to_panel());
        assert_eq!(tb.panel_count(), 5);
        let html = tb.render_html();
        assert!(html.contains("Request Timing") && html.contains("Query Log"));
        assert!(html.contains("Cache Stats") && html.contains("Memory"));
    }
}
