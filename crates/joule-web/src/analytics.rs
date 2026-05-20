//! Analytics framework — event tracking, page views, user properties,
//! session tracking, funnel analysis, cohort assignment, and privacy-respecting consent.
//!
//! Pure Rust analytics engine. No external HTTP or browser deps.

use std::collections::HashMap;
use std::fmt;

// ── Consent ──────────────────────────────────────────────────────

/// Privacy consent levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConsentLevel {
    /// No tracking at all.
    None,
    /// Essential analytics only (page views, no PII).
    Essential,
    /// Functional analytics (sessions, funnels).
    Functional,
    /// Full analytics including user properties.
    Full,
}

impl fmt::Display for ConsentLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Essential => write!(f, "essential"),
            Self::Functional => write!(f, "functional"),
            Self::Full => write!(f, "full"),
        }
    }
}

/// Consent manager tracks user's privacy choices.
#[derive(Debug, Clone)]
pub struct ConsentManager {
    level: ConsentLevel,
    categories: HashMap<String, bool>,
}

impl ConsentManager {
    pub fn new(level: ConsentLevel) -> Self {
        Self { level, categories: HashMap::new() }
    }

    pub fn set_level(&mut self, level: ConsentLevel) { self.level = level; }
    pub fn level(&self) -> ConsentLevel { self.level }

    /// Set consent for a specific category (e.g., "marketing", "performance").
    pub fn set_category(&mut self, cat: &str, allowed: bool) {
        self.categories.insert(cat.to_string(), allowed);
    }

    /// Check if a specific category is consented.
    pub fn is_category_allowed(&self, cat: &str) -> bool {
        *self.categories.get(cat).unwrap_or(&false)
    }

    /// Check if tracking is allowed at a given minimum level.
    pub fn allows(&self, min_level: ConsentLevel) -> bool {
        self.level >= min_level
    }
}

// ── Event ────────────────────────────────────────────────────────

/// Analytics event.
#[derive(Debug, Clone)]
pub struct Event {
    pub name: String,
    pub category: Option<String>,
    pub action: Option<String>,
    pub label: Option<String>,
    pub value: Option<f64>,
    pub properties: HashMap<String, EventValue>,
    pub timestamp_ms: u64,
}

/// Typed event property value.
#[derive(Debug, Clone, PartialEq)]
pub enum EventValue {
    Str(String),
    Num(f64),
    Bool(bool),
    Null,
}

impl fmt::Display for EventValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Str(s) => write!(f, "{s}"),
            Self::Num(n) => write!(f, "{n}"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::Null => write!(f, "null"),
        }
    }
}

impl Event {
    pub fn new(name: impl Into<String>, timestamp_ms: u64) -> Self {
        Self {
            name: name.into(),
            category: None,
            action: None,
            label: None,
            value: None,
            properties: HashMap::new(),
            timestamp_ms,
        }
    }

    pub fn category(mut self, c: impl Into<String>) -> Self { self.category = Some(c.into()); self }
    pub fn action(mut self, a: impl Into<String>) -> Self { self.action = Some(a.into()); self }
    pub fn label(mut self, l: impl Into<String>) -> Self { self.label = Some(l.into()); self }
    pub fn value(mut self, v: f64) -> Self { self.value = Some(v); self }

    pub fn set_str(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.properties.insert(key.into(), EventValue::Str(val.into()));
        self
    }

    pub fn set_num(mut self, key: impl Into<String>, val: f64) -> Self {
        self.properties.insert(key.into(), EventValue::Num(val));
        self
    }

    pub fn set_bool(mut self, key: impl Into<String>, val: bool) -> Self {
        self.properties.insert(key.into(), EventValue::Bool(val));
        self
    }

    /// Required consent level: events with PII need Full, basic events need Essential.
    pub fn min_consent(&self) -> ConsentLevel {
        if self.properties.keys().any(|k| {
            k.contains("email") || k.contains("user_id") || k.contains("phone") || k.contains("name")
        }) {
            ConsentLevel::Full
        } else {
            ConsentLevel::Essential
        }
    }
}

// ── Page View ────────────────────────────────────────────────────

/// Standardized page view event.
#[derive(Debug, Clone)]
pub struct PageView {
    pub path: String,
    pub title: Option<String>,
    pub referrer: Option<String>,
    pub timestamp_ms: u64,
}

impl PageView {
    pub fn new(path: impl Into<String>, timestamp_ms: u64) -> Self {
        Self { path: path.into(), title: None, referrer: None, timestamp_ms }
    }

    pub fn title(mut self, t: impl Into<String>) -> Self { self.title = Some(t.into()); self }
    pub fn referrer(mut self, r: impl Into<String>) -> Self { self.referrer = Some(r.into()); self }

    /// Convert to a generic event.
    pub fn to_event(&self) -> Event {
        let mut e = Event::new("page_view", self.timestamp_ms)
            .set_str("path", &self.path);
        if let Some(ref t) = self.title {
            e = e.set_str("title", t);
        }
        if let Some(ref r) = self.referrer {
            e = e.set_str("referrer", r);
        }
        e
    }
}

// ── User Properties ──────────────────────────────────────────────

/// User properties for analytics profiling.
#[derive(Debug, Clone, Default)]
pub struct UserProperties {
    props: HashMap<String, EventValue>,
}

impl UserProperties {
    pub fn new() -> Self { Self::default() }

    pub fn set(mut self, key: impl Into<String>, val: EventValue) -> Self {
        self.props.insert(key.into(), val);
        self
    }

    pub fn set_str(self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.set(key, EventValue::Str(val.into()))
    }

    pub fn set_num(self, key: impl Into<String>, val: f64) -> Self {
        self.set(key, EventValue::Num(val))
    }

    pub fn get(&self, key: &str) -> Option<&EventValue> {
        self.props.get(key)
    }

    pub fn count(&self) -> usize { self.props.len() }
}

// ── Session ──────────────────────────────────────────────────────

/// Session tracking.
#[derive(Debug, Clone)]
pub struct Session {
    pub session_id: String,
    pub start_ms: u64,
    pub last_activity_ms: u64,
    pub page_views: u32,
    pub events: u32,
    pub timeout_ms: u64,
}

impl Session {
    pub fn new(session_id: impl Into<String>, start_ms: u64) -> Self {
        Self {
            session_id: session_id.into(),
            start_ms,
            last_activity_ms: start_ms,
            page_views: 0,
            events: 0,
            timeout_ms: 30 * 60 * 1000, // 30 minutes
        }
    }

    pub fn with_timeout(mut self, ms: u64) -> Self { self.timeout_ms = ms; self }

    /// Record activity at a given timestamp.
    pub fn touch(&mut self, timestamp_ms: u64) {
        self.last_activity_ms = timestamp_ms;
    }

    /// Record a page view.
    pub fn record_page_view(&mut self, timestamp_ms: u64) {
        self.page_views += 1;
        self.touch(timestamp_ms);
    }

    /// Record an event.
    pub fn record_event(&mut self, timestamp_ms: u64) {
        self.events += 1;
        self.touch(timestamp_ms);
    }

    /// Check if session has expired.
    pub fn is_expired(&self, current_ms: u64) -> bool {
        current_ms.saturating_sub(self.last_activity_ms) > self.timeout_ms
    }

    /// Duration of the session in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.last_activity_ms.saturating_sub(self.start_ms)
    }
}

// ── Funnel Analysis ──────────────────────────────────────────────

/// Funnel definition with ordered steps.
#[derive(Debug, Clone)]
pub struct Funnel {
    pub name: String,
    pub steps: Vec<FunnelStep>,
}

/// A single step in a funnel.
#[derive(Debug, Clone)]
pub struct FunnelStep {
    pub name: String,
    pub event_name: String,
    pub count: u64,
}

impl FunnelStep {
    pub fn new(name: impl Into<String>, event_name: impl Into<String>) -> Self {
        Self { name: name.into(), event_name: event_name.into(), count: 0 }
    }
}

impl Funnel {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), steps: Vec::new() }
    }

    pub fn step(mut self, s: FunnelStep) -> Self {
        self.steps.push(s);
        self
    }

    /// Record an event, incrementing the matching step.
    pub fn record(&mut self, event_name: &str) {
        for step in &mut self.steps {
            if step.event_name == event_name {
                step.count += 1;
            }
        }
    }

    /// Calculate conversion rate between consecutive steps.
    pub fn conversion_rates(&self) -> Vec<FunnelConversion> {
        let mut rates = Vec::new();
        for window in self.steps.windows(2) {
            let from = &window[0];
            let to = &window[1];
            let rate = if from.count > 0 {
                to.count as f64 / from.count as f64
            } else {
                0.0
            };
            rates.push(FunnelConversion {
                from_step: from.name.clone(),
                to_step: to.name.clone(),
                from_count: from.count,
                to_count: to.count,
                rate,
            });
        }
        rates
    }

    /// Overall funnel conversion (first step to last step).
    pub fn overall_conversion(&self) -> f64 {
        if self.steps.len() < 2 { return 0.0; }
        let first = self.steps.first().unwrap().count;
        let last = self.steps.last().unwrap().count;
        if first > 0 { last as f64 / first as f64 } else { 0.0 }
    }

    /// Drop-off count between each step.
    pub fn dropoffs(&self) -> Vec<(String, u64)> {
        self.steps.windows(2).map(|w| {
            let drop = w[0].count.saturating_sub(w[1].count);
            (w[0].name.clone(), drop)
        }).collect()
    }

    /// Number of steps.
    pub fn step_count(&self) -> usize { self.steps.len() }
}

/// Conversion data between two funnel steps.
#[derive(Debug, Clone)]
pub struct FunnelConversion {
    pub from_step: String,
    pub to_step: String,
    pub from_count: u64,
    pub to_count: u64,
    pub rate: f64,
}

// ── Cohort ───────────────────────────────────────────────────────

/// Cohort assignment based on user property or date.
#[derive(Debug, Clone)]
pub struct Cohort {
    pub name: String,
    pub members: Vec<String>,
}

impl Cohort {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), members: Vec::new() }
    }

    pub fn add_member(&mut self, user_id: impl Into<String>) {
        let id = user_id.into();
        if !self.members.contains(&id) {
            self.members.push(id);
        }
    }

    pub fn contains(&self, user_id: &str) -> bool {
        self.members.iter().any(|m| m == user_id)
    }

    pub fn size(&self) -> usize { self.members.len() }
}

/// Assign users to weekly cohorts based on signup timestamp.
pub fn weekly_cohort(signup_day: u32, user_id: &str, cohorts: &mut HashMap<u32, Cohort>) {
    let week = signup_day / 7;
    let cohort = cohorts.entry(week).or_insert_with(|| Cohort::new(format!("week-{week}")));
    cohort.add_member(user_id);
}

// ── Event Collector ──────────────────────────────────────────────

/// Batched event collector with consent enforcement.
#[derive(Debug)]
pub struct EventCollector {
    events: Vec<Event>,
    consent: ConsentManager,
    max_batch_size: usize,
}

impl EventCollector {
    pub fn new(consent: ConsentManager) -> Self {
        Self { events: Vec::new(), consent, max_batch_size: 100 }
    }

    pub fn with_batch_size(mut self, size: usize) -> Self { self.max_batch_size = size; self }

    /// Track an event, respecting consent.
    pub fn track(&mut self, event: Event) -> bool {
        if !self.consent.allows(event.min_consent()) {
            return false;
        }
        self.events.push(event);
        true
    }

    /// Track a page view.
    pub fn track_page_view(&mut self, pv: PageView) -> bool {
        if !self.consent.allows(ConsentLevel::Essential) {
            return false;
        }
        self.events.push(pv.to_event());
        true
    }

    /// Drain events up to batch_size for sending.
    pub fn drain_batch(&mut self) -> Vec<Event> {
        let n = self.events.len().min(self.max_batch_size);
        self.events.drain(..n).collect()
    }

    /// Number of pending events.
    pub fn pending_count(&self) -> usize { self.events.len() }

    /// Update consent level.
    pub fn set_consent(&mut self, level: ConsentLevel) {
        self.consent.set_level(level);
    }

    /// Current consent level.
    pub fn consent_level(&self) -> ConsentLevel { self.consent.level() }

    /// Clear all pending events.
    pub fn clear(&mut self) { self.events.clear(); }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consent_levels() {
        assert!(ConsentLevel::Full > ConsentLevel::Functional);
        assert!(ConsentLevel::Functional > ConsentLevel::Essential);
        assert!(ConsentLevel::Essential > ConsentLevel::None);
    }

    #[test]
    fn test_consent_manager() {
        let mut cm = ConsentManager::new(ConsentLevel::Essential);
        assert!(cm.allows(ConsentLevel::Essential));
        assert!(!cm.allows(ConsentLevel::Full));

        cm.set_level(ConsentLevel::Full);
        assert!(cm.allows(ConsentLevel::Full));
    }

    #[test]
    fn test_consent_categories() {
        let mut cm = ConsentManager::new(ConsentLevel::Functional);
        assert!(!cm.is_category_allowed("marketing"));

        cm.set_category("marketing", true);
        assert!(cm.is_category_allowed("marketing"));
    }

    #[test]
    fn test_event_creation() {
        let e = Event::new("click", 1000)
            .category("ui")
            .action("button_click")
            .label("submit")
            .value(1.0)
            .set_str("page", "/home")
            .set_num("x", 100.0)
            .set_bool("first_time", true);

        assert_eq!(e.name, "click");
        assert_eq!(e.category.as_deref(), Some("ui"));
        assert_eq!(e.properties.get("page"), Some(&EventValue::Str("/home".into())));
        assert_eq!(e.properties.get("x"), Some(&EventValue::Num(100.0)));
    }

    #[test]
    fn test_event_min_consent_basic() {
        let e = Event::new("click", 1000);
        assert_eq!(e.min_consent(), ConsentLevel::Essential);
    }

    #[test]
    fn test_event_min_consent_pii() {
        let e = Event::new("signup", 1000)
            .set_str("email", "user@example.com");
        assert_eq!(e.min_consent(), ConsentLevel::Full);
    }

    #[test]
    fn test_page_view() {
        let pv = PageView::new("/about", 5000)
            .title("About Us")
            .referrer("https://google.com");

        assert_eq!(pv.path, "/about");
        assert_eq!(pv.title.as_deref(), Some("About Us"));

        let event = pv.to_event();
        assert_eq!(event.name, "page_view");
        assert_eq!(event.properties.get("path"), Some(&EventValue::Str("/about".into())));
    }

    #[test]
    fn test_user_properties() {
        let up = UserProperties::new()
            .set_str("plan", "pro")
            .set_num("age", 30.0);

        assert_eq!(up.count(), 2);
        assert_eq!(up.get("plan"), Some(&EventValue::Str("pro".into())));
        assert_eq!(up.get("age"), Some(&EventValue::Num(30.0)));
    }

    #[test]
    fn test_session_basic() {
        let mut s = Session::new("sess-1", 0);
        s.record_page_view(1000);
        s.record_page_view(5000);
        s.record_event(6000);

        assert_eq!(s.page_views, 2);
        assert_eq!(s.events, 1);
        assert_eq!(s.duration_ms(), 6000);
    }

    #[test]
    fn test_session_expiry() {
        let s = Session::new("sess-1", 0).with_timeout(10_000);
        assert!(!s.is_expired(5000));
        assert!(s.is_expired(15_000));
    }

    #[test]
    fn test_session_timeout_default() {
        let s = Session::new("s", 0);
        assert_eq!(s.timeout_ms, 30 * 60 * 1000);
    }

    #[test]
    fn test_funnel_basic() {
        let mut funnel = Funnel::new("signup")
            .step(FunnelStep::new("Landing", "page_view"))
            .step(FunnelStep::new("Sign Up Form", "signup_start"))
            .step(FunnelStep::new("Completed", "signup_complete"));

        for _ in 0..100 { funnel.record("page_view"); }
        for _ in 0..60 { funnel.record("signup_start"); }
        for _ in 0..30 { funnel.record("signup_complete"); }

        let rates = funnel.conversion_rates();
        assert_eq!(rates.len(), 2);
        assert!((rates[0].rate - 0.6).abs() < 0.001);
        assert!((rates[1].rate - 0.5).abs() < 0.001);

        assert!((funnel.overall_conversion() - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_funnel_dropoffs() {
        let mut funnel = Funnel::new("checkout")
            .step(FunnelStep::new("Cart", "view_cart"))
            .step(FunnelStep::new("Shipping", "enter_shipping"))
            .step(FunnelStep::new("Payment", "enter_payment"));

        for _ in 0..100 { funnel.record("view_cart"); }
        for _ in 0..70 { funnel.record("enter_shipping"); }
        for _ in 0..50 { funnel.record("enter_payment"); }

        let dropoffs = funnel.dropoffs();
        assert_eq!(dropoffs[0].1, 30); // cart->shipping
        assert_eq!(dropoffs[1].1, 20); // shipping->payment
    }

    #[test]
    fn test_funnel_empty() {
        let funnel = Funnel::new("empty");
        assert_eq!(funnel.overall_conversion(), 0.0);
        assert!(funnel.conversion_rates().is_empty());
    }

    #[test]
    fn test_funnel_step_count() {
        let funnel = Funnel::new("f")
            .step(FunnelStep::new("A", "a"))
            .step(FunnelStep::new("B", "b"));
        assert_eq!(funnel.step_count(), 2);
    }

    #[test]
    fn test_cohort() {
        let mut c = Cohort::new("week-1");
        c.add_member("user-1");
        c.add_member("user-2");
        c.add_member("user-1"); // duplicate

        assert_eq!(c.size(), 2);
        assert!(c.contains("user-1"));
        assert!(!c.contains("user-3"));
    }

    #[test]
    fn test_weekly_cohort() {
        let mut cohorts = HashMap::new();
        weekly_cohort(0, "u1", &mut cohorts);
        weekly_cohort(3, "u2", &mut cohorts);
        weekly_cohort(7, "u3", &mut cohorts);
        weekly_cohort(8, "u4", &mut cohorts);

        assert_eq!(cohorts.len(), 2); // week 0 and week 1
        assert_eq!(cohorts[&0].size(), 2);
        assert_eq!(cohorts[&1].size(), 2);
    }

    #[test]
    fn test_collector_with_consent() {
        let consent = ConsentManager::new(ConsentLevel::Essential);
        let mut collector = EventCollector::new(consent);

        // Basic event should be tracked
        let e1 = Event::new("click", 1000);
        assert!(collector.track(e1));

        // PII event should be rejected
        let e2 = Event::new("signup", 2000).set_str("email", "a@b.com");
        assert!(!collector.track(e2));

        assert_eq!(collector.pending_count(), 1);
    }

    #[test]
    fn test_collector_no_consent() {
        let consent = ConsentManager::new(ConsentLevel::None);
        let mut collector = EventCollector::new(consent);

        assert!(!collector.track(Event::new("click", 1000)));
        assert!(!collector.track_page_view(PageView::new("/", 1000)));
        assert_eq!(collector.pending_count(), 0);
    }

    #[test]
    fn test_collector_drain_batch() {
        let consent = ConsentManager::new(ConsentLevel::Full);
        let mut collector = EventCollector::new(consent).with_batch_size(2);

        for i in 0..5 {
            collector.track(Event::new(format!("e{i}"), i as u64));
        }

        let batch1 = collector.drain_batch();
        assert_eq!(batch1.len(), 2);
        assert_eq!(collector.pending_count(), 3);

        let batch2 = collector.drain_batch();
        assert_eq!(batch2.len(), 2);
        assert_eq!(collector.pending_count(), 1);
    }

    #[test]
    fn test_collector_clear() {
        let consent = ConsentManager::new(ConsentLevel::Full);
        let mut collector = EventCollector::new(consent);
        collector.track(Event::new("e", 0));
        collector.clear();
        assert_eq!(collector.pending_count(), 0);
    }

    #[test]
    fn test_collector_set_consent() {
        let consent = ConsentManager::new(ConsentLevel::None);
        let mut collector = EventCollector::new(consent);
        assert_eq!(collector.consent_level(), ConsentLevel::None);

        collector.set_consent(ConsentLevel::Full);
        assert_eq!(collector.consent_level(), ConsentLevel::Full);
    }

    #[test]
    fn test_event_value_display() {
        assert_eq!(EventValue::Str("hello".into()).to_string(), "hello");
        assert_eq!(EventValue::Num(42.0).to_string(), "42");
        assert_eq!(EventValue::Bool(true).to_string(), "true");
        assert_eq!(EventValue::Null.to_string(), "null");
    }

    #[test]
    fn test_consent_level_display() {
        assert_eq!(ConsentLevel::None.to_string(), "none");
        assert_eq!(ConsentLevel::Essential.to_string(), "essential");
        assert_eq!(ConsentLevel::Functional.to_string(), "functional");
        assert_eq!(ConsentLevel::Full.to_string(), "full");
    }

    #[test]
    fn test_page_view_minimal() {
        let pv = PageView::new("/", 0);
        let event = pv.to_event();
        assert!(event.properties.get("referrer").is_none());
    }

    #[test]
    fn test_funnel_zero_first_step() {
        let funnel = Funnel::new("f")
            .step(FunnelStep::new("A", "a"))
            .step(FunnelStep::new("B", "b"));

        let rates = funnel.conversion_rates();
        assert!((rates[0].rate - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_session_duration_initial() {
        let s = Session::new("s", 5000);
        assert_eq!(s.duration_ms(), 0);
    }

    #[test]
    fn test_collector_track_page_view() {
        let consent = ConsentManager::new(ConsentLevel::Essential);
        let mut collector = EventCollector::new(consent);
        assert!(collector.track_page_view(PageView::new("/test", 1000)));
        assert_eq!(collector.pending_count(), 1);
    }
}
