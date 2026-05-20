//! Incident management: lifecycle states (detected/acknowledged/investigating/
//! mitigated/resolved), severity levels, timeline events, responder assignment,
//! postmortem templates, SLA tracking, and escalation rules.

use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;
use uuid::Uuid;

// ── Types ──

/// Incident lifecycle states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncidentState {
    Detected,
    Acknowledged,
    Investigating,
    Mitigated,
    Resolved,
    Closed,
}

impl IncidentState {
    pub fn as_str(&self) -> &'static str {
        match self {
            IncidentState::Detected => "detected",
            IncidentState::Acknowledged => "acknowledged",
            IncidentState::Investigating => "investigating",
            IncidentState::Mitigated => "mitigated",
            IncidentState::Resolved => "resolved",
            IncidentState::Closed => "closed",
        }
    }

    /// Valid transitions from this state.
    pub fn valid_transitions(&self) -> &'static [IncidentState] {
        match self {
            IncidentState::Detected => &[IncidentState::Acknowledged, IncidentState::Resolved],
            IncidentState::Acknowledged => &[IncidentState::Investigating, IncidentState::Resolved],
            IncidentState::Investigating => &[IncidentState::Mitigated, IncidentState::Resolved],
            IncidentState::Mitigated => &[IncidentState::Resolved],
            IncidentState::Resolved => &[IncidentState::Closed],
            IncidentState::Closed => &[],
        }
    }

    pub fn can_transition_to(&self, target: IncidentState) -> bool {
        self.valid_transitions().contains(&target)
    }
}

/// Severity levels (SEV-1 is highest).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Sev1,
    Sev2,
    Sev3,
    Sev4,
    Sev5,
}

impl Severity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Severity::Sev1 => "SEV-1",
            Severity::Sev2 => "SEV-2",
            Severity::Sev3 => "SEV-3",
            Severity::Sev4 => "SEV-4",
            Severity::Sev5 => "SEV-5",
        }
    }

    pub fn from_level(level: u8) -> Option<Self> {
        match level {
            1 => Some(Severity::Sev1),
            2 => Some(Severity::Sev2),
            3 => Some(Severity::Sev3),
            4 => Some(Severity::Sev4),
            5 => Some(Severity::Sev5),
            _ => None,
        }
    }

    /// Default response SLA in minutes for this severity.
    pub fn default_response_sla_minutes(&self) -> u64 {
        match self {
            Severity::Sev1 => 15,
            Severity::Sev2 => 30,
            Severity::Sev3 => 120,
            Severity::Sev4 => 480,
            Severity::Sev5 => 1440,
        }
    }

    /// Default resolution SLA in minutes.
    pub fn default_resolution_sla_minutes(&self) -> u64 {
        match self {
            Severity::Sev1 => 240,
            Severity::Sev2 => 480,
            Severity::Sev3 => 2880,
            Severity::Sev4 => 10080,
            Severity::Sev5 => 43200,
        }
    }
}

/// A responder assigned to an incident.
#[derive(Debug, Clone)]
pub struct Responder {
    pub id: String,
    pub name: String,
    pub role: ResponderRole,
    pub assigned_at: DateTime<Utc>,
    pub acknowledged_at: Option<DateTime<Utc>>,
}

impl Responder {
    pub fn new(name: &str, role: ResponderRole) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.to_string(),
            role,
            assigned_at: Utc::now(),
            acknowledged_at: None,
        }
    }

    pub fn acknowledge(&mut self) {
        self.acknowledged_at = Some(Utc::now());
    }
}

/// Role of a responder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponderRole {
    IncidentCommander,
    TechnicalLead,
    CommunicationsLead,
    Responder,
    Observer,
}

impl ResponderRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            ResponderRole::IncidentCommander => "incident_commander",
            ResponderRole::TechnicalLead => "technical_lead",
            ResponderRole::CommunicationsLead => "communications_lead",
            ResponderRole::Responder => "responder",
            ResponderRole::Observer => "observer",
        }
    }
}

/// A timeline event in an incident.
#[derive(Debug, Clone)]
pub struct TimelineEvent {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub kind: TimelineEventKind,
    pub description: String,
    pub author: Option<String>,
}

impl TimelineEvent {
    pub fn new(kind: TimelineEventKind, description: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            kind,
            description: description.to_string(),
            author: None,
        }
    }

    pub fn with_author(mut self, author: &str) -> Self {
        self.author = Some(author.to_string());
        self
    }
}

/// Kinds of timeline events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimelineEventKind {
    StateChange,
    Note,
    ResponderAssigned,
    EscalationTriggered,
    Communication,
    ActionTaken,
    RootCauseIdentified,
}

impl TimelineEventKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            TimelineEventKind::StateChange => "state_change",
            TimelineEventKind::Note => "note",
            TimelineEventKind::ResponderAssigned => "responder_assigned",
            TimelineEventKind::EscalationTriggered => "escalation_triggered",
            TimelineEventKind::Communication => "communication",
            TimelineEventKind::ActionTaken => "action_taken",
            TimelineEventKind::RootCauseIdentified => "root_cause_identified",
        }
    }
}

/// SLA tracking for an incident.
#[derive(Debug, Clone)]
pub struct SlaStatus {
    pub response_deadline: DateTime<Utc>,
    pub resolution_deadline: DateTime<Utc>,
    pub response_met: Option<bool>,
    pub resolution_met: Option<bool>,
    pub response_at: Option<DateTime<Utc>>,
    pub resolution_at: Option<DateTime<Utc>>,
}

impl SlaStatus {
    pub fn new(severity: Severity, created_at: DateTime<Utc>) -> Self {
        let response_deadline =
            created_at + Duration::minutes(severity.default_response_sla_minutes() as i64);
        let resolution_deadline =
            created_at + Duration::minutes(severity.default_resolution_sla_minutes() as i64);
        Self {
            response_deadline,
            resolution_deadline,
            response_met: None,
            resolution_met: None,
            response_at: None,
            resolution_at: None,
        }
    }

    pub fn record_response(&mut self, at: DateTime<Utc>) {
        self.response_at = Some(at);
        self.response_met = Some(at <= self.response_deadline);
    }

    pub fn record_resolution(&mut self, at: DateTime<Utc>) {
        self.resolution_at = Some(at);
        self.resolution_met = Some(at <= self.resolution_deadline);
    }

    pub fn response_remaining(&self) -> Duration {
        let now = Utc::now();
        if now >= self.response_deadline {
            Duration::zero()
        } else {
            self.response_deadline - now
        }
    }

    pub fn resolution_remaining(&self) -> Duration {
        let now = Utc::now();
        if now >= self.resolution_deadline {
            Duration::zero()
        } else {
            self.resolution_deadline - now
        }
    }

    pub fn is_response_breached(&self) -> bool {
        match self.response_met {
            Some(met) => !met,
            None => Utc::now() > self.response_deadline,
        }
    }

    pub fn is_resolution_breached(&self) -> bool {
        match self.resolution_met {
            Some(met) => !met,
            None => Utc::now() > self.resolution_deadline,
        }
    }
}

/// Escalation rule.
#[derive(Debug, Clone)]
pub struct EscalationRule {
    pub name: String,
    pub condition: EscalationCondition,
    pub target: String,
    pub message: String,
}

impl EscalationRule {
    pub fn new(name: &str, condition: EscalationCondition, target: &str, message: &str) -> Self {
        Self {
            name: name.to_string(),
            condition,
            target: target.to_string(),
            message: message.to_string(),
        }
    }

    /// Check whether this escalation should fire.
    pub fn should_escalate(
        &self,
        incident_age: Duration,
        state: IncidentState,
        severity: Severity,
    ) -> bool {
        match &self.condition {
            EscalationCondition::TimeInState {
                state: target_state,
                max_duration_minutes,
            } => state == *target_state && incident_age.num_minutes() > *max_duration_minutes as i64,
            EscalationCondition::SeverityAtLeast { min_severity } => severity <= *min_severity,
            EscalationCondition::NotAcknowledgedWithin { minutes } => {
                state == IncidentState::Detected && incident_age.num_minutes() > *minutes as i64
            }
        }
    }
}

/// Conditions that trigger escalation.
#[derive(Debug, Clone)]
pub enum EscalationCondition {
    TimeInState {
        state: IncidentState,
        max_duration_minutes: u64,
    },
    SeverityAtLeast {
        min_severity: Severity,
    },
    NotAcknowledgedWithin {
        minutes: u64,
    },
}

// ── Postmortem ──

/// A postmortem template/report.
#[derive(Debug, Clone)]
pub struct Postmortem {
    pub incident_id: String,
    pub title: String,
    pub summary: String,
    pub root_cause: String,
    pub impact: String,
    pub timeline_summary: Vec<String>,
    pub action_items: Vec<ActionItem>,
    pub lessons_learned: Vec<String>,
    pub created_at: DateTime<Utc>,
}

impl Postmortem {
    pub fn new(incident_id: &str, title: &str) -> Self {
        Self {
            incident_id: incident_id.to_string(),
            title: title.to_string(),
            summary: String::new(),
            root_cause: String::new(),
            impact: String::new(),
            timeline_summary: Vec::new(),
            action_items: Vec::new(),
            lessons_learned: Vec::new(),
            created_at: Utc::now(),
        }
    }

    pub fn with_summary(mut self, summary: &str) -> Self {
        self.summary = summary.to_string();
        self
    }

    pub fn with_root_cause(mut self, cause: &str) -> Self {
        self.root_cause = cause.to_string();
        self
    }

    pub fn with_impact(mut self, impact: &str) -> Self {
        self.impact = impact.to_string();
        self
    }

    pub fn add_timeline_entry(&mut self, entry: &str) {
        self.timeline_summary.push(entry.to_string());
    }

    pub fn add_action_item(&mut self, item: ActionItem) {
        self.action_items.push(item);
    }

    pub fn add_lesson(&mut self, lesson: &str) {
        self.lessons_learned.push(lesson.to_string());
    }

    pub fn is_complete(&self) -> bool {
        !self.summary.is_empty()
            && !self.root_cause.is_empty()
            && !self.impact.is_empty()
            && !self.action_items.is_empty()
    }
}

/// An action item from a postmortem.
#[derive(Debug, Clone)]
pub struct ActionItem {
    pub id: String,
    pub description: String,
    pub owner: String,
    pub priority: ActionPriority,
    pub due_date: Option<DateTime<Utc>>,
    pub completed: bool,
}

impl ActionItem {
    pub fn new(description: &str, owner: &str, priority: ActionPriority) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            description: description.to_string(),
            owner: owner.to_string(),
            priority,
            due_date: None,
            completed: false,
        }
    }

    pub fn with_due_date(mut self, date: DateTime<Utc>) -> Self {
        self.due_date = Some(date);
        self
    }

    pub fn complete(&mut self) {
        self.completed = true;
    }
}

/// Priority of an action item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionPriority {
    Urgent,
    High,
    Medium,
    Low,
}

impl ActionPriority {
    pub fn as_str(&self) -> &'static str {
        match self {
            ActionPriority::Urgent => "urgent",
            ActionPriority::High => "high",
            ActionPriority::Medium => "medium",
            ActionPriority::Low => "low",
        }
    }
}

// ── Incident ──

/// A single incident.
#[derive(Debug, Clone)]
pub struct Incident {
    pub id: String,
    pub title: String,
    pub description: String,
    pub severity: Severity,
    pub state: IncidentState,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
    pub responders: Vec<Responder>,
    pub timeline: Vec<TimelineEvent>,
    pub sla: SlaStatus,
    pub tags: HashMap<String, String>,
    pub affected_services: Vec<String>,
}

impl Incident {
    pub fn new(title: &str, description: &str, severity: Severity) -> Self {
        let now = Utc::now();
        let sla = SlaStatus::new(severity, now);
        let mut incident = Self {
            id: Uuid::new_v4().to_string(),
            title: title.to_string(),
            description: description.to_string(),
            severity,
            state: IncidentState::Detected,
            created_at: now,
            updated_at: now,
            resolved_at: None,
            responders: Vec::new(),
            timeline: Vec::new(),
            sla,
            tags: HashMap::new(),
            affected_services: Vec::new(),
        };
        incident.timeline.push(TimelineEvent::new(
            TimelineEventKind::StateChange,
            "Incident detected",
        ));
        incident
    }

    /// Transition to a new state, returning Ok if valid.
    pub fn transition_to(&mut self, new_state: IncidentState) -> Result<(), String> {
        if !self.state.can_transition_to(new_state) {
            return Err(format!(
                "Cannot transition from {} to {}",
                self.state.as_str(),
                new_state.as_str()
            ));
        }
        let old = self.state;
        self.state = new_state;
        self.updated_at = Utc::now();

        if new_state == IncidentState::Acknowledged {
            self.sla.record_response(Utc::now());
        }
        if new_state == IncidentState::Resolved {
            self.resolved_at = Some(Utc::now());
            self.sla.record_resolution(Utc::now());
        }

        self.timeline.push(TimelineEvent::new(
            TimelineEventKind::StateChange,
            &format!("State changed from {} to {}", old.as_str(), new_state.as_str()),
        ));
        Ok(())
    }

    pub fn assign_responder(&mut self, responder: Responder) {
        let desc = format!("{} assigned as {}", responder.name, responder.role.as_str());
        self.timeline.push(TimelineEvent::new(
            TimelineEventKind::ResponderAssigned,
            &desc,
        ));
        self.responders.push(responder);
        self.updated_at = Utc::now();
    }

    pub fn add_note(&mut self, text: &str, author: Option<&str>) {
        let mut event = TimelineEvent::new(TimelineEventKind::Note, text);
        if let Some(a) = author {
            event = event.with_author(a);
        }
        self.timeline.push(event);
        self.updated_at = Utc::now();
    }

    pub fn add_affected_service(&mut self, service: &str) {
        if !self.affected_services.iter().any(|s| s == service) {
            self.affected_services.push(service.to_string());
        }
    }

    pub fn set_tag(&mut self, key: &str, value: &str) {
        self.tags.insert(key.to_string(), value.to_string());
    }

    pub fn duration(&self) -> Duration {
        let end = self.resolved_at.unwrap_or_else(Utc::now);
        end - self.created_at
    }

    pub fn is_active(&self) -> bool {
        !matches!(self.state, IncidentState::Resolved | IncidentState::Closed)
    }

    pub fn incident_commander(&self) -> Option<&Responder> {
        self.responders
            .iter()
            .find(|r| r.role == ResponderRole::IncidentCommander)
    }

    /// Generate a postmortem template pre-filled from incident data.
    pub fn generate_postmortem(&self) -> Postmortem {
        let mut pm = Postmortem::new(&self.id, &self.title);
        for event in &self.timeline {
            pm.add_timeline_entry(&format!(
                "[{}] {}",
                event.timestamp.format("%Y-%m-%d %H:%M:%S"),
                event.description
            ));
        }
        pm
    }
}

// ── Incident Tracker ──

/// Tracks multiple incidents.
#[derive(Debug)]
pub struct IncidentTracker {
    incidents: Vec<Incident>,
    escalation_rules: Vec<EscalationRule>,
}

impl IncidentTracker {
    pub fn new() -> Self {
        Self {
            incidents: Vec::new(),
            escalation_rules: Vec::new(),
        }
    }

    pub fn create_incident(&mut self, title: &str, description: &str, severity: Severity) -> String {
        let incident = Incident::new(title, description, severity);
        let id = incident.id.clone();
        self.incidents.push(incident);
        id
    }

    pub fn get(&self, id: &str) -> Option<&Incident> {
        self.incidents.iter().find(|i| i.id == id)
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut Incident> {
        self.incidents.iter_mut().find(|i| i.id == id)
    }

    pub fn active_incidents(&self) -> Vec<&Incident> {
        self.incidents.iter().filter(|i| i.is_active()).collect()
    }

    pub fn by_severity(&self, severity: Severity) -> Vec<&Incident> {
        self.incidents
            .iter()
            .filter(|i| i.severity == severity)
            .collect()
    }

    pub fn incident_count(&self) -> usize {
        self.incidents.len()
    }

    pub fn add_escalation_rule(&mut self, rule: EscalationRule) {
        self.escalation_rules.push(rule);
    }

    /// Check all active incidents against escalation rules.
    pub fn check_escalations(&self) -> Vec<(String, String)> {
        let mut triggered = Vec::new();
        for incident in self.incidents.iter().filter(|i| i.is_active()) {
            let age = incident.duration();
            for rule in &self.escalation_rules {
                if rule.should_escalate(age, incident.state, incident.severity) {
                    triggered.push((incident.id.clone(), rule.name.clone()));
                }
            }
        }
        triggered
    }

    /// Mean time to acknowledge (MTTA) for resolved incidents, in minutes.
    pub fn mtta_minutes(&self) -> Option<f64> {
        let times: Vec<f64> = self
            .incidents
            .iter()
            .filter(|i| i.sla.response_at.is_some())
            .map(|i| {
                let resp = i.sla.response_at.unwrap();
                (resp - i.created_at).num_seconds() as f64 / 60.0
            })
            .collect();
        if times.is_empty() {
            return None;
        }
        Some(times.iter().sum::<f64>() / times.len() as f64)
    }

    /// Mean time to resolve (MTTR) for resolved incidents, in minutes.
    pub fn mttr_minutes(&self) -> Option<f64> {
        let times: Vec<f64> = self
            .incidents
            .iter()
            .filter_map(|i| {
                i.resolved_at
                    .map(|r| (r - i.created_at).num_seconds() as f64 / 60.0)
            })
            .collect();
        if times.is_empty() {
            return None;
        }
        Some(times.iter().sum::<f64>() / times.len() as f64)
    }
}

impl Default for IncidentTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incident_creation() {
        let inc = Incident::new("DB outage", "Primary DB is unreachable", Severity::Sev1);
        assert_eq!(inc.state, IncidentState::Detected);
        assert_eq!(inc.severity, Severity::Sev1);
        assert!(inc.is_active());
        assert_eq!(inc.timeline.len(), 1);
    }

    #[test]
    fn test_state_transitions_valid() {
        let mut inc = Incident::new("Test", "Test incident", Severity::Sev2);
        assert!(inc.transition_to(IncidentState::Acknowledged).is_ok());
        assert!(inc.transition_to(IncidentState::Investigating).is_ok());
        assert!(inc.transition_to(IncidentState::Mitigated).is_ok());
        assert!(inc.transition_to(IncidentState::Resolved).is_ok());
        assert!(inc.transition_to(IncidentState::Closed).is_ok());
        assert!(!inc.is_active());
    }

    #[test]
    fn test_state_transitions_invalid() {
        let mut inc = Incident::new("Test", "Test", Severity::Sev3);
        assert!(inc.transition_to(IncidentState::Mitigated).is_err());
    }

    #[test]
    fn test_assign_responder() {
        let mut inc = Incident::new("Test", "Test", Severity::Sev2);
        inc.assign_responder(Responder::new("Alice", ResponderRole::IncidentCommander));
        assert_eq!(inc.responders.len(), 1);
        assert!(inc.incident_commander().is_some());
        assert_eq!(inc.incident_commander().unwrap().name, "Alice");
    }

    #[test]
    fn test_responder_acknowledge() {
        let mut r = Responder::new("Bob", ResponderRole::Responder);
        assert!(r.acknowledged_at.is_none());
        r.acknowledge();
        assert!(r.acknowledged_at.is_some());
    }

    #[test]
    fn test_add_note() {
        let mut inc = Incident::new("Test", "Test", Severity::Sev3);
        inc.add_note("Investigating logs", Some("alice"));
        let notes: Vec<&TimelineEvent> = inc
            .timeline
            .iter()
            .filter(|e| e.kind == TimelineEventKind::Note)
            .collect();
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].author.as_deref(), Some("alice"));
    }

    #[test]
    fn test_affected_services() {
        let mut inc = Incident::new("Test", "Test", Severity::Sev2);
        inc.add_affected_service("api-gateway");
        inc.add_affected_service("api-gateway"); // Duplicate
        inc.add_affected_service("user-service");
        assert_eq!(inc.affected_services.len(), 2);
    }

    #[test]
    fn test_tags() {
        let mut inc = Incident::new("Test", "Test", Severity::Sev3);
        inc.set_tag("env", "production");
        assert_eq!(inc.tags.get("env").unwrap(), "production");
    }

    #[test]
    fn test_sla_creation() {
        let now = Utc::now();
        let sla = SlaStatus::new(Severity::Sev1, now);
        assert_eq!(
            sla.response_deadline,
            now + Duration::minutes(15)
        );
        assert_eq!(
            sla.resolution_deadline,
            now + Duration::minutes(240)
        );
    }

    #[test]
    fn test_sla_response_met() {
        let now = Utc::now();
        let mut sla = SlaStatus::new(Severity::Sev1, now);
        sla.record_response(now + Duration::minutes(10));
        assert_eq!(sla.response_met, Some(true));
    }

    #[test]
    fn test_sla_response_breached() {
        let created = Utc::now() - Duration::hours(1);
        let mut sla = SlaStatus::new(Severity::Sev1, created);
        sla.record_response(created + Duration::minutes(20));
        assert_eq!(sla.response_met, Some(false));
        assert!(sla.is_response_breached());
    }

    #[test]
    fn test_severity_levels() {
        assert_eq!(Severity::from_level(1), Some(Severity::Sev1));
        assert_eq!(Severity::from_level(5), Some(Severity::Sev5));
        assert_eq!(Severity::from_level(0), None);
        assert_eq!(Severity::from_level(6), None);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Sev1 < Severity::Sev2);
        assert!(Severity::Sev4 < Severity::Sev5);
    }

    #[test]
    fn test_postmortem_generation() {
        let mut inc = Incident::new("Outage", "Major outage", Severity::Sev1);
        inc.add_note("Root cause identified", None);
        let pm = inc.generate_postmortem();
        assert_eq!(pm.incident_id, inc.id);
        assert!(!pm.timeline_summary.is_empty());
        assert!(!pm.is_complete()); // No summary/root_cause/impact yet
    }

    #[test]
    fn test_postmortem_complete() {
        let mut pm = Postmortem::new("inc-1", "Outage Postmortem");
        assert!(!pm.is_complete());
        let pm = pm
            .with_summary("Summary")
            .with_root_cause("Bad deploy")
            .with_impact("50% of users affected");
        assert!(!pm.is_complete()); // No action items
        let mut pm = pm;
        pm.add_action_item(ActionItem::new(
            "Add canary deploys",
            "Alice",
            ActionPriority::High,
        ));
        assert!(pm.is_complete());
    }

    #[test]
    fn test_action_item() {
        let mut item = ActionItem::new("Fix monitoring", "Bob", ActionPriority::Urgent);
        assert!(!item.completed);
        item.complete();
        assert!(item.completed);
    }

    #[test]
    fn test_escalation_time_in_state() {
        let rule = EscalationRule::new(
            "slow_ack",
            EscalationCondition::NotAcknowledgedWithin { minutes: 10 },
            "oncall-manager",
            "Incident not acknowledged",
        );
        assert!(rule.should_escalate(
            Duration::minutes(15),
            IncidentState::Detected,
            Severity::Sev1,
        ));
        assert!(!rule.should_escalate(
            Duration::minutes(5),
            IncidentState::Detected,
            Severity::Sev1,
        ));
    }

    #[test]
    fn test_escalation_severity() {
        let rule = EscalationRule::new(
            "sev1_escalation",
            EscalationCondition::SeverityAtLeast {
                min_severity: Severity::Sev1,
            },
            "vp-engineering",
            "SEV-1 incident requires VP notification",
        );
        assert!(rule.should_escalate(
            Duration::minutes(0),
            IncidentState::Detected,
            Severity::Sev1,
        ));
        // Sev2 should NOT trigger since Sev2 is less severe than Sev1
        assert!(!rule.should_escalate(
            Duration::minutes(0),
            IncidentState::Detected,
            Severity::Sev2,
        ));
    }

    #[test]
    fn test_incident_tracker_basics() {
        let mut tracker = IncidentTracker::new();
        let id = tracker.create_incident("Test", "Test incident", Severity::Sev3);
        assert_eq!(tracker.incident_count(), 1);
        assert!(tracker.get(&id).is_some());
        assert_eq!(tracker.active_incidents().len(), 1);
    }

    #[test]
    fn test_incident_tracker_by_severity() {
        let mut tracker = IncidentTracker::new();
        tracker.create_incident("A", "A", Severity::Sev1);
        tracker.create_incident("B", "B", Severity::Sev2);
        tracker.create_incident("C", "C", Severity::Sev1);
        assert_eq!(tracker.by_severity(Severity::Sev1).len(), 2);
        assert_eq!(tracker.by_severity(Severity::Sev2).len(), 1);
    }

    #[test]
    fn test_mtta_and_mttr() {
        let mut tracker = IncidentTracker::new();
        let id = tracker.create_incident("Test", "Test", Severity::Sev2);
        {
            let inc = tracker.get_mut(&id).unwrap();
            inc.transition_to(IncidentState::Acknowledged).unwrap();
            inc.transition_to(IncidentState::Investigating).unwrap();
            inc.transition_to(IncidentState::Resolved).unwrap();
        }
        assert!(tracker.mtta_minutes().is_some());
        assert!(tracker.mttr_minutes().is_some());
    }

    #[test]
    fn test_responder_role_as_str() {
        assert_eq!(ResponderRole::IncidentCommander.as_str(), "incident_commander");
        assert_eq!(ResponderRole::TechnicalLead.as_str(), "technical_lead");
        assert_eq!(ResponderRole::Responder.as_str(), "responder");
    }

    #[test]
    fn test_timeline_event_kind_as_str() {
        assert_eq!(TimelineEventKind::StateChange.as_str(), "state_change");
        assert_eq!(TimelineEventKind::Note.as_str(), "note");
        assert_eq!(
            TimelineEventKind::RootCauseIdentified.as_str(),
            "root_cause_identified"
        );
    }

    #[test]
    fn test_incident_duration() {
        let inc = Incident::new("Test", "Test", Severity::Sev3);
        let dur = inc.duration();
        // Just created, duration should be very small
        assert!(dur.num_seconds() >= 0);
    }

    #[test]
    fn test_sla_remaining() {
        let now = Utc::now();
        let sla = SlaStatus::new(Severity::Sev3, now);
        // Response deadline is 120 minutes from now
        let remaining = sla.response_remaining();
        assert!(remaining.num_minutes() > 0);
    }
}
