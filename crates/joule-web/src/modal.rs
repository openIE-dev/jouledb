//! Modal dialog manager: open/close state, modal stack (nested modals),
//! backdrop click handling, escape key dismissal, focus lock inside modal,
//! animation states, modal result (confirm/cancel/custom), role=dialog attributes.

use std::collections::HashMap;

// ── Animation state ────────────────────────────────────────────────

/// Animation lifecycle for modal transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationState {
    Entering,
    Entered,
    Exiting,
    Exited,
}

// ── Modal result ───────────────────────────────────────────────────

/// The result returned when a modal closes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModalResult {
    Confirm,
    Cancel,
    Custom(String),
}

// ── Dismiss policy ─────────────────────────────────────────────────

/// Controls how a modal can be dismissed.
#[derive(Debug, Clone)]
pub struct DismissPolicy {
    /// Close on backdrop click.
    pub backdrop_click: bool,
    /// Close on Escape key.
    pub escape_key: bool,
    /// Close on programmatic request only.
    pub programmatic_only: bool,
}

impl Default for DismissPolicy {
    fn default() -> Self {
        Self {
            backdrop_click: true,
            escape_key: true,
            programmatic_only: false,
        }
    }
}

// ── Modal config ───────────────────────────────────────────────────

/// Configuration for a single modal instance.
#[derive(Debug, Clone)]
pub struct ModalConfig {
    pub id: String,
    pub title: String,
    pub content: String,
    pub dismiss_policy: DismissPolicy,
    pub focus_lock: bool,
    pub role: String,
    pub aria_label: Option<String>,
    pub aria_described_by: Option<String>,
    /// Custom CSS class for the modal container.
    pub class: Option<String>,
    /// Custom attributes as key-value pairs.
    pub attributes: HashMap<String, String>,
}

impl ModalConfig {
    pub fn new(id: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            content: String::new(),
            dismiss_policy: DismissPolicy::default(),
            focus_lock: true,
            role: "dialog".into(),
            aria_label: None,
            aria_described_by: None,
            class: None,
            attributes: HashMap::new(),
        }
    }

    pub fn content(mut self, content: impl Into<String>) -> Self {
        self.content = content.into();
        self
    }

    pub fn dismiss_policy(mut self, policy: DismissPolicy) -> Self {
        self.dismiss_policy = policy;
        self
    }

    pub fn focus_lock(mut self, enabled: bool) -> Self {
        self.focus_lock = enabled;
        self
    }

    pub fn role(mut self, role: impl Into<String>) -> Self {
        self.role = role.into();
        self
    }

    pub fn aria_label(mut self, label: impl Into<String>) -> Self {
        self.aria_label = Some(label.into());
        self
    }

    pub fn class(mut self, class: impl Into<String>) -> Self {
        self.class = Some(class.into());
        self
    }
}

// ── Modal instance ─────────────────────────────────────────────────

/// Runtime state for a single modal on the stack.
#[derive(Debug, Clone)]
pub struct ModalInstance {
    pub config: ModalConfig,
    pub animation: AnimationState,
    pub result: Option<ModalResult>,
    pub z_index: u32,
}

// ── Modal manager ──────────────────────────────────────────────────

/// Manages a stack of nested modal dialogs.
#[derive(Debug)]
pub struct ModalManager {
    stack: Vec<ModalInstance>,
    base_z_index: u32,
    /// Results from closed modals, keyed by modal ID.
    closed_results: HashMap<String, ModalResult>,
}

impl Default for ModalManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ModalManager {
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            base_z_index: 1000,
            closed_results: HashMap::new(),
        }
    }

    pub fn with_base_z_index(mut self, z: u32) -> Self {
        self.base_z_index = z;
        self
    }

    /// Open a new modal, pushing it onto the stack.
    pub fn open(&mut self, config: ModalConfig) -> &ModalInstance {
        let z_index = self.base_z_index + self.stack.len() as u32 * 10;
        let instance = ModalInstance {
            config,
            animation: AnimationState::Entering,
            result: None,
            z_index,
        };
        self.stack.push(instance);
        self.stack.last().unwrap()
    }

    /// Transition the top modal's animation to Entered.
    pub fn finish_enter(&mut self) -> bool {
        if let Some(top) = self.stack.last_mut() {
            if top.animation == AnimationState::Entering {
                top.animation = AnimationState::Entered;
                return true;
            }
        }
        false
    }

    /// Begin closing the top modal with a result.
    pub fn close_top(&mut self, result: ModalResult) -> bool {
        if let Some(top) = self.stack.last_mut() {
            top.animation = AnimationState::Exiting;
            top.result = Some(result);
            return true;
        }
        false
    }

    /// Finish the exit animation and remove the top modal from the stack.
    pub fn finish_exit(&mut self) -> Option<(String, ModalResult)> {
        if let Some(top) = self.stack.last() {
            if top.animation == AnimationState::Exiting {
                let instance = self.stack.pop().unwrap();
                let result = instance.result.unwrap_or(ModalResult::Cancel);
                let id = instance.config.id.clone();
                self.closed_results.insert(id.clone(), result.clone());
                return Some((id, result));
            }
        }
        None
    }

    /// Close a specific modal by ID.
    pub fn close_by_id(&mut self, id: &str, result: ModalResult) -> bool {
        if let Some(pos) = self.stack.iter().position(|m| m.config.id == id) {
            self.stack[pos].animation = AnimationState::Exiting;
            self.stack[pos].result = Some(result);
            true
        } else {
            false
        }
    }

    /// Remove all modals whose animation is Exiting (bulk cleanup).
    pub fn remove_exited(&mut self) -> Vec<(String, ModalResult)> {
        let mut removed = Vec::new();
        self.stack.retain(|m| {
            if m.animation == AnimationState::Exiting {
                let id = m.config.id.clone();
                let result = m.result.clone().unwrap_or(ModalResult::Cancel);
                removed.push((id, result));
                false
            } else {
                true
            }
        });
        for (id, result) in &removed {
            self.closed_results.insert(id.clone(), result.clone());
        }
        removed
    }

    /// Handle backdrop click on the top modal.
    pub fn handle_backdrop_click(&mut self) -> Option<ModalResult> {
        if let Some(top) = self.stack.last() {
            if top.config.dismiss_policy.backdrop_click
                && !top.config.dismiss_policy.programmatic_only
            {
                let result = ModalResult::Cancel;
                self.close_top(result.clone());
                return Some(result);
            }
        }
        None
    }

    /// Handle Escape key on the top modal.
    pub fn handle_escape_key(&mut self) -> Option<ModalResult> {
        if let Some(top) = self.stack.last() {
            if top.config.dismiss_policy.escape_key
                && !top.config.dismiss_policy.programmatic_only
            {
                let result = ModalResult::Cancel;
                self.close_top(result.clone());
                return Some(result);
            }
        }
        None
    }

    /// Number of modals currently on the stack.
    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// True if any modals are open.
    pub fn is_open(&self) -> bool {
        !self.stack.is_empty()
    }

    /// Reference to the topmost modal, if any.
    pub fn top(&self) -> Option<&ModalInstance> {
        self.stack.last()
    }

    /// All modals on the stack (bottom to top).
    pub fn stack(&self) -> &[ModalInstance] {
        &self.stack
    }

    /// Retrieve the result of a previously closed modal.
    pub fn closed_result(&self, id: &str) -> Option<&ModalResult> {
        self.closed_results.get(id)
    }

    /// Check whether focus lock is active (top modal has focus_lock enabled).
    pub fn is_focus_locked(&self) -> bool {
        self.stack.last().map_or(false, |m| m.config.focus_lock)
    }

    /// Render the top modal to an HTML string.
    pub fn render_top(&self) -> Option<String> {
        self.stack.last().map(|m| render_modal(m))
    }
}

/// Render a single modal instance to HTML.
pub fn render_modal(modal: &ModalInstance) -> String {
    let animation_class = match modal.animation {
        AnimationState::Entering => "modal--entering",
        AnimationState::Entered => "modal--entered",
        AnimationState::Exiting => "modal--exiting",
        AnimationState::Exited => "modal--exited",
    };

    let class = modal
        .config
        .class
        .as_deref()
        .unwrap_or("modal");

    let aria_label = modal
        .config
        .aria_label
        .as_deref()
        .map(|l| format!(" aria-label=\"{}\"", l))
        .unwrap_or_default();

    let aria_desc = modal
        .config
        .aria_described_by
        .as_deref()
        .map(|d| format!(" aria-describedby=\"{}\"", d))
        .unwrap_or_default();

    format!(
        "<div class=\"modal-backdrop\" style=\"z-index:{}\">\
         <div class=\"{} {}\" role=\"{}\" aria-modal=\"true\"{}{} data-modal-id=\"{}\">\
         <div class=\"modal-header\"><h2>{}</h2></div>\
         <div class=\"modal-body\">{}</div>\
         </div></div>",
        modal.z_index,
        class,
        animation_class,
        modal.config.role,
        aria_label,
        aria_desc,
        modal.config.id,
        modal.config.title,
        modal.config.content,
    )
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_and_depth() {
        let mut mgr = ModalManager::new();
        assert_eq!(mgr.depth(), 0);
        assert!(!mgr.is_open());
        mgr.open(ModalConfig::new("m1", "First"));
        assert_eq!(mgr.depth(), 1);
        assert!(mgr.is_open());
    }

    #[test]
    fn test_nested_modals_z_index() {
        let mut mgr = ModalManager::new().with_base_z_index(500);
        mgr.open(ModalConfig::new("a", "A"));
        mgr.open(ModalConfig::new("b", "B"));
        mgr.open(ModalConfig::new("c", "C"));
        assert_eq!(mgr.stack()[0].z_index, 500);
        assert_eq!(mgr.stack()[1].z_index, 510);
        assert_eq!(mgr.stack()[2].z_index, 520);
    }

    #[test]
    fn test_animation_lifecycle() {
        let mut mgr = ModalManager::new();
        mgr.open(ModalConfig::new("m", "Modal"));
        assert_eq!(mgr.top().unwrap().animation, AnimationState::Entering);
        assert!(mgr.finish_enter());
        assert_eq!(mgr.top().unwrap().animation, AnimationState::Entered);
        mgr.close_top(ModalResult::Confirm);
        assert_eq!(mgr.top().unwrap().animation, AnimationState::Exiting);
        let (id, result) = mgr.finish_exit().unwrap();
        assert_eq!(id, "m");
        assert_eq!(result, ModalResult::Confirm);
        assert_eq!(mgr.depth(), 0);
    }

    #[test]
    fn test_backdrop_click_dismissal() {
        let mut mgr = ModalManager::new();
        mgr.open(ModalConfig::new("m", "Modal"));
        mgr.finish_enter();
        let result = mgr.handle_backdrop_click();
        assert_eq!(result, Some(ModalResult::Cancel));
        assert_eq!(mgr.top().unwrap().animation, AnimationState::Exiting);
    }

    #[test]
    fn test_backdrop_click_blocked() {
        let mut mgr = ModalManager::new();
        let policy = DismissPolicy {
            backdrop_click: false,
            escape_key: true,
            programmatic_only: false,
        };
        mgr.open(ModalConfig::new("m", "Modal").dismiss_policy(policy));
        assert!(mgr.handle_backdrop_click().is_none());
        assert_eq!(mgr.top().unwrap().animation, AnimationState::Entering);
    }

    #[test]
    fn test_escape_key_dismissal() {
        let mut mgr = ModalManager::new();
        mgr.open(ModalConfig::new("m", "Modal"));
        mgr.finish_enter();
        let result = mgr.handle_escape_key();
        assert_eq!(result, Some(ModalResult::Cancel));
    }

    #[test]
    fn test_programmatic_only() {
        let mut mgr = ModalManager::new();
        let policy = DismissPolicy {
            backdrop_click: true,
            escape_key: true,
            programmatic_only: true,
        };
        mgr.open(ModalConfig::new("m", "Modal").dismiss_policy(policy));
        assert!(mgr.handle_backdrop_click().is_none());
        assert!(mgr.handle_escape_key().is_none());
    }

    #[test]
    fn test_close_by_id() {
        let mut mgr = ModalManager::new();
        mgr.open(ModalConfig::new("a", "A"));
        mgr.open(ModalConfig::new("b", "B"));
        assert!(mgr.close_by_id("a", ModalResult::Custom("done".into())));
        assert_eq!(mgr.stack()[0].animation, AnimationState::Exiting);
        assert_eq!(mgr.stack()[1].animation, AnimationState::Entering);
    }

    #[test]
    fn test_remove_exited() {
        let mut mgr = ModalManager::new();
        mgr.open(ModalConfig::new("a", "A"));
        mgr.open(ModalConfig::new("b", "B"));
        mgr.close_by_id("a", ModalResult::Cancel);
        let removed = mgr.remove_exited();
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0].0, "a");
        assert_eq!(mgr.depth(), 1);
    }

    #[test]
    fn test_focus_lock() {
        let mut mgr = ModalManager::new();
        assert!(!mgr.is_focus_locked());
        mgr.open(ModalConfig::new("m", "Modal").focus_lock(true));
        assert!(mgr.is_focus_locked());
        mgr.open(ModalConfig::new("n", "NoLock").focus_lock(false));
        assert!(!mgr.is_focus_locked());
    }

    #[test]
    fn test_closed_result_history() {
        let mut mgr = ModalManager::new();
        mgr.open(ModalConfig::new("m", "Modal"));
        mgr.close_top(ModalResult::Custom("save".into()));
        mgr.finish_exit();
        assert_eq!(
            mgr.closed_result("m"),
            Some(&ModalResult::Custom("save".into()))
        );
    }

    #[test]
    fn test_render_modal_html() {
        let mut mgr = ModalManager::new();
        mgr.open(
            ModalConfig::new("dlg", "Title")
                .content("Body text")
                .aria_label("my dialog"),
        );
        let html = mgr.render_top().unwrap();
        assert!(html.contains("role=\"dialog\""));
        assert!(html.contains("aria-modal=\"true\""));
        assert!(html.contains("aria-label=\"my dialog\""));
        assert!(html.contains("data-modal-id=\"dlg\""));
        assert!(html.contains("Body text"));
    }

    #[test]
    fn test_custom_modal_result() {
        let r = ModalResult::Custom("submit-form".into());
        assert_ne!(r, ModalResult::Confirm);
        assert_ne!(r, ModalResult::Cancel);
        assert_eq!(r, ModalResult::Custom("submit-form".into()));
    }
}
