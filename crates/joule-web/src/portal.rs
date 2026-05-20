//! Render portals and slots — mount children into different DOM subtrees,
//! slot distribution with named/default slots, teleport, portal event bubbling,
//! slot fallback content, and multi-slot layout.
//!
//! Replaces React portals, Vue `<Teleport>`, and Web Component `<slot>`
//! distribution with a pure-Rust model. Portals render content outside the
//! normal parent tree; slots distribute projected content into named outlets.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use uuid::Uuid;

// ── Portal types ────────────────────────────────────────────────────────

/// A named render target that portals can mount into.
#[derive(Debug, Clone)]
pub struct PortalTarget {
    pub id: String,
    pub z_index: i32,
    pub mount_point: String,
}

/// A piece of content mounted into a portal target.
#[derive(Debug, Clone)]
pub struct PortalContent {
    pub id: Uuid,
    pub target_id: String,
    pub content: String,
    pub priority: i32,
    pub visible: bool,
    pub created_at: DateTime<Utc>,
    pub source_component: Option<String>,
}

/// An event that bubbled through a portal boundary.
#[derive(Debug, Clone)]
pub struct PortalEvent {
    pub event_type: String,
    pub portal_id: Uuid,
    pub target_id: String,
    pub source_component: Option<String>,
    pub propagated: bool,
}

// ── Slot types ──────────────────────────────────────────────────────────

/// A named slot definition.
#[derive(Debug, Clone)]
pub struct SlotDefinition {
    pub name: String,
    pub fallback: Option<String>,
    pub multi: bool,
}

impl SlotDefinition {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            fallback: None,
            multi: false,
        }
    }

    pub fn with_fallback(mut self, fallback: &str) -> Self {
        self.fallback = Some(fallback.to_string());
        self
    }

    pub fn with_multi(mut self) -> Self {
        self.multi = true;
        self
    }
}

/// Content projected into a slot.
#[derive(Debug, Clone)]
pub struct SlotContent {
    pub id: Uuid,
    pub slot_name: String,
    pub content: String,
    pub order: i32,
}

// ── Teleport ────────────────────────────────────────────────────────────

/// A teleport directive — moves content from one location to another.
#[derive(Debug, Clone)]
pub struct Teleport {
    pub id: Uuid,
    pub source_path: String,
    pub target_selector: String,
    pub content: String,
    pub disabled: bool,
}

impl Teleport {
    pub fn new(source: &str, target: &str, content: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            source_path: source.to_string(),
            target_selector: target.to_string(),
            content: content.to_string(),
            disabled: false,
        }
    }

    pub fn disable(&mut self) {
        self.disabled = true;
    }

    pub fn enable(&mut self) {
        self.disabled = false;
    }

    /// Effective content: content if enabled, empty if disabled.
    pub fn effective_content(&self) -> &str {
        if self.disabled { "" } else { &self.content }
    }
}

// ── PortalManager ───────────────────────────────────────────────────────

/// Central manager for portal targets, mounted content, slots, teleports,
/// and modal/tooltip convenience layers.
pub struct PortalManager {
    targets: HashMap<String, PortalTarget>,
    contents: Vec<PortalContent>,
    active_modals: Vec<Uuid>,
    slot_definitions: HashMap<String, SlotDefinition>,
    slot_contents: Vec<SlotContent>,
    teleports: Vec<Teleport>,
    event_log: Vec<PortalEvent>,
}

impl PortalManager {
    pub fn new() -> Self {
        Self {
            targets: HashMap::new(),
            contents: Vec::new(),
            active_modals: Vec::new(),
            slot_definitions: HashMap::new(),
            slot_contents: Vec::new(),
            teleports: Vec::new(),
            event_log: Vec::new(),
        }
    }

    // ── Targets ─────────────────────────────────────────────────────────

    pub fn register_target(&mut self, id: &str, z_index: i32) {
        self.targets.insert(
            id.to_string(),
            PortalTarget {
                id: id.to_string(),
                z_index,
                mount_point: format!("#{id}"),
            },
        );
    }

    pub fn unregister_target(&mut self, id: &str) -> bool {
        self.targets.remove(id).is_some()
    }

    pub fn has_target(&self, id: &str) -> bool {
        self.targets.contains_key(id)
    }

    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    // ── Mount / unmount ─────────────────────────────────────────────────

    /// Mount content into a target. Returns `None` if the target does not exist.
    pub fn mount(
        &mut self,
        target_id: &str,
        content: &str,
        priority: i32,
    ) -> Option<Uuid> {
        if !self.has_target(target_id) {
            return None;
        }
        let id = Uuid::new_v4();
        self.contents.push(PortalContent {
            id,
            target_id: target_id.to_string(),
            content: content.to_string(),
            priority,
            visible: true,
            created_at: Utc::now(),
            source_component: None,
        });
        Some(id)
    }

    /// Mount content with a source component tag for event routing.
    pub fn mount_from(
        &mut self,
        target_id: &str,
        content: &str,
        priority: i32,
        source: &str,
    ) -> Option<Uuid> {
        if !self.has_target(target_id) {
            return None;
        }
        let id = Uuid::new_v4();
        self.contents.push(PortalContent {
            id,
            target_id: target_id.to_string(),
            content: content.to_string(),
            priority,
            visible: true,
            created_at: Utc::now(),
            source_component: Some(source.to_string()),
        });
        Some(id)
    }

    pub fn unmount(&mut self, content_id: &Uuid) -> bool {
        let before = self.contents.len();
        self.contents.retain(|c| c.id != *content_id);
        self.contents.len() < before
    }

    pub fn update(&mut self, content_id: &Uuid, new_content: &str) -> bool {
        if let Some(c) = self.contents.iter_mut().find(|c| c.id == *content_id) {
            c.content = new_content.to_string();
            true
        } else {
            false
        }
    }

    pub fn set_visible(&mut self, content_id: &Uuid, visible: bool) {
        if let Some(c) = self.contents.iter_mut().find(|c| c.id == *content_id) {
            c.visible = visible;
        }
    }

    /// Return visible content for a target, sorted by priority (highest first).
    pub fn contents_for_target(&self, target_id: &str) -> Vec<&PortalContent> {
        let mut out: Vec<&PortalContent> = self
            .contents
            .iter()
            .filter(|c| c.target_id == target_id && c.visible)
            .collect();
        out.sort_by(|a, b| b.priority.cmp(&a.priority));
        out
    }

    /// Total mounted content count.
    pub fn content_count(&self) -> usize {
        self.contents.len()
    }

    // ── Portal event bubbling ───────────────────────────────────────────

    /// Simulate an event bubbling through a portal boundary.
    pub fn dispatch_portal_event(
        &mut self,
        event_type: &str,
        portal_id: &Uuid,
    ) -> Option<PortalEvent> {
        let content = self.contents.iter().find(|c| c.id == *portal_id)?;
        let target_id = content.target_id.clone();
        let source = content.source_component.clone();

        let event = PortalEvent {
            event_type: event_type.to_string(),
            portal_id: *portal_id,
            target_id,
            source_component: source,
            propagated: true,
        };

        self.event_log.push(event.clone());
        Some(event)
    }

    /// Get the portal event log.
    pub fn event_log(&self) -> &[PortalEvent] {
        &self.event_log
    }

    // ── Modal helpers ───────────────────────────────────────────────────

    /// Open a modal. Auto-registers `modal-root` if needed.
    pub fn open_modal(&mut self, content: &str) -> Uuid {
        if !self.has_target("modal-root") {
            self.register_target("modal-root", 1000);
        }
        let priority = self.active_modals.len() as i32;
        let id = self
            .mount("modal-root", content, priority)
            .expect("modal-root registered above");
        self.active_modals.push(id);
        id
    }

    /// Close the top-most modal (LIFO).
    pub fn close_modal(&mut self) -> Option<Uuid> {
        let id = self.active_modals.pop()?;
        self.unmount(&id);
        Some(id)
    }

    /// Close a specific modal by ID.
    pub fn close_modal_by_id(&mut self, id: &Uuid) -> bool {
        if self.active_modals.contains(id) {
            self.active_modals.retain(|mid| mid != id);
            self.unmount(id);
            true
        } else {
            false
        }
    }

    pub fn modal_count(&self) -> usize {
        self.active_modals.len()
    }

    pub fn is_modal_open(&self) -> bool {
        !self.active_modals.is_empty()
    }

    pub fn top_modal(&self) -> Option<&PortalContent> {
        let id = self.active_modals.last()?;
        self.contents.iter().find(|c| c.id == *id)
    }

    // ── Tooltip helpers ─────────────────────────────────────────────────

    pub fn show_tooltip(&mut self, _anchor_id: &str, content: &str) -> Uuid {
        if !self.has_target("tooltip-root") {
            self.register_target("tooltip-root", 2000);
        }
        self.mount("tooltip-root", content, 0)
            .expect("tooltip-root registered above")
    }

    pub fn hide_tooltip(&mut self, id: &Uuid) -> bool {
        self.unmount(id)
    }

    // ── Slots ───────────────────────────────────────────────────────────

    /// Define a named slot.
    pub fn define_slot(&mut self, def: SlotDefinition) {
        self.slot_definitions.insert(def.name.clone(), def);
    }

    /// Remove a slot definition and all its content.
    pub fn remove_slot(&mut self, name: &str) -> bool {
        self.slot_contents.retain(|sc| sc.slot_name != name);
        self.slot_definitions.remove(name).is_some()
    }

    /// Project content into a named slot.
    pub fn project_to_slot(&mut self, slot_name: &str, content: &str, order: i32) -> Option<Uuid> {
        let def = self.slot_definitions.get(slot_name)?;
        let multi = def.multi;

        // If not multi-slot, replace existing content
        if !multi {
            self.slot_contents.retain(|sc| sc.slot_name != slot_name);
        }

        let id = Uuid::new_v4();
        self.slot_contents.push(SlotContent {
            id,
            slot_name: slot_name.to_string(),
            content: content.to_string(),
            order,
        });
        Some(id)
    }

    /// Get resolved content for a slot (sorted by order). Falls back to
    /// the slot's fallback content if nothing is projected.
    pub fn resolve_slot(&self, slot_name: &str) -> Vec<String> {
        let mut items: Vec<&SlotContent> = self
            .slot_contents
            .iter()
            .filter(|sc| sc.slot_name == slot_name)
            .collect();

        if items.is_empty() {
            // Check fallback
            if let Some(def) = self.slot_definitions.get(slot_name) {
                if let Some(fb) = &def.fallback {
                    return vec![fb.clone()];
                }
            }
            return Vec::new();
        }

        items.sort_by_key(|sc| sc.order);
        items.iter().map(|sc| sc.content.clone()).collect()
    }

    /// Number of defined slots.
    pub fn slot_count(&self) -> usize {
        self.slot_definitions.len()
    }

    // ── Teleports ───────────────────────────────────────────────────────

    /// Register a teleport.
    pub fn add_teleport(&mut self, teleport: Teleport) -> Uuid {
        let id = teleport.id;
        self.teleports.push(teleport);
        id
    }

    /// Remove a teleport by ID.
    pub fn remove_teleport(&mut self, id: &Uuid) -> bool {
        let before = self.teleports.len();
        self.teleports.retain(|t| t.id != *id);
        self.teleports.len() < before
    }

    /// Get all active (non-disabled) teleport targets.
    pub fn active_teleports(&self) -> Vec<&Teleport> {
        self.teleports.iter().filter(|t| !t.disabled).collect()
    }

    /// Teleport count.
    pub fn teleport_count(&self) -> usize {
        self.teleports.len()
    }
}

impl Default for PortalManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Multi-slot layout ───────────────────────────────────────────────────

/// A multi-slot layout definition (e.g., header/content/footer).
#[derive(Debug, Clone)]
pub struct MultiSlotLayout {
    pub name: String,
    pub slots: Vec<String>,
}

impl MultiSlotLayout {
    pub fn new(name: &str, slots: Vec<&str>) -> Self {
        Self {
            name: name.to_string(),
            slots: slots.into_iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Resolve all slots for this layout from a PortalManager.
    pub fn resolve(&self, manager: &PortalManager) -> HashMap<String, Vec<String>> {
        let mut result = HashMap::new();
        for slot_name in &self.slots {
            result.insert(slot_name.clone(), manager.resolve_slot(slot_name));
        }
        result
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_unregister_target() {
        let mut pm = PortalManager::new();
        pm.register_target("overlay", 100);
        assert!(pm.has_target("overlay"));
        assert!(pm.unregister_target("overlay"));
        assert!(!pm.has_target("overlay"));
    }

    #[test]
    fn mount_content_to_target() {
        let mut pm = PortalManager::new();
        pm.register_target("root", 0);
        let id = pm.mount("root", "hello", 1);
        assert!(id.is_some());
        assert_eq!(pm.content_count(), 1);
    }

    #[test]
    fn unmount_removes() {
        let mut pm = PortalManager::new();
        pm.register_target("root", 0);
        let id = pm.mount("root", "hello", 1).unwrap();
        assert!(pm.unmount(&id));
        assert!(pm.contents_for_target("root").is_empty());
    }

    #[test]
    fn contents_for_target_sorted_by_priority() {
        let mut pm = PortalManager::new();
        pm.register_target("root", 0);
        pm.mount("root", "low", 1);
        pm.mount("root", "high", 10);
        pm.mount("root", "mid", 5);

        let items = pm.contents_for_target("root");
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].content, "high");
        assert_eq!(items[1].content, "mid");
        assert_eq!(items[2].content, "low");
    }

    #[test]
    fn modal_open_close_stack() {
        let mut pm = PortalManager::new();
        let _a = pm.open_modal("modal-a");
        let b = pm.open_modal("modal-b");
        assert_eq!(pm.modal_count(), 2);
        assert!(pm.is_modal_open());

        let closed = pm.close_modal().unwrap();
        assert_eq!(closed, b);
        assert_eq!(pm.modal_count(), 1);
    }

    #[test]
    fn close_modal_lifo_order() {
        let mut pm = PortalManager::new();
        let a = pm.open_modal("first");
        let b = pm.open_modal("second");
        let c = pm.open_modal("third");

        assert_eq!(pm.close_modal(), Some(c));
        assert_eq!(pm.close_modal(), Some(b));
        assert_eq!(pm.close_modal(), Some(a));
        assert_eq!(pm.close_modal(), None);
    }

    #[test]
    fn update_content() {
        let mut pm = PortalManager::new();
        pm.register_target("root", 0);
        let id = pm.mount("root", "old", 0).unwrap();
        assert!(pm.update(&id, "new"));

        let items = pm.contents_for_target("root");
        assert_eq!(items[0].content, "new");
    }

    #[test]
    fn invisible_content_filtered() {
        let mut pm = PortalManager::new();
        pm.register_target("root", 0);
        let id = pm.mount("root", "hidden", 0).unwrap();
        pm.set_visible(&id, false);

        assert!(pm.contents_for_target("root").is_empty());
    }

    #[test]
    fn tooltip_show_hide() {
        let mut pm = PortalManager::new();
        let id = pm.show_tooltip("btn-1", "Tooltip text");
        assert!(!pm.contents_for_target("tooltip-root").is_empty());
        assert!(pm.hide_tooltip(&id));
        assert!(pm.contents_for_target("tooltip-root").is_empty());
    }

    #[test]
    fn mount_to_nonexistent_target_returns_none() {
        let mut pm = PortalManager::new();
        assert!(pm.mount("nope", "content", 0).is_none());
    }

    #[test]
    fn multiple_modals_stack_correctly() {
        let mut pm = PortalManager::new();
        pm.open_modal("a");
        pm.open_modal("b");
        pm.open_modal("c");

        let top = pm.top_modal().unwrap();
        assert_eq!(top.content, "c");
        assert_eq!(pm.modal_count(), 3);
    }

    #[test]
    fn close_modal_by_id() {
        let mut pm = PortalManager::new();
        let a = pm.open_modal("a");
        let _b = pm.open_modal("b");

        assert!(pm.close_modal_by_id(&a));
        assert_eq!(pm.modal_count(), 1);
        assert!(!pm.close_modal_by_id(&a));
    }

    // ── Slot tests ──────────────────────────────────────────────────────

    #[test]
    fn define_and_project_slot() {
        let mut pm = PortalManager::new();
        pm.define_slot(SlotDefinition::new("header"));
        let id = pm.project_to_slot("header", "<h1>Title</h1>", 0);
        assert!(id.is_some());

        let resolved = pm.resolve_slot("header");
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0], "<h1>Title</h1>");
    }

    #[test]
    fn slot_fallback() {
        let mut pm = PortalManager::new();
        pm.define_slot(SlotDefinition::new("footer").with_fallback("<p>Default footer</p>"));

        let resolved = pm.resolve_slot("footer");
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0], "<p>Default footer</p>");
    }

    #[test]
    fn slot_fallback_overridden() {
        let mut pm = PortalManager::new();
        pm.define_slot(SlotDefinition::new("footer").with_fallback("<p>Default</p>"));
        pm.project_to_slot("footer", "<p>Custom</p>", 0);

        let resolved = pm.resolve_slot("footer");
        assert_eq!(resolved[0], "<p>Custom</p>");
    }

    #[test]
    fn multi_slot_accepts_multiple() {
        let mut pm = PortalManager::new();
        pm.define_slot(SlotDefinition::new("items").with_multi());
        pm.project_to_slot("items", "item-1", 1);
        pm.project_to_slot("items", "item-2", 2);
        pm.project_to_slot("items", "item-0", 0);

        let resolved = pm.resolve_slot("items");
        assert_eq!(resolved.len(), 3);
        assert_eq!(resolved[0], "item-0");
        assert_eq!(resolved[1], "item-1");
        assert_eq!(resolved[2], "item-2");
    }

    #[test]
    fn single_slot_replaces() {
        let mut pm = PortalManager::new();
        pm.define_slot(SlotDefinition::new("main"));
        pm.project_to_slot("main", "first", 0);
        pm.project_to_slot("main", "second", 0);

        let resolved = pm.resolve_slot("main");
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0], "second");
    }

    #[test]
    fn remove_slot() {
        let mut pm = PortalManager::new();
        pm.define_slot(SlotDefinition::new("temp"));
        pm.project_to_slot("temp", "data", 0);
        assert!(pm.remove_slot("temp"));
        assert_eq!(pm.slot_count(), 0);
        assert!(pm.resolve_slot("temp").is_empty());
    }

    // ── Teleport tests ──────────────────────────────────────────────────

    #[test]
    fn teleport_add_and_active() {
        let mut pm = PortalManager::new();
        let t = Teleport::new("/src", "#target", "<div>content</div>");
        pm.add_teleport(t);
        assert_eq!(pm.teleport_count(), 1);
        assert_eq!(pm.active_teleports().len(), 1);
    }

    #[test]
    fn teleport_disable() {
        let mut pm = PortalManager::new();
        let mut t = Teleport::new("/src", "#target", "content");
        t.disable();
        pm.add_teleport(t);
        assert_eq!(pm.active_teleports().len(), 0);
    }

    #[test]
    fn teleport_effective_content() {
        let mut t = Teleport::new("/s", "#t", "hello");
        assert_eq!(t.effective_content(), "hello");
        t.disable();
        assert_eq!(t.effective_content(), "");
        t.enable();
        assert_eq!(t.effective_content(), "hello");
    }

    #[test]
    fn remove_teleport() {
        let mut pm = PortalManager::new();
        let t = Teleport::new("/s", "#t", "c");
        let id = pm.add_teleport(t);
        assert!(pm.remove_teleport(&id));
        assert_eq!(pm.teleport_count(), 0);
    }

    // ── Portal event bubbling tests ─────────────────────────────────────

    #[test]
    fn portal_event_dispatched() {
        let mut pm = PortalManager::new();
        pm.register_target("root", 0);
        let pid = pm.mount_from("root", "content", 0, "MyComponent").unwrap();
        let event = pm.dispatch_portal_event("click", &pid).unwrap();
        assert_eq!(event.event_type, "click");
        assert_eq!(event.source_component, Some("MyComponent".to_string()));
        assert!(event.propagated);
        assert_eq!(pm.event_log().len(), 1);
    }

    #[test]
    fn portal_event_nonexistent_returns_none() {
        let mut pm = PortalManager::new();
        let fake_id = Uuid::new_v4();
        assert!(pm.dispatch_portal_event("click", &fake_id).is_none());
    }

    // ── Multi-slot layout tests ─────────────────────────────────────────

    #[test]
    fn multi_slot_layout_resolve() {
        let mut pm = PortalManager::new();
        pm.define_slot(SlotDefinition::new("header").with_fallback("<header>Default</header>"));
        pm.define_slot(SlotDefinition::new("content"));
        pm.define_slot(SlotDefinition::new("footer"));

        pm.project_to_slot("content", "<main>Body</main>", 0);

        let layout = MultiSlotLayout::new("page", vec!["header", "content", "footer"]);
        let resolved = layout.resolve(&pm);

        assert_eq!(resolved["header"], vec!["<header>Default</header>"]);
        assert_eq!(resolved["content"], vec!["<main>Body</main>"]);
        assert!(resolved["footer"].is_empty());
    }
}
