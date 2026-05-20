//! Live cursors / collaboration — cursor position broadcast, user color
//! assignment, viewport awareness, throttled updates, and cursor smoothing.
//!
//! Pure-Rust cursor tracking layer with no real timers or graphics. Callers
//! feed positions and time, and the module produces outbound update messages.

use std::collections::HashMap;
use std::fmt;

// ── Cursor position ────────────────────────────────────────────────

/// 2D cursor position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CursorPos {
    pub x: f64,
    pub y: f64,
}

impl CursorPos {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    /// Euclidean distance to another position.
    pub fn distance_to(&self, other: &CursorPos) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Linear interpolation toward another position.
    pub fn lerp(&self, target: &CursorPos, t: f64) -> CursorPos {
        let t = t.clamp(0.0, 1.0);
        CursorPos {
            x: self.x + (target.x - self.x) * t,
            y: self.y + (target.y - self.y) * t,
        }
    }
}

impl fmt::Display for CursorPos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.1}, {:.1})", self.x, self.y)
    }
}

// ── Viewport ───────────────────────────────────────────────────────

/// A rectangular viewport defining the visible area.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Viewport {
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self { x, y, width, height }
    }

    /// Whether a cursor position is within this viewport.
    pub fn contains(&self, pos: &CursorPos) -> bool {
        pos.x >= self.x && pos.x <= self.x + self.width
            && pos.y >= self.y && pos.y <= self.y + self.height
    }

    /// Whether two viewports overlap.
    pub fn overlaps(&self, other: &Viewport) -> bool {
        self.x < other.x + other.width
            && self.x + self.width > other.x
            && self.y < other.y + other.height
            && self.y + self.height > other.y
    }
}

// ── Color assignment ───────────────────────────────────────────────

/// A set of colors to cycle through for user assignment.
const PALETTE: &[&str] = &[
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#264653", "#D62828", "#023E8A",
    "#6A0572", "#1B998B", "#FF6B6B", "#4ECDC4",
];

/// Assigns a deterministic color to a user based on their ID.
pub fn assign_color(user_id: &str) -> &'static str {
    let hash = user_id.bytes().fold(0u64, |acc, b| {
        acc.wrapping_mul(31).wrapping_add(b as u64)
    });
    PALETTE[(hash as usize) % PALETTE.len()]
}

// ── Cursor state per user ──────────────────────────────────────────

/// State of a single remote user's cursor.
#[derive(Debug, Clone)]
pub struct RemoteCursor {
    pub user_id: String,
    pub display_name: String,
    pub color: &'static str,
    pub position: CursorPos,
    pub target_position: CursorPos,
    pub viewport: Option<Viewport>,
    pub last_update_ms: u64,
    pub visible: bool,
}

impl RemoteCursor {
    fn new(user_id: impl Into<String>, display_name: impl Into<String>) -> Self {
        let uid: String = user_id.into();
        let color = assign_color(&uid);
        Self {
            user_id: uid,
            display_name: display_name.into(),
            color,
            position: CursorPos::zero(),
            target_position: CursorPos::zero(),
            viewport: None,
            last_update_ms: 0,
            visible: true,
        }
    }

    /// Smooth-step the cursor toward its target position.
    pub fn smooth_step(&mut self, factor: f64) {
        self.position = self.position.lerp(&self.target_position, factor);
    }

    /// Whether the cursor has essentially reached its target.
    pub fn is_at_target(&self, epsilon: f64) -> bool {
        self.position.distance_to(&self.target_position) < epsilon
    }
}

// ── Throttle state ─────────────────────────────────────────────────

/// Throttle configuration for outbound cursor updates.
#[derive(Debug, Clone)]
pub struct ThrottleConfig {
    /// Minimum interval between outbound updates in ms.
    pub interval_ms: u64,
    /// Minimum distance (px) the cursor must move to trigger an update.
    pub min_distance: f64,
}

impl Default for ThrottleConfig {
    fn default() -> Self {
        Self {
            interval_ms: 50,
            min_distance: 2.0,
        }
    }
}

/// Throttle state for the local cursor.
#[derive(Debug, Clone)]
struct ThrottleState {
    last_sent_pos: CursorPos,
    last_sent_ms: u64,
}

// ── Outbound messages ──────────────────────────────────────────────

/// An outbound cursor update message to broadcast.
#[derive(Debug, Clone, PartialEq)]
pub struct CursorUpdate {
    pub user_id: String,
    pub position: CursorPos,
    pub viewport: Option<Viewport>,
    pub timestamp_ms: u64,
}

/// An event emitted by the cursor manager.
#[derive(Debug, Clone)]
pub enum CursorEvent {
    /// A cursor update should be sent to peers.
    Broadcast(CursorUpdate),
    /// A remote cursor was added.
    CursorJoined { user_id: String, color: &'static str },
    /// A remote cursor was removed.
    CursorLeft { user_id: String },
    /// A remote cursor timed out (no updates for a while).
    CursorTimedOut { user_id: String },
}

// ── Cursor manager ─────────────────────────────────────────────────

/// Configuration for the cursor manager.
#[derive(Debug, Clone)]
pub struct CursorManagerConfig {
    pub throttle: ThrottleConfig,
    /// Smoothing factor per frame (0.0 = no smoothing, 1.0 = instant snap).
    pub smoothing_factor: f64,
    /// Timeout in ms after which a remote cursor is considered stale.
    pub stale_timeout_ms: u64,
    /// Hide cursors beyond this distance from local viewport center (-1 = disabled).
    pub visibility_radius: f64,
}

impl Default for CursorManagerConfig {
    fn default() -> Self {
        Self {
            throttle: ThrottleConfig::default(),
            smoothing_factor: 0.3,
            stale_timeout_ms: 10_000,
            visibility_radius: -1.0,
        }
    }
}

/// Manages local and remote cursors.
#[derive(Debug)]
pub struct CursorManager {
    config: CursorManagerConfig,
    local_user_id: String,
    local_position: CursorPos,
    local_viewport: Option<Viewport>,
    throttle: ThrottleState,
    remotes: HashMap<String, RemoteCursor>,
    events: Vec<CursorEvent>,
}

impl CursorManager {
    pub fn new(local_user_id: impl Into<String>, config: CursorManagerConfig) -> Self {
        Self {
            config,
            local_user_id: local_user_id.into(),
            local_position: CursorPos::zero(),
            local_viewport: None,
            throttle: ThrottleState {
                last_sent_pos: CursorPos::zero(),
                last_sent_ms: 0,
            },
            remotes: HashMap::new(),
            events: Vec::new(),
        }
    }

    /// Drain pending events.
    pub fn take_events(&mut self) -> Vec<CursorEvent> {
        std::mem::take(&mut self.events)
    }

    /// Update the local cursor position. Returns true if a broadcast was queued.
    pub fn move_local(&mut self, pos: CursorPos, now_ms: u64) -> bool {
        self.local_position = pos;
        let first_send = self.throttle.last_sent_ms == 0 && self.throttle.last_sent_pos.x == 0.0 && self.throttle.last_sent_pos.y == 0.0 && (pos.x != 0.0 || pos.y != 0.0);
        let time_ok = now_ms.saturating_sub(self.throttle.last_sent_ms) >= self.config.throttle.interval_ms;
        let dist_ok = pos.distance_to(&self.throttle.last_sent_pos) >= self.config.throttle.min_distance;

        if first_send || (time_ok && dist_ok) {
            self.throttle.last_sent_pos = pos;
            self.throttle.last_sent_ms = now_ms;
            self.events.push(CursorEvent::Broadcast(CursorUpdate {
                user_id: self.local_user_id.clone(),
                position: pos,
                viewport: self.local_viewport,
                timestamp_ms: now_ms,
            }));
            true
        } else {
            false
        }
    }

    /// Update the local viewport.
    pub fn set_local_viewport(&mut self, vp: Viewport) {
        self.local_viewport = Some(vp);
    }

    /// Add a remote cursor.
    pub fn add_remote(&mut self, user_id: impl Into<String>, display_name: impl Into<String>) {
        let uid: String = user_id.into();
        let cursor = RemoteCursor::new(&uid, display_name);
        let color = cursor.color;
        self.events.push(CursorEvent::CursorJoined { user_id: uid.clone(), color });
        self.remotes.insert(uid, cursor);
    }

    /// Remove a remote cursor.
    pub fn remove_remote(&mut self, user_id: &str) {
        if self.remotes.remove(user_id).is_some() {
            self.events.push(CursorEvent::CursorLeft { user_id: user_id.to_string() });
        }
    }

    /// Apply an incoming cursor update from a remote user.
    pub fn apply_remote_update(&mut self, update: &CursorUpdate) {
        if let Some(cursor) = self.remotes.get_mut(&update.user_id) {
            cursor.target_position = update.position;
            cursor.last_update_ms = update.timestamp_ms;
            if let Some(vp) = update.viewport {
                cursor.viewport = Some(vp);
            }
        }
    }

    /// Advance cursor smoothing by one frame.
    pub fn tick_smoothing(&mut self) {
        let factor = self.config.smoothing_factor;
        for cursor in self.remotes.values_mut() {
            cursor.smooth_step(factor);
        }
    }

    /// Check for stale cursors and update visibility.
    pub fn tick_stale(&mut self, now_ms: u64) {
        let timeout = self.config.stale_timeout_ms;
        let radius = self.config.visibility_radius;
        let local_vp = self.local_viewport;

        let mut timed_out = Vec::new();
        for (uid, cursor) in &mut self.remotes {
            // Stale check
            if now_ms.saturating_sub(cursor.last_update_ms) > timeout {
                timed_out.push(uid.clone());
                continue;
            }
            // Visibility check
            if radius > 0.0 {
                if let Some(vp) = local_vp {
                    let center = CursorPos::new(vp.x + vp.width / 2.0, vp.y + vp.height / 2.0);
                    cursor.visible = cursor.position.distance_to(&center) <= radius;
                }
            }
        }

        for uid in timed_out {
            self.remotes.remove(&uid);
            self.events.push(CursorEvent::CursorTimedOut { user_id: uid });
        }
    }

    /// Get all visible remote cursors.
    pub fn visible_cursors(&self) -> Vec<&RemoteCursor> {
        self.remotes.values().filter(|c| c.visible).collect()
    }

    /// Get a specific remote cursor.
    pub fn get_remote(&self, user_id: &str) -> Option<&RemoteCursor> {
        self.remotes.get(user_id)
    }

    /// Number of remote cursors tracked.
    pub fn remote_count(&self) -> usize {
        self.remotes.len()
    }

    pub fn local_position(&self) -> CursorPos {
        self.local_position
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CursorPos ──────────────────────────────────────────────────

    #[test]
    fn cursor_pos_distance() {
        let a = CursorPos::new(0.0, 0.0);
        let b = CursorPos::new(3.0, 4.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn cursor_pos_lerp() {
        let a = CursorPos::new(0.0, 0.0);
        let b = CursorPos::new(10.0, 20.0);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.x - 5.0).abs() < 1e-10);
        assert!((mid.y - 10.0).abs() < 1e-10);
    }

    #[test]
    fn cursor_pos_lerp_clamped() {
        let a = CursorPos::new(0.0, 0.0);
        let b = CursorPos::new(10.0, 10.0);
        let over = a.lerp(&b, 2.0);
        assert!((over.x - 10.0).abs() < 1e-10); // clamped to 1.0
    }

    #[test]
    fn cursor_pos_display() {
        let p = CursorPos::new(3.14, 2.71);
        assert_eq!(p.to_string(), "(3.1, 2.7)");
    }

    #[test]
    fn cursor_pos_zero() {
        let z = CursorPos::zero();
        assert!((z.x).abs() < 1e-10);
        assert!((z.y).abs() < 1e-10);
    }

    // ── Viewport ───────────────────────────────────────────────────

    #[test]
    fn viewport_contains() {
        let vp = Viewport::new(0.0, 0.0, 100.0, 100.0);
        assert!(vp.contains(&CursorPos::new(50.0, 50.0)));
        assert!(vp.contains(&CursorPos::new(0.0, 0.0)));
        assert!(vp.contains(&CursorPos::new(100.0, 100.0)));
        assert!(!vp.contains(&CursorPos::new(-1.0, 50.0)));
        assert!(!vp.contains(&CursorPos::new(50.0, 101.0)));
    }

    #[test]
    fn viewport_overlaps() {
        let a = Viewport::new(0.0, 0.0, 100.0, 100.0);
        let b = Viewport::new(50.0, 50.0, 100.0, 100.0);
        let c = Viewport::new(200.0, 200.0, 50.0, 50.0);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
        assert!(!a.overlaps(&c));
    }

    // ── Color assignment ───────────────────────────────────────────

    #[test]
    fn color_deterministic() {
        let c1 = assign_color("alice");
        let c2 = assign_color("alice");
        assert_eq!(c1, c2);
    }

    #[test]
    fn color_different_users() {
        // Different users will likely get different colors (not guaranteed but
        // with 12 colors and different hash values it is very probable)
        let c1 = assign_color("alice");
        let c2 = assign_color("bob");
        // At minimum, verify they are valid palette entries
        assert!(PALETTE.contains(&c1));
        assert!(PALETTE.contains(&c2));
    }

    // ── Remote cursor ──────────────────────────────────────────────

    #[test]
    fn remote_cursor_smooth_step() {
        let mut c = RemoteCursor::new("u1", "User 1");
        c.target_position = CursorPos::new(100.0, 100.0);
        c.smooth_step(0.5);
        assert!((c.position.x - 50.0).abs() < 1e-10);
        assert!((c.position.y - 50.0).abs() < 1e-10);
    }

    #[test]
    fn remote_cursor_is_at_target() {
        let mut c = RemoteCursor::new("u1", "User 1");
        c.position = CursorPos::new(99.99, 99.99);
        c.target_position = CursorPos::new(100.0, 100.0);
        assert!(c.is_at_target(0.1));
        assert!(!c.is_at_target(0.001));
    }

    // ── Cursor manager: throttle ───────────────────────────────────

    #[test]
    fn throttle_time_gate() {
        let config = CursorManagerConfig {
            throttle: ThrottleConfig { interval_ms: 100, min_distance: 0.0 },
            ..Default::default()
        };
        let mut mgr = CursorManager::new("me", config);

        assert!(mgr.move_local(CursorPos::new(10.0, 10.0), 0));
        // Too soon
        assert!(!mgr.move_local(CursorPos::new(20.0, 20.0), 50));
        // Enough time passed
        assert!(mgr.move_local(CursorPos::new(30.0, 30.0), 101));
    }

    #[test]
    fn throttle_distance_gate() {
        let config = CursorManagerConfig {
            throttle: ThrottleConfig { interval_ms: 0, min_distance: 10.0 },
            ..Default::default()
        };
        let mut mgr = CursorManager::new("me", config);

        assert!(mgr.move_local(CursorPos::new(10.0, 0.0), 0)); // first always sends (dist from 0,0 >= 10)
        // Not enough distance
        assert!(!mgr.move_local(CursorPos::new(11.0, 0.0), 10));
        // Enough distance
        assert!(mgr.move_local(CursorPos::new(25.0, 0.0), 20));
    }

    #[test]
    fn move_local_generates_broadcast() {
        let config = CursorManagerConfig {
            throttle: ThrottleConfig { interval_ms: 0, min_distance: 0.0 },
            ..Default::default()
        };
        let mut mgr = CursorManager::new("me", config);
        mgr.move_local(CursorPos::new(5.0, 5.0), 0);
        let events = mgr.take_events();
        assert_eq!(events.len(), 1);
        match &events[0] {
            CursorEvent::Broadcast(u) => {
                assert_eq!(u.user_id, "me");
                assert!((u.position.x - 5.0).abs() < 1e-10);
            }
            _ => panic!("expected Broadcast"),
        }
    }

    // ── Cursor manager: remotes ────────────────────────────────────

    #[test]
    fn add_and_remove_remote() {
        let mut mgr = CursorManager::new("me", CursorManagerConfig::default());
        mgr.add_remote("alice", "Alice");
        assert_eq!(mgr.remote_count(), 1);
        assert!(mgr.get_remote("alice").is_some());

        mgr.remove_remote("alice");
        assert_eq!(mgr.remote_count(), 0);

        let events = mgr.take_events();
        assert!(events.iter().any(|e| matches!(e, CursorEvent::CursorJoined { user_id, .. } if user_id == "alice")));
        assert!(events.iter().any(|e| matches!(e, CursorEvent::CursorLeft { user_id } if user_id == "alice")));
    }

    #[test]
    fn remove_nonexistent_remote() {
        let mut mgr = CursorManager::new("me", CursorManagerConfig::default());
        mgr.remove_remote("ghost");
        assert!(mgr.take_events().is_empty());
    }

    #[test]
    fn apply_remote_update() {
        let mut mgr = CursorManager::new("me", CursorManagerConfig::default());
        mgr.add_remote("alice", "Alice");

        let update = CursorUpdate {
            user_id: "alice".to_string(),
            position: CursorPos::new(50.0, 60.0),
            viewport: Some(Viewport::new(0.0, 0.0, 200.0, 200.0)),
            timestamp_ms: 100,
        };
        mgr.apply_remote_update(&update);

        let cursor = mgr.get_remote("alice").unwrap();
        assert!((cursor.target_position.x - 50.0).abs() < 1e-10);
        assert_eq!(cursor.last_update_ms, 100);
        assert!(cursor.viewport.is_some());
    }

    // ── Cursor manager: smoothing ──────────────────────────────────

    #[test]
    fn tick_smoothing_moves_toward_target() {
        let config = CursorManagerConfig {
            smoothing_factor: 0.5,
            ..Default::default()
        };
        let mut mgr = CursorManager::new("me", config);
        mgr.add_remote("alice", "Alice");

        let update = CursorUpdate {
            user_id: "alice".to_string(),
            position: CursorPos::new(100.0, 100.0),
            viewport: None,
            timestamp_ms: 0,
        };
        mgr.apply_remote_update(&update);
        mgr.tick_smoothing();

        let c = mgr.get_remote("alice").unwrap();
        assert!((c.position.x - 50.0).abs() < 1e-10);
        assert!((c.position.y - 50.0).abs() < 1e-10);
    }

    // ── Cursor manager: stale detection ────────────────────────────

    #[test]
    fn stale_cursor_removed() {
        let config = CursorManagerConfig {
            stale_timeout_ms: 100,
            ..Default::default()
        };
        let mut mgr = CursorManager::new("me", config);
        mgr.add_remote("alice", "Alice");
        mgr.take_events(); // clear join event

        mgr.tick_stale(200);
        assert_eq!(mgr.remote_count(), 0);
        let events = mgr.take_events();
        assert!(events.iter().any(|e| matches!(e, CursorEvent::CursorTimedOut { user_id } if user_id == "alice")));
    }

    #[test]
    fn active_cursor_not_stale() {
        let config = CursorManagerConfig {
            stale_timeout_ms: 100,
            ..Default::default()
        };
        let mut mgr = CursorManager::new("me", config);
        mgr.add_remote("alice", "Alice");
        let update = CursorUpdate {
            user_id: "alice".to_string(),
            position: CursorPos::new(1.0, 1.0),
            viewport: None,
            timestamp_ms: 50,
        };
        mgr.apply_remote_update(&update);

        mgr.tick_stale(100);
        assert_eq!(mgr.remote_count(), 1);
    }

    // ── Cursor manager: visibility ─────────────────────────────────

    #[test]
    fn visibility_radius_filtering() {
        let config = CursorManagerConfig {
            visibility_radius: 50.0,
            stale_timeout_ms: 99999,
            ..Default::default()
        };
        let mut mgr = CursorManager::new("me", config);
        mgr.set_local_viewport(Viewport::new(0.0, 0.0, 100.0, 100.0)); // center at (50,50)

        mgr.add_remote("near", "Near");
        mgr.add_remote("far", "Far");

        // Near is close to center
        let near_update = CursorUpdate {
            user_id: "near".to_string(),
            position: CursorPos::new(55.0, 55.0),
            viewport: None,
            timestamp_ms: 10,
        };
        mgr.apply_remote_update(&near_update);
        mgr.tick_smoothing(); // snap near cursor

        // Far is outside radius
        let far_update = CursorUpdate {
            user_id: "far".to_string(),
            position: CursorPos::new(500.0, 500.0),
            viewport: None,
            timestamp_ms: 10,
        };
        mgr.apply_remote_update(&far_update);
        // Need multiple smoothing ticks to move far cursor closer to target
        for _ in 0..20 {
            mgr.tick_smoothing();
        }

        mgr.tick_stale(20);
        let visible = mgr.visible_cursors();
        // Near should be visible, far should not
        assert!(visible.iter().any(|c| c.user_id == "near"));
        assert!(!visible.iter().any(|c| c.user_id == "far"));
    }

    #[test]
    fn local_position_getter() {
        let mut mgr = CursorManager::new("me", CursorManagerConfig {
            throttle: ThrottleConfig { interval_ms: 0, min_distance: 0.0 },
            ..Default::default()
        });
        mgr.move_local(CursorPos::new(42.0, 99.0), 0);
        let p = mgr.local_position();
        assert!((p.x - 42.0).abs() < 1e-10);
        assert!((p.y - 99.0).abs() < 1e-10);
    }

    #[test]
    fn set_local_viewport() {
        let mut mgr = CursorManager::new("me", CursorManagerConfig::default());
        mgr.set_local_viewport(Viewport::new(10.0, 20.0, 800.0, 600.0));
        assert!(mgr.local_viewport.is_some());
    }

    #[test]
    fn viewport_included_in_broadcast() {
        let config = CursorManagerConfig {
            throttle: ThrottleConfig { interval_ms: 0, min_distance: 0.0 },
            ..Default::default()
        };
        let mut mgr = CursorManager::new("me", config);
        let vp = Viewport::new(0.0, 0.0, 800.0, 600.0);
        mgr.set_local_viewport(vp);
        mgr.move_local(CursorPos::new(10.0, 10.0), 0);
        let events = mgr.take_events();
        match &events[0] {
            CursorEvent::Broadcast(u) => {
                assert!(u.viewport.is_some());
                let v = u.viewport.unwrap();
                assert!((v.width - 800.0).abs() < 1e-10);
            }
            _ => panic!("expected Broadcast"),
        }
    }
}
