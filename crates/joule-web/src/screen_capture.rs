//! Screen capture — config, constraints, session state machine, permission model.

use std::collections::VecDeque;

// ── Display Surface ─────────────────────────────────────────────

/// Type of display surface to capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplaySurface {
    Monitor,
    Window,
    Tab,
}

// ── Cursor Mode ─────────────────────────────────────────────────

/// Whether the cursor should be captured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorMode {
    Always,
    Motion,
    Never,
}

// ── Config ──────────────────────────────────────────────────────

/// Capture configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct CaptureConfig {
    pub video: bool,
    pub audio: bool,
    pub cursor: CursorMode,
    pub display_surface: DisplaySurface,
    pub self_browser_surface: SelfBrowserSurface,
    pub system_audio: SystemAudio,
}

/// Whether to allow capturing the current tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelfBrowserSurface {
    Include,
    Exclude,
}

/// Whether to include system audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemAudio {
    Include,
    Exclude,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            video: true,
            audio: false,
            cursor: CursorMode::Always,
            display_surface: DisplaySurface::Monitor,
            self_browser_surface: SelfBrowserSurface::Exclude,
            system_audio: SystemAudio::Exclude,
        }
    }
}

// ── Constraints ─────────────────────────────────────────────────

/// Capture constraints for video.
#[derive(Debug, Clone, PartialEq)]
pub struct CaptureConstraints {
    pub max_width: Option<u32>,
    pub max_height: Option<u32>,
    pub max_framerate: Option<f64>,
    pub min_width: Option<u32>,
    pub min_height: Option<u32>,
    pub min_framerate: Option<f64>,
}

impl Default for CaptureConstraints {
    fn default() -> Self {
        Self {
            max_width: None,
            max_height: None,
            max_framerate: None,
            min_width: None,
            min_height: None,
            min_framerate: None,
        }
    }
}

impl CaptureConstraints {
    /// Check whether the given dimensions satisfy the constraints.
    pub fn satisfies(&self, width: u32, height: u32) -> bool {
        if let Some(max_w) = self.max_width {
            if width > max_w {
                return false;
            }
        }
        if let Some(max_h) = self.max_height {
            if height > max_h {
                return false;
            }
        }
        if let Some(min_w) = self.min_width {
            if width < min_w {
                return false;
            }
        }
        if let Some(min_h) = self.min_height {
            if height < min_h {
                return false;
            }
        }
        true
    }
}

// ── Frame Metadata ──────────────────────────────────────────────

/// Metadata about a captured frame.
#[derive(Debug, Clone, PartialEq)]
pub struct CaptureFrameMetadata {
    pub timestamp_us: i64,
    pub width: u32,
    pub height: u32,
    pub capture_timestamp_us: i64,
}

// ── Permission ──────────────────────────────────────────────────

/// Permission state for screen capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionState {
    Prompt,
    Granted,
    Denied,
}

/// Permission model for screen capture.
#[derive(Debug, Clone)]
pub struct PermissionModel {
    pub state: PermissionState,
}

impl PermissionModel {
    pub fn new() -> Self {
        Self {
            state: PermissionState::Prompt,
        }
    }

    /// Grant permission.
    pub fn grant(&mut self) {
        self.state = PermissionState::Granted;
    }

    /// Deny permission.
    pub fn deny(&mut self) {
        self.state = PermissionState::Denied;
    }

    /// Reset to prompt.
    pub fn reset(&mut self) {
        self.state = PermissionState::Prompt;
    }

    pub fn is_granted(&self) -> bool {
        self.state == PermissionState::Granted
    }
}

impl Default for PermissionModel {
    fn default() -> Self {
        Self::new()
    }
}

// ── Session State ───────────────────────────────────────────────

/// State of a capture session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    Idle,
    Requesting,
    Active,
    Stopped,
}

// ── Session Events ──────────────────────────────────────────────

/// Events from the capture session.
#[derive(Debug, Clone, PartialEq)]
pub enum CaptureEvent {
    PermissionGranted,
    PermissionDenied,
    Started(CaptureConfig),
    Frame(CaptureFrameMetadata),
    Stopped,
    Error(String),
}

// ── Capture Session ─────────────────────────────────────────────

/// Screen capture session state machine.
#[derive(Debug, Clone)]
pub struct CaptureSession {
    pub state: SessionState,
    pub config: CaptureConfig,
    pub constraints: CaptureConstraints,
    pub permission: PermissionModel,
    frame_count: u64,
    events: VecDeque<CaptureEvent>,
}

impl CaptureSession {
    /// Create a new session with the given config and constraints.
    pub fn new(config: CaptureConfig, constraints: CaptureConstraints) -> Self {
        Self {
            state: SessionState::Idle,
            config,
            constraints,
            permission: PermissionModel::new(),
            frame_count: 0,
            events: VecDeque::new(),
        }
    }

    /// Drain pending events.
    pub fn drain_events(&mut self) -> Vec<CaptureEvent> {
        self.events.drain(..).collect()
    }

    /// Request to start capture (transitions to Requesting).
    pub fn request_start(&mut self) {
        if self.state != SessionState::Idle {
            return;
        }
        self.state = SessionState::Requesting;
    }

    /// Grant permission and activate the session.
    pub fn grant_permission(&mut self) {
        if self.state != SessionState::Requesting {
            return;
        }
        self.permission.grant();
        self.state = SessionState::Active;
        self.frame_count = 0;
        self.events.push_back(CaptureEvent::PermissionGranted);
        self.events.push_back(CaptureEvent::Started(self.config.clone()));
    }

    /// Deny permission.
    pub fn deny_permission(&mut self) {
        if self.state != SessionState::Requesting {
            return;
        }
        self.permission.deny();
        self.state = SessionState::Idle;
        self.events.push_back(CaptureEvent::PermissionDenied);
    }

    /// Record a captured frame.
    pub fn on_frame(&mut self, metadata: CaptureFrameMetadata) {
        if self.state != SessionState::Active {
            return;
        }
        if !self.constraints.satisfies(metadata.width, metadata.height) {
            self.events.push_back(CaptureEvent::Error(format!(
                "frame {}x{} does not satisfy constraints",
                metadata.width, metadata.height
            )));
            return;
        }
        self.frame_count += 1;
        self.events.push_back(CaptureEvent::Frame(metadata));
    }

    /// Stop the capture session.
    pub fn stop(&mut self) {
        if self.state == SessionState::Active || self.state == SessionState::Requesting {
            self.state = SessionState::Stopped;
            self.events.push_back(CaptureEvent::Stopped);
        }
    }

    /// Number of frames captured.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Report an error.
    pub fn report_error(&mut self, msg: impl Into<String>) {
        self.events.push_back(CaptureEvent::Error(msg.into()));
        self.state = SessionState::Stopped;
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_session() -> CaptureSession {
        CaptureSession::new(CaptureConfig::default(), CaptureConstraints::default())
    }

    fn make_frame(ts: i64, w: u32, h: u32) -> CaptureFrameMetadata {
        CaptureFrameMetadata {
            timestamp_us: ts,
            width: w,
            height: h,
            capture_timestamp_us: ts + 100,
        }
    }

    #[test]
    fn full_lifecycle() {
        let mut s = default_session();
        assert_eq!(s.state, SessionState::Idle);

        s.request_start();
        assert_eq!(s.state, SessionState::Requesting);

        s.grant_permission();
        assert_eq!(s.state, SessionState::Active);
        assert!(s.permission.is_granted());

        s.on_frame(make_frame(0, 1920, 1080));
        assert_eq!(s.frame_count(), 1);

        s.stop();
        assert_eq!(s.state, SessionState::Stopped);
    }

    #[test]
    fn permission_denied() {
        let mut s = default_session();
        s.request_start();
        s.deny_permission();
        assert_eq!(s.state, SessionState::Idle);
        assert_eq!(s.permission.state, PermissionState::Denied);
        let evts = s.drain_events();
        assert!(evts.iter().any(|e| matches!(e, CaptureEvent::PermissionDenied)));
    }

    #[test]
    fn no_frames_before_active() {
        let mut s = default_session();
        s.on_frame(make_frame(0, 1920, 1080));
        assert_eq!(s.frame_count(), 0);
    }

    #[test]
    fn constraints_check() {
        let c = CaptureConstraints {
            max_width: Some(1920),
            max_height: Some(1080),
            ..Default::default()
        };
        assert!(c.satisfies(1920, 1080));
        assert!(!c.satisfies(3840, 2160));
        assert!(c.satisfies(640, 480));
    }

    #[test]
    fn constraints_min() {
        let c = CaptureConstraints {
            min_width: Some(640),
            min_height: Some(480),
            ..Default::default()
        };
        assert!(c.satisfies(1920, 1080));
        assert!(!c.satisfies(320, 240));
    }

    #[test]
    fn frame_violates_constraints() {
        let mut s = CaptureSession::new(
            CaptureConfig::default(),
            CaptureConstraints {
                max_width: Some(1920),
                max_height: Some(1080),
                ..Default::default()
            },
        );
        s.request_start();
        s.grant_permission();
        s.drain_events();

        s.on_frame(make_frame(0, 3840, 2160));
        assert_eq!(s.frame_count(), 0);
        let evts = s.drain_events();
        assert!(evts.iter().any(|e| matches!(e, CaptureEvent::Error(_))));
    }

    #[test]
    fn config_defaults() {
        let c = CaptureConfig::default();
        assert!(c.video);
        assert!(!c.audio);
        assert_eq!(c.cursor, CursorMode::Always);
        assert_eq!(c.display_surface, DisplaySurface::Monitor);
    }

    #[test]
    fn events_emitted() {
        let mut s = default_session();
        s.request_start();
        s.grant_permission();
        s.on_frame(make_frame(0, 1920, 1080));
        s.stop();

        let evts = s.drain_events();
        assert!(evts.iter().any(|e| matches!(e, CaptureEvent::PermissionGranted)));
        assert!(evts.iter().any(|e| matches!(e, CaptureEvent::Started(_))));
        assert!(evts.iter().any(|e| matches!(e, CaptureEvent::Frame(_))));
        assert!(evts.iter().any(|e| matches!(e, CaptureEvent::Stopped)));
    }

    #[test]
    fn error_stops_session() {
        let mut s = default_session();
        s.request_start();
        s.grant_permission();
        s.report_error("device lost");
        assert_eq!(s.state, SessionState::Stopped);
    }

    #[test]
    fn permission_model_reset() {
        let mut p = PermissionModel::new();
        assert_eq!(p.state, PermissionState::Prompt);
        p.grant();
        assert!(p.is_granted());
        p.reset();
        assert_eq!(p.state, PermissionState::Prompt);
    }

    #[test]
    fn display_surface_variants() {
        let mut s = CaptureSession::new(
            CaptureConfig {
                display_surface: DisplaySurface::Tab,
                ..Default::default()
            },
            CaptureConstraints::default(),
        );
        s.request_start();
        s.grant_permission();
        assert_eq!(s.config.display_surface, DisplaySurface::Tab);
    }

    #[test]
    fn stop_from_requesting() {
        let mut s = default_session();
        s.request_start();
        s.stop();
        assert_eq!(s.state, SessionState::Stopped);
    }
}
