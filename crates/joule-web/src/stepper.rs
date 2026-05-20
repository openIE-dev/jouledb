//! Step indicator component: horizontal/vertical orientation, step states
//! (pending/active/completed/error), step validation, clickable steps,
//! linear vs free navigation, step content, connector styling, optional steps.

// ── Orientation ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    Horizontal,
    Vertical,
}

// ── Step state ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepState {
    Pending,
    Active,
    Completed,
    Error,
}

impl StepState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Active => "active",
            Self::Completed => "completed",
            Self::Error => "error",
        }
    }

    pub fn is_actionable(self) -> bool {
        matches!(self, Self::Active | Self::Completed)
    }
}

// ── Navigation mode ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavigationMode {
    /// Must complete steps in order.
    Linear,
    /// Can jump to any step.
    Free,
}

// ── Step ────────────────────────────────────────────────────────────

/// A single step in the stepper.
#[derive(Debug, Clone)]
pub struct Step {
    pub label: String,
    pub description: Option<String>,
    pub state: StepState,
    pub optional: bool,
    pub clickable: bool,
    pub content: Option<String>,
    /// Optional icon or step number override.
    pub icon: Option<String>,
    /// Custom validation error message.
    pub error_message: Option<String>,
}

impl Step {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            description: None,
            state: StepState::Pending,
            optional: false,
            clickable: true,
            content: None,
            icon: None,
            error_message: None,
        }
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn optional(mut self, opt: bool) -> Self {
        self.optional = opt;
        self
    }

    pub fn clickable(mut self, c: bool) -> Self {
        self.clickable = c;
        self
    }

    pub fn content(mut self, c: impl Into<String>) -> Self {
        self.content = Some(c.into());
        self
    }

    pub fn icon(mut self, i: impl Into<String>) -> Self {
        self.icon = Some(i.into());
        self
    }
}

// ── Connector style ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectorStyle {
    Solid,
    Dashed,
    Dotted,
}

impl ConnectorStyle {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Solid => "solid",
            Self::Dashed => "dashed",
            Self::Dotted => "dotted",
        }
    }
}

// ── Stepper ────────────────────────────────────────────────────────

/// A multi-step indicator / wizard component.
#[derive(Debug)]
pub struct Stepper {
    pub steps: Vec<Step>,
    pub active_index: usize,
    pub orientation: Orientation,
    pub navigation: NavigationMode,
    pub connector_style: ConnectorStyle,
}

impl Stepper {
    pub fn new(steps: Vec<Step>) -> Self {
        let mut s = Self {
            steps,
            active_index: 0,
            orientation: Orientation::Horizontal,
            navigation: NavigationMode::Linear,
            connector_style: ConnectorStyle::Solid,
        };
        s.sync_states();
        s
    }

    pub fn orientation(mut self, o: Orientation) -> Self {
        self.orientation = o;
        self
    }

    pub fn navigation(mut self, n: NavigationMode) -> Self {
        self.navigation = n;
        self
    }

    pub fn connector_style(mut self, c: ConnectorStyle) -> Self {
        self.connector_style = c;
        self
    }

    /// Synchronize step states based on active_index.
    fn sync_states(&mut self) {
        for (i, step) in self.steps.iter_mut().enumerate() {
            if step.state == StepState::Error {
                continue; // preserve errors
            }
            if i < self.active_index {
                step.state = StepState::Completed;
            } else if i == self.active_index {
                step.state = StepState::Active;
            } else {
                step.state = StepState::Pending;
            }
        }
    }

    /// Move to the next step. Returns false if already at the end.
    pub fn next(&mut self) -> bool {
        if self.active_index + 1 < self.steps.len() {
            self.active_index += 1;
            self.sync_states();
            true
        } else {
            false
        }
    }

    /// Move to the previous step.
    pub fn prev(&mut self) -> bool {
        if self.active_index > 0 {
            self.active_index -= 1;
            self.sync_states();
            true
        } else {
            false
        }
    }

    /// Jump to a specific step. Respects navigation mode.
    pub fn go_to(&mut self, index: usize) -> bool {
        if index >= self.steps.len() {
            return false;
        }
        if !self.steps[index].clickable {
            return false;
        }

        match self.navigation {
            NavigationMode::Free => {
                self.active_index = index;
                self.sync_states();
                true
            }
            NavigationMode::Linear => {
                // Can only go to completed steps or the next step
                if index <= self.active_index || index == self.active_index + 1 {
                    self.active_index = index;
                    self.sync_states();
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Mark the current step as error.
    pub fn set_error(&mut self, message: impl Into<String>) {
        if let Some(step) = self.steps.get_mut(self.active_index) {
            step.state = StepState::Error;
            step.error_message = Some(message.into());
        }
    }

    /// Clear error on the current step and reset to active.
    pub fn clear_error(&mut self) {
        if let Some(step) = self.steps.get_mut(self.active_index) {
            if step.state == StepState::Error {
                step.state = StepState::Active;
                step.error_message = None;
            }
        }
    }

    /// Validate: returns indices of steps in error state.
    pub fn errors(&self) -> Vec<usize> {
        self.steps
            .iter()
            .enumerate()
            .filter(|(_, s)| s.state == StepState::Error)
            .map(|(i, _)| i)
            .collect()
    }

    /// Total number of steps.
    pub fn total(&self) -> usize {
        self.steps.len()
    }

    /// Progress as a fraction [0.0, 1.0].
    pub fn progress(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let completed = self
            .steps
            .iter()
            .filter(|s| s.state == StepState::Completed)
            .count();
        completed as f64 / self.steps.len() as f64
    }

    /// True if all non-optional steps are completed.
    pub fn is_complete(&self) -> bool {
        self.steps
            .iter()
            .filter(|s| !s.optional)
            .all(|s| s.state == StepState::Completed)
    }

    /// Render stepper to HTML.
    pub fn render(&self) -> String {
        let orient = match self.orientation {
            Orientation::Horizontal => "horizontal",
            Orientation::Vertical => "vertical",
        };

        let mut html = format!("<div class=\"stepper stepper--{}\" role=\"navigation\">", orient);

        for (i, step) in self.steps.iter().enumerate() {
            let state_class = step.state.as_str();
            let optional_label = if step.optional { " (optional)" } else { "" };
            let default_num = format!("{}", i + 1);
            let step_num = step.icon.as_deref().unwrap_or(&default_num);

            html.push_str(&format!(
                "<div class=\"stepper-step stepper-step--{}\" aria-current=\"{}\">\
                 <div class=\"stepper-indicator\">{}</div>\
                 <div class=\"stepper-label\">{}{}</div>",
                state_class,
                if step.state == StepState::Active { "step" } else { "false" },
                step_num,
                step.label,
                optional_label,
            ));

            if let Some(desc) = &step.description {
                html.push_str(&format!("<div class=\"stepper-description\">{}</div>", desc));
            }

            if let Some(err) = &step.error_message {
                html.push_str(&format!("<div class=\"stepper-error\">{}</div>", err));
            }

            html.push_str("</div>");

            // Connector between steps
            if i + 1 < self.steps.len() {
                html.push_str(&format!(
                    "<div class=\"stepper-connector stepper-connector--{}\"></div>",
                    self.connector_style.as_str()
                ));
            }
        }

        html.push_str("</div>");
        html
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn three_steps() -> Vec<Step> {
        vec![
            Step::new("Account"),
            Step::new("Details"),
            Step::new("Confirm"),
        ]
    }

    #[test]
    fn test_initial_state() {
        let stepper = Stepper::new(three_steps());
        assert_eq!(stepper.active_index, 0);
        assert_eq!(stepper.steps[0].state, StepState::Active);
        assert_eq!(stepper.steps[1].state, StepState::Pending);
        assert_eq!(stepper.steps[2].state, StepState::Pending);
    }

    #[test]
    fn test_next_step() {
        let mut stepper = Stepper::new(three_steps());
        assert!(stepper.next());
        assert_eq!(stepper.active_index, 1);
        assert_eq!(stepper.steps[0].state, StepState::Completed);
        assert_eq!(stepper.steps[1].state, StepState::Active);
    }

    #[test]
    fn test_prev_step() {
        let mut stepper = Stepper::new(three_steps());
        stepper.next();
        assert!(stepper.prev());
        assert_eq!(stepper.active_index, 0);
        assert_eq!(stepper.steps[0].state, StepState::Active);
    }

    #[test]
    fn test_cannot_go_past_end() {
        let mut stepper = Stepper::new(three_steps());
        stepper.next();
        stepper.next();
        assert!(!stepper.next());
        assert_eq!(stepper.active_index, 2);
    }

    #[test]
    fn test_cannot_go_before_start() {
        let mut stepper = Stepper::new(three_steps());
        assert!(!stepper.prev());
        assert_eq!(stepper.active_index, 0);
    }

    #[test]
    fn test_linear_navigation_blocked() {
        let mut stepper = Stepper::new(three_steps()).navigation(NavigationMode::Linear);
        // Can't jump from step 0 to step 2
        assert!(!stepper.go_to(2));
        assert_eq!(stepper.active_index, 0);
    }

    #[test]
    fn test_free_navigation() {
        let mut stepper = Stepper::new(three_steps()).navigation(NavigationMode::Free);
        assert!(stepper.go_to(2));
        assert_eq!(stepper.active_index, 2);
    }

    #[test]
    fn test_error_state() {
        let mut stepper = Stepper::new(three_steps());
        stepper.set_error("Invalid email");
        assert_eq!(stepper.steps[0].state, StepState::Error);
        assert_eq!(stepper.steps[0].error_message.as_deref(), Some("Invalid email"));
        assert_eq!(stepper.errors(), vec![0]);
    }

    #[test]
    fn test_clear_error() {
        let mut stepper = Stepper::new(three_steps());
        stepper.set_error("Bad");
        stepper.clear_error();
        assert_eq!(stepper.steps[0].state, StepState::Active);
        assert!(stepper.steps[0].error_message.is_none());
    }

    #[test]
    fn test_progress() {
        let mut stepper = Stepper::new(three_steps());
        assert!((stepper.progress() - 0.0).abs() < f64::EPSILON);
        stepper.next();
        assert!((stepper.progress() - 1.0 / 3.0).abs() < 0.01);
        stepper.next();
        assert!((stepper.progress() - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_is_complete() {
        let mut stepper = Stepper::new(vec![
            Step::new("A"),
            Step::new("B").optional(true),
            Step::new("C"),
        ]);
        stepper.next(); // A completed
        stepper.next(); // B completed
        // C is active, not completed — but A and C are required
        assert!(!stepper.is_complete());
    }

    #[test]
    fn test_non_clickable_step() {
        let mut stepper = Stepper::new(vec![
            Step::new("A"),
            Step::new("B").clickable(false),
        ])
        .navigation(NavigationMode::Free);
        assert!(!stepper.go_to(1));
    }

    #[test]
    fn test_render_contains_role() {
        let stepper = Stepper::new(three_steps());
        let html = stepper.render();
        assert!(html.contains("role=\"navigation\""));
        assert!(html.contains("stepper--horizontal"));
        assert!(html.contains("stepper-step--active"));
        assert!(html.contains("stepper-connector--solid"));
    }

    #[test]
    fn test_vertical_orientation() {
        let stepper = Stepper::new(three_steps()).orientation(Orientation::Vertical);
        let html = stepper.render();
        assert!(html.contains("stepper--vertical"));
    }
}
