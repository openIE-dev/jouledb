//! Chip/tag component: label text, removable (with onRemove), selectable,
//! disabled, icon slot, avatar slot, variants (filled/outlined), size (sm/md),
//! chip group with single/multi select, overflow handling.

// ── Variant ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChipVariant {
    Filled,
    Outlined,
}

impl ChipVariant {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Filled => "filled",
            Self::Outlined => "outlined",
        }
    }
}

// ── Size ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChipSize {
    Sm,
    Md,
}

impl ChipSize {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sm => "sm",
            Self::Md => "md",
        }
    }

    pub fn height_px(self) -> u32 {
        match self {
            Self::Sm => 24,
            Self::Md => 32,
        }
    }
}

// ── Selection mode ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionMode {
    Single,
    Multi,
}

// ── Chip ───────────────────────────────────────────────────────────

/// A single chip / tag element.
#[derive(Debug, Clone)]
pub struct Chip {
    pub id: String,
    pub label: String,
    pub variant: ChipVariant,
    pub size: ChipSize,
    pub selected: bool,
    pub disabled: bool,
    pub removable: bool,
    pub icon: Option<String>,
    pub avatar_url: Option<String>,
    pub color: Option<String>,
}

impl Chip {
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            variant: ChipVariant::Filled,
            size: ChipSize::Md,
            selected: false,
            disabled: false,
            removable: false,
            icon: None,
            avatar_url: None,
            color: None,
        }
    }

    pub fn variant(mut self, v: ChipVariant) -> Self {
        self.variant = v;
        self
    }

    pub fn size(mut self, s: ChipSize) -> Self {
        self.size = s;
        self
    }

    pub fn selected(mut self, s: bool) -> Self {
        self.selected = s;
        self
    }

    pub fn disabled(mut self, d: bool) -> Self {
        self.disabled = d;
        self
    }

    pub fn removable(mut self, r: bool) -> Self {
        self.removable = r;
        self
    }

    pub fn icon(mut self, i: impl Into<String>) -> Self {
        self.icon = Some(i.into());
        self
    }

    pub fn avatar(mut self, url: impl Into<String>) -> Self {
        self.avatar_url = Some(url.into());
        self
    }

    pub fn color(mut self, c: impl Into<String>) -> Self {
        self.color = Some(c.into());
        self
    }

    /// Toggle selection state if not disabled.
    pub fn toggle_select(&mut self) -> bool {
        if self.disabled {
            return false;
        }
        self.selected = !self.selected;
        true
    }

    /// Render chip to HTML.
    pub fn render(&self) -> String {
        let variant_class = self.variant.as_str();
        let size_class = self.size.as_str();
        let selected_class = if self.selected { " chip--selected" } else { "" };
        let disabled_class = if self.disabled { " chip--disabled" } else { "" };

        let color_style = self
            .color
            .as_deref()
            .map(|c| format!(";background:{}", c))
            .unwrap_or_default();

        let avatar_html = if let Some(url) = &self.avatar_url {
            format!(
                "<img class=\"chip-avatar\" src=\"{}\" alt=\"\" width=\"{}\" height=\"{}\" />",
                url,
                self.size.height_px() - 8,
                self.size.height_px() - 8,
            )
        } else {
            String::new()
        };

        let icon_html = if let Some(icon_name) = &self.icon {
            format!("<span class=\"chip-icon\">{}</span>", icon_name)
        } else {
            String::new()
        };

        let remove_html = if self.removable && !self.disabled {
            "<button class=\"chip-remove\" aria-label=\"Remove\">\u{00d7}</button>".to_string()
        } else {
            String::new()
        };

        format!(
            "<div class=\"chip chip--{} chip--{}{}{}\" \
             role=\"option\" aria-selected=\"{}\" aria-disabled=\"{}\" \
             data-chip-id=\"{}\" \
             style=\"height:{}px{}\">\
             {}{}<span class=\"chip-label\">{}</span>{}</div>",
            variant_class,
            size_class,
            selected_class,
            disabled_class,
            self.selected,
            self.disabled,
            self.id,
            self.size.height_px(),
            color_style,
            avatar_html,
            icon_html,
            self.label,
            remove_html,
        )
    }
}

// ── ChipGroup ──────────────────────────────────────────────────────

/// A group of chips with selection management and overflow handling.
#[derive(Debug)]
pub struct ChipGroup {
    pub chips: Vec<Chip>,
    pub selection_mode: SelectionMode,
    /// Maximum visible chips before showing overflow indicator.
    pub max_visible: Option<usize>,
    pub label: Option<String>,
}

impl ChipGroup {
    pub fn new(chips: Vec<Chip>) -> Self {
        Self {
            chips,
            selection_mode: SelectionMode::Multi,
            max_visible: None,
            label: None,
        }
    }

    pub fn selection_mode(mut self, mode: SelectionMode) -> Self {
        self.selection_mode = mode;
        self
    }

    pub fn max_visible(mut self, n: usize) -> Self {
        self.max_visible = Some(n);
        self
    }

    pub fn label(mut self, l: impl Into<String>) -> Self {
        self.label = Some(l.into());
        self
    }

    /// Select a chip by index. Respects single/multi mode.
    pub fn select(&mut self, index: usize) -> bool {
        if index >= self.chips.len() || self.chips[index].disabled {
            return false;
        }

        match self.selection_mode {
            SelectionMode::Single => {
                // Deselect all others
                for (i, chip) in self.chips.iter_mut().enumerate() {
                    chip.selected = i == index;
                }
                true
            }
            SelectionMode::Multi => {
                self.chips[index].toggle_select()
            }
        }
    }

    /// Deselect all chips.
    pub fn clear_selection(&mut self) {
        for chip in &mut self.chips {
            chip.selected = false;
        }
    }

    /// Remove a chip by index. Returns the removed chip if found.
    pub fn remove(&mut self, index: usize) -> Option<Chip> {
        if index < self.chips.len() && self.chips[index].removable {
            Some(self.chips.remove(index))
        } else {
            None
        }
    }

    /// Get indices of selected chips.
    pub fn selected_indices(&self) -> Vec<usize> {
        self.chips
            .iter()
            .enumerate()
            .filter(|(_, c)| c.selected)
            .map(|(i, _)| i)
            .collect()
    }

    /// Get selected chip labels.
    pub fn selected_labels(&self) -> Vec<&str> {
        self.chips
            .iter()
            .filter(|c| c.selected)
            .map(|c| c.label.as_str())
            .collect()
    }

    /// Number of chips hidden by overflow.
    pub fn overflow_count(&self) -> usize {
        match self.max_visible {
            Some(max) if self.chips.len() > max => self.chips.len() - max,
            _ => 0,
        }
    }

    /// Render the group to HTML.
    pub fn render(&self) -> String {
        let display_count = self
            .max_visible
            .unwrap_or(self.chips.len())
            .min(self.chips.len());

        let mut html = String::from("<div class=\"chip-group\" role=\"listbox\"");
        if let Some(lbl) = &self.label {
            html.push_str(&format!(" aria-label=\"{}\"", lbl));
        }
        html.push('>');

        for chip in self.chips.iter().take(display_count) {
            html.push_str(&chip.render());
        }

        let overflow = self.overflow_count();
        if overflow > 0 {
            html.push_str(&format!(
                "<span class=\"chip-overflow\">+{} more</span>",
                overflow
            ));
        }

        html.push_str("</div>");
        html
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_chips() -> Vec<Chip> {
        vec![
            Chip::new("c1", "Rust"),
            Chip::new("c2", "Python"),
            Chip::new("c3", "Go").disabled(true),
            Chip::new("c4", "Java").removable(true),
        ]
    }

    #[test]
    fn test_chip_toggle_select() {
        let mut chip = Chip::new("c1", "Rust");
        assert!(!chip.selected);
        assert!(chip.toggle_select());
        assert!(chip.selected);
        assert!(chip.toggle_select());
        assert!(!chip.selected);
    }

    #[test]
    fn test_chip_disabled_no_toggle() {
        let mut chip = Chip::new("c1", "Rust").disabled(true);
        assert!(!chip.toggle_select());
        assert!(!chip.selected);
    }

    #[test]
    fn test_single_select_mode() {
        let mut group = ChipGroup::new(sample_chips()).selection_mode(SelectionMode::Single);
        group.select(0);
        assert_eq!(group.selected_indices(), vec![0]);
        group.select(1);
        assert_eq!(group.selected_indices(), vec![1]);
        // First should be deselected
        assert!(!group.chips[0].selected);
    }

    #[test]
    fn test_multi_select_mode() {
        let mut group = ChipGroup::new(sample_chips()).selection_mode(SelectionMode::Multi);
        group.select(0);
        group.select(1);
        assert_eq!(group.selected_indices(), vec![0, 1]);
    }

    #[test]
    fn test_disabled_chip_not_selectable() {
        let mut group = ChipGroup::new(sample_chips());
        assert!(!group.select(2)); // Go is disabled
        assert!(group.selected_indices().is_empty());
    }

    #[test]
    fn test_remove_chip() {
        let mut group = ChipGroup::new(sample_chips());
        let removed = group.remove(3); // Java is removable
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().label, "Java");
        assert_eq!(group.chips.len(), 3);
    }

    #[test]
    fn test_remove_non_removable() {
        let mut group = ChipGroup::new(sample_chips());
        assert!(group.remove(0).is_none()); // Rust is not removable
    }

    #[test]
    fn test_clear_selection() {
        let mut group = ChipGroup::new(sample_chips());
        group.select(0);
        group.select(1);
        group.clear_selection();
        assert!(group.selected_indices().is_empty());
    }

    #[test]
    fn test_selected_labels() {
        let mut group = ChipGroup::new(sample_chips());
        group.select(0);
        group.select(1);
        let labels = group.selected_labels();
        assert_eq!(labels, vec!["Rust", "Python"]);
    }

    #[test]
    fn test_overflow_count() {
        let group = ChipGroup::new(sample_chips()).max_visible(2);
        assert_eq!(group.overflow_count(), 2);
    }

    #[test]
    fn test_no_overflow() {
        let group = ChipGroup::new(sample_chips());
        assert_eq!(group.overflow_count(), 0);
    }

    #[test]
    fn test_render_chip_html() {
        let chip = Chip::new("c1", "Rust")
            .variant(ChipVariant::Outlined)
            .size(ChipSize::Sm);
        let html = chip.render();
        assert!(html.contains("chip--outlined"));
        assert!(html.contains("chip--sm"));
        assert!(html.contains("Rust"));
        assert!(html.contains("role=\"option\""));
    }

    #[test]
    fn test_render_removable_chip() {
        let chip = Chip::new("c1", "Rust").removable(true);
        let html = chip.render();
        assert!(html.contains("chip-remove"));
        assert!(html.contains("\u{00d7}"));
    }

    #[test]
    fn test_render_chip_with_icon() {
        let chip = Chip::new("c1", "Star").icon("star-icon");
        let html = chip.render();
        assert!(html.contains("chip-icon"));
        assert!(html.contains("star-icon"));
    }

    #[test]
    fn test_render_group_overflow() {
        let group = ChipGroup::new(sample_chips()).max_visible(2);
        let html = group.render();
        assert!(html.contains("+2 more"));
        assert!(html.contains("chip-group"));
    }

    #[test]
    fn test_chip_variant_as_str() {
        assert_eq!(ChipVariant::Filled.as_str(), "filled");
        assert_eq!(ChipVariant::Outlined.as_str(), "outlined");
    }
}
