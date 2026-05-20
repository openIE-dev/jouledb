//! ARIA attribute management: role definitions, aria-* property builders,
//! state management, live region types, and relationship attributes.
//!
//! Pure data — no browser dependency. Builds WAI-ARIA attribute maps that
//! renderers can apply to DOM elements.

use std::collections::HashMap;
use std::fmt;

// ── Roles ──────────────────────────────────────────────────────

/// WAI-ARIA roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AriaRole {
    // Landmark
    Banner,
    Complementary,
    ContentInfo,
    Form,
    Main,
    Navigation,
    Region,
    Search,
    // Widget
    Alert,
    AlertDialog,
    Button,
    Checkbox,
    Combobox,
    Dialog,
    Grid,
    GridCell,
    Link,
    Listbox,
    Log,
    Marquee,
    Menu,
    MenuBar,
    MenuItem,
    MenuItemCheckbox,
    MenuItemRadio,
    Option,
    Progressbar,
    Radio,
    RadioGroup,
    Scrollbar,
    Separator,
    Slider,
    Spinbutton,
    Status,
    Switch,
    Tab,
    TabList,
    TabPanel,
    Textbox,
    Timer,
    Toolbar,
    Tooltip,
    Tree,
    TreeGrid,
    TreeItem,
    // Document structure
    Application,
    Article,
    Cell,
    ColumnHeader,
    Definition,
    Directory,
    Document,
    Feed,
    Figure,
    Group,
    Heading,
    Img,
    List,
    ListItem,
    Math,
    None,
    Note,
    Presentation,
    Row,
    RowGroup,
    RowHeader,
    Table,
    Term,
}

impl fmt::Display for AriaRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            AriaRole::Banner => "banner",
            AriaRole::Complementary => "complementary",
            AriaRole::ContentInfo => "contentinfo",
            AriaRole::Form => "form",
            AriaRole::Main => "main",
            AriaRole::Navigation => "navigation",
            AriaRole::Region => "region",
            AriaRole::Search => "search",
            AriaRole::Alert => "alert",
            AriaRole::AlertDialog => "alertdialog",
            AriaRole::Button => "button",
            AriaRole::Checkbox => "checkbox",
            AriaRole::Combobox => "combobox",
            AriaRole::Dialog => "dialog",
            AriaRole::Grid => "grid",
            AriaRole::GridCell => "gridcell",
            AriaRole::Link => "link",
            AriaRole::Listbox => "listbox",
            AriaRole::Log => "log",
            AriaRole::Marquee => "marquee",
            AriaRole::Menu => "menu",
            AriaRole::MenuBar => "menubar",
            AriaRole::MenuItem => "menuitem",
            AriaRole::MenuItemCheckbox => "menuitemcheckbox",
            AriaRole::MenuItemRadio => "menuitemradio",
            AriaRole::Option => "option",
            AriaRole::Progressbar => "progressbar",
            AriaRole::Radio => "radio",
            AriaRole::RadioGroup => "radiogroup",
            AriaRole::Scrollbar => "scrollbar",
            AriaRole::Separator => "separator",
            AriaRole::Slider => "slider",
            AriaRole::Spinbutton => "spinbutton",
            AriaRole::Status => "status",
            AriaRole::Switch => "switch",
            AriaRole::Tab => "tab",
            AriaRole::TabList => "tablist",
            AriaRole::TabPanel => "tabpanel",
            AriaRole::Textbox => "textbox",
            AriaRole::Timer => "timer",
            AriaRole::Toolbar => "toolbar",
            AriaRole::Tooltip => "tooltip",
            AriaRole::Tree => "tree",
            AriaRole::TreeGrid => "treegrid",
            AriaRole::TreeItem => "treeitem",
            AriaRole::Application => "application",
            AriaRole::Article => "article",
            AriaRole::Cell => "cell",
            AriaRole::ColumnHeader => "columnheader",
            AriaRole::Definition => "definition",
            AriaRole::Directory => "directory",
            AriaRole::Document => "document",
            AriaRole::Feed => "feed",
            AriaRole::Figure => "figure",
            AriaRole::Group => "group",
            AriaRole::Heading => "heading",
            AriaRole::Img => "img",
            AriaRole::List => "list",
            AriaRole::ListItem => "listitem",
            AriaRole::Math => "math",
            AriaRole::None => "none",
            AriaRole::Note => "note",
            AriaRole::Presentation => "presentation",
            AriaRole::Row => "row",
            AriaRole::RowGroup => "rowgroup",
            AriaRole::RowHeader => "rowheader",
            AriaRole::Table => "table",
            AriaRole::Term => "term",
        };
        f.write_str(s)
    }
}

impl AriaRole {
    /// Whether this role is a landmark role.
    pub fn is_landmark(&self) -> bool {
        matches!(
            self,
            AriaRole::Banner
                | AriaRole::Complementary
                | AriaRole::ContentInfo
                | AriaRole::Form
                | AriaRole::Main
                | AriaRole::Navigation
                | AriaRole::Region
                | AriaRole::Search
        )
    }

    /// Whether this role is a widget (interactive) role.
    pub fn is_widget(&self) -> bool {
        matches!(
            self,
            AriaRole::Alert
                | AriaRole::AlertDialog
                | AriaRole::Button
                | AriaRole::Checkbox
                | AriaRole::Combobox
                | AriaRole::Dialog
                | AriaRole::Grid
                | AriaRole::GridCell
                | AriaRole::Link
                | AriaRole::Listbox
                | AriaRole::Log
                | AriaRole::Menu
                | AriaRole::MenuBar
                | AriaRole::MenuItem
                | AriaRole::MenuItemCheckbox
                | AriaRole::MenuItemRadio
                | AriaRole::Option
                | AriaRole::Progressbar
                | AriaRole::Radio
                | AriaRole::RadioGroup
                | AriaRole::Scrollbar
                | AriaRole::Slider
                | AriaRole::Spinbutton
                | AriaRole::Status
                | AriaRole::Switch
                | AriaRole::Tab
                | AriaRole::TabList
                | AriaRole::TabPanel
                | AriaRole::Textbox
                | AriaRole::Toolbar
                | AriaRole::Tooltip
                | AriaRole::Tree
                | AriaRole::TreeGrid
                | AriaRole::TreeItem
        )
    }
}

// ── Live Region Type ───────────────────────────────────────────

/// ARIA live region politeness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveRegionType {
    Off,
    Polite,
    Assertive,
}

impl fmt::Display for LiveRegionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LiveRegionType::Off => f.write_str("off"),
            LiveRegionType::Polite => f.write_str("polite"),
            LiveRegionType::Assertive => f.write_str("assertive"),
        }
    }
}

// ── Tristate (checked / pressed) ──────────────────────────────

/// A tristate value for `aria-checked` / `aria-pressed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tristate {
    True,
    False,
    Mixed,
}

impl fmt::Display for Tristate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tristate::True => f.write_str("true"),
            Tristate::False => f.write_str("false"),
            Tristate::Mixed => f.write_str("mixed"),
        }
    }
}

// ── AriaAttrs builder ──────────────────────────────────────────

/// Builder for a set of ARIA attributes on a single element.
#[derive(Debug, Clone, Default)]
pub struct AriaAttrs {
    attrs: HashMap<String, String>,
}

impl AriaAttrs {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the `role` attribute.
    pub fn role(mut self, role: AriaRole) -> Self {
        self.attrs.insert("role".into(), role.to_string());
        self
    }

    /// Set `aria-label`.
    pub fn label(mut self, label: &str) -> Self {
        self.attrs.insert("aria-label".into(), label.into());
        self
    }

    /// Set `aria-labelledby` (space-separated IDs).
    pub fn labelledby(mut self, ids: &[&str]) -> Self {
        self.attrs.insert("aria-labelledby".into(), ids.join(" "));
        self
    }

    /// Set `aria-describedby` (space-separated IDs).
    pub fn describedby(mut self, ids: &[&str]) -> Self {
        self.attrs.insert("aria-describedby".into(), ids.join(" "));
        self
    }

    /// Set `aria-controls` (space-separated IDs).
    pub fn controls(mut self, ids: &[&str]) -> Self {
        self.attrs.insert("aria-controls".into(), ids.join(" "));
        self
    }

    /// Set `aria-owns` (space-separated IDs).
    pub fn owns(mut self, ids: &[&str]) -> Self {
        self.attrs.insert("aria-owns".into(), ids.join(" "));
        self
    }

    /// Set `aria-expanded`.
    pub fn expanded(mut self, val: bool) -> Self {
        self.attrs.insert("aria-expanded".into(), val.to_string());
        self
    }

    /// Set `aria-selected`.
    pub fn selected(mut self, val: bool) -> Self {
        self.attrs.insert("aria-selected".into(), val.to_string());
        self
    }

    /// Set `aria-checked`.
    pub fn checked(mut self, val: Tristate) -> Self {
        self.attrs.insert("aria-checked".into(), val.to_string());
        self
    }

    /// Set `aria-disabled`.
    pub fn disabled(mut self, val: bool) -> Self {
        self.attrs.insert("aria-disabled".into(), val.to_string());
        self
    }

    /// Set `aria-hidden`.
    pub fn hidden(mut self, val: bool) -> Self {
        self.attrs.insert("aria-hidden".into(), val.to_string());
        self
    }

    /// Set `aria-required`.
    pub fn required(mut self, val: bool) -> Self {
        self.attrs.insert("aria-required".into(), val.to_string());
        self
    }

    /// Set `aria-invalid`.
    pub fn invalid(mut self, val: bool) -> Self {
        self.attrs.insert("aria-invalid".into(), val.to_string());
        self
    }

    /// Set `aria-live`.
    pub fn live(mut self, region: LiveRegionType) -> Self {
        self.attrs.insert("aria-live".into(), region.to_string());
        self
    }

    /// Set `aria-atomic`.
    pub fn atomic(mut self, val: bool) -> Self {
        self.attrs.insert("aria-atomic".into(), val.to_string());
        self
    }

    /// Set `aria-busy`.
    pub fn busy(mut self, val: bool) -> Self {
        self.attrs.insert("aria-busy".into(), val.to_string());
        self
    }

    /// Set `aria-current`.
    pub fn current(mut self, val: &str) -> Self {
        self.attrs.insert("aria-current".into(), val.into());
        self
    }

    /// Set `aria-haspopup`.
    pub fn haspopup(mut self, val: &str) -> Self {
        self.attrs.insert("aria-haspopup".into(), val.into());
        self
    }

    /// Set `aria-level`.
    pub fn level(mut self, val: u32) -> Self {
        self.attrs.insert("aria-level".into(), val.to_string());
        self
    }

    /// Set `aria-valuemin`.
    pub fn valuemin(mut self, val: f64) -> Self {
        self.attrs.insert("aria-valuemin".into(), val.to_string());
        self
    }

    /// Set `aria-valuemax`.
    pub fn valuemax(mut self, val: f64) -> Self {
        self.attrs.insert("aria-valuemax".into(), val.to_string());
        self
    }

    /// Set `aria-valuenow`.
    pub fn valuenow(mut self, val: f64) -> Self {
        self.attrs.insert("aria-valuenow".into(), val.to_string());
        self
    }

    /// Set `aria-valuetext`.
    pub fn valuetext(mut self, val: &str) -> Self {
        self.attrs.insert("aria-valuetext".into(), val.into());
        self
    }

    /// Set an arbitrary attribute.
    pub fn attr(mut self, key: &str, value: &str) -> Self {
        self.attrs.insert(key.into(), value.into());
        self
    }

    /// Get an attribute value.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.attrs.get(key).map(|s| s.as_str())
    }

    /// Iterate all attributes.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.attrs.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Number of attributes.
    pub fn len(&self) -> usize {
        self.attrs.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.attrs.is_empty()
    }

    /// Render to an HTML attribute string fragment.
    pub fn to_html(&self) -> String {
        let mut pairs: Vec<_> = self.attrs.iter().collect();
        pairs.sort_by_key(|(k, _)| (*k).clone());
        pairs
            .iter()
            .map(|(k, v)| format!("{}=\"{}\"", k, v))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_display() {
        assert_eq!(AriaRole::Button.to_string(), "button");
        assert_eq!(AriaRole::Navigation.to_string(), "navigation");
        assert_eq!(AriaRole::TreeItem.to_string(), "treeitem");
    }

    #[test]
    fn role_is_landmark() {
        assert!(AriaRole::Main.is_landmark());
        assert!(AriaRole::Navigation.is_landmark());
        assert!(!AriaRole::Button.is_landmark());
    }

    #[test]
    fn role_is_widget() {
        assert!(AriaRole::Button.is_widget());
        assert!(AriaRole::Slider.is_widget());
        assert!(!AriaRole::Article.is_widget());
    }

    #[test]
    fn builder_role_and_label() {
        let a = AriaAttrs::new().role(AriaRole::Button).label("Submit");
        assert_eq!(a.get("role"), Some("button"));
        assert_eq!(a.get("aria-label"), Some("Submit"));
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn builder_states() {
        let a = AriaAttrs::new()
            .expanded(true)
            .selected(false)
            .checked(Tristate::Mixed)
            .disabled(true);
        assert_eq!(a.get("aria-expanded"), Some("true"));
        assert_eq!(a.get("aria-selected"), Some("false"));
        assert_eq!(a.get("aria-checked"), Some("mixed"));
        assert_eq!(a.get("aria-disabled"), Some("true"));
    }

    #[test]
    fn builder_relationships() {
        let a = AriaAttrs::new()
            .labelledby(&["title1", "title2"])
            .describedby(&["desc1"])
            .controls(&["panel1"])
            .owns(&["child1", "child2"]);
        assert_eq!(a.get("aria-labelledby"), Some("title1 title2"));
        assert_eq!(a.get("aria-describedby"), Some("desc1"));
        assert_eq!(a.get("aria-controls"), Some("panel1"));
        assert_eq!(a.get("aria-owns"), Some("child1 child2"));
    }

    #[test]
    fn builder_live_region() {
        let a = AriaAttrs::new()
            .live(LiveRegionType::Assertive)
            .atomic(true)
            .busy(false);
        assert_eq!(a.get("aria-live"), Some("assertive"));
        assert_eq!(a.get("aria-atomic"), Some("true"));
        assert_eq!(a.get("aria-busy"), Some("false"));
    }

    #[test]
    fn builder_range_values() {
        let a = AriaAttrs::new()
            .role(AriaRole::Slider)
            .valuemin(0.0)
            .valuemax(100.0)
            .valuenow(42.0)
            .valuetext("42%");
        assert_eq!(a.get("aria-valuenow"), Some("42"));
        assert_eq!(a.get("aria-valuetext"), Some("42%"));
    }

    #[test]
    fn builder_heading_level() {
        let a = AriaAttrs::new().role(AriaRole::Heading).level(3);
        assert_eq!(a.get("aria-level"), Some("3"));
    }

    #[test]
    fn to_html_sorted() {
        let a = AriaAttrs::new().role(AriaRole::Button).label("Go");
        let html = a.to_html();
        assert!(html.contains("aria-label=\"Go\""));
        assert!(html.contains("role=\"button\""));
    }

    #[test]
    fn tristate_display() {
        assert_eq!(Tristate::True.to_string(), "true");
        assert_eq!(Tristate::False.to_string(), "false");
        assert_eq!(Tristate::Mixed.to_string(), "mixed");
    }

    #[test]
    fn live_region_type_display() {
        assert_eq!(LiveRegionType::Off.to_string(), "off");
        assert_eq!(LiveRegionType::Polite.to_string(), "polite");
        assert_eq!(LiveRegionType::Assertive.to_string(), "assertive");
    }

    #[test]
    fn empty_attrs() {
        let a = AriaAttrs::new();
        assert!(a.is_empty());
        assert_eq!(a.len(), 0);
        assert_eq!(a.to_html(), "");
    }

    #[test]
    fn custom_attr() {
        let a = AriaAttrs::new().attr("data-custom", "value");
        assert_eq!(a.get("data-custom"), Some("value"));
    }
}
