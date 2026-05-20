//! Command Palette: fuzzy-filtered command launcher (replaces cmdk).
//!
//! Provides a searchable, categorized command palette with keyboard
//! navigation, recent-command tracking, and handler dispatch.

use std::collections::{HashMap, VecDeque};

// ── Command ─────────────────────────────────────────────────────

/// A single command that can be invoked from the palette.
#[derive(Debug, Clone)]
pub struct Command {
    pub id: String,
    pub label: String,
    pub description: Option<String>,
    pub shortcut: Option<String>,
    pub category: Option<String>,
    pub handler_id: u64,
    pub enabled: bool,
    pub icon: Option<String>,
}

impl Command {
    pub fn new(id: impl Into<String>, label: impl Into<String>, handler_id: u64) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            description: None,
            shortcut: None,
            category: None,
            handler_id,
            enabled: true,
            icon: None,
        }
    }

    pub fn description(mut self, d: impl Into<String>) -> Self { self.description = Some(d.into()); self }
    pub fn shortcut(mut self, s: impl Into<String>) -> Self { self.shortcut = Some(s.into()); self }
    pub fn category(mut self, c: impl Into<String>) -> Self { self.category = Some(c.into()); self }
    pub fn icon(mut self, i: impl Into<String>) -> Self { self.icon = Some(i.into()); self }
    pub fn enabled(mut self, e: bool) -> Self { self.enabled = e; self }
}

// ── CommandPalette ──────────────────────────────────────────────

/// The palette state: commands, filter, selection, open/close.
#[derive(Debug)]
pub struct CommandPalette {
    commands: Vec<Command>,
    query: String,
    filtered: Vec<usize>,
    selected_index: usize,
    open: bool,
    max_results: usize,
    recent: VecDeque<String>,
}

impl CommandPalette {
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
            query: String::new(),
            filtered: Vec::new(),
            selected_index: 0,
            open: false,
            max_results: 50,
            recent: VecDeque::with_capacity(16),
        }
    }

    pub fn register(&mut self, cmd: Command) {
        // Replace if same id already exists.
        if let Some(pos) = self.commands.iter().position(|c| c.id == cmd.id) {
            self.commands[pos] = cmd;
        } else {
            self.commands.push(cmd);
        }
        self.refilter();
    }

    pub fn unregister(&mut self, id: &str) -> bool {
        let before = self.commands.len();
        self.commands.retain(|c| c.id != id);
        let removed = self.commands.len() < before;
        if removed { self.refilter(); }
        removed
    }

    pub fn open(&mut self) { self.open = true; self.selected_index = 0; }
    pub fn close(&mut self) { self.open = false; self.query.clear(); self.refilter(); }
    pub fn toggle(&mut self) { if self.open { self.close(); } else { self.open(); } }
    pub fn is_open(&self) -> bool { self.open }

    pub fn set_query(&mut self, query: impl Into<String>) {
        self.query = query.into();
        self.refilter();
        self.selected_index = 0;
    }

    pub fn query(&self) -> &str { &self.query }

    pub fn select_next(&mut self) {
        if self.filtered.is_empty() { return; }
        // Find next enabled
        let len = self.filtered.len();
        for offset in 1..=len {
            let idx = (self.selected_index + offset) % len;
            if self.commands[self.filtered[idx]].enabled {
                self.selected_index = idx;
                return;
            }
        }
    }

    pub fn select_previous(&mut self) {
        if self.filtered.is_empty() { return; }
        let len = self.filtered.len();
        for offset in 1..=len {
            let idx = (self.selected_index + len - offset) % len;
            if self.commands[self.filtered[idx]].enabled {
                self.selected_index = idx;
                return;
            }
        }
    }

    pub fn selected(&self) -> Option<&Command> {
        self.filtered.get(self.selected_index).map(|i| &self.commands[*i])
    }

    /// Execute the selected command. Returns handler_id, adds to recent, closes.
    pub fn execute(&mut self) -> Option<u64> {
        let cmd_idx = *self.filtered.get(self.selected_index)?;
        let cmd = &self.commands[cmd_idx];
        if !cmd.enabled { return None; }
        let handler_id = cmd.handler_id;
        let id = cmd.id.clone();

        // Update recent: remove if present, push front
        self.recent.retain(|r| r != &id);
        if self.recent.len() >= 16 { self.recent.pop_back(); }
        self.recent.push_front(id);

        self.close();
        Some(handler_id)
    }

    pub fn recent_commands(&self) -> Vec<&Command> {
        self.recent.iter()
            .filter_map(|id| self.commands.iter().find(|c| c.id == *id))
            .collect()
    }

    pub fn commands_by_category(&self) -> HashMap<&str, Vec<&Command>> {
        let mut map: HashMap<&str, Vec<&Command>> = HashMap::new();
        for cmd in &self.commands {
            let cat = cmd.category.as_deref().unwrap_or("Uncategorized");
            map.entry(cat).or_default().push(cmd);
        }
        map
    }

    pub fn filtered_commands(&self) -> Vec<&Command> {
        self.filtered.iter().map(|i| &self.commands[*i]).collect()
    }

    // ── Internal ────────────────────────────────────────────────

    fn refilter(&mut self) {
        let q = self.query.to_lowercase();
        self.filtered = self.commands.iter().enumerate()
            .filter(|(_, cmd)| {
                if q.is_empty() { return true; }
                self.fuzzy_match(&cmd.label, &q)
                    || cmd.description.as_deref().is_some_and(|d| self.fuzzy_match(d, &q))
                    || cmd.category.as_deref().is_some_and(|c| self.fuzzy_match(c, &q))
            })
            .take(self.max_results)
            .map(|(i, _)| i)
            .collect();
    }

    fn fuzzy_match(&self, haystack: &str, needle: &str) -> bool {
        let hay = haystack.to_lowercase();
        let mut hay_chars = hay.chars();
        for nc in needle.chars() {
            loop {
                match hay_chars.next() {
                    Some(hc) if hc == nc => break,
                    Some(_) => continue,
                    None => return false,
                }
            }
        }
        true
    }
}

impl Default for CommandPalette {
    fn default() -> Self { Self::new() }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn palette_with_cmds() -> CommandPalette {
        let mut p = CommandPalette::new();
        p.register(Command::new("open", "Open File", 1).category("File"));
        p.register(Command::new("save", "Save File", 2).category("File"));
        p.register(Command::new("fmt", "Format Document", 3).category("Edit"));
        p.register(Command::new("theme", "Toggle Theme", 4).category("View"));
        p
    }

    #[test]
    fn register_and_unregister() {
        let mut p = CommandPalette::new();
        p.register(Command::new("a", "Alpha", 1));
        assert_eq!(p.filtered.len(), 1);
        assert!(p.unregister("a"));
        assert_eq!(p.filtered.len(), 0);
        assert!(!p.unregister("a"));
    }

    #[test]
    fn query_filters_by_label() {
        let mut p = palette_with_cmds();
        p.set_query("open");
        let names: Vec<_> = p.filtered_commands().iter().map(|c| c.label.as_str()).collect();
        assert!(names.contains(&"Open File"));
        assert!(!names.contains(&"Format Document"));
    }

    #[test]
    fn query_fuzzy() {
        let mut p = palette_with_cmds();
        p.set_query("fmtd"); // fuzzy: F-or-m-a-t D-ocument
        let cmds = p.filtered_commands();
        assert_eq!(cmds.len(), 1);
        assert_eq!(cmds[0].id, "fmt");
    }

    #[test]
    fn select_next_previous() {
        let mut p = palette_with_cmds();
        assert_eq!(p.selected().unwrap().id, "open");
        p.select_next();
        assert_eq!(p.selected().unwrap().id, "save");
        p.select_previous();
        assert_eq!(p.selected().unwrap().id, "open");
        // Wrap
        p.select_previous();
        assert_eq!(p.selected().unwrap().id, "theme");
    }

    #[test]
    fn execute_returns_handler_and_closes() {
        let mut p = palette_with_cmds();
        p.open();
        p.select_next(); // save
        let hid = p.execute();
        assert_eq!(hid, Some(2));
        assert!(!p.is_open());
    }

    #[test]
    fn recent_tracked() {
        let mut p = palette_with_cmds();
        p.open();
        p.execute(); // open
        p.open();
        p.select_next();
        p.execute(); // save
        let recent: Vec<_> = p.recent_commands().iter().map(|c| c.id.as_str()).collect();
        assert_eq!(recent, vec!["save", "open"]);
    }

    #[test]
    fn category_grouping() {
        let p = palette_with_cmds();
        let cats = p.commands_by_category();
        assert_eq!(cats["File"].len(), 2);
        assert_eq!(cats["Edit"].len(), 1);
        assert_eq!(cats["View"].len(), 1);
    }

    #[test]
    fn disabled_skipped_on_nav() {
        let mut p = CommandPalette::new();
        p.register(Command::new("a", "Alpha", 1));
        p.register(Command::new("b", "Beta", 2).enabled(false));
        p.register(Command::new("c", "Gamma", 3));
        assert_eq!(p.selected().unwrap().id, "a");
        p.select_next();
        assert_eq!(p.selected().unwrap().id, "c"); // skips disabled b
    }

    #[test]
    fn disabled_cannot_execute() {
        let mut p = CommandPalette::new();
        p.register(Command::new("d", "Disabled", 99).enabled(false));
        p.open();
        assert_eq!(p.execute(), None);
    }

    #[test]
    fn toggle_open_close() {
        let mut p = CommandPalette::new();
        assert!(!p.is_open());
        p.toggle();
        assert!(p.is_open());
        p.toggle();
        assert!(!p.is_open());
    }

    #[test]
    fn query_by_category() {
        let mut p = palette_with_cmds();
        p.set_query("edit");
        let cmds = p.filtered_commands();
        assert!(cmds.iter().any(|c| c.id == "fmt"));
    }

    #[test]
    fn reregister_replaces() {
        let mut p = CommandPalette::new();
        p.register(Command::new("x", "Old", 1));
        p.register(Command::new("x", "New", 2));
        assert_eq!(p.commands.len(), 1);
        assert_eq!(p.commands[0].label, "New");
        assert_eq!(p.commands[0].handler_id, 2);
    }
}
