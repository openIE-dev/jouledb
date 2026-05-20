//! Tree shaking / dead code elimination — symbol-level dependency analysis.
//!
//! Replaces Rollup / Webpack tree shaking with a pure Rust mark-and-sweep
//! model. Tracks exports, imports, side effects, and re-export chains.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Symbols ─────────────────────────────────────────────────────

/// An exported symbol from a module.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExportedSymbol {
    pub name: String,
    pub module_id: u64,
    pub used: bool,
}

impl ExportedSymbol {
    pub fn new(name: impl Into<String>, module_id: u64) -> Self {
        Self {
            name: name.into(),
            module_id,
            used: false,
        }
    }
}

/// An imported symbol consumed by a module.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ImportedSymbol {
    pub name: String,
    pub from_module: u64,
    pub local_name: String,
}

impl ImportedSymbol {
    pub fn new(
        name: impl Into<String>,
        from_module: u64,
        local_name: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            from_module,
            local_name: local_name.into(),
        }
    }
}

// ── Module Info ─────────────────────────────────────────────────

/// Metadata about a module for tree shaking.
#[derive(Debug, Clone)]
pub struct ModuleInfo {
    pub id: u64,
    pub path: String,
    pub exports: Vec<ExportedSymbol>,
    pub imports: Vec<ImportedSymbol>,
    pub has_side_effects: bool,
    /// Re-exports: (local_export_name, source_module_id, source_export_name)
    pub re_exports: Vec<(String, u64, String)>,
}

impl ModuleInfo {
    pub fn new(id: u64, path: impl Into<String>) -> Self {
        Self {
            id,
            path: path.into(),
            exports: Vec::new(),
            imports: Vec::new(),
            has_side_effects: false,
            re_exports: Vec::new(),
        }
    }

    pub fn add_export(&mut self, name: impl Into<String>) {
        self.exports
            .push(ExportedSymbol::new(name, self.id));
    }

    pub fn add_import(&mut self, name: impl Into<String>, from: u64, local: impl Into<String>) {
        self.imports
            .push(ImportedSymbol::new(name, from, local));
    }

    pub fn add_re_export(
        &mut self,
        local_name: impl Into<String>,
        source_module: u64,
        source_name: impl Into<String>,
    ) {
        self.re_exports
            .push((local_name.into(), source_module, source_name.into()));
    }
}

// ── Tree Shaker ─────────────────────────────────────────────────

/// Result of tree shaking analysis.
#[derive(Debug, Clone)]
pub struct ShakeResult {
    /// Symbols that are used (reachable from entry exports).
    pub used_symbols: Vec<(u64, String)>,
    /// Symbols that are unused and can be eliminated.
    pub unused_symbols: Vec<(u64, String)>,
    /// Modules that are included due to side effects.
    pub side_effect_modules: Vec<u64>,
    /// Number of removed symbols.
    pub removed_count: usize,
    /// Number of total symbols.
    pub total_count: usize,
}

/// The tree shaker engine.
#[derive(Debug, Default)]
pub struct TreeShaker {
    modules: HashMap<u64, ModuleInfo>,
}

impl TreeShaker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a module to the analysis.
    pub fn add_module(&mut self, module: ModuleInfo) {
        self.modules.insert(module.id, module);
    }

    /// Get a module by ID.
    pub fn get_module(&self, id: u64) -> Option<&ModuleInfo> {
        self.modules.get(&id)
    }

    /// Resolve a re-export chain to find the original module and export name.
    pub fn resolve_re_export(&self, module_id: u64, name: &str) -> (u64, String) {
        let mut current_module = module_id;
        let mut current_name = name.to_string();
        let mut visited = HashSet::new();

        loop {
            if !visited.insert((current_module, current_name.clone())) {
                break; // Circular re-export
            }
            if let Some(module) = self.modules.get(&current_module) {
                let found = module
                    .re_exports
                    .iter()
                    .find(|(local, _, _)| local == &current_name);
                if let Some((_, source_mod, source_name)) = found {
                    current_module = *source_mod;
                    current_name = source_name.clone();
                    continue;
                }
            }
            break;
        }

        (current_module, current_name)
    }

    /// Mark phase: starting from entry point exports, trace all used symbols.
    fn mark(&self, entry_ids: &[u64]) -> HashSet<(u64, String)> {
        let mut used: HashSet<(u64, String)> = HashSet::new();
        let mut queue: VecDeque<(u64, String)> = VecDeque::new();

        // Seed: all exports of entry modules
        for &entry_id in entry_ids {
            if let Some(module) = self.modules.get(&entry_id) {
                for exp in &module.exports {
                    queue.push_back((entry_id, exp.name.clone()));
                }
            }
        }

        while let Some((mod_id, sym_name)) = queue.pop_front() {
            // Resolve re-exports
            let (resolved_mod, resolved_name) = self.resolve_re_export(mod_id, &sym_name);

            if !used.insert((resolved_mod, resolved_name.clone())) {
                continue;
            }

            // Find the module that defines this symbol, look at its imports
            if let Some(module) = self.modules.get(&resolved_mod) {
                // All imports of this module are needed (conservative)
                for imp in &module.imports {
                    queue.push_back((imp.from_module, imp.name.clone()));
                }
            }
        }

        used
    }

    /// Run the full tree shaking analysis.
    pub fn shake(&self, entry_ids: &[u64]) -> ShakeResult {
        let used = self.mark(entry_ids);

        let mut used_symbols = Vec::new();
        let mut unused_symbols = Vec::new();
        let mut side_effect_modules = Vec::new();
        let mut total = 0;

        for module in self.modules.values() {
            if module.has_side_effects {
                side_effect_modules.push(module.id);
            }

            for exp in &module.exports {
                total += 1;
                let key = (module.id, exp.name.clone());
                if used.contains(&key) || module.has_side_effects {
                    used_symbols.push((module.id, exp.name.clone()));
                } else {
                    unused_symbols.push((module.id, exp.name.clone()));
                }
            }
        }

        side_effect_modules.sort();
        used_symbols.sort();
        unused_symbols.sort();

        let removed = unused_symbols.len();

        ShakeResult {
            used_symbols,
            unused_symbols,
            side_effect_modules,
            removed_count: removed,
            total_count: total,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_modules() -> TreeShaker {
        let mut shaker = TreeShaker::new();

        // Module 1 (entry): exports `main`, imports `helper` from 2
        let mut m1 = ModuleInfo::new(1, "src/main.js");
        m1.add_export("main");
        m1.add_import("helper", 2, "helper");
        shaker.add_module(m1);

        // Module 2: exports `helper` and `unused_fn`
        let mut m2 = ModuleInfo::new(2, "src/utils.js");
        m2.add_export("helper");
        m2.add_export("unused_fn");
        shaker.add_module(m2);

        // Module 3: exports `orphan` — no one imports it
        let mut m3 = ModuleInfo::new(3, "src/orphan.js");
        m3.add_export("orphan");
        shaker.add_module(m3);

        shaker
    }

    #[test]
    fn mark_entry_exports() {
        let shaker = setup_modules();
        let result = shaker.shake(&[1]);
        // `main` from module 1 should be used
        assert!(result.used_symbols.contains(&(1, "main".into())));
    }

    #[test]
    fn mark_transitive_imports() {
        let shaker = setup_modules();
        let result = shaker.shake(&[1]);
        // `helper` from module 2 should be used (imported by entry)
        assert!(result.used_symbols.contains(&(2, "helper".into())));
    }

    #[test]
    fn unused_export_detected() {
        let shaker = setup_modules();
        let result = shaker.shake(&[1]);
        // `unused_fn` from module 2 is not imported by anyone reachable
        assert!(result.unused_symbols.contains(&(2, "unused_fn".into())));
    }

    #[test]
    fn orphan_module_unused() {
        let shaker = setup_modules();
        let result = shaker.shake(&[1]);
        assert!(result.unused_symbols.contains(&(3, "orphan".into())));
    }

    #[test]
    fn removed_count_correct() {
        let shaker = setup_modules();
        let result = shaker.shake(&[1]);
        // unused_fn + orphan = 2 removed
        assert_eq!(result.removed_count, 2);
    }

    #[test]
    fn total_count_correct() {
        let shaker = setup_modules();
        let result = shaker.shake(&[1]);
        assert_eq!(result.total_count, 4); // main + helper + unused_fn + orphan
    }

    #[test]
    fn side_effect_module_preserved() {
        let mut shaker = setup_modules();
        // Make module 3 have side effects
        if let Some(m) = shaker.modules.get_mut(&3) {
            m.has_side_effects = true;
        }
        let result = shaker.shake(&[1]);
        // `orphan` should now be in used because module has side effects
        assert!(result.used_symbols.contains(&(3, "orphan".into())));
        assert!(result.side_effect_modules.contains(&3));
    }

    #[test]
    fn re_export_resolution() {
        let mut shaker = TreeShaker::new();

        // Module 1 re-exports `foo` from module 2
        let mut m1 = ModuleInfo::new(1, "index.js");
        m1.add_export("foo");
        m1.add_re_export("foo", 2, "originalFoo");
        shaker.add_module(m1);

        let mut m2 = ModuleInfo::new(2, "lib.js");
        m2.add_export("originalFoo");
        shaker.add_module(m2);

        let (mod_id, name) = shaker.resolve_re_export(1, "foo");
        assert_eq!(mod_id, 2);
        assert_eq!(name, "originalFoo");
    }

    #[test]
    fn multiple_entry_points() {
        let mut shaker = setup_modules();
        // Add module 3 as another entry
        let result = shaker.shake(&[1, 3]);
        assert!(result.used_symbols.contains(&(3, "orphan".into())));
    }

    #[test]
    fn empty_module_graph() {
        let shaker = TreeShaker::new();
        let result = shaker.shake(&[]);
        assert_eq!(result.total_count, 0);
        assert_eq!(result.removed_count, 0);
    }

    #[test]
    fn exported_symbol_equality() {
        let s1 = ExportedSymbol::new("foo", 1);
        let s2 = ExportedSymbol::new("foo", 1);
        assert_eq!(s1, s2);
    }

    #[test]
    fn imported_symbol_fields() {
        let s = ImportedSymbol::new("default", 5, "myDefault");
        assert_eq!(s.name, "default");
        assert_eq!(s.from_module, 5);
        assert_eq!(s.local_name, "myDefault");
    }
}
