//! Symbol table with scopes — nested scope chain, symbol lookup (local then
//! parent), symbol kinds (variable/function/type), redefinition checking,
//! scope enter/exit, symbol enumeration, import/export tracking.

use std::collections::HashMap;
use std::fmt;

// ── Symbol kinds ────────────────────────────────────────────────────────────

/// The kind of a symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolKind {
    /// A variable binding.
    Variable,
    /// A constant binding.
    Constant,
    /// A function definition.
    Function {
        /// Number of parameters.
        arity: usize,
    },
    /// A type / struct / enum definition.
    Type,
    /// A module.
    Module,
    /// A macro.
    Macro,
    /// A parameter (function argument).
    Parameter,
    /// A label (for gotos or loop labels).
    Label,
}

impl fmt::Display for SymbolKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Variable => write!(f, "variable"),
            Self::Constant => write!(f, "constant"),
            Self::Function { arity } => write!(f, "function/{arity}"),
            Self::Type => write!(f, "type"),
            Self::Module => write!(f, "module"),
            Self::Macro => write!(f, "macro"),
            Self::Parameter => write!(f, "parameter"),
            Self::Label => write!(f, "label"),
        }
    }
}

// ── Symbol ──────────────────────────────────────────────────────────────────

/// A symbol entry in the table.
#[derive(Debug, Clone, PartialEq)]
pub struct Symbol {
    /// The symbol's name.
    pub name: String,
    /// What kind of symbol this is.
    pub kind: SymbolKind,
    /// Optional type annotation (as a string for simplicity).
    pub type_annotation: Option<String>,
    /// Whether this symbol is mutable.
    pub mutable: bool,
    /// Whether this symbol is exported (visible outside its module).
    pub exported: bool,
    /// The scope depth at which this symbol was defined.
    pub scope_depth: usize,
    /// Optional documentation string.
    pub doc: Option<String>,
}

impl Symbol {
    /// Create a new variable symbol.
    pub fn variable(name: &str, mutable: bool) -> Self {
        Self {
            name: name.into(),
            kind: SymbolKind::Variable,
            type_annotation: None,
            mutable,
            exported: false,
            scope_depth: 0,
            doc: None,
        }
    }

    /// Create a new constant symbol.
    pub fn constant(name: &str) -> Self {
        Self {
            name: name.into(),
            kind: SymbolKind::Constant,
            type_annotation: None,
            mutable: false,
            exported: false,
            scope_depth: 0,
            doc: None,
        }
    }

    /// Create a new function symbol.
    pub fn function(name: &str, arity: usize) -> Self {
        Self {
            name: name.into(),
            kind: SymbolKind::Function { arity },
            type_annotation: None,
            mutable: false,
            exported: false,
            scope_depth: 0,
            doc: None,
        }
    }

    /// Create a new type symbol.
    pub fn type_sym(name: &str) -> Self {
        Self {
            name: name.into(),
            kind: SymbolKind::Type,
            type_annotation: None,
            mutable: false,
            exported: false,
            scope_depth: 0,
            doc: None,
        }
    }

    /// Create a parameter symbol.
    pub fn parameter(name: &str) -> Self {
        Self {
            name: name.into(),
            kind: SymbolKind::Parameter,
            type_annotation: None,
            mutable: false,
            exported: false,
            scope_depth: 0,
            doc: None,
        }
    }

    /// Attach a type annotation.
    pub fn with_type(mut self, ty: &str) -> Self {
        self.type_annotation = Some(ty.into());
        self
    }

    /// Mark as exported.
    pub fn with_export(mut self) -> Self {
        self.exported = true;
        self
    }

    /// Attach documentation.
    pub fn with_doc(mut self, doc: &str) -> Self {
        self.doc = Some(doc.into());
        self
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name, self.kind)?;
        if let Some(ref ty) = self.type_annotation {
            write!(f, ": {ty}")?;
        }
        if self.exported {
            write!(f, " [exported]")?;
        }
        Ok(())
    }
}

// ── Import / Export ─────────────────────────────────────────────────────────

/// An import declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Import {
    /// The module path being imported from.
    pub module_path: String,
    /// The specific symbol being imported (or "*" for wildcard).
    pub symbol_name: String,
    /// Local alias (if different from original name).
    pub alias: Option<String>,
}

impl Import {
    pub fn new(module_path: &str, symbol_name: &str) -> Self {
        Self {
            module_path: module_path.into(),
            symbol_name: symbol_name.into(),
            alias: None,
        }
    }

    pub fn with_alias(mut self, alias: &str) -> Self {
        self.alias = Some(alias.into());
        self
    }

    /// The local name this import binds to.
    pub fn local_name(&self) -> &str {
        self.alias.as_deref().unwrap_or(&self.symbol_name)
    }
}

impl fmt::Display for Import {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "import {} from {}", self.symbol_name, self.module_path)?;
        if let Some(ref alias) = self.alias {
            write!(f, " as {alias}")?;
        }
        Ok(())
    }
}

/// An export declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Export {
    /// The local name being exported.
    pub local_name: String,
    /// The external name (if renamed).
    pub external_name: Option<String>,
}

impl Export {
    pub fn new(local_name: &str) -> Self {
        Self {
            local_name: local_name.into(),
            external_name: None,
        }
    }

    pub fn with_rename(mut self, external: &str) -> Self {
        self.external_name = Some(external.into());
        self
    }

    pub fn public_name(&self) -> &str {
        self.external_name.as_deref().unwrap_or(&self.local_name)
    }
}

// ── Scope ───────────────────────────────────────────────────────────────────

/// A single scope level.
#[derive(Debug, Clone)]
struct Scope {
    /// Symbols defined in this scope.
    symbols: HashMap<String, Symbol>,
    /// Whether this scope is a function boundary (prevents leaking to parent).
    is_function_scope: bool,
    /// Name of this scope (for debugging).
    name: Option<String>,
}

impl Scope {
    fn new(is_function_scope: bool) -> Self {
        Self {
            symbols: HashMap::new(),
            is_function_scope,
            name: None,
        }
    }

    fn named(name: &str, is_function_scope: bool) -> Self {
        Self {
            symbols: HashMap::new(),
            is_function_scope,
            name: Some(name.into()),
        }
    }
}

// ── Symbol table errors ─────────────────────────────────────────────────────

/// Errors from symbol table operations.
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolError {
    /// Symbol already defined in current scope.
    Redefined(String),
    /// Symbol not found.
    Undefined(String),
    /// Cannot mutate an immutable symbol.
    Immutable(String),
    /// Scope underflow (tried to exit the global scope).
    ScopeUnderflow,
    /// Import not resolved.
    UnresolvedImport(String),
    /// Other.
    Other(String),
}

impl fmt::Display for SymbolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Redefined(s) => write!(f, "symbol '{s}' already defined in this scope"),
            Self::Undefined(s) => write!(f, "symbol '{s}' not found"),
            Self::Immutable(s) => write!(f, "cannot mutate immutable symbol '{s}'"),
            Self::ScopeUnderflow => write!(f, "cannot exit global scope"),
            Self::UnresolvedImport(s) => write!(f, "unresolved import: {s}"),
            Self::Other(msg) => write!(f, "{msg}"),
        }
    }
}

// ── Symbol table ────────────────────────────────────────────────────────────

/// A symbol table with nested scopes, import/export tracking.
pub struct SymbolTable {
    scopes: Vec<Scope>,
    imports: Vec<Import>,
    exports: Vec<Export>,
}

impl SymbolTable {
    /// Create a new symbol table with a global scope.
    pub fn new() -> Self {
        Self {
            scopes: vec![Scope::new(false)],
            imports: Vec::new(),
            exports: Vec::new(),
        }
    }

    /// Current scope depth (0 = global).
    pub fn depth(&self) -> usize {
        self.scopes.len().saturating_sub(1)
    }

    /// Enter a new scope.
    pub fn enter_scope(&mut self) {
        self.scopes.push(Scope::new(false));
    }

    /// Enter a named scope.
    pub fn enter_named_scope(&mut self, name: &str) {
        self.scopes.push(Scope::named(name, false));
    }

    /// Enter a function scope (blocks variable lookup from leaking up).
    pub fn enter_function_scope(&mut self, name: &str) {
        self.scopes.push(Scope::named(name, true));
    }

    /// Exit the current scope.
    pub fn exit_scope(&mut self) -> Result<(), SymbolError> {
        if self.scopes.len() <= 1 {
            return Err(SymbolError::ScopeUnderflow);
        }
        self.scopes.pop();
        Ok(())
    }

    /// Define a symbol in the current scope.
    pub fn define(&mut self, mut sym: Symbol) -> Result<(), SymbolError> {
        sym.scope_depth = self.depth();
        let name = sym.name.clone();
        let scope = self.scopes.last_mut().unwrap();
        if scope.symbols.contains_key(&name) {
            return Err(SymbolError::Redefined(name));
        }
        scope.symbols.insert(name, sym);
        Ok(())
    }

    /// Look up a symbol by name, searching from innermost to outermost scope.
    /// Stops at function scope boundaries for local variables.
    pub fn lookup(&self, name: &str) -> Result<&Symbol, SymbolError> {
        for scope in self.scopes.iter().rev() {
            if let Some(sym) = scope.symbols.get(name) {
                return Ok(sym);
            }
            // If we hit a function scope boundary, only continue for
            // functions, types, and modules (not variables/parameters)
            if scope.is_function_scope {
                // Still search parent scopes but only for non-locals
                for parent in self.scopes.iter().rev().skip(1) {
                    if let Some(sym) = parent.symbols.get(name) {
                        match sym.kind {
                            SymbolKind::Function { .. }
                            | SymbolKind::Type
                            | SymbolKind::Module
                            | SymbolKind::Constant => return Ok(sym),
                            _ => {}
                        }
                    }
                }
                break;
            }
        }
        Err(SymbolError::Undefined(name.into()))
    }

    /// Look up a symbol only in the current scope (no parent search).
    pub fn lookup_local(&self, name: &str) -> Option<&Symbol> {
        self.scopes.last().and_then(|s| s.symbols.get(name))
    }

    /// Check if a symbol is defined in any reachable scope.
    pub fn is_defined(&self, name: &str) -> bool {
        self.lookup(name).is_ok()
    }

    /// Update a symbol (e.g., change its type annotation).
    pub fn update(&mut self, name: &str, updater: impl FnOnce(&mut Symbol)) -> Result<(), SymbolError> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(sym) = scope.symbols.get_mut(name) {
                updater(sym);
                return Ok(());
            }
        }
        Err(SymbolError::Undefined(name.into()))
    }

    /// Enumerate all symbols in the current scope.
    pub fn current_scope_symbols(&self) -> Vec<&Symbol> {
        self.scopes
            .last()
            .map(|s| s.symbols.values().collect())
            .unwrap_or_default()
    }

    /// Enumerate all symbols across all reachable scopes.
    pub fn all_symbols(&self) -> Vec<&Symbol> {
        let mut out = Vec::new();
        for scope in &self.scopes {
            for sym in scope.symbols.values() {
                out.push(sym);
            }
        }
        out
    }

    /// Enumerate all symbols of a given kind.
    pub fn symbols_of_kind(&self, kind_match: &SymbolKind) -> Vec<&Symbol> {
        self.all_symbols()
            .into_iter()
            .filter(|s| std::mem::discriminant(&s.kind) == std::mem::discriminant(kind_match))
            .collect()
    }

    /// Count symbols in the current scope.
    pub fn current_scope_count(&self) -> usize {
        self.scopes.last().map_or(0, |s| s.symbols.len())
    }

    /// Name of the current scope.
    pub fn current_scope_name(&self) -> Option<&str> {
        self.scopes.last().and_then(|s| s.name.as_deref())
    }

    // ── Imports ─────────────────────────────────────────────────────────

    /// Register an import.
    pub fn add_import(&mut self, import: Import) {
        self.imports.push(import);
    }

    /// Get all registered imports.
    pub fn imports(&self) -> &[Import] {
        &self.imports
    }

    /// Check if a symbol name comes from an import.
    pub fn is_imported(&self, name: &str) -> bool {
        self.imports.iter().any(|i| i.local_name() == name)
    }

    /// Resolve an import: look it up and define it in the current scope.
    pub fn resolve_import(
        &mut self,
        import: &Import,
        resolved_sym: Symbol,
    ) -> Result<(), SymbolError> {
        let local_name = import.local_name().to_string();
        let mut sym = resolved_sym;
        sym.name = local_name;
        self.define(sym)
    }

    // ── Exports ─────────────────────────────────────────────────────────

    /// Register an export.
    pub fn add_export(&mut self, export: Export) {
        self.exports.push(export);
    }

    /// Get all registered exports.
    pub fn exports(&self) -> &[Export] {
        &self.exports
    }

    /// Get all exported symbols (resolved against the table).
    pub fn exported_symbols(&self) -> Vec<(&Export, Option<&Symbol>)> {
        self.exports
            .iter()
            .map(|e| {
                let sym = self.lookup(&e.local_name).ok();
                (e, sym)
            })
            .collect()
    }

    /// Mark a symbol as exported (in-place).
    pub fn mark_exported(&mut self, name: &str) -> Result<(), SymbolError> {
        self.update(name, |sym| {
            sym.exported = true;
        })?;
        self.exports.push(Export::new(name));
        Ok(())
    }

    // ── Scope chain info ────────────────────────────────────────────────

    /// Return the scope chain as a list of scope names/depths.
    pub fn scope_chain(&self) -> Vec<(usize, Option<String>)> {
        self.scopes
            .iter()
            .enumerate()
            .map(|(i, s)| (i, s.name.clone()))
            .collect()
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for SymbolTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SymbolTable (depth: {}):", self.depth())?;
        for (i, scope) in self.scopes.iter().enumerate() {
            let name = scope.name.as_deref().unwrap_or("anonymous");
            writeln!(f, "  Scope {i} ({name}):")?;
            for (k, v) in &scope.symbols {
                writeln!(f, "    {k}: {v}")?;
            }
        }
        Ok(())
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_define_and_lookup() {
        let mut st = SymbolTable::new();
        st.define(Symbol::variable("x", true)).unwrap();
        let sym = st.lookup("x").unwrap();
        assert_eq!(sym.name, "x");
        assert!(sym.mutable);
    }

    #[test]
    fn test_undefined_error() {
        let st = SymbolTable::new();
        let err = st.lookup("x").unwrap_err();
        assert_eq!(err, SymbolError::Undefined("x".into()));
    }

    #[test]
    fn test_redefinition_error() {
        let mut st = SymbolTable::new();
        st.define(Symbol::variable("x", false)).unwrap();
        let err = st.define(Symbol::variable("x", false)).unwrap_err();
        assert_eq!(err, SymbolError::Redefined("x".into()));
    }

    #[test]
    fn test_scope_shadowing() {
        let mut st = SymbolTable::new();
        st.define(Symbol::variable("x", false).with_type("int")).unwrap();
        st.enter_scope();
        st.define(Symbol::variable("x", true).with_type("bool")).unwrap();
        let inner = st.lookup("x").unwrap();
        assert_eq!(inner.type_annotation.as_deref(), Some("bool"));
        st.exit_scope().unwrap();
        let outer = st.lookup("x").unwrap();
        assert_eq!(outer.type_annotation.as_deref(), Some("int"));
    }

    #[test]
    fn test_scope_depth() {
        let mut st = SymbolTable::new();
        assert_eq!(st.depth(), 0);
        st.enter_scope();
        assert_eq!(st.depth(), 1);
        st.enter_scope();
        assert_eq!(st.depth(), 2);
        st.exit_scope().unwrap();
        assert_eq!(st.depth(), 1);
    }

    #[test]
    fn test_scope_underflow() {
        let mut st = SymbolTable::new();
        let err = st.exit_scope().unwrap_err();
        assert_eq!(err, SymbolError::ScopeUnderflow);
    }

    #[test]
    fn test_function_scope_blocks_locals() {
        let mut st = SymbolTable::new();
        st.define(Symbol::variable("outer_var", false)).unwrap();
        st.define(Symbol::function("global_fn", 0)).unwrap();
        st.enter_function_scope("my_func");
        // Should NOT see outer_var (it's a variable across function boundary)
        assert!(st.lookup("outer_var").is_err());
        // Should still see global_fn (functions cross boundaries)
        assert!(st.lookup("global_fn").is_ok());
    }

    #[test]
    fn test_lookup_local() {
        let mut st = SymbolTable::new();
        st.define(Symbol::variable("x", false)).unwrap();
        st.enter_scope();
        st.define(Symbol::variable("y", false)).unwrap();
        assert!(st.lookup_local("y").is_some());
        assert!(st.lookup_local("x").is_none()); // x is in parent scope
    }

    #[test]
    fn test_is_defined() {
        let mut st = SymbolTable::new();
        st.define(Symbol::variable("x", false)).unwrap();
        assert!(st.is_defined("x"));
        assert!(!st.is_defined("y"));
    }

    #[test]
    fn test_update_symbol() {
        let mut st = SymbolTable::new();
        st.define(Symbol::variable("x", false)).unwrap();
        st.update("x", |sym| {
            sym.type_annotation = Some("int".into());
        })
        .unwrap();
        let sym = st.lookup("x").unwrap();
        assert_eq!(sym.type_annotation.as_deref(), Some("int"));
    }

    #[test]
    fn test_current_scope_symbols() {
        let mut st = SymbolTable::new();
        st.define(Symbol::variable("a", false)).unwrap();
        st.define(Symbol::variable("b", false)).unwrap();
        st.enter_scope();
        st.define(Symbol::variable("c", false)).unwrap();
        let syms = st.current_scope_symbols();
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "c");
    }

    #[test]
    fn test_all_symbols() {
        let mut st = SymbolTable::new();
        st.define(Symbol::variable("a", false)).unwrap();
        st.enter_scope();
        st.define(Symbol::variable("b", false)).unwrap();
        let all = st.all_symbols();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_symbols_of_kind() {
        let mut st = SymbolTable::new();
        st.define(Symbol::variable("x", false)).unwrap();
        st.define(Symbol::function("f", 2)).unwrap();
        st.define(Symbol::type_sym("MyType")).unwrap();
        let fns = st.symbols_of_kind(&SymbolKind::Function { arity: 0 });
        assert_eq!(fns.len(), 1);
        assert_eq!(fns[0].name, "f");
    }

    #[test]
    fn test_import_add_and_query() {
        let mut st = SymbolTable::new();
        st.add_import(Import::new("std.io", "println"));
        assert!(st.is_imported("println"));
        assert!(!st.is_imported("readln"));
        assert_eq!(st.imports().len(), 1);
    }

    #[test]
    fn test_import_with_alias() {
        let import = Import::new("std.collections", "HashMap").with_alias("Map");
        assert_eq!(import.local_name(), "Map");
    }

    #[test]
    fn test_resolve_import() {
        let mut st = SymbolTable::new();
        let import = Import::new("math", "sqrt");
        let resolved = Symbol::function("sqrt", 1);
        st.resolve_import(&import, resolved).unwrap();
        let sym = st.lookup("sqrt").unwrap();
        assert!(matches!(sym.kind, SymbolKind::Function { arity: 1 }));
    }

    #[test]
    fn test_export() {
        let mut st = SymbolTable::new();
        st.define(Symbol::function("my_func", 0)).unwrap();
        st.mark_exported("my_func").unwrap();
        let exported = st.exported_symbols();
        assert_eq!(exported.len(), 1);
        assert!(exported[0].1.is_some());
        let sym = st.lookup("my_func").unwrap();
        assert!(sym.exported);
    }

    #[test]
    fn test_export_with_rename() {
        let mut st = SymbolTable::new();
        st.define(Symbol::variable("internal_name", false)).unwrap();
        st.add_export(Export::new("internal_name").with_rename("publicName"));
        let exports = st.exports();
        assert_eq!(exports[0].public_name(), "publicName");
    }

    #[test]
    fn test_named_scope() {
        let mut st = SymbolTable::new();
        st.enter_named_scope("block_1");
        assert_eq!(st.current_scope_name(), Some("block_1"));
        st.exit_scope().unwrap();
        assert_eq!(st.current_scope_name(), None);
    }

    #[test]
    fn test_scope_chain() {
        let mut st = SymbolTable::new();
        st.enter_named_scope("func");
        st.enter_named_scope("if_body");
        let chain = st.scope_chain();
        assert_eq!(chain.len(), 3);
        assert_eq!(chain[1].1.as_deref(), Some("func"));
        assert_eq!(chain[2].1.as_deref(), Some("if_body"));
    }

    #[test]
    fn test_symbol_display() {
        let sym = Symbol::function("add", 2).with_type("(int, int) -> int").with_export();
        let s = format!("{sym}");
        assert!(s.contains("add"));
        assert!(s.contains("function/2"));
        assert!(s.contains("exported"));
    }

    #[test]
    fn test_constant_across_function_scope() {
        let mut st = SymbolTable::new();
        st.define(Symbol::constant("PI").with_type("float")).unwrap();
        st.enter_function_scope("calc");
        // Constants should be visible across function scope boundaries
        assert!(st.lookup("PI").is_ok());
    }

    #[test]
    fn test_current_scope_count() {
        let mut st = SymbolTable::new();
        st.define(Symbol::variable("a", false)).unwrap();
        st.define(Symbol::variable("b", false)).unwrap();
        assert_eq!(st.current_scope_count(), 2);
        st.enter_scope();
        assert_eq!(st.current_scope_count(), 0);
    }

    #[test]
    fn test_parameter_symbol() {
        let mut st = SymbolTable::new();
        st.define(Symbol::parameter("x").with_type("int")).unwrap();
        let sym = st.lookup("x").unwrap();
        assert!(matches!(sym.kind, SymbolKind::Parameter));
    }

    #[test]
    fn test_import_display() {
        let import = Import::new("std.collections", "Vec").with_alias("MyVec");
        let s = format!("{import}");
        assert!(s.contains("Vec"));
        assert!(s.contains("std.collections"));
        assert!(s.contains("MyVec"));
    }
}
