//! WASM module model — module sections (type/func/memory/table/export/import/
//! data/code), function types, linear memory model, table elements,
//! import/export resolution, module validation, binary format concepts.

use std::collections::HashMap;
use std::fmt;

// ── Value Types ────────────────────────────────────────────────────────────

/// WASM value types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValType {
    I32,
    I64,
    F32,
    F64,
    FuncRef,
    ExternRef,
}

impl fmt::Display for ValType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I32 => write!(f, "i32"),
            Self::I64 => write!(f, "i64"),
            Self::F32 => write!(f, "f32"),
            Self::F64 => write!(f, "f64"),
            Self::FuncRef => write!(f, "funcref"),
            Self::ExternRef => write!(f, "externref"),
        }
    }
}

// ── Function Types ─────────────────────────────────────────────────────────

/// A function signature: parameter types → result types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FuncType {
    pub params: Vec<ValType>,
    pub results: Vec<ValType>,
}

impl FuncType {
    pub fn new(params: Vec<ValType>, results: Vec<ValType>) -> Self {
        Self { params, results }
    }

    /// Arity of the function (number of parameters).
    pub fn arity(&self) -> usize {
        self.params.len()
    }

    /// Number of result values.
    pub fn result_count(&self) -> usize {
        self.results.len()
    }

    /// Check if this type signature matches another.
    pub fn matches(&self, other: &FuncType) -> bool {
        self.params == other.params && self.results == other.results
    }
}

impl fmt::Display for FuncType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, p) in self.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{p}")?;
        }
        write!(f, ") -> (")?;
        for (i, r) in self.results.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{r}")?;
        }
        write!(f, ")")
    }
}

// ── Limits ─────────────────────────────────────────────────────────────────

/// Limits for memories and tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub min: u32,
    pub max: Option<u32>,
}

impl Limits {
    pub fn new(min: u32, max: Option<u32>) -> Self {
        Self { min, max }
    }

    /// Validate that limits are well-formed.
    pub fn validate(&self) -> Result<(), ModuleError> {
        if let Some(mx) = self.max {
            if mx < self.min {
                return Err(ModuleError::Validation(format!(
                    "max ({mx}) < min ({})",
                    self.min
                )));
            }
        }
        Ok(())
    }

    /// Check if `n` satisfies these limits.
    pub fn contains(&self, n: u32) -> bool {
        n >= self.min && self.max.map_or(true, |mx| n <= mx)
    }
}

// ── Memory Type ────────────────────────────────────────────────────────────

/// WASM linear memory type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryType {
    pub limits: Limits,
}

impl MemoryType {
    pub fn new(min_pages: u32, max_pages: Option<u32>) -> Self {
        Self {
            limits: Limits::new(min_pages, max_pages),
        }
    }

    /// Byte size of `min` pages (each page = 64 KiB).
    pub fn min_bytes(&self) -> u64 {
        self.limits.min as u64 * 65536
    }

    /// Byte size of `max` pages if bounded.
    pub fn max_bytes(&self) -> Option<u64> {
        self.limits.max.map(|m| m as u64 * 65536)
    }
}

// ── Table Type ─────────────────────────────────────────────────────────────

/// WASM table type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableType {
    pub elem_type: ValType,
    pub limits: Limits,
}

impl TableType {
    pub fn new(elem_type: ValType, min: u32, max: Option<u32>) -> Self {
        Self {
            elem_type,
            limits: Limits::new(min, max),
        }
    }
}

// ── Global Type ────────────────────────────────────────────────────────────

/// Mutability of a global.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mutability {
    Const,
    Var,
}

/// WASM global type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalType {
    pub val_type: ValType,
    pub mutability: Mutability,
}

impl GlobalType {
    pub fn new(val_type: ValType, mutability: Mutability) -> Self {
        Self {
            val_type,
            mutability,
        }
    }
}

// ── Import / Export ────────────────────────────────────────────────────────

/// What an import or export refers to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExternKind {
    Func(u32),
    Table(TableType),
    Memory(MemoryType),
    Global(GlobalType),
}

/// An import descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Import {
    pub module: String,
    pub name: String,
    pub kind: ExternKind,
}

/// An export descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Export {
    pub name: String,
    pub kind: ExternKind,
}

// ── Data Segment ───────────────────────────────────────────────────────────

/// A data segment that initializes linear memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataSegment {
    pub memory_index: u32,
    pub offset: u32,
    pub data: Vec<u8>,
}

impl DataSegment {
    pub fn new(memory_index: u32, offset: u32, data: Vec<u8>) -> Self {
        Self {
            memory_index,
            offset,
            data,
        }
    }

    /// End address (non-inclusive) of this segment.
    pub fn end_offset(&self) -> u64 {
        self.offset as u64 + self.data.len() as u64
    }
}

// ── Element Segment ────────────────────────────────────────────────────────

/// An element segment that initializes a table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElemSegment {
    pub table_index: u32,
    pub offset: u32,
    pub func_indices: Vec<u32>,
}

// ── Code Body ──────────────────────────────────────────────────────────────

/// Local variable declarations inside a function body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalDecl {
    pub count: u32,
    pub val_type: ValType,
}

/// A function body (locals + byte-level code placeholder).
#[derive(Debug, Clone, PartialEq)]
pub struct FuncBody {
    pub locals: Vec<LocalDecl>,
    /// Raw instruction bytes (in a real impl these would be decoded instructions).
    pub code_size: u32,
    /// Symbolic index of the type this body belongs to.
    pub type_index: u32,
}

// ── Module Errors ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleError {
    Validation(String),
    ImportNotFound { module: String, name: String },
    ExportNotFound(String),
    DuplicateExport(String),
    TypeMismatch(String),
    MemoryLimitExceeded { requested: u32, max: u32 },
    TooManySections(String),
}

impl fmt::Display for ModuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(msg) => write!(f, "validation error: {msg}"),
            Self::ImportNotFound { module, name } => {
                write!(f, "import not found: {module}.{name}")
            }
            Self::ExportNotFound(n) => write!(f, "export not found: {n}"),
            Self::DuplicateExport(n) => write!(f, "duplicate export: {n}"),
            Self::TypeMismatch(msg) => write!(f, "type mismatch: {msg}"),
            Self::MemoryLimitExceeded { requested, max } => {
                write!(f, "memory limit exceeded: {requested} > {max}")
            }
            Self::TooManySections(msg) => write!(f, "too many sections: {msg}"),
        }
    }
}

// ── Section IDs (binary format concept) ────────────────────────────────────

/// WASM binary section IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SectionId {
    Custom = 0,
    Type = 1,
    Import = 2,
    Function = 3,
    Table = 4,
    Memory = 5,
    Global = 6,
    Export = 7,
    Start = 8,
    Element = 9,
    Code = 10,
    Data = 11,
    DataCount = 12,
}

impl SectionId {
    /// Parse a section ID from a byte.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Custom),
            1 => Some(Self::Type),
            2 => Some(Self::Import),
            3 => Some(Self::Function),
            4 => Some(Self::Table),
            5 => Some(Self::Memory),
            6 => Some(Self::Global),
            7 => Some(Self::Export),
            8 => Some(Self::Start),
            9 => Some(Self::Element),
            10 => Some(Self::Code),
            11 => Some(Self::Data),
            12 => Some(Self::DataCount),
            _ => None,
        }
    }

    /// Expected order of this section in the binary.
    pub fn order(&self) -> u8 {
        *self as u8
    }
}

// ── Module ─────────────────────────────────────────────────────────────────

/// A WASM module — the top-level container.
#[derive(Debug, Clone)]
pub struct WasmModule {
    /// Type section: function signatures.
    pub types: Vec<FuncType>,
    /// Import section.
    pub imports: Vec<Import>,
    /// Function section: index into `types` for each function.
    pub func_type_indices: Vec<u32>,
    /// Table section.
    pub tables: Vec<TableType>,
    /// Memory section.
    pub memories: Vec<MemoryType>,
    /// Global section.
    pub globals: Vec<GlobalType>,
    /// Export section.
    pub exports: Vec<Export>,
    /// Start function index, if any.
    pub start: Option<u32>,
    /// Element segments.
    pub elements: Vec<ElemSegment>,
    /// Code section (function bodies).
    pub code: Vec<FuncBody>,
    /// Data segments.
    pub data: Vec<DataSegment>,
    /// Custom section names → payloads.
    pub custom_sections: HashMap<String, Vec<u8>>,
}

impl WasmModule {
    /// Create an empty module.
    pub fn new() -> Self {
        Self {
            types: Vec::new(),
            imports: Vec::new(),
            func_type_indices: Vec::new(),
            tables: Vec::new(),
            memories: Vec::new(),
            globals: Vec::new(),
            exports: Vec::new(),
            start: None,
            elements: Vec::new(),
            code: Vec::new(),
            data: Vec::new(),
            custom_sections: HashMap::new(),
        }
    }

    // ── Type section helpers ───────────────────────────────────────────

    /// Add a function type and return its index.
    pub fn add_type(&mut self, ft: FuncType) -> u32 {
        // Deduplicate: return existing index if an identical signature exists.
        for (i, existing) in self.types.iter().enumerate() {
            if existing == &ft {
                return i as u32;
            }
        }
        let idx = self.types.len() as u32;
        self.types.push(ft);
        idx
    }

    // ── Import helpers ─────────────────────────────────────────────────

    /// Add an import and return its index within the import list.
    pub fn add_import(&mut self, imp: Import) -> u32 {
        let idx = self.imports.len() as u32;
        self.imports.push(imp);
        idx
    }

    /// Count imports that are functions.
    pub fn imported_func_count(&self) -> u32 {
        self.imports
            .iter()
            .filter(|i| matches!(i.kind, ExternKind::Func(_)))
            .count() as u32
    }

    /// Count imports that are memories.
    pub fn imported_memory_count(&self) -> u32 {
        self.imports
            .iter()
            .filter(|i| matches!(i.kind, ExternKind::Memory(_)))
            .count() as u32
    }

    // ── Function helpers ───────────────────────────────────────────────

    /// Add a function (type index + body). Returns the function index
    /// (which starts after imported functions).
    pub fn add_function(&mut self, type_idx: u32, body: FuncBody) -> Result<u32, ModuleError> {
        if type_idx as usize >= self.types.len() {
            return Err(ModuleError::Validation(format!(
                "type index {type_idx} out of range (have {} types)",
                self.types.len()
            )));
        }
        self.func_type_indices.push(type_idx);
        self.code.push(body);
        Ok(self.imported_func_count() + self.func_type_indices.len() as u32 - 1)
    }

    /// Total function count (imported + defined).
    pub fn total_func_count(&self) -> u32 {
        self.imported_func_count() + self.func_type_indices.len() as u32
    }

    // ── Memory helpers ─────────────────────────────────────────────────

    /// Add a memory section entry.
    pub fn add_memory(&mut self, mem: MemoryType) -> Result<u32, ModuleError> {
        mem.limits.validate()?;
        let idx = self.imported_memory_count() + self.memories.len() as u32;
        self.memories.push(mem);
        Ok(idx)
    }

    /// Total memory count (imported + defined).
    pub fn total_memory_count(&self) -> u32 {
        self.imported_memory_count() + self.memories.len() as u32
    }

    // ── Table helpers ──────────────────────────────────────────────────

    /// Add a table section entry.
    pub fn add_table(&mut self, tt: TableType) -> Result<u32, ModuleError> {
        tt.limits.validate()?;
        let idx = self.tables.len() as u32;
        self.tables.push(tt);
        Ok(idx)
    }

    // ── Export helpers ──────────────────────────────────────────────────

    /// Add an export. Rejects duplicates.
    pub fn add_export(&mut self, exp: Export) -> Result<(), ModuleError> {
        if self.exports.iter().any(|e| e.name == exp.name) {
            return Err(ModuleError::DuplicateExport(exp.name));
        }
        self.exports.push(exp);
        Ok(())
    }

    /// Look up an export by name.
    pub fn find_export(&self, name: &str) -> Result<&Export, ModuleError> {
        self.exports
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| ModuleError::ExportNotFound(name.to_string()))
    }

    /// Exported function names.
    pub fn exported_function_names(&self) -> Vec<&str> {
        self.exports
            .iter()
            .filter(|e| matches!(e.kind, ExternKind::Func(_)))
            .map(|e| e.name.as_str())
            .collect()
    }

    // ── Data segment helpers ───────────────────────────────────────────

    /// Add a data segment.
    pub fn add_data(&mut self, seg: DataSegment) -> u32 {
        let idx = self.data.len() as u32;
        self.data.push(seg);
        idx
    }

    // ── Global helpers ─────────────────────────────────────────────────

    /// Add a global.
    pub fn add_global(&mut self, gt: GlobalType) -> u32 {
        let idx = self.globals.len() as u32;
        self.globals.push(gt);
        idx
    }

    // ── Element helpers ────────────────────────────────────────────────

    /// Add an element segment.
    pub fn add_element(&mut self, seg: ElemSegment) -> u32 {
        let idx = self.elements.len() as u32;
        self.elements.push(seg);
        idx
    }

    // ── Validation ─────────────────────────────────────────────────────

    /// Validate the module for internal consistency.
    pub fn validate(&self) -> Result<(), Vec<ModuleError>> {
        let mut errors = Vec::new();

        // 1) func_type_indices must reference valid types
        for (i, &ti) in self.func_type_indices.iter().enumerate() {
            if ti as usize >= self.types.len() {
                errors.push(ModuleError::Validation(format!(
                    "function {i} references type index {ti} but only {} types exist",
                    self.types.len()
                )));
            }
        }

        // 2) code bodies must match func count
        if self.code.len() != self.func_type_indices.len() {
            errors.push(ModuleError::Validation(format!(
                "code section has {} entries but function section has {}",
                self.code.len(),
                self.func_type_indices.len()
            )));
        }

        // 3) export names must be unique
        let mut seen_exports = std::collections::HashSet::new();
        for exp in &self.exports {
            if !seen_exports.insert(&exp.name) {
                errors.push(ModuleError::DuplicateExport(exp.name.clone()));
            }
        }

        // 4) export references must be in range
        let total_funcs = self.total_func_count();
        for exp in &self.exports {
            match &exp.kind {
                ExternKind::Func(idx) if *idx >= total_funcs => {
                    errors.push(ModuleError::Validation(format!(
                        "export '{}' references func {idx} but only {total_funcs} exist",
                        exp.name
                    )));
                }
                _ => {}
            }
        }

        // 5) start function must be in range
        if let Some(s) = self.start {
            if s >= total_funcs {
                errors.push(ModuleError::Validation(format!(
                    "start function {s} out of range (total funcs: {total_funcs})"
                )));
            }
        }

        // 6) memory limits
        for (i, mem) in self.memories.iter().enumerate() {
            if let Err(e) = mem.limits.validate() {
                errors.push(ModuleError::Validation(format!(
                    "memory {i}: {e}"
                )));
            }
        }

        // 7) data segments must reference valid memories
        let total_mems = self.total_memory_count();
        for (i, seg) in self.data.iter().enumerate() {
            if seg.memory_index >= total_mems {
                errors.push(ModuleError::Validation(format!(
                    "data segment {i} references memory {}, but only {total_mems} exist",
                    seg.memory_index
                )));
            }
        }

        // 8) element segment func indices in range
        for (i, elem) in self.elements.iter().enumerate() {
            for &fi in &elem.func_indices {
                if fi >= total_funcs {
                    errors.push(ModuleError::Validation(format!(
                        "element segment {i} references func {fi} but only {total_funcs} exist",
                    )));
                }
            }
        }

        // 9) imports with func kind must reference valid types
        for imp in &self.imports {
            if let ExternKind::Func(ti) = &imp.kind {
                if *ti as usize >= self.types.len() {
                    errors.push(ModuleError::Validation(format!(
                        "import {}.{} references type {ti} but only {} types exist",
                        imp.module,
                        imp.name,
                        self.types.len()
                    )));
                }
            }
        }

        // 10) at most one memory (MVP restriction)
        if self.total_memory_count() > 1 {
            errors.push(ModuleError::Validation(
                "MVP allows at most one memory".to_string(),
            ));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    // ── Resolution helpers ─────────────────────────────────────────────

    /// Resolve an import against a set of provided externals.
    /// Returns the matching provided kind, or an error.
    pub fn resolve_import<'a>(
        &self,
        imp: &Import,
        provided: &'a HashMap<(String, String), ExternKind>,
    ) -> Result<&'a ExternKind, ModuleError> {
        let key = (imp.module.clone(), imp.name.clone());
        provided.get(&key).ok_or_else(|| ModuleError::ImportNotFound {
            module: imp.module.clone(),
            name: imp.name.clone(),
        })
    }

    /// Check that all imports can be resolved.
    pub fn resolve_all_imports(
        &self,
        provided: &HashMap<(String, String), ExternKind>,
    ) -> Result<(), Vec<ModuleError>> {
        let mut errors = Vec::new();
        for imp in &self.imports {
            if let Err(e) = self.resolve_import(imp, provided) {
                errors.push(e);
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    // ── Summary ────────────────────────────────────────────────────────

    /// Summary of the module for debugging.
    pub fn summary(&self) -> ModuleSummary {
        ModuleSummary {
            type_count: self.types.len(),
            import_count: self.imports.len(),
            func_count: self.func_type_indices.len(),
            table_count: self.tables.len(),
            memory_count: self.memories.len(),
            global_count: self.globals.len(),
            export_count: self.exports.len(),
            data_segment_count: self.data.len(),
            element_segment_count: self.elements.len(),
            has_start: self.start.is_some(),
            custom_section_count: self.custom_sections.len(),
        }
    }
}

impl Default for WasmModule {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level module summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleSummary {
    pub type_count: usize,
    pub import_count: usize,
    pub func_count: usize,
    pub table_count: usize,
    pub memory_count: usize,
    pub global_count: usize,
    pub export_count: usize,
    pub data_segment_count: usize,
    pub element_segment_count: usize,
    pub has_start: bool,
    pub custom_section_count: usize,
}

// ── Section ordering validator ─────────────────────────────────────────────

/// Validate that a sequence of section IDs is in ascending order
/// (custom sections may appear anywhere, so they are skipped).
pub fn validate_section_order(ids: &[SectionId]) -> Result<(), ModuleError> {
    let mut last_non_custom: Option<u8> = None;
    for id in ids {
        if *id == SectionId::Custom {
            continue;
        }
        let order = id.order();
        if let Some(prev) = last_non_custom {
            if order <= prev {
                return Err(ModuleError::Validation(format!(
                    "section {:?} (order {order}) must come after section with order {prev}",
                    id
                )));
            }
        }
        last_non_custom = Some(order);
    }
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn i32_to_i32() -> FuncType {
        FuncType::new(vec![ValType::I32], vec![ValType::I32])
    }

    fn void_sig() -> FuncType {
        FuncType::new(vec![], vec![])
    }

    fn make_body(type_index: u32) -> FuncBody {
        FuncBody {
            locals: vec![],
            code_size: 4,
            type_index,
        }
    }

    #[test]
    fn func_type_display() {
        let ft = FuncType::new(vec![ValType::I32, ValType::I64], vec![ValType::F64]);
        assert_eq!(ft.to_string(), "(i32, i64) -> (f64)");
    }

    #[test]
    fn func_type_arity_and_results() {
        let ft = FuncType::new(vec![ValType::I32, ValType::I32], vec![ValType::I64]);
        assert_eq!(ft.arity(), 2);
        assert_eq!(ft.result_count(), 1);
    }

    #[test]
    fn func_type_matches() {
        let a = i32_to_i32();
        let b = i32_to_i32();
        let c = void_sig();
        assert!(a.matches(&b));
        assert!(!a.matches(&c));
    }

    #[test]
    fn limits_validate_ok() {
        assert!(Limits::new(1, Some(10)).validate().is_ok());
        assert!(Limits::new(0, None).validate().is_ok());
    }

    #[test]
    fn limits_validate_fail() {
        assert!(Limits::new(10, Some(5)).validate().is_err());
    }

    #[test]
    fn limits_contains() {
        let l = Limits::new(2, Some(8));
        assert!(!l.contains(1));
        assert!(l.contains(2));
        assert!(l.contains(5));
        assert!(l.contains(8));
        assert!(!l.contains(9));
    }

    #[test]
    fn memory_type_bytes() {
        let mt = MemoryType::new(1, Some(4));
        assert_eq!(mt.min_bytes(), 65536);
        assert_eq!(mt.max_bytes(), Some(4 * 65536));
    }

    #[test]
    fn section_id_round_trip() {
        for b in 0..=12u8 {
            let sid = SectionId::from_byte(b).unwrap();
            assert_eq!(sid.order(), b);
        }
        assert!(SectionId::from_byte(13).is_none());
    }

    #[test]
    fn section_order_valid() {
        let ids = vec![
            SectionId::Type,
            SectionId::Custom,
            SectionId::Import,
            SectionId::Function,
        ];
        assert!(validate_section_order(&ids).is_ok());
    }

    #[test]
    fn section_order_invalid() {
        let ids = vec![SectionId::Export, SectionId::Import];
        assert!(validate_section_order(&ids).is_err());
    }

    #[test]
    fn add_type_deduplicates() {
        let mut m = WasmModule::new();
        let idx1 = m.add_type(i32_to_i32());
        let idx2 = m.add_type(i32_to_i32());
        assert_eq!(idx1, idx2);
        assert_eq!(m.types.len(), 1);
    }

    #[test]
    fn add_function_validates_type() {
        let mut m = WasmModule::new();
        // No types yet, so type_idx=0 is out of range.
        let res = m.add_function(0, make_body(0));
        assert!(res.is_err());
    }

    #[test]
    fn add_function_and_export() {
        let mut m = WasmModule::new();
        let ti = m.add_type(i32_to_i32());
        let fi = m.add_function(ti, make_body(ti)).unwrap();
        m.add_export(Export {
            name: "inc".to_string(),
            kind: ExternKind::Func(fi),
        })
        .unwrap();
        assert_eq!(m.total_func_count(), 1);
        assert_eq!(m.exported_function_names(), vec!["inc"]);
    }

    #[test]
    fn duplicate_export_rejected() {
        let mut m = WasmModule::new();
        let ti = m.add_type(void_sig());
        let fi = m.add_function(ti, make_body(ti)).unwrap();
        m.add_export(Export {
            name: "f".to_string(),
            kind: ExternKind::Func(fi),
        })
        .unwrap();
        let res = m.add_export(Export {
            name: "f".to_string(),
            kind: ExternKind::Func(fi),
        });
        assert!(matches!(res, Err(ModuleError::DuplicateExport(_))));
    }

    #[test]
    fn find_export() {
        let mut m = WasmModule::new();
        let ti = m.add_type(void_sig());
        let fi = m.add_function(ti, make_body(ti)).unwrap();
        m.add_export(Export {
            name: "main".to_string(),
            kind: ExternKind::Func(fi),
        })
        .unwrap();
        assert!(m.find_export("main").is_ok());
        assert!(m.find_export("nope").is_err());
    }

    #[test]
    fn import_resolution() {
        let mut m = WasmModule::new();
        let ti = m.add_type(void_sig());
        m.add_import(Import {
            module: "env".to_string(),
            name: "log".to_string(),
            kind: ExternKind::Func(ti),
        });
        let mut provided = HashMap::new();
        provided.insert(
            ("env".to_string(), "log".to_string()),
            ExternKind::Func(ti),
        );
        assert!(m.resolve_all_imports(&provided).is_ok());
    }

    #[test]
    fn import_resolution_missing() {
        let mut m = WasmModule::new();
        let ti = m.add_type(void_sig());
        m.add_import(Import {
            module: "env".to_string(),
            name: "missing".to_string(),
            kind: ExternKind::Func(ti),
        });
        let provided = HashMap::new();
        assert!(m.resolve_all_imports(&provided).is_err());
    }

    #[test]
    fn data_segment_end_offset() {
        let seg = DataSegment::new(0, 100, vec![1, 2, 3, 4, 5]);
        assert_eq!(seg.end_offset(), 105);
    }

    #[test]
    fn validate_valid_module() {
        let mut m = WasmModule::new();
        let ti = m.add_type(void_sig());
        m.add_function(ti, make_body(ti)).unwrap();
        m.add_memory(MemoryType::new(1, Some(10))).unwrap();
        assert!(m.validate().is_ok());
    }

    #[test]
    fn validate_bad_start() {
        let mut m = WasmModule::new();
        m.start = Some(99);
        let errs = m.validate().unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, ModuleError::Validation(msg) if msg.contains("start"))));
    }

    #[test]
    fn validate_data_seg_bad_mem() {
        let mut m = WasmModule::new();
        m.data.push(DataSegment::new(5, 0, vec![0]));
        let errs = m.validate().unwrap_err();
        assert!(!errs.is_empty());
    }

    #[test]
    fn module_summary() {
        let mut m = WasmModule::new();
        let ti = m.add_type(void_sig());
        m.add_function(ti, make_body(ti)).unwrap();
        m.add_memory(MemoryType::new(1, None)).unwrap();
        let s = m.summary();
        assert_eq!(s.type_count, 1);
        assert_eq!(s.func_count, 1);
        assert_eq!(s.memory_count, 1);
        assert!(!s.has_start);
    }

    #[test]
    fn table_type_validation() {
        let mut m = WasmModule::new();
        let ok = m.add_table(TableType::new(ValType::FuncRef, 0, Some(100)));
        assert!(ok.is_ok());
        let bad = m.add_table(TableType::new(ValType::FuncRef, 10, Some(5)));
        assert!(bad.is_err());
    }

    #[test]
    fn global_and_element() {
        let mut m = WasmModule::new();
        let gi = m.add_global(GlobalType::new(ValType::I32, Mutability::Var));
        assert_eq!(gi, 0);
        let ti = m.add_type(void_sig());
        let fi = m.add_function(ti, make_body(ti)).unwrap();
        let ei = m.add_element(ElemSegment {
            table_index: 0,
            offset: 0,
            func_indices: vec![fi],
        });
        assert_eq!(ei, 0);
    }

    #[test]
    fn imported_func_counted() {
        let mut m = WasmModule::new();
        let ti = m.add_type(void_sig());
        m.add_import(Import {
            module: "env".to_string(),
            name: "f1".to_string(),
            kind: ExternKind::Func(ti),
        });
        m.add_import(Import {
            module: "env".to_string(),
            name: "f2".to_string(),
            kind: ExternKind::Func(ti),
        });
        assert_eq!(m.imported_func_count(), 2);
        let fi = m.add_function(ti, make_body(ti)).unwrap();
        // Function index accounts for imported funcs.
        assert_eq!(fi, 2);
        assert_eq!(m.total_func_count(), 3);
    }

    #[test]
    fn custom_section_storage() {
        let mut m = WasmModule::new();
        m.custom_sections
            .insert("name".to_string(), b"hello".to_vec());
        assert_eq!(m.custom_sections.len(), 1);
        assert_eq!(m.summary().custom_section_count, 1);
    }

    #[test]
    fn module_error_display() {
        let e = ModuleError::ImportNotFound {
            module: "env".to_string(),
            name: "log".to_string(),
        };
        assert!(e.to_string().contains("env.log"));
    }
}
