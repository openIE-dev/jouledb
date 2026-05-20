//! Linker simulation — object file model (sections + symbols + relocations),
//! symbol resolution, section merging, relocation application, undefined symbol
//! detection, duplicate symbol handling, link map output.

use std::collections::HashMap;
use std::fmt;

// ── Section model ───────────────────────────────────────────────────────────

/// Section flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SectionFlags {
    /// Section is writable.
    pub writable: bool,
    /// Section is executable.
    pub executable: bool,
    /// Section is loaded into memory.
    pub allocatable: bool,
}

impl SectionFlags {
    /// Code section flags.
    pub fn code() -> Self {
        Self { writable: false, executable: true, allocatable: true }
    }

    /// Data section flags.
    pub fn data() -> Self {
        Self { writable: true, executable: false, allocatable: true }
    }

    /// Read-only data.
    pub fn rodata() -> Self {
        Self { writable: false, executable: false, allocatable: true }
    }

    /// BSS (uninitialized).
    pub fn bss() -> Self {
        Self { writable: true, executable: false, allocatable: true }
    }
}

/// A section within an object file.
#[derive(Debug, Clone)]
pub struct Section {
    /// Section name (e.g. ".text", ".data").
    pub name: String,
    /// Section data bytes.
    pub data: Vec<u8>,
    /// Flags.
    pub flags: SectionFlags,
    /// Alignment (power of 2).
    pub alignment: u32,
}

impl Section {
    /// Create a new section.
    pub fn new(name: &str, flags: SectionFlags) -> Self {
        Self {
            name: name.to_string(),
            data: Vec::new(),
            flags,
            alignment: 4,
        }
    }

    /// Create a section with data.
    pub fn with_data(name: &str, data: Vec<u8>, flags: SectionFlags) -> Self {
        Self {
            name: name.to_string(),
            data,
            flags,
            alignment: 4,
        }
    }

    /// Size in bytes.
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

// ── Symbol model ────────────────────────────────────────────────────────────

/// Symbol binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolBinding {
    /// Local: only visible within the object file.
    Local,
    /// Global: visible to all object files.
    Global,
    /// Weak: global but can be overridden.
    Weak,
}

impl fmt::Display for SymbolBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Local => write!(f, "LOCAL"),
            Self::Global => write!(f, "GLOBAL"),
            Self::Weak => write!(f, "WEAK"),
        }
    }
}

/// Symbol type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolType {
    /// Function.
    Function,
    /// Data object.
    Object,
    /// No type.
    NoType,
}

/// A symbol in an object file.
#[derive(Debug, Clone)]
pub struct Symbol {
    /// Symbol name.
    pub name: String,
    /// Section name this symbol is defined in (None = undefined).
    pub section: Option<String>,
    /// Offset within the section.
    pub offset: u64,
    /// Size of the symbol.
    pub size: u64,
    /// Binding.
    pub binding: SymbolBinding,
    /// Type.
    pub sym_type: SymbolType,
}

impl Symbol {
    /// Create a defined global symbol.
    pub fn global(name: &str, section: &str, offset: u64) -> Self {
        Self {
            name: name.to_string(),
            section: Some(section.to_string()),
            offset,
            size: 0,
            binding: SymbolBinding::Global,
            sym_type: SymbolType::NoType,
        }
    }

    /// Create a defined function symbol.
    pub fn function(name: &str, section: &str, offset: u64, size: u64) -> Self {
        Self {
            name: name.to_string(),
            section: Some(section.to_string()),
            offset,
            size,
            binding: SymbolBinding::Global,
            sym_type: SymbolType::Function,
        }
    }

    /// Create an undefined symbol (external reference).
    pub fn undefined(name: &str) -> Self {
        Self {
            name: name.to_string(),
            section: None,
            offset: 0,
            size: 0,
            binding: SymbolBinding::Global,
            sym_type: SymbolType::NoType,
        }
    }

    /// Create a weak symbol.
    pub fn weak(name: &str, section: &str, offset: u64) -> Self {
        Self {
            name: name.to_string(),
            section: Some(section.to_string()),
            offset,
            size: 0,
            binding: SymbolBinding::Weak,
            sym_type: SymbolType::NoType,
        }
    }

    /// Whether the symbol is defined.
    pub fn is_defined(&self) -> bool {
        self.section.is_some()
    }

    /// Whether the symbol is undefined.
    pub fn is_undefined(&self) -> bool {
        self.section.is_none()
    }
}

// ── Relocation model ────────────────────────────────────────────────────────

/// Relocation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocType {
    /// Absolute 32-bit address.
    Abs32,
    /// Absolute 64-bit address.
    Abs64,
    /// PC-relative 32-bit.
    Rel32,
}

impl fmt::Display for RelocType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Abs32 => write!(f, "R_ABS32"),
            Self::Abs64 => write!(f, "R_ABS64"),
            Self::Rel32 => write!(f, "R_REL32"),
        }
    }
}

/// A relocation entry.
#[derive(Debug, Clone)]
pub struct Relocation {
    /// Section this relocation applies to.
    pub section: String,
    /// Offset within the section.
    pub offset: u64,
    /// Relocation type.
    pub reloc_type: RelocType,
    /// Target symbol name.
    pub symbol: String,
    /// Addend.
    pub addend: i64,
}

impl Relocation {
    /// Create a new relocation.
    pub fn new(section: &str, offset: u64, reloc_type: RelocType, symbol: &str) -> Self {
        Self {
            section: section.to_string(),
            offset,
            reloc_type,
            symbol: symbol.to_string(),
            addend: 0,
        }
    }

    /// Set the addend.
    pub fn with_addend(mut self, addend: i64) -> Self {
        self.addend = addend;
        self
    }
}

// ── Object file ─────────────────────────────────────────────────────────────

/// An object file containing sections, symbols, and relocations.
#[derive(Debug, Clone)]
pub struct ObjectFile {
    /// File name.
    pub name: String,
    /// Sections.
    pub sections: Vec<Section>,
    /// Symbols.
    pub symbols: Vec<Symbol>,
    /// Relocations.
    pub relocations: Vec<Relocation>,
}

impl ObjectFile {
    /// Create a new object file.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            sections: Vec::new(),
            symbols: Vec::new(),
            relocations: Vec::new(),
        }
    }

    /// Add a section.
    pub fn add_section(&mut self, section: Section) {
        self.sections.push(section);
    }

    /// Add a symbol.
    pub fn add_symbol(&mut self, symbol: Symbol) {
        self.symbols.push(symbol);
    }

    /// Add a relocation.
    pub fn add_relocation(&mut self, reloc: Relocation) {
        self.relocations.push(reloc);
    }

    /// Find a section by name.
    pub fn find_section(&self, name: &str) -> Option<&Section> {
        self.sections.iter().find(|s| s.name == name)
    }

    /// Find a symbol by name.
    pub fn find_symbol(&self, name: &str) -> Option<&Symbol> {
        self.symbols.iter().find(|s| s.name == name)
    }
}

// ── Linker errors ───────────────────────────────────────────────────────────

/// Linker error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkerError {
    /// Undefined symbol — no definition found.
    UndefinedSymbol(String),
    /// Multiple definitions of a strong symbol.
    DuplicateSymbol {
        name: String,
        first_file: String,
        second_file: String,
    },
    /// Relocation target not found.
    RelocationTargetNotFound {
        symbol: String,
        in_file: String,
    },
    /// Section not found during relocation.
    SectionNotFound {
        section: String,
        in_file: String,
    },
    /// Relocation out of range.
    RelocationOutOfRange {
        symbol: String,
        reloc_type: String,
    },
}

impl fmt::Display for LinkerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UndefinedSymbol(name) => {
                write!(f, "undefined reference to '{name}'")
            }
            Self::DuplicateSymbol { name, first_file, second_file } => {
                write!(f, "multiple definition of '{name}': first defined in {first_file}, also in {second_file}")
            }
            Self::RelocationTargetNotFound { symbol, in_file } => {
                write!(f, "relocation target '{symbol}' not found (in {in_file})")
            }
            Self::SectionNotFound { section, in_file } => {
                write!(f, "section '{section}' not found in {in_file}")
            }
            Self::RelocationOutOfRange { symbol, reloc_type } => {
                write!(f, "relocation {reloc_type} for '{symbol}' out of range")
            }
        }
    }
}

// ── Resolved symbol ─────────────────────────────────────────────────────────

/// A resolved symbol with its final address.
#[derive(Debug, Clone)]
struct ResolvedSymbol {
    /// Final virtual address.
    address: u64,
    /// Source object file name.
    source_file: String,
    /// Binding.
    binding: SymbolBinding,
    /// Size.
    size: u64,
    /// Type.
    sym_type: SymbolType,
}

// ── Link map entry ──────────────────────────────────────────────────────────

/// An entry in the link map.
#[derive(Debug, Clone)]
pub struct LinkMapEntry {
    /// Symbol name.
    pub name: String,
    /// Final virtual address.
    pub address: u64,
    /// Size.
    pub size: u64,
    /// Source file.
    pub source: String,
    /// Section name.
    pub section: String,
}

// ── Merged section tracking ─────────────────────────────────────────────────

/// Tracks how sections from different object files are merged.
#[derive(Debug, Clone)]
struct MergedSection {
    /// Section name.
    name: String,
    /// Final data.
    data: Vec<u8>,
    /// Flags.
    flags: SectionFlags,
    /// Alignment.
    alignment: u32,
    /// Base virtual address (assigned during layout).
    base_address: u64,
    /// Contributions: (object_file_name, original_offset_in_merged, original_size).
    contributions: Vec<(String, u64, usize)>,
}

// ── Linker ──────────────────────────────────────────────────────────────────

/// The linker: merges object files into a single output.
pub struct Linker {
    /// Input object files.
    objects: Vec<ObjectFile>,
    /// Base address for layout.
    base_address: u64,
    /// Merged sections.
    merged_sections: Vec<MergedSection>,
    /// Global symbol table: name -> resolved symbol.
    global_symbols: HashMap<String, ResolvedSymbol>,
    /// Link map entries.
    link_map: Vec<LinkMapEntry>,
    /// Errors encountered.
    errors: Vec<LinkerError>,
    /// Warnings.
    warnings: Vec<String>,
}

impl Linker {
    /// Create a new linker with the given base address.
    pub fn new(base_address: u64) -> Self {
        Self {
            objects: Vec::new(),
            base_address,
            merged_sections: Vec::new(),
            global_symbols: HashMap::new(),
            link_map: Vec::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Add an object file to the link.
    pub fn add_object(&mut self, obj: ObjectFile) {
        self.objects.push(obj);
    }

    /// Run the full link process.
    pub fn link(&mut self) -> Result<Vec<u8>, Vec<LinkerError>> {
        self.errors.clear();
        self.warnings.clear();
        self.merged_sections.clear();
        self.global_symbols.clear();
        self.link_map.clear();

        // Phase 1: Merge sections
        self.merge_sections();

        // Phase 2: Resolve symbols
        self.resolve_symbols();

        if !self.errors.is_empty() {
            return Err(self.errors.clone());
        }

        // Phase 3: Layout
        self.layout();

        // Phase 4: Update symbol addresses after layout
        self.update_symbol_addresses();

        // Phase 5: Apply relocations
        self.apply_relocations();

        if !self.errors.is_empty() {
            return Err(self.errors.clone());
        }

        // Phase 6: Check for undefined symbols
        self.check_undefined();

        if !self.errors.is_empty() {
            return Err(self.errors.clone());
        }

        // Phase 7: Build link map
        self.build_link_map();

        // Produce output
        Ok(self.flatten_output())
    }

    /// Phase 1: Merge sections with the same name across object files.
    fn merge_sections(&mut self) {
        let mut section_order: Vec<String> = Vec::new();
        let mut section_map: HashMap<String, usize> = HashMap::new();

        for obj in &self.objects {
            for sec in &obj.sections {
                let idx = if let Some(&existing) = section_map.get(&sec.name) {
                    existing
                } else {
                    let idx = self.merged_sections.len();
                    section_map.insert(sec.name.clone(), idx);
                    section_order.push(sec.name.clone());
                    self.merged_sections.push(MergedSection {
                        name: sec.name.clone(),
                        data: Vec::new(),
                        flags: sec.flags,
                        alignment: sec.alignment,
                        base_address: 0,
                        contributions: Vec::new(),
                    });
                    idx
                };

                let merged = &mut self.merged_sections[idx];

                // Align before appending
                let align = sec.alignment.max(1) as usize;
                let remainder = merged.data.len() % align;
                if remainder != 0 {
                    let padding = align - remainder;
                    merged.data.extend(std::iter::repeat(0u8).take(padding));
                }

                let offset = merged.data.len() as u64;
                merged.data.extend_from_slice(&sec.data);
                merged.contributions.push((obj.name.clone(), offset, sec.data.len()));

                // Update alignment to max
                if sec.alignment > merged.alignment {
                    merged.alignment = sec.alignment;
                }
            }
        }
    }

    /// Phase 2: Resolve global symbols.
    fn resolve_symbols(&mut self) {
        for obj in &self.objects {
            for sym in &obj.symbols {
                if sym.binding == SymbolBinding::Local {
                    continue;
                }

                if sym.is_undefined() {
                    // Just register as needed; will check later
                    self.global_symbols.entry(sym.name.clone()).or_insert(ResolvedSymbol {
                        address: 0,
                        source_file: obj.name.clone(),
                        binding: sym.binding,
                        size: sym.size,
                        sym_type: sym.sym_type,
                    });
                    continue;
                }

                if let Some(existing) = self.global_symbols.get(&sym.name) {
                    match (existing.binding, sym.binding) {
                        // Two strong definitions = error
                        (SymbolBinding::Global, SymbolBinding::Global) => {
                            if existing.address != 0 || !existing.source_file.is_empty() {
                                self.errors.push(LinkerError::DuplicateSymbol {
                                    name: sym.name.clone(),
                                    first_file: existing.source_file.clone(),
                                    second_file: obj.name.clone(),
                                });
                                continue;
                            }
                        }
                        // Strong overrides weak
                        (SymbolBinding::Weak, SymbolBinding::Global) => {
                            // Will override below
                        }
                        // Weak doesn't override strong
                        (SymbolBinding::Global, SymbolBinding::Weak) => continue,
                        // Two weak: keep the first
                        (SymbolBinding::Weak, SymbolBinding::Weak) => continue,
                        _ => {}
                    }
                }

                self.global_symbols.insert(sym.name.clone(), ResolvedSymbol {
                    address: 0, // Will be filled during layout
                    source_file: obj.name.clone(),
                    binding: sym.binding,
                    size: sym.size,
                    sym_type: sym.sym_type,
                });
            }
        }
    }

    /// Phase 3: Assign virtual addresses to merged sections.
    fn layout(&mut self) {
        let mut addr = self.base_address;
        for sec in &mut self.merged_sections {
            let align = sec.alignment.max(1) as u64;
            let remainder = addr % align;
            if remainder != 0 {
                addr += align - remainder;
            }
            sec.base_address = addr;
            addr += sec.data.len() as u64;
        }
    }

    /// Phase 4: Update resolved symbol addresses based on layout.
    fn update_symbol_addresses(&mut self) {
        // Build a map of (file_name, section_name) -> offset in merged section
        let mut contrib_map: HashMap<(String, String), u64> = HashMap::new();
        for sec in &self.merged_sections {
            for (file_name, offset, _size) in &sec.contributions {
                contrib_map.insert(
                    (file_name.clone(), sec.name.clone()),
                    sec.base_address + offset,
                );
            }
        }

        for obj in &self.objects {
            for sym in &obj.symbols {
                if sym.is_undefined() || sym.binding == SymbolBinding::Local {
                    continue;
                }

                if let Some(section_name) = &sym.section {
                    let key = (obj.name.clone(), section_name.clone());
                    if let Some(&section_base) = contrib_map.get(&key) {
                        let addr = section_base + sym.offset;
                        if let Some(resolved) = self.global_symbols.get_mut(&sym.name) {
                            if resolved.source_file == obj.name {
                                resolved.address = addr;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Phase 5: Apply relocations.
    fn apply_relocations(&mut self) {
        // Build a map of (file_name, section_name) -> (merged_section_index, offset)
        let mut contrib_map: HashMap<(String, String), (usize, u64)> = HashMap::new();
        for (sec_idx, sec) in self.merged_sections.iter().enumerate() {
            for (file_name, offset, _size) in &sec.contributions {
                contrib_map.insert(
                    (file_name.clone(), sec.name.clone()),
                    (sec_idx, *offset),
                );
            }
        }

        let objects = self.objects.clone();
        for obj in &objects {
            for reloc in &obj.relocations {
                // Find the merged section and offset
                let key = (obj.name.clone(), reloc.section.clone());
                let (sec_idx, contrib_offset) = match contrib_map.get(&key) {
                    Some(v) => *v,
                    None => {
                        self.errors.push(LinkerError::SectionNotFound {
                            section: reloc.section.clone(),
                            in_file: obj.name.clone(),
                        });
                        continue;
                    }
                };

                // Find the target symbol address
                let target_addr = match self.global_symbols.get(&reloc.symbol) {
                    Some(resolved) if resolved.address != 0 => resolved.address,
                    _ => {
                        self.errors.push(LinkerError::RelocationTargetNotFound {
                            symbol: reloc.symbol.clone(),
                            in_file: obj.name.clone(),
                        });
                        continue;
                    }
                };

                // Apply the relocation
                let patch_offset = (contrib_offset + reloc.offset) as usize;
                let sec = &mut self.merged_sections[sec_idx];
                let reloc_addr = sec.base_address + contrib_offset + reloc.offset;

                match reloc.reloc_type {
                    RelocType::Abs32 => {
                        let val = (target_addr as i64 + reloc.addend) as u32;
                        if patch_offset + 4 <= sec.data.len() {
                            sec.data[patch_offset..patch_offset + 4]
                                .copy_from_slice(&val.to_le_bytes());
                        }
                    }
                    RelocType::Abs64 => {
                        let val = (target_addr as i64 + reloc.addend) as u64;
                        if patch_offset + 8 <= sec.data.len() {
                            sec.data[patch_offset..patch_offset + 8]
                                .copy_from_slice(&val.to_le_bytes());
                        }
                    }
                    RelocType::Rel32 => {
                        let rel = target_addr as i64 - reloc_addr as i64 + reloc.addend;
                        let val = rel as i32;
                        if patch_offset + 4 <= sec.data.len() {
                            sec.data[patch_offset..patch_offset + 4]
                                .copy_from_slice(&val.to_le_bytes());
                        }
                    }
                }
            }
        }
    }

    /// Phase 6: Check for undefined symbols that were never resolved.
    fn check_undefined(&mut self) {
        // Collect referenced symbols from relocations
        let mut referenced: std::collections::HashSet<String> = std::collections::HashSet::new();
        for obj in &self.objects {
            for reloc in &obj.relocations {
                referenced.insert(reloc.symbol.clone());
            }
        }

        for sym_name in &referenced {
            match self.global_symbols.get(sym_name) {
                Some(resolved) if resolved.address == 0 => {
                    self.errors.push(LinkerError::UndefinedSymbol(sym_name.clone()));
                }
                None => {
                    self.errors.push(LinkerError::UndefinedSymbol(sym_name.clone()));
                }
                _ => {}
            }
        }
    }

    /// Phase 7: Build the link map.
    fn build_link_map(&mut self) {
        let mut entries: Vec<LinkMapEntry> = Vec::new();
        for (name, resolved) in &self.global_symbols {
            if resolved.address == 0 {
                continue;
            }
            // Find which section this belongs to
            let section = self.merged_sections.iter()
                .find(|s| {
                    resolved.address >= s.base_address
                        && resolved.address < s.base_address + s.data.len() as u64
                })
                .map_or_else(String::new, |s| s.name.clone());

            entries.push(LinkMapEntry {
                name: name.clone(),
                address: resolved.address,
                size: resolved.size,
                source: resolved.source_file.clone(),
                section,
            });
        }
        entries.sort_by_key(|e| e.address);
        self.link_map = entries;
    }

    /// Flatten all merged sections into the output bytes.
    fn flatten_output(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for sec in &self.merged_sections {
            out.extend_from_slice(&sec.data);
        }
        out
    }

    /// Get the link map.
    pub fn link_map(&self) -> &[LinkMapEntry] {
        &self.link_map
    }

    /// Get warnings.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Produce a human-readable link map string.
    pub fn link_map_string(&self) -> String {
        let mut out = String::new();
        out.push_str("Link Map\n");
        out.push_str(&"=".repeat(70));
        out.push('\n');

        // Sections
        out.push_str("\nSections:\n");
        for sec in &self.merged_sections {
            out.push_str(&format!(
                "  {:<16} base=0x{:08X}  size={:<8}  align={}\n",
                sec.name,
                sec.base_address,
                sec.data.len(),
                sec.alignment,
            ));
        }

        // Symbols
        out.push_str("\nSymbols:\n");
        for entry in &self.link_map {
            out.push_str(&format!(
                "  0x{:08X}  {:<20}  size={:<6}  from {}\n",
                entry.address, entry.name, entry.size, entry.source,
            ));
        }

        out
    }

    /// Get errors from the last link attempt.
    pub fn errors(&self) -> &[LinkerError] {
        &self.errors
    }

    /// Look up a symbol's resolved address.
    pub fn symbol_address(&self, name: &str) -> Option<u64> {
        self.global_symbols.get(name).map(|s| s.address)
    }

    /// Number of merged sections.
    pub fn section_count(&self) -> usize {
        self.merged_sections.len()
    }

    /// Total output size.
    pub fn output_size(&self) -> usize {
        self.merged_sections.iter().map(|s| s.data.len()).sum()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple object file with code and a symbol.
    fn make_obj(name: &str, code: &[u8], sym_name: &str, sym_offset: u64) -> ObjectFile {
        let mut obj = ObjectFile::new(name);
        obj.add_section(Section::with_data(".text", code.to_vec(), SectionFlags::code()));
        obj.add_symbol(Symbol::function(sym_name, ".text", sym_offset, code.len() as u64));
        obj
    }

    #[test]
    fn test_simple_link() {
        let obj = make_obj("a.o", &[0x90, 0x90, 0xC3], "main", 0);
        let mut linker = Linker::new(0x1000);
        linker.add_object(obj);
        let result = linker.link();
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output, vec![0x90, 0x90, 0xC3]);
    }

    #[test]
    fn test_two_objects_merge() {
        let obj_a = make_obj("a.o", &[0x90, 0x90], "foo", 0);
        let obj_b = make_obj("b.o", &[0xCC, 0xCC], "bar", 0);

        let mut linker = Linker::new(0x1000);
        linker.add_object(obj_a);
        linker.add_object(obj_b);
        let result = linker.link();
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.len() >= 4);
    }

    #[test]
    fn test_symbol_resolution() {
        let obj_a = make_obj("a.o", &[0x90; 4], "main", 0);
        let mut linker = Linker::new(0x1000);
        linker.add_object(obj_a);
        linker.link().unwrap();

        let addr = linker.symbol_address("main");
        assert!(addr.is_some());
        assert_eq!(addr.unwrap(), 0x1000);
    }

    #[test]
    fn test_undefined_symbol() {
        let mut obj = ObjectFile::new("a.o");
        obj.add_section(Section::with_data(".text", vec![0xE8, 0, 0, 0, 0], SectionFlags::code()));
        obj.add_symbol(Symbol::undefined("missing_func"));
        obj.add_relocation(Relocation::new(".text", 1, RelocType::Rel32, "missing_func"));

        let mut linker = Linker::new(0x1000);
        linker.add_object(obj);
        let result = linker.link();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        // The undefined symbol may surface as UndefinedSymbol or RelocationTargetNotFound
        let has_undef_error = errors.iter().any(|e| {
            matches!(e, LinkerError::UndefinedSymbol(n) if n == "missing_func")
                || matches!(e, LinkerError::RelocationTargetNotFound { symbol, .. } if symbol == "missing_func")
        });
        assert!(has_undef_error);
    }

    #[test]
    fn test_duplicate_symbol() {
        let obj_a = make_obj("a.o", &[0x90], "dup_sym", 0);
        let obj_b = make_obj("b.o", &[0xCC], "dup_sym", 0);

        let mut linker = Linker::new(0x1000);
        linker.add_object(obj_a);
        linker.add_object(obj_b);
        let result = linker.link();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(e, LinkerError::DuplicateSymbol { name, .. } if name == "dup_sym")));
    }

    #[test]
    fn test_weak_symbol_override() {
        let mut obj_a = ObjectFile::new("a.o");
        obj_a.add_section(Section::with_data(".text", vec![0x01], SectionFlags::code()));
        obj_a.add_symbol(Symbol::weak("shared", ".text", 0));

        let mut obj_b = ObjectFile::new("b.o");
        obj_b.add_section(Section::with_data(".text", vec![0x02], SectionFlags::code()));
        obj_b.add_symbol(Symbol::global("shared", ".text", 0));

        let mut linker = Linker::new(0x1000);
        linker.add_object(obj_a);
        linker.add_object(obj_b);
        let result = linker.link();
        assert!(result.is_ok());
    }

    #[test]
    fn test_relocation_abs32() {
        let mut obj_a = ObjectFile::new("a.o");
        obj_a.add_section(Section::with_data(".text", vec![0; 4], SectionFlags::code()));
        obj_a.add_relocation(Relocation::new(".text", 0, RelocType::Abs32, "target"));

        let mut obj_b = ObjectFile::new("b.o");
        obj_b.add_section(Section::with_data(".text", vec![0xCC; 4], SectionFlags::code()));
        obj_b.add_symbol(Symbol::global("target", ".text", 0));

        let mut linker = Linker::new(0x1000);
        linker.add_object(obj_a);
        linker.add_object(obj_b);
        let result = linker.link();
        assert!(result.is_ok());
        let output = result.unwrap();
        // The first 4 bytes should contain the absolute address of "target"
        let addr = u32::from_le_bytes([output[0], output[1], output[2], output[3]]);
        assert!(addr >= 0x1000);
    }

    #[test]
    fn test_section_merging() {
        let obj_a = make_obj("a.o", &[1, 2], "foo", 0);
        let obj_b = make_obj("b.o", &[3, 4], "bar", 0);

        let mut linker = Linker::new(0x0);
        linker.add_object(obj_a);
        linker.add_object(obj_b);
        linker.link().unwrap();

        assert_eq!(linker.section_count(), 1);
    }

    #[test]
    fn test_link_map_output() {
        let obj_a = make_obj("a.o", &[0x90; 8], "main", 0);
        let mut linker = Linker::new(0x1000);
        linker.add_object(obj_a);
        linker.link().unwrap();

        let map = linker.link_map_string();
        assert!(map.contains("Link Map"));
        assert!(map.contains("main"));
        assert!(map.contains(".text"));
    }

    #[test]
    fn test_link_map_entries() {
        let obj = make_obj("a.o", &[0x90], "entry", 0);
        let mut linker = Linker::new(0x1000);
        linker.add_object(obj);
        linker.link().unwrap();

        let entries = linker.link_map();
        assert!(!entries.is_empty());
        assert_eq!(entries[0].name, "entry");
    }

    #[test]
    fn test_data_section() {
        let mut obj = ObjectFile::new("a.o");
        obj.add_section(Section::with_data(".text", vec![0x90], SectionFlags::code()));
        obj.add_section(Section::with_data(
            ".data",
            vec![0xDE, 0xAD, 0xBE, 0xEF],
            SectionFlags::data(),
        ));
        obj.add_symbol(Symbol::global("code_start", ".text", 0));
        obj.add_symbol(Symbol::global("data_start", ".data", 0));

        let mut linker = Linker::new(0x1000);
        linker.add_object(obj);
        linker.link().unwrap();

        assert!(linker.symbol_address("data_start").is_some());
    }

    #[test]
    fn test_empty_link() {
        let mut linker = Linker::new(0x1000);
        let result = linker.link();
        assert!(result.is_ok());
        assert_eq!(linker.output_size(), 0);
    }

    #[test]
    fn test_object_file_find() {
        let obj = make_obj("a.o", &[0x90], "main", 0);
        assert!(obj.find_section(".text").is_some());
        assert!(obj.find_section(".data").is_none());
        assert!(obj.find_symbol("main").is_some());
        assert!(obj.find_symbol("missing").is_none());
    }

    #[test]
    fn test_linker_error_display() {
        let err = LinkerError::UndefinedSymbol("foo".into());
        let s = format!("{err}");
        assert!(s.contains("undefined"));
        assert!(s.contains("foo"));
    }

    #[test]
    fn test_reloc_type_display() {
        assert_eq!(format!("{}", RelocType::Abs32), "R_ABS32");
        assert_eq!(format!("{}", RelocType::Rel32), "R_REL32");
    }

    #[test]
    fn test_symbol_binding_display() {
        assert_eq!(format!("{}", SymbolBinding::Global), "GLOBAL");
        assert_eq!(format!("{}", SymbolBinding::Weak), "WEAK");
    }
}
