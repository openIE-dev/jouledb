//! Code emission — instruction encoding, label resolution, fixup/relocation,
//! section management (.text/.data/.bss), symbol table emission, output buffer,
//! binary format concepts.

use std::collections::HashMap;
use std::fmt;

// ── Sections ────────────────────────────────────────────────────────────────

/// Well-known section names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SectionKind {
    /// Executable code.
    Text,
    /// Initialized data.
    Data,
    /// Uninitialized data (zero-filled).
    Bss,
    /// Read-only data.
    Rodata,
    /// Custom section.
    Custom,
}

impl fmt::Display for SectionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Text => ".text",
            Self::Data => ".data",
            Self::Bss => ".bss",
            Self::Rodata => ".rodata",
            Self::Custom => ".custom",
        };
        write!(f, "{s}")
    }
}

/// Section flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SectionFlags {
    /// Section is writable.
    pub writable: bool,
    /// Section is executable.
    pub executable: bool,
    /// Section is allocatable (loaded into memory).
    pub allocatable: bool,
}

impl SectionFlags {
    /// Flags for a code section.
    pub fn code() -> Self {
        Self {
            writable: false,
            executable: true,
            allocatable: true,
        }
    }

    /// Flags for an initialized data section.
    pub fn data() -> Self {
        Self {
            writable: true,
            executable: false,
            allocatable: true,
        }
    }

    /// Flags for a BSS section.
    pub fn bss() -> Self {
        Self {
            writable: true,
            executable: false,
            allocatable: true,
        }
    }

    /// Flags for read-only data.
    pub fn rodata() -> Self {
        Self {
            writable: false,
            executable: false,
            allocatable: true,
        }
    }
}

/// A section in the output.
#[derive(Debug, Clone)]
pub struct Section {
    /// Section name.
    pub name: String,
    /// Section kind.
    pub kind: SectionKind,
    /// Section flags.
    pub flags: SectionFlags,
    /// Section content bytes.
    pub data: Vec<u8>,
    /// Alignment requirement (power of 2).
    pub alignment: u32,
    /// Base address (assigned during layout).
    pub base_address: u64,
}

impl Section {
    /// Create a new section.
    pub fn new(name: &str, kind: SectionKind, flags: SectionFlags) -> Self {
        Self {
            name: name.to_string(),
            kind,
            flags,
            data: Vec::new(),
            alignment: 4,
            base_address: 0,
        }
    }

    /// Create a .text section.
    pub fn text() -> Self {
        Self::new(".text", SectionKind::Text, SectionFlags::code())
    }

    /// Create a .data section.
    pub fn data() -> Self {
        Self::new(".data", SectionKind::Data, SectionFlags::data())
    }

    /// Create a .bss section.
    pub fn bss(size: usize) -> Self {
        let mut s = Self::new(".bss", SectionKind::Bss, SectionFlags::bss());
        s.data = vec![0u8; size];
        s
    }

    /// Create a .rodata section.
    pub fn rodata() -> Self {
        Self::new(".rodata", SectionKind::Rodata, SectionFlags::rodata())
    }

    /// Current size in bytes.
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Append bytes.
    pub fn emit_bytes(&mut self, bytes: &[u8]) -> usize {
        let offset = self.data.len();
        self.data.extend_from_slice(bytes);
        offset
    }

    /// Emit a u8.
    pub fn emit_u8(&mut self, val: u8) -> usize {
        let offset = self.data.len();
        self.data.push(val);
        offset
    }

    /// Emit a little-endian u16.
    pub fn emit_u16_le(&mut self, val: u16) -> usize {
        let offset = self.data.len();
        self.data.extend_from_slice(&val.to_le_bytes());
        offset
    }

    /// Emit a little-endian u32.
    pub fn emit_u32_le(&mut self, val: u32) -> usize {
        let offset = self.data.len();
        self.data.extend_from_slice(&val.to_le_bytes());
        offset
    }

    /// Emit a little-endian u64.
    pub fn emit_u64_le(&mut self, val: u64) -> usize {
        let offset = self.data.len();
        self.data.extend_from_slice(&val.to_le_bytes());
        offset
    }

    /// Write a u32 at a specific offset (for fixups).
    pub fn patch_u32_le(&mut self, offset: usize, val: u32) -> bool {
        if offset + 4 > self.data.len() {
            return false;
        }
        self.data[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
        true
    }

    /// Pad to alignment boundary.
    pub fn align_to(&mut self, alignment: usize) {
        if alignment == 0 {
            return;
        }
        let remainder = self.data.len() % alignment;
        if remainder != 0 {
            let padding = alignment - remainder;
            self.data.extend(std::iter::repeat(0u8).take(padding));
        }
    }
}

// ── Labels ──────────────────────────────────────────────────────────────────

/// A label in the code — resolved to an offset within a section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Label {
    /// Label name.
    pub name: String,
    /// Section index.
    pub section_idx: usize,
    /// Offset within the section.
    pub offset: u64,
    /// Whether the label is defined.
    pub defined: bool,
}

impl Label {
    /// Create a new undefined label.
    pub fn undefined(name: &str) -> Self {
        Self {
            name: name.to_string(),
            section_idx: 0,
            offset: 0,
            defined: false,
        }
    }

    /// Define the label at a specific location.
    pub fn define(&mut self, section_idx: usize, offset: u64) {
        self.section_idx = section_idx;
        self.offset = offset;
        self.defined = true;
    }

    /// Resolved address (base_address + offset).
    pub fn resolved_address(&self, section_base: u64) -> u64 {
        section_base + self.offset
    }
}

// ── Fixups / Relocations ────────────────────────────────────────────────────

/// The kind of fixup to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixupKind {
    /// Absolute 32-bit address.
    Abs32,
    /// Absolute 64-bit address.
    Abs64,
    /// PC-relative 32-bit offset.
    Rel32,
    /// PC-relative 8-bit offset (short branch).
    Rel8,
}

impl fmt::Display for FixupKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Abs32 => "ABS32",
            Self::Abs64 => "ABS64",
            Self::Rel32 => "REL32",
            Self::Rel8 => "REL8",
        };
        write!(f, "{s}")
    }
}

/// A fixup that needs to be applied after layout.
#[derive(Debug, Clone)]
pub struct Fixup {
    /// Section index where the fixup is.
    pub section_idx: usize,
    /// Offset within the section where the fixup starts.
    pub offset: usize,
    /// The fixup kind.
    pub kind: FixupKind,
    /// Target label name.
    pub target: String,
    /// Addend (added to the resolved address).
    pub addend: i64,
}

impl Fixup {
    /// Create a new fixup.
    pub fn new(section_idx: usize, offset: usize, kind: FixupKind, target: &str) -> Self {
        Self {
            section_idx,
            offset,
            kind,
            target: target.to_string(),
            addend: 0,
        }
    }

    /// Set addend.
    pub fn with_addend(mut self, addend: i64) -> Self {
        self.addend = addend;
        self
    }
}

// ── Symbol table ────────────────────────────────────────────────────────────

/// Symbol binding (local or global).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolBinding {
    Local,
    Global,
    Weak,
}

/// Symbol type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolType {
    Function,
    Object,
    Section,
    NoType,
}

/// A symbol table entry.
#[derive(Debug, Clone)]
pub struct Symbol {
    /// Symbol name.
    pub name: String,
    /// Section index (None for undefined/external).
    pub section_idx: Option<usize>,
    /// Offset within the section.
    pub offset: u64,
    /// Size of the symbol (e.g. function size).
    pub size: u64,
    /// Binding.
    pub binding: SymbolBinding,
    /// Type.
    pub sym_type: SymbolType,
}

impl Symbol {
    /// Create a new symbol.
    pub fn new(name: &str, section_idx: Option<usize>, offset: u64) -> Self {
        Self {
            name: name.to_string(),
            section_idx,
            offset,
            size: 0,
            binding: SymbolBinding::Global,
            sym_type: SymbolType::NoType,
        }
    }

    /// Create a function symbol.
    pub fn function(name: &str, section_idx: usize, offset: u64, size: u64) -> Self {
        Self {
            name: name.to_string(),
            section_idx: Some(section_idx),
            offset,
            size,
            binding: SymbolBinding::Global,
            sym_type: SymbolType::Function,
        }
    }

    /// Whether the symbol is defined.
    pub fn is_defined(&self) -> bool {
        self.section_idx.is_some()
    }

    /// Whether the symbol is external (undefined).
    pub fn is_external(&self) -> bool {
        self.section_idx.is_none()
    }

    /// Resolved address given the section base.
    pub fn resolved_address(&self, section_base: u64) -> u64 {
        section_base + self.offset
    }
}

// ── Code emitter ────────────────────────────────────────────────────────────

/// Errors from code emission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitError {
    /// Undefined label referenced.
    UndefinedLabel(String),
    /// Section not found.
    SectionNotFound(String),
    /// Fixup out of range.
    FixupOutOfRange { fixup: String, offset: i64 },
    /// Duplicate label.
    DuplicateLabel(String),
    /// Duplicate symbol.
    DuplicateSymbol(String),
}

impl fmt::Display for EmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UndefinedLabel(l) => write!(f, "undefined label: {l}"),
            Self::SectionNotFound(s) => write!(f, "section not found: {s}"),
            Self::FixupOutOfRange { fixup, offset } => {
                write!(f, "fixup '{fixup}' out of range at offset {offset}")
            }
            Self::DuplicateLabel(l) => write!(f, "duplicate label: {l}"),
            Self::DuplicateSymbol(s) => write!(f, "duplicate symbol: {s}"),
        }
    }
}

/// The code emitter manages sections, labels, fixups, and symbols.
pub struct CodeEmitter {
    /// Output sections.
    sections: Vec<Section>,
    /// Current section index.
    current_section: usize,
    /// Labels: name -> label.
    labels: HashMap<String, Label>,
    /// Fixups to resolve.
    fixups: Vec<Fixup>,
    /// Symbol table.
    symbols: Vec<Symbol>,
    /// Symbol name index for dedup.
    symbol_names: HashMap<String, usize>,
}

impl CodeEmitter {
    /// Create a new emitter with default .text, .data, .bss sections.
    pub fn new() -> Self {
        let sections = vec![
            Section::text(),
            Section::data(),
            Section::bss(0),
            Section::rodata(),
        ];
        Self {
            sections,
            current_section: 0,
            labels: HashMap::new(),
            fixups: Vec::new(),
            symbols: Vec::new(),
            symbol_names: HashMap::new(),
        }
    }

    /// Switch to a section by name.
    pub fn switch_section(&mut self, name: &str) -> Result<(), EmitError> {
        for (i, sec) in self.sections.iter().enumerate() {
            if sec.name == name {
                self.current_section = i;
                return Ok(());
            }
        }
        Err(EmitError::SectionNotFound(name.to_string()))
    }

    /// Add a custom section.
    pub fn add_section(&mut self, section: Section) -> usize {
        let idx = self.sections.len();
        self.sections.push(section);
        idx
    }

    /// Get the current section index.
    pub fn current_section_index(&self) -> usize {
        self.current_section
    }

    /// Get the current offset in the current section.
    pub fn current_offset(&self) -> usize {
        self.sections[self.current_section].size()
    }

    /// Emit raw bytes into the current section.
    pub fn emit_bytes(&mut self, bytes: &[u8]) -> usize {
        self.sections[self.current_section].emit_bytes(bytes)
    }

    /// Emit a u8 into the current section.
    pub fn emit_u8(&mut self, val: u8) -> usize {
        self.sections[self.current_section].emit_u8(val)
    }

    /// Emit a u32 (LE) into the current section.
    pub fn emit_u32_le(&mut self, val: u32) -> usize {
        self.sections[self.current_section].emit_u32_le(val)
    }

    /// Emit a u64 (LE) into the current section.
    pub fn emit_u64_le(&mut self, val: u64) -> usize {
        self.sections[self.current_section].emit_u64_le(val)
    }

    /// Define a label at the current position.
    pub fn define_label(&mut self, name: &str) -> Result<(), EmitError> {
        if let Some(existing) = self.labels.get(name) {
            if existing.defined {
                return Err(EmitError::DuplicateLabel(name.to_string()));
            }
        }
        let sec_idx = self.current_section;
        let offset = self.sections[sec_idx].size() as u64;
        let label = Label {
            name: name.to_string(),
            section_idx: sec_idx,
            offset,
            defined: true,
        };
        self.labels.insert(name.to_string(), label);
        Ok(())
    }

    /// Reference a label (creates it undefined if not yet defined).
    pub fn reference_label(&mut self, name: &str) {
        self.labels
            .entry(name.to_string())
            .or_insert_with(|| Label::undefined(name));
    }

    /// Add a fixup at the current position.
    pub fn add_fixup(&mut self, kind: FixupKind, target: &str, addend: i64) {
        let sec_idx = self.current_section;
        let offset = self.sections[sec_idx].size();
        self.reference_label(target);
        self.fixups.push(
            Fixup::new(sec_idx, offset, kind, target).with_addend(addend),
        );
    }

    /// Add a symbol.
    pub fn add_symbol(&mut self, symbol: Symbol) -> Result<usize, EmitError> {
        if self.symbol_names.contains_key(&symbol.name) {
            return Err(EmitError::DuplicateSymbol(symbol.name.clone()));
        }
        let idx = self.symbols.len();
        self.symbol_names.insert(symbol.name.clone(), idx);
        self.symbols.push(symbol);
        Ok(idx)
    }

    /// Resolve all fixups after layout.
    pub fn resolve_fixups(&mut self) -> Result<(), EmitError> {
        let fixups = self.fixups.clone();
        for fixup in &fixups {
            let label = self.labels.get(&fixup.target).ok_or_else(|| {
                EmitError::UndefinedLabel(fixup.target.clone())
            })?;

            if !label.defined {
                return Err(EmitError::UndefinedLabel(fixup.target.clone()));
            }

            let target_base = self.sections[label.section_idx].base_address;
            let target_addr = target_base + label.offset;

            let fixup_base = self.sections[fixup.section_idx].base_address;
            let fixup_addr = fixup_base + fixup.offset as u64;

            match fixup.kind {
                FixupKind::Abs32 => {
                    let val = (target_addr as i64 + fixup.addend) as u32;
                    self.sections[fixup.section_idx].patch_u32_le(fixup.offset, val);
                }
                FixupKind::Abs64 => {
                    let val = (target_addr as i64 + fixup.addend) as u64;
                    let bytes = val.to_le_bytes();
                    let sec = &mut self.sections[fixup.section_idx];
                    if fixup.offset + 8 <= sec.data.len() {
                        sec.data[fixup.offset..fixup.offset + 8]
                            .copy_from_slice(&bytes);
                    }
                }
                FixupKind::Rel32 => {
                    let rel = target_addr as i64 - fixup_addr as i64 + fixup.addend;
                    let val = rel as i32;
                    self.sections[fixup.section_idx]
                        .patch_u32_le(fixup.offset, val as u32);
                }
                FixupKind::Rel8 => {
                    let rel = target_addr as i64 - fixup_addr as i64 + fixup.addend;
                    if rel < -128 || rel > 127 {
                        return Err(EmitError::FixupOutOfRange {
                            fixup: fixup.target.clone(),
                            offset: rel,
                        });
                    }
                    let sec = &mut self.sections[fixup.section_idx];
                    if fixup.offset < sec.data.len() {
                        sec.data[fixup.offset] = rel as u8;
                    }
                }
            }
        }
        Ok(())
    }

    /// Assign base addresses to sections sequentially from a start address.
    pub fn layout(&mut self, start_address: u64) {
        let mut addr = start_address;
        for section in &mut self.sections {
            // Align section start
            let align = section.alignment as u64;
            if align > 0 {
                let remainder = addr % align;
                if remainder != 0 {
                    addr += align - remainder;
                }
            }
            section.base_address = addr;
            addr += section.size() as u64;
        }
    }

    /// Get all sections.
    pub fn sections(&self) -> &[Section] {
        &self.sections
    }

    /// Get a section by name.
    pub fn section_by_name(&self, name: &str) -> Option<&Section> {
        self.sections.iter().find(|s| s.name == name)
    }

    /// Get all symbols.
    pub fn symbols(&self) -> &[Symbol] {
        &self.symbols
    }

    /// Get all fixups.
    pub fn fixups(&self) -> &[Fixup] {
        &self.fixups
    }

    /// Get all labels.
    pub fn labels(&self) -> &HashMap<String, Label> {
        &self.labels
    }

    /// Total output size across all sections.
    pub fn total_size(&self) -> usize {
        self.sections.iter().map(|s| s.size()).sum()
    }

    /// Flatten all sections into a single byte vector (for output).
    pub fn flatten(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for section in &self.sections {
            out.extend_from_slice(&section.data);
        }
        out
    }

    /// Produce a human-readable listing.
    pub fn listing(&self) -> String {
        let mut out = String::new();
        for section in &self.sections {
            out.push_str(&format!(
                "Section {} (base=0x{:08X}, size={}):\n",
                section.name, section.base_address, section.size()
            ));
            // Hex dump first 64 bytes
            for (i, byte) in section.data.iter().take(64).enumerate() {
                if i > 0 && i % 16 == 0 {
                    out.push('\n');
                }
                out.push_str(&format!("{byte:02X} "));
            }
            if !section.data.is_empty() {
                out.push('\n');
            }
            out.push('\n');
        }
        out
    }
}

impl Default for CodeEmitter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_section_emit_bytes() {
        let mut sec = Section::text();
        let off = sec.emit_bytes(&[0x90, 0x90, 0x90]);
        assert_eq!(off, 0);
        assert_eq!(sec.size(), 3);
        assert_eq!(sec.data, vec![0x90, 0x90, 0x90]);
    }

    #[test]
    fn test_section_emit_u32() {
        let mut sec = Section::data();
        sec.emit_u32_le(0xDEADBEEF);
        assert_eq!(sec.size(), 4);
        assert_eq!(sec.data, vec![0xEF, 0xBE, 0xAD, 0xDE]);
    }

    #[test]
    fn test_section_patch() {
        let mut sec = Section::text();
        sec.emit_u32_le(0);
        assert!(sec.patch_u32_le(0, 0x12345678));
        assert_eq!(sec.data, vec![0x78, 0x56, 0x34, 0x12]);
    }

    #[test]
    fn test_section_align() {
        let mut sec = Section::text();
        sec.emit_bytes(&[1, 2, 3]);
        sec.align_to(8);
        assert_eq!(sec.size(), 8);
        assert_eq!(sec.data[3..8], vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_emitter_default_sections() {
        let em = CodeEmitter::new();
        assert_eq!(em.sections().len(), 4);
        assert!(em.section_by_name(".text").is_some());
        assert!(em.section_by_name(".data").is_some());
        assert!(em.section_by_name(".bss").is_some());
        assert!(em.section_by_name(".rodata").is_some());
    }

    #[test]
    fn test_emitter_switch_section() {
        let mut em = CodeEmitter::new();
        em.emit_bytes(&[0x90]);
        assert!(em.switch_section(".data").is_ok());
        em.emit_bytes(&[0xFF]);
        assert_eq!(em.section_by_name(".text").unwrap().size(), 1);
        assert_eq!(em.section_by_name(".data").unwrap().size(), 1);
    }

    #[test]
    fn test_emitter_switch_nonexistent() {
        let mut em = CodeEmitter::new();
        assert!(em.switch_section(".nosuch").is_err());
    }

    #[test]
    fn test_label_define_and_resolve() {
        let mut em = CodeEmitter::new();
        em.emit_bytes(&[0x90, 0x90]);
        em.define_label("loop_start").unwrap();
        let label = em.labels().get("loop_start").unwrap();
        assert!(label.defined);
        assert_eq!(label.offset, 2);
    }

    #[test]
    fn test_duplicate_label() {
        let mut em = CodeEmitter::new();
        em.define_label("foo").unwrap();
        assert!(em.define_label("foo").is_err());
    }

    #[test]
    fn test_fixup_rel32() {
        let mut em = CodeEmitter::new();
        // Emit some code
        em.emit_bytes(&[0xE8]); // call opcode
        em.add_fixup(FixupKind::Rel32, "target", -4);
        em.emit_u32_le(0); // placeholder
        em.define_label("target").unwrap();
        em.emit_bytes(&[0x90]); // nop at target

        em.layout(0x1000);
        assert!(em.resolve_fixups().is_ok());

        let text = em.section_by_name(".text").unwrap();
        // Target offset = 5, fixup at offset 1, base = 0x1000
        // rel = (0x1000 + 5) - (0x1000 + 1) + (-4) = 0
        let patched = u32::from_le_bytes([text.data[1], text.data[2], text.data[3], text.data[4]]);
        assert_eq!(patched, 0);
    }

    #[test]
    fn test_fixup_abs32() {
        let mut em = CodeEmitter::new();
        em.add_fixup(FixupKind::Abs32, "data_label", 0);
        em.emit_u32_le(0); // placeholder

        assert!(em.switch_section(".data").is_ok());
        em.define_label("data_label").unwrap();
        em.emit_u32_le(0xCAFEBABE);

        em.layout(0x0);
        assert!(em.resolve_fixups().is_ok());
    }

    #[test]
    fn test_undefined_label_error() {
        let mut em = CodeEmitter::new();
        em.add_fixup(FixupKind::Abs32, "nowhere", 0);
        em.emit_u32_le(0);
        em.layout(0);
        assert!(em.resolve_fixups().is_err());
    }

    #[test]
    fn test_symbol_table() {
        let mut em = CodeEmitter::new();
        let sym = Symbol::function("main", 0, 0, 32);
        assert!(em.add_symbol(sym).is_ok());
        assert_eq!(em.symbols().len(), 1);
        assert_eq!(em.symbols()[0].name, "main");
    }

    #[test]
    fn test_duplicate_symbol() {
        let mut em = CodeEmitter::new();
        em.add_symbol(Symbol::new("x", Some(0), 0)).unwrap();
        assert!(em.add_symbol(Symbol::new("x", Some(0), 4)).is_err());
    }

    #[test]
    fn test_layout_alignment() {
        let mut em = CodeEmitter::new();
        em.emit_bytes(&[0x90; 5]);
        em.layout(0x1000);
        // .text starts at 0x1000, .data starts at next aligned address
        let data_sec = em.section_by_name(".data").unwrap();
        assert!(data_sec.base_address >= 0x1005);
        assert_eq!(data_sec.base_address % 4, 0);
    }

    #[test]
    fn test_total_size_and_flatten() {
        let mut em = CodeEmitter::new();
        em.emit_bytes(&[1, 2, 3]);
        assert!(em.switch_section(".data").is_ok());
        em.emit_bytes(&[4, 5]);
        assert_eq!(em.total_size(), 5);
        let flat = em.flatten();
        assert_eq!(flat.len(), 5);
    }

    #[test]
    fn test_listing_output() {
        let mut em = CodeEmitter::new();
        em.emit_bytes(&[0xCC; 4]);
        em.layout(0x0);
        let listing = em.listing();
        assert!(listing.contains(".text"));
        assert!(listing.contains("CC"));
    }

    #[test]
    fn test_section_kind_display() {
        assert_eq!(format!("{}", SectionKind::Text), ".text");
        assert_eq!(format!("{}", SectionKind::Data), ".data");
        assert_eq!(format!("{}", SectionKind::Bss), ".bss");
    }

    #[test]
    fn test_fixup_kind_display() {
        assert_eq!(format!("{}", FixupKind::Abs32), "ABS32");
        assert_eq!(format!("{}", FixupKind::Rel32), "REL32");
    }
}
