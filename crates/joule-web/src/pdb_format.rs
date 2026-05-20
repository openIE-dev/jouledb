//! PDB format writer/serializer — ATOM/HETATM records, coordinate output.
//!
//! Generates Protein Data Bank (PDB) formatted text from structured atom
//! data, supporting ATOM, HETATM, TER, MODEL/ENDMDL, CONECT records,
//! B-factor and occupancy fields, and coordinate transformations.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum PdbError {
    EmptyInput,
    InvalidAtomName(String),
    InvalidResidue(String),
    InvalidChain(String),
    CoordinateOverflow { atom: usize, axis: char, value: f64 },
    SerialOverflow(usize),
}

impl fmt::Display for PdbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "empty PDB input"),
            Self::InvalidAtomName(s) => write!(f, "invalid atom name: {s}"),
            Self::InvalidResidue(s) => write!(f, "invalid residue: {s}"),
            Self::InvalidChain(s) => write!(f, "invalid chain: {s}"),
            Self::CoordinateOverflow { atom, axis, value } => {
                write!(f, "atom {atom}: {axis} coordinate {value} exceeds PDB range")
            }
            Self::SerialOverflow(n) => write!(f, "serial number {n} exceeds 99999"),
        }
    }
}

impl std::error::Error for PdbError {}

// ── Record type ─────────────────────────────────────────────────

/// Atom record type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordType {
    Atom,
    Hetatm,
}

impl fmt::Display for RecordType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Atom => write!(f, "ATOM"),
            Self::Hetatm => write!(f, "HETATM"),
        }
    }
}

// ── 3D coordinate ───────────────────────────────────────────────

/// A 3D coordinate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coord {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Coord {
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0, z: 0.0 } }

    /// Euclidean distance to another coordinate.
    pub fn distance_to(self, other: Coord) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Translate by dx, dy, dz.
    pub fn translate(self, dx: f64, dy: f64, dz: f64) -> Self {
        Self {
            x: self.x + dx,
            y: self.y + dy,
            z: self.z + dz,
        }
    }

    /// Scale all axes uniformly.
    pub fn scale(self, factor: f64) -> Self {
        Self {
            x: self.x * factor,
            y: self.y * factor,
            z: self.z * factor,
        }
    }
}

impl fmt::Display for Coord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.3}, {:.3}, {:.3})", self.x, self.y, self.z)
    }
}

// ── Atom record ─────────────────────────────────────────────────

/// A PDB atom record.
#[derive(Debug, Clone)]
pub struct AtomRecord {
    pub record_type: RecordType,
    pub serial: usize,
    pub name: String,
    pub alt_loc: char,
    pub residue_name: String,
    pub chain_id: char,
    pub residue_seq: i32,
    pub insertion_code: char,
    pub coord: Coord,
    pub occupancy: f64,
    pub b_factor: f64,
    pub element: String,
    pub charge: String,
}

impl AtomRecord {
    pub fn new(serial: usize, name: &str, residue: &str, chain: char, seq: i32, coord: Coord) -> Self {
        Self {
            record_type: RecordType::Atom,
            serial,
            name: name.to_string(),
            alt_loc: ' ',
            residue_name: residue.to_string(),
            chain_id: chain,
            residue_seq: seq,
            insertion_code: ' ',
            coord,
            occupancy: 1.0,
            b_factor: 0.0,
            element: guess_element(name),
            charge: String::new(),
        }
    }

    pub fn with_record_type(mut self, rt: RecordType) -> Self {
        self.record_type = rt;
        self
    }

    pub fn with_alt_loc(mut self, a: char) -> Self { self.alt_loc = a; self }
    pub fn with_occupancy(mut self, o: f64) -> Self { self.occupancy = o; self }
    pub fn with_b_factor(mut self, b: f64) -> Self { self.b_factor = b; self }
    pub fn with_element(mut self, e: &str) -> Self { self.element = e.to_string(); self }
    pub fn with_charge(mut self, c: &str) -> Self { self.charge = c.to_string(); self }

    /// Format as a PDB ATOM/HETATM record (80-char fixed-width).
    pub fn to_pdb_line(&self) -> Result<String, PdbError> {
        if self.serial > 99999 {
            return Err(PdbError::SerialOverflow(self.serial));
        }
        for (axis, val) in [('x', self.coord.x), ('y', self.coord.y), ('z', self.coord.z)] {
            if val.abs() > 9999.999 {
                return Err(PdbError::CoordinateOverflow {
                    atom: self.serial,
                    axis,
                    value: val,
                });
            }
        }

        let record_tag = match self.record_type {
            RecordType::Atom => "ATOM  ",
            RecordType::Hetatm => "HETATM",
        };

        // PDB format columns (1-indexed):
        //  1-6   record name
        //  7-11  serial
        // 13-16  atom name
        // 17     alt loc
        // 18-20  residue name
        // 22     chain
        // 23-26  residue seq
        // 27     insertion code
        // 31-38  x (8.3)
        // 39-46  y (8.3)
        // 47-54  z (8.3)
        // 55-60  occupancy (6.2)
        // 61-66  B-factor (6.2)
        // 77-78  element
        // 79-80  charge
        let atom_name = format_atom_name(&self.name);
        let line = format!(
            "{:<6}{:>5} {:<4}{}{:>3} {}{:>4}{}   {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}{:<2}",
            record_tag,
            self.serial,
            atom_name,
            self.alt_loc,
            self.residue_name,
            self.chain_id,
            self.residue_seq,
            self.insertion_code,
            self.coord.x,
            self.coord.y,
            self.coord.z,
            self.occupancy,
            self.b_factor,
            self.element,
            self.charge,
        );
        Ok(line)
    }
}

impl fmt::Display for AtomRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {:>5} {} {} {}{:>4} {}",
               self.record_type, self.serial, self.name,
               self.residue_name, self.chain_id, self.residue_seq,
               self.coord)
    }
}

fn format_atom_name(name: &str) -> String {
    if name.len() >= 4 {
        name[..4].to_string()
    } else if name.len() == 1 {
        format!(" {name}  ")
    } else if name.len() == 2 {
        format!(" {name} ")
    } else {
        format!(" {name}")
    }
}

fn guess_element(name: &str) -> String {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    // First non-digit character is often the element
    let first_alpha: String = trimmed.chars().filter(|c| c.is_ascii_alphabetic()).take(1).collect();
    first_alpha.to_uppercase()
}

// ── CONECT record ───────────────────────────────────────────────

/// A PDB CONECT record (bonding information).
#[derive(Debug, Clone)]
pub struct ConectRecord {
    pub atom_serial: usize,
    pub bonded: Vec<usize>,
}

impl ConectRecord {
    pub fn new(serial: usize) -> Self {
        Self { atom_serial: serial, bonded: Vec::new() }
    }

    pub fn with_bond(mut self, other: usize) -> Self {
        self.bonded.push(other);
        self
    }

    pub fn to_pdb_line(&self) -> String {
        let mut s = format!("CONECT{:>5}", self.atom_serial);
        for b in &self.bonded {
            s.push_str(&format!("{:>5}", b));
        }
        s
    }
}

impl fmt::Display for ConectRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CONECT {} -> {:?}", self.atom_serial, self.bonded)
    }
}

// ── PDB document ────────────────────────────────────────────────

/// A PDB structure ready for serialization.
#[derive(Debug, Clone)]
pub struct PdbDocument {
    pub title: Option<String>,
    pub models: Vec<PdbModel>,
    pub conects: Vec<ConectRecord>,
}

/// A single MODEL in a PDB file.
#[derive(Debug, Clone)]
pub struct PdbModel {
    pub model_number: usize,
    pub atoms: Vec<AtomRecord>,
}

impl PdbDocument {
    pub fn new() -> Self {
        Self {
            title: None,
            models: Vec::new(),
            conects: Vec::new(),
        }
    }

    pub fn with_title(mut self, t: &str) -> Self { self.title = Some(t.to_string()); self }

    pub fn with_model(mut self, model: PdbModel) -> Self {
        self.models.push(model);
        self
    }

    pub fn with_conect(mut self, c: ConectRecord) -> Self {
        self.conects.push(c);
        self
    }

    /// Total atom count across all models.
    pub fn atom_count(&self) -> usize {
        self.models.iter().map(|m| m.atoms.len()).sum()
    }

    /// Center of mass for the first model.
    pub fn center_of_mass(&self) -> Coord {
        let atoms = match self.models.first() {
            Some(m) => &m.atoms,
            None => return Coord::zero(),
        };
        if atoms.is_empty() { return Coord::zero(); }
        let n = atoms.len() as f64;
        let sum_x: f64 = atoms.iter().map(|a| a.coord.x).sum();
        let sum_y: f64 = atoms.iter().map(|a| a.coord.y).sum();
        let sum_z: f64 = atoms.iter().map(|a| a.coord.z).sum();
        Coord::new(sum_x / n, sum_y / n, sum_z / n)
    }

    /// Bounding box (min, max) for the first model.
    pub fn bounding_box(&self) -> (Coord, Coord) {
        let atoms = match self.models.first() {
            Some(m) => &m.atoms,
            None => return (Coord::zero(), Coord::zero()),
        };
        if atoms.is_empty() { return (Coord::zero(), Coord::zero()); }
        let mut min = Coord::new(f64::MAX, f64::MAX, f64::MAX);
        let mut max = Coord::new(f64::MIN, f64::MIN, f64::MIN);
        for a in atoms {
            if a.coord.x < min.x { min.x = a.coord.x; }
            if a.coord.y < min.y { min.y = a.coord.y; }
            if a.coord.z < min.z { min.z = a.coord.z; }
            if a.coord.x > max.x { max.x = a.coord.x; }
            if a.coord.y > max.y { max.y = a.coord.y; }
            if a.coord.z > max.z { max.z = a.coord.z; }
        }
        (min, max)
    }

    /// Serialize the complete PDB document.
    pub fn to_pdb(&self) -> Result<String, PdbError> {
        let mut out = String::new();

        if let Some(ref t) = self.title {
            out.push_str(&format!("TITLE     {t}\n"));
        }

        let multi_model = self.models.len() > 1;
        for model in &self.models {
            if multi_model {
                out.push_str(&format!("MODEL     {:>4}\n", model.model_number));
            }
            let mut prev_chain: Option<char> = None;
            for atom in &model.atoms {
                // Insert TER between chains
                if let Some(pc) = prev_chain {
                    if pc != atom.chain_id {
                        out.push_str("TER\n");
                    }
                }
                out.push_str(&atom.to_pdb_line()?);
                out.push('\n');
                prev_chain = Some(atom.chain_id);
            }
            out.push_str("TER\n");
            if multi_model {
                out.push_str("ENDMDL\n");
            }
        }

        for c in &self.conects {
            out.push_str(&c.to_pdb_line());
            out.push('\n');
        }
        out.push_str("END\n");
        Ok(out)
    }
}

impl fmt::Display for PdbDocument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PdbDocument(models={}, atoms={})", self.models.len(), self.atom_count())
    }
}

impl PdbModel {
    pub fn new(number: usize) -> Self {
        Self { model_number: number, atoms: Vec::new() }
    }

    pub fn with_atom(mut self, atom: AtomRecord) -> Self {
        self.atoms.push(atom);
        self
    }
}

impl fmt::Display for PdbModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Model {} ({} atoms)", self.model_number, self.atoms.len())
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_atom(serial: usize, name: &str, x: f64, y: f64, z: f64) -> AtomRecord {
        AtomRecord::new(serial, name, "ALA", 'A', 1, Coord::new(x, y, z))
    }

    #[test]
    fn t01_atom_line_format() {
        let atom = sample_atom(1, "CA", 1.0, 2.0, 3.0);
        let line = atom.to_pdb_line().unwrap();
        assert!(line.starts_with("ATOM  "));
        assert!(line.contains("ALA"));
    }

    #[test]
    fn t02_hetatm() {
        let atom = sample_atom(1, "O", 0.0, 0.0, 0.0)
            .with_record_type(RecordType::Hetatm);
        let line = atom.to_pdb_line().unwrap();
        assert!(line.starts_with("HETATM"));
    }

    #[test]
    fn t03_serial_overflow() {
        let atom = sample_atom(100000, "CA", 0.0, 0.0, 0.0);
        assert!(matches!(atom.to_pdb_line(), Err(PdbError::SerialOverflow(_))));
    }

    #[test]
    fn t04_coordinate_overflow() {
        let atom = sample_atom(1, "CA", 99999.0, 0.0, 0.0);
        assert!(matches!(atom.to_pdb_line(), Err(PdbError::CoordinateOverflow { .. })));
    }

    #[test]
    fn t05_coord_distance() {
        let a = Coord::new(0.0, 0.0, 0.0);
        let b = Coord::new(3.0, 4.0, 0.0);
        assert!((a.distance_to(b) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn t06_coord_translate() {
        let c = Coord::new(1.0, 2.0, 3.0).translate(10.0, 20.0, 30.0);
        assert!((c.x - 11.0).abs() < 1e-9);
    }

    #[test]
    fn t07_coord_scale() {
        let c = Coord::new(1.0, 2.0, 3.0).scale(2.0);
        assert!((c.y - 4.0).abs() < 1e-9);
    }

    #[test]
    fn t08_document_single_model() {
        let model = PdbModel::new(1)
            .with_atom(sample_atom(1, "N", 0.0, 0.0, 0.0))
            .with_atom(sample_atom(2, "CA", 1.5, 0.0, 0.0));
        let doc = PdbDocument::new()
            .with_title("Test")
            .with_model(model);
        let pdb = doc.to_pdb().unwrap();
        assert!(pdb.contains("TITLE     Test"));
        assert!(pdb.contains("END"));
        assert_eq!(doc.atom_count(), 2);
    }

    #[test]
    fn t09_multi_model() {
        let m1 = PdbModel::new(1).with_atom(sample_atom(1, "CA", 0.0, 0.0, 0.0));
        let m2 = PdbModel::new(2).with_atom(sample_atom(1, "CA", 1.0, 0.0, 0.0));
        let doc = PdbDocument::new().with_model(m1).with_model(m2);
        let pdb = doc.to_pdb().unwrap();
        assert!(pdb.contains("MODEL"));
        assert!(pdb.contains("ENDMDL"));
    }

    #[test]
    fn t10_conect_record() {
        let c = ConectRecord::new(1).with_bond(2).with_bond(3);
        let line = c.to_pdb_line();
        assert!(line.starts_with("CONECT"));
        assert!(line.contains("    1"));
    }

    #[test]
    fn t11_center_of_mass() {
        let model = PdbModel::new(1)
            .with_atom(sample_atom(1, "CA", 0.0, 0.0, 0.0))
            .with_atom(sample_atom(2, "CA", 10.0, 0.0, 0.0));
        let doc = PdbDocument::new().with_model(model);
        let com = doc.center_of_mass();
        assert!((com.x - 5.0).abs() < 1e-9);
    }

    #[test]
    fn t12_bounding_box() {
        let model = PdbModel::new(1)
            .with_atom(sample_atom(1, "CA", -5.0, 0.0, 10.0))
            .with_atom(sample_atom(2, "CA", 5.0, 8.0, -3.0));
        let doc = PdbDocument::new().with_model(model);
        let (bmin, bmax) = doc.bounding_box();
        assert!((bmin.x - (-5.0)).abs() < 1e-9);
        assert!((bmax.y - 8.0).abs() < 1e-9);
    }

    #[test]
    fn t13_b_factor() {
        let atom = sample_atom(1, "CA", 0.0, 0.0, 0.0).with_b_factor(25.5);
        let line = atom.to_pdb_line().unwrap();
        assert!(line.contains("25.50"));
    }

    #[test]
    fn t14_occupancy() {
        let atom = sample_atom(1, "CA", 0.0, 0.0, 0.0).with_occupancy(0.75);
        let line = atom.to_pdb_line().unwrap();
        assert!(line.contains("0.75"));
    }

    #[test]
    fn t15_chain_ter() {
        let model = PdbModel::new(1)
            .with_atom(AtomRecord::new(1, "CA", "ALA", 'A', 1, Coord::zero()))
            .with_atom(AtomRecord::new(2, "CA", "GLY", 'B', 1, Coord::zero()));
        let doc = PdbDocument::new().with_model(model);
        let pdb = doc.to_pdb().unwrap();
        // Should have TER between chains + final TER
        let ter_count = pdb.matches("TER\n").count();
        assert_eq!(ter_count, 2);
    }

    #[test]
    fn t16_element_guess() {
        assert_eq!(guess_element("CA"), "C");
        assert_eq!(guess_element("N"), "N");
    }

    #[test]
    fn t17_empty_doc() {
        let doc = PdbDocument::new();
        assert_eq!(doc.atom_count(), 0);
        let pdb = doc.to_pdb().unwrap();
        assert!(pdb.contains("END"));
    }

    #[test]
    fn t18_display_atom() {
        let atom = sample_atom(42, "CA", 1.0, 2.0, 3.0);
        let s = format!("{atom}");
        assert!(s.contains("42"));
        assert!(s.contains("CA"));
    }

    #[test]
    fn t19_display_document() {
        let doc = PdbDocument::new();
        let s = format!("{doc}");
        assert!(s.contains("models=0"));
    }

    #[test]
    fn t20_coord_display() {
        let c = Coord::new(1.234, 5.678, 9.012);
        let s = format!("{c}");
        assert!(s.contains("1.234"));
        assert!(s.contains("5.678"));
    }
}
