//! PDB file format parser for atomic coordinate data.
//!
//! Parses ATOM, HETATM, MODEL, ENDMDL, TER, and HEADER records from
//! PDB-format strings. Extracts chain identifiers, residue sequences,
//! and 3D coordinates for structural analysis.

use std::fmt;

// ── Record Kind ─────────────────────────────────────────────────────

/// Kind of PDB record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecordKind {
    Atom,
    Hetatm,
    Ter,
    Model,
    EndModel,
    Header,
    Remark,
    Other,
}

impl fmt::Display for RecordKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Atom => write!(f, "ATOM"),
            Self::Hetatm => write!(f, "HETATM"),
            Self::Ter => write!(f, "TER"),
            Self::Model => write!(f, "MODEL"),
            Self::EndModel => write!(f, "ENDMDL"),
            Self::Header => write!(f, "HEADER"),
            Self::Remark => write!(f, "REMARK"),
            Self::Other => write!(f, "OTHER"),
        }
    }
}

// ── Atom Record ─────────────────────────────────────────────────────

/// A parsed ATOM or HETATM record.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomRecord {
    /// Serial number.
    pub serial: u32,
    /// Atom name (e.g. "CA", "N", "CB").
    pub name: String,
    /// Alternate location indicator.
    pub alt_loc: char,
    /// Residue name (e.g. "ALA", "HOH").
    pub res_name: String,
    /// Chain identifier.
    pub chain: char,
    /// Residue sequence number.
    pub res_seq: i32,
    /// Insertion code.
    pub ins_code: char,
    /// Cartesian coordinates [x, y, z] in angstroms.
    pub coords: [f64; 3],
    /// Occupancy factor.
    pub occupancy: f64,
    /// Temperature factor (B-factor).
    pub b_factor: f64,
    /// Element symbol.
    pub element: String,
    /// True if HETATM, false if ATOM.
    pub is_hetatm: bool,
}

impl AtomRecord {
    /// Distance to another atom in angstroms.
    pub fn distance_to(&self, other: &AtomRecord) -> f64 {
        let dx = self.coords[0] - other.coords[0];
        let dy = self.coords[1] - other.coords[1];
        let dz = self.coords[2] - other.coords[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// True if this is a backbone atom (N, CA, C, O).
    pub fn is_backbone(&self) -> bool {
        let trimmed = self.name.trim();
        matches!(trimmed, "N" | "CA" | "C" | "O")
    }

    /// True if this is a C-alpha atom.
    pub fn is_ca(&self) -> bool {
        self.name.trim() == "CA"
    }

    /// True if this atom belongs to a water molecule.
    pub fn is_water(&self) -> bool {
        let r = self.res_name.trim();
        r == "HOH" || r == "WAT" || r == "H2O"
    }
}

impl fmt::Display for AtomRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{:>5} {:4} {:3} {}{:>4} ({:.3},{:.3},{:.3})",
            if self.is_hetatm { "HETATM" } else { "ATOM  " },
            self.serial,
            self.name,
            self.res_name,
            self.chain,
            self.res_seq,
            self.coords[0], self.coords[1], self.coords[2],
        )
    }
}

// ── Chain ───────────────────────────────────────────────────────────

/// A chain extracted from a PDB structure.
#[derive(Debug, Clone)]
pub struct Chain {
    /// Chain identifier.
    pub id: char,
    /// Atom records in this chain.
    pub atoms: Vec<AtomRecord>,
}

impl Chain {
    pub fn new(id: char) -> Self {
        Self { id, atoms: Vec::new() }
    }

    /// Number of atoms.
    pub fn atom_count(&self) -> usize {
        self.atoms.len()
    }

    /// Number of unique residues.
    pub fn residue_count(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        for a in &self.atoms {
            if !a.is_hetatm {
                seen.insert(a.res_seq);
            }
        }
        seen.len()
    }

    /// Extract C-alpha atoms.
    pub fn ca_atoms(&self) -> Vec<&AtomRecord> {
        self.atoms.iter().filter(|a| a.is_ca() && !a.is_hetatm).collect()
    }

    /// Extract C-alpha coordinates.
    pub fn ca_coordinates(&self) -> Vec<[f64; 3]> {
        self.ca_atoms().iter().map(|a| a.coords).collect()
    }

    /// Average B-factor across all atoms.
    pub fn average_b_factor(&self) -> f64 {
        if self.atoms.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.atoms.iter().map(|a| a.b_factor).sum();
        sum / self.atoms.len() as f64
    }
}

impl fmt::Display for Chain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Chain {} ({} atoms, {} residues)", self.id, self.atom_count(), self.residue_count())
    }
}

// ── PDB Structure ───────────────────────────────────────────────────

/// A parsed PDB structure (single model).
#[derive(Debug, Clone)]
pub struct PdbStructure {
    /// Header classification.
    pub classification: String,
    /// PDB identifier.
    pub pdb_id: String,
    /// All atom records.
    pub atoms: Vec<AtomRecord>,
    /// HETATM records.
    pub hetatms: Vec<AtomRecord>,
    /// Model number (1-based, 0 if no MODEL record).
    pub model_num: u32,
    /// Remark lines.
    pub remarks: Vec<String>,
}

impl PdbStructure {
    pub fn new() -> Self {
        Self {
            classification: String::new(),
            pdb_id: String::new(),
            atoms: Vec::new(),
            hetatms: Vec::new(),
            model_num: 0,
            remarks: Vec::new(),
        }
    }

    /// Total atom count (ATOM + HETATM).
    pub fn total_atoms(&self) -> usize {
        self.atoms.len() + self.hetatms.len()
    }

    /// Extract chains from ATOM records.
    pub fn chains(&self) -> Vec<Chain> {
        let mut chain_map: std::collections::BTreeMap<char, Vec<AtomRecord>> =
            std::collections::BTreeMap::new();
        for a in &self.atoms {
            chain_map.entry(a.chain).or_default().push(a.clone());
        }
        chain_map.into_iter().map(|(id, atoms)| Chain { id, atoms }).collect()
    }

    /// Get all C-alpha coordinates across all chains.
    pub fn all_ca_coordinates(&self) -> Vec<[f64; 3]> {
        self.atoms.iter().filter(|a| a.is_ca()).map(|a| a.coords).collect()
    }

    /// Center of mass of all ATOM records.
    pub fn center_of_mass(&self) -> [f64; 3] {
        if self.atoms.is_empty() {
            return [0.0, 0.0, 0.0];
        }
        let n = self.atoms.len() as f64;
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        for a in &self.atoms {
            cx += a.coords[0];
            cy += a.coords[1];
            cz += a.coords[2];
        }
        [cx / n, cy / n, cz / n]
    }

    /// Bounding box: ([min_x, min_y, min_z], [max_x, max_y, max_z]).
    pub fn bounding_box(&self) -> ([f64; 3], [f64; 3]) {
        if self.atoms.is_empty() {
            return ([0.0; 3], [0.0; 3]);
        }
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for a in &self.atoms {
            for k in 0..3 {
                lo[k] = lo[k].min(a.coords[k]);
                hi[k] = hi[k].max(a.coords[k]);
            }
        }
        (lo, hi)
    }

    /// Count non-water HETATM records.
    pub fn ligand_count(&self) -> usize {
        self.hetatms.iter().filter(|a| !a.is_water()).count()
    }

    /// Count water molecules.
    pub fn water_count(&self) -> usize {
        self.hetatms.iter().filter(|a| a.is_water()).count()
    }
}

impl fmt::Display for PdbStructure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PDB({}, atoms={}, hetatm={}, chains={})",
            if self.pdb_id.is_empty() { "????" } else { &self.pdb_id },
            self.atoms.len(),
            self.hetatms.len(),
            self.chains().len(),
        )
    }
}

// ── Parser ──────────────────────────────────────────────────────────

/// PDB format parser configuration.
#[derive(Debug, Clone)]
pub struct PdbParser {
    /// If true, only parse the first MODEL.
    first_model_only: bool,
    /// If true, skip HETATM records.
    skip_hetatm: bool,
    /// If true, skip hydrogen atoms.
    skip_hydrogens: bool,
}

impl PdbParser {
    pub fn new() -> Self {
        Self { first_model_only: true, skip_hetatm: false, skip_hydrogens: false }
    }

    pub fn with_all_models(mut self) -> Self {
        self.first_model_only = false;
        self
    }

    pub fn with_skip_hetatm(mut self, skip: bool) -> Self {
        self.skip_hetatm = skip;
        self
    }

    pub fn with_skip_hydrogens(mut self, skip: bool) -> Self {
        self.skip_hydrogens = skip;
        self
    }

    /// Parse a PDB-format string.
    pub fn parse(&self, content: &str) -> Result<PdbStructure, PdbError> {
        let mut structure = PdbStructure::new();
        let mut in_model = false;
        let mut model_done = false;

        for line in content.lines() {
            if model_done && self.first_model_only {
                break;
            }

            let record_type = if line.len() >= 6 { &line[..6] } else { line };

            match record_type.trim() {
                "HEADER" => {
                    if line.len() >= 66 {
                        structure.classification = line[10..50].trim().to_string();
                        structure.pdb_id = line[62..66].trim().to_string();
                    }
                }
                "MODEL" => {
                    in_model = true;
                    if line.len() >= 14 {
                        structure.model_num = line[6..14].trim().parse().unwrap_or(1);
                    }
                }
                "ENDMDL" => {
                    model_done = true;
                    in_model = !self.first_model_only;
                }
                "ATOM" => {
                    if let Some(atom) = self.parse_atom_line(line, false)? {
                        structure.atoms.push(atom);
                    }
                }
                "HETATM" => {
                    if !self.skip_hetatm {
                        if let Some(atom) = self.parse_atom_line(line, true)? {
                            structure.hetatms.push(atom);
                        }
                    }
                }
                "REMARK" => {
                    structure.remarks.push(line.to_string());
                }
                _ => {}
            }
            let _ = in_model; // suppress unused warning
        }

        Ok(structure)
    }

    fn parse_atom_line(&self, line: &str, is_hetatm: bool) -> Result<Option<AtomRecord>, PdbError> {
        if line.len() < 54 {
            return Err(PdbError::ShortLine(line.len()));
        }

        let name = line[12..16].to_string();
        let element = if line.len() >= 78 {
            line[76..78].trim().to_string()
        } else {
            name.trim().chars().next().map(|c| c.to_string()).unwrap_or_default()
        };

        if self.skip_hydrogens && (element == "H" || element == "D") {
            return Ok(None);
        }

        let serial = line[6..11].trim().parse::<u32>()
            .map_err(|_| PdbError::ParseField("serial"))?;
        let alt_loc = line.as_bytes().get(16).map(|b| *b as char).unwrap_or(' ');
        let res_name = line[17..20].to_string();
        let chain = line.as_bytes().get(21).map(|b| *b as char).unwrap_or(' ');
        let res_seq = line[22..26].trim().parse::<i32>()
            .map_err(|_| PdbError::ParseField("res_seq"))?;
        let ins_code = line.as_bytes().get(26).map(|b| *b as char).unwrap_or(' ');

        let x = line[30..38].trim().parse::<f64>()
            .map_err(|_| PdbError::ParseField("x"))?;
        let y = line[38..46].trim().parse::<f64>()
            .map_err(|_| PdbError::ParseField("y"))?;
        let z = line[46..54].trim().parse::<f64>()
            .map_err(|_| PdbError::ParseField("z"))?;

        let occupancy = if line.len() >= 60 {
            line[54..60].trim().parse::<f64>().unwrap_or(1.0)
        } else {
            1.0
        };

        let b_factor = if line.len() >= 66 {
            line[60..66].trim().parse::<f64>().unwrap_or(0.0)
        } else {
            0.0
        };

        Ok(Some(AtomRecord {
            serial, name, alt_loc, res_name, chain, res_seq, ins_code,
            coords: [x, y, z], occupancy, b_factor, element, is_hetatm,
        }))
    }
}

impl fmt::Display for PdbParser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PdbParser(first_model={}, skip_het={}, skip_H={})",
            self.first_model_only, self.skip_hetatm, self.skip_hydrogens,
        )
    }
}

// ── Error ───────────────────────────────────────────────────────────

/// PDB parsing error.
#[derive(Debug, Clone)]
pub enum PdbError {
    ShortLine(usize),
    ParseField(&'static str),
}

impl fmt::Display for PdbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShortLine(len) => write!(f, "PDB line too short ({} chars)", len),
            Self::ParseField(field) => write!(f, "Failed to parse PDB field: {}", field),
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_pdb() -> &'static str {
        concat!(
            "HEADER    HYDROLASE                               01-JAN-00   1ABC\n",
            "REMARK   1 TEST STRUCTURE\n",
            "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N\n",
            "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 12.00           C\n",
            "ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00 11.00           C\n",
            "ATOM      4  O   ALA A   1       4.000   5.000   6.000  1.00 13.00           O\n",
            "ATOM      5  N   GLY A   2       5.000   6.000   7.000  1.00  9.00           N\n",
            "ATOM      6  CA  GLY A   2       6.000   7.000   8.000  1.00 11.00           C\n",
            "HETATM    7  O   HOH A 100      10.000  11.000  12.000  1.00 20.00           O\n",
            "HETATM    8  C1  LIG B 200      15.000  16.000  17.000  1.00  5.00           C\n",
            "END\n",
        )
    }

    #[test]
    fn test_parse_basic() {
        let parser = PdbParser::new();
        let pdb = parser.parse(sample_pdb()).unwrap();
        assert_eq!(pdb.pdb_id, "1ABC");
        assert_eq!(pdb.atoms.len(), 6);
        assert_eq!(pdb.hetatms.len(), 2);
    }

    #[test]
    fn test_parse_header() {
        let parser = PdbParser::new();
        let pdb = parser.parse(sample_pdb()).unwrap();
        assert!(pdb.classification.contains("HYDROLASE"));
    }

    #[test]
    fn test_atom_coordinates() {
        let parser = PdbParser::new();
        let pdb = parser.parse(sample_pdb()).unwrap();
        let ca = &pdb.atoms[1];
        assert!((ca.coords[0] - 2.0).abs() < 1e-3);
        assert!((ca.coords[1] - 3.0).abs() < 1e-3);
        assert!((ca.coords[2] - 4.0).abs() < 1e-3);
    }

    #[test]
    fn test_atom_is_backbone() {
        let parser = PdbParser::new();
        let pdb = parser.parse(sample_pdb()).unwrap();
        assert!(pdb.atoms[0].is_backbone()); // N
        assert!(pdb.atoms[1].is_backbone()); // CA
    }

    #[test]
    fn test_atom_is_ca() {
        let parser = PdbParser::new();
        let pdb = parser.parse(sample_pdb()).unwrap();
        assert!(pdb.atoms[1].is_ca());
        assert!(!pdb.atoms[0].is_ca());
    }

    #[test]
    fn test_atom_distance() {
        let parser = PdbParser::new();
        let pdb = parser.parse(sample_pdb()).unwrap();
        let d = pdb.atoms[0].distance_to(&pdb.atoms[1]);
        // sqrt(1+1+1) = sqrt(3)
        assert!((d - 3.0_f64.sqrt()).abs() < 1e-3);
    }

    #[test]
    fn test_chains() {
        let parser = PdbParser::new();
        let pdb = parser.parse(sample_pdb()).unwrap();
        let chains = pdb.chains();
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].id, 'A');
        assert_eq!(chains[0].residue_count(), 2);
    }

    #[test]
    fn test_ca_coordinates() {
        let parser = PdbParser::new();
        let pdb = parser.parse(sample_pdb()).unwrap();
        let ca_coords = pdb.all_ca_coordinates();
        assert_eq!(ca_coords.len(), 2);
    }

    #[test]
    fn test_center_of_mass() {
        let parser = PdbParser::new();
        let pdb = parser.parse(sample_pdb()).unwrap();
        let com = pdb.center_of_mass();
        // Mean of x: (1+2+3+4+5+6)/6 = 3.5
        assert!((com[0] - 3.5).abs() < 1e-3);
    }

    #[test]
    fn test_bounding_box() {
        let parser = PdbParser::new();
        let pdb = parser.parse(sample_pdb()).unwrap();
        let (lo, hi) = pdb.bounding_box();
        assert!((lo[0] - 1.0).abs() < 1e-3);
        assert!((hi[0] - 6.0).abs() < 1e-3);
    }

    #[test]
    fn test_water_and_ligand_counts() {
        let parser = PdbParser::new();
        let pdb = parser.parse(sample_pdb()).unwrap();
        assert_eq!(pdb.water_count(), 1);
        assert_eq!(pdb.ligand_count(), 1);
    }

    #[test]
    fn test_skip_hetatm() {
        let parser = PdbParser::new().with_skip_hetatm(true);
        let pdb = parser.parse(sample_pdb()).unwrap();
        assert_eq!(pdb.hetatms.len(), 0);
    }

    #[test]
    fn test_chain_average_b_factor() {
        let parser = PdbParser::new();
        let pdb = parser.parse(sample_pdb()).unwrap();
        let chains = pdb.chains();
        let avg = chains[0].average_b_factor();
        // (10+12+11+13+9+11)/6 = 11.0
        assert!((avg - 11.0).abs() < 1e-3);
    }

    #[test]
    fn test_record_kind_display() {
        assert_eq!(RecordKind::Atom.to_string(), "ATOM");
        assert_eq!(RecordKind::Hetatm.to_string(), "HETATM");
    }

    #[test]
    fn test_parser_display() {
        let p = PdbParser::new();
        assert!(p.to_string().contains("PdbParser"));
    }

    #[test]
    fn test_pdb_structure_display() {
        let parser = PdbParser::new();
        let pdb = parser.parse(sample_pdb()).unwrap();
        assert!(pdb.to_string().contains("1ABC"));
    }

    #[test]
    fn test_error_short_line() {
        let parser = PdbParser::new();
        let result = parser.parse("ATOM  short\n");
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_pdb() {
        let parser = PdbParser::new();
        let pdb = parser.parse("").unwrap();
        assert_eq!(pdb.total_atoms(), 0);
    }

    #[test]
    fn test_hetatm_is_water() {
        let parser = PdbParser::new();
        let pdb = parser.parse(sample_pdb()).unwrap();
        assert!(pdb.hetatms[0].is_water());
        assert!(!pdb.hetatms[1].is_water());
    }
}
