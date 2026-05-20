//! Register allocation — linear scan allocator, live interval computation,
//! register assignment, spill handling, register classes, interference graph,
//! allocation statistics, instruction rewriting.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

// ── Register classes ────────────────────────────────────────────────────────

/// A class of physical registers (e.g. general-purpose, floating-point).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegisterClass {
    /// General-purpose integer registers.
    GeneralPurpose,
    /// Floating-point / SIMD registers.
    FloatingPoint,
    /// Special-purpose (flags, stack pointer, etc.).
    Special,
}

impl fmt::Display for RegisterClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GeneralPurpose => write!(f, "GP"),
            Self::FloatingPoint => write!(f, "FP"),
            Self::Special => write!(f, "SP"),
        }
    }
}

/// A physical register.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PhysReg {
    /// Register identifier (e.g. 0 = r0, 1 = r1).
    pub id: u32,
    /// Name for display (e.g. "rax", "xmm0").
    pub name: String,
    /// Register class.
    pub class: RegisterClass,
}

impl fmt::Display for PhysReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// A virtual register (before allocation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VirtReg(pub u32);

impl fmt::Display for VirtReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

// ── Live intervals ──────────────────────────────────────────────────────────

/// A live interval for a virtual register — [start, end).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveInterval {
    /// Virtual register this interval belongs to.
    pub vreg: VirtReg,
    /// Start point (instruction index, inclusive).
    pub start: u32,
    /// End point (instruction index, exclusive).
    pub end: u32,
    /// Required register class.
    pub class: RegisterClass,
    /// Use positions within the interval (for spill heuristics).
    pub use_positions: Vec<u32>,
}

impl LiveInterval {
    /// Create a new live interval.
    pub fn new(vreg: VirtReg, start: u32, end: u32, class: RegisterClass) -> Self {
        Self {
            vreg,
            start,
            end,
            class,
            use_positions: Vec::new(),
        }
    }

    /// Whether this interval overlaps another.
    pub fn overlaps(&self, other: &LiveInterval) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Length of the interval.
    pub fn length(&self) -> u32 {
        self.end.saturating_sub(self.start)
    }

    /// Add a use position.
    pub fn add_use(&mut self, pos: u32) {
        if !self.use_positions.contains(&pos) {
            self.use_positions.push(pos);
            self.use_positions.sort();
        }
    }

    /// Next use after a given point (for spill heuristics).
    pub fn next_use_after(&self, pos: u32) -> Option<u32> {
        self.use_positions.iter().copied().find(|u| *u > pos)
    }
}

// ── Interference graph ──────────────────────────────────────────────────────

/// Interference graph — undirected graph where edges mean two vregs are
/// simultaneously live and cannot share a physical register.
#[derive(Debug, Clone)]
pub struct InterferenceGraph {
    /// Adjacency sets: vreg -> set of interfering vregs.
    adjacency: HashMap<VirtReg, HashSet<VirtReg>>,
}

impl InterferenceGraph {
    /// Create an empty interference graph.
    pub fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
        }
    }

    /// Build the interference graph from live intervals.
    pub fn build(intervals: &[LiveInterval]) -> Self {
        let mut graph = Self::new();
        for iv in intervals {
            graph.adjacency.entry(iv.vreg).or_default();
        }
        for i in 0..intervals.len() {
            for j in (i + 1)..intervals.len() {
                if intervals[i].overlaps(&intervals[j])
                    && intervals[i].class == intervals[j].class
                {
                    graph.add_edge(intervals[i].vreg, intervals[j].vreg);
                }
            }
        }
        graph
    }

    /// Add an interference edge.
    pub fn add_edge(&mut self, a: VirtReg, b: VirtReg) {
        self.adjacency.entry(a).or_default().insert(b);
        self.adjacency.entry(b).or_default().insert(a);
    }

    /// Whether two vregs interfere.
    pub fn interferes(&self, a: VirtReg, b: VirtReg) -> bool {
        self.adjacency
            .get(&a)
            .map_or(false, |s| s.contains(&b))
    }

    /// Degree of a vreg (number of interferences).
    pub fn degree(&self, v: VirtReg) -> usize {
        self.adjacency.get(&v).map_or(0, |s| s.len())
    }

    /// All vregs in the graph.
    pub fn vregs(&self) -> Vec<VirtReg> {
        let mut v: Vec<_> = self.adjacency.keys().copied().collect();
        v.sort();
        v
    }

    /// Neighbors of a vreg.
    pub fn neighbors(&self, v: VirtReg) -> Vec<VirtReg> {
        let mut n: Vec<_> = self.adjacency.get(&v).map_or(Vec::new(), |s| {
            s.iter().copied().collect()
        });
        n.sort();
        n
    }

    /// Total number of edges (each edge counted once).
    pub fn edge_count(&self) -> usize {
        let total: usize = self.adjacency.values().map(|s| s.len()).sum();
        total / 2
    }
}

impl Default for InterferenceGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ── Instructions (simplified) ───────────────────────────────────────────────

/// An operand in an instruction — either virtual or physical.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operand {
    /// Virtual register (pre-allocation).
    Virtual(VirtReg),
    /// Physical register (post-allocation).
    Physical(u32),
    /// Immediate / constant.
    Immediate(i64),
    /// Stack slot (spill).
    StackSlot(u32),
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Virtual(v) => write!(f, "{v}"),
            Self::Physical(id) => write!(f, "r{id}"),
            Self::Immediate(val) => write!(f, "#{val}"),
            Self::StackSlot(slot) => write!(f, "[sp+{slot}]"),
        }
    }
}

/// A simplified instruction for register allocation purposes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instruction {
    /// Instruction index (program point).
    pub index: u32,
    /// Opcode name.
    pub opcode: String,
    /// Defined (written) operands.
    pub defs: Vec<Operand>,
    /// Used (read) operands.
    pub uses: Vec<Operand>,
}

impl Instruction {
    /// Create a new instruction.
    pub fn new(index: u32, opcode: &str, defs: Vec<Operand>, uses: Vec<Operand>) -> Self {
        Self {
            index,
            opcode: opcode.to_string(),
            defs,
            uses,
        }
    }

    /// Collect all virtual registers referenced.
    pub fn vregs(&self) -> Vec<VirtReg> {
        let mut result = Vec::new();
        for op in self.defs.iter().chain(self.uses.iter()) {
            if let Operand::Virtual(v) = op {
                if !result.contains(v) {
                    result.push(*v);
                }
            }
        }
        result
    }
}

// ── Live interval computation ───────────────────────────────────────────────

/// Compute live intervals from a sequence of instructions.
/// Returns intervals sorted by start point.
pub fn compute_live_intervals(
    instructions: &[Instruction],
    classes: &HashMap<VirtReg, RegisterClass>,
) -> Vec<LiveInterval> {
    let mut intervals: BTreeMap<VirtReg, LiveInterval> = BTreeMap::new();

    for instr in instructions {
        // Process defs
        for op in &instr.defs {
            if let Operand::Virtual(v) = op {
                let class = classes.get(v).copied().unwrap_or(RegisterClass::GeneralPurpose);
                intervals.entry(*v).or_insert_with(|| {
                    LiveInterval::new(*v, instr.index, instr.index + 1, class)
                });
                // Def starts the interval if not yet started
                let iv = intervals.get_mut(v).unwrap();
                if instr.index < iv.start {
                    iv.start = instr.index;
                }
            }
        }

        // Process uses
        for op in &instr.uses {
            if let Operand::Virtual(v) = op {
                let class = classes.get(v).copied().unwrap_or(RegisterClass::GeneralPurpose);
                let iv = intervals.entry(*v).or_insert_with(|| {
                    LiveInterval::new(*v, instr.index, instr.index + 1, class)
                });
                // Extend end to cover this use
                if instr.index + 1 > iv.end {
                    iv.end = instr.index + 1;
                }
                iv.add_use(instr.index);
            }
        }
    }

    let mut result: Vec<_> = intervals.into_values().collect();
    result.sort_by_key(|iv| iv.start);
    result
}

// ── Allocation result ───────────────────────────────────────────────────────

/// The result of allocating a single virtual register.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Allocation {
    /// Assigned to a physical register.
    Register(u32),
    /// Spilled to a stack slot.
    Spill(u32),
}

impl fmt::Display for Allocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Register(id) => write!(f, "r{id}"),
            Self::Spill(slot) => write!(f, "[sp+{slot}]"),
        }
    }
}

// ── Allocation statistics ───────────────────────────────────────────────────

/// Statistics from an allocation run.
#[derive(Debug, Clone, Default)]
pub struct AllocStats {
    /// Total virtual registers processed.
    pub total_vregs: u32,
    /// Vregs assigned to physical registers.
    pub assigned: u32,
    /// Vregs spilled to stack.
    pub spilled: u32,
    /// Maximum number of registers active at once.
    pub max_active: u32,
    /// Number of spill loads inserted.
    pub spill_loads: u32,
    /// Number of spill stores inserted.
    pub spill_stores: u32,
}

impl AllocStats {
    /// Spill ratio (0.0 = no spills, 1.0 = all spilled).
    pub fn spill_ratio(&self) -> f64 {
        if self.total_vregs == 0 {
            return 0.0;
        }
        self.spilled as f64 / self.total_vregs as f64
    }
}

impl fmt::Display for AllocStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "vregs={}, assigned={}, spilled={}, max_active={}, loads={}, stores={}",
            self.total_vregs,
            self.assigned,
            self.spilled,
            self.max_active,
            self.spill_loads,
            self.spill_stores,
        )
    }
}

// ── Linear scan allocator ───────────────────────────────────────────────────

/// Linear-scan register allocator.
///
/// Processes live intervals sorted by start point. Maintains an "active" list
/// of intervals currently occupying registers. When all registers are busy,
/// spills the interval with the farthest next use.
pub struct LinearScanAllocator {
    /// Available physical registers per class.
    registers: HashMap<RegisterClass, Vec<u32>>,
    /// Current allocation map: vreg -> allocation.
    allocation: HashMap<VirtReg, Allocation>,
    /// Active intervals, sorted by end point.
    active: Vec<LiveInterval>,
    /// Free registers per class (stack — LIFO).
    free_regs: HashMap<RegisterClass, Vec<u32>>,
    /// Next stack slot for spills.
    next_slot: u32,
    /// Statistics.
    stats: AllocStats,
}

impl LinearScanAllocator {
    /// Create a new allocator with the given physical register sets.
    pub fn new(registers: HashMap<RegisterClass, Vec<u32>>) -> Self {
        let free_regs = registers.clone();
        Self {
            registers,
            allocation: HashMap::new(),
            active: Vec::new(),
            free_regs,
            next_slot: 0,
            stats: AllocStats::default(),
        }
    }

    /// Create an allocator with `n` general-purpose and `m` floating-point registers.
    pub fn with_counts(gp_count: u32, fp_count: u32) -> Self {
        let mut registers = HashMap::new();
        let gp: Vec<u32> = (0..gp_count).collect();
        let fp: Vec<u32> = (gp_count..gp_count + fp_count).collect();
        registers.insert(RegisterClass::GeneralPurpose, gp);
        registers.insert(RegisterClass::FloatingPoint, fp);
        Self::new(registers)
    }

    /// Run allocation on the given live intervals. Returns the allocation map.
    pub fn allocate(&mut self, intervals: &[LiveInterval]) -> HashMap<VirtReg, Allocation> {
        // Reset state
        self.allocation.clear();
        self.active.clear();
        self.free_regs = self.registers.clone();
        self.next_slot = 0;
        self.stats = AllocStats::default();

        // Sort by start point (should already be, but ensure).
        let mut sorted: Vec<LiveInterval> = intervals.to_vec();
        sorted.sort_by_key(|iv| iv.start);

        for interval in &sorted {
            self.stats.total_vregs += 1;

            // Expire old intervals
            self.expire_old(interval.start);

            // Track max active
            let active_count = self.active.len() as u32 + 1;
            if active_count > self.stats.max_active {
                self.stats.max_active = active_count;
            }

            // Try to allocate a free register
            let class = interval.class;
            let free = self.free_regs.entry(class).or_default();

            if let Some(reg) = free.pop() {
                // Assign register
                self.allocation
                    .insert(interval.vreg, Allocation::Register(reg));
                self.active.push(interval.clone());
                self.active.sort_by_key(|iv| iv.end);
                self.stats.assigned += 1;
            } else {
                // Must spill — choose the interval with the farthest end
                self.spill_at_interval(interval);
            }
        }

        self.allocation.clone()
    }

    /// Expire intervals that end before the given point.
    fn expire_old(&mut self, point: u32) {
        let mut i = 0;
        while i < self.active.len() {
            if self.active[i].end <= point {
                let expired = self.active.remove(i);
                // Return the register to the free pool
                if let Some(Allocation::Register(reg)) = self.allocation.get(&expired.vreg) {
                    self.free_regs
                        .entry(expired.class)
                        .or_default()
                        .push(*reg);
                }
            } else {
                i += 1;
            }
        }
    }

    /// Spill: either the current interval or the active one with the farthest end.
    fn spill_at_interval(&mut self, current: &LiveInterval) {
        // Find the active interval with the farthest end in the same class
        let candidate_idx = self.active.iter().enumerate()
            .filter(|(_, a)| a.class == current.class)
            .max_by_key(|(_, a)| a.end)
            .map(|(idx, _)| idx);

        if let Some(idx) = candidate_idx {
            let farthest_end = self.active[idx].end;
            if farthest_end > current.end {
                // Spill the active one, give its register to current
                let spilled_active = self.active.remove(idx);
                let reg = match self.allocation.get(&spilled_active.vreg) {
                    Some(Allocation::Register(r)) => *r,
                    _ => {
                        // Shouldn't happen, but spill current as fallback
                        self.do_spill(current.vreg);
                        return;
                    }
                };
                // Re-assign the spilled one to a stack slot
                let slot = self.next_slot;
                self.next_slot += 1;
                self.allocation
                    .insert(spilled_active.vreg, Allocation::Spill(slot));
                self.stats.spilled += 1;
                self.stats.assigned -= 1; // was counted as assigned

                // Give the register to current
                self.allocation
                    .insert(current.vreg, Allocation::Register(reg));
                self.active.push(current.clone());
                self.active.sort_by_key(|iv| iv.end);
                self.stats.assigned += 1;
                return;
            }
        }

        // Spill the current interval
        self.do_spill(current.vreg);
    }

    fn do_spill(&mut self, vreg: VirtReg) {
        let slot = self.next_slot;
        self.next_slot += 1;
        self.allocation.insert(vreg, Allocation::Spill(slot));
        self.stats.spilled += 1;
    }

    /// Get allocation statistics.
    pub fn statistics(&self) -> &AllocStats {
        &self.stats
    }

    /// Get the allocation for a specific vreg.
    pub fn get_allocation(&self, vreg: VirtReg) -> Option<&Allocation> {
        self.allocation.get(&vreg)
    }
}

// ── Instruction rewriting ───────────────────────────────────────────────────

/// Rewrite instructions by replacing virtual registers with their allocations.
/// Inserts spill loads/stores as needed.
pub fn rewrite_instructions(
    instructions: &[Instruction],
    allocation: &HashMap<VirtReg, Allocation>,
) -> (Vec<Instruction>, AllocStats) {
    let mut result = Vec::new();
    let mut stats = AllocStats::default();
    let mut next_index = instructions.last().map_or(0, |i| i.index + 100);

    for instr in instructions {
        // Insert loads before the instruction for spilled uses
        for op in &instr.uses {
            if let Operand::Virtual(v) = op {
                if let Some(Allocation::Spill(slot)) = allocation.get(v) {
                    result.push(Instruction::new(
                        next_index,
                        "spill_load",
                        vec![Operand::Virtual(*v)],
                        vec![Operand::StackSlot(*slot)],
                    ));
                    next_index += 1;
                    stats.spill_loads += 1;
                }
            }
        }

        // Rewrite the instruction operands
        let new_defs: Vec<Operand> = instr
            .defs
            .iter()
            .map(|op| rewrite_operand(op, allocation))
            .collect();
        let new_uses: Vec<Operand> = instr
            .uses
            .iter()
            .map(|op| rewrite_operand(op, allocation))
            .collect();

        result.push(Instruction::new(
            instr.index,
            &instr.opcode,
            new_defs,
            new_uses,
        ));

        // Insert stores after the instruction for spilled defs
        for op in &instr.defs {
            if let Operand::Virtual(v) = op {
                if let Some(Allocation::Spill(slot)) = allocation.get(v) {
                    result.push(Instruction::new(
                        next_index,
                        "spill_store",
                        vec![Operand::StackSlot(*slot)],
                        vec![Operand::Virtual(*v)],
                    ));
                    next_index += 1;
                    stats.spill_stores += 1;
                }
            }
        }
    }

    (result, stats)
}

/// Map a single operand according to the allocation.
fn rewrite_operand(op: &Operand, allocation: &HashMap<VirtReg, Allocation>) -> Operand {
    match op {
        Operand::Virtual(v) => match allocation.get(v) {
            Some(Allocation::Register(r)) => Operand::Physical(*r),
            Some(Allocation::Spill(s)) => Operand::StackSlot(*s),
            None => op.clone(),
        },
        other => other.clone(),
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn gp_interval(vreg: u32, start: u32, end: u32) -> LiveInterval {
        LiveInterval::new(VirtReg(vreg), start, end, RegisterClass::GeneralPurpose)
    }

    fn fp_interval(vreg: u32, start: u32, end: u32) -> LiveInterval {
        LiveInterval::new(VirtReg(vreg), start, end, RegisterClass::FloatingPoint)
    }

    #[test]
    fn test_live_interval_overlap() {
        let a = gp_interval(0, 0, 10);
        let b = gp_interval(1, 5, 15);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_live_interval_no_overlap() {
        let a = gp_interval(0, 0, 5);
        let b = gp_interval(1, 5, 10);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_live_interval_length() {
        let a = gp_interval(0, 3, 10);
        assert_eq!(a.length(), 7);
    }

    #[test]
    fn test_live_interval_next_use() {
        let mut iv = gp_interval(0, 0, 20);
        iv.add_use(3);
        iv.add_use(7);
        iv.add_use(15);
        assert_eq!(iv.next_use_after(5), Some(7));
        assert_eq!(iv.next_use_after(15), None);
        assert_eq!(iv.next_use_after(0), Some(3));
    }

    #[test]
    fn test_interference_graph_basic() {
        let intervals = vec![gp_interval(0, 0, 10), gp_interval(1, 5, 15)];
        let graph = InterferenceGraph::build(&intervals);
        assert!(graph.interferes(VirtReg(0), VirtReg(1)));
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_interference_graph_no_overlap() {
        let intervals = vec![gp_interval(0, 0, 5), gp_interval(1, 5, 10)];
        let graph = InterferenceGraph::build(&intervals);
        assert!(!graph.interferes(VirtReg(0), VirtReg(1)));
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_interference_graph_degree() {
        let intervals = vec![
            gp_interval(0, 0, 10),
            gp_interval(1, 5, 15),
            gp_interval(2, 8, 20),
        ];
        let graph = InterferenceGraph::build(&intervals);
        // v0=[0,10) overlaps v1=[5,15) and v2=[8,20) (8 < 10)
        assert_eq!(graph.degree(VirtReg(0)), 2); // overlaps 1, 2
        assert_eq!(graph.degree(VirtReg(1)), 2); // overlaps 0 and 2
        assert_eq!(graph.degree(VirtReg(2)), 2); // overlaps 0 and 1
    }

    #[test]
    fn test_interference_graph_different_classes() {
        // GP and FP don't interfere even if overlapping
        let intervals = vec![gp_interval(0, 0, 10), fp_interval(1, 0, 10)];
        let graph = InterferenceGraph::build(&intervals);
        assert!(!graph.interferes(VirtReg(0), VirtReg(1)));
    }

    #[test]
    fn test_simple_allocation_no_spills() {
        let intervals = vec![
            gp_interval(0, 0, 5),
            gp_interval(1, 5, 10),
            gp_interval(2, 10, 15),
        ];
        let mut alloc = LinearScanAllocator::with_counts(2, 0);
        let result = alloc.allocate(&intervals);

        // All should get registers since they don't overlap
        for vreg_id in 0..3 {
            assert!(matches!(
                result.get(&VirtReg(vreg_id)),
                Some(Allocation::Register(_))
            ));
        }
        assert_eq!(alloc.statistics().spilled, 0);
    }

    #[test]
    fn test_allocation_with_spill() {
        // 3 overlapping intervals but only 2 registers
        let intervals = vec![
            gp_interval(0, 0, 10),
            gp_interval(1, 2, 8),
            gp_interval(2, 4, 12),
        ];
        let mut alloc = LinearScanAllocator::with_counts(2, 0);
        let result = alloc.allocate(&intervals);

        // One must be spilled
        let spilled: Vec<_> = result
            .values()
            .filter(|a| matches!(a, Allocation::Spill(_)))
            .collect();
        assert_eq!(spilled.len(), 1);
        assert_eq!(alloc.statistics().spilled, 1);
        assert_eq!(alloc.statistics().assigned, 2);
    }

    #[test]
    fn test_allocation_reuses_expired_registers() {
        let intervals = vec![
            gp_interval(0, 0, 3),
            gp_interval(1, 3, 6),
            gp_interval(2, 6, 9),
        ];
        let mut alloc = LinearScanAllocator::with_counts(1, 0);
        let result = alloc.allocate(&intervals);

        // All get the same register since they don't overlap
        for vreg_id in 0..3 {
            assert!(matches!(
                result.get(&VirtReg(vreg_id)),
                Some(Allocation::Register(_))
            ));
        }
        assert_eq!(alloc.statistics().spilled, 0);
    }

    #[test]
    fn test_allocation_mixed_classes() {
        let intervals = vec![
            gp_interval(0, 0, 10),
            fp_interval(1, 0, 10),
        ];
        let mut alloc = LinearScanAllocator::with_counts(1, 1);
        let result = alloc.allocate(&intervals);

        // Both get registers (different classes)
        assert!(matches!(
            result.get(&VirtReg(0)),
            Some(Allocation::Register(_))
        ));
        assert!(matches!(
            result.get(&VirtReg(1)),
            Some(Allocation::Register(_))
        ));
    }

    #[test]
    fn test_compute_live_intervals() {
        let instructions = vec![
            Instruction::new(0, "mov", vec![Operand::Virtual(VirtReg(0))], vec![Operand::Immediate(42)]),
            Instruction::new(1, "mov", vec![Operand::Virtual(VirtReg(1))], vec![Operand::Immediate(10)]),
            Instruction::new(2, "add", vec![Operand::Virtual(VirtReg(2))], vec![
                Operand::Virtual(VirtReg(0)),
                Operand::Virtual(VirtReg(1)),
            ]),
        ];
        let classes = HashMap::new();
        let intervals = compute_live_intervals(&instructions, &classes);

        assert_eq!(intervals.len(), 3);
        // v0 defined at 0, used at 2
        let iv0 = intervals.iter().find(|iv| iv.vreg == VirtReg(0)).unwrap();
        assert_eq!(iv0.start, 0);
        assert_eq!(iv0.end, 3);
    }

    #[test]
    fn test_rewrite_instructions_register() {
        let instructions = vec![Instruction::new(
            0,
            "add",
            vec![Operand::Virtual(VirtReg(2))],
            vec![Operand::Virtual(VirtReg(0)), Operand::Virtual(VirtReg(1))],
        )];
        let mut alloc_map = HashMap::new();
        alloc_map.insert(VirtReg(0), Allocation::Register(0));
        alloc_map.insert(VirtReg(1), Allocation::Register(1));
        alloc_map.insert(VirtReg(2), Allocation::Register(2));

        let (rewritten, _stats) = rewrite_instructions(&instructions, &alloc_map);
        assert_eq!(rewritten.len(), 1);
        assert_eq!(rewritten[0].defs, vec![Operand::Physical(2)]);
        assert_eq!(
            rewritten[0].uses,
            vec![Operand::Physical(0), Operand::Physical(1)]
        );
    }

    #[test]
    fn test_rewrite_instructions_with_spills() {
        let instructions = vec![Instruction::new(
            0,
            "add",
            vec![Operand::Virtual(VirtReg(2))],
            vec![Operand::Virtual(VirtReg(0))],
        )];
        let mut alloc_map = HashMap::new();
        alloc_map.insert(VirtReg(0), Allocation::Spill(0));
        alloc_map.insert(VirtReg(2), Allocation::Spill(1));

        let (rewritten, stats) = rewrite_instructions(&instructions, &alloc_map);
        // Should have: load before, instruction, store after
        assert_eq!(rewritten.len(), 3);
        assert_eq!(rewritten[0].opcode, "spill_load");
        assert_eq!(rewritten[1].opcode, "add");
        assert_eq!(rewritten[2].opcode, "spill_store");
        assert_eq!(stats.spill_loads, 1);
        assert_eq!(stats.spill_stores, 1);
    }

    #[test]
    fn test_operand_display() {
        assert_eq!(format!("{}", Operand::Virtual(VirtReg(3))), "v3");
        assert_eq!(format!("{}", Operand::Physical(7)), "r7");
        assert_eq!(format!("{}", Operand::Immediate(42)), "#42");
        assert_eq!(format!("{}", Operand::StackSlot(2)), "[sp+2]");
    }

    #[test]
    fn test_alloc_stats_spill_ratio() {
        let mut stats = AllocStats::default();
        stats.total_vregs = 10;
        stats.spilled = 3;
        let ratio = stats.spill_ratio();
        assert!((ratio - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_alloc_stats_zero_vregs() {
        let stats = AllocStats::default();
        assert_eq!(stats.spill_ratio(), 0.0);
    }

    #[test]
    fn test_instruction_vregs() {
        let instr = Instruction::new(
            0,
            "add",
            vec![Operand::Virtual(VirtReg(2))],
            vec![
                Operand::Virtual(VirtReg(0)),
                Operand::Immediate(1),
                Operand::Virtual(VirtReg(0)),
            ],
        );
        let vregs = instr.vregs();
        assert_eq!(vregs.len(), 2); // v2, v0 (no duplicates)
        assert!(vregs.contains(&VirtReg(0)));
        assert!(vregs.contains(&VirtReg(2)));
    }

    #[test]
    fn test_interference_graph_neighbors() {
        let intervals = vec![
            gp_interval(0, 0, 10),
            gp_interval(1, 5, 15),
            gp_interval(2, 12, 20),
        ];
        let graph = InterferenceGraph::build(&intervals);
        let n = graph.neighbors(VirtReg(1));
        assert!(n.contains(&VirtReg(0)));
        assert!(n.contains(&VirtReg(2)));
    }

    #[test]
    fn test_heavy_spill_pressure() {
        // 5 overlapping intervals, 1 register
        let intervals: Vec<_> = (0..5)
            .map(|i| gp_interval(i, 0, 20))
            .collect();
        let mut alloc = LinearScanAllocator::with_counts(1, 0);
        let result = alloc.allocate(&intervals);

        let assigned: usize = result
            .values()
            .filter(|a| matches!(a, Allocation::Register(_)))
            .count();
        assert_eq!(assigned, 1);
        assert_eq!(alloc.statistics().spilled, 4);
    }

    #[test]
    fn test_register_class_display() {
        assert_eq!(format!("{}", RegisterClass::GeneralPurpose), "GP");
        assert_eq!(format!("{}", RegisterClass::FloatingPoint), "FP");
        assert_eq!(format!("{}", RegisterClass::Special), "SP");
    }
}
