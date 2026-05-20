//! System execution scheduler with dependency ordering for a game-engine ECS.
//!
//! Systems are registered with a run phase (PreUpdate, Update, PostUpdate,
//! Render) and optional dependencies on other systems. The scheduler performs
//! topological sort within each phase and groups independent systems into
//! parallel-ready batches.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Run phase ──

/// The phase in the game loop during which a system executes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RunPhase {
    PreUpdate = 0,
    Update = 1,
    PostUpdate = 2,
    Render = 3,
}

impl RunPhase {
    /// All phases in execution order.
    pub fn all() -> &'static [RunPhase] {
        &[
            RunPhase::PreUpdate,
            RunPhase::Update,
            RunPhase::PostUpdate,
            RunPhase::Render,
        ]
    }
}

// ── System descriptor ──

/// Metadata about a registered system.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemDescriptor {
    pub name: String,
    pub phase: RunPhase,
    pub dependencies: Vec<String>,
    pub enabled: bool,
    /// Priority within the same phase (higher = earlier). Used to break ties.
    pub priority: i32,
}

impl SystemDescriptor {
    /// Create a new system descriptor.
    pub fn new(name: impl Into<String>, phase: RunPhase) -> Self {
        Self {
            name: name.into(),
            phase,
            dependencies: Vec::new(),
            enabled: true,
            priority: 0,
        }
    }

    /// Builder: add a dependency.
    pub fn depends_on(mut self, dep: impl Into<String>) -> Self {
        self.dependencies.push(dep.into());
        self
    }

    /// Builder: set priority.
    pub fn with_priority(mut self, p: i32) -> Self {
        self.priority = p;
        self
    }

    /// Builder: set enabled flag.
    pub fn with_enabled(mut self, e: bool) -> Self {
        self.enabled = e;
        self
    }
}

// ── Parallel batch ──

/// A group of systems that can execute concurrently (no inter-dependencies).
#[derive(Debug, Clone, PartialEq)]
pub struct ParallelBatch {
    pub systems: Vec<String>,
}

// ── Schedule result ──

/// The output of the scheduler: an ordered list of parallel batches per phase.
#[derive(Debug, Clone, PartialEq)]
pub struct Schedule {
    /// Batches ordered by phase, then by dependency order within phase.
    pub batches: Vec<(RunPhase, Vec<ParallelBatch>)>,
}

impl Schedule {
    /// Flatten the schedule into a sequential execution order.
    pub fn sequential_order(&self) -> Vec<&str> {
        let mut result = Vec::new();
        for (_, phase_batches) in &self.batches {
            for batch in phase_batches {
                for sys in &batch.systems {
                    result.push(sys.as_str());
                }
            }
        }
        result
    }

    /// Total number of systems in the schedule.
    pub fn system_count(&self) -> usize {
        self.batches
            .iter()
            .flat_map(|(_, bs)| bs)
            .map(|b| b.systems.len())
            .sum()
    }

    /// Total number of parallel batches.
    pub fn batch_count(&self) -> usize {
        self.batches.iter().map(|(_, bs)| bs.len()).sum()
    }
}

// ── Scheduler error ──

/// Errors that can occur during scheduling.
#[derive(Debug, Clone, PartialEq)]
pub enum ScheduleError {
    /// A dependency cycle was detected.
    CyclicDependency(Vec<String>),
    /// A system depends on a non-existent system.
    MissingDependency {
        system: String,
        missing: String,
    },
    /// A system depends on a system in a later phase.
    CrossPhaseDependency {
        system: String,
        dependency: String,
        system_phase: RunPhase,
        dep_phase: RunPhase,
    },
    /// Duplicate system name.
    DuplicateName(String),
}

impl std::fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CyclicDependency(cycle) => write!(f, "cycle: {}", cycle.join(" -> ")),
            Self::MissingDependency { system, missing } => {
                write!(f, "{system} depends on unknown {missing}")
            }
            Self::CrossPhaseDependency {
                system, dependency, ..
            } => write!(f, "{system} depends on later-phase {dependency}"),
            Self::DuplicateName(n) => write!(f, "duplicate system: {n}"),
        }
    }
}

// ── SystemScheduler ──

/// Builds and schedules systems with dependency ordering.
pub struct SystemScheduler {
    systems: HashMap<String, SystemDescriptor>,
}

impl SystemScheduler {
    pub fn new() -> Self {
        Self {
            systems: HashMap::new(),
        }
    }

    /// Add a system. Returns error on duplicate name.
    pub fn add_system(&mut self, desc: SystemDescriptor) -> Result<(), ScheduleError> {
        if self.systems.contains_key(&desc.name) {
            return Err(ScheduleError::DuplicateName(desc.name.clone()));
        }
        self.systems.insert(desc.name.clone(), desc);
        Ok(())
    }

    /// Remove a system by name.
    pub fn remove_system(&mut self, name: &str) -> bool {
        self.systems.remove(name).is_some()
    }

    /// Enable or disable a system.
    pub fn set_enabled(&mut self, name: &str, enabled: bool) -> bool {
        if let Some(sys) = self.systems.get_mut(name) {
            sys.enabled = enabled;
            true
        } else {
            false
        }
    }

    /// Get a system descriptor by name.
    pub fn get_system(&self, name: &str) -> Option<&SystemDescriptor> {
        self.systems.get(name)
    }

    /// Number of registered systems.
    pub fn system_count(&self) -> usize {
        self.systems.len()
    }

    /// Build the schedule. Only enabled systems are included.
    pub fn build_schedule(&self) -> Result<Schedule, ScheduleError> {
        // Collect enabled systems grouped by phase.
        let enabled: HashMap<&str, &SystemDescriptor> = self
            .systems
            .values()
            .filter(|s| s.enabled)
            .map(|s| (s.name.as_str(), s))
            .collect();

        // Validate dependencies.
        for sys in enabled.values() {
            for dep in &sys.dependencies {
                let dep_desc = match self.systems.get(dep.as_str()) {
                    Some(d) => d,
                    None => {
                        return Err(ScheduleError::MissingDependency {
                            system: sys.name.clone(),
                            missing: dep.clone(),
                        });
                    }
                };
                // Skip disabled dependencies (they won't be in the schedule).
                if !dep_desc.enabled {
                    continue;
                }
                if dep_desc.phase as u8 > sys.phase as u8 {
                    return Err(ScheduleError::CrossPhaseDependency {
                        system: sys.name.clone(),
                        dependency: dep.clone(),
                        system_phase: sys.phase,
                        dep_phase: dep_desc.phase,
                    });
                }
            }
        }

        let mut batches = Vec::new();

        // Already-completed systems from previous phases (for cross-phase deps).
        let mut completed: HashSet<String> = HashSet::new();

        for &phase in RunPhase::all() {
            let phase_systems: Vec<&SystemDescriptor> = enabled
                .values()
                .filter(|s| s.phase == phase)
                .copied()
                .collect();

            if phase_systems.is_empty() {
                continue;
            }

            let phase_batches = Self::topo_sort_phase(&phase_systems, &completed)?;
            for batch in &phase_batches {
                for sys_name in &batch.systems {
                    completed.insert(sys_name.clone());
                }
            }
            batches.push((phase, phase_batches));
        }

        Ok(Schedule { batches })
    }

    /// Topological sort + parallel batching for a single phase.
    fn topo_sort_phase(
        systems: &[&SystemDescriptor],
        completed: &HashSet<String>,
    ) -> Result<Vec<ParallelBatch>, ScheduleError> {
        let names: HashSet<&str> = systems.iter().map(|s| s.name.as_str()).collect();

        // In-degree only counts deps within this phase that are also enabled.
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut dependents: HashMap<&str, Vec<&str>> = HashMap::new();

        for sys in systems {
            in_degree.entry(sys.name.as_str()).or_insert(0);
            for dep in &sys.dependencies {
                if names.contains(dep.as_str()) {
                    *in_degree.entry(sys.name.as_str()).or_insert(0) += 1;
                    dependents
                        .entry(dep.as_str())
                        .or_default()
                        .push(sys.name.as_str());
                }
                // Cross-phase deps are already completed, don't count.
            }
        }

        let sys_map: HashMap<&str, &SystemDescriptor> =
            systems.iter().map(|s| (s.name.as_str(), *s)).collect();

        let mut result_batches: Vec<ParallelBatch> = Vec::new();
        let mut ready: VecDeque<&str> = in_degree
            .iter()
            .filter(|&(_, deg)| *deg == 0)
            .map(|(&n, _)| n)
            .collect();

        let mut processed = 0;
        let total = systems.len();

        while !ready.is_empty() {
            // Drain all ready systems into one parallel batch.
            let mut batch_systems: Vec<&str> = ready.drain(..).collect();
            // Sort by priority descending, then name for determinism.
            batch_systems.sort_by(|a, b| {
                let pa = sys_map.get(a).map(|s| s.priority).unwrap_or(0);
                let pb = sys_map.get(b).map(|s| s.priority).unwrap_or(0);
                pb.cmp(&pa).then_with(|| a.cmp(b))
            });

            processed += batch_systems.len();

            for &sys_name in &batch_systems {
                if let Some(deps) = dependents.get(sys_name) {
                    for &dep in deps {
                        if let Some(deg) = in_degree.get_mut(dep) {
                            *deg -= 1;
                            if *deg == 0 {
                                ready.push_back(dep);
                            }
                        }
                    }
                }
            }

            result_batches.push(ParallelBatch {
                systems: batch_systems.into_iter().map(String::from).collect(),
            });
        }

        if processed != total {
            // Cycle detected — collect remaining nodes.
            let cycle: Vec<String> = in_degree
                .iter()
                .filter(|&(_, deg)| *deg > 0)
                .map(|(&n, _)| n.to_string())
                .collect();
            return Err(ScheduleError::CyclicDependency(cycle));
        }

        Ok(result_batches)
    }

    /// Convenience: get a flat execution order (sequential).
    pub fn execution_order(&self) -> Result<Vec<String>, ScheduleError> {
        let schedule = self.build_schedule()?;
        Ok(schedule
            .sequential_order()
            .into_iter()
            .map(String::from)
            .collect())
    }
}

impl Default for SystemScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_system() {
        let mut sched = SystemScheduler::new();
        sched
            .add_system(SystemDescriptor::new("physics", RunPhase::Update))
            .unwrap();
        let order = sched.execution_order().unwrap();
        assert_eq!(order, vec!["physics"]);
    }

    #[test]
    fn duplicate_name_error() {
        let mut sched = SystemScheduler::new();
        sched
            .add_system(SystemDescriptor::new("a", RunPhase::Update))
            .unwrap();
        let err = sched
            .add_system(SystemDescriptor::new("a", RunPhase::Update))
            .unwrap_err();
        assert_eq!(err, ScheduleError::DuplicateName("a".into()));
    }

    #[test]
    fn phases_in_order() {
        let mut sched = SystemScheduler::new();
        sched
            .add_system(SystemDescriptor::new("render", RunPhase::Render))
            .unwrap();
        sched
            .add_system(SystemDescriptor::new("physics", RunPhase::Update))
            .unwrap();
        sched
            .add_system(SystemDescriptor::new("input", RunPhase::PreUpdate))
            .unwrap();
        sched
            .add_system(SystemDescriptor::new("cleanup", RunPhase::PostUpdate))
            .unwrap();
        let order = sched.execution_order().unwrap();
        let input_pos = order.iter().position(|s| s == "input").unwrap();
        let phys_pos = order.iter().position(|s| s == "physics").unwrap();
        let clean_pos = order.iter().position(|s| s == "cleanup").unwrap();
        let render_pos = order.iter().position(|s| s == "render").unwrap();
        assert!(input_pos < phys_pos);
        assert!(phys_pos < clean_pos);
        assert!(clean_pos < render_pos);
    }

    #[test]
    fn dependency_ordering() {
        let mut sched = SystemScheduler::new();
        sched
            .add_system(
                SystemDescriptor::new("movement", RunPhase::Update).depends_on("physics"),
            )
            .unwrap();
        sched
            .add_system(SystemDescriptor::new("physics", RunPhase::Update))
            .unwrap();
        let order = sched.execution_order().unwrap();
        let phys_pos = order.iter().position(|s| s == "physics").unwrap();
        let move_pos = order.iter().position(|s| s == "movement").unwrap();
        assert!(phys_pos < move_pos);
    }

    #[test]
    fn cycle_detection() {
        let mut sched = SystemScheduler::new();
        sched
            .add_system(SystemDescriptor::new("a", RunPhase::Update).depends_on("b"))
            .unwrap();
        sched
            .add_system(SystemDescriptor::new("b", RunPhase::Update).depends_on("a"))
            .unwrap();
        let err = sched.build_schedule().unwrap_err();
        match err {
            ScheduleError::CyclicDependency(cycle) => {
                assert!(cycle.contains(&"a".to_string()));
                assert!(cycle.contains(&"b".to_string()));
            }
            _ => panic!("expected CyclicDependency"),
        }
    }

    #[test]
    fn missing_dependency() {
        let mut sched = SystemScheduler::new();
        sched
            .add_system(
                SystemDescriptor::new("a", RunPhase::Update).depends_on("nonexistent"),
            )
            .unwrap();
        let err = sched.build_schedule().unwrap_err();
        match err {
            ScheduleError::MissingDependency { system, missing } => {
                assert_eq!(system, "a");
                assert_eq!(missing, "nonexistent");
            }
            _ => panic!("expected MissingDependency"),
        }
    }

    #[test]
    fn cross_phase_dependency_error() {
        let mut sched = SystemScheduler::new();
        sched
            .add_system(
                SystemDescriptor::new("early", RunPhase::PreUpdate).depends_on("late"),
            )
            .unwrap();
        sched
            .add_system(SystemDescriptor::new("late", RunPhase::Render))
            .unwrap();
        let err = sched.build_schedule().unwrap_err();
        match err {
            ScheduleError::CrossPhaseDependency { system, .. } => {
                assert_eq!(system, "early");
            }
            _ => panic!("expected CrossPhaseDependency"),
        }
    }

    #[test]
    fn disabled_system_excluded() {
        let mut sched = SystemScheduler::new();
        sched
            .add_system(SystemDescriptor::new("a", RunPhase::Update))
            .unwrap();
        sched
            .add_system(SystemDescriptor::new("b", RunPhase::Update).with_enabled(false))
            .unwrap();
        let order = sched.execution_order().unwrap();
        assert_eq!(order, vec!["a"]);
    }

    #[test]
    fn enable_disable() {
        let mut sched = SystemScheduler::new();
        sched
            .add_system(SystemDescriptor::new("a", RunPhase::Update))
            .unwrap();
        assert!(sched.set_enabled("a", false));
        let order = sched.execution_order().unwrap();
        assert!(order.is_empty());
        assert!(sched.set_enabled("a", true));
        let order = sched.execution_order().unwrap();
        assert_eq!(order, vec!["a"]);
    }

    #[test]
    fn set_enabled_nonexistent() {
        let mut sched = SystemScheduler::new();
        assert!(!sched.set_enabled("nope", true));
    }

    #[test]
    fn remove_system() {
        let mut sched = SystemScheduler::new();
        sched
            .add_system(SystemDescriptor::new("a", RunPhase::Update))
            .unwrap();
        assert!(sched.remove_system("a"));
        assert!(!sched.remove_system("a"));
        assert_eq!(sched.system_count(), 0);
    }

    #[test]
    fn parallel_batching() {
        let mut sched = SystemScheduler::new();
        // a, b, c are independent; d depends on a and b.
        sched
            .add_system(SystemDescriptor::new("a", RunPhase::Update))
            .unwrap();
        sched
            .add_system(SystemDescriptor::new("b", RunPhase::Update))
            .unwrap();
        sched
            .add_system(SystemDescriptor::new("c", RunPhase::Update))
            .unwrap();
        sched
            .add_system(
                SystemDescriptor::new("d", RunPhase::Update)
                    .depends_on("a")
                    .depends_on("b"),
            )
            .unwrap();
        let schedule = sched.build_schedule().unwrap();
        // a,b,c should be in batch 0; d in batch 1.
        assert!(schedule.batch_count() >= 2);
        let seq = schedule.sequential_order();
        let d_pos = seq.iter().position(|s| *s == "d").unwrap();
        let a_pos = seq.iter().position(|s| *s == "a").unwrap();
        let b_pos = seq.iter().position(|s| *s == "b").unwrap();
        assert!(a_pos < d_pos);
        assert!(b_pos < d_pos);
    }

    #[test]
    fn priority_within_batch() {
        let mut sched = SystemScheduler::new();
        sched
            .add_system(SystemDescriptor::new("low", RunPhase::Update).with_priority(0))
            .unwrap();
        sched
            .add_system(SystemDescriptor::new("high", RunPhase::Update).with_priority(10))
            .unwrap();
        let order = sched.execution_order().unwrap();
        let high_pos = order.iter().position(|s| s == "high").unwrap();
        let low_pos = order.iter().position(|s| s == "low").unwrap();
        assert!(high_pos < low_pos);
    }

    #[test]
    fn schedule_counts() {
        let mut sched = SystemScheduler::new();
        sched
            .add_system(SystemDescriptor::new("a", RunPhase::Update))
            .unwrap();
        sched
            .add_system(SystemDescriptor::new("b", RunPhase::Render))
            .unwrap();
        let schedule = sched.build_schedule().unwrap();
        assert_eq!(schedule.system_count(), 2);
    }

    #[test]
    fn get_system() {
        let mut sched = SystemScheduler::new();
        sched
            .add_system(SystemDescriptor::new("a", RunPhase::Update).with_priority(5))
            .unwrap();
        let desc = sched.get_system("a").unwrap();
        assert_eq!(desc.priority, 5);
        assert!(sched.get_system("nope").is_none());
    }

    #[test]
    fn disabled_dependency_skipped() {
        let mut sched = SystemScheduler::new();
        sched
            .add_system(SystemDescriptor::new("a", RunPhase::Update).with_enabled(false))
            .unwrap();
        sched
            .add_system(SystemDescriptor::new("b", RunPhase::Update).depends_on("a"))
            .unwrap();
        // a is disabled, so b should still schedule (dep is ignored).
        let order = sched.execution_order().unwrap();
        assert_eq!(order, vec!["b"]);
    }

    #[test]
    fn cross_phase_dep_same_or_earlier_ok() {
        let mut sched = SystemScheduler::new();
        sched
            .add_system(SystemDescriptor::new("early", RunPhase::PreUpdate))
            .unwrap();
        sched
            .add_system(
                SystemDescriptor::new("late", RunPhase::Update).depends_on("early"),
            )
            .unwrap();
        let order = sched.execution_order().unwrap();
        let early_pos = order.iter().position(|s| s == "early").unwrap();
        let late_pos = order.iter().position(|s| s == "late").unwrap();
        assert!(early_pos < late_pos);
    }

    #[test]
    fn empty_scheduler() {
        let sched = SystemScheduler::new();
        let schedule = sched.build_schedule().unwrap();
        assert_eq!(schedule.system_count(), 0);
        assert!(schedule.sequential_order().is_empty());
    }

    #[test]
    fn chain_of_three() {
        let mut sched = SystemScheduler::new();
        sched
            .add_system(SystemDescriptor::new("c", RunPhase::Update).depends_on("b"))
            .unwrap();
        sched
            .add_system(SystemDescriptor::new("b", RunPhase::Update).depends_on("a"))
            .unwrap();
        sched
            .add_system(SystemDescriptor::new("a", RunPhase::Update))
            .unwrap();
        let order = sched.execution_order().unwrap();
        assert_eq!(order, vec!["a", "b", "c"]);
    }

    #[test]
    fn three_node_cycle() {
        let mut sched = SystemScheduler::new();
        sched.add_system(SystemDescriptor::new("a", RunPhase::Update).depends_on("c")).unwrap();
        sched.add_system(SystemDescriptor::new("b", RunPhase::Update).depends_on("a")).unwrap();
        sched.add_system(SystemDescriptor::new("c", RunPhase::Update).depends_on("b")).unwrap();
        let err = sched.build_schedule().unwrap_err();
        match err {
            ScheduleError::CyclicDependency(_) => {}
            _ => panic!("expected cycle"),
        }
    }

    #[test]
    fn error_display() {
        let err = ScheduleError::DuplicateName("test".into());
        assert!(err.to_string().contains("test"));
    }
}
