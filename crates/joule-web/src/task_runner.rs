//! Task runner (make/just-like).
//!
//! Provides task definitions with dependencies, topological execution order,
//! parallel task groups, task parameters, conditional execution, output capture,
//! dry-run mode, and task file parsing. Pure Rust — no shell or process deps.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Errors from task runner operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskRunnerError {
    /// Task not found.
    TaskNotFound(String),
    /// Circular dependency detected.
    CyclicDependency(Vec<String>),
    /// Duplicate task name.
    DuplicateTask(String),
    /// Missing required parameter.
    MissingParameter { task: String, param: String },
    /// Task condition not met.
    ConditionNotMet(String),
    /// Task execution failed.
    ExecutionFailed { task: String, message: String },
    /// Parse error in task file.
    ParseError { line: usize, message: String },
}

impl fmt::Display for TaskRunnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TaskNotFound(name) => write!(f, "task not found: {name}"),
            Self::CyclicDependency(cycle) => {
                write!(f, "cyclic dependency: {}", cycle.join(" -> "))
            }
            Self::DuplicateTask(name) => write!(f, "duplicate task: {name}"),
            Self::MissingParameter { task, param } => {
                write!(f, "task '{task}' missing required parameter '{param}'")
            }
            Self::ConditionNotMet(name) => write!(f, "condition not met for task: {name}"),
            Self::ExecutionFailed { task, message } => {
                write!(f, "task '{task}' failed: {message}")
            }
            Self::ParseError { line, message } => {
                write!(f, "parse error at line {line}: {message}")
            }
        }
    }
}

impl std::error::Error for TaskRunnerError {}

// ── Task Parameter ──────────────────────────────────────────────

/// A parameter definition for a task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskParam {
    pub name: String,
    pub description: String,
    pub default: Option<String>,
    pub required: bool,
}

impl TaskParam {
    pub fn required(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            default: None,
            required: true,
        }
    }

    pub fn optional(name: &str, description: &str, default: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            default: Some(default.to_string()),
            required: false,
        }
    }
}

// ── Task Condition ──────────────────────────────────────────────

/// A condition that determines whether a task should run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskCondition {
    /// Always run.
    Always,
    /// Run only if a variable is set.
    VarSet(String),
    /// Run only if a variable equals a specific value.
    VarEquals(String, String),
    /// Run only if a variable is NOT set.
    VarNotSet(String),
    /// Negate a condition.
    Not(Box<TaskCondition>),
    /// All conditions must be true.
    All(Vec<TaskCondition>),
    /// Any condition must be true.
    Any(Vec<TaskCondition>),
}

impl TaskCondition {
    /// Evaluate the condition against a variable context.
    pub fn evaluate(&self, vars: &HashMap<String, String>) -> bool {
        match self {
            Self::Always => true,
            Self::VarSet(name) => vars.contains_key(name),
            Self::VarEquals(name, value) => vars.get(name).map_or(false, |v| v == value),
            Self::VarNotSet(name) => !vars.contains_key(name),
            Self::Not(inner) => !inner.evaluate(vars),
            Self::All(conditions) => conditions.iter().all(|c| c.evaluate(vars)),
            Self::Any(conditions) => conditions.iter().any(|c| c.evaluate(vars)),
        }
    }
}

impl Default for TaskCondition {
    fn default() -> Self {
        Self::Always
    }
}

// ── Task Step ───────────────────────────────────────────────────

/// A single step within a task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStep {
    /// The command to execute (or description for dry-run).
    pub command: String,
    /// Working directory override.
    pub working_dir: Option<String>,
    /// Whether to continue if this step fails.
    pub continue_on_error: bool,
}

impl TaskStep {
    pub fn new(command: &str) -> Self {
        Self {
            command: command.to_string(),
            working_dir: None,
            continue_on_error: false,
        }
    }

    pub fn with_dir(command: &str, dir: &str) -> Self {
        Self {
            command: command.to_string(),
            working_dir: Some(dir.to_string()),
            continue_on_error: false,
        }
    }

    /// Substitute parameters into the command string.
    pub fn substitute(&self, params: &HashMap<String, String>) -> Self {
        let mut cmd = self.command.clone();
        for (key, value) in params {
            let placeholder = format!("{{{{{key}}}}}"); // {{key}}
            cmd = cmd.replace(&placeholder, value);
        }
        Self {
            command: cmd,
            working_dir: self.working_dir.clone(),
            continue_on_error: self.continue_on_error,
        }
    }
}

// ── Task Definition ─────────────────────────────────────────────

/// A complete task definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDef {
    pub name: String,
    pub description: String,
    pub dependencies: Vec<String>,
    pub steps: Vec<TaskStep>,
    pub params: Vec<TaskParam>,
    pub condition: TaskCondition,
    /// Whether this task can run in parallel with sibling tasks.
    pub parallel: bool,
    /// Group name for parallel grouping.
    pub group: Option<String>,
}

impl TaskDef {
    /// Create a new task.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: String::new(),
            dependencies: Vec::new(),
            steps: Vec::new(),
            params: Vec::new(),
            condition: TaskCondition::Always,
            parallel: false,
            group: None,
        }
    }

    /// Add a description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Add a dependency.
    pub fn depends_on(mut self, dep: &str) -> Self {
        self.dependencies.push(dep.to_string());
        self
    }

    /// Add a step.
    pub fn step(mut self, command: &str) -> Self {
        self.steps.push(TaskStep::new(command));
        self
    }

    /// Add a parameter.
    pub fn param(mut self, p: TaskParam) -> Self {
        self.params.push(p);
        self
    }

    /// Set the condition.
    pub fn when(mut self, condition: TaskCondition) -> Self {
        self.condition = condition;
        self
    }

    /// Mark as parallel-capable.
    pub fn parallel(mut self) -> Self {
        self.parallel = true;
        self
    }

    /// Set a group.
    pub fn in_group(mut self, group: &str) -> Self {
        self.group = Some(group.to_string());
        self
    }
}

// ── Execution Plan ──────────────────────────────────────────────

/// A group of tasks that can execute in parallel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelGroup {
    pub tasks: Vec<String>,
}

/// The planned execution order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// Ordered list of parallel groups. Groups run sequentially;
    /// tasks within a group can run in parallel.
    pub groups: Vec<ParallelGroup>,
    /// Total number of tasks.
    pub total_tasks: usize,
}

impl ExecutionPlan {
    /// Flatten the plan into a sequential task list.
    pub fn flatten(&self) -> Vec<&str> {
        self.groups
            .iter()
            .flat_map(|g| g.tasks.iter().map(|s| s.as_str()))
            .collect()
    }
}

impl fmt::Display for ExecutionPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, group) in self.groups.iter().enumerate() {
            if group.tasks.len() == 1 {
                writeln!(f, "Step {}: {}", i + 1, group.tasks[0])?;
            } else {
                writeln!(
                    f,
                    "Step {} (parallel): {}",
                    i + 1,
                    group.tasks.join(", ")
                )?;
            }
        }
        Ok(())
    }
}

// ── Task Output ─────────────────────────────────────────────────

/// Captured output of a single task execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskOutput {
    pub task_name: String,
    pub steps_output: Vec<StepOutput>,
    pub success: bool,
    pub skipped: bool,
    pub duration_ms: u64,
}

/// Output from a single step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepOutput {
    pub command: String,
    pub stdout: String,
    pub stderr: String,
    pub success: bool,
}

// ── Execution Result ────────────────────────────────────────────

/// Overall result of running a task (and its dependencies).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    pub task_outputs: Vec<TaskOutput>,
    pub success: bool,
    pub total_duration_ms: u64,
    pub dry_run: bool,
}

impl RunResult {
    pub fn successful_tasks(&self) -> Vec<&str> {
        self.task_outputs
            .iter()
            .filter(|o| o.success && !o.skipped)
            .map(|o| o.task_name.as_str())
            .collect()
    }

    pub fn failed_tasks(&self) -> Vec<&str> {
        self.task_outputs
            .iter()
            .filter(|o| !o.success && !o.skipped)
            .map(|o| o.task_name.as_str())
            .collect()
    }

    pub fn skipped_tasks(&self) -> Vec<&str> {
        self.task_outputs
            .iter()
            .filter(|o| o.skipped)
            .map(|o| o.task_name.as_str())
            .collect()
    }
}

// ── Task Executor Trait ─────────────────────────────────────────

/// Trait for executing a task step. Allows injecting custom execution logic.
pub trait StepExecutor {
    fn execute_step(
        &mut self,
        task_name: &str,
        step: &TaskStep,
        params: &HashMap<String, String>,
    ) -> StepOutput;
}

/// A simulated executor that records commands without running them.
#[derive(Debug, Default)]
pub struct SimulatedExecutor {
    pub executed: Vec<String>,
}

impl StepExecutor for SimulatedExecutor {
    fn execute_step(
        &mut self,
        _task_name: &str,
        step: &TaskStep,
        params: &HashMap<String, String>,
    ) -> StepOutput {
        let substituted = step.substitute(params);
        self.executed.push(substituted.command.clone());
        StepOutput {
            command: substituted.command,
            stdout: String::new(),
            stderr: String::new(),
            success: true,
        }
    }
}

// ── Task Runner ─────────────────────────────────────────────────

/// The main task runner.
pub struct TaskRunner {
    tasks: HashMap<String, TaskDef>,
    variables: HashMap<String, String>,
}

impl TaskRunner {
    /// Create a new empty task runner.
    pub fn new() -> Self {
        Self {
            tasks: HashMap::new(),
            variables: HashMap::new(),
        }
    }

    /// Register a task.
    pub fn add_task(&mut self, task: TaskDef) -> Result<(), TaskRunnerError> {
        if self.tasks.contains_key(&task.name) {
            return Err(TaskRunnerError::DuplicateTask(task.name.clone()));
        }
        self.tasks.insert(task.name.clone(), task);
        Ok(())
    }

    /// Set a variable.
    pub fn set_var(&mut self, key: &str, value: &str) {
        self.variables.insert(key.to_string(), value.to_string());
    }

    /// Get a task definition.
    pub fn get_task(&self, name: &str) -> Option<&TaskDef> {
        self.tasks.get(name)
    }

    /// List all task names.
    pub fn task_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tasks.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    /// Compute the topological execution order for a target task.
    pub fn execution_order(&self, target: &str) -> Result<Vec<String>, TaskRunnerError> {
        if !self.tasks.contains_key(target) {
            return Err(TaskRunnerError::TaskNotFound(target.to_string()));
        }

        let mut order = Vec::new();
        let mut visited = HashSet::new();
        let mut in_stack = HashSet::new();
        self.topo_visit(target, &mut order, &mut visited, &mut in_stack)?;
        Ok(order)
    }

    fn topo_visit(
        &self,
        name: &str,
        order: &mut Vec<String>,
        visited: &mut HashSet<String>,
        in_stack: &mut HashSet<String>,
    ) -> Result<(), TaskRunnerError> {
        if in_stack.contains(name) {
            // Build cycle path.
            let cycle = vec![name.to_string()];
            return Err(TaskRunnerError::CyclicDependency(cycle));
        }
        if visited.contains(name) {
            return Ok(());
        }
        in_stack.insert(name.to_string());

        if let Some(task) = self.tasks.get(name) {
            for dep in &task.dependencies {
                if !self.tasks.contains_key(dep) {
                    return Err(TaskRunnerError::TaskNotFound(dep.clone()));
                }
                self.topo_visit(dep, order, visited, in_stack)?;
            }
        }

        in_stack.remove(name);
        visited.insert(name.to_string());
        order.push(name.to_string());
        Ok(())
    }

    /// Build an execution plan with parallel groups.
    pub fn plan(&self, target: &str) -> Result<ExecutionPlan, TaskRunnerError> {
        let order = self.execution_order(target)?;

        // Compute levels using in-degree approach.
        let mut completed: HashSet<String> = HashSet::new();
        let mut groups: Vec<ParallelGroup> = Vec::new();

        let mut remaining: VecDeque<String> = order.into_iter().collect();

        while !remaining.is_empty() {
            let mut current_group = Vec::new();
            let mut next_remaining = VecDeque::new();

            for task_name in remaining {
                let task = &self.tasks[&task_name];
                let deps_satisfied = task.dependencies.iter().all(|d| completed.contains(d));
                if deps_satisfied {
                    current_group.push(task_name);
                } else {
                    next_remaining.push_back(task_name);
                }
            }

            if current_group.is_empty() && !next_remaining.is_empty() {
                // Should not happen if topo-sort is correct, but safety net.
                break;
            }

            for t in &current_group {
                completed.insert(t.clone());
            }

            groups.push(ParallelGroup {
                tasks: current_group,
            });
            remaining = next_remaining;
        }

        let total = groups.iter().map(|g| g.tasks.len()).sum();
        Ok(ExecutionPlan {
            groups,
            total_tasks: total,
        })
    }

    /// Run a target task with all dependencies.
    pub fn run<E: StepExecutor>(
        &self,
        target: &str,
        executor: &mut E,
        params: &HashMap<String, String>,
        dry_run: bool,
    ) -> Result<RunResult, TaskRunnerError> {
        let order = self.execution_order(target)?;
        let mut outputs = Vec::new();
        let mut all_success = true;

        for task_name in &order {
            let task = &self.tasks[task_name];

            // Check required parameters.
            for p in &task.params {
                if p.required && !params.contains_key(&p.name) && p.default.is_none() {
                    return Err(TaskRunnerError::MissingParameter {
                        task: task_name.clone(),
                        param: p.name.clone(),
                    });
                }
            }

            // Check condition.
            if !task.condition.evaluate(&self.variables) {
                outputs.push(TaskOutput {
                    task_name: task_name.clone(),
                    steps_output: Vec::new(),
                    success: true,
                    skipped: true,
                    duration_ms: 0,
                });
                continue;
            }

            // Build effective params (task defaults + caller params).
            let mut effective_params = HashMap::new();
            for p in &task.params {
                if let Some(def) = &p.default {
                    effective_params.insert(p.name.clone(), def.clone());
                }
            }
            for (k, v) in params {
                effective_params.insert(k.clone(), v.clone());
            }
            // Also add runner variables.
            for (k, v) in &self.variables {
                effective_params
                    .entry(k.clone())
                    .or_insert_with(|| v.clone());
            }

            if dry_run {
                let steps_output: Vec<StepOutput> = task
                    .steps
                    .iter()
                    .map(|s| {
                        let sub = s.substitute(&effective_params);
                        StepOutput {
                            command: format!("[dry-run] {}", sub.command),
                            stdout: String::new(),
                            stderr: String::new(),
                            success: true,
                        }
                    })
                    .collect();
                outputs.push(TaskOutput {
                    task_name: task_name.clone(),
                    steps_output,
                    success: true,
                    skipped: false,
                    duration_ms: 0,
                });
            } else {
                let mut step_outputs = Vec::new();
                let mut task_success = true;
                for step in &task.steps {
                    let output = executor.execute_step(task_name, step, &effective_params);
                    if !output.success && !step.continue_on_error {
                        task_success = false;
                        step_outputs.push(output);
                        break;
                    }
                    step_outputs.push(output);
                }
                if !task_success {
                    all_success = false;
                }
                outputs.push(TaskOutput {
                    task_name: task_name.clone(),
                    steps_output: step_outputs,
                    success: task_success,
                    skipped: false,
                    duration_ms: 0,
                });
            }
        }

        Ok(RunResult {
            task_outputs: outputs,
            success: all_success,
            total_duration_ms: 0,
            dry_run,
        })
    }

    /// Detect circular dependencies across all tasks.
    pub fn detect_cycles(&self) -> Vec<Vec<String>> {
        let mut all_cycles = Vec::new();
        for name in self.tasks.keys() {
            let mut visited = HashSet::new();
            let mut in_stack = HashSet::new();
            let mut order = Vec::new();
            if self
                .topo_visit(name, &mut order, &mut visited, &mut in_stack)
                .is_err()
            {
                all_cycles.push(vec![name.clone()]);
            }
        }
        all_cycles
    }
}

impl Default for TaskRunner {
    fn default() -> Self {
        Self::new()
    }
}

// ── Task File Parser ────────────────────────────────────────────

/// Parse a simple task file format.
///
/// Format:
/// ```text
/// task build
///   description: Build the project
///   depends: clean, prepare
///   step: cargo build --release
///   step: echo done
///
/// task clean
///   description: Clean artifacts
///   step: rm -rf target/
/// ```
pub fn parse_taskfile(input: &str) -> Result<Vec<TaskDef>, TaskRunnerError> {
    let mut tasks = Vec::new();
    let mut current: Option<TaskDef> = None;
    let lines: Vec<&str> = input.lines().collect();

    for (line_idx, line) in lines.iter().enumerate() {
        let line_num = line_idx + 1;
        let trimmed = line.trim();

        // Skip blank lines and comments.
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if trimmed.starts_with("task ") {
            // Save previous task.
            if let Some(t) = current.take() {
                tasks.push(t);
            }
            let name = trimmed[5..].trim();
            if name.is_empty() {
                return Err(TaskRunnerError::ParseError {
                    line: line_num,
                    message: "task name is empty".to_string(),
                });
            }
            current = Some(TaskDef::new(name));
            continue;
        }

        // Properties must be indented.
        if let Some(task) = current.as_mut() {
            if let Some(rest) = trimmed.strip_prefix("description:") {
                task.description = rest.trim().to_string();
            } else if let Some(rest) = trimmed.strip_prefix("depends:") {
                task.dependencies = rest
                    .split(',')
                    .map(|d| d.trim().to_string())
                    .filter(|d| !d.is_empty())
                    .collect();
            } else if let Some(rest) = trimmed.strip_prefix("step:") {
                task.steps.push(TaskStep::new(rest.trim()));
            } else if let Some(rest) = trimmed.strip_prefix("param:") {
                let name = rest.trim();
                task.params.push(TaskParam::required(name, ""));
            } else if trimmed.strip_prefix("parallel").is_some() {
                task.parallel = true;
            } else if let Some(rest) = trimmed.strip_prefix("group:") {
                task.group = Some(rest.trim().to_string());
            } else {
                return Err(TaskRunnerError::ParseError {
                    line: line_num,
                    message: format!("unknown property: {trimmed}"),
                });
            }
        } else {
            return Err(TaskRunnerError::ParseError {
                line: line_num,
                message: "property outside of task".to_string(),
            });
        }
    }

    if let Some(t) = current {
        tasks.push(t);
    }

    Ok(tasks)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_runner() -> TaskRunner {
        let mut runner = TaskRunner::new();
        runner
            .add_task(TaskDef::new("clean").step("rm -rf target/"))
            .unwrap();
        runner
            .add_task(
                TaskDef::new("compile")
                    .depends_on("clean")
                    .step("cargo build"),
            )
            .unwrap();
        runner
            .add_task(
                TaskDef::new("test")
                    .depends_on("compile")
                    .step("cargo test"),
            )
            .unwrap();
        runner
            .add_task(
                TaskDef::new("lint")
                    .depends_on("compile")
                    .step("cargo clippy"),
            )
            .unwrap();
        runner
            .add_task(
                TaskDef::new("all")
                    .depends_on("test")
                    .depends_on("lint")
                    .step("echo all done"),
            )
            .unwrap();
        runner
    }

    #[test]
    fn execution_order_simple() {
        let runner = make_runner();
        let order = runner.execution_order("test").unwrap();
        assert_eq!(order, vec!["clean", "compile", "test"]);
    }

    #[test]
    fn execution_order_diamond() {
        let runner = make_runner();
        let order = runner.execution_order("all").unwrap();
        // clean, compile must be before test and lint; all must be last.
        let clean_pos = order.iter().position(|s| s == "clean").unwrap();
        let compile_pos = order.iter().position(|s| s == "compile").unwrap();
        let test_pos = order.iter().position(|s| s == "test").unwrap();
        let lint_pos = order.iter().position(|s| s == "lint").unwrap();
        let all_pos = order.iter().position(|s| s == "all").unwrap();
        assert!(clean_pos < compile_pos);
        assert!(compile_pos < test_pos);
        assert!(compile_pos < lint_pos);
        assert!(test_pos < all_pos);
        assert!(lint_pos < all_pos);
    }

    #[test]
    fn task_not_found() {
        let runner = make_runner();
        let err = runner.execution_order("nonexistent").unwrap_err();
        assert!(matches!(err, TaskRunnerError::TaskNotFound(_)));
    }

    #[test]
    fn cyclic_dependency() {
        let mut runner = TaskRunner::new();
        runner
            .add_task(TaskDef::new("a").depends_on("b"))
            .unwrap();
        runner
            .add_task(TaskDef::new("b").depends_on("a"))
            .unwrap();
        let err = runner.execution_order("a").unwrap_err();
        assert!(matches!(err, TaskRunnerError::CyclicDependency(_)));
    }

    #[test]
    fn duplicate_task() {
        let mut runner = TaskRunner::new();
        runner.add_task(TaskDef::new("build")).unwrap();
        let err = runner.add_task(TaskDef::new("build")).unwrap_err();
        assert!(matches!(err, TaskRunnerError::DuplicateTask(_)));
    }

    #[test]
    fn execution_plan_groups() {
        let runner = make_runner();
        let plan = runner.plan("all").unwrap();
        assert!(plan.total_tasks >= 5);
        // First group should only contain "clean" (no deps).
        assert!(plan.groups[0].tasks.contains(&"clean".to_string()));
    }

    #[test]
    fn dry_run() {
        let runner = make_runner();
        let mut exec = SimulatedExecutor::default();
        let params = HashMap::new();
        let result = runner.run("test", &mut exec, &params, true).unwrap();
        assert!(result.dry_run);
        assert!(result.success);
        // In dry-run mode, executor should not have been called.
        assert!(exec.executed.is_empty());
        // But steps should be recorded with [dry-run] prefix.
        for output in &result.task_outputs {
            if !output.skipped {
                for step in &output.steps_output {
                    assert!(step.command.starts_with("[dry-run]"));
                }
            }
        }
    }

    #[test]
    fn run_with_executor() {
        let runner = make_runner();
        let mut exec = SimulatedExecutor::default();
        let params = HashMap::new();
        let result = runner.run("test", &mut exec, &params, false).unwrap();
        assert!(result.success);
        assert_eq!(exec.executed.len(), 3); // clean, compile, test steps
    }

    #[test]
    fn parameter_substitution() {
        let step = TaskStep::new("echo {{name}} is {{version}}");
        let mut params = HashMap::new();
        params.insert("name".to_string(), "myapp".to_string());
        params.insert("version".to_string(), "1.0".to_string());
        let result = step.substitute(&params);
        assert_eq!(result.command, "echo myapp is 1.0");
    }

    #[test]
    fn missing_required_param() {
        let mut runner = TaskRunner::new();
        runner
            .add_task(
                TaskDef::new("deploy")
                    .param(TaskParam::required("env", "target environment"))
                    .step("deploy to {{env}}"),
            )
            .unwrap();
        let mut exec = SimulatedExecutor::default();
        let params = HashMap::new();
        let err = runner.run("deploy", &mut exec, &params, false).unwrap_err();
        assert!(matches!(err, TaskRunnerError::MissingParameter { .. }));
    }

    #[test]
    fn conditional_task_skipped() {
        let mut runner = TaskRunner::new();
        runner
            .add_task(
                TaskDef::new("deploy")
                    .when(TaskCondition::VarEquals(
                        "env".to_string(),
                        "prod".to_string(),
                    ))
                    .step("deploy"),
            )
            .unwrap();
        // Don't set env=prod, so condition fails.
        runner.set_var("env", "staging");
        let mut exec = SimulatedExecutor::default();
        let result = runner
            .run("deploy", &mut exec, &HashMap::new(), false)
            .unwrap();
        assert!(result.success);
        assert_eq!(result.skipped_tasks(), vec!["deploy"]);
    }

    #[test]
    fn conditional_task_runs() {
        let mut runner = TaskRunner::new();
        runner
            .add_task(
                TaskDef::new("deploy")
                    .when(TaskCondition::VarEquals(
                        "env".to_string(),
                        "prod".to_string(),
                    ))
                    .step("deploy cmd"),
            )
            .unwrap();
        runner.set_var("env", "prod");
        let mut exec = SimulatedExecutor::default();
        let result = runner
            .run("deploy", &mut exec, &HashMap::new(), false)
            .unwrap();
        assert_eq!(result.successful_tasks(), vec!["deploy"]);
    }

    #[test]
    fn condition_evaluation() {
        let mut vars = HashMap::new();
        vars.insert("CI".to_string(), "true".to_string());

        assert!(TaskCondition::Always.evaluate(&vars));
        assert!(TaskCondition::VarSet("CI".to_string()).evaluate(&vars));
        assert!(!TaskCondition::VarSet("NOPE".to_string()).evaluate(&vars));
        assert!(TaskCondition::VarEquals("CI".to_string(), "true".to_string()).evaluate(&vars));
        assert!(!TaskCondition::VarEquals("CI".to_string(), "false".to_string()).evaluate(&vars));
        assert!(TaskCondition::VarNotSet("NOPE".to_string()).evaluate(&vars));
        assert!(TaskCondition::Not(Box::new(TaskCondition::VarNotSet(
            "CI".to_string()
        )))
        .evaluate(&vars));
    }

    #[test]
    fn parse_taskfile_basic() {
        let input = "\
task clean
  description: Clean build artifacts
  step: rm -rf target/

task build
  description: Build the project
  depends: clean
  step: cargo build --release
";
        let tasks = parse_taskfile(input).unwrap();
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].name, "clean");
        assert_eq!(tasks[0].description, "Clean build artifacts");
        assert_eq!(tasks[1].name, "build");
        assert_eq!(tasks[1].dependencies, vec!["clean"]);
    }

    #[test]
    fn parse_taskfile_with_comments() {
        let input = "\
# This is a comment
task hello
  step: echo hello

# Another comment
task world
  depends: hello
  step: echo world
";
        let tasks = parse_taskfile(input).unwrap();
        assert_eq!(tasks.len(), 2);
    }

    #[test]
    fn parse_taskfile_error() {
        let input = "description: orphan property\n";
        let err = parse_taskfile(input).unwrap_err();
        assert!(matches!(err, TaskRunnerError::ParseError { .. }));
    }

    #[test]
    fn task_list() {
        let runner = make_runner();
        let names = runner.task_names();
        assert!(names.contains(&"clean"));
        assert!(names.contains(&"compile"));
        assert!(names.contains(&"test"));
    }

    #[test]
    fn execution_plan_display() {
        let runner = make_runner();
        let plan = runner.plan("test").unwrap();
        let display = format!("{plan}");
        assert!(display.contains("Step 1"));
    }

    #[test]
    fn run_result_helpers() {
        let result = RunResult {
            task_outputs: vec![
                TaskOutput {
                    task_name: "a".to_string(),
                    steps_output: vec![],
                    success: true,
                    skipped: false,
                    duration_ms: 0,
                },
                TaskOutput {
                    task_name: "b".to_string(),
                    steps_output: vec![],
                    success: false,
                    skipped: false,
                    duration_ms: 0,
                },
                TaskOutput {
                    task_name: "c".to_string(),
                    steps_output: vec![],
                    success: true,
                    skipped: true,
                    duration_ms: 0,
                },
            ],
            success: false,
            total_duration_ms: 0,
            dry_run: false,
        };
        assert_eq!(result.successful_tasks(), vec!["a"]);
        assert_eq!(result.failed_tasks(), vec!["b"]);
        assert_eq!(result.skipped_tasks(), vec!["c"]);
    }

    #[test]
    fn task_step_working_dir() {
        let step = TaskStep::with_dir("ls", "/tmp");
        assert_eq!(step.working_dir, Some("/tmp".to_string()));
    }

    #[test]
    fn task_def_builder() {
        let task = TaskDef::new("build")
            .with_description("Build project")
            .depends_on("clean")
            .step("cargo build")
            .parallel()
            .in_group("compile");
        assert_eq!(task.description, "Build project");
        assert_eq!(task.dependencies, vec!["clean"]);
        assert!(task.parallel);
        assert_eq!(task.group, Some("compile".to_string()));
    }

    #[test]
    fn error_display() {
        let e = TaskRunnerError::CyclicDependency(vec!["a".to_string(), "b".to_string()]);
        assert!(format!("{e}").contains("a -> b"));
    }

    #[test]
    fn detect_all_cycles() {
        let mut runner = TaskRunner::new();
        runner
            .add_task(TaskDef::new("x").depends_on("y"))
            .unwrap();
        runner
            .add_task(TaskDef::new("y").depends_on("x"))
            .unwrap();
        let cycles = runner.detect_cycles();
        assert!(!cycles.is_empty());
    }
}
