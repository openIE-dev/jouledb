//! Behavior Planning — Finite state machine, decision tree, rule-based lane
//! change logic, and intersection handling for autonomous driving.
//!
//! This module implements the behavioural decision layer that sits between
//! perception/prediction and the motion planner. It decides *what* the vehicle
//! should do (lane keep, lane change, stop, yield, etc.) based on the current
//! driving context.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Behavior planning errors.
#[derive(Debug, Clone, PartialEq)]
pub enum BehaviorError {
    /// Unknown state encountered.
    UnknownState(String),
    /// No valid transition from current state.
    NoTransition(String),
    /// Decision tree evaluation failed.
    DecisionFailed(String),
    /// Invalid configuration.
    InvalidConfig(String),
}

impl fmt::Display for BehaviorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownState(m) => write!(f, "unknown state: {m}"),
            Self::NoTransition(m) => write!(f, "no transition: {m}"),
            Self::DecisionFailed(m) => write!(f, "decision failed: {m}"),
            Self::InvalidConfig(m) => write!(f, "invalid config: {m}"),
        }
    }
}

impl std::error::Error for BehaviorError {}

// ── Driving Context ─────────────────────────────────────────────

/// Snapshot of the driving context used for behavioral decisions.
#[derive(Debug, Clone)]
pub struct DrivingContext {
    /// Current vehicle speed (m/s).
    pub speed: f64,
    /// Current lane index (0 = leftmost).
    pub current_lane: usize,
    /// Total number of lanes.
    pub num_lanes: usize,
    /// Distance to the nearest leading vehicle (m). `None` if no vehicle ahead.
    pub lead_distance: Option<f64>,
    /// Speed of the leading vehicle (m/s).
    pub lead_speed: Option<f64>,
    /// Distance to the next intersection (m). `None` if none ahead.
    pub intersection_distance: Option<f64>,
    /// True if traffic light is red or yellow.
    pub traffic_light_stop: bool,
    /// True if a stop sign is detected.
    pub stop_sign_detected: bool,
    /// Speed limit (m/s).
    pub speed_limit: f64,
    /// Whether the left lane is clear for a lane change.
    pub left_clear: bool,
    /// Whether the right lane is clear for a lane change.
    pub right_clear: bool,
    /// Time since last lane change (s).
    pub time_since_lane_change: f64,
}

impl DrivingContext {
    pub fn new(speed: f64, current_lane: usize, num_lanes: usize) -> Self {
        Self {
            speed,
            current_lane,
            num_lanes,
            lead_distance: None,
            lead_speed: None,
            intersection_distance: None,
            traffic_light_stop: false,
            stop_sign_detected: false,
            speed_limit: 30.0,
            left_clear: true,
            right_clear: true,
            time_since_lane_change: 10.0,
        }
    }

    pub fn with_lead(mut self, distance: f64, speed: f64) -> Self {
        self.lead_distance = Some(distance);
        self.lead_speed = Some(speed);
        self
    }

    pub fn with_intersection(mut self, distance: f64) -> Self {
        self.intersection_distance = Some(distance);
        self
    }

    pub fn with_traffic_light(mut self, stop: bool) -> Self {
        self.traffic_light_stop = stop;
        self
    }

    pub fn with_stop_sign(mut self, detected: bool) -> Self {
        self.stop_sign_detected = detected;
        self
    }
}

impl fmt::Display for DrivingContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Context(v={:.1}m/s, lane={}/{}, lead={:?}m)",
            self.speed, self.current_lane, self.num_lanes, self.lead_distance
        )
    }
}

// ── Behavior Command ────────────────────────────────────────────

/// High-level behavior command issued by the planner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BehaviorCmd {
    /// Maintain current lane and speed.
    LaneKeep,
    /// Follow the lead vehicle at safe distance.
    Follow,
    /// Change to the left lane.
    LaneChangeLeft,
    /// Change to the right lane.
    LaneChangeRight,
    /// Decelerate to a stop.
    Stop,
    /// Yield to traffic (slow down, prepare to stop).
    Yield,
    /// Accelerate to the speed limit.
    Accelerate,
    /// Prepare for an intersection (slow down, check traffic).
    IntersectionApproach,
    /// Emergency stop.
    EmergencyStop,
}

impl fmt::Display for BehaviorCmd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LaneKeep => write!(f, "LANE_KEEP"),
            Self::Follow => write!(f, "FOLLOW"),
            Self::LaneChangeLeft => write!(f, "LANE_CHANGE_LEFT"),
            Self::LaneChangeRight => write!(f, "LANE_CHANGE_RIGHT"),
            Self::Stop => write!(f, "STOP"),
            Self::Yield => write!(f, "YIELD"),
            Self::Accelerate => write!(f, "ACCELERATE"),
            Self::IntersectionApproach => write!(f, "INTERSECTION_APPROACH"),
            Self::EmergencyStop => write!(f, "EMERGENCY_STOP"),
        }
    }
}

// ── Finite State Machine ────────────────────────────────────────

/// State identifier for the behavior FSM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FsmState {
    Cruising,
    Following,
    PrepareLaneChange,
    LaneChanging,
    Stopping,
    Stopped,
    IntersectionWait,
}

impl fmt::Display for FsmState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cruising => write!(f, "Cruising"),
            Self::Following => write!(f, "Following"),
            Self::PrepareLaneChange => write!(f, "PrepareLaneChange"),
            Self::LaneChanging => write!(f, "LaneChanging"),
            Self::Stopping => write!(f, "Stopping"),
            Self::Stopped => write!(f, "Stopped"),
            Self::IntersectionWait => write!(f, "IntersectionWait"),
        }
    }
}

/// Transition condition: a closure-like predicate encoded as an enum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransitionCond {
    LeadTooClose,
    LeadFarEnough,
    LaneChangePossible,
    LaneChangeComplete,
    MustStop,
    StoppedComplete,
    IntersectionNear,
    IntersectionClear,
    Always,
}

/// Behavior FSM with explicit state transitions.
#[derive(Debug, Clone)]
pub struct BehaviorFsm {
    current: FsmState,
    transitions: Vec<(FsmState, TransitionCond, FsmState)>,
    follow_distance: f64,
    lane_change_cooldown: f64,
    intersection_slow_dist: f64,
}

impl BehaviorFsm {
    pub fn new() -> Self {
        let transitions = vec![
            (FsmState::Cruising, TransitionCond::LeadTooClose, FsmState::Following),
            (FsmState::Cruising, TransitionCond::MustStop, FsmState::Stopping),
            (FsmState::Cruising, TransitionCond::IntersectionNear, FsmState::IntersectionWait),
            (FsmState::Following, TransitionCond::LeadFarEnough, FsmState::Cruising),
            (FsmState::Following, TransitionCond::LaneChangePossible, FsmState::PrepareLaneChange),
            (FsmState::Following, TransitionCond::MustStop, FsmState::Stopping),
            (FsmState::PrepareLaneChange, TransitionCond::Always, FsmState::LaneChanging),
            (FsmState::LaneChanging, TransitionCond::LaneChangeComplete, FsmState::Cruising),
            (FsmState::Stopping, TransitionCond::StoppedComplete, FsmState::Stopped),
            (FsmState::Stopped, TransitionCond::LeadFarEnough, FsmState::Cruising),
            (FsmState::Stopped, TransitionCond::IntersectionClear, FsmState::Cruising),
            (FsmState::IntersectionWait, TransitionCond::IntersectionClear, FsmState::Cruising),
            (FsmState::IntersectionWait, TransitionCond::MustStop, FsmState::Stopping),
        ];
        Self {
            current: FsmState::Cruising,
            transitions,
            follow_distance: 20.0,
            lane_change_cooldown: 5.0,
            intersection_slow_dist: 50.0,
        }
    }

    pub fn with_follow_distance(mut self, d: f64) -> Self {
        self.follow_distance = d.max(1.0);
        self
    }

    pub fn with_lane_change_cooldown(mut self, t: f64) -> Self {
        self.lane_change_cooldown = t.max(0.0);
        self
    }

    pub fn with_intersection_slow_dist(mut self, d: f64) -> Self {
        self.intersection_slow_dist = d.max(1.0);
        self
    }

    pub fn state(&self) -> FsmState {
        self.current
    }

    /// Update the FSM given the driving context. Returns the behavior command.
    pub fn update(&mut self, ctx: &DrivingContext) -> BehaviorCmd {
        // Evaluate all conditions and find the first matching transition.
        for &(from, cond, to) in &self.transitions {
            if from == self.current && self.evaluate_cond(cond, ctx) {
                self.current = to;
                break;
            }
        }
        self.state_to_cmd(ctx)
    }

    fn evaluate_cond(&self, cond: TransitionCond, ctx: &DrivingContext) -> bool {
        match cond {
            TransitionCond::LeadTooClose => {
                ctx.lead_distance.map_or(false, |d| d < self.follow_distance)
            }
            TransitionCond::LeadFarEnough => {
                ctx.lead_distance.map_or(true, |d| d > self.follow_distance * 1.5)
            }
            TransitionCond::LaneChangePossible => {
                ctx.time_since_lane_change > self.lane_change_cooldown
                    && (ctx.left_clear || ctx.right_clear)
                    && ctx.lead_distance.map_or(false, |d| d < self.follow_distance)
            }
            TransitionCond::LaneChangeComplete => true,
            TransitionCond::MustStop => {
                ctx.traffic_light_stop || ctx.stop_sign_detected
            }
            TransitionCond::StoppedComplete => ctx.speed < 0.1,
            TransitionCond::IntersectionNear => {
                ctx.intersection_distance.map_or(false, |d| d < self.intersection_slow_dist)
            }
            TransitionCond::IntersectionClear => {
                !ctx.traffic_light_stop && !ctx.stop_sign_detected
            }
            TransitionCond::Always => true,
        }
    }

    fn state_to_cmd(&self, ctx: &DrivingContext) -> BehaviorCmd {
        match self.current {
            FsmState::Cruising => {
                if ctx.speed < ctx.speed_limit * 0.9 {
                    BehaviorCmd::Accelerate
                } else {
                    BehaviorCmd::LaneKeep
                }
            }
            FsmState::Following => BehaviorCmd::Follow,
            FsmState::PrepareLaneChange | FsmState::LaneChanging => {
                if ctx.left_clear && ctx.current_lane > 0 {
                    BehaviorCmd::LaneChangeLeft
                } else if ctx.right_clear {
                    BehaviorCmd::LaneChangeRight
                } else {
                    BehaviorCmd::Follow
                }
            }
            FsmState::Stopping => BehaviorCmd::Stop,
            FsmState::Stopped => BehaviorCmd::Stop,
            FsmState::IntersectionWait => BehaviorCmd::IntersectionApproach,
        }
    }
}

impl fmt::Display for BehaviorFsm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BehaviorFSM(state={})", self.current)
    }
}

// ── Decision Tree ───────────────────────────────────────────────

/// A node in a decision tree for behavior selection.
#[derive(Debug, Clone)]
pub enum DecisionNode {
    /// Leaf node: emit a behavior command.
    Leaf(BehaviorCmd),
    /// Branch on a condition.
    Branch {
        condition: DecisionCondition,
        if_true: Box<DecisionNode>,
        if_false: Box<DecisionNode>,
    },
}

/// Conditions that can be evaluated against the driving context.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecisionCondition {
    LeadDistLessThan(f64),
    SpeedAbove(f64),
    SpeedBelow(f64),
    TrafficLightRed,
    StopSignDetected,
    LeftLaneClear,
    RightLaneClear,
    IntersectionWithin(f64),
}

impl DecisionCondition {
    fn evaluate(&self, ctx: &DrivingContext) -> bool {
        match self {
            Self::LeadDistLessThan(d) => ctx.lead_distance.map_or(false, |ld| ld < *d),
            Self::SpeedAbove(v) => ctx.speed > *v,
            Self::SpeedBelow(v) => ctx.speed < *v,
            Self::TrafficLightRed => ctx.traffic_light_stop,
            Self::StopSignDetected => ctx.stop_sign_detected,
            Self::LeftLaneClear => ctx.left_clear,
            Self::RightLaneClear => ctx.right_clear,
            Self::IntersectionWithin(d) => {
                ctx.intersection_distance.map_or(false, |id| id < *d)
            }
        }
    }
}

impl DecisionNode {
    /// Evaluate the decision tree.
    pub fn evaluate(&self, ctx: &DrivingContext) -> BehaviorCmd {
        match self {
            Self::Leaf(cmd) => *cmd,
            Self::Branch { condition, if_true, if_false } => {
                if condition.evaluate(ctx) {
                    if_true.evaluate(ctx)
                } else {
                    if_false.evaluate(ctx)
                }
            }
        }
    }
}

/// Build a default decision tree for highway driving.
pub fn build_highway_tree() -> DecisionNode {
    DecisionNode::Branch {
        condition: DecisionCondition::TrafficLightRed,
        if_true: Box::new(DecisionNode::Leaf(BehaviorCmd::Stop)),
        if_false: Box::new(DecisionNode::Branch {
            condition: DecisionCondition::StopSignDetected,
            if_true: Box::new(DecisionNode::Leaf(BehaviorCmd::Stop)),
            if_false: Box::new(DecisionNode::Branch {
                condition: DecisionCondition::LeadDistLessThan(15.0),
                if_true: Box::new(DecisionNode::Branch {
                    condition: DecisionCondition::LeftLaneClear,
                    if_true: Box::new(DecisionNode::Leaf(BehaviorCmd::LaneChangeLeft)),
                    if_false: Box::new(DecisionNode::Leaf(BehaviorCmd::Follow)),
                }),
                if_false: Box::new(DecisionNode::Branch {
                    condition: DecisionCondition::SpeedBelow(25.0),
                    if_true: Box::new(DecisionNode::Leaf(BehaviorCmd::Accelerate)),
                    if_false: Box::new(DecisionNode::Leaf(BehaviorCmd::LaneKeep)),
                }),
            }),
        }),
    }
}

// ── Lane Change Rules ───────────────────────────────────────────

/// Rule-based lane change evaluator.
#[derive(Debug, Clone)]
pub struct LaneChangeRules {
    min_gap: f64,
    min_speed_advantage: f64,
    cooldown: f64,
    max_lanes_at_once: usize,
}

impl LaneChangeRules {
    pub fn new() -> Self {
        Self {
            min_gap: 15.0,
            min_speed_advantage: 2.0,
            cooldown: 5.0,
            max_lanes_at_once: 1,
        }
    }

    pub fn with_min_gap(mut self, g: f64) -> Self {
        self.min_gap = g.max(1.0);
        self
    }

    pub fn with_speed_advantage(mut self, a: f64) -> Self {
        self.min_speed_advantage = a.max(0.0);
        self
    }

    pub fn with_cooldown(mut self, c: f64) -> Self {
        self.cooldown = c.max(0.0);
        self
    }

    /// Evaluate whether a lane change is recommended.
    pub fn evaluate(&self, ctx: &DrivingContext) -> Option<BehaviorCmd> {
        if ctx.time_since_lane_change < self.cooldown {
            return None;
        }
        let dominated = ctx.lead_distance.map_or(false, |d| d < self.min_gap);
        let slow_lead = ctx.lead_speed.map_or(false, |ls| {
            ctx.speed - ls > self.min_speed_advantage
        });

        if !dominated && !slow_lead {
            return None;
        }

        if ctx.left_clear && ctx.current_lane > 0 {
            Some(BehaviorCmd::LaneChangeLeft)
        } else if ctx.right_clear && ctx.current_lane + 1 < ctx.num_lanes {
            Some(BehaviorCmd::LaneChangeRight)
        } else {
            None
        }
    }
}

impl fmt::Display for LaneChangeRules {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LaneChangeRules(gap={:.1}, cd={:.1})", self.min_gap, self.cooldown)
    }
}

// ── Intersection Handler ────────────────────────────────────────

/// Intersection handling state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntersectionPhase {
    Approaching,
    Waiting,
    Entering,
    Crossing,
    Exited,
}

impl fmt::Display for IntersectionPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Approaching => write!(f, "Approaching"),
            Self::Waiting => write!(f, "Waiting"),
            Self::Entering => write!(f, "Entering"),
            Self::Crossing => write!(f, "Crossing"),
            Self::Exited => write!(f, "Exited"),
        }
    }
}

/// Intersection handler with phase tracking.
#[derive(Debug, Clone)]
pub struct IntersectionHandler {
    phase: IntersectionPhase,
    slow_distance: f64,
    stop_distance: f64,
    wait_time: f64,
    elapsed_wait: f64,
}

impl IntersectionHandler {
    pub fn new() -> Self {
        Self {
            phase: IntersectionPhase::Approaching,
            slow_distance: 50.0,
            stop_distance: 5.0,
            wait_time: 3.0,
            elapsed_wait: 0.0,
        }
    }

    pub fn with_distances(mut self, slow: f64, stop: f64) -> Self {
        self.slow_distance = slow.max(stop + 1.0);
        self.stop_distance = stop.max(0.5);
        self
    }

    pub fn with_wait_time(mut self, t: f64) -> Self {
        self.wait_time = t.max(0.0);
        self
    }

    pub fn phase(&self) -> IntersectionPhase {
        self.phase
    }

    /// Update the intersection handler. Returns a behavior command.
    pub fn update(&mut self, ctx: &DrivingContext, dt: f64) -> BehaviorCmd {
        let int_dist = ctx.intersection_distance.unwrap_or(f64::INFINITY);

        match self.phase {
            IntersectionPhase::Approaching => {
                if int_dist < self.stop_distance {
                    self.phase = IntersectionPhase::Waiting;
                    self.elapsed_wait = 0.0;
                    BehaviorCmd::Stop
                } else if int_dist < self.slow_distance {
                    BehaviorCmd::IntersectionApproach
                } else {
                    BehaviorCmd::LaneKeep
                }
            }
            IntersectionPhase::Waiting => {
                self.elapsed_wait += dt;
                if self.elapsed_wait >= self.wait_time && !ctx.traffic_light_stop {
                    self.phase = IntersectionPhase::Entering;
                    BehaviorCmd::Accelerate
                } else {
                    BehaviorCmd::Stop
                }
            }
            IntersectionPhase::Entering => {
                self.phase = IntersectionPhase::Crossing;
                BehaviorCmd::Accelerate
            }
            IntersectionPhase::Crossing => {
                if int_dist > self.slow_distance {
                    self.phase = IntersectionPhase::Exited;
                }
                BehaviorCmd::LaneKeep
            }
            IntersectionPhase::Exited => BehaviorCmd::LaneKeep,
        }
    }

    /// Reset to approaching for a new intersection.
    pub fn reset(&mut self) {
        self.phase = IntersectionPhase::Approaching;
        self.elapsed_wait = 0.0;
    }
}

impl fmt::Display for IntersectionHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IntersectionHandler(phase={})", self.phase)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cruise_ctx() -> DrivingContext {
        DrivingContext::new(25.0, 1, 3)
    }

    #[test]
    fn test_context_display() {
        let ctx = cruise_ctx();
        assert!(format!("{ctx}").contains("25.0"));
    }

    #[test]
    fn test_behavior_cmd_display() {
        assert_eq!(format!("{}", BehaviorCmd::LaneKeep), "LANE_KEEP");
        assert_eq!(format!("{}", BehaviorCmd::EmergencyStop), "EMERGENCY_STOP");
    }

    #[test]
    fn test_fsm_initial_state() {
        let fsm = BehaviorFsm::new();
        assert_eq!(fsm.state(), FsmState::Cruising);
    }

    #[test]
    fn test_fsm_cruising_no_lead() {
        let mut fsm = BehaviorFsm::new();
        let ctx = cruise_ctx();
        let cmd = fsm.update(&ctx);
        assert_eq!(fsm.state(), FsmState::Cruising);
        assert!(cmd == BehaviorCmd::LaneKeep || cmd == BehaviorCmd::Accelerate);
    }

    #[test]
    fn test_fsm_transition_to_following() {
        let mut fsm = BehaviorFsm::new().with_follow_distance(30.0);
        let ctx = cruise_ctx().with_lead(10.0, 15.0);
        let cmd = fsm.update(&ctx);
        assert_eq!(fsm.state(), FsmState::Following);
        assert_eq!(cmd, BehaviorCmd::Follow);
    }

    #[test]
    fn test_fsm_stop_on_red_light() {
        let mut fsm = BehaviorFsm::new();
        let ctx = cruise_ctx().with_traffic_light(true);
        fsm.update(&ctx);
        assert_eq!(fsm.state(), FsmState::Stopping);
    }

    #[test]
    fn test_fsm_intersection() {
        let mut fsm = BehaviorFsm::new().with_intersection_slow_dist(60.0);
        let ctx = cruise_ctx().with_intersection(30.0);
        let cmd = fsm.update(&ctx);
        assert_eq!(fsm.state(), FsmState::IntersectionWait);
        assert_eq!(cmd, BehaviorCmd::IntersectionApproach);
    }

    #[test]
    fn test_decision_tree_red_light() {
        let tree = build_highway_tree();
        let ctx = cruise_ctx().with_traffic_light(true);
        assert_eq!(tree.evaluate(&ctx), BehaviorCmd::Stop);
    }

    #[test]
    fn test_decision_tree_clear_road() {
        let tree = build_highway_tree();
        let ctx = DrivingContext::new(30.0, 1, 3);
        let cmd = tree.evaluate(&ctx);
        assert!(cmd == BehaviorCmd::LaneKeep || cmd == BehaviorCmd::Accelerate);
    }

    #[test]
    fn test_decision_tree_slow_lead() {
        let tree = build_highway_tree();
        let ctx = cruise_ctx().with_lead(10.0, 10.0);
        let cmd = tree.evaluate(&ctx);
        // Lead < 15m, left clear → lane change left
        assert_eq!(cmd, BehaviorCmd::LaneChangeLeft);
    }

    #[test]
    fn test_lane_change_rules_cooldown() {
        let rules = LaneChangeRules::new().with_cooldown(10.0);
        let mut ctx = cruise_ctx().with_lead(5.0, 10.0);
        ctx.time_since_lane_change = 3.0;
        assert!(rules.evaluate(&ctx).is_none());
    }

    #[test]
    fn test_lane_change_rules_gap() {
        let rules = LaneChangeRules::new().with_min_gap(20.0);
        let ctx = cruise_ctx().with_lead(10.0, 20.0);
        let cmd = rules.evaluate(&ctx);
        assert!(cmd.is_some());
    }

    #[test]
    fn test_lane_change_rules_no_need() {
        let rules = LaneChangeRules::new();
        let ctx = cruise_ctx(); // no lead vehicle
        assert!(rules.evaluate(&ctx).is_none());
    }

    #[test]
    fn test_intersection_handler_approach() {
        let mut handler = IntersectionHandler::new().with_distances(50.0, 5.0);
        let ctx = cruise_ctx().with_intersection(30.0);
        let cmd = handler.update(&ctx, 0.1);
        assert_eq!(handler.phase(), IntersectionPhase::Approaching);
        assert_eq!(cmd, BehaviorCmd::IntersectionApproach);
    }

    #[test]
    fn test_intersection_handler_stop() {
        let mut handler = IntersectionHandler::new();
        let ctx = cruise_ctx().with_intersection(2.0);
        let cmd = handler.update(&ctx, 0.1);
        assert_eq!(handler.phase(), IntersectionPhase::Waiting);
        assert_eq!(cmd, BehaviorCmd::Stop);
    }

    #[test]
    fn test_intersection_handler_proceed() {
        let mut handler = IntersectionHandler::new().with_wait_time(1.0);
        let ctx = cruise_ctx().with_intersection(2.0);
        handler.update(&ctx, 0.1); // → Waiting
        // Wait enough time.
        for _ in 0..20 {
            handler.update(&ctx, 0.1);
        }
        let cmd = handler.update(&ctx, 0.1);
        assert!(
            handler.phase() == IntersectionPhase::Entering
                || handler.phase() == IntersectionPhase::Crossing
                || cmd == BehaviorCmd::Accelerate
        );
    }

    #[test]
    fn test_intersection_reset() {
        let mut handler = IntersectionHandler::new();
        let ctx = cruise_ctx().with_intersection(2.0);
        handler.update(&ctx, 0.1);
        handler.reset();
        assert_eq!(handler.phase(), IntersectionPhase::Approaching);
    }

    #[test]
    fn test_fsm_display() {
        let fsm = BehaviorFsm::new();
        assert!(format!("{fsm}").contains("Cruising"));
    }

    #[test]
    fn test_lane_change_rules_display() {
        let r = LaneChangeRules::new();
        assert!(format!("{r}").contains("LaneChangeRules"));
    }

    #[test]
    fn test_intersection_display() {
        let h = IntersectionHandler::new();
        assert!(format!("{h}").contains("Approaching"));
    }
}
