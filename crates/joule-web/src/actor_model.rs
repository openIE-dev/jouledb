//! Actor model — actors with mailbox, supervision, hierarchy, and ask pattern.
//!
//! Replaces Akka.js / CAF / Comedy with a pure-Rust actor model.
//! Supports actor spawning, message passing via mailboxes, supervisor
//! strategies (one-for-one, all-for-one), actor hierarchy, request-response
//! (ask pattern), and lifecycle management.

use std::collections::{HashMap, VecDeque};

// ── Errors ──────────────────────────────────────────────────────

/// Actor system domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActorError {
    /// Actor not found.
    ActorNotFound(u64),
    /// Mailbox is full.
    MailboxFull { actor_id: u64, capacity: usize },
    /// Actor is stopped.
    ActorStopped(u64),
    /// Duplicate actor ID.
    DuplicateActor(u64),
    /// Ask timed out (no response within tick budget).
    AskTimeout { actor_id: u64, correlation_id: u64 },
    /// Max restart intensity exceeded.
    MaxRestartsExceeded { actor_id: u64, restarts: u32, window: u32 },
}

impl std::fmt::Display for ActorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ActorNotFound(id) => write!(f, "actor not found: {id}"),
            Self::MailboxFull { actor_id, capacity } => {
                write!(f, "mailbox full for actor {actor_id} (cap {capacity})")
            }
            Self::ActorStopped(id) => write!(f, "actor {id} is stopped"),
            Self::DuplicateActor(id) => write!(f, "actor already exists: {id}"),
            Self::AskTimeout { actor_id, correlation_id } => {
                write!(f, "ask timeout: actor {actor_id}, correlation {correlation_id}")
            }
            Self::MaxRestartsExceeded { actor_id, restarts, window } => {
                write!(f, "actor {actor_id}: {restarts} restarts in {window} ticks")
            }
        }
    }
}

impl std::error::Error for ActorError {}

// ── Actor Lifecycle ─────────────────────────────────────────────

/// Lifecycle state of an actor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActorState {
    Created,
    Starting,
    Running,
    Restarting,
    Stopping,
    Stopped,
    Failed,
}

// ── Message ─────────────────────────────────────────────────────

/// A message in an actor's mailbox.
#[derive(Debug, Clone)]
pub struct Message {
    pub id: u64,
    pub sender_id: Option<u64>,
    pub payload: String,
    pub correlation_id: Option<u64>,
    pub timestamp_tick: u64,
}

impl Message {
    pub fn new(id: u64, payload: impl Into<String>) -> Self {
        Self {
            id,
            sender_id: None,
            payload: payload.into(),
            correlation_id: None,
            timestamp_tick: 0,
        }
    }

    /// Set the sender.
    pub fn from_actor(mut self, sender_id: u64) -> Self {
        self.sender_id = Some(sender_id);
        self
    }

    /// Set a correlation ID for ask-pattern replies.
    pub fn with_correlation(mut self, cid: u64) -> Self {
        self.correlation_id = Some(cid);
        self
    }
}

// ── Supervisor Strategy ─────────────────────────────────────────

/// How a supervisor handles child failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupervisorStrategy {
    /// Restart only the failed child.
    OneForOne,
    /// Restart all children when one fails.
    AllForOne,
}

impl Default for SupervisorStrategy {
    fn default() -> Self {
        Self::OneForOne
    }
}

// ── Restart Policy ──────────────────────────────────────────────

/// Controls how many restarts are allowed in a time window.
#[derive(Debug, Clone)]
pub struct RestartPolicy {
    pub max_restarts: u32,
    pub within_ticks: u32,
}

impl Default for RestartPolicy {
    fn default() -> Self {
        Self {
            max_restarts: 5,
            within_ticks: 60,
        }
    }
}

// ── Actor ───────────────────────────────────────────────────────

/// A logical actor with a mailbox and state.
#[derive(Debug)]
pub struct Actor {
    pub id: u64,
    pub name: String,
    pub state: ActorState,
    pub parent_id: Option<u64>,
    pub children: Vec<u64>,
    mailbox: VecDeque<Message>,
    mailbox_capacity: usize,
    pub messages_processed: u64,
    pub restart_count: u32,
    restart_timestamps: Vec<u64>,
    pub supervisor_strategy: SupervisorStrategy,
    pub restart_policy: RestartPolicy,
    /// Responses waiting for the ask pattern.
    responses: HashMap<u64, Message>,
}

impl Actor {
    pub fn new(id: u64, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            state: ActorState::Created,
            parent_id: None,
            children: Vec::new(),
            mailbox: VecDeque::new(),
            mailbox_capacity: 1000,
            messages_processed: 0,
            restart_count: 0,
            restart_timestamps: Vec::new(),
            supervisor_strategy: SupervisorStrategy::default(),
            restart_policy: RestartPolicy::default(),
            responses: HashMap::new(),
        }
    }

    /// Set mailbox capacity.
    pub fn with_mailbox_capacity(mut self, cap: usize) -> Self {
        self.mailbox_capacity = cap;
        self
    }

    /// Set this actor's supervisor strategy for its children.
    pub fn with_strategy(mut self, strategy: SupervisorStrategy) -> Self {
        self.supervisor_strategy = strategy;
        self
    }

    /// Set restart policy.
    pub fn with_restart_policy(mut self, policy: RestartPolicy) -> Self {
        self.restart_policy = policy;
        self
    }

    /// Mailbox size.
    pub fn mailbox_len(&self) -> usize {
        self.mailbox.len()
    }

    /// Whether the mailbox is full.
    pub fn mailbox_full(&self) -> bool {
        self.mailbox.len() >= self.mailbox_capacity
    }
}

// ── Actor System Events ─────────────────────────────────────────

/// Events emitted by the actor system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActorEvent {
    ActorStarted(u64),
    ActorStopped(u64),
    ActorRestarted(u64),
    ActorFailed(u64),
    MessageDelivered { to: u64, msg_id: u64 },
    MessageProcessed { actor_id: u64, msg_id: u64 },
    ChildSpawned { parent: u64, child: u64 },
    SupervisorEscalated { actor_id: u64 },
}

// ── Actor System ────────────────────────────────────────────────

/// The actor system manages actors, message passing, and supervision.
pub struct ActorSystem {
    actors: HashMap<u64, Actor>,
    next_actor_id: u64,
    next_msg_id: u64,
    next_correlation_id: u64,
    events: Vec<ActorEvent>,
    current_tick: u64,
}

impl ActorSystem {
    pub fn new() -> Self {
        Self {
            actors: HashMap::new(),
            next_actor_id: 1,
            next_msg_id: 1,
            next_correlation_id: 1,
            events: Vec::new(),
            current_tick: 0,
        }
    }

    /// Advance the clock.
    pub fn tick(&mut self, ticks: u64) {
        self.current_tick += ticks;
    }

    /// Spawn a new actor. Returns the actor ID.
    pub fn spawn(&mut self, mut actor: Actor) -> u64 {
        let id = self.next_actor_id;
        self.next_actor_id += 1;
        actor.id = id;
        actor.state = ActorState::Starting;
        self.actors.insert(id, actor);
        self.events.push(ActorEvent::ActorStarted(id));
        // Transition to running
        if let Some(a) = self.actors.get_mut(&id) {
            a.state = ActorState::Running;
        }
        id
    }

    /// Spawn a child actor under a parent. Returns child ID.
    pub fn spawn_child(&mut self, parent_id: u64, mut child: Actor) -> Result<u64, ActorError> {
        if !self.actors.contains_key(&parent_id) {
            return Err(ActorError::ActorNotFound(parent_id));
        }
        child.parent_id = Some(parent_id);
        let child_id = self.spawn(child);
        // Add child to parent's child list
        if let Some(parent) = self.actors.get_mut(&parent_id) {
            parent.children.push(child_id);
        }
        self.events.push(ActorEvent::ChildSpawned {
            parent: parent_id,
            child: child_id,
        });
        Ok(child_id)
    }

    /// Send a message to an actor.
    pub fn send(&mut self, to: u64, mut msg: Message) -> Result<u64, ActorError> {
        let actor = self.actors.get(&to).ok_or(ActorError::ActorNotFound(to))?;
        if actor.state == ActorState::Stopped || actor.state == ActorState::Failed {
            return Err(ActorError::ActorStopped(to));
        }
        let capacity = actor.mailbox_capacity;
        if actor.mailbox_full() {
            return Err(ActorError::MailboxFull {
                actor_id: to,
                capacity,
            });
        }
        let msg_id = self.next_msg_id;
        self.next_msg_id += 1;
        msg.id = msg_id;
        msg.timestamp_tick = self.current_tick;
        let actor = self.actors.get_mut(&to).unwrap();
        actor.mailbox.push_back(msg);
        self.events.push(ActorEvent::MessageDelivered {
            to,
            msg_id,
        });
        Ok(msg_id)
    }

    /// Ask pattern: send a message and register a correlation for response.
    /// Returns the correlation ID to use when calling `take_response`.
    pub fn ask(&mut self, to: u64, mut msg: Message) -> Result<u64, ActorError> {
        let cid = self.next_correlation_id;
        self.next_correlation_id += 1;
        msg.correlation_id = Some(cid);
        self.send(to, msg)?;
        Ok(cid)
    }

    /// Deliver a response from an actor (e.g., after processing an ask message).
    pub fn respond(
        &mut self,
        actor_id: u64,
        correlation_id: u64,
        response_payload: impl Into<String>,
    ) -> Result<(), ActorError> {
        if !self.actors.contains_key(&actor_id) {
            return Err(ActorError::ActorNotFound(actor_id));
        }
        let resp = Message {
            id: self.next_msg_id,
            sender_id: Some(actor_id),
            payload: response_payload.into(),
            correlation_id: Some(correlation_id),
            timestamp_tick: self.current_tick,
        };
        self.next_msg_id += 1;
        let actor = self.actors.get_mut(&actor_id).unwrap();
        actor.responses.insert(correlation_id, resp);
        Ok(())
    }

    /// Take a response for a correlation ID. Returns None if not yet available.
    pub fn take_response(&mut self, actor_id: u64, correlation_id: u64) -> Option<Message> {
        self.actors
            .get_mut(&actor_id)
            .and_then(|a| a.responses.remove(&correlation_id))
    }

    /// Process one message from an actor's mailbox. Returns the message if any.
    pub fn process_one(&mut self, actor_id: u64) -> Result<Option<Message>, ActorError> {
        let actor = self
            .actors
            .get_mut(&actor_id)
            .ok_or(ActorError::ActorNotFound(actor_id))?;
        if actor.state != ActorState::Running {
            return Ok(None);
        }
        match actor.mailbox.pop_front() {
            Some(msg) => {
                actor.messages_processed += 1;
                let msg_id = msg.id;
                self.events.push(ActorEvent::MessageProcessed {
                    actor_id,
                    msg_id,
                });
                Ok(Some(msg))
            }
            None => Ok(None),
        }
    }

    /// Simulate actor failure and invoke supervisor strategy.
    pub fn fail_actor(&mut self, actor_id: u64) -> Result<Vec<ActorEvent>, ActorError> {
        if !self.actors.contains_key(&actor_id) {
            return Err(ActorError::ActorNotFound(actor_id));
        }
        let mut new_events = Vec::new();

        // Mark actor as failed
        let parent_id = {
            let actor = self.actors.get_mut(&actor_id).unwrap();
            actor.state = ActorState::Failed;
            new_events.push(ActorEvent::ActorFailed(actor_id));
            actor.parent_id
        };

        // If there is a supervisor (parent), apply strategy
        if let Some(pid) = parent_id {
            let (strategy, children) = {
                let parent = self.actors.get(&pid).unwrap();
                (parent.supervisor_strategy, parent.children.clone())
            };
            match strategy {
                SupervisorStrategy::OneForOne => {
                    self.restart_actor(actor_id, &mut new_events)?;
                }
                SupervisorStrategy::AllForOne => {
                    for child_id in children {
                        self.restart_actor(child_id, &mut new_events)?;
                    }
                }
            }
        }

        self.events.extend(new_events.clone());
        Ok(new_events)
    }

    /// Restart an actor (clear mailbox, increment restart counter).
    fn restart_actor(
        &mut self,
        actor_id: u64,
        events: &mut Vec<ActorEvent>,
    ) -> Result<(), ActorError> {
        let actor = self
            .actors
            .get_mut(&actor_id)
            .ok_or(ActorError::ActorNotFound(actor_id))?;

        // Check restart intensity
        let current_tick = self.current_tick;
        let window = actor.restart_policy.within_ticks as u64;
        actor
            .restart_timestamps
            .retain(|t| current_tick.saturating_sub(*t) < window);
        let max = actor.restart_policy.max_restarts;
        if actor.restart_timestamps.len() as u32 >= max {
            actor.state = ActorState::Failed;
            events.push(ActorEvent::SupervisorEscalated { actor_id });
            return Err(ActorError::MaxRestartsExceeded {
                actor_id,
                restarts: max,
                window: actor.restart_policy.within_ticks,
            });
        }

        actor.restart_count += 1;
        actor.restart_timestamps.push(current_tick);
        actor.mailbox.clear();
        actor.state = ActorState::Running;
        events.push(ActorEvent::ActorRestarted(actor_id));
        Ok(())
    }

    /// Stop an actor and all its children.
    pub fn stop(&mut self, actor_id: u64) -> Result<(), ActorError> {
        let children = {
            let actor = self
                .actors
                .get_mut(&actor_id)
                .ok_or(ActorError::ActorNotFound(actor_id))?;
            actor.state = ActorState::Stopped;
            actor.mailbox.clear();
            self.events.push(ActorEvent::ActorStopped(actor_id));
            actor.children.clone()
        };
        // Recursively stop children
        for child_id in children {
            let _ = self.stop(child_id);
        }
        Ok(())
    }

    /// Get a reference to an actor.
    pub fn get_actor(&self, id: u64) -> Option<&Actor> {
        self.actors.get(&id)
    }

    /// Total number of actors.
    pub fn actor_count(&self) -> usize {
        self.actors.len()
    }

    /// Running actors.
    pub fn running_count(&self) -> usize {
        self.actors
            .values()
            .filter(|a| a.state == ActorState::Running)
            .count()
    }

    /// Get all events.
    pub fn events(&self) -> &[ActorEvent] {
        &self.events
    }

    /// Drain events.
    pub fn drain_events(&mut self) -> Vec<ActorEvent> {
        std::mem::take(&mut self.events)
    }
}

impl Default for ActorSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spawn_actor() {
        let mut sys = ActorSystem::new();
        let id = sys.spawn(Actor::new(0, "a1"));
        assert!(id > 0);
        assert_eq!(sys.get_actor(id).unwrap().state, ActorState::Running);
    }

    #[test]
    fn test_send_message() {
        let mut sys = ActorSystem::new();
        let id = sys.spawn(Actor::new(0, "a1"));
        let msg_id = sys.send(id, Message::new(0, "hello")).unwrap();
        assert!(msg_id > 0);
        assert_eq!(sys.get_actor(id).unwrap().mailbox_len(), 1);
    }

    #[test]
    fn test_process_message() {
        let mut sys = ActorSystem::new();
        let id = sys.spawn(Actor::new(0, "a1"));
        sys.send(id, Message::new(0, "hello")).unwrap();
        let msg = sys.process_one(id).unwrap().unwrap();
        assert_eq!(msg.payload, "hello");
        assert_eq!(sys.get_actor(id).unwrap().messages_processed, 1);
        assert_eq!(sys.get_actor(id).unwrap().mailbox_len(), 0);
    }

    #[test]
    fn test_send_to_stopped_actor() {
        let mut sys = ActorSystem::new();
        let id = sys.spawn(Actor::new(0, "a1"));
        sys.stop(id).unwrap();
        let err = sys.send(id, Message::new(0, "fail")).unwrap_err();
        assert_eq!(err, ActorError::ActorStopped(id));
    }

    #[test]
    fn test_mailbox_capacity() {
        let mut sys = ActorSystem::new();
        let id = sys.spawn(Actor::new(0, "a1").with_mailbox_capacity(2));
        sys.send(id, Message::new(0, "m1")).unwrap();
        sys.send(id, Message::new(0, "m2")).unwrap();
        let err = sys.send(id, Message::new(0, "m3")).unwrap_err();
        assert!(matches!(err, ActorError::MailboxFull { .. }));
    }

    #[test]
    fn test_spawn_child() {
        let mut sys = ActorSystem::new();
        let parent = sys.spawn(Actor::new(0, "parent"));
        let child = sys.spawn_child(parent, Actor::new(0, "child")).unwrap();
        assert_eq!(sys.get_actor(child).unwrap().parent_id, Some(parent));
        assert!(sys.get_actor(parent).unwrap().children.contains(&child));
    }

    #[test]
    fn test_spawn_child_invalid_parent() {
        let mut sys = ActorSystem::new();
        let err = sys.spawn_child(999, Actor::new(0, "orphan")).unwrap_err();
        assert_eq!(err, ActorError::ActorNotFound(999));
    }

    #[test]
    fn test_one_for_one_restart() {
        let mut sys = ActorSystem::new();
        let parent = sys.spawn(Actor::new(0, "supervisor").with_strategy(SupervisorStrategy::OneForOne));
        let c1 = sys.spawn_child(parent, Actor::new(0, "c1")).unwrap();
        let c2 = sys.spawn_child(parent, Actor::new(0, "c2")).unwrap();

        sys.drain_events();
        let events = sys.fail_actor(c1).unwrap();
        // Only c1 should restart
        assert!(events.contains(&ActorEvent::ActorRestarted(c1)));
        assert_eq!(sys.get_actor(c1).unwrap().state, ActorState::Running);
        assert_eq!(sys.get_actor(c2).unwrap().state, ActorState::Running);
    }

    #[test]
    fn test_all_for_one_restart() {
        let mut sys = ActorSystem::new();
        let parent = sys.spawn(
            Actor::new(0, "supervisor").with_strategy(SupervisorStrategy::AllForOne),
        );
        let c1 = sys.spawn_child(parent, Actor::new(0, "c1")).unwrap();
        let c2 = sys.spawn_child(parent, Actor::new(0, "c2")).unwrap();

        sys.drain_events();
        let events = sys.fail_actor(c1).unwrap();
        // Both should restart
        assert!(events.contains(&ActorEvent::ActorRestarted(c1)));
        assert!(events.contains(&ActorEvent::ActorRestarted(c2)));
    }

    #[test]
    fn test_ask_pattern() {
        let mut sys = ActorSystem::new();
        let id = sys.spawn(Actor::new(0, "responder"));
        let cid = sys.ask(id, Message::new(0, "question")).unwrap();

        // Process the question
        let msg = sys.process_one(id).unwrap().unwrap();
        assert_eq!(msg.correlation_id, Some(cid));

        // Respond
        sys.respond(id, cid, "answer").unwrap();
        let resp = sys.take_response(id, cid).unwrap();
        assert_eq!(resp.payload, "answer");
    }

    #[test]
    fn test_stop_cascades_to_children() {
        let mut sys = ActorSystem::new();
        let parent = sys.spawn(Actor::new(0, "parent"));
        let child = sys.spawn_child(parent, Actor::new(0, "child")).unwrap();
        let grandchild = sys
            .spawn_child(child, Actor::new(0, "grandchild"))
            .unwrap();

        sys.stop(parent).unwrap();
        assert_eq!(sys.get_actor(parent).unwrap().state, ActorState::Stopped);
        assert_eq!(sys.get_actor(child).unwrap().state, ActorState::Stopped);
        assert_eq!(
            sys.get_actor(grandchild).unwrap().state,
            ActorState::Stopped
        );
    }

    #[test]
    fn test_max_restart_intensity() {
        let mut sys = ActorSystem::new();
        let policy = RestartPolicy {
            max_restarts: 2,
            within_ticks: 100,
        };
        let parent = sys.spawn(Actor::new(0, "sup").with_strategy(SupervisorStrategy::OneForOne));
        let child = sys
            .spawn_child(parent, Actor::new(0, "fragile").with_restart_policy(policy))
            .unwrap();

        // First two restarts succeed
        sys.fail_actor(child).unwrap();
        sys.fail_actor(child).unwrap();
        // Third should escalate
        let result = sys.fail_actor(child);
        assert!(result.is_err());
    }

    #[test]
    fn test_actor_count() {
        let mut sys = ActorSystem::new();
        sys.spawn(Actor::new(0, "a1"));
        sys.spawn(Actor::new(0, "a2"));
        assert_eq!(sys.actor_count(), 2);
    }

    #[test]
    fn test_running_count() {
        let mut sys = ActorSystem::new();
        let a1 = sys.spawn(Actor::new(0, "a1"));
        sys.spawn(Actor::new(0, "a2"));
        sys.stop(a1).unwrap();
        assert_eq!(sys.running_count(), 1);
    }

    #[test]
    fn test_process_empty_mailbox() {
        let mut sys = ActorSystem::new();
        let id = sys.spawn(Actor::new(0, "empty"));
        let result = sys.process_one(id).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_message_ordering() {
        let mut sys = ActorSystem::new();
        let id = sys.spawn(Actor::new(0, "fifo"));
        sys.send(id, Message::new(0, "first")).unwrap();
        sys.send(id, Message::new(0, "second")).unwrap();
        sys.send(id, Message::new(0, "third")).unwrap();

        let m1 = sys.process_one(id).unwrap().unwrap();
        let m2 = sys.process_one(id).unwrap().unwrap();
        let m3 = sys.process_one(id).unwrap().unwrap();
        assert_eq!(m1.payload, "first");
        assert_eq!(m2.payload, "second");
        assert_eq!(m3.payload, "third");
    }

    #[test]
    fn test_events_emitted() {
        let mut sys = ActorSystem::new();
        let id = sys.spawn(Actor::new(0, "a1"));
        sys.send(id, Message::new(0, "hello")).unwrap();
        sys.process_one(id).unwrap();
        let events = sys.drain_events();
        assert!(events.contains(&ActorEvent::ActorStarted(id)));
        assert!(events.iter().any(|e| matches!(e, ActorEvent::MessageDelivered { .. })));
        assert!(events.iter().any(|e| matches!(e, ActorEvent::MessageProcessed { .. })));
    }

    #[test]
    fn test_send_to_nonexistent_actor() {
        let mut sys = ActorSystem::new();
        let err = sys.send(999, Message::new(0, "nope")).unwrap_err();
        assert_eq!(err, ActorError::ActorNotFound(999));
    }

    #[test]
    fn test_restart_clears_mailbox() {
        let mut sys = ActorSystem::new();
        let parent = sys.spawn(Actor::new(0, "sup").with_strategy(SupervisorStrategy::OneForOne));
        let child = sys.spawn_child(parent, Actor::new(0, "c1")).unwrap();
        sys.send(child, Message::new(0, "stale")).unwrap();
        assert_eq!(sys.get_actor(child).unwrap().mailbox_len(), 1);
        sys.fail_actor(child).unwrap();
        assert_eq!(sys.get_actor(child).unwrap().mailbox_len(), 0);
    }

    #[test]
    fn test_message_sender() {
        let mut sys = ActorSystem::new();
        let a1 = sys.spawn(Actor::new(0, "sender"));
        let a2 = sys.spawn(Actor::new(0, "receiver"));
        sys.send(a2, Message::new(0, "hi").from_actor(a1)).unwrap();
        let msg = sys.process_one(a2).unwrap().unwrap();
        assert_eq!(msg.sender_id, Some(a1));
    }
}
