//! Phantom type / typestate patterns — state machines via types, type-level
//! permissions, unit-safe arithmetic (meters * meters = square_meters),
//! type-level flags, and compile-time state validation concepts.
//!
//! Replaces ad-hoc phantom type patterns with a cohesive library for
//! typestate programming in Rust.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};

// ── State Markers ───────────────────────────────────────────────

/// Marker for the "open" state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Open;

/// Marker for the "closed" state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Closed;

/// Marker for the "locked" state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Locked;

/// Marker for a "draft" state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Draft;

/// Marker for a "published" state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Published;

/// Marker for an "archived" state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Archived;

// ── Permission Markers ──────────────────────────────────────────

/// Marker for read permission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadPerm;

/// Marker for write permission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WritePerm;

/// Marker for read+write permission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadWritePerm;

// ── Typestate Channel ───────────────────────────────────────────

/// A channel that transitions through typestates: Open -> Closed -> Locked.
/// Each state exposes different operations.
#[derive(Debug)]
pub struct Channel<State> {
    name: String,
    messages: Vec<String>,
    _state: PhantomData<State>,
}

impl Channel<Open> {
    /// Create a new open channel.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            messages: Vec::new(),
            _state: PhantomData,
        }
    }

    /// Send a message (only available on Open channels).
    pub fn send(&mut self, msg: impl Into<String>) {
        self.messages.push(msg.into());
    }

    /// Close the channel, transitioning to Closed state.
    pub fn close(self) -> Channel<Closed> {
        Channel {
            name: self.name,
            messages: self.messages,
            _state: PhantomData,
        }
    }

    /// Number of messages.
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }
}

impl Channel<Closed> {
    /// Read messages (only available on Closed channels).
    pub fn messages(&self) -> &[String] {
        &self.messages
    }

    /// Lock the channel, transitioning to Locked state.
    pub fn lock(self) -> Channel<Locked> {
        Channel {
            name: self.name,
            messages: self.messages,
            _state: PhantomData,
        }
    }

    /// Reopen the channel.
    pub fn reopen(self) -> Channel<Open> {
        Channel {
            name: self.name,
            messages: self.messages,
            _state: PhantomData,
        }
    }
}

impl Channel<Locked> {
    /// Get the channel name (the only thing you can do with a locked channel).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of archived messages.
    pub fn archived_count(&self) -> usize {
        self.messages.len()
    }
}

// ── Typestate Document ──────────────────────────────────────────

/// A document that transitions: Draft -> Published -> Archived.
#[derive(Debug)]
pub struct Document<State> {
    title: String,
    content: String,
    revision: u32,
    _state: PhantomData<State>,
}

impl Document<Draft> {
    /// Create a new draft document.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            content: String::new(),
            revision: 0,
            _state: PhantomData,
        }
    }

    /// Edit the content (only in Draft).
    pub fn edit(&mut self, content: impl Into<String>) {
        self.content = content.into();
        self.revision += 1;
    }

    /// Publish the document.
    pub fn publish(self) -> Document<Published> {
        Document {
            title: self.title,
            content: self.content,
            revision: self.revision,
            _state: PhantomData,
        }
    }

    /// Get the current content.
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Get the revision number.
    pub fn revision(&self) -> u32 {
        self.revision
    }
}

impl Document<Published> {
    /// Read the content (available in Published).
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Get the title.
    pub fn title(&self) -> &str {
        &self.title
    }

    /// Archive the document.
    pub fn archive(self) -> Document<Archived> {
        Document {
            title: self.title,
            content: self.content,
            revision: self.revision,
            _state: PhantomData,
        }
    }

    /// Unpublish back to draft.
    pub fn unpublish(self) -> Document<Draft> {
        Document {
            title: self.title,
            content: self.content,
            revision: self.revision,
            _state: PhantomData,
        }
    }
}

impl Document<Archived> {
    /// Get the title.
    pub fn title(&self) -> &str {
        &self.title
    }

    /// Get the content (read-only in archived state).
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Get the final revision.
    pub fn revision(&self) -> u32 {
        self.revision
    }
}

// ── Type-level Handle with Permissions ──────────────────────────

/// A handle to a resource, parameterized by permission level.
#[derive(Debug)]
pub struct Handle<Perm> {
    resource_id: String,
    data: Vec<u8>,
    _perm: PhantomData<Perm>,
}

impl Handle<ReadPerm> {
    /// Create a read-only handle.
    pub fn read_only(id: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            resource_id: id.into(),
            data,
            _perm: PhantomData,
        }
    }

    /// Read the data.
    pub fn read(&self) -> &[u8] {
        &self.data
    }

    /// Upgrade to read-write (would require auth in real code).
    pub fn upgrade(self) -> Handle<ReadWritePerm> {
        Handle {
            resource_id: self.resource_id,
            data: self.data,
            _perm: PhantomData,
        }
    }

    /// Get the resource id.
    pub fn resource_id(&self) -> &str {
        &self.resource_id
    }
}

impl Handle<WritePerm> {
    /// Create a write-only handle.
    pub fn write_only(id: impl Into<String>) -> Self {
        Self {
            resource_id: id.into(),
            data: Vec::new(),
            _perm: PhantomData,
        }
    }

    /// Write data.
    pub fn write(&mut self, data: Vec<u8>) {
        self.data = data;
    }

    /// Get the resource id.
    pub fn resource_id(&self) -> &str {
        &self.resource_id
    }
}

impl Handle<ReadWritePerm> {
    /// Create a read-write handle.
    pub fn read_write(id: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            resource_id: id.into(),
            data,
            _perm: PhantomData,
        }
    }

    /// Read the data.
    pub fn read(&self) -> &[u8] {
        &self.data
    }

    /// Write data.
    pub fn write(&mut self, data: Vec<u8>) {
        self.data = data;
    }

    /// Downgrade to read-only.
    pub fn downgrade(self) -> Handle<ReadPerm> {
        Handle {
            resource_id: self.resource_id,
            data: self.data,
            _perm: PhantomData,
        }
    }
}

// ── Unit-Safe Arithmetic ────────────────────────────────────────

/// A unit marker trait.
pub trait Unit: fmt::Debug + Clone + Copy + PartialEq {
    /// Unit name for display.
    fn name() -> &'static str;
}

/// A quantity: a numeric value tagged with a unit.
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Quantity<U: Unit> {
    value: f64,
    #[serde(skip)]
    _unit: PhantomData<U>,
}

impl<U: Unit> fmt::Debug for Quantity<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.value, U::name())
    }
}

impl<U: Unit> fmt::Display for Quantity<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.value, U::name())
    }
}

impl<U: Unit> PartialEq for Quantity<U> {
    fn eq(&self, other: &Self) -> bool {
        (self.value - other.value).abs() < f64::EPSILON
    }
}

impl<U: Unit> Quantity<U> {
    /// Create a new quantity.
    pub fn new(value: f64) -> Self {
        Self { value, _unit: PhantomData }
    }

    /// Get the raw value.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Scale by a dimensionless factor.
    pub fn scale(self, factor: f64) -> Self {
        Self::new(self.value * factor)
    }

    /// Absolute value.
    pub fn abs(self) -> Self {
        Self::new(self.value.abs())
    }

    /// Whether the quantity is zero.
    pub fn is_zero(&self) -> bool {
        self.value.abs() < f64::EPSILON
    }
}

impl<U: Unit> Add for Quantity<U> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.value + rhs.value)
    }
}

impl<U: Unit> Sub for Quantity<U> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.value - rhs.value)
    }
}

// ── Concrete Units ──────────────────────────────────────────────

/// Meters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Meters;
impl Unit for Meters {
    fn name() -> &'static str { "m" }
}

/// Square meters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SquareMeters;
impl Unit for SquareMeters {
    fn name() -> &'static str { "m^2" }
}

/// Seconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Seconds;
impl Unit for Seconds {
    fn name() -> &'static str { "s" }
}

/// Meters per second.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetersPerSecond;
impl Unit for MetersPerSecond {
    fn name() -> &'static str { "m/s" }
}

/// Kilograms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Kilograms;
impl Unit for Kilograms {
    fn name() -> &'static str { "kg" }
}

/// Joules (energy).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Joules;
impl Unit for Joules {
    fn name() -> &'static str { "J" }
}

// ── Unit multiplication ─────────────────────────────────────────

/// meters * meters = square_meters.
impl Mul<Quantity<Meters>> for Quantity<Meters> {
    type Output = Quantity<SquareMeters>;
    fn mul(self, rhs: Quantity<Meters>) -> Quantity<SquareMeters> {
        Quantity::new(self.value * rhs.value)
    }
}

/// distance / time = velocity (as a function, since Div is tricky).
pub fn velocity(distance: Quantity<Meters>, time: Quantity<Seconds>) -> Quantity<MetersPerSecond> {
    Quantity::new(distance.value() / time.value())
}

/// energy = mass * velocity^2 (simplified E = 0.5 * m * v^2).
pub fn kinetic_energy(mass: Quantity<Kilograms>, vel: Quantity<MetersPerSecond>) -> Quantity<Joules> {
    Quantity::new(0.5 * mass.value() * vel.value() * vel.value())
}

// ── Type-level Flag ─────────────────────────────────────────────

/// A value with a boolean flag at the type level.
#[derive(Debug, Clone)]
pub struct Flagged<T, Flag> {
    value: T,
    _flag: PhantomData<Flag>,
}

/// Flag: validated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IsValidated;

/// Flag: not validated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NotValidated;

/// Flag: sanitized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sanitized;

/// Flag: not sanitized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NotSanitized;

impl<T> Flagged<T, NotValidated> {
    /// Create an unvalidated value.
    pub fn new(value: T) -> Self {
        Self { value, _flag: PhantomData }
    }

    /// Validate, transitioning to IsValidated on success.
    pub fn validate(self, f: impl FnOnce(&T) -> bool) -> Option<Flagged<T, IsValidated>> {
        if f(&self.value) {
            Some(Flagged { value: self.value, _flag: PhantomData })
        } else {
            None
        }
    }

    /// Access the raw value.
    pub fn raw(&self) -> &T {
        &self.value
    }
}

impl<T> Flagged<T, IsValidated> {
    /// Access the validated value.
    pub fn value(&self) -> &T {
        &self.value
    }

    /// Consume and return the validated value.
    pub fn into_inner(self) -> T {
        self.value
    }
}

impl<T> Flagged<T, NotSanitized> {
    /// Create an unsanitized value.
    pub fn unsanitized(value: T) -> Self {
        Self { value, _flag: PhantomData }
    }

    /// Sanitize, transitioning to Sanitized.
    pub fn sanitize(self, f: impl FnOnce(T) -> T) -> Flagged<T, Sanitized> {
        Flagged { value: f(self.value), _flag: PhantomData }
    }
}

impl<T> Flagged<T, Sanitized> {
    /// Access the sanitized value.
    pub fn value(&self) -> &T {
        &self.value
    }

    /// Consume and return.
    pub fn into_inner(self) -> T {
        self.value
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_lifecycle() {
        let mut ch = Channel::<Open>::new("test");
        ch.send("hello");
        ch.send("world");
        assert_eq!(ch.message_count(), 2);

        let closed = ch.close();
        assert_eq!(closed.messages().len(), 2);
        assert_eq!(closed.messages()[0], "hello");

        let locked = closed.lock();
        assert_eq!(locked.name(), "test");
        assert_eq!(locked.archived_count(), 2);
    }

    #[test]
    fn test_channel_reopen() {
        let mut ch = Channel::<Open>::new("ch");
        ch.send("first");
        let closed = ch.close();
        let mut reopened = closed.reopen();
        reopened.send("second");
        assert_eq!(reopened.message_count(), 2);
    }

    #[test]
    fn test_document_lifecycle() {
        let mut doc = Document::<Draft>::new("My Doc");
        doc.edit("Hello, world!");
        assert_eq!(doc.content(), "Hello, world!");
        assert_eq!(doc.revision(), 1);

        let published = doc.publish();
        assert_eq!(published.content(), "Hello, world!");
        assert_eq!(published.title(), "My Doc");

        let archived = published.archive();
        assert_eq!(archived.title(), "My Doc");
        assert_eq!(archived.revision(), 1);
    }

    #[test]
    fn test_document_unpublish() {
        let mut doc = Document::<Draft>::new("Test");
        doc.edit("v1");
        let published = doc.publish();
        let mut draft = published.unpublish();
        draft.edit("v2");
        assert_eq!(draft.content(), "v2");
        assert_eq!(draft.revision(), 2);
    }

    #[test]
    fn test_handle_read_only() {
        let h = Handle::<ReadPerm>::read_only("file1", vec![1, 2, 3]);
        assert_eq!(h.read(), &[1, 2, 3]);
        assert_eq!(h.resource_id(), "file1");
    }

    #[test]
    fn test_handle_upgrade_downgrade() {
        let h = Handle::<ReadPerm>::read_only("f1", vec![1, 2]);
        let mut rw = h.upgrade();
        assert_eq!(rw.read(), &[1, 2]);
        rw.write(vec![3, 4]);
        assert_eq!(rw.read(), &[3, 4]);
        let ro = rw.downgrade();
        assert_eq!(ro.read(), &[3, 4]);
    }

    #[test]
    fn test_handle_write_only() {
        let mut h = Handle::<WritePerm>::write_only("f2");
        h.write(vec![10, 20]);
        assert_eq!(h.resource_id(), "f2");
    }

    #[test]
    fn test_quantity_add_sub() {
        let a = Quantity::<Meters>::new(3.0);
        let b = Quantity::<Meters>::new(2.0);
        let sum = a + b;
        assert_eq!(sum.value(), 5.0);
        let diff = a - b;
        assert_eq!(diff.value(), 1.0);
    }

    #[test]
    fn test_quantity_scale() {
        let a = Quantity::<Meters>::new(5.0);
        let scaled = a.scale(2.0);
        assert_eq!(scaled.value(), 10.0);
    }

    #[test]
    fn test_quantity_mul_meters() {
        let width = Quantity::<Meters>::new(3.0);
        let height = Quantity::<Meters>::new(4.0);
        let area: Quantity<SquareMeters> = width * height;
        assert_eq!(area.value(), 12.0);
    }

    #[test]
    fn test_velocity() {
        let d = Quantity::<Meters>::new(100.0);
        let t = Quantity::<Seconds>::new(10.0);
        let v = velocity(d, t);
        assert_eq!(v.value(), 10.0);
        assert_eq!(format!("{v}"), "10 m/s");
    }

    #[test]
    fn test_kinetic_energy() {
        let m = Quantity::<Kilograms>::new(2.0);
        let v = Quantity::<MetersPerSecond>::new(3.0);
        let e = kinetic_energy(m, v);
        assert_eq!(e.value(), 9.0); // 0.5 * 2 * 9
    }

    #[test]
    fn test_quantity_display() {
        let m = Quantity::<Meters>::new(42.0);
        assert_eq!(format!("{m}"), "42 m");
    }

    #[test]
    fn test_quantity_abs() {
        let m = Quantity::<Meters>::new(-5.0);
        assert_eq!(m.abs().value(), 5.0);
    }

    #[test]
    fn test_quantity_is_zero() {
        let zero = Quantity::<Meters>::new(0.0);
        assert!(zero.is_zero());
        let nonzero = Quantity::<Meters>::new(1.0);
        assert!(!nonzero.is_zero());
    }

    #[test]
    fn test_flagged_validation_pass() {
        let raw = Flagged::<i32, NotValidated>::new(42);
        let validated = raw.validate(|x| *x > 0).unwrap();
        assert_eq!(*validated.value(), 42);
    }

    #[test]
    fn test_flagged_validation_fail() {
        let raw = Flagged::<i32, NotValidated>::new(-1);
        assert!(raw.validate(|x| *x > 0).is_none());
    }

    #[test]
    fn test_flagged_sanitize() {
        let raw = Flagged::<String, NotSanitized>::unsanitized("<script>alert(1)</script>".to_string());
        let clean = raw.sanitize(|s| s.replace('<', "&lt;").replace('>', "&gt;"));
        assert_eq!(clean.value(), "&lt;script&gt;alert(1)&lt;/script&gt;");
    }

    #[test]
    fn test_flagged_into_inner() {
        let raw = Flagged::<i32, NotValidated>::new(10);
        let validated = raw.validate(|_| true).unwrap();
        assert_eq!(validated.into_inner(), 10);
    }

    #[test]
    fn test_flagged_raw_access() {
        let raw = Flagged::<i32, NotValidated>::new(99);
        assert_eq!(*raw.raw(), 99);
    }

    #[test]
    fn test_quantity_equality() {
        let a = Quantity::<Meters>::new(3.0);
        let b = Quantity::<Meters>::new(3.0);
        assert_eq!(a, b);
    }
}
