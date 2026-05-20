//! IPC abstractions — named pipes (FIFO buffer), shared memory regions,
//! message passing (typed mailbox), semaphore, reader-writer lock simulation,
//! select/poll for multiple channels, channel statistics.

use std::collections::{HashMap, VecDeque};

// ── Error ───────────────────────────────────────────────────────────────────

/// IPC errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IpcError {
    ChannelClosed(String),
    ChannelFull(String),
    ChannelEmpty(String),
    NotFound(String),
    AlreadyExists(String),
    WouldBlock,
    PermissionDenied(String),
    InvalidOperation(String),
}

impl std::fmt::Display for IpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IpcError::ChannelClosed(n) => write!(f, "channel closed: {n}"),
            IpcError::ChannelFull(n) => write!(f, "channel full: {n}"),
            IpcError::ChannelEmpty(n) => write!(f, "channel empty: {n}"),
            IpcError::NotFound(n) => write!(f, "not found: {n}"),
            IpcError::AlreadyExists(n) => write!(f, "already exists: {n}"),
            IpcError::WouldBlock => write!(f, "would block"),
            IpcError::PermissionDenied(n) => write!(f, "permission denied: {n}"),
            IpcError::InvalidOperation(msg) => write!(f, "invalid operation: {msg}"),
        }
    }
}

// ── Named Pipe (FIFO) ──────────────────────────────────────────────────────

/// A named pipe (FIFO) with a bounded buffer.
#[derive(Debug)]
pub struct NamedPipe {
    name: String,
    buffer: VecDeque<Vec<u8>>,
    capacity: usize,
    closed: bool,
    bytes_written: u64,
    bytes_read: u64,
    write_count: u64,
    read_count: u64,
}

impl NamedPipe {
    pub fn new(name: &str, capacity: usize) -> Self {
        Self {
            name: name.to_string(),
            buffer: VecDeque::new(),
            capacity,
            closed: false,
            bytes_written: 0,
            bytes_read: 0,
            write_count: 0,
            read_count: 0,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Write data into the pipe.
    pub fn write(&mut self, data: &[u8]) -> Result<(), IpcError> {
        if self.closed {
            return Err(IpcError::ChannelClosed(self.name.clone()));
        }
        if self.buffer.len() >= self.capacity {
            return Err(IpcError::ChannelFull(self.name.clone()));
        }
        self.bytes_written += data.len() as u64;
        self.write_count += 1;
        self.buffer.push_back(data.to_vec());
        Ok(())
    }

    /// Read data from the pipe.
    pub fn read(&mut self) -> Result<Vec<u8>, IpcError> {
        if let Some(data) = self.buffer.pop_front() {
            self.bytes_read += data.len() as u64;
            self.read_count += 1;
            Ok(data)
        } else if self.closed {
            Err(IpcError::ChannelClosed(self.name.clone()))
        } else {
            Err(IpcError::ChannelEmpty(self.name.clone()))
        }
    }

    /// Try to read without blocking (returns None if empty).
    pub fn try_read(&mut self) -> Option<Vec<u8>> {
        if let Some(data) = self.buffer.pop_front() {
            self.bytes_read += data.len() as u64;
            self.read_count += 1;
            Some(data)
        } else {
            None
        }
    }

    /// Close the pipe.
    pub fn close(&mut self) {
        self.closed = true;
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }
}

// ── Shared Memory Region ───────────────────────────────────────────────────

/// A shared memory region accessible by name.
#[derive(Debug)]
pub struct SharedMemory {
    name: String,
    data: Vec<u8>,
    readers: u32,
    writers: u32,
    access_count: u64,
}

impl SharedMemory {
    pub fn new(name: &str, size: usize) -> Self {
        Self {
            name: name.to_string(),
            data: vec![0; size],
            readers: 0,
            writers: 0,
            access_count: 0,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Read bytes from the shared memory at offset.
    pub fn read(&mut self, offset: usize, len: usize) -> Result<Vec<u8>, IpcError> {
        if offset + len > self.data.len() {
            return Err(IpcError::InvalidOperation(format!(
                "read out of bounds: offset={offset}, len={len}, size={}",
                self.data.len()
            )));
        }
        self.access_count += 1;
        Ok(self.data[offset..offset + len].to_vec())
    }

    /// Write bytes to shared memory at offset.
    pub fn write(&mut self, offset: usize, data: &[u8]) -> Result<(), IpcError> {
        if offset + data.len() > self.data.len() {
            return Err(IpcError::InvalidOperation(format!(
                "write out of bounds: offset={offset}, len={}, size={}",
                data.len(),
                self.data.len()
            )));
        }
        self.access_count += 1;
        self.data[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Attach as reader.
    pub fn attach_reader(&mut self) {
        self.readers += 1;
    }

    /// Attach as writer.
    pub fn attach_writer(&mut self) {
        self.writers += 1;
    }

    /// Detach reader.
    pub fn detach_reader(&mut self) {
        self.readers = self.readers.saturating_sub(1);
    }

    /// Detach writer.
    pub fn detach_writer(&mut self) {
        self.writers = self.writers.saturating_sub(1);
    }

    pub fn reader_count(&self) -> u32 {
        self.readers
    }

    pub fn writer_count(&self) -> u32 {
        self.writers
    }

    pub fn access_count(&self) -> u64 {
        self.access_count
    }
}

// ── Mailbox (Typed Message Passing) ─────────────────────────────────────────

/// A typed message for the mailbox.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub sender: String,
    pub payload: String,
    pub priority: u8,
    pub timestamp: u64,
}

/// A typed mailbox for message passing between processes.
#[derive(Debug)]
pub struct Mailbox {
    name: String,
    messages: VecDeque<Message>,
    capacity: usize,
    closed: bool,
    total_sent: u64,
    total_received: u64,
}

impl Mailbox {
    pub fn new(name: &str, capacity: usize) -> Self {
        Self {
            name: name.to_string(),
            messages: VecDeque::new(),
            capacity,
            closed: false,
            total_sent: 0,
            total_received: 0,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Send a message.
    pub fn send(&mut self, msg: Message) -> Result<(), IpcError> {
        if self.closed {
            return Err(IpcError::ChannelClosed(self.name.clone()));
        }
        if self.messages.len() >= self.capacity {
            return Err(IpcError::ChannelFull(self.name.clone()));
        }
        self.total_sent += 1;
        self.messages.push_back(msg);
        Ok(())
    }

    /// Receive the next message (FIFO).
    pub fn receive(&mut self) -> Result<Message, IpcError> {
        if let Some(msg) = self.messages.pop_front() {
            self.total_received += 1;
            Ok(msg)
        } else if self.closed {
            Err(IpcError::ChannelClosed(self.name.clone()))
        } else {
            Err(IpcError::ChannelEmpty(self.name.clone()))
        }
    }

    /// Receive the highest priority message.
    pub fn receive_priority(&mut self) -> Result<Message, IpcError> {
        if self.messages.is_empty() {
            return if self.closed {
                Err(IpcError::ChannelClosed(self.name.clone()))
            } else {
                Err(IpcError::ChannelEmpty(self.name.clone()))
            };
        }
        let idx = self
            .messages
            .iter()
            .enumerate()
            .max_by_key(|(_, m)| m.priority)
            .unwrap()
            .0;
        let msg = self.messages.remove(idx).unwrap();
        self.total_received += 1;
        Ok(msg)
    }

    /// Peek at the next message without removing it.
    pub fn peek(&self) -> Option<&Message> {
        self.messages.front()
    }

    pub fn close(&mut self) {
        self.closed = true;
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    pub fn total_sent(&self) -> u64 {
        self.total_sent
    }

    pub fn total_received(&self) -> u64 {
        self.total_received
    }
}

// ── Semaphore ───────────────────────────────────────────────────────────────

/// Counting semaphore.
#[derive(Debug)]
pub struct Semaphore {
    name: String,
    value: i32,
    max_value: i32,
    waiters: u32,
    acquire_count: u64,
    release_count: u64,
}

impl Semaphore {
    pub fn new(name: &str, initial: i32) -> Self {
        Self {
            name: name.to_string(),
            value: initial,
            max_value: initial,
            waiters: 0,
            acquire_count: 0,
            release_count: 0,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Try to acquire the semaphore (P/wait).
    pub fn acquire(&mut self) -> Result<(), IpcError> {
        if self.value > 0 {
            self.value -= 1;
            self.acquire_count += 1;
            Ok(())
        } else {
            self.waiters += 1;
            Err(IpcError::WouldBlock)
        }
    }

    /// Release the semaphore (V/signal).
    pub fn release(&mut self) -> Result<(), IpcError> {
        if self.value >= self.max_value {
            return Err(IpcError::InvalidOperation(format!(
                "semaphore {} already at max",
                self.name
            )));
        }
        self.value += 1;
        self.release_count += 1;
        if self.waiters > 0 {
            self.waiters -= 1;
        }
        Ok(())
    }

    pub fn value(&self) -> i32 {
        self.value
    }

    pub fn waiters(&self) -> u32 {
        self.waiters
    }

    pub fn acquire_count(&self) -> u64 {
        self.acquire_count
    }

    pub fn release_count(&self) -> u64 {
        self.release_count
    }
}

// ── Reader-Writer Lock ─────────────────────────────────────────────────────

/// Reader-writer lock simulation.
#[derive(Debug)]
pub struct RwLockSim {
    name: String,
    readers: u32,
    writer: bool,
    waiting_readers: u32,
    waiting_writers: u32,
    read_count: u64,
    write_count: u64,
}

impl RwLockSim {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            readers: 0,
            writer: false,
            waiting_readers: 0,
            waiting_writers: 0,
            read_count: 0,
            write_count: 0,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Acquire a read lock. Fails if a writer is active.
    pub fn read_lock(&mut self) -> Result<(), IpcError> {
        if self.writer {
            self.waiting_readers += 1;
            Err(IpcError::WouldBlock)
        } else {
            self.readers += 1;
            self.read_count += 1;
            Ok(())
        }
    }

    /// Release a read lock.
    pub fn read_unlock(&mut self) -> Result<(), IpcError> {
        if self.readers == 0 {
            return Err(IpcError::InvalidOperation("no read lock held".into()));
        }
        self.readers -= 1;
        Ok(())
    }

    /// Acquire a write lock. Fails if readers or another writer are active.
    pub fn write_lock(&mut self) -> Result<(), IpcError> {
        if self.writer || self.readers > 0 {
            self.waiting_writers += 1;
            Err(IpcError::WouldBlock)
        } else {
            self.writer = true;
            self.write_count += 1;
            Ok(())
        }
    }

    /// Release a write lock.
    pub fn write_unlock(&mut self) -> Result<(), IpcError> {
        if !self.writer {
            return Err(IpcError::InvalidOperation("no write lock held".into()));
        }
        self.writer = false;
        // Wake waiting readers
        let to_wake = self.waiting_readers;
        self.waiting_readers = 0;
        self.readers += to_wake;
        self.read_count += to_wake as u64;
        Ok(())
    }

    pub fn active_readers(&self) -> u32 {
        self.readers
    }

    pub fn is_write_locked(&self) -> bool {
        self.writer
    }

    pub fn waiting_readers(&self) -> u32 {
        self.waiting_readers
    }

    pub fn waiting_writers(&self) -> u32 {
        self.waiting_writers
    }

    pub fn total_reads(&self) -> u64 {
        self.read_count
    }

    pub fn total_writes(&self) -> u64 {
        self.write_count
    }
}

// ── Channel Registry with Select/Poll ──────────────────────────────────────

/// Event type returned by select/poll.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChannelEvent {
    PipeReadable(String),
    MailboxReadable(String),
}

/// Registry for managing multiple IPC channels.
#[derive(Debug)]
pub struct ChannelRegistry {
    pipes: HashMap<String, NamedPipe>,
    mailboxes: HashMap<String, Mailbox>,
    shared_memory: HashMap<String, SharedMemory>,
    semaphores: HashMap<String, Semaphore>,
    rw_locks: HashMap<String, RwLockSim>,
}

impl ChannelRegistry {
    pub fn new() -> Self {
        Self {
            pipes: HashMap::new(),
            mailboxes: HashMap::new(),
            shared_memory: HashMap::new(),
            semaphores: HashMap::new(),
            rw_locks: HashMap::new(),
        }
    }

    /// Create a named pipe.
    pub fn create_pipe(&mut self, name: &str, capacity: usize) -> Result<(), IpcError> {
        if self.pipes.contains_key(name) {
            return Err(IpcError::AlreadyExists(name.into()));
        }
        self.pipes
            .insert(name.to_string(), NamedPipe::new(name, capacity));
        Ok(())
    }

    /// Create a mailbox.
    pub fn create_mailbox(&mut self, name: &str, capacity: usize) -> Result<(), IpcError> {
        if self.mailboxes.contains_key(name) {
            return Err(IpcError::AlreadyExists(name.into()));
        }
        self.mailboxes
            .insert(name.to_string(), Mailbox::new(name, capacity));
        Ok(())
    }

    /// Create shared memory.
    pub fn create_shared_memory(&mut self, name: &str, size: usize) -> Result<(), IpcError> {
        if self.shared_memory.contains_key(name) {
            return Err(IpcError::AlreadyExists(name.into()));
        }
        self.shared_memory
            .insert(name.to_string(), SharedMemory::new(name, size));
        Ok(())
    }

    /// Create a semaphore.
    pub fn create_semaphore(&mut self, name: &str, initial: i32) -> Result<(), IpcError> {
        if self.semaphores.contains_key(name) {
            return Err(IpcError::AlreadyExists(name.into()));
        }
        self.semaphores
            .insert(name.to_string(), Semaphore::new(name, initial));
        Ok(())
    }

    /// Create a reader-writer lock.
    pub fn create_rw_lock(&mut self, name: &str) -> Result<(), IpcError> {
        if self.rw_locks.contains_key(name) {
            return Err(IpcError::AlreadyExists(name.into()));
        }
        self.rw_locks
            .insert(name.to_string(), RwLockSim::new(name));
        Ok(())
    }

    /// Get a mutable reference to a pipe.
    pub fn pipe_mut(&mut self, name: &str) -> Result<&mut NamedPipe, IpcError> {
        self.pipes
            .get_mut(name)
            .ok_or_else(|| IpcError::NotFound(name.into()))
    }

    /// Get a mutable reference to a mailbox.
    pub fn mailbox_mut(&mut self, name: &str) -> Result<&mut Mailbox, IpcError> {
        self.mailboxes
            .get_mut(name)
            .ok_or_else(|| IpcError::NotFound(name.into()))
    }

    /// Get a mutable reference to shared memory.
    pub fn shared_memory_mut(&mut self, name: &str) -> Result<&mut SharedMemory, IpcError> {
        self.shared_memory
            .get_mut(name)
            .ok_or_else(|| IpcError::NotFound(name.into()))
    }

    /// Get a mutable reference to a semaphore.
    pub fn semaphore_mut(&mut self, name: &str) -> Result<&mut Semaphore, IpcError> {
        self.semaphores
            .get_mut(name)
            .ok_or_else(|| IpcError::NotFound(name.into()))
    }

    /// Get a mutable reference to a rw lock.
    pub fn rw_lock_mut(&mut self, name: &str) -> Result<&mut RwLockSim, IpcError> {
        self.rw_locks
            .get_mut(name)
            .ok_or_else(|| IpcError::NotFound(name.into()))
    }

    /// Poll all channels and return which ones have data ready.
    pub fn poll(&self) -> Vec<ChannelEvent> {
        let mut events = Vec::new();
        for (name, pipe) in &self.pipes {
            if !pipe.is_empty() {
                events.push(ChannelEvent::PipeReadable(name.clone()));
            }
        }
        for (name, mb) in &self.mailboxes {
            if !mb.is_empty() {
                events.push(ChannelEvent::MailboxReadable(name.clone()));
            }
        }
        events.sort_by(|a, b| {
            let name_a = match a {
                ChannelEvent::PipeReadable(n) | ChannelEvent::MailboxReadable(n) => n,
            };
            let name_b = match b {
                ChannelEvent::PipeReadable(n) | ChannelEvent::MailboxReadable(n) => n,
            };
            name_a.cmp(name_b)
        });
        events
    }

    /// Select: wait for any of the named channels to have data.
    /// Returns the first event found.
    pub fn select(&self, names: &[&str]) -> Option<ChannelEvent> {
        for name in names {
            if let Some(pipe) = self.pipes.get(*name) {
                if !pipe.is_empty() {
                    return Some(ChannelEvent::PipeReadable(name.to_string()));
                }
            }
            if let Some(mb) = self.mailboxes.get(*name) {
                if !mb.is_empty() {
                    return Some(ChannelEvent::MailboxReadable(name.to_string()));
                }
            }
        }
        None
    }

    /// Number of registered channels (pipes + mailboxes).
    pub fn channel_count(&self) -> usize {
        self.pipes.len() + self.mailboxes.len()
    }

    /// Destroy a pipe.
    pub fn destroy_pipe(&mut self, name: &str) -> Result<(), IpcError> {
        self.pipes
            .remove(name)
            .map(|_| ())
            .ok_or_else(|| IpcError::NotFound(name.into()))
    }

    /// Destroy a mailbox.
    pub fn destroy_mailbox(&mut self, name: &str) -> Result<(), IpcError> {
        self.mailboxes
            .remove(name)
            .map(|_| ())
            .ok_or_else(|| IpcError::NotFound(name.into()))
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipe_write_read() {
        let mut pipe = NamedPipe::new("test", 10);
        pipe.write(b"hello").unwrap();
        let data = pipe.read().unwrap();
        assert_eq!(data, b"hello");
    }

    #[test]
    fn test_pipe_fifo_order() {
        let mut pipe = NamedPipe::new("test", 10);
        pipe.write(b"first").unwrap();
        pipe.write(b"second").unwrap();
        assert_eq!(pipe.read().unwrap(), b"first");
        assert_eq!(pipe.read().unwrap(), b"second");
    }

    #[test]
    fn test_pipe_full() {
        let mut pipe = NamedPipe::new("test", 2);
        pipe.write(b"a").unwrap();
        pipe.write(b"b").unwrap();
        let result = pipe.write(b"c");
        assert!(matches!(result, Err(IpcError::ChannelFull(_))));
    }

    #[test]
    fn test_pipe_closed() {
        let mut pipe = NamedPipe::new("test", 10);
        pipe.close();
        assert!(pipe.write(b"x").is_err());
        assert!(pipe.read().is_err());
    }

    #[test]
    fn test_pipe_stats() {
        let mut pipe = NamedPipe::new("test", 10);
        pipe.write(b"hello").unwrap();
        pipe.write(b"world").unwrap();
        let _ = pipe.read().unwrap();
        assert_eq!(pipe.bytes_written(), 10);
        assert_eq!(pipe.bytes_read(), 5);
    }

    #[test]
    fn test_shared_memory_read_write() {
        let mut shm = SharedMemory::new("shm0", 256);
        shm.write(0, b"hello").unwrap();
        let data = shm.read(0, 5).unwrap();
        assert_eq!(data, b"hello");
    }

    #[test]
    fn test_shared_memory_bounds() {
        let mut shm = SharedMemory::new("shm0", 10);
        let result = shm.write(8, b"abc");
        assert!(result.is_err());
    }

    #[test]
    fn test_shared_memory_attachments() {
        let mut shm = SharedMemory::new("shm0", 64);
        shm.attach_reader();
        shm.attach_reader();
        shm.attach_writer();
        assert_eq!(shm.reader_count(), 2);
        assert_eq!(shm.writer_count(), 1);
        shm.detach_reader();
        assert_eq!(shm.reader_count(), 1);
    }

    #[test]
    fn test_mailbox_send_receive() {
        let mut mb = Mailbox::new("mbox", 10);
        mb.send(Message {
            sender: "proc1".into(),
            payload: "hi".into(),
            priority: 1,
            timestamp: 100,
        })
        .unwrap();
        let msg = mb.receive().unwrap();
        assert_eq!(msg.sender, "proc1");
        assert_eq!(msg.payload, "hi");
    }

    #[test]
    fn test_mailbox_priority() {
        let mut mb = Mailbox::new("mbox", 10);
        mb.send(Message {
            sender: "a".into(),
            payload: "low".into(),
            priority: 1,
            timestamp: 1,
        })
        .unwrap();
        mb.send(Message {
            sender: "b".into(),
            payload: "high".into(),
            priority: 10,
            timestamp: 2,
        })
        .unwrap();
        let msg = mb.receive_priority().unwrap();
        assert_eq!(msg.priority, 10);
        assert_eq!(msg.payload, "high");
    }

    #[test]
    fn test_semaphore_acquire_release() {
        let mut sem = Semaphore::new("mutex", 1);
        sem.acquire().unwrap();
        assert_eq!(sem.value(), 0);
        // Second acquire should block
        assert!(sem.acquire().is_err());
        assert_eq!(sem.waiters(), 1);
        sem.release().unwrap();
        assert_eq!(sem.value(), 1);
    }

    #[test]
    fn test_semaphore_counting() {
        let mut sem = Semaphore::new("slots", 3);
        sem.acquire().unwrap();
        sem.acquire().unwrap();
        sem.acquire().unwrap();
        assert_eq!(sem.value(), 0);
        assert!(sem.acquire().is_err());
    }

    #[test]
    fn test_rw_lock_multiple_readers() {
        let mut lock = RwLockSim::new("data");
        lock.read_lock().unwrap();
        lock.read_lock().unwrap();
        lock.read_lock().unwrap();
        assert_eq!(lock.active_readers(), 3);
        assert!(!lock.is_write_locked());
    }

    #[test]
    fn test_rw_lock_writer_blocks_reader() {
        let mut lock = RwLockSim::new("data");
        lock.write_lock().unwrap();
        assert!(lock.read_lock().is_err());
        assert_eq!(lock.waiting_readers(), 1);
    }

    #[test]
    fn test_rw_lock_reader_blocks_writer() {
        let mut lock = RwLockSim::new("data");
        lock.read_lock().unwrap();
        assert!(lock.write_lock().is_err());
        assert_eq!(lock.waiting_writers(), 1);
    }

    #[test]
    fn test_rw_lock_unlock_wakes_readers() {
        let mut lock = RwLockSim::new("data");
        lock.write_lock().unwrap();
        // Simulate waiting readers
        let _ = lock.read_lock(); // Would block
        let _ = lock.read_lock(); // Would block
        assert_eq!(lock.waiting_readers(), 2);
        lock.write_unlock().unwrap();
        assert_eq!(lock.active_readers(), 2);
        assert_eq!(lock.waiting_readers(), 0);
    }

    #[test]
    fn test_registry_create_and_use() {
        let mut reg = ChannelRegistry::new();
        reg.create_pipe("p1", 10).unwrap();
        reg.create_mailbox("mb1", 10).unwrap();
        assert_eq!(reg.channel_count(), 2);
        reg.pipe_mut("p1").unwrap().write(b"data").unwrap();
    }

    #[test]
    fn test_registry_poll() {
        let mut reg = ChannelRegistry::new();
        reg.create_pipe("p1", 10).unwrap();
        reg.create_pipe("p2", 10).unwrap();
        reg.pipe_mut("p1").unwrap().write(b"data").unwrap();
        let events = reg.poll();
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], ChannelEvent::PipeReadable(n) if n == "p1"));
    }

    #[test]
    fn test_registry_select() {
        let mut reg = ChannelRegistry::new();
        reg.create_pipe("p1", 10).unwrap();
        reg.create_mailbox("mb1", 10).unwrap();
        reg.mailbox_mut("mb1").unwrap().send(Message {
            sender: "x".into(),
            payload: "y".into(),
            priority: 1,
            timestamp: 0,
        }).unwrap();
        let event = reg.select(&["p1", "mb1"]);
        assert!(matches!(event, Some(ChannelEvent::MailboxReadable(n)) if n == "mb1"));
    }

    #[test]
    fn test_registry_destroy() {
        let mut reg = ChannelRegistry::new();
        reg.create_pipe("p1", 10).unwrap();
        reg.destroy_pipe("p1").unwrap();
        assert!(reg.pipe_mut("p1").is_err());
    }

    #[test]
    fn test_pipe_try_read() {
        let mut pipe = NamedPipe::new("test", 10);
        assert!(pipe.try_read().is_none());
        pipe.write(b"data").unwrap();
        assert!(pipe.try_read().is_some());
    }

    #[test]
    fn test_mailbox_peek() {
        let mut mb = Mailbox::new("mbox", 10);
        assert!(mb.peek().is_none());
        mb.send(Message {
            sender: "s".into(),
            payload: "p".into(),
            priority: 1,
            timestamp: 0,
        }).unwrap();
        assert_eq!(mb.peek().unwrap().payload, "p");
        assert_eq!(mb.len(), 1); // Peek doesn't remove
    }
}
