//! Streams API — ReadableStream, WritableStream, TransformStream.
//!
//! Headless implementation of the WHATWG Streams pattern with backpressure
//! support and transform piping.

use std::collections::VecDeque;

// ── Chunk ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Chunk<T> {
    Data(T),
    End,
    Error(String),
}

// ── ReadableStream ──────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct ReadableStream<T> {
    buffer: VecDeque<Chunk<T>>,
    closed: bool,
}

impl<T> ReadableStream<T> {
    pub fn new() -> Self {
        Self {
            buffer: VecDeque::new(),
            closed: false,
        }
    }

    pub fn push(&mut self, item: T) {
        if !self.closed {
            self.buffer.push_back(Chunk::Data(item));
        }
    }

    pub fn push_error(&mut self, e: impl Into<String>) {
        if !self.closed {
            self.buffer.push_back(Chunk::Error(e.into()));
        }
    }

    pub fn close(&mut self) {
        self.closed = true;
        self.buffer.push_back(Chunk::End);
    }

    pub fn read(&mut self) -> Option<Chunk<T>> {
        self.buffer.pop_front()
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    pub fn buffered_count(&self) -> usize {
        self.buffer.len()
    }
}

impl<T> Default for ReadableStream<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ── WritableStream ──────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct WritableStream<T> {
    buffer: VecDeque<T>,
    closed: bool,
    high_water_mark: usize,
}

impl<T> WritableStream<T> {
    pub fn new(high_water_mark: usize) -> Self {
        Self {
            buffer: VecDeque::new(),
            closed: false,
            high_water_mark,
        }
    }

    /// Write an item. Returns `false` if backpressure is active (buffer full)
    /// or the stream is closed.
    pub fn write(&mut self, item: T) -> bool {
        if self.closed || self.buffer.len() >= self.high_water_mark {
            return false;
        }
        self.buffer.push_back(item);
        true
    }

    pub fn close(&mut self) {
        self.closed = true;
    }

    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.high_water_mark
    }

    pub fn drain(&mut self) -> Vec<T> {
        self.buffer.drain(..).collect()
    }
}

// ── TransformStream ─────────────────────────────────────────────────────────

pub struct TransformStream<I, O> {
    transform: Box<dyn Fn(I) -> Vec<O>>,
    pub readable: ReadableStream<O>,
}

impl<I, O> TransformStream<I, O> {
    pub fn new(f: impl Fn(I) -> Vec<O> + 'static) -> Self {
        Self {
            transform: Box::new(f),
            readable: ReadableStream::new(),
        }
    }

    pub fn write(&mut self, input: I) {
        let outputs = (self.transform)(input);
        for o in outputs {
            self.readable.push(o);
        }
    }
}

// ── pipe ────────────────────────────────────────────────────────────────────

/// Drain all `Data` chunks from `readable` into `writable`, respecting
/// backpressure. Returns the number of items successfully written.
pub fn pipe<T>(readable: &mut ReadableStream<T>, writable: &mut WritableStream<T>) -> usize {
    let mut count = 0;
    while let Some(chunk) = readable.read() {
        match chunk {
            Chunk::Data(item) => {
                if writable.write(item) {
                    count += 1;
                } else {
                    break;
                }
            }
            Chunk::End | Chunk::Error(_) => break,
        }
    }
    count
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_read_fifo() {
        let mut rs = ReadableStream::new();
        rs.push(1);
        rs.push(2);
        rs.push(3);
        assert_eq!(rs.read(), Some(Chunk::Data(1)));
        assert_eq!(rs.read(), Some(Chunk::Data(2)));
        assert_eq!(rs.read(), Some(Chunk::Data(3)));
        assert_eq!(rs.read(), None);
    }

    #[test]
    fn close_stops_push() {
        let mut rs = ReadableStream::new();
        rs.push(1);
        rs.close();
        rs.push(2); // ignored
        assert_eq!(rs.read(), Some(Chunk::Data(1)));
        assert_eq!(rs.read(), Some(Chunk::End));
        assert_eq!(rs.read(), None);
        assert!(rs.is_closed());
    }

    #[test]
    fn backpressure() {
        let mut ws = WritableStream::new(2);
        assert!(ws.write(1));
        assert!(ws.write(2));
        assert!(!ws.write(3)); // full
        assert!(ws.is_full());
    }

    #[test]
    fn writable_drain() {
        let mut ws = WritableStream::new(10);
        ws.write(10);
        ws.write(20);
        let drained = ws.drain();
        assert_eq!(drained, vec![10, 20]);
        assert!(!ws.is_full());
    }

    #[test]
    fn writable_closed_rejects() {
        let mut ws = WritableStream::new(10);
        ws.close();
        assert!(!ws.write(1));
    }

    #[test]
    fn transform_maps() {
        let mut ts = TransformStream::new(|x: i32| vec![x * 2]);
        ts.write(5);
        ts.write(10);
        assert_eq!(ts.readable.read(), Some(Chunk::Data(10)));
        assert_eq!(ts.readable.read(), Some(Chunk::Data(20)));
    }

    #[test]
    fn transform_one_to_many() {
        let mut ts = TransformStream::new(|s: &str| s.chars().collect::<Vec<_>>());
        ts.write("ab");
        assert_eq!(ts.readable.read(), Some(Chunk::Data('a')));
        assert_eq!(ts.readable.read(), Some(Chunk::Data('b')));
    }

    #[test]
    fn pipe_drains() {
        let mut rs = ReadableStream::new();
        rs.push(1);
        rs.push(2);
        rs.push(3);
        let mut ws = WritableStream::new(10);
        let count = pipe(&mut rs, &mut ws);
        assert_eq!(count, 3);
        assert_eq!(ws.drain(), vec![1, 2, 3]);
    }

    #[test]
    fn pipe_respects_backpressure() {
        let mut rs = ReadableStream::new();
        rs.push(1);
        rs.push(2);
        rs.push(3);
        let mut ws = WritableStream::new(2);
        let count = pipe(&mut rs, &mut ws);
        assert_eq!(count, 2);
    }

    #[test]
    fn error_propagation() {
        let mut rs: ReadableStream<i32> = ReadableStream::new();
        rs.push(1);
        rs.push_error("oops");
        rs.push(2);
        assert_eq!(rs.read(), Some(Chunk::Data(1)));
        assert_eq!(rs.read(), Some(Chunk::Error("oops".into())));
        assert_eq!(rs.read(), Some(Chunk::Data(2)));
    }

    #[test]
    fn buffered_count() {
        let mut rs = ReadableStream::new();
        rs.push(1);
        rs.push(2);
        assert_eq!(rs.buffered_count(), 2);
        rs.read();
        assert_eq!(rs.buffered_count(), 1);
    }
}
