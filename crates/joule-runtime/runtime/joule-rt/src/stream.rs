//! Async streams (async iterators) for Joule runtime
//!
//! This module provides a `Stream` trait similar to Rust's futures::Stream,
//! along with common combinators and adapters.
//!
//! # Example
//!
//! ```ignore
//! use joule_rt::stream::{Stream, StreamExt, iter};
//!
//! let mut stream = iter([1, 2, 3, 4, 5])
//!     .map(|x| x * 2)
//!     .filter(|x| *x > 4);
//!
//! while let Some(value) = stream.next().await {
//!     println!("{}", value);
//! }
//! ```

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::channel::Receiver;

// ============================================================================
// Stream Trait
// ============================================================================

/// A stream of values produced asynchronously.
///
/// This is the async equivalent of `Iterator`.
pub trait Stream {
    /// The type of items yielded by the stream.
    type Item;

    /// Attempt to pull out the next value of this stream.
    ///
    /// Returns `Poll::Ready(Some(item))` if an item is ready,
    /// `Poll::Ready(None)` if the stream is exhausted,
    /// or `Poll::Pending` if the next item is not yet available.
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>>;

    /// Returns the bounds on the remaining length of the stream.
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

// ============================================================================
// StreamExt - Extension methods for Stream
// ============================================================================

/// Extension trait providing combinator methods for streams.
pub trait StreamExt: Stream {
    /// Get the next item from the stream.
    fn next(&mut self) -> Next<'_, Self>
    where
        Self: Unpin,
    {
        Next { stream: self }
    }

    /// Map each item to a new value.
    fn map<T, F>(self, f: F) -> Map<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Item) -> T,
    {
        Map { stream: self, f }
    }

    /// Filter items based on a predicate.
    fn filter<F>(self, predicate: F) -> Filter<Self, F>
    where
        Self: Sized,
        F: FnMut(&Self::Item) -> bool,
    {
        Filter {
            stream: self,
            predicate,
        }
    }

    /// Filter and map in one step.
    fn filter_map<T, F>(self, f: F) -> FilterMap<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Item) -> Option<T>,
    {
        FilterMap { stream: self, f }
    }

    /// Take only the first n items.
    fn take(self, n: usize) -> Take<Self>
    where
        Self: Sized,
    {
        Take {
            stream: self,
            remaining: n,
        }
    }

    /// Skip the first n items.
    fn skip(self, n: usize) -> Skip<Self>
    where
        Self: Sized,
    {
        Skip {
            stream: self,
            remaining: n,
        }
    }

    /// Chain two streams together.
    fn chain<S>(self, other: S) -> Chain<Self, S>
    where
        Self: Sized,
        S: Stream<Item = Self::Item>,
    {
        Chain {
            first: Some(self),
            second: other,
        }
    }

    /// Enumerate items with their index.
    fn enumerate(self) -> Enumerate<Self>
    where
        Self: Sized,
    {
        Enumerate {
            stream: self,
            count: 0,
        }
    }

    /// Inspect each item without modifying it.
    fn inspect<F>(self, f: F) -> Inspect<Self, F>
    where
        Self: Sized,
        F: FnMut(&Self::Item),
    {
        Inspect { stream: self, f }
    }

    /// Fold all items into a single value.
    fn fold<T, F>(self, init: T, f: F) -> Fold<Self, T, F>
    where
        Self: Sized,
        F: FnMut(T, Self::Item) -> T,
    {
        Fold {
            stream: self,
            acc: Some(init),
            f,
        }
    }

    /// Collect all items into a Vec.
    fn collect_vec(self) -> CollectVec<Self>
    where
        Self: Sized,
    {
        CollectVec {
            stream: self,
            items: Vec::new(),
        }
    }

    /// Check if any item satisfies a predicate.
    fn any<F>(self, predicate: F) -> Any<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Item) -> bool,
    {
        Any {
            stream: self,
            predicate,
        }
    }

    /// Check if all items satisfy a predicate.
    fn all<F>(self, predicate: F) -> All<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Item) -> bool,
    {
        All {
            stream: self,
            predicate,
        }
    }

    /// Count the number of items.
    fn count(self) -> Count<Self>
    where
        Self: Sized,
    {
        Count {
            stream: self,
            count: 0,
        }
    }

    /// Execute a closure for each item.
    fn for_each<F>(self, f: F) -> ForEach<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Item),
    {
        ForEach { stream: self, f }
    }
}

impl<S: Stream + ?Sized> StreamExt for S {}

// ============================================================================
// Next future
// ============================================================================

/// Future for getting the next item from a stream.
pub struct Next<'a, S: ?Sized> {
    stream: &'a mut S,
}

impl<S: Stream + Unpin + ?Sized> Future for Next<'_, S> {
    type Output = Option<S::Item>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut *self.stream).poll_next(cx)
    }
}

// ============================================================================
// Stream Adapters
// ============================================================================

/// Stream adapter that maps each item.
pub struct Map<S, F> {
    stream: S,
    f: F,
}

impl<S, F> Unpin for Map<S, F> where S: Unpin {}

impl<S: Stream + Unpin, T, F: FnMut(S::Item) -> T> Stream for Map<S, F> {
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.stream).poll_next(cx) {
            Poll::Ready(Some(item)) => Poll::Ready(Some((self.f)(item))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.stream.size_hint()
    }
}

/// Stream adapter that filters items.
pub struct Filter<S, F> {
    stream: S,
    predicate: F,
}

impl<S, F> Unpin for Filter<S, F> where S: Unpin {}

impl<S: Stream + Unpin, F: FnMut(&S::Item) -> bool> Stream for Filter<S, F> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match Pin::new(&mut self.stream).poll_next(cx) {
                Poll::Ready(Some(item)) => {
                    if (self.predicate)(&item) {
                        return Poll::Ready(Some(item));
                    }
                }
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Stream adapter that filters and maps.
pub struct FilterMap<S, F> {
    stream: S,
    f: F,
}

impl<S, F> Unpin for FilterMap<S, F> where S: Unpin {}

impl<S: Stream + Unpin, T, F: FnMut(S::Item) -> Option<T>> Stream for FilterMap<S, F> {
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match Pin::new(&mut self.stream).poll_next(cx) {
                Poll::Ready(Some(item)) => {
                    if let Some(mapped) = (self.f)(item) {
                        return Poll::Ready(Some(mapped));
                    }
                }
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Stream adapter that takes the first n items.
pub struct Take<S> {
    stream: S,
    remaining: usize,
}

impl<S> Unpin for Take<S> where S: Unpin {}

impl<S: Stream + Unpin> Stream for Take<S> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.remaining == 0 {
            return Poll::Ready(None);
        }
        match Pin::new(&mut self.stream).poll_next(cx) {
            Poll::Ready(Some(item)) => {
                self.remaining -= 1;
                Poll::Ready(Some(item))
            }
            other => other,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.stream.size_hint();
        (
            lower.min(self.remaining),
            upper.map(|u| u.min(self.remaining)),
        )
    }
}

/// Stream adapter that skips the first n items.
pub struct Skip<S> {
    stream: S,
    remaining: usize,
}

impl<S> Unpin for Skip<S> where S: Unpin {}

impl<S: Stream + Unpin> Stream for Skip<S> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        while self.remaining > 0 {
            match Pin::new(&mut self.stream).poll_next(cx) {
                Poll::Ready(Some(_)) => {
                    self.remaining -= 1;
                }
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
        Pin::new(&mut self.stream).poll_next(cx)
    }
}

/// Stream adapter that chains two streams.
pub struct Chain<S1, S2> {
    first: Option<S1>,
    second: S2,
}

impl<S1, S2> Unpin for Chain<S1, S2>
where
    S1: Unpin,
    S2: Unpin,
{
}

impl<S1: Stream + Unpin, S2: Stream<Item = S1::Item> + Unpin> Stream for Chain<S1, S2> {
    type Item = S1::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(ref mut first) = self.first {
            match Pin::new(first).poll_next(cx) {
                Poll::Ready(Some(item)) => return Poll::Ready(Some(item)),
                Poll::Ready(None) => {
                    self.first = None;
                }
                Poll::Pending => return Poll::Pending,
            }
        }
        Pin::new(&mut self.second).poll_next(cx)
    }
}

/// Stream adapter that enumerates items.
pub struct Enumerate<S> {
    stream: S,
    count: usize,
}

impl<S> Unpin for Enumerate<S> where S: Unpin {}

impl<S: Stream + Unpin> Stream for Enumerate<S> {
    type Item = (usize, S::Item);

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.stream).poll_next(cx) {
            Poll::Ready(Some(item)) => {
                let index = self.count;
                self.count += 1;
                Poll::Ready(Some((index, item)))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Stream adapter that inspects each item.
pub struct Inspect<S, F> {
    stream: S,
    f: F,
}

impl<S, F> Unpin for Inspect<S, F> where S: Unpin {}

impl<S: Stream + Unpin, F: FnMut(&S::Item)> Stream for Inspect<S, F> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.stream).poll_next(cx) {
            Poll::Ready(Some(item)) => {
                (self.f)(&item);
                Poll::Ready(Some(item))
            }
            other => other,
        }
    }
}

// ============================================================================
// Terminal Futures
// ============================================================================

/// Future for folding a stream.
pub struct Fold<S, T, F> {
    stream: S,
    acc: Option<T>,
    f: F,
}

impl<S, T, F> Unpin for Fold<S, T, F> where S: Unpin {}

impl<S: Stream + Unpin, T, F: FnMut(T, S::Item) -> T> Future for Fold<S, T, F> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        loop {
            match Pin::new(&mut self.stream).poll_next(cx) {
                Poll::Ready(Some(item)) => {
                    let acc = self.acc.take().unwrap();
                    self.acc = Some((self.f)(acc, item));
                }
                Poll::Ready(None) => {
                    return Poll::Ready(self.acc.take().unwrap());
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Future for collecting a stream into a Vec.
pub struct CollectVec<S: Stream> {
    stream: S,
    items: Vec<S::Item>,
}

impl<S: Stream> Unpin for CollectVec<S> where S: Unpin {}

impl<S: Stream + Unpin> Future for CollectVec<S> {
    type Output = Vec<S::Item>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        loop {
            match Pin::new(&mut self.stream).poll_next(cx) {
                Poll::Ready(Some(item)) => {
                    self.items.push(item);
                }
                Poll::Ready(None) => {
                    return Poll::Ready(std::mem::take(&mut self.items));
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Future for checking if any item satisfies a predicate.
pub struct Any<S, F> {
    stream: S,
    predicate: F,
}

impl<S, F> Unpin for Any<S, F> where S: Unpin {}

impl<S: Stream + Unpin, F: FnMut(S::Item) -> bool> Future for Any<S, F> {
    type Output = bool;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        loop {
            match Pin::new(&mut self.stream).poll_next(cx) {
                Poll::Ready(Some(item)) => {
                    if (self.predicate)(item) {
                        return Poll::Ready(true);
                    }
                }
                Poll::Ready(None) => return Poll::Ready(false),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Future for checking if all items satisfy a predicate.
pub struct All<S, F> {
    stream: S,
    predicate: F,
}

impl<S, F> Unpin for All<S, F> where S: Unpin {}

impl<S: Stream + Unpin, F: FnMut(S::Item) -> bool> Future for All<S, F> {
    type Output = bool;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        loop {
            match Pin::new(&mut self.stream).poll_next(cx) {
                Poll::Ready(Some(item)) => {
                    if !(self.predicate)(item) {
                        return Poll::Ready(false);
                    }
                }
                Poll::Ready(None) => return Poll::Ready(true),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Future for counting items.
pub struct Count<S> {
    stream: S,
    count: usize,
}

impl<S> Unpin for Count<S> where S: Unpin {}

impl<S: Stream + Unpin> Future for Count<S> {
    type Output = usize;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        loop {
            match Pin::new(&mut self.stream).poll_next(cx) {
                Poll::Ready(Some(_)) => {
                    self.count += 1;
                }
                Poll::Ready(None) => return Poll::Ready(self.count),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Future for executing a closure on each item.
pub struct ForEach<S, F> {
    stream: S,
    f: F,
}

impl<S, F> Unpin for ForEach<S, F> where S: Unpin {}

impl<S: Stream + Unpin, F: FnMut(S::Item)> Future for ForEach<S, F> {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        loop {
            match Pin::new(&mut self.stream).poll_next(cx) {
                Poll::Ready(Some(item)) => {
                    (self.f)(item);
                }
                Poll::Ready(None) => return Poll::Ready(()),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

// ============================================================================
// Stream Constructors
// ============================================================================

/// Create a stream from an iterator.
pub fn iter<I: IntoIterator>(iter: I) -> Iter<I::IntoIter> {
    Iter {
        iter: iter.into_iter(),
    }
}

/// Stream from an iterator.
pub struct Iter<I> {
    iter: I,
}

impl<I: Iterator> Stream for Iter<I> {
    type Item = I::Item;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(self.iter.next())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<I: Iterator> Unpin for Iter<I> {}

/// Create a stream that yields a single item.
pub fn once<T>(item: T) -> Once<T> {
    Once { item: Some(item) }
}

/// Stream that yields a single item.
pub struct Once<T> {
    item: Option<T>,
}

impl<T> Stream for Once<T> {
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(self.item.take())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = if self.item.is_some() { 1 } else { 0 };
        (len, Some(len))
    }
}

impl<T> Unpin for Once<T> {}

/// Create an empty stream.
pub fn empty<T>() -> Empty<T> {
    Empty {
        _marker: std::marker::PhantomData,
    }
}

/// Empty stream.
pub struct Empty<T> {
    _marker: std::marker::PhantomData<T>,
}

impl<T> Stream for Empty<T> {
    type Item = T;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(None)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(0))
    }
}

impl<T> Unpin for Empty<T> {}

/// Create a stream that repeats a value.
pub fn repeat<T: Clone>(item: T) -> Repeat<T> {
    Repeat { item }
}

/// Stream that repeats a value.
pub struct Repeat<T> {
    item: T,
}

impl<T: Clone> Stream for Repeat<T> {
    type Item = T;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(Some(self.item.clone()))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None)
    }
}

impl<T> Unpin for Repeat<T> {}

// ============================================================================
// Channel to Stream adapter
// ============================================================================

/// A stream that receives items from a channel.
pub struct ReceiverStream<T> {
    receiver: Receiver<T>,
}

impl<T> ReceiverStream<T> {
    /// Create a new receiver stream.
    pub fn new(receiver: Receiver<T>) -> Self {
        Self { receiver }
    }

    /// Get the underlying receiver.
    pub fn into_inner(self) -> Receiver<T> {
        self.receiver
    }
}

impl<T> Stream for ReceiverStream<T> {
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Use try_recv for non-blocking receive
        let this = self.as_mut().get_mut();
        match this.receiver.try_recv() {
            Ok(item) => Poll::Ready(Some(item)),
            Err(crate::channel::TryRecvError::Empty) => {
                // Register waker so the sender side can wake us when data arrives.
                // The waker is stored in the channel's recv_wakers and called by
                // wake_one_receiver() when a new value is sent.
                this.receiver.register_waker(cx.waker().clone());
                Poll::Pending
            }
            Err(crate::channel::TryRecvError::Closed) => Poll::Ready(None),
        }
    }
}

impl<T> Unpin for ReceiverStream<T> {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iter_stream() {
        let stream = iter([1, 2, 3]);
        assert_eq!(stream.size_hint(), (3, Some(3)));
    }

    #[test]
    fn test_once_stream() {
        let stream = once(42);
        assert_eq!(stream.size_hint(), (1, Some(1)));
    }

    #[test]
    fn test_empty_stream() {
        let stream: Empty<i32> = empty();
        assert_eq!(stream.size_hint(), (0, Some(0)));
    }

    #[test]
    fn test_repeat_stream() {
        let stream = repeat(42);
        assert_eq!(stream.size_hint(), (usize::MAX, None));
    }

    #[test]
    fn test_stream_combinators() {
        // Basic sanity check for stream types
        let _map = iter([1, 2, 3]).map(|x| x * 2);
        let _filter = iter([1, 2, 3]).filter(|x| *x > 1);
        let _take = iter([1, 2, 3]).take(2);
        let _skip = iter([1, 2, 3]).skip(1);
    }

    #[test]
    fn test_receiver_stream() {
        let (tx, rx) = crate::channel::channel::<i32>(10);

        tx.try_send(1).unwrap();
        tx.try_send(2).unwrap();

        let _stream = ReceiverStream::new(rx);
    }
}
