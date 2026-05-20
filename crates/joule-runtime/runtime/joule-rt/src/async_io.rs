//! Async I/O for Joule runtime
//!
//! This module provides platform-specific async I/O using:
//! - kqueue on macOS/BSD
//! - epoll on Linux
//!
//! # Design
//!
//! The I/O reactor runs in a dedicated thread and notifies waiting tasks
//! when file descriptors become ready for reading or writing.

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::os::unix::io::{AsRawFd, RawFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Waker;
use std::thread;
use std::time::Duration;

// ============================================================================
// Interest flags
// ============================================================================

/// Interest in read/write events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interest {
    /// Interested in read events.
    pub read: bool,
    /// Interested in write events.
    pub write: bool,
}

impl Interest {
    /// Interest in read events only.
    pub const READ: Self = Self {
        read: true,
        write: false,
    };

    /// Interest in write events only.
    pub const WRITE: Self = Self {
        read: false,
        write: true,
    };

    /// Interest in both read and write events.
    pub const BOTH: Self = Self {
        read: true,
        write: true,
    };
}

// ============================================================================
// Readiness state
// ============================================================================

/// Readiness state for a file descriptor.
#[derive(Debug, Clone, Copy, Default)]
pub struct Readiness {
    /// FD is ready for reading.
    pub readable: bool,
    /// FD is ready for writing.
    pub writable: bool,
    /// An error occurred.
    pub error: bool,
}

// ============================================================================
// Registration
// ============================================================================

/// A registration with the I/O reactor.
struct Registration {
    /// Current interest.
    interest: Interest,
    /// Waker to notify when ready.
    waker: Option<Waker>,
    /// Current readiness.
    readiness: Readiness,
}

// ============================================================================
// Reactor (kqueue implementation for macOS)
// ============================================================================

#[cfg(target_os = "macos")]
struct ReactorInner {
    /// The kqueue file descriptor.
    kq: RawFd,
    /// Registered file descriptors.
    registrations: Mutex<HashMap<RawFd, Registration>>,
    /// Whether the reactor is running.
    running: AtomicBool,
}

#[cfg(target_os = "macos")]
impl ReactorInner {
    fn new() -> io::Result<Self> {
        let kq = unsafe { libc::kqueue() };
        if kq < 0 {
            return Err(io::Error::last_os_error());
        }

        Ok(Self {
            kq,
            registrations: Mutex::new(HashMap::new()),
            running: AtomicBool::new(false),
        })
    }

    fn register(&self, fd: RawFd, interest: Interest) -> io::Result<()> {
        let mut regs = self.registrations.lock().unwrap();

        if regs.contains_key(&fd) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "fd already registered",
            ));
        }

        // Add to kqueue
        self.update_kqueue(fd, interest, true)?;

        regs.insert(
            fd,
            Registration {
                interest,
                waker: None,
                readiness: Readiness::default(),
            },
        );

        Ok(())
    }

    fn deregister(&self, fd: RawFd) -> io::Result<()> {
        let mut regs = self.registrations.lock().unwrap();

        if let Some(reg) = regs.remove(&fd) {
            // Remove from kqueue
            self.update_kqueue(fd, reg.interest, false)?;
        }

        Ok(())
    }

    fn update_kqueue(&self, fd: RawFd, interest: Interest, add: bool) -> io::Result<()> {
        let mut changes = Vec::new();

        let flags = if add {
            (libc::EV_ADD | libc::EV_ENABLE | libc::EV_ONESHOT) as u16
        } else {
            libc::EV_DELETE as u16
        };

        if interest.read {
            changes.push(libc::kevent {
                ident: fd as usize,
                filter: libc::EVFILT_READ,
                flags,
                fflags: 0,
                data: 0,
                udata: fd as *mut libc::c_void,
            });
        }

        if interest.write {
            changes.push(libc::kevent {
                ident: fd as usize,
                filter: libc::EVFILT_WRITE,
                flags,
                fflags: 0,
                data: 0,
                udata: fd as *mut libc::c_void,
            });
        }

        if !changes.is_empty() {
            let result = unsafe {
                libc::kevent(
                    self.kq,
                    changes.as_ptr(),
                    changes.len() as i32,
                    std::ptr::null_mut(),
                    0,
                    std::ptr::null(),
                )
            };

            if result < 0 {
                return Err(io::Error::last_os_error());
            }
        }

        Ok(())
    }

    fn set_waker(&self, fd: RawFd, waker: Waker) {
        let mut regs = self.registrations.lock().unwrap();
        if let Some(reg) = regs.get_mut(&fd) {
            reg.waker = Some(waker);
        }
    }

    fn poll_once(&self, timeout: Option<Duration>) -> io::Result<()> {
        let mut events: [libc::kevent; 64] = unsafe { std::mem::zeroed() };

        let timeout_spec = timeout.map(|d| libc::timespec {
            tv_sec: d.as_secs() as i64,
            tv_nsec: d.subsec_nanos() as i64,
        });

        let timeout_ptr = timeout_spec
            .as_ref()
            .map(|t| t as *const _)
            .unwrap_or(std::ptr::null());

        let n = unsafe {
            libc::kevent(
                self.kq,
                std::ptr::null(),
                0,
                events.as_mut_ptr(),
                events.len() as i32,
                timeout_ptr,
            )
        };

        if n < 0 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::Interrupted {
                return Ok(());
            }
            return Err(err);
        }

        // Process events
        let mut regs = self.registrations.lock().unwrap();

        for event in &events[..n as usize] {
            let fd = event.udata as RawFd;

            if let Some(reg) = regs.get_mut(&fd) {
                if event.filter == libc::EVFILT_READ {
                    reg.readiness.readable = true;
                }
                if event.filter == libc::EVFILT_WRITE {
                    reg.readiness.writable = true;
                }

                // Wake the task
                if let Some(waker) = reg.waker.take() {
                    waker.wake();
                }

                // Re-arm the event (oneshot) - need to do this after releasing lock
                let interest = reg.interest;
                drop(regs);
                let _ = self.update_kqueue(fd, interest, true);
                regs = self.registrations.lock().unwrap();
            }
        }

        Ok(())
    }

    fn get_readiness(&self, fd: RawFd) -> Readiness {
        let regs = self.registrations.lock().unwrap();
        regs.get(&fd).map(|r| r.readiness).unwrap_or_default()
    }

    fn clear_readiness(&self, fd: RawFd) {
        let mut regs = self.registrations.lock().unwrap();
        if let Some(reg) = regs.get_mut(&fd) {
            reg.readiness = Readiness::default();
        }
    }
}

#[cfg(target_os = "macos")]
impl Drop for ReactorInner {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.kq);
        }
    }
}

// ============================================================================
// Reactor (epoll implementation for Linux)
// ============================================================================

#[cfg(target_os = "linux")]
struct ReactorInner {
    /// The epoll file descriptor.
    epfd: RawFd,
    /// Registered file descriptors.
    registrations: Mutex<HashMap<RawFd, Registration>>,
    /// Whether the reactor is running.
    running: AtomicBool,
}

#[cfg(target_os = "linux")]
impl ReactorInner {
    fn new() -> io::Result<Self> {
        let epfd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
        if epfd < 0 {
            return Err(io::Error::last_os_error());
        }

        Ok(Self {
            epfd,
            registrations: Mutex::new(HashMap::new()),
            running: AtomicBool::new(false),
        })
    }

    fn register(&self, fd: RawFd, interest: Interest) -> io::Result<()> {
        let mut regs = self.registrations.lock().unwrap();

        if regs.contains_key(&fd) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "fd already registered",
            ));
        }

        let mut events = libc::EPOLLONESHOT;
        if interest.read {
            events |= libc::EPOLLIN;
        }
        if interest.write {
            events |= libc::EPOLLOUT;
        }

        let mut event = libc::epoll_event {
            events: events as u32,
            u64: fd as u64,
        };

        let result = unsafe { libc::epoll_ctl(self.epfd, libc::EPOLL_CTL_ADD, fd, &mut event) };

        if result < 0 {
            return Err(io::Error::last_os_error());
        }

        regs.insert(
            fd,
            Registration {
                interest,
                waker: None,
                readiness: Readiness::default(),
            },
        );

        Ok(())
    }

    fn deregister(&self, fd: RawFd) -> io::Result<()> {
        let mut regs = self.registrations.lock().unwrap();

        if regs.remove(&fd).is_some() {
            let result = unsafe {
                libc::epoll_ctl(self.epfd, libc::EPOLL_CTL_DEL, fd, std::ptr::null_mut())
            };

            if result < 0 {
                return Err(io::Error::last_os_error());
            }
        }

        Ok(())
    }

    fn set_waker(&self, fd: RawFd, waker: Waker) {
        let mut regs = self.registrations.lock().unwrap();
        if let Some(reg) = regs.get_mut(&fd) {
            reg.waker = Some(waker);
        }
    }

    fn poll_once(&self, timeout: Option<Duration>) -> io::Result<()> {
        let mut events: [libc::epoll_event; 64] = unsafe { std::mem::zeroed() };

        let timeout_ms = timeout.map(|d| d.as_millis() as i32).unwrap_or(-1);

        let n = unsafe {
            libc::epoll_wait(
                self.epfd,
                events.as_mut_ptr(),
                events.len() as i32,
                timeout_ms,
            )
        };

        if n < 0 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::Interrupted {
                return Ok(());
            }
            return Err(err);
        }

        let mut regs = self.registrations.lock().unwrap();

        for event in &events[..n as usize] {
            let fd = event.u64 as RawFd;

            if let Some(reg) = regs.get_mut(&fd) {
                if event.events & libc::EPOLLIN as u32 != 0 {
                    reg.readiness.readable = true;
                }
                if event.events & libc::EPOLLOUT as u32 != 0 {
                    reg.readiness.writable = true;
                }

                if let Some(waker) = reg.waker.take() {
                    waker.wake();
                }

                // Re-arm (EPOLLONESHOT)
                let mut events_flags = libc::EPOLLONESHOT;
                if reg.interest.read {
                    events_flags |= libc::EPOLLIN;
                }
                if reg.interest.write {
                    events_flags |= libc::EPOLLOUT;
                }

                let mut ev = libc::epoll_event {
                    events: events_flags as u32,
                    u64: fd as u64,
                };

                unsafe {
                    libc::epoll_ctl(self.epfd, libc::EPOLL_CTL_MOD, fd, &mut ev);
                }
            }
        }

        Ok(())
    }

    fn get_readiness(&self, fd: RawFd) -> Readiness {
        let regs = self.registrations.lock().unwrap();
        regs.get(&fd).map(|r| r.readiness).unwrap_or_default()
    }

    fn clear_readiness(&self, fd: RawFd) {
        let mut regs = self.registrations.lock().unwrap();
        if let Some(reg) = regs.get_mut(&fd) {
            reg.readiness = Readiness::default();
        }
    }
}

#[cfg(target_os = "linux")]
impl Drop for ReactorInner {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.epfd);
        }
    }
}

// ============================================================================
// Global Reactor
// ============================================================================

static GLOBAL_REACTOR: std::sync::OnceLock<Arc<ReactorInner>> = std::sync::OnceLock::new();

/// The global I/O reactor.
pub struct Reactor;

impl Reactor {
    /// Get the global reactor, initializing it if necessary.
    fn global() -> &'static Arc<ReactorInner> {
        GLOBAL_REACTOR.get_or_init(|| {
            let inner = Arc::new(ReactorInner::new().expect("Failed to create I/O reactor"));
            let inner_clone = inner.clone();

            // Start the reactor thread
            inner.running.store(true, Ordering::Release);

            thread::Builder::new()
                .name("joule-io-reactor".to_string())
                .spawn(move || {
                    while inner_clone.running.load(Ordering::Acquire) {
                        if let Err(e) = inner_clone.poll_once(Some(Duration::from_millis(100))) {
                            eprintln!("I/O reactor error: {}", e);
                        }
                    }
                })
                .expect("Failed to spawn I/O reactor thread");

            inner
        })
    }

    /// Register a file descriptor with the reactor.
    pub fn register(fd: RawFd, interest: Interest) -> io::Result<()> {
        Self::global().register(fd, interest)
    }

    /// Deregister a file descriptor.
    pub fn deregister(fd: RawFd) -> io::Result<()> {
        Self::global().deregister(fd)
    }

    /// Set the waker for a file descriptor.
    pub fn set_waker(fd: RawFd, waker: Waker) {
        Self::global().set_waker(fd, waker)
    }

    /// Get the readiness state of a file descriptor.
    pub fn get_readiness(fd: RawFd) -> Readiness {
        Self::global().get_readiness(fd)
    }

    /// Clear the readiness state.
    pub fn clear_readiness(fd: RawFd) {
        Self::global().clear_readiness(fd)
    }
}

// ============================================================================
// Async TCP
// ============================================================================

/// Async TCP stream.
pub struct AsyncTcpStream {
    inner: std::net::TcpStream,
}

impl AsyncTcpStream {
    /// Create from a standard TcpStream.
    ///
    /// The stream must be set to non-blocking mode.
    pub fn from_std(stream: std::net::TcpStream) -> io::Result<Self> {
        stream.set_nonblocking(true)?;
        Ok(Self { inner: stream })
    }

    /// Connect to an address.
    pub fn connect(addr: &str) -> io::Result<Self> {
        let stream = std::net::TcpStream::connect(addr)?;
        Self::from_std(stream)
    }

    /// Blocking read with async registration.
    pub fn read_sync(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }

    /// Blocking write with async registration.
    pub fn write_sync(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }

    /// Get the underlying TcpStream.
    pub fn into_inner(self) -> std::net::TcpStream {
        self.inner
    }

    /// Get the raw file descriptor.
    pub fn as_raw_fd(&self) -> RawFd {
        self.inner.as_raw_fd()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interest_flags() {
        assert!(Interest::READ.read);
        assert!(!Interest::READ.write);

        assert!(!Interest::WRITE.read);
        assert!(Interest::WRITE.write);

        assert!(Interest::BOTH.read);
        assert!(Interest::BOTH.write);
    }

    #[test]
    fn test_readiness_default() {
        let r = Readiness::default();
        assert!(!r.readable);
        assert!(!r.writable);
        assert!(!r.error);
    }

    #[test]
    fn test_reactor_creation() {
        // Just test that we can get the global reactor
        let _reactor = Reactor::global();
    }
}
