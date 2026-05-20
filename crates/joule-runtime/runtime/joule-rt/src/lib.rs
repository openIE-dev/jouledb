// Runtime FFI code has legitimate patterns that trigger these lints:
// - ptr_as_ptr, ptr_cast_constness, not_unsafe_ptr_arg_deref: FFI pointer operations
// - missing_transmute_annotations: runtime type conversions
// - significant_drop_in_scrutinee: lock guards in match
// - type_repetition_in_bounds: trait bound clarity
// - unwrap_or_default: HashMap entry API patterns
#![allow(
    clippy::ptr_as_ptr,
    clippy::ptr_cast_constness,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::missing_transmute_annotations,
    clippy::significant_drop_in_scrutinee,
    clippy::type_repetition_in_bounds,
    clippy::unwrap_or_default,
    clippy::assertions_on_constants,
    clippy::bool_assert_comparison,
    clippy::float_cmp,
    clippy::borrow_as_ptr
)]
//! Joule Runtime Library
//!
//! This library provides the core runtime support for compiled Joule programs.
//! It includes implementations for:
//! - I/O functions (println, print, etc.)
//! - Panic handling
//! - Memory allocation helpers
//! - Standard library runtime support
//! - Async executor (single-threaded and work-stealing)
//! - Bounded channels for concurrency
//! - Task groups for structured concurrency
//! - Async I/O (kqueue/epoll)
//! - Async streams (async iterators)

#[cfg(unix)]
pub mod async_io;
pub mod channel;
pub mod dataflow_executor;
pub mod executor;
pub mod ffi;
pub mod stream;
pub mod task_group;
pub mod work_stealing;

use std::io::{self, Write};
use std::process;

/// Print a string with a newline
///
/// This function is called from compiled Joule code to implement println!().
/// It takes a pointer to UTF-8 bytes and a length.
///
/// # Safety
/// The caller must ensure that `ptr` points to valid UTF-8 data of at least `len` bytes.
#[unsafe(no_mangle)]
pub extern "C" fn joule_println(ptr: *const u8, len: usize) {
    if ptr.is_null() {
        eprintln!("Error: null pointer passed to joule_println");
        return;
    }

    // SAFETY: We checked that ptr is not null, and caller guarantees valid data
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };

    match std::str::from_utf8(bytes) {
        Ok(s) => {
            println!("{s}");
        }
        Err(e) => {
            eprintln!("Error: invalid UTF-8 in joule_println: {e}");
        }
    }
}

/// Print a string without a newline
///
/// This function is called from compiled Joule code to implement print!().
///
/// # Safety
/// The caller must ensure that `ptr` points to valid UTF-8 data of at least `len` bytes.
#[unsafe(no_mangle)]
pub extern "C" fn joule_print(ptr: *const u8, len: usize) {
    if ptr.is_null() {
        eprintln!("Error: null pointer passed to joule_print");
        return;
    }

    // SAFETY: We checked that ptr is not null, and caller guarantees valid data
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };

    match std::str::from_utf8(bytes) {
        Ok(s) => {
            print!("{s}");
            let _ = io::stdout().flush();
        }
        Err(e) => {
            eprintln!("Error: invalid UTF-8 in joule_print: {e}");
        }
    }
}

/// Panic handler for Joule programs
///
/// This function is called when a Joule program panics.
/// It prints the panic message and aborts the program.
///
/// # Safety
/// The caller must ensure that the pointers point to valid UTF-8 data.
#[unsafe(no_mangle)]
pub extern "C" fn joule_panic(
    ptr: *const u8,
    len: usize,
    file: *const u8,
    file_len: usize,
    line: u32,
) -> ! {
    eprintln!("Joule program panicked!");

    if !ptr.is_null() && len > 0 {
        // SAFETY: We checked that ptr is not null
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        if let Ok(msg) = std::str::from_utf8(bytes) {
            eprintln!("  Message: {msg}");
        }
    }

    if !file.is_null() && file_len > 0 {
        // SAFETY: We checked that file is not null
        let file_bytes = unsafe { std::slice::from_raw_parts(file, file_len) };
        if let Ok(file_str) = std::str::from_utf8(file_bytes) {
            eprintln!("  Location: {file_str}:{line}");
        }
    }

    process::abort();
}

/// Simple panic handler without location info
///
/// # Safety
/// The caller must ensure that `ptr` points to valid UTF-8 data of at least `len` bytes.
#[unsafe(no_mangle)]
pub extern "C" fn joule_panic_simple(ptr: *const u8, len: usize) -> ! {
    eprintln!("Joule program panicked!");

    if !ptr.is_null() && len > 0 {
        // SAFETY: We checked that ptr is not null
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        if let Ok(msg) = std::str::from_utf8(bytes) {
            eprintln!("  Message: {msg}");
        }
    }

    process::abort();
}

/// Allocate memory (wrapper around global allocator)
///
/// # Safety
/// This is an FFI function that allocates memory. The caller is responsible for
/// deallocating the returned memory with `joule_dealloc`.
#[unsafe(no_mangle)]
pub extern "C" fn joule_alloc(size: usize) -> *mut u8 {
    if size == 0 {
        return std::ptr::null_mut();
    }

    let Ok(layout) = std::alloc::Layout::from_size_align(size, 8) else {
        return std::ptr::null_mut();
    };

    // SAFETY: Layout is valid (non-zero size, valid alignment)
    unsafe { std::alloc::alloc(layout) }
}

/// Deallocate memory (wrapper around global allocator)
///
/// # Safety
/// The caller must ensure that `ptr` was allocated by `joule_alloc` with the same `size`.
#[unsafe(no_mangle)]
pub extern "C" fn joule_dealloc(ptr: *mut u8, size: usize) {
    if ptr.is_null() || size == 0 {
        return;
    }

    let Ok(layout) = std::alloc::Layout::from_size_align(size, 8) else {
        return;
    };

    // SAFETY: Caller guarantees ptr was allocated with the same layout
    unsafe { std::alloc::dealloc(ptr, layout) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_println_with_valid_string() {
        let s = "Hello, World!";
        joule_println(s.as_ptr(), s.len());
    }

    #[test]
    fn test_println_with_empty_string() {
        let s = "";
        joule_println(s.as_ptr(), s.len());
    }

    #[test]
    fn test_print_with_valid_string() {
        let s = "Test";
        joule_print(s.as_ptr(), s.len());
    }
}
