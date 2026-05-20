/*
 * Joule Embedded Runtime — Reference Implementation
 *
 * This header provides default implementations for the jrt_* functions
 * required by Joule-compiled freestanding C code. Users can include this
 * header directly or use it as a template for custom implementations.
 *
 * Required functions:
 *   void* jrt_alloc(size_t size)
 *   void  jrt_dealloc(void* ptr, size_t size)
 *   void* jrt_realloc(void* ptr, size_t old_size, size_t new_size)
 *   void  jrt_write_stdout(const uint8_t* data, size_t len)
 *   void  jrt_write_stderr(const uint8_t* data, size_t len)
 *   _Noreturn void jrt_panic_handler(const char* msg, size_t len)
 *   _Noreturn void jrt_exit(int code)
 *
 * Usage:
 *   1. Define JRT_HEAP_SIZE before including to set heap size (default 8K)
 *   2. Define JRT_UART_PUTCHAR(c) to wire up UART output
 *   3. Include this header in ONE .c file (it contains implementations)
 */

#ifndef JOULE_EMBEDDED_RT_H
#define JOULE_EMBEDDED_RT_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* Configurable heap size */
#ifndef JRT_HEAP_SIZE
#define JRT_HEAP_SIZE 8192
#endif

/* UART output — user must define this macro */
#ifndef JRT_UART_PUTCHAR
#define JRT_UART_PUTCHAR(c) ((void)(c))  /* No-op by default */
#endif

/* ======================================================================
 * Simple bump allocator
 * ====================================================================== */

static uint8_t jrt_heap[JRT_HEAP_SIZE];
static size_t  jrt_heap_offset = 0;

void* jrt_alloc(size_t size) {
    /* Align to 8 bytes */
    size = (size + 7) & ~(size_t)7;
    if (jrt_heap_offset + size > JRT_HEAP_SIZE) {
        return (void*)0;  /* Out of memory */
    }
    void* ptr = &jrt_heap[jrt_heap_offset];
    jrt_heap_offset += size;
    return ptr;
}

void jrt_dealloc(void* ptr, size_t size) {
    /* Bump allocator cannot free individual blocks */
    (void)ptr;
    (void)size;
}

void* jrt_realloc(void* ptr, size_t old_size, size_t new_size) {
    void* new_ptr = jrt_alloc(new_size);
    if (new_ptr && ptr && old_size > 0) {
        size_t copy_size = old_size < new_size ? old_size : new_size;
        memcpy(new_ptr, ptr, copy_size);
    }
    return new_ptr;
}

/* ======================================================================
 * I/O via UART
 * ====================================================================== */

void jrt_write_stdout(const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        JRT_UART_PUTCHAR(data[i]);
    }
}

void jrt_write_stderr(const uint8_t* data, size_t len) {
    /* On most MCUs, stderr goes to the same UART as stdout */
    jrt_write_stdout(data, len);
}

/* ======================================================================
 * Panic and exit
 * ====================================================================== */

_Noreturn void jrt_panic_handler(const char* msg, size_t len) {
    jrt_write_stderr((const uint8_t*)msg, len);
    jrt_write_stderr((const uint8_t*)"\n", 1);
    /* Halt: infinite loop with interrupts disabled */
    while (1) {
        __asm__ volatile("" ::: "memory");
    }
}

_Noreturn void jrt_exit(int code) {
    (void)code;
    while (1) {
        __asm__ volatile("" ::: "memory");
    }
}

#endif /* JOULE_EMBEDDED_RT_H */
