//! HDC similarity on the Apple GPU (Metal).
//!
//! GPU fallback for the fabric's HDC lane when the Core ML / Neural Engine
//! lane is not used. Bipolar (+1/-1) vectors are stored as `char`, products
//! are accumulated in `int`, so the result is exact.

use crate::error::{MetalError, MetalResult};

/// One thread per (query, codebook) pair: `out[i*k + j] = sum_t q[i,t] * m[j,t]`.
pub const FABRIC_HDC_DOT_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fabric_hdc_dot(
    device const char* q [[buffer(0)]],
    device const char* m [[buffer(1)]],
    device int* out [[buffer(2)]],
    constant uint* dims [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    uint n = dims[0];
    uint k = dims[1];
    uint d = dims[2];
    if (idx >= n * k) {
        return;
    }
    uint i = idx / k;
    uint j = idx - i * k;
    device const char* qi = q + (ulong)i * d;
    device const char* mj = m + (ulong)j * d;
    int acc = 0;
    for (uint t = 0; t < d; ++t) {
        acc += int(qi[t]) * int(mj[t]);
    }
    out[idx] = acc;
}
"#;

/// CPU reference for [`hdc_dot_i8`].
pub fn reference_hdc_dot(queries: &[i8], memory: &[i8], d: usize) -> Vec<i32> {
    let d = d.max(1);
    let n = queries.len() / d;
    let k = memory.len() / d;
    let mut out = Vec::with_capacity(n * k);
    for i in 0..n {
        for j in 0..k {
            out.push(
                queries[i * d..(i + 1) * d]
                    .iter()
                    .zip(&memory[j * d..(j + 1) * d])
                    .map(|(a, b)| i32::from(*a) * i32::from(*b))
                    .sum(),
            );
        }
    }
    out
}

/// Wall-clock split of one [`hdc_dot_i8_timed`] call.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HdcDotTiming {
    /// One-time setup in this call: device lookup, shader compile, pipeline
    /// and queue creation. `0.0` when the per-process kernel was reused.
    pub setup_s: f64,
    /// Per-job work: buffer upload, dispatch, wait, readback.
    pub kernel_s: f64,
}

/// Dot products of `n x d` queries against `k x d` memory, on the GPU.
///
/// Off macOS returns [`MetalError::NotAvailable`].
pub fn hdc_dot_i8(queries: &[i8], memory: &[i8], d: usize) -> MetalResult<Vec<i32>> {
    hdc_dot_i8_timed(queries, memory, d).map(|(scores, _)| scores)
}

/// [`hdc_dot_i8`] plus the setup / per-job time split. The compiled
/// pipeline is cached per process, so only the first call pays `setup_s`.
pub fn hdc_dot_i8_timed(
    queries: &[i8],
    memory: &[i8],
    d: usize,
) -> MetalResult<(Vec<i32>, HdcDotTiming)> {
    if d == 0 || queries.is_empty() || memory.is_empty() || queries.len() % d != 0 || memory.len() % d != 0 {
        return Err(MetalError::InvalidBufferAccess {
            message: format!(
                "hdc_dot_i8: queries={} memory={} not multiples of d={d}",
                queries.len(),
                memory.len()
            ),
        });
    }
    #[cfg(target_os = "macos")]
    {
        hdc_dot_metal(queries, memory, d)
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err(MetalError::not_macos())
    }
}

#[cfg(target_os = "macos")]
struct HdcKernel {
    device: crate::MetalDevice,
    pipeline: crate::MetalComputePipeline,
    queue: crate::MetalCommandQueue,
}

// SAFETY: Metal devices, pipeline states and command queues are documented
// as thread-safe; every use below is additionally serialised by the mutex.
#[cfg(target_os = "macos")]
unsafe impl Send for HdcKernel {}

#[cfg(target_os = "macos")]
fn hdc_dot_metal(queries: &[i8], memory: &[i8], d: usize) -> MetalResult<(Vec<i32>, HdcDotTiming)> {
    use crate::{BufferUsage, MetalDevice};
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;

    static KERNEL: OnceLock<Mutex<Option<HdcKernel>>> = OnceLock::new();
    let slot = KERNEL.get_or_init(|| Mutex::new(None));
    let mut guard = slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

    let setup_started = Instant::now();
    let mut setup_s = 0.0;
    if guard.is_none() {
        let device = MetalDevice::system_default()?;
        let library = device.create_library_from_source(FABRIC_HDC_DOT_MSL)?;
        let pipeline = device.create_compute_pipeline(&library, "fabric_hdc_dot")?;
        let queue = device.create_command_queue()?;
        *guard = Some(HdcKernel { device, pipeline, queue });
        setup_s = setup_started.elapsed().as_secs_f64();
    }
    let Some(kernel) = guard.as_ref() else {
        return Err(MetalError::not_macos());
    };

    let started = Instant::now();
    let n = queries.len() / d;
    let k = memory.len() / d;
    let total = n * k;
    let q = kernel.device.create_buffer_with_data(queries, BufferUsage::Shared)?;
    let m = kernel.device.create_buffer_with_data(memory, BufferUsage::Shared)?;
    let out = kernel.device.create_buffer_with_data(&vec![0i32; total], BufferUsage::Shared)?;
    let dims = kernel
        .device
        .create_buffer_with_data(&[n as u32, k as u32, d as u32], BufferUsage::Shared)?;
    let width = kernel.pipeline.max_total_threads_per_threadgroup().clamp(1, 256);
    let groups = total.div_ceil(width as usize) as u32;
    kernel
        .queue
        .launch_kernel(&kernel.pipeline, (groups, 1, 1), (width, 1, 1), &[&q, &m, &out, &dims])?;
    let scores = out.read::<i32>(total)?;
    Ok((
        scores,
        HdcDotTiming {
            setup_s,
            kernel_s: started.elapsed().as_secs_f64(),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_dot_is_correct() {
        assert_eq!(reference_hdc_dot(&[1, -1, 1, 1], &[1, 1, -1, -1], 2), vec![0, 0, 2, -2]);
    }

    #[test]
    fn rejects_bad_shapes_everywhere() {
        assert!(hdc_dot_i8(&[1, 1, 1], &[1, 1], 2).is_err());
        assert!(hdc_dot_i8(&[], &[1, 1], 2).is_err());
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn not_available_off_macos() {
        let err = hdc_dot_i8(&[1, -1], &[1, 1], 2).unwrap_err();
        assert!(matches!(err, MetalError::NotAvailable { .. }));
    }
}
