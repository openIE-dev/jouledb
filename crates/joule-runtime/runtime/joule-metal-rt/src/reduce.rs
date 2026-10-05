//! Fabric reduction on the Apple GPU.
//!
//! The Apple Neural Engine is not this path. The only public ANE entry point
//! is Core ML: the `joule-ane-rt` crate generates and runs an ML Program for
//! HDC similarity and reads `MLComputePlan` to see where Core ML put it. This
//! scalar reduction does not go through Core ML and there is no public ANE
//! command queue. BNNS is Accelerate (CPU / AMX), not the Neural Engine.
//! A Metal compute shader runs on the GPU.

use crate::error::{MetalError, MetalResult};

/// Why `sum_f64` never submits work to the Apple Neural Engine.
pub const ANE_DOES_NOT_RUN_THIS_KERNEL: &str = "Apple Neural Engine work goes through Core ML only (see joule-ane-rt for the HDC similarity ML Program). This scalar reduction is not a Core ML model and there is no public ANE command queue, so it is not submitted to the ANE. BNNS is Accelerate (CPU/AMX), not the ANE. This shader runs on the GPU via Metal.";

/// Metal Shading Language for the fabric sum. One threadgroup, f32 accumulation.
pub const FABRIC_REDUCE_SUM_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fabric_reduce_sum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]]
) {
    threadgroup float scratch[256];
    float acc = 0.0;
    for (uint i = tid; i < count; i += tg) {
        acc += input[i];
    }
    scratch[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        output[0] = scratch[0];
    }
}
"#;

/// f32 sequential sum matching the Metal kernel's accumulation type.
///
/// Descriptor tests on hosts that cannot submit Metal call this. It is not a
/// GPU dispatch and it is not an ANE dispatch.
pub fn reference_f32_sum(values: &[f64]) -> f64 {
    let mut acc = 0f32;
    for value in values {
        acc += *value as f32;
    }
    f64::from(acc)
}

/// Largest power of two not above `cap`, at least 1, at most 256.
fn threadgroup_width(max_threads: u32) -> u32 {
    let cap = max_threads.min(256).max(1);
    if cap <= 1 {
        return 1;
    }
    1u32 << (31 - cap.leading_zeros())
}

/// Sum `values` by compiling `FABRIC_REDUCE_SUM_MSL` and dispatching it.
///
/// On macOS this submits a real Metal compute command buffer. Elsewhere it
/// returns [`MetalError::NotAvailable`] and does not pretend the GPU or the
/// ANE ran.
pub fn sum_f64(values: &[f64]) -> MetalResult<f64> {
    #[cfg(target_os = "macos")]
    {
        sum_f64_metal(values)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = values;
        Err(MetalError::not_macos())
    }
}

#[cfg(target_os = "macos")]
fn sum_f64_metal(values: &[f64]) -> MetalResult<f64> {
    use crate::{BufferUsage, MetalDevice};

    let device = MetalDevice::system_default()?;
    let f32s: Vec<f32> = values.iter().map(|value| *value as f32).collect();
    let count = f32s.len() as u32;
    // Shared buffers so the CPU can read the GPU result without a blit.
    let input_words: Vec<f32> = if f32s.is_empty() { vec![0.0] } else { f32s };
    let input = device.create_buffer_with_data(&input_words, BufferUsage::Shared)?;
    let output = device.create_buffer_with_data(&[0f32], BufferUsage::Shared)?;
    let count_buf = device.create_buffer_with_data(&[count], BufferUsage::Shared)?;
    let library = device.create_library_from_source(FABRIC_REDUCE_SUM_MSL)?;
    let pipeline = device.create_compute_pipeline(&library, "fabric_reduce_sum")?;
    let queue = device.create_command_queue()?;
    let threads = threadgroup_width(device.max_threads_per_threadgroup());
    queue.launch_kernel(
        &pipeline,
        (1, 1, 1),
        (threads, 1, 1),
        &[&input, &output, &count_buf],
    )?;
    let out: Vec<f32> = output.read(1)?;
    Ok(f64::from(out.first().copied().unwrap_or(0.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_sum_matches_small_integers() {
        let sum = reference_f32_sum(&[1.0, 2.0, 3.0]);
        assert!((sum - 6.0).abs() < 1e-6, "sum={sum}");
        assert_eq!(reference_f32_sum(&[]), 0.0);
    }

    #[test]
    fn shader_is_a_gpu_kernel_not_an_ane_graph() {
        assert!(FABRIC_REDUCE_SUM_MSL.contains("kernel void fabric_reduce_sum"));
        assert!(FABRIC_REDUCE_SUM_MSL.contains("threadgroup_barrier"));
        assert!(ANE_DOES_NOT_RUN_THIS_KERNEL.contains("Core ML"));
        assert!(ANE_DOES_NOT_RUN_THIS_KERNEL.contains("GPU"));
        assert!(!ANE_DOES_NOT_RUN_THIS_KERNEL.to_ascii_lowercase().contains("submitted to the ane and ran"));
    }

    #[test]
    fn threadgroup_width_is_power_of_two_at_most_256() {
        assert_eq!(threadgroup_width(1024), 256);
        assert_eq!(threadgroup_width(256), 256);
        assert_eq!(threadgroup_width(100), 64);
        assert_eq!(threadgroup_width(1), 1);
        assert_eq!(threadgroup_width(0), 1);
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn sum_f64_is_not_available_off_macos() {
        let err = sum_f64(&[1.0, 2.0, 3.0]).unwrap_err();
        assert!(matches!(err, MetalError::NotAvailable { .. }));
        let msg = err.to_string();
        assert!(msg.contains("macOS") || msg.contains("Metal"), "{msg}");
    }
}
