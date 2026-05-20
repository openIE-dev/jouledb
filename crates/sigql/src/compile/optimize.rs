//! Query Optimization
//!
//! Optimizes execution plans for performance.

use super::plan::{ExecutionPlan, FirCoeffs, IirCoeffs, PlanStep, RegisterId};
use std::collections::{HashMap, HashSet};

/// Optimization passes
pub enum OptimizationPass {
    /// Fuse consecutive filters into one
    FilterFusion,
    /// Reorder operations to minimize memory
    MemoryOptimization,
    /// Choose optimal FFT sizes
    FftSizeSelection,
    /// Eliminate redundant computations
    CommonSubexpressionElimination,
    /// Parallelize independent operations
    Parallelization,
}

/// Apply optimization passes to a plan
pub fn optimize(plan: ExecutionPlan, passes: &[OptimizationPass]) -> ExecutionPlan {
    let mut optimized = plan;

    for pass in passes {
        optimized = match pass {
            OptimizationPass::FilterFusion => fuse_filters(optimized),
            OptimizationPass::MemoryOptimization => optimize_memory(optimized),
            OptimizationPass::FftSizeSelection => select_fft_sizes(optimized),
            OptimizationPass::CommonSubexpressionElimination => eliminate_cse(optimized),
            OptimizationPass::Parallelization => parallelize(optimized),
        };
    }

    optimized
}

/// Fuse consecutive IIR and FIR filters that operate on the same signal path
fn fuse_filters(mut plan: ExecutionPlan) -> ExecutionPlan {
    let mut optimized_steps = Vec::new();
    let mut i = 0;

    while i < plan.steps.len() {
        let step = &plan.steps[i];

        // Look for consecutive IIR filters
        if let PlanStep::IirFilter {
            input,
            output,
            coeffs,
        } = step
        {
            let mut fused_sections = coeffs.sections.clone();
            let original_input = *input;
            let mut final_output = *output;
            let mut j = i + 1;

            // Look ahead for more IIR filters that chain from our output
            while j < plan.steps.len() {
                if let PlanStep::IirFilter {
                    input: next_input,
                    output: next_output,
                    coeffs: next_coeffs,
                } = &plan.steps[j]
                {
                    if next_input.0 == final_output.0 {
                        // Cascade the filter sections
                        fused_sections.extend(next_coeffs.sections.clone());
                        final_output = *next_output;
                        j += 1;
                        continue;
                    }
                }
                break;
            }

            // If we fused multiple filters, create a single combined filter
            if j > i + 1 {
                optimized_steps.push(PlanStep::IirFilter {
                    input: original_input,
                    output: final_output,
                    coeffs: IirCoeffs {
                        sections: fused_sections,
                    },
                });
                i = j;
                continue;
            }
        }

        // Look for consecutive FIR filters
        if let PlanStep::FirFilter {
            input,
            output,
            coeffs,
        } = step
        {
            let mut combined_taps = coeffs.taps.clone();
            let original_input = *input;
            let mut final_output = *output;
            let mut j = i + 1;

            // Look ahead for more FIR filters that chain from our output
            while j < plan.steps.len() {
                if let PlanStep::FirFilter {
                    input: next_input,
                    output: next_output,
                    coeffs: next_coeffs,
                } = &plan.steps[j]
                {
                    if next_input.0 == final_output.0 {
                        // Convolve filter taps to create combined filter
                        combined_taps = convolve_taps(&combined_taps, &next_coeffs.taps);
                        final_output = *next_output;
                        j += 1;
                        continue;
                    }
                }
                break;
            }

            // If we fused multiple filters, create a single combined filter
            if j > i + 1 {
                optimized_steps.push(PlanStep::FirFilter {
                    input: original_input,
                    output: final_output,
                    coeffs: FirCoeffs {
                        taps: combined_taps,
                    },
                });
                i = j;
                continue;
            }
        }

        optimized_steps.push(plan.steps[i].clone());
        i += 1;
    }

    plan.steps = optimized_steps;
    plan
}

/// Convolve two FIR filter tap vectors
fn convolve_taps(a: &[f64], b: &[f64]) -> Vec<f64> {
    let len = a.len() + b.len() - 1;
    let mut result = vec![0.0; len];

    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            result[i + j] += av * bv;
        }
    }

    result
}

/// Optimize memory usage by reusing registers when no longer needed
fn optimize_memory(mut plan: ExecutionPlan) -> ExecutionPlan {
    // Build a liveness map: track which registers are alive at each step
    let mut last_use: HashMap<u32, usize> = HashMap::new();

    // First pass: find last use of each register
    for (step_idx, step) in plan.steps.iter().enumerate() {
        let inputs = get_step_inputs(step);
        for input in inputs {
            last_use.insert(input.0, step_idx);
        }
    }

    // Second pass: build a pool of dead registers to reuse
    let mut free_registers: Vec<u32> = Vec::new();
    let mut register_map: HashMap<u32, u32> = HashMap::new();
    let mut next_register = 0u32;

    for (step_idx, step) in plan.steps.iter_mut().enumerate() {
        // Free registers that died at the previous step
        for (&reg, &last_step) in &last_use {
            if last_step + 1 == step_idx {
                if let Some(&mapped) = register_map.get(&reg) {
                    free_registers.push(mapped);
                }
            }
        }

        // Remap inputs
        remap_step_inputs(step, &register_map);

        // Assign output register (reuse from pool if available)
        if let Some(output) = get_step_output_mut(step) {
            let old_reg = output.0;
            let new_reg = free_registers.pop().unwrap_or_else(|| {
                let r = next_register;
                next_register += 1;
                r
            });
            register_map.insert(old_reg, new_reg);
            output.0 = new_reg;
        }
    }

    plan
}

/// Get input registers for a step
fn get_step_inputs(step: &PlanStep) -> Vec<RegisterId> {
    match step {
        PlanStep::LoadSignal { .. } => vec![],
        PlanStep::Fft { input, .. } => vec![*input],
        PlanStep::Ifft { input, .. } => vec![*input],
        PlanStep::IirFilter { input, .. } => vec![*input],
        PlanStep::FirFilter { input, .. } => vec![*input],
        PlanStep::Resample { input, .. } => vec![*input],
        PlanStep::ComplexToMagnitude { input, .. } => vec![*input],
        PlanStep::Envelope { input, .. } => vec![*input],
        PlanStep::Reduce { input, .. } => vec![*input],
        PlanStep::Window { input, .. } => vec![*input],
        PlanStep::CrossCorrelate {
            input_a, input_b, ..
        } => vec![*input_a, *input_b],
        PlanStep::BandPower { input, .. } => vec![*input],
        PlanStep::Store { input, .. } => vec![*input],
        PlanStep::ZScore { input, .. } => vec![*input],
        PlanStep::Detrend { input, .. } => vec![*input],
        PlanStep::ElementWise { input, .. } => vec![*input],
        PlanStep::Diff { input, .. } => vec![*input],
        PlanStep::Cumsum { input, .. } => vec![*input],
        PlanStep::MedianFilter { input, .. } => vec![*input],
        PlanStep::Decimate { input, .. } => vec![*input],
        PlanStep::Interpolate { input, .. } => vec![*input],
        PlanStep::DominantFrequency { input, .. } => vec![*input],
        PlanStep::SpectralEntropy { input, .. } => vec![*input],
        PlanStep::SpectralCentroid { input, .. } => vec![*input],
        PlanStep::Passthrough { input, .. } => vec![*input],
        PlanStep::Fft2d { input, .. } => vec![*input],
        PlanStep::Ifft2d { input, .. } => vec![*input],
        PlanStep::Dct2d { input, .. } => vec![*input],
        PlanStep::Idct2d { input, .. } => vec![*input],
        PlanStep::Mfcc { input, .. } => vec![*input],
        PlanStep::PerceptualHash { input, .. } => vec![*input],
        PlanStep::EdgeDetect { input, .. } => vec![*input],
    }
}

/// Get mutable reference to output register
fn get_step_output_mut(step: &mut PlanStep) -> Option<&mut RegisterId> {
    match step {
        PlanStep::LoadSignal { output, .. } => Some(output),
        PlanStep::Fft { output, .. } => Some(output),
        PlanStep::Ifft { output, .. } => Some(output),
        PlanStep::IirFilter { output, .. } => Some(output),
        PlanStep::FirFilter { output, .. } => Some(output),
        PlanStep::Resample { output, .. } => Some(output),
        PlanStep::ComplexToMagnitude { output, .. } => Some(output),
        PlanStep::Envelope { output, .. } => Some(output),
        PlanStep::Reduce { output, .. } => Some(output),
        PlanStep::Window { output, .. } => Some(output),
        PlanStep::CrossCorrelate { output, .. } => Some(output),
        PlanStep::BandPower { output, .. } => Some(output),
        PlanStep::Store { .. } => None,
        PlanStep::ZScore { output, .. } => Some(output),
        PlanStep::Detrend { output, .. } => Some(output),
        PlanStep::ElementWise { output, .. } => Some(output),
        PlanStep::Diff { output, .. } => Some(output),
        PlanStep::Cumsum { output, .. } => Some(output),
        PlanStep::MedianFilter { output, .. } => Some(output),
        PlanStep::Decimate { output, .. } => Some(output),
        PlanStep::Interpolate { output, .. } => Some(output),
        PlanStep::DominantFrequency { output, .. } => Some(output),
        PlanStep::SpectralEntropy { output, .. } => Some(output),
        PlanStep::SpectralCentroid { output, .. } => Some(output),
        PlanStep::Passthrough { output, .. } => Some(output),
        PlanStep::Fft2d { output, .. } => Some(output),
        PlanStep::Ifft2d { output, .. } => Some(output),
        PlanStep::Dct2d { output, .. } => Some(output),
        PlanStep::Idct2d { output, .. } => Some(output),
        PlanStep::Mfcc { output, .. } => Some(output),
        PlanStep::PerceptualHash { output, .. } => Some(output),
        PlanStep::EdgeDetect { output, .. } => Some(output),
    }
}

/// Remap input registers based on mapping
fn remap_step_inputs(step: &mut PlanStep, map: &HashMap<u32, u32>) {
    let remap = |r: &mut RegisterId| {
        if let Some(&mapped) = map.get(&r.0) {
            r.0 = mapped;
        }
    };

    match step {
        PlanStep::LoadSignal { .. } => {}
        PlanStep::Fft { input, .. } => remap(input),
        PlanStep::Ifft { input, .. } => remap(input),
        PlanStep::IirFilter { input, .. } => remap(input),
        PlanStep::FirFilter { input, .. } => remap(input),
        PlanStep::Resample { input, .. } => remap(input),
        PlanStep::ComplexToMagnitude { input, .. } => remap(input),
        PlanStep::Envelope { input, .. } => remap(input),
        PlanStep::Reduce { input, .. } => remap(input),
        PlanStep::Window { input, .. } => remap(input),
        PlanStep::CrossCorrelate {
            input_a, input_b, ..
        } => {
            remap(input_a);
            remap(input_b);
        }
        PlanStep::BandPower { input, .. } => remap(input),
        PlanStep::Store { input, .. } => remap(input),
        PlanStep::ZScore { input, .. } => remap(input),
        PlanStep::Detrend { input, .. } => remap(input),
        PlanStep::ElementWise { input, .. } => remap(input),
        PlanStep::Diff { input, .. } => remap(input),
        PlanStep::Cumsum { input, .. } => remap(input),
        PlanStep::MedianFilter { input, .. } => remap(input),
        PlanStep::Decimate { input, .. } => remap(input),
        PlanStep::Interpolate { input, .. } => remap(input),
        PlanStep::DominantFrequency { input, .. } => remap(input),
        PlanStep::SpectralEntropy { input, .. } => remap(input),
        PlanStep::SpectralCentroid { input, .. } => remap(input),
        PlanStep::Passthrough { input, .. } => remap(input),
        PlanStep::Fft2d { input, .. } => remap(input),
        PlanStep::Ifft2d { input, .. } => remap(input),
        PlanStep::Dct2d { input, .. } => remap(input),
        PlanStep::Idct2d { input, .. } => remap(input),
        PlanStep::Mfcc { input, .. } => remap(input),
        PlanStep::PerceptualHash { input, .. } => remap(input),
        PlanStep::EdgeDetect { input, .. } => remap(input),
    }
}

/// Select optimal FFT sizes (power of 2 or efficient composites for speed)
fn select_fft_sizes(mut plan: ExecutionPlan) -> ExecutionPlan {
    for step in &mut plan.steps {
        if let PlanStep::Fft { size, .. } = step {
            *size = optimal_fft_size(*size);
        }
    }
    plan
}

/// Find the optimal FFT size >= n
fn optimal_fft_size(n: usize) -> usize {
    if n == 0 {
        return 0;
    }

    // Try to find a composite with small prime factors (2, 3, 5)
    // These are highly optimized in most FFT libraries
    let _best = n;
    let mut candidate = n;

    // Search for efficient sizes up to 2x the requested size
    while candidate <= n * 2 {
        if is_efficient_fft_size(candidate) {
            return candidate;
        }
        candidate += 1;
    }

    // Fall back to next power of 2
    let mut power = 1;
    while power < n {
        power <<= 1;
    }
    power
}

/// Check if n has only small prime factors (2, 3, 5) - efficient for FFT
fn is_efficient_fft_size(mut n: usize) -> bool {
    if n == 0 {
        return false;
    }

    // Factor out 2s, 3s, and 5s
    while n % 2 == 0 {
        n /= 2;
    }
    while n % 3 == 0 {
        n /= 3;
    }
    while n % 5 == 0 {
        n /= 5;
    }

    // If we've reduced to 1, only small factors were present
    n == 1
}

/// Eliminate common subexpressions by detecting identical operations
fn eliminate_cse(mut plan: ExecutionPlan) -> ExecutionPlan {
    // Hash each step to detect duplicates
    let mut seen: HashMap<String, RegisterId> = HashMap::new();
    let mut optimized_steps = Vec::new();
    let mut register_substitutions: HashMap<u32, u32> = HashMap::new();

    for mut step in plan.steps {
        // Apply any pending substitutions
        remap_step_inputs(&mut step, &register_substitutions);

        // Generate a hash key for this step
        let hash_key = step_hash_key(&step);

        if let Some(&existing_output) = seen.get(&hash_key) {
            // This is a duplicate - redirect to existing output
            if let Some(output) = get_step_output_mut(&mut step) {
                register_substitutions.insert(output.0, existing_output.0);
                // Skip this step entirely
                continue;
            }
        }

        // Record this step's output
        if let Some(output) = get_step_output_mut(&mut step) {
            seen.insert(hash_key, *output);
        }

        optimized_steps.push(step);
    }

    plan.steps = optimized_steps;
    plan
}

/// Generate a hash key for a step (excluding output register)
fn step_hash_key(step: &PlanStep) -> String {
    match step {
        PlanStep::Fft { input, size, .. } => format!("FFT:{}:{}", input.0, size),
        PlanStep::Ifft { input, .. } => format!("IFFT:{}", input.0),
        PlanStep::IirFilter { input, coeffs, .. } => {
            format!("IIR:{}:{:?}", input.0, coeffs.sections.len())
        }
        PlanStep::FirFilter { input, coeffs, .. } => {
            format!("FIR:{}:{}", input.0, coeffs.taps.len())
        }
        PlanStep::ComplexToMagnitude { input, .. } => format!("MAG:{}", input.0),
        PlanStep::Envelope { input, .. } => format!("ENV:{}", input.0),
        PlanStep::Reduce { input, op, .. } => format!("RED:{}:{:?}", input.0, op),
        PlanStep::ZScore { input, .. } => format!("ZSCORE:{}", input.0),
        PlanStep::ElementWise { input, op, .. } => format!("EW:{}:{:?}", input.0, op),
        // Other steps are less likely to be duplicated or aren't safe to dedupe
        _ => format!("UNIQUE:{:p}", step),
    }
}

/// Mark independent operations for parallel execution
fn parallelize(mut plan: ExecutionPlan) -> ExecutionPlan {
    // Build dependency graph
    let mut dependencies: Vec<HashSet<usize>> = vec![HashSet::new(); plan.steps.len()];
    let mut output_to_step: HashMap<u32, usize> = HashMap::new();

    for (i, step) in plan.steps.iter().enumerate() {
        // Record output register
        if let Some(output) = match step {
            PlanStep::LoadSignal { output, .. } => Some(output),
            PlanStep::Fft { output, .. } => Some(output),
            PlanStep::Ifft { output, .. } => Some(output),
            PlanStep::IirFilter { output, .. } => Some(output),
            PlanStep::FirFilter { output, .. } => Some(output),
            PlanStep::Resample { output, .. } => Some(output),
            PlanStep::ComplexToMagnitude { output, .. } => Some(output),
            PlanStep::Envelope { output, .. } => Some(output),
            PlanStep::Reduce { output, .. } => Some(output),
            PlanStep::Window { output, .. } => Some(output),
            PlanStep::CrossCorrelate { output, .. } => Some(output),
            PlanStep::BandPower { output, .. } => Some(output),
            PlanStep::ZScore { output, .. } => Some(output),
            PlanStep::Detrend { output, .. } => Some(output),
            PlanStep::ElementWise { output, .. } => Some(output),
            PlanStep::Diff { output, .. } => Some(output),
            PlanStep::Cumsum { output, .. } => Some(output),
            PlanStep::MedianFilter { output, .. } => Some(output),
            PlanStep::Decimate { output, .. } => Some(output),
            PlanStep::Interpolate { output, .. } => Some(output),
            PlanStep::DominantFrequency { output, .. } => Some(output),
            PlanStep::SpectralEntropy { output, .. } => Some(output),
            PlanStep::SpectralCentroid { output, .. } => Some(output),
            PlanStep::Passthrough { output, .. } => Some(output),
            PlanStep::Fft2d { output, .. } => Some(output),
            PlanStep::Ifft2d { output, .. } => Some(output),
            PlanStep::Dct2d { output, .. } => Some(output),
            PlanStep::Idct2d { output, .. } => Some(output),
            PlanStep::Mfcc { output, .. } => Some(output),
            PlanStep::PerceptualHash { output, .. } => Some(output),
            PlanStep::EdgeDetect { output, .. } => Some(output),
            PlanStep::Store { .. } => None,
        } {
            output_to_step.insert(output.0, i);
        }

        // Find dependencies from inputs
        let inputs = get_step_inputs(step);
        for input in inputs {
            if let Some(&dep_step) = output_to_step.get(&input.0) {
                dependencies[i].insert(dep_step);
            }
        }
    }

    // Group steps into parallel batches
    // Steps with no unprocessed dependencies can run in parallel
    let mut processed: HashSet<usize> = HashSet::new();
    let mut parallel_groups: Vec<Vec<usize>> = Vec::new();

    while processed.len() < plan.steps.len() {
        let mut current_group = Vec::new();

        for i in 0..plan.steps.len() {
            if processed.contains(&i) {
                continue;
            }

            // Check if all dependencies are processed
            if dependencies[i].iter().all(|dep| processed.contains(dep)) {
                current_group.push(i);
            }
        }

        // Mark this group as processed
        for &i in &current_group {
            processed.insert(i);
        }

        parallel_groups.push(current_group);
    }

    // Reorder steps based on parallel groups
    // (In a real implementation, we'd add parallel execution markers)
    let mut reordered_steps = Vec::with_capacity(plan.steps.len());
    for group in parallel_groups {
        for i in group {
            reordered_steps.push(plan.steps[i].clone());
        }
    }

    plan.steps = reordered_steps;
    plan
}
