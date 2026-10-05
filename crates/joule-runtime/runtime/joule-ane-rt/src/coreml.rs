//! macOS Core ML driver: compile, load with `CPUAndNeuralEngine`, predict
//! with Float16 `MLMultiArray`, and read `MLComputePlan` placement.

use crate::model::{INPUT_NAME, ModelShape, OUTPUT_NAME, write_mlpackage};
use crate::{
    AneError, EXACT_TILE_DIM, HdcOptions, HdcRun, Layout, OpPlacement, Placement, PlannedDevice,
    PowerSampling, fingerprint, power, tile_columns,
};
use block2::{RcBlock, StackBlock};
use core::ffi::c_void;
use core::ptr::NonNull;
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_ml::{
    MLCPUComputeDevice, MLComputeDeviceProtocol, MLComputePlan, MLComputeUnits,
    MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLGPUComputeDevice, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType, MLNeuralEngineComputeDevice,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSInteger, NSNumber, NSString, NSURL};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const COMPUTE_UNITS_NAME: &str = "cpu_and_neural_engine";

fn ns_error(context: &str, e: &NSError) -> AneError {
    AneError::CoreMl(format!("{context}: {}", e.localizedDescription()))
}

fn file_url(path: &Path) -> Result<Retained<NSURL>, AneError> {
    let s = path
        .to_str()
        .ok_or_else(|| AneError::Io(format!("non-UTF-8 path {}", path.display())))?;
    Ok(NSURL::fileURLWithPath(&NSString::from_str(s)))
}

fn cache_dir() -> PathBuf {
    std::env::temp_dir().join("joule-ane-rt")
}

fn copy_dir_all(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_all(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

/// Generate and compile the model for `shape` + `codebook` tile, caching the
/// `.mlmodelc` by fingerprint. Returns `(mlmodelc path, compile seconds)`.
fn compile(shape: &ModelShape, codebook: &[i8]) -> Result<(PathBuf, f64), AneError> {
    let key = fingerprint(shape, codebook);
    let dir = cache_dir();
    let compiled = dir.join(format!("hdc-{key:016x}.mlmodelc"));
    if compiled.join("coremldata.bin").exists() {
        return Ok((compiled, 0.0));
    }
    let io = |e: std::io::Error| AneError::Io(e.to_string());
    std::fs::create_dir_all(&dir).map_err(io)?;
    let started = Instant::now();
    let package = dir.join(format!("hdc-{key:016x}-{}.mlpackage", std::process::id()));
    let _ = std::fs::remove_dir_all(&package);
    write_mlpackage(&package, shape, codebook)?;
    let url = file_url(&package)?;
    // The synchronous form is deprecated in favour of the completion-handler
    // form but remains supported; the fabric call path is synchronous.
    #[allow(deprecated)]
    let out = unsafe { MLModel::compileModelAtURL_error(&url) }
        .map_err(|e| ns_error("compileModelAtURL", &e))?;
    let out_path = out
        .path()
        .map(|p| PathBuf::from(p.to_string()))
        .ok_or_else(|| AneError::CoreMl("compiled model URL has no path".into()))?;
    let staging = dir.join(format!("hdc-{key:016x}-{}.staging", std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);
    if std::fs::rename(&out_path, &staging).is_err() {
        copy_dir_all(&out_path, &staging).map_err(io)?;
        let _ = std::fs::remove_dir_all(&out_path);
    }
    let _ = std::fs::remove_dir_all(&compiled);
    std::fs::rename(&staging, &compiled).map_err(io)?;
    let _ = std::fs::remove_dir_all(&package);
    Ok((compiled, started.elapsed().as_secs_f64()))
}

fn configuration() -> Retained<MLModelConfiguration> {
    let config = unsafe { MLModelConfiguration::new() };
    unsafe { config.setComputeUnits(MLComputeUnits::CPUAndNeuralEngine) };
    config
}

fn load(path: &Path, config: &MLModelConfiguration) -> Result<Retained<MLModel>, AneError> {
    let url = file_url(path)?;
    unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, config) }
        .map_err(|e| ns_error("modelWithContentsOfURL", &e))
}

fn ns_shape(dims: &[i64]) -> Retained<NSArray<NSNumber>> {
    let numbers: Vec<Retained<NSNumber>> = dims
        .iter()
        .map(|d| NSNumber::numberWithInteger(*d as NSInteger))
        .collect();
    NSArray::from_retained_slice(&numbers)
}

fn ns_ints(array: &NSArray<NSNumber>) -> Vec<isize> {
    array.iter().map(|n| n.integerValue()).collect()
}

/// Logical (query i, column t or codebook j) -> element offset via strides.
fn element_offset(layout: Layout, strides: &[isize], row: usize, col: usize) -> isize {
    match layout {
        // [n, x]
        Layout::MatMul => row as isize * strides[0] + col as isize * strides[1],
        // [1, x, 1, n]
        Layout::ChannelsFirstConv => col as isize * strides[1] + row as isize * strides[3],
    }
}

/// Float16 input array holding the `n x dt` query tile.
fn make_input(shape: &ModelShape, q_tile: &[i8]) -> Result<Retained<MLMultiArray>, AneError> {
    let dims = shape.input_shape();
    let array = unsafe {
        MLMultiArray::initWithShape_dataType_error(
            MLMultiArray::alloc(),
            &ns_shape(&dims),
            MLMultiArrayDataType::Float16,
        )
    }
    .map_err(|e| ns_error("MLMultiArray initWithShape", &e))?;
    let (n, dt, layout) = (shape.n, shape.d, shape.layout);
    let failed = RefCell::new(None::<String>);
    let block = StackBlock::new(
        |ptr: NonNull<c_void>, len: NSInteger, strides: NonNull<NSArray<NSNumber>>| {
            let strides = ns_ints(unsafe { strides.as_ref() });
            if strides.len() != dims.len() {
                *failed.borrow_mut() = Some(format!("input strides {strides:?} vs dims {dims:?}"));
                return;
            }
            let base = ptr.as_ptr().cast::<u16>();
            let elems = len as usize / 2;
            let one = half::f16::from_f32(1.0).to_bits();
            let minus = half::f16::from_f32(-1.0).to_bits();
            for i in 0..n {
                for t in 0..dt {
                    let off = element_offset(layout, &strides, i, t);
                    if off < 0 || off as usize >= elems {
                        *failed.borrow_mut() = Some(format!("input offset {off} out of {elems}"));
                        return;
                    }
                    let v = if q_tile[i * dt + t] > 0 { one } else { minus };
                    unsafe { base.offset(off).write(v) };
                }
            }
        },
    );
    unsafe { array.getMutableBytesWithHandler(&block) };
    if let Some(msg) = failed.into_inner() {
        return Err(AneError::CoreMl(msg));
    }
    Ok(array)
}

fn predict(model: &MLModel, input: &MLMultiArray) -> Result<Retained<MLMultiArray>, AneError> {
    let key = NSString::from_str(INPUT_NAME);
    let value: Retained<AnyObject> = unsafe { MLFeatureValue::featureValueWithMultiArray(input) }.into();
    let dict: Retained<NSDictionary<NSString, AnyObject>> =
        NSDictionary::from_retained_objects(&[&*key], &[value]);
    let provider = unsafe {
        MLDictionaryFeatureProvider::initWithDictionary_error(
            MLDictionaryFeatureProvider::alloc(),
            &dict,
        )
    }
    .map_err(|e| ns_error("MLDictionaryFeatureProvider", &e))?;
    let provider: &ProtocolObject<dyn MLFeatureProvider> = ProtocolObject::from_ref(&*provider);
    let out = unsafe { model.predictionFromFeatures_error(provider) }
        .map_err(|e| ns_error("predictionFromFeatures", &e))?;
    let feature = unsafe { out.featureValueForName(&NSString::from_str(OUTPUT_NAME)) }
        .ok_or_else(|| AneError::CoreMl(format!("prediction has no '{OUTPUT_NAME}' output")))?;
    unsafe { feature.multiArrayValue() }
        .ok_or_else(|| AneError::CoreMl("output is not a multi-array".into()))
}

/// Read the `n x k` scores (row-major) out of the output array, honouring
/// its strides and data type.
fn read_scores(shape: &ModelShape, out: &MLMultiArray) -> Result<Vec<f32>, AneError> {
    let dims = ns_ints(&*unsafe { out.shape() });
    let want: Vec<isize> = shape.output_shape().iter().map(|d| *d as isize).collect();
    if dims != want {
        return Err(AneError::CoreMl(format!("output shape {dims:?}, expected {want:?}")));
    }
    let strides = ns_ints(&*unsafe { out.strides() });
    let dtype = unsafe { out.dataType() };
    let (n, k, layout) = (shape.n, shape.k, shape.layout);
    let result = RefCell::new(Err::<Vec<f32>, String>("handler not called".into()));
    let block = StackBlock::new(|ptr: NonNull<c_void>, len: NSInteger| {
        let width = if dtype == MLMultiArrayDataType::Float16 {
            2
        } else if dtype == MLMultiArrayDataType::Float32 {
            4
        } else if dtype == MLMultiArrayDataType::Double {
            8
        } else {
            *result.borrow_mut() = Err(format!("unsupported output data type {dtype:?}"));
            return;
        };
        let elems = len as usize / width;
        let mut values = Vec::with_capacity(n * k);
        for i in 0..n {
            for j in 0..k {
                let off = element_offset(layout, &strides, i, j);
                if off < 0 || off as usize >= elems {
                    *result.borrow_mut() = Err(format!("output offset {off} out of {elems}"));
                    return;
                }
                let v = unsafe {
                    match width {
                        2 => half::f16::from_bits(ptr.as_ptr().cast::<u16>().offset(off).read())
                            .to_f32(),
                        4 => ptr.as_ptr().cast::<f32>().offset(off).read(),
                        _ => ptr.as_ptr().cast::<f64>().offset(off).read() as f32,
                    }
                };
                values.push(v);
            }
        }
        *result.borrow_mut() = Ok(values);
    });
    unsafe { out.getBytesWithHandler(&block) };
    result.into_inner().map_err(AneError::CoreMl)
}

fn classify(device: &ProtocolObject<dyn MLComputeDeviceProtocol>) -> PlannedDevice {
    let obj: &AnyObject = device.as_ref();
    if obj.downcast_ref::<MLNeuralEngineComputeDevice>().is_some() {
        PlannedDevice::NeuralEngine
    } else if obj.downcast_ref::<MLGPUComputeDevice>().is_some() {
        PlannedDevice::Gpu
    } else if obj.downcast_ref::<MLCPUComputeDevice>().is_some() {
        PlannedDevice::Cpu
    } else {
        PlannedDevice::Unknown
    }
}

/// Walk the plan's `main` function and record per-op device usage.
fn extract_plan(plan: &MLComputePlan) -> Result<Vec<OpPlacement>, String> {
    let structure = unsafe { plan.modelStructure() };
    let program = unsafe { structure.program() }
        .ok_or_else(|| "compute plan model structure is not an ML Program".to_string())?;
    let functions = unsafe { program.functions() };
    let main = functions
        .objectForKey(&NSString::from_str("main"))
        .ok_or_else(|| "ML Program has no 'main' function".to_string())?;
    let block = unsafe { main.block() };
    let mut ops = Vec::new();
    for op in unsafe { block.operations() }.iter() {
        let operator = unsafe { op.operatorName() }.to_string();
        let outputs = unsafe { op.outputs() }
            .iter()
            .map(|o| unsafe { o.name() }.to_string())
            .collect();
        let usage = unsafe { plan.computeDeviceUsageForMLProgramOperation(&op) };
        let (preferred, supported) = match usage {
            Some(u) => (
                Some(classify(&*unsafe { u.preferredComputeDevice() })),
                unsafe { u.supportedComputeDevices() }
                    .iter()
                    .map(|d| classify(&d))
                    .collect(),
            ),
            None => (None, Vec::new()),
        };
        let estimated_cost =
            unsafe { plan.estimatedCostOfMLProgramOperation(&op) }.map(|c| unsafe { c.weight() });
        ops.push(OpPlacement {
            operator,
            outputs,
            preferred,
            supported,
            estimated_cost,
        });
    }
    Ok(ops)
}

/// `MLComputePlan loadContentsOfURL:configuration:completionHandler:` for the
/// compiled model, waiting for the completion handler.
fn compute_plan(
    path: &Path,
    config: &MLModelConfiguration,
    layout: Layout,
) -> Result<Placement, AneError> {
    let url = file_url(path)?;
    let (tx, rx) = std::sync::mpsc::channel::<Result<Vec<OpPlacement>, String>>();
    let handler = RcBlock::new(move |plan: *mut MLComputePlan, error: *mut NSError| {
        let result = if let Some(plan) = unsafe { plan.as_ref() } {
            extract_plan(plan)
        } else if let Some(error) = unsafe { error.as_ref() } {
            Err(format!("MLComputePlan: {}", error.localizedDescription()))
        } else {
            Err("MLComputePlan returned neither a plan nor an error".to_string())
        };
        let _ = tx.send(result);
    });
    unsafe {
        MLComputePlan::loadContentsOfURL_configuration_completionHandler(&url, config, &handler)
    };
    let ops = rx
        .recv_timeout(Duration::from_secs(120))
        .map_err(|_| AneError::CoreMl("MLComputePlan completion handler timed out".into()))?
        .map_err(AneError::CoreMl)?;
    Ok(Placement {
        layout,
        compute_units: COMPUTE_UNITS_NAME,
        ops,
    })
}

/// A compiled + loaded model kept for the life of the process, so repeat
/// jobs with the same shape and codebook skip compile, load and plan.
struct LoadedModel {
    model: Retained<MLModel>,
    placement: Placement,
    /// Serialises predictions on this `MLModel`.
    predict_lock: Mutex<()>,
}

// SAFETY: Objective-C retain/release is atomic, so moving the `Retained`
// handle between threads is sound. Predictions on one model are serialised
// by `predict_lock`; placement is plain Rust data.
unsafe impl Send for LoadedModel {}
unsafe impl Sync for LoadedModel {}

/// Bound on resident models; the cache is cleared when exceeded.
const MAX_LOADED_MODELS: usize = 32;

fn loaded_models() -> &'static Mutex<HashMap<u64, Arc<LoadedModel>>> {
    static MODELS: OnceLock<Mutex<HashMap<u64, Arc<LoadedModel>>>> = OnceLock::new();
    MODELS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Get the loaded model for `shape` + `codebook`, doing compile / plan /
/// load only on a miss. Returns `(model, did_setup, compile_s)`.
fn model_for(
    shape: &ModelShape,
    codebook: &[i8],
    config: &MLModelConfiguration,
) -> Result<(Arc<LoadedModel>, bool, f64), AneError> {
    let key = fingerprint(shape, codebook);
    if let Some(hit) = loaded_models()
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get(&key)
        .cloned()
    {
        return Ok((hit, false, 0.0));
    }
    let (path, compile_s) = compile(shape, codebook)?;
    let placement = compute_plan(&path, config, shape.layout)?;
    let model = load(&path, config)?;
    let loaded = Arc::new(LoadedModel {
        model,
        placement,
        predict_lock: Mutex::new(()),
    });
    let mut map = loaded_models().lock().unwrap_or_else(|p| p.into_inner());
    if map.len() >= MAX_LOADED_MODELS {
        map.clear();
    }
    map.insert(key, Arc::clone(&loaded));
    Ok((loaded, true, compile_s))
}

struct Tile {
    shape: ModelShape,
    model: Arc<LoadedModel>,
    q_tile: Vec<i8>,
}

pub(crate) fn hdc_similarity(
    queries: &[i8],
    memory: &[i8],
    n: usize,
    k: usize,
    d: usize,
    options: HdcOptions,
) -> Result<HdcRun, AneError> {
    let config = configuration();
    let tiles = d.div_ceil(EXACT_TILE_DIM);
    let mut compile_s = 0.0;
    let mut setup_performed = false;
    let setup_started = Instant::now();
    let mut prepared = Vec::with_capacity(tiles);
    for t in 0..tiles {
        let start = t * EXACT_TILE_DIM;
        let end = ((t + 1) * EXACT_TILE_DIM).min(d);
        let shape = ModelShape {
            layout: options.layout,
            n,
            d: end - start,
            k,
        };
        let (q_tile, m_tile) = if tiles == 1 {
            (queries.to_vec(), memory.to_vec())
        } else {
            (tile_columns(queries, d, start, end), tile_columns(memory, d, start, end))
        };
        let (model, did_setup, secs) = model_for(&shape, &m_tile, &config)?;
        setup_performed |= did_setup;
        compile_s += secs;
        prepared.push(Tile { shape, model, q_tile });
    }
    let setup_s = if setup_performed {
        setup_started.elapsed().as_secs_f64()
    } else {
        0.0
    };
    let placement = prepared
        .first()
        .map(|t| t.model.placement.clone())
        .ok_or_else(|| AneError::CoreMl("no tiles".into()))?;

    // Per-job work: input conversion, prediction, readback.
    let run_all = |prepared: &[Tile]| -> Result<Vec<i32>, AneError> {
        let mut total = vec![0i32; n * k];
        for tile in prepared {
            let input = make_input(&tile.shape, &tile.q_tile)?;
            let out = {
                let _guard = tile.model.predict_lock.lock().unwrap_or_else(|p| p.into_inner());
                predict(&tile.model.model, &input)?
            };
            let scores = read_scores(&tile.shape, &out)?;
            for (acc, s) in total.iter_mut().zip(scores) {
                if s.fract() != 0.0 || !s.is_finite() {
                    return Err(AneError::CoreMl(format!("non-integer score {s} from Core ML")));
                }
                *acc += s as i32;
            }
        }
        Ok(total)
    };

    let started = Instant::now();
    let scores = run_all(&prepared)?;
    let predict_s = started.elapsed().as_secs_f64();

    let sample = match options.power {
        PowerSampling::Off => false,
        PowerSampling::OnNeuralEngine => placement.similarity_on_neural_engine(),
        PowerSampling::Always => true,
    };
    let power = if sample {
        power::measure(20, 10, || run_all(&prepared).map(|_| ()))?
    } else {
        None
    };

    let device = placement.similarity_device();
    Ok(HdcRun {
        scores,
        n,
        k,
        d,
        tiles,
        attempts: vec![(options.layout, device)],
        placement,
        compile_s,
        setup_s,
        setup_performed,
        predict_s,
        power,
    })
}
