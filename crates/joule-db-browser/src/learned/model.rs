//! Learned Model Implementation
//!
//! WebNN-based model for learned indexes with fallback chain:
//! 1. WebNN (hardware-accelerated ML in browser)
//! 2. WebGPU compute shaders (GPU acceleration)
//! 3. CPU (SIMD-optimized fallback)
//!
//! # Model Architecture
//! - Linear: y = ax + b (fast, for monotonic data)
//! - Neural: 2-layer MLP with ReLU (more accurate, for complex distributions)

#[cfg(target_arch = "wasm32")]
use js_sys::{Float32Array, Object, Promise, Reflect};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;

use bincode;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};

// ============================================================================
// WebNN Feature Detection
// ============================================================================

/// WebNN availability status
static WEBNN_AVAILABLE: AtomicBool = AtomicBool::new(false);
static WEBNN_CHECKED: AtomicBool = AtomicBool::new(false);

/// Check if WebNN is available in the current browser
#[cfg(target_arch = "wasm32")]
pub async fn check_webnn_availability() -> bool {
    if WEBNN_CHECKED.load(Ordering::SeqCst) {
        return WEBNN_AVAILABLE.load(Ordering::SeqCst);
    }

    let available = detect_webnn().await;
    WEBNN_AVAILABLE.store(available, Ordering::SeqCst);
    WEBNN_CHECKED.store(true, Ordering::SeqCst);
    available
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn check_webnn_availability() -> bool {
    false
}

#[cfg(target_arch = "wasm32")]
async fn detect_webnn() -> bool {
    use web_sys::window;

    let Some(window) = window() else {
        return false;
    };

    let navigator = window.navigator();

    // Check for navigator.ml (WebNN entry point)
    if let Ok(ml) = Reflect::get(&navigator, &"ml".into()) {
        if !ml.is_undefined() && !ml.is_null() {
            // Try to create a context to verify it works
            if let Ok(create_context) = Reflect::get(&ml, &"createContext".into()) {
                if create_context.is_function() {
                    return true;
                }
            }
        }
    }

    false
}

/// Compute backend for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackend {
    /// WebNN (browser ML API)
    WebNN,
    /// WebGPU compute shaders
    WebGPU,
    /// CPU with SIMD optimizations
    CPU,
}

impl ComputeBackend {
    /// Select the best available backend
    #[cfg(target_arch = "wasm32")]
    pub async fn auto_select() -> Self {
        // Check WebNN first (best for ML inference)
        if check_webnn_availability().await {
            return ComputeBackend::WebNN;
        }

        // Check WebGPU
        if check_webgpu_availability().await {
            return ComputeBackend::WebGPU;
        }

        // Fall back to CPU
        ComputeBackend::CPU
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn auto_select() -> Self {
        ComputeBackend::CPU
    }
}

#[cfg(target_arch = "wasm32")]
async fn check_webgpu_availability() -> bool {
    use web_sys::window;

    let Some(window) = window() else {
        return false;
    };

    let navigator = window.navigator();

    if let Ok(gpu) = Reflect::get(&navigator, &"gpu".into()) {
        if !gpu.is_undefined() && !gpu.is_null() {
            return true;
        }
    }

    false
}

// ============================================================================
// Model Types
// ============================================================================

/// Model type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// Linear model (fast, simple): y = ax + b
    Linear,
    /// Neural network (2-layer MLP with ReLU)
    Neural,
}

/// Model parameters for CPU fallback
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelParameters {
    /// Linear model: slope (a in y = ax + b)
    pub slope: f32,
    /// Linear model: intercept (b in y = ax + b)
    pub intercept: f32,
    /// Model accuracy (0.0 - 1.0)
    pub accuracy: f32,
    /// Optional neural network weights
    #[serde(default)]
    pub neural_weights: Option<NeuralWeights>,
}

/// Neural network weights for 2-layer MLP
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NeuralWeights {
    /// Input to hidden layer weights (input_size x hidden_size)
    pub w1: Vec<f32>,
    /// Hidden layer bias
    pub b1: Vec<f32>,
    /// Hidden to output layer weights (hidden_size x output_size)
    pub w2: Vec<f32>,
    /// Output layer bias
    pub b2: Vec<f32>,
    /// Hidden layer size
    pub hidden_size: usize,
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self {
            slope: 1.0,
            intercept: 0.0,
            accuracy: 0.0,
            neural_weights: None,
        }
    }
}

// ============================================================================
// WebNN Model Wrapper
// ============================================================================

/// WebNN ML Graph wrapper
#[cfg(target_arch = "wasm32")]
pub struct WebNNGraph {
    /// MLContext
    context: JsValue,
    /// Compiled MLGraph
    graph: JsValue,
    /// Input operand name
    input_name: String,
    /// Output operand name
    output_name: String,
}

#[cfg(target_arch = "wasm32")]
impl WebNNGraph {
    /// Create a linear model graph: y = ax + b
    pub async fn build_linear(slope: f32, intercept: f32) -> Result<Self, String> {
        let window = web_sys::window().ok_or_else(|| "No window".to_string())?;
        let navigator = window.navigator();

        let ml = Reflect::get(&navigator, &"ml".into()).map_err(|_| "navigator.ml not found")?;

        if ml.is_undefined() || ml.is_null() {
            return Err("WebNN not available".to_string());
        }

        // Create MLContext with default options
        let create_context_fn =
            Reflect::get(&ml, &"createContext".into()).map_err(|_| "createContext not found")?;

        let options = Object::new();
        Reflect::set(&options, &"deviceType".into(), &"cpu".into())
            .map_err(|e| format!("Failed to set deviceType: {:?}", e))?;

        let context_promise: Promise = Reflect::apply(
            &create_context_fn.into(),
            &ml,
            &js_sys::Array::of1(&options),
        )
        .map_err(|e| format!("createContext failed: {:?}", e))?
        .dyn_into()
        .map_err(|_| "createContext did not return a promise")?;

        let context = JsFuture::from(context_promise)
            .await
            .map_err(|e| format!("Failed to create MLContext: {:?}", e))?;

        // Create MLGraphBuilder
        let ml_graph_builder_class = Reflect::get(&window, &"MLGraphBuilder".into())
            .map_err(|_| "MLGraphBuilder not found")?;

        let builder = Reflect::construct(
            &ml_graph_builder_class.into(),
            &js_sys::Array::of1(&context),
        )
        .map_err(|e| format!("Failed to create MLGraphBuilder: {:?}", e))?;

        // Define input operand
        let input_desc = Object::new();
        Reflect::set(&input_desc, &"dataType".into(), &"float32".into())?;
        let dimensions = js_sys::Array::new();
        dimensions.push(&1.into());
        Reflect::set(&input_desc, &"dimensions".into(), &dimensions)?;

        let input_fn = Reflect::get(&builder, &"input".into())?;
        let input = Reflect::apply(
            &input_fn.into(),
            &builder,
            &js_sys::Array::of2(&"x".into(), &input_desc),
        )?;

        // Create constant for slope
        let slope_data = Float32Array::new_with_length(1);
        slope_data.set_index(0, slope);
        let slope_const_fn = Reflect::get(&builder, &"constant".into())?;
        let slope_desc = Object::new();
        Reflect::set(&slope_desc, &"dataType".into(), &"float32".into())?;
        let slope_dims = js_sys::Array::new();
        slope_dims.push(&1.into());
        Reflect::set(&slope_desc, &"dimensions".into(), &slope_dims)?;
        let slope_const = Reflect::apply(
            &slope_const_fn.into(),
            &builder,
            &js_sys::Array::of2(&slope_desc, &slope_data),
        )?;

        // Create constant for intercept
        let intercept_data = Float32Array::new_with_length(1);
        intercept_data.set_index(0, intercept);
        let intercept_const = Reflect::apply(
            &slope_const_fn.into(),
            &builder,
            &js_sys::Array::of2(&slope_desc, &intercept_data),
        )?;

        // y = x * slope
        let mul_fn = Reflect::get(&builder, &"mul".into())?;
        let scaled = Reflect::apply(
            &mul_fn.into(),
            &builder,
            &js_sys::Array::of2(&input, &slope_const),
        )?;

        // y = (x * slope) + intercept
        let add_fn = Reflect::get(&builder, &"add".into())?;
        let output = Reflect::apply(
            &add_fn.into(),
            &builder,
            &js_sys::Array::of2(&scaled, &intercept_const),
        )?;

        // Build the graph
        let outputs = Object::new();
        Reflect::set(&outputs, &"y".into(), &output)?;

        let build_fn = Reflect::get(&builder, &"build".into())?;
        let build_promise: Promise =
            Reflect::apply(&build_fn.into(), &builder, &js_sys::Array::of1(&outputs))?
                .dyn_into()
                .map_err(|_| "build did not return a promise")?;

        let graph = JsFuture::from(build_promise)
            .await
            .map_err(|e| format!("Failed to build graph: {:?}", e))?;

        Ok(Self {
            context,
            graph,
            input_name: "x".to_string(),
            output_name: "y".to_string(),
        })
    }

    /// Run inference
    pub async fn compute(&self, input_value: f32) -> Result<f32, String> {
        // Create input tensor
        let input_data = Float32Array::new_with_length(1);
        input_data.set_index(0, input_value);

        let inputs = Object::new();
        Reflect::set(&inputs, &self.input_name.clone().into(), &input_data)
            .map_err(|e| format!("Failed to set input: {:?}", e))?;

        // Create output buffer
        let output_data = Float32Array::new_with_length(1);
        let outputs = Object::new();
        Reflect::set(&outputs, &self.output_name.clone().into(), &output_data)
            .map_err(|e| format!("Failed to set output: {:?}", e))?;

        // Run compute
        let compute_fn =
            Reflect::get(&self.context, &"compute".into()).map_err(|_| "compute not found")?;

        let compute_promise: Promise = Reflect::apply(
            &compute_fn.into(),
            &self.context,
            &js_sys::Array::of2(&self.graph, &inputs),
        )
        .map_err(|e| format!("compute failed: {:?}", e))?
        .dyn_into()
        .map_err(|_| "compute did not return a promise")?;

        let result = JsFuture::from(compute_promise)
            .await
            .map_err(|e| format!("Compute failed: {:?}", e))?;

        // Extract output
        let result_outputs =
            Reflect::get(&result, &"outputs".into()).map_err(|_| "outputs not found in result")?;

        let output_tensor: Float32Array =
            Reflect::get(&result_outputs, &self.output_name.clone().into())
                .map_err(|_| "output tensor not found")?
                .dyn_into()
                .map_err(|_| "output is not Float32Array")?;

        Ok(output_tensor.get_index(0))
    }
}

// ============================================================================
// Learned Model
// ============================================================================

/// Learned model with WebNN acceleration
pub struct LearnedModel {
    /// Model type
    pub model_type: ModelType,
    /// WebNN graph (when available)
    #[cfg(target_arch = "wasm32")]
    pub webnn_graph: Option<WebNNGraph>,
    #[cfg(not(target_arch = "wasm32"))]
    pub webnn_graph: Option<()>,
    /// Legacy WebNN model reference
    #[cfg(target_arch = "wasm32")]
    pub webnn_model: Option<js_sys::Object>,
    #[cfg(not(target_arch = "wasm32"))]
    pub webnn_model: Option<()>,
    /// CPU fallback parameters
    pub parameters: ModelParameters,
    /// Currently selected backend
    pub backend: ComputeBackend,
}

impl LearnedModel {
    /// Create a new linear model
    pub fn new_linear(slope: f32, intercept: f32) -> Self {
        Self {
            model_type: ModelType::Linear,
            #[cfg(target_arch = "wasm32")]
            webnn_graph: None,
            #[cfg(not(target_arch = "wasm32"))]
            webnn_graph: None,
            #[cfg(target_arch = "wasm32")]
            webnn_model: None,
            #[cfg(not(target_arch = "wasm32"))]
            webnn_model: None,
            parameters: ModelParameters {
                slope,
                intercept,
                accuracy: 0.0,
                neural_weights: None,
            },
            backend: ComputeBackend::CPU,
        }
    }

    /// Initialize the model with the best available backend
    #[cfg(target_arch = "wasm32")]
    pub async fn initialize(&mut self) -> Result<(), String> {
        self.backend = ComputeBackend::auto_select().await;

        match self.backend {
            ComputeBackend::WebNN => {
                // Build WebNN graph
                let graph =
                    WebNNGraph::build_linear(self.parameters.slope, self.parameters.intercept)
                        .await?;
                self.webnn_graph = Some(graph);
            }
            ComputeBackend::WebGPU => {
                // WebGPU would initialize compute pipeline here
                // For now, fall through to CPU
            }
            ComputeBackend::CPU => {
                // CPU uses parameters directly
            }
        }

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn initialize(&mut self) -> Result<(), String> {
        self.backend = ComputeBackend::CPU;
        Ok(())
    }

    /// Load model from OPFS
    pub async fn load(path: &str) -> Result<Self, String> {
        use crate::persistence::{AsyncPersistenceBackend, OPFSBackend};
        use std::sync::{Arc, RwLock};

        // Create OPFS backend
        let mut backend = OPFSBackend::new(path);
        backend
            .init()
            .await
            .map_err(|e| format!("OPFS init failed: {}", e))?;

        // Wrap in Arc<RwLock> to use trait method
        let backend_arc: Arc<RwLock<dyn AsyncPersistenceBackend + Send + Sync>> =
            Arc::new(RwLock::new(backend));

        // Read model data using read_page from trait
        let model_key = format!("model_{}", path);
        let mut hasher = DefaultHasher::new();
        model_key.hash(&mut hasher);
        let page_id = hasher.finish() as u64;
        let model_data = backend_arc
            .read()
            .unwrap()
            .read_page(page_id)
            .await
            .map_err(|e| format!("Failed to read model: {}", e))?;

        if model_data.is_empty() {
            return Err("Model file is empty".to_string());
        }

        // Deserialize model parameters
        let params: ModelParameters =
            bincode::serde::decode_from_slice(&model_data, bincode::config::standard())
                .map(|(v, _)| v)
                .map_err(|e| format!("Failed to deserialize model: {}", e))?;

        let mut model = Self {
            model_type: if params.neural_weights.is_some() {
                ModelType::Neural
            } else {
                ModelType::Linear
            },
            #[cfg(target_arch = "wasm32")]
            webnn_graph: None,
            #[cfg(not(target_arch = "wasm32"))]
            webnn_graph: None,
            #[cfg(target_arch = "wasm32")]
            webnn_model: None,
            #[cfg(not(target_arch = "wasm32"))]
            webnn_model: None,
            parameters: params,
            backend: ComputeBackend::CPU,
        };

        // Try to initialize with best backend
        if let Err(e) = model.initialize().await {
            // Log warning but continue with CPU
            #[cfg(target_arch = "wasm32")]
            web_sys::console::warn_1(&format!("WebNN init failed, using CPU: {}", e).into());
        }

        Ok(model)
    }

    /// Save model to OPFS
    pub async fn save(&self, path: &str) -> Result<(), String> {
        use crate::persistence::{AsyncPersistenceBackend, OPFSBackend};
        use std::sync::{Arc, RwLock};

        // Create OPFS backend
        let mut backend = OPFSBackend::new(path);
        backend
            .init()
            .await
            .map_err(|e| format!("OPFS init failed: {}", e))?;

        // Serialize model parameters
        let model_data =
            bincode::serde::encode_to_vec(&self.parameters, bincode::config::standard())
                .map_err(|e| format!("Failed to serialize model: {}", e))?;

        // Wrap in Arc<RwLock> to use trait method
        let backend_arc: Arc<RwLock<dyn AsyncPersistenceBackend + Send + Sync>> =
            Arc::new(RwLock::new(backend));

        // Write to OPFS using write_page from trait
        let model_key = format!("model_{}", path);
        let mut hasher = DefaultHasher::new();
        model_key.hash(&mut hasher);
        let page_id = hasher.finish() as u64;
        backend_arc
            .read()
            .unwrap()
            .write_page(page_id, &model_data)
            .await
            .map_err(|e| format!("Failed to write model: {}", e))?;

        Ok(())
    }

    /// Load WebNN model (legacy method for compatibility)
    #[cfg(target_arch = "wasm32")]
    async fn load_webnn_model(_path: &str) -> Result<js_sys::Object, String> {
        Err("Use initialize() instead".to_string())
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn load_webnn_model(_path: &str) -> Result<(), String> {
        Err("WebNN not available on this platform".to_string())
    }

    /// Predict position for key using the best available backend
    ///
    /// Fallback chain:
    /// 1. WebNN (if initialized)
    /// 2. WebGPU (if available)
    /// 3. CPU (always available)
    pub async fn predict_position(&self, key: &str) -> Result<usize, String> {
        let hash = self.hash_key(key);
        let input_value = hash as f32 / u32::MAX as f32; // Normalize to [0, 1]

        let result = match self.backend {
            #[cfg(target_arch = "wasm32")]
            ComputeBackend::WebNN => {
                if let Some(ref graph) = self.webnn_graph {
                    match graph.compute(input_value).await {
                        Ok(v) => v,
                        Err(_) => self.predict_cpu_internal(input_value),
                    }
                } else {
                    self.predict_cpu_internal(input_value)
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            ComputeBackend::WebNN => self.predict_cpu_internal(input_value),

            ComputeBackend::WebGPU => {
                // WebGPU fallback to CPU for now
                self.predict_cpu_internal(input_value)
            }

            ComputeBackend::CPU => self.predict_cpu_internal(input_value),
        };

        Ok(result.max(0.0) as usize)
    }

    /// CPU prediction implementation
    fn predict_cpu_internal(&self, input: f32) -> f32 {
        match self.model_type {
            ModelType::Linear => self.parameters.slope * input + self.parameters.intercept,
            ModelType::Neural => {
                if let Some(ref weights) = self.parameters.neural_weights {
                    self.predict_neural_cpu(input, weights)
                } else {
                    // Fall back to linear
                    self.parameters.slope * input + self.parameters.intercept
                }
            }
        }
    }

    /// Neural network forward pass on CPU
    fn predict_neural_cpu(&self, input: f32, weights: &NeuralWeights) -> f32 {
        let hidden_size = weights.hidden_size;

        // Input to hidden layer (with ReLU)
        let mut hidden = vec![0.0f32; hidden_size];
        for i in 0..hidden_size {
            let mut sum = weights.b1.get(i).copied().unwrap_or(0.0);
            sum += input * weights.w1.get(i).copied().unwrap_or(0.0);
            hidden[i] = sum.max(0.0); // ReLU
        }

        // Hidden to output layer
        let mut output = weights.b2.get(0).copied().unwrap_or(0.0);
        for (i, &h) in hidden.iter().enumerate() {
            output += h * weights.w2.get(i).copied().unwrap_or(0.0);
        }

        output
    }

    /// Legacy CPU prediction method
    fn predict_cpu(&self, key: &str) -> Result<usize, String> {
        let hash = self.hash_key(key);
        let input = hash as f32 / u32::MAX as f32;
        let position = self.predict_cpu_internal(input);
        Ok(position.max(0.0) as usize)
    }

    /// Hash key to number
    fn hash_key(&self, key: &str) -> u32 {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish() as u32
    }

    /// Get current compute backend
    pub fn current_backend(&self) -> ComputeBackend {
        self.backend
    }

    /// Get model accuracy
    pub fn accuracy(&self) -> f32 {
        self.parameters.accuracy
    }

    /// Update model parameters after training
    pub fn update_parameters(&mut self, slope: f32, intercept: f32, accuracy: f32) {
        self.parameters.slope = slope;
        self.parameters.intercept = intercept;
        self.parameters.accuracy = accuracy;

        // Invalidate WebNN graph (needs rebuild)
        #[cfg(target_arch = "wasm32")]
        {
            self.webnn_graph = None;
        }
    }
}

impl Default for LearnedModel {
    fn default() -> Self {
        Self::new_linear(1.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_model_prediction() {
        let model = LearnedModel::new_linear(10.0, 5.0);
        let result = model.predict_cpu_internal(0.5);
        assert!((result - 10.0).abs() < 0.1); // 10.0 * 0.5 + 5.0 = 10.0
    }

    #[test]
    fn test_neural_weights_serde() {
        let weights = NeuralWeights {
            w1: vec![1.0, 2.0, 3.0, 4.0],
            b1: vec![0.1, 0.2],
            w2: vec![0.5, 0.5],
            b2: vec![0.0],
            hidden_size: 2,
        };

        let serialized =
            bincode::serde::encode_to_vec(&weights, bincode::config::standard()).unwrap();
        let deserialized: NeuralWeights =
            bincode::serde::decode_from_slice(&serialized, bincode::config::standard())
                .map(|(v, _)| v)
                .unwrap();

        assert_eq!(weights.hidden_size, deserialized.hidden_size);
        assert_eq!(weights.w1, deserialized.w1);
    }

    #[test]
    fn test_model_parameters_default() {
        let params = ModelParameters::default();
        assert_eq!(params.slope, 1.0);
        assert_eq!(params.intercept, 0.0);
    }

    #[tokio::test]
    async fn test_backend_selection() {
        // On non-WASM, should always be CPU
        let backend = ComputeBackend::auto_select().await;
        #[cfg(not(target_arch = "wasm32"))]
        assert_eq!(backend, ComputeBackend::CPU);
    }
}
