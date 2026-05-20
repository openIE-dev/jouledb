//! WebGPU abstraction model — device, buffers, textures, pipelines, and commands.
//!
//! Pure Rust model of the WebGPU API for command recording, validation, and
//! offline pipeline configuration. No actual GPU calls — this is a command
//! buffer that can be serialized or replayed on a real WebGPU backend.

use std::fmt;
use std::collections::HashMap;

// ── Texture Formats ──────────────────────────────────────────

/// GPU texture format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    Rgba8Unorm,
    Rgba8Snorm,
    Rgba16Float,
    Rgba32Float,
    Bgra8Unorm,
    Depth24Plus,
    Depth24PlusStencil8,
    Depth32Float,
    R8Unorm,
    Rg8Unorm,
}

impl TextureFormat {
    /// Bytes per pixel for this format.
    pub fn bytes_per_pixel(&self) -> u32 {
        match self {
            Self::R8Unorm => 1,
            Self::Rg8Unorm => 2,
            Self::Rgba8Unorm | Self::Rgba8Snorm | Self::Bgra8Unorm => 4,
            Self::Depth24Plus | Self::Depth24PlusStencil8 | Self::Depth32Float => 4,
            Self::Rgba16Float => 8,
            Self::Rgba32Float => 16,
        }
    }

    /// Whether this format has a depth component.
    pub fn is_depth(&self) -> bool {
        matches!(self, Self::Depth24Plus | Self::Depth24PlusStencil8 | Self::Depth32Float)
    }
}

/// Simple bitflags macro since we can't use external crates.
macro_rules! bitflags_manual {
    ($(#[$meta:meta])* $Name:ident : $T:ty { $($FLAG:ident = $val:expr),* $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $Name(pub $T);

        impl $Name {
            $(pub const $FLAG: Self = Self($val);)*

            pub fn contains(self, other: Self) -> bool {
                (self.0 & other.0) == other.0
            }

            pub fn is_empty(self) -> bool {
                self.0 == 0
            }
        }

        impl std::ops::BitOr for $Name {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }
        }

        impl std::ops::BitAnd for $Name {
            type Output = Self;
            fn bitand(self, rhs: Self) -> Self {
                Self(self.0 & rhs.0)
            }
        }
    };
}
use bitflags_manual;

// ── Buffer Usage ─────────────────────────────────────────────

bitflags_manual! {
    /// GPU buffer usage flags.
    BufferUsage: u32 {
        VERTEX = 0x0001,
        INDEX = 0x0002,
        UNIFORM = 0x0004,
        STORAGE = 0x0008,
        COPY_SRC = 0x0010,
        COPY_DST = 0x0020,
        INDIRECT = 0x0040,
        MAP_READ = 0x0080,
        MAP_WRITE = 0x0100,
    }
}

bitflags_manual! {
    /// GPU texture usage flags.
    TextureUsage: u32 {
        COPY_SRC = 0x01,
        COPY_DST = 0x02,
        TEXTURE_BINDING = 0x04,
        STORAGE_BINDING = 0x08,
        RENDER_ATTACHMENT = 0x10,
    }
}

// ── Device Limits ────────────────────────────────────────────

/// GPU device limits.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceLimits {
    pub max_texture_dimension_2d: u32,
    pub max_buffer_size: u64,
    pub max_bind_groups: u32,
    pub max_uniforms_per_stage: u32,
    pub max_storage_buffers_per_stage: u32,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
}

impl Default for DeviceLimits {
    fn default() -> Self {
        Self {
            max_texture_dimension_2d: 8192,
            max_buffer_size: 256 * 1024 * 1024,
            max_bind_groups: 4,
            max_uniforms_per_stage: 12,
            max_storage_buffers_per_stage: 8,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
        }
    }
}

/// GPU device feature flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceFeature {
    TextureCompressionBc,
    TextureCompressionEtc2,
    TextureCompressionAstc,
    TimestampQuery,
    IndirectFirstInstance,
    Float32Filterable,
    DepthClipControl,
}

// ── GPUDevice ────────────────────────────────────────────────

/// Modeled GPU device with limits and features.
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub label: String,
    pub limits: DeviceLimits,
    pub features: Vec<DeviceFeature>,
    next_id: u64,
}

impl GpuDevice {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            limits: DeviceLimits::default(),
            features: Vec::new(),
            next_id: 1,
        }
    }

    pub fn with_limits(mut self, limits: DeviceLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn with_features(mut self, features: Vec<DeviceFeature>) -> Self {
        self.features = features;
        self
    }

    pub fn has_feature(&self, feature: DeviceFeature) -> bool {
        self.features.contains(&feature)
    }

    fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Create a GPU buffer.
    pub fn create_buffer(&mut self, desc: &BufferDescriptor) -> Result<GpuBuffer, GpuError> {
        if desc.size == 0 {
            return Err(GpuError::InvalidSize("buffer size must be > 0".into()));
        }
        if desc.size > self.limits.max_buffer_size {
            return Err(GpuError::ExceedsLimit(format!(
                "buffer size {} exceeds max {}",
                desc.size, self.limits.max_buffer_size
            )));
        }
        Ok(GpuBuffer {
            id: self.next_id(),
            label: desc.label.clone(),
            size: desc.size,
            usage: desc.usage,
            mapped: false,
        })
    }

    /// Create a GPU texture.
    pub fn create_texture(&mut self, desc: &TextureDescriptor) -> Result<GpuTexture, GpuError> {
        if desc.width == 0 || desc.height == 0 {
            return Err(GpuError::InvalidSize("texture dimensions must be > 0".into()));
        }
        let max = self.limits.max_texture_dimension_2d;
        if desc.width > max || desc.height > max {
            return Err(GpuError::ExceedsLimit(format!(
                "texture {}x{} exceeds max dimension {}",
                desc.width, desc.height, max
            )));
        }
        Ok(GpuTexture {
            id: self.next_id(),
            label: desc.label.clone(),
            width: desc.width,
            height: desc.height,
            depth: desc.depth.unwrap_or(1),
            format: desc.format,
            usage: desc.usage,
            mip_level_count: desc.mip_level_count.unwrap_or(1),
        })
    }

    /// Create a render pipeline.
    pub fn create_render_pipeline(&mut self, desc: &RenderPipelineDescriptor) -> RenderPipeline {
        RenderPipeline {
            id: self.next_id(),
            label: desc.label.clone(),
            vertex_shader: desc.vertex_shader.clone(),
            fragment_shader: desc.fragment_shader.clone(),
            primitive_topology: desc.primitive_topology,
            depth_stencil: desc.depth_stencil,
            vertex_buffers: desc.vertex_buffers.clone(),
        }
    }

    /// Create a compute pipeline.
    pub fn create_compute_pipeline(&mut self, desc: &ComputePipelineDescriptor) -> ComputePipeline {
        ComputePipeline {
            id: self.next_id(),
            label: desc.label.clone(),
            shader: desc.shader.clone(),
            entry_point: desc.entry_point.clone(),
        }
    }

    /// Create a bind group.
    pub fn create_bind_group(&mut self, desc: &BindGroupDescriptor) -> Result<BindGroup, GpuError> {
        if desc.entries.is_empty() {
            return Err(GpuError::InvalidBindGroup("bind group must have at least one entry".into()));
        }
        Ok(BindGroup {
            id: self.next_id(),
            label: desc.label.clone(),
            entries: desc.entries.clone(),
        })
    }

    /// Create a command encoder.
    pub fn create_command_encoder(&mut self, label: impl Into<String>) -> CommandEncoder {
        CommandEncoder {
            id: self.next_id(),
            label: label.into(),
            commands: Vec::new(),
        }
    }
}

// ── Buffer ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BufferDescriptor {
    pub label: String,
    pub size: u64,
    pub usage: BufferUsage,
}

#[derive(Debug, Clone)]
pub struct GpuBuffer {
    pub id: u64,
    pub label: String,
    pub size: u64,
    pub usage: BufferUsage,
    pub mapped: bool,
}

// ── Texture ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TextureDescriptor {
    pub label: String,
    pub width: u32,
    pub height: u32,
    pub depth: Option<u32>,
    pub format: TextureFormat,
    pub usage: TextureUsage,
    pub mip_level_count: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct GpuTexture {
    pub id: u64,
    pub label: String,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub format: TextureFormat,
    pub usage: TextureUsage,
    pub mip_level_count: u32,
}

impl GpuTexture {
    /// Calculate total size in bytes.
    pub fn size_bytes(&self) -> u64 {
        self.width as u64 * self.height as u64 * self.depth as u64
            * self.format.bytes_per_pixel() as u64
            * self.mip_level_count as u64
    }
}

// ── Primitives ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareFunction {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthStencilState {
    pub format: TextureFormat,
    pub depth_write_enabled: bool,
    pub depth_compare: CompareFunction,
}

// ── Vertex Buffer Layout ─────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct VertexBufferLayout {
    pub array_stride: u64,
    pub step_mode: VertexStepMode,
    pub attributes: Vec<VertexAttribute>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexStepMode {
    Vertex,
    Instance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexAttribute {
    pub format: VertexFormat,
    pub offset: u64,
    pub shader_location: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexFormat {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Uint32,
    Sint32,
}

impl VertexFormat {
    pub fn size(&self) -> u64 {
        match self {
            Self::Float32 | Self::Uint32 | Self::Sint32 => 4,
            Self::Float32x2 => 8,
            Self::Float32x3 => 12,
            Self::Float32x4 => 16,
        }
    }
}

// ── Pipelines ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RenderPipelineDescriptor {
    pub label: String,
    pub vertex_shader: String,
    pub fragment_shader: String,
    pub primitive_topology: PrimitiveTopology,
    pub depth_stencil: Option<DepthStencilState>,
    pub vertex_buffers: Vec<VertexBufferLayout>,
}

#[derive(Debug, Clone)]
pub struct RenderPipeline {
    pub id: u64,
    pub label: String,
    pub vertex_shader: String,
    pub fragment_shader: String,
    pub primitive_topology: PrimitiveTopology,
    pub depth_stencil: Option<DepthStencilState>,
    pub vertex_buffers: Vec<VertexBufferLayout>,
}

#[derive(Debug, Clone)]
pub struct ComputePipelineDescriptor {
    pub label: String,
    pub shader: String,
    pub entry_point: String,
}

#[derive(Debug, Clone)]
pub struct ComputePipeline {
    pub id: u64,
    pub label: String,
    pub shader: String,
    pub entry_point: String,
}

// ── Bind Group ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BindGroupDescriptor {
    pub label: String,
    pub entries: Vec<BindGroupEntry>,
}

#[derive(Debug, Clone)]
pub struct BindGroupEntry {
    pub binding: u32,
    pub resource: BindingResource,
}

#[derive(Debug, Clone)]
pub enum BindingResource {
    Buffer { buffer_id: u64, offset: u64, size: u64 },
    Sampler { sampler_id: u64 },
    TextureView { texture_id: u64 },
}

#[derive(Debug, Clone)]
pub struct BindGroup {
    pub id: u64,
    pub label: String,
    pub entries: Vec<BindGroupEntry>,
}

// ── Command Encoder ──────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CommandEncoder {
    pub id: u64,
    pub label: String,
    pub commands: Vec<GpuCommand>,
}

#[derive(Debug, Clone)]
pub enum GpuCommand {
    BeginRenderPass {
        color_attachments: Vec<ColorAttachment>,
        depth_attachment: Option<DepthAttachment>,
    },
    EndRenderPass,
    SetPipeline { pipeline_id: u64 },
    SetBindGroup { index: u32, bind_group_id: u64 },
    SetVertexBuffer { slot: u32, buffer_id: u64 },
    SetIndexBuffer { buffer_id: u64, format: IndexFormat },
    Draw { vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32 },
    DrawIndexed { index_count: u32, instance_count: u32, first_index: u32, base_vertex: i32, first_instance: u32 },
    BeginComputePass,
    EndComputePass,
    Dispatch { x: u32, y: u32, z: u32 },
    CopyBufferToBuffer { src: u64, src_offset: u64, dst: u64, dst_offset: u64, size: u64 },
    CopyTextureToTexture { src: u64, dst: u64 },
}

#[derive(Debug, Clone)]
pub struct ColorAttachment {
    pub texture_id: u64,
    pub clear_color: [f64; 4],
    pub load_op: LoadOp,
    pub store_op: StoreOp,
}

#[derive(Debug, Clone)]
pub struct DepthAttachment {
    pub texture_id: u64,
    pub depth_clear_value: f32,
    pub depth_load_op: LoadOp,
    pub depth_store_op: StoreOp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadOp {
    Clear,
    Load,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreOp {
    Store,
    Discard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexFormat {
    Uint16,
    Uint32,
}

impl CommandEncoder {
    pub fn begin_render_pass(&mut self, color_attachments: Vec<ColorAttachment>, depth: Option<DepthAttachment>) {
        self.commands.push(GpuCommand::BeginRenderPass {
            color_attachments,
            depth_attachment: depth,
        });
    }

    pub fn end_render_pass(&mut self) {
        self.commands.push(GpuCommand::EndRenderPass);
    }

    pub fn set_pipeline(&mut self, pipeline_id: u64) {
        self.commands.push(GpuCommand::SetPipeline { pipeline_id });
    }

    pub fn set_bind_group(&mut self, index: u32, bind_group_id: u64) {
        self.commands.push(GpuCommand::SetBindGroup { index, bind_group_id });
    }

    pub fn set_vertex_buffer(&mut self, slot: u32, buffer_id: u64) {
        self.commands.push(GpuCommand::SetVertexBuffer { slot, buffer_id });
    }

    pub fn set_index_buffer(&mut self, buffer_id: u64, format: IndexFormat) {
        self.commands.push(GpuCommand::SetIndexBuffer { buffer_id, format });
    }

    pub fn draw(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) {
        self.commands.push(GpuCommand::Draw { vertex_count, instance_count, first_vertex, first_instance });
    }

    pub fn draw_indexed(&mut self, index_count: u32, instance_count: u32, first_index: u32, base_vertex: i32, first_instance: u32) {
        self.commands.push(GpuCommand::DrawIndexed { index_count, instance_count, first_index, base_vertex, first_instance });
    }

    pub fn begin_compute_pass(&mut self) {
        self.commands.push(GpuCommand::BeginComputePass);
    }

    pub fn end_compute_pass(&mut self) {
        self.commands.push(GpuCommand::EndComputePass);
    }

    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        self.commands.push(GpuCommand::Dispatch { x, y, z });
    }

    pub fn copy_buffer_to_buffer(&mut self, src: u64, src_offset: u64, dst: u64, dst_offset: u64, size: u64) {
        self.commands.push(GpuCommand::CopyBufferToBuffer { src, src_offset, dst, dst_offset, size });
    }

    /// Finish encoding and return commands.
    pub fn finish(self) -> Vec<GpuCommand> {
        self.commands
    }

    /// Number of recorded commands.
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }
}

// ── Errors ───────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum GpuError {
    InvalidSize(String),
    ExceedsLimit(String),
    InvalidBindGroup(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSize(msg) => write!(f, "invalid size: {msg}"),
            Self::ExceedsLimit(msg) => write!(f, "exceeds limit: {msg}"),
            Self::InvalidBindGroup(msg) => write!(f, "invalid bind group: {msg}"),
        }
    }
}

impl std::error::Error for GpuError {}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_device() {
        let device = GpuDevice::new("test-device");
        assert_eq!(device.label, "test-device");
        assert_eq!(device.limits.max_texture_dimension_2d, 8192);
    }

    #[test]
    fn create_buffer() {
        let mut device = GpuDevice::new("test");
        let buf = device.create_buffer(&BufferDescriptor {
            label: "vertex-buf".into(),
            size: 1024,
            usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
        }).unwrap();
        assert_eq!(buf.size, 1024);
        assert!(buf.usage.contains(BufferUsage::VERTEX));
        assert!(buf.usage.contains(BufferUsage::COPY_DST));
    }

    #[test]
    fn create_buffer_zero_size() {
        let mut device = GpuDevice::new("test");
        let result = device.create_buffer(&BufferDescriptor {
            label: "empty".into(),
            size: 0,
            usage: BufferUsage::VERTEX,
        });
        assert!(result.is_err());
    }

    #[test]
    fn create_buffer_exceeds_limit() {
        let mut device = GpuDevice::new("test");
        device.limits.max_buffer_size = 100;
        let result = device.create_buffer(&BufferDescriptor {
            label: "big".into(),
            size: 200,
            usage: BufferUsage::STORAGE,
        });
        assert!(result.is_err());
    }

    #[test]
    fn create_texture() {
        let mut device = GpuDevice::new("test");
        let tex = device.create_texture(&TextureDescriptor {
            label: "albedo".into(),
            width: 256,
            height: 256,
            depth: None,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
            mip_level_count: None,
        }).unwrap();
        assert_eq!(tex.width, 256);
        assert_eq!(tex.height, 256);
        assert_eq!(tex.size_bytes(), 256 * 256 * 4);
    }

    #[test]
    fn create_texture_exceeds_limit() {
        let mut device = GpuDevice::new("test");
        device.limits.max_texture_dimension_2d = 512;
        let result = device.create_texture(&TextureDescriptor {
            label: "huge".into(),
            width: 1024,
            height: 1024,
            depth: None,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsage::TEXTURE_BINDING,
            mip_level_count: None,
        });
        assert!(result.is_err());
    }

    #[test]
    fn create_render_pipeline() {
        let mut device = GpuDevice::new("test");
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: "main".into(),
            vertex_shader: "vs_main".into(),
            fragment_shader: "fs_main".into(),
            primitive_topology: PrimitiveTopology::TriangleList,
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth24Plus,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
            }),
            vertex_buffers: vec![],
        });
        assert_eq!(pipeline.primitive_topology, PrimitiveTopology::TriangleList);
        assert!(pipeline.depth_stencil.is_some());
    }

    #[test]
    fn create_compute_pipeline() {
        let mut device = GpuDevice::new("test");
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: "compute".into(),
            shader: "cs_main".into(),
            entry_point: "main".into(),
        });
        assert_eq!(pipeline.entry_point, "main");
    }

    #[test]
    fn create_bind_group() {
        let mut device = GpuDevice::new("test");
        let bg = device.create_bind_group(&BindGroupDescriptor {
            label: "bg0".into(),
            entries: vec![
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer { buffer_id: 1, offset: 0, size: 64 },
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView { texture_id: 2 },
                },
            ],
        }).unwrap();
        assert_eq!(bg.entries.len(), 2);
    }

    #[test]
    fn create_bind_group_empty() {
        let mut device = GpuDevice::new("test");
        let result = device.create_bind_group(&BindGroupDescriptor {
            label: "empty".into(),
            entries: vec![],
        });
        assert!(result.is_err());
    }

    #[test]
    fn command_encoder_render_pass() {
        let mut device = GpuDevice::new("test");
        let mut encoder = device.create_command_encoder("main");

        encoder.begin_render_pass(
            vec![ColorAttachment {
                texture_id: 1,
                clear_color: [0.0, 0.0, 0.0, 1.0],
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
            }],
            None,
        );
        encoder.set_pipeline(1);
        encoder.set_vertex_buffer(0, 2);
        encoder.draw(36, 1, 0, 0);
        encoder.end_render_pass();

        assert_eq!(encoder.command_count(), 5);
        let cmds = encoder.finish();
        assert!(matches!(cmds[0], GpuCommand::BeginRenderPass { .. }));
        assert!(matches!(cmds[4], GpuCommand::EndRenderPass));
    }

    #[test]
    fn command_encoder_compute_pass() {
        let mut device = GpuDevice::new("test");
        let mut encoder = device.create_command_encoder("compute");
        encoder.begin_compute_pass();
        encoder.set_pipeline(5);
        encoder.set_bind_group(0, 3);
        encoder.dispatch(64, 64, 1);
        encoder.end_compute_pass();

        assert_eq!(encoder.command_count(), 5);
    }

    #[test]
    fn texture_format_depth() {
        assert!(TextureFormat::Depth24Plus.is_depth());
        assert!(TextureFormat::Depth32Float.is_depth());
        assert!(!TextureFormat::Rgba8Unorm.is_depth());
    }

    #[test]
    fn vertex_format_sizes() {
        assert_eq!(VertexFormat::Float32.size(), 4);
        assert_eq!(VertexFormat::Float32x3.size(), 12);
        assert_eq!(VertexFormat::Float32x4.size(), 16);
    }

    #[test]
    fn buffer_usage_flags() {
        let usage = BufferUsage::VERTEX | BufferUsage::INDEX;
        assert!(usage.contains(BufferUsage::VERTEX));
        assert!(usage.contains(BufferUsage::INDEX));
        assert!(!usage.contains(BufferUsage::UNIFORM));
    }

    #[test]
    fn device_features() {
        let device = GpuDevice::new("test")
            .with_features(vec![DeviceFeature::TimestampQuery, DeviceFeature::Float32Filterable]);
        assert!(device.has_feature(DeviceFeature::TimestampQuery));
        assert!(!device.has_feature(DeviceFeature::TextureCompressionBc));
    }

    #[test]
    fn copy_buffer_command() {
        let mut device = GpuDevice::new("test");
        let mut encoder = device.create_command_encoder("copy");
        encoder.copy_buffer_to_buffer(1, 0, 2, 0, 512);
        let cmds = encoder.finish();
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            GpuCommand::CopyBufferToBuffer { src, dst, size, .. } => {
                assert_eq!(*src, 1);
                assert_eq!(*dst, 2);
                assert_eq!(*size, 512);
            }
            _ => panic!("expected CopyBufferToBuffer"),
        }
    }
}
