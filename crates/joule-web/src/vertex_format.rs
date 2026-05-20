//! Vertex attribute format system — define vertex layouts: position, normal,
//! UV, color, tangent, bone weights, custom attributes. Stride and offset
//! calculation. Interleaved vs separate attribute arrays. Vertex format
//! hashing for pipeline compatibility. Format conversion.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

// ── Attribute Types ──────────────────────────────────────────

/// Data type of a vertex attribute component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComponentType {
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Uint8,
    Uint16,
    Uint32,
    Unorm8,
    Unorm16,
}

impl ComponentType {
    pub fn byte_size(self) -> usize {
        match self {
            ComponentType::Float32 => 4,
            ComponentType::Float64 => 8,
            ComponentType::Int8 | ComponentType::Uint8 | ComponentType::Unorm8 => 1,
            ComponentType::Int16 | ComponentType::Uint16 | ComponentType::Unorm16 => 2,
            ComponentType::Int32 | ComponentType::Uint32 => 4,
        }
    }

    pub fn is_normalized(self) -> bool {
        matches!(self, ComponentType::Unorm8 | ComponentType::Unorm16)
    }
}

/// Well-known vertex attribute semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttributeSemantic {
    Position,
    Normal,
    Tangent,
    Bitangent,
    TexCoord(u8),
    Color(u8),
    BoneWeights,
    BoneIndices,
    Custom(u16),
}

// ── VertexAttribute ──────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VertexAttribute {
    pub semantic: AttributeSemantic,
    pub component_type: ComponentType,
    pub component_count: u8,
    pub normalized: bool,
}

impl VertexAttribute {
    pub fn new(semantic: AttributeSemantic, component_type: ComponentType, component_count: u8) -> Self {
        Self {
            semantic,
            component_type,
            component_count,
            normalized: component_type.is_normalized(),
        }
    }

    pub fn byte_size(&self) -> usize {
        self.component_type.byte_size() * self.component_count as usize
    }

    /// Common: position as 3 floats.
    pub fn position_f32() -> Self {
        Self::new(AttributeSemantic::Position, ComponentType::Float32, 3)
    }
    /// Common: normal as 3 floats.
    pub fn normal_f32() -> Self {
        Self::new(AttributeSemantic::Normal, ComponentType::Float32, 3)
    }
    /// Common: UV as 2 floats.
    pub fn texcoord_f32(set: u8) -> Self {
        Self::new(AttributeSemantic::TexCoord(set), ComponentType::Float32, 2)
    }
    /// Common: RGBA color as 4 normalized u8.
    pub fn color_unorm8(set: u8) -> Self {
        Self::new(AttributeSemantic::Color(set), ComponentType::Unorm8, 4)
    }
    /// Common: RGBA color as 4 floats.
    pub fn color_f32(set: u8) -> Self {
        Self::new(AttributeSemantic::Color(set), ComponentType::Float32, 4)
    }
    /// Common: tangent as Vec4 (xyz + handedness w).
    pub fn tangent_f32() -> Self {
        Self::new(AttributeSemantic::Tangent, ComponentType::Float32, 4)
    }
    /// Common: bone weights as 4 floats.
    pub fn bone_weights_f32() -> Self {
        Self::new(AttributeSemantic::BoneWeights, ComponentType::Float32, 4)
    }
    /// Common: bone indices as 4 u16.
    pub fn bone_indices_u16() -> Self {
        Self::new(AttributeSemantic::BoneIndices, ComponentType::Uint16, 4)
    }
}

// ── VertexFormat ─────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttributeBinding {
    pub attribute: VertexAttribute,
    pub offset: usize,
    pub buffer_index: u32,
}

/// Describes the layout of vertices in memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexFormat {
    pub bindings: Vec<AttributeBinding>,
    pub stride: usize,
    pub interleaved: bool,
}

impl VertexFormat {
    /// Create an interleaved vertex format from a list of attributes.
    pub fn interleaved(attributes: &[VertexAttribute]) -> Self {
        let mut bindings = Vec::new();
        let mut offset = 0usize;
        for attr in attributes {
            bindings.push(AttributeBinding {
                attribute: attr.clone(),
                offset,
                buffer_index: 0,
            });
            offset += attr.byte_size();
        }
        Self { bindings, stride: offset, interleaved: true }
    }

    /// Create a separate (struct-of-arrays) format — each attribute in its own buffer.
    pub fn separate(attributes: &[VertexAttribute]) -> Self {
        let mut bindings = Vec::new();
        for (i, attr) in attributes.iter().enumerate() {
            bindings.push(AttributeBinding {
                attribute: attr.clone(),
                offset: 0,
                buffer_index: i as u32,
            });
        }
        let stride = attributes.iter().map(|a| a.byte_size()).sum();
        Self { bindings, stride, interleaved: false }
    }

    pub fn attribute_count(&self) -> usize { self.bindings.len() }

    pub fn has_semantic(&self, semantic: AttributeSemantic) -> bool {
        self.bindings.iter().any(|b| b.attribute.semantic == semantic)
    }

    pub fn find_attribute(&self, semantic: AttributeSemantic) -> Option<&AttributeBinding> {
        self.bindings.iter().find(|b| b.attribute.semantic == semantic)
    }

    /// Compute a 64-bit hash of this format for pipeline compatibility checks.
    pub fn format_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        for b in &self.bindings {
            b.attribute.hash(&mut hasher);
            b.offset.hash(&mut hasher);
            b.buffer_index.hash(&mut hasher);
        }
        self.stride.hash(&mut hasher);
        self.interleaved.hash(&mut hasher);
        hasher.finish()
    }

    /// Total bytes needed for `vertex_count` vertices.
    pub fn buffer_size(&self, vertex_count: usize) -> usize {
        self.stride * vertex_count
    }

    /// Number of distinct buffers used.
    pub fn buffer_count(&self) -> u32 {
        self.bindings.iter().map(|b| b.buffer_index).max().map_or(0, |m| m + 1)
    }

    /// Stride per buffer for separate layouts.
    pub fn strides_per_buffer(&self) -> HashMap<u32, usize> {
        let mut map = HashMap::new();
        for b in &self.bindings {
            *map.entry(b.buffer_index).or_insert(0) += b.attribute.byte_size();
        }
        map
    }
}

// ── Format Conversion ────────────────────────────────────────

/// Vertex data stored as flat byte arrays per buffer.
#[derive(Debug, Clone)]
pub struct VertexData {
    pub format: VertexFormat,
    pub buffers: Vec<Vec<u8>>,
    pub vertex_count: usize,
}

impl VertexData {
    pub fn new(format: VertexFormat, vertex_count: usize) -> Self {
        let buf_count = format.buffer_count().max(1) as usize;
        let strides = format.strides_per_buffer();
        let mut buffers = Vec::with_capacity(buf_count);
        for i in 0..buf_count {
            let stride = strides.get(&(i as u32)).copied().unwrap_or(format.stride);
            buffers.push(vec![0u8; stride * vertex_count]);
        }
        Self { format, buffers, vertex_count }
    }

    /// Write a f32 component to the given vertex index and attribute.
    pub fn write_f32(&mut self, vertex: usize, semantic: AttributeSemantic, values: &[f32]) {
        if let Some(binding) = self.format.find_attribute(semantic) {
            let buf_idx = binding.buffer_index as usize;
            let stride = self.format.strides_per_buffer()
                .get(&binding.buffer_index).copied()
                .unwrap_or(self.format.stride);
            let base = vertex * stride + binding.offset;
            let count = values.len().min(binding.attribute.component_count as usize);
            for i in 0..count {
                let bytes = values[i].to_le_bytes();
                let off = base + i * 4;
                if off + 4 <= self.buffers[buf_idx].len() {
                    self.buffers[buf_idx][off..off + 4].copy_from_slice(&bytes);
                }
            }
        }
    }

    /// Read f32 components from the given vertex index and attribute.
    pub fn read_f32(&self, vertex: usize, semantic: AttributeSemantic) -> Vec<f32> {
        if let Some(binding) = self.format.find_attribute(semantic) {
            let buf_idx = binding.buffer_index as usize;
            let stride = self.format.strides_per_buffer()
                .get(&binding.buffer_index).copied()
                .unwrap_or(self.format.stride);
            let base = vertex * stride + binding.offset;
            let count = binding.attribute.component_count as usize;
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let off = base + i * 4;
                if off + 4 <= self.buffers[buf_idx].len() {
                    let bytes = [
                        self.buffers[buf_idx][off],
                        self.buffers[buf_idx][off + 1],
                        self.buffers[buf_idx][off + 2],
                        self.buffers[buf_idx][off + 3],
                    ];
                    result.push(f32::from_le_bytes(bytes));
                }
            }
            result
        } else {
            Vec::new()
        }
    }

    /// Convert to a new format, adding zero-filled attributes or removing unused ones.
    pub fn convert_to(&self, new_format: &VertexFormat) -> VertexData {
        let mut new_data = VertexData::new(new_format.clone(), self.vertex_count);
        for new_binding in &new_format.bindings {
            let sem = new_binding.attribute.semantic;
            for v in 0..self.vertex_count {
                let old_vals = self.read_f32(v, sem);
                if !old_vals.is_empty() {
                    new_data.write_f32(v, sem, &old_vals);
                }
            }
        }
        new_data
    }
}

// ── Predefined Formats ──────────────────────────────────────

/// Common vertex format presets.
pub struct VertexFormats;

impl VertexFormats {
    /// Position only (12 bytes).
    pub fn position() -> VertexFormat {
        VertexFormat::interleaved(&[VertexAttribute::position_f32()])
    }

    /// Position + Normal (24 bytes).
    pub fn position_normal() -> VertexFormat {
        VertexFormat::interleaved(&[
            VertexAttribute::position_f32(),
            VertexAttribute::normal_f32(),
        ])
    }

    /// Position + Normal + TexCoord (32 bytes).
    pub fn position_normal_uv() -> VertexFormat {
        VertexFormat::interleaved(&[
            VertexAttribute::position_f32(),
            VertexAttribute::normal_f32(),
            VertexAttribute::texcoord_f32(0),
        ])
    }

    /// Full PBR: Position + Normal + UV + Tangent (48 bytes).
    pub fn pbr() -> VertexFormat {
        VertexFormat::interleaved(&[
            VertexAttribute::position_f32(),
            VertexAttribute::normal_f32(),
            VertexAttribute::texcoord_f32(0),
            VertexAttribute::tangent_f32(),
        ])
    }

    /// Skinned: Position + Normal + UV + BoneWeights + BoneIndices (48 bytes).
    pub fn skinned() -> VertexFormat {
        VertexFormat::interleaved(&[
            VertexAttribute::position_f32(),
            VertexAttribute::normal_f32(),
            VertexAttribute::texcoord_f32(0),
            VertexAttribute::bone_weights_f32(),
            VertexAttribute::bone_indices_u16(),
        ])
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_byte_sizes() {
        assert_eq!(ComponentType::Float32.byte_size(), 4);
        assert_eq!(ComponentType::Float64.byte_size(), 8);
        assert_eq!(ComponentType::Int8.byte_size(), 1);
        assert_eq!(ComponentType::Int16.byte_size(), 2);
        assert_eq!(ComponentType::Uint32.byte_size(), 4);
    }

    #[test]
    fn test_attribute_byte_size() {
        let a = VertexAttribute::position_f32();
        assert_eq!(a.byte_size(), 12); // 3 * 4
    }

    #[test]
    fn test_interleaved_stride() {
        let fmt = VertexFormats::position_normal_uv();
        assert_eq!(fmt.stride, 32); // 12 + 12 + 8
    }

    #[test]
    fn test_interleaved_offsets() {
        let fmt = VertexFormats::position_normal_uv();
        assert_eq!(fmt.bindings[0].offset, 0);
        assert_eq!(fmt.bindings[1].offset, 12);
        assert_eq!(fmt.bindings[2].offset, 24);
    }

    #[test]
    fn test_separate_layout() {
        let attrs = vec![
            VertexAttribute::position_f32(),
            VertexAttribute::normal_f32(),
        ];
        let fmt = VertexFormat::separate(&attrs);
        assert!(!fmt.interleaved);
        assert_eq!(fmt.bindings[0].buffer_index, 0);
        assert_eq!(fmt.bindings[1].buffer_index, 1);
        assert_eq!(fmt.buffer_count(), 2);
    }

    #[test]
    fn test_has_semantic() {
        let fmt = VertexFormats::pbr();
        assert!(fmt.has_semantic(AttributeSemantic::Position));
        assert!(fmt.has_semantic(AttributeSemantic::Tangent));
        assert!(!fmt.has_semantic(AttributeSemantic::BoneWeights));
    }

    #[test]
    fn test_find_attribute() {
        let fmt = VertexFormats::position_normal();
        let binding = fmt.find_attribute(AttributeSemantic::Normal).unwrap();
        assert_eq!(binding.offset, 12);
    }

    #[test]
    fn test_format_hash_differs() {
        let h1 = VertexFormats::position().format_hash();
        let h2 = VertexFormats::position_normal().format_hash();
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_format_hash_same() {
        let h1 = VertexFormats::pbr().format_hash();
        let h2 = VertexFormats::pbr().format_hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_buffer_size() {
        let fmt = VertexFormats::position_normal_uv();
        assert_eq!(fmt.buffer_size(100), 3200);
    }

    #[test]
    fn test_vertex_data_write_read() {
        let fmt = VertexFormats::position_normal();
        let mut data = VertexData::new(fmt, 2);
        data.write_f32(0, AttributeSemantic::Position, &[1.0, 2.0, 3.0]);
        let vals = data.read_f32(0, AttributeSemantic::Position);
        assert_eq!(vals.len(), 3);
        assert!((vals[0] - 1.0).abs() < 1e-6);
        assert!((vals[1] - 2.0).abs() < 1e-6);
        assert!((vals[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_vertex_data_normal_write_read() {
        let fmt = VertexFormats::position_normal();
        let mut data = VertexData::new(fmt, 1);
        data.write_f32(0, AttributeSemantic::Normal, &[0.0, 1.0, 0.0]);
        let vals = data.read_f32(0, AttributeSemantic::Normal);
        assert_eq!(vals.len(), 3);
        assert!((vals[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_convert_add_attribute() {
        let src_fmt = VertexFormats::position();
        let mut data = VertexData::new(src_fmt, 1);
        data.write_f32(0, AttributeSemantic::Position, &[5.0, 6.0, 7.0]);
        let dst_fmt = VertexFormats::position_normal();
        let converted = data.convert_to(&dst_fmt);
        let pos = converted.read_f32(0, AttributeSemantic::Position);
        assert!((pos[0] - 5.0).abs() < 1e-6);
        let norm = converted.read_f32(0, AttributeSemantic::Normal);
        assert!((norm[0]).abs() < 1e-6); // zero-filled
    }

    #[test]
    fn test_convert_remove_attribute() {
        let src_fmt = VertexFormats::position_normal_uv();
        let mut data = VertexData::new(src_fmt, 1);
        data.write_f32(0, AttributeSemantic::Position, &[1.0, 2.0, 3.0]);
        data.write_f32(0, AttributeSemantic::TexCoord(0), &[0.5, 0.5]);
        let dst_fmt = VertexFormats::position();
        let converted = data.convert_to(&dst_fmt);
        let pos = converted.read_f32(0, AttributeSemantic::Position);
        assert!((pos[0] - 1.0).abs() < 1e-6);
        let uv = converted.read_f32(0, AttributeSemantic::TexCoord(0));
        assert!(uv.is_empty());
    }

    #[test]
    fn test_color_unorm8() {
        let attr = VertexAttribute::color_unorm8(0);
        assert_eq!(attr.byte_size(), 4);
        assert!(attr.normalized);
    }

    #[test]
    fn test_bone_indices_u16() {
        let attr = VertexAttribute::bone_indices_u16();
        assert_eq!(attr.byte_size(), 8); // 4 * 2
        assert_eq!(attr.component_type, ComponentType::Uint16);
    }

    #[test]
    fn test_position_format_stride() {
        let fmt = VertexFormats::position();
        assert_eq!(fmt.stride, 12);
    }

    #[test]
    fn test_pbr_format_stride() {
        let fmt = VertexFormats::pbr();
        assert_eq!(fmt.stride, 48); // 12+12+8+16
    }

    #[test]
    fn test_skinned_format() {
        let fmt = VertexFormats::skinned();
        assert!(fmt.has_semantic(AttributeSemantic::BoneWeights));
        assert!(fmt.has_semantic(AttributeSemantic::BoneIndices));
    }

    #[test]
    fn test_strides_per_buffer_separate() {
        let attrs = vec![
            VertexAttribute::position_f32(),
            VertexAttribute::normal_f32(),
        ];
        let fmt = VertexFormat::separate(&attrs);
        let strides = fmt.strides_per_buffer();
        assert_eq!(*strides.get(&0).unwrap(), 12);
        assert_eq!(*strides.get(&1).unwrap(), 12);
    }

    #[test]
    fn test_multiple_texcoord_sets() {
        let fmt = VertexFormat::interleaved(&[
            VertexAttribute::position_f32(),
            VertexAttribute::texcoord_f32(0),
            VertexAttribute::texcoord_f32(1),
        ]);
        assert!(fmt.has_semantic(AttributeSemantic::TexCoord(0)));
        assert!(fmt.has_semantic(AttributeSemantic::TexCoord(1)));
        assert!(!fmt.has_semantic(AttributeSemantic::TexCoord(2)));
    }

    #[test]
    fn test_custom_attribute() {
        let custom = VertexAttribute::new(
            AttributeSemantic::Custom(42),
            ComponentType::Float32,
            2,
        );
        assert_eq!(custom.byte_size(), 8);
        let fmt = VertexFormat::interleaved(&[
            VertexAttribute::position_f32(),
            custom,
        ]);
        assert!(fmt.has_semantic(AttributeSemantic::Custom(42)));
    }

    #[test]
    fn test_empty_format() {
        let fmt = VertexFormat::interleaved(&[]);
        assert_eq!(fmt.stride, 0);
        assert_eq!(fmt.attribute_count(), 0);
    }
}
