//! Index buffer management — 16-bit and 32-bit indices, triangle list/strip/fan,
//! line list/strip, point list topologies. Primitive restart support.
//! Index optimization (vertex cache, Forsyth-like scoring). Degenerate triangle
//! detection and removal. Index buffer statistics.

// ── Topology ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Topology {
    TriangleList,
    TriangleStrip,
    TriangleFan,
    LineList,
    LineStrip,
    PointList,
}

impl Topology {
    /// How many indices does one primitive need?
    pub fn indices_per_primitive(self) -> usize {
        match self {
            Topology::TriangleList => 3,
            Topology::LineList => 2,
            Topology::PointList => 1,
            // Strips and fans: first primitive costs more, subsequent cost 1
            Topology::TriangleStrip | Topology::TriangleFan | Topology::LineStrip => 1,
        }
    }
}

// ── IndexType ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    U16,
    U32,
}

impl IndexType {
    pub fn byte_size(self) -> usize {
        match self {
            IndexType::U16 => 2,
            IndexType::U32 => 4,
        }
    }

    pub fn max_value(self) -> u32 {
        match self {
            IndexType::U16 => 0xFFFF,
            IndexType::U32 => 0xFFFFFFFF,
        }
    }

    /// The restart sentinel for this index type.
    pub fn restart_index(self) -> u32 {
        self.max_value()
    }
}

// ── IndexBufferStats ─────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexBufferStats {
    pub index_count: usize,
    pub primitive_count: usize,
    pub min_index: u32,
    pub max_index: u32,
    pub unique_index_count: usize,
    pub degenerate_triangle_count: usize,
    pub index_type: IndexType,
    pub byte_size: usize,
}

// ── IndexBuffer ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct IndexBuffer {
    indices: Vec<u32>,
    topology: Topology,
    index_type: IndexType,
    primitive_restart: bool,
}

impl IndexBuffer {
    pub fn new(topology: Topology, index_type: IndexType) -> Self {
        Self { indices: Vec::new(), topology, index_type, primitive_restart: false }
    }

    pub fn from_indices(indices: Vec<u32>, topology: Topology) -> Self {
        let max_val = indices.iter().copied().max().unwrap_or(0);
        let index_type = if max_val <= 0xFFFF { IndexType::U16 } else { IndexType::U32 };
        Self { indices, topology, index_type, primitive_restart: false }
    }

    pub fn set_primitive_restart(&mut self, enabled: bool) { self.primitive_restart = enabled; }
    pub fn primitive_restart_enabled(&self) -> bool { self.primitive_restart }
    pub fn topology(&self) -> Topology { self.topology }
    pub fn index_type(&self) -> IndexType { self.index_type }
    pub fn indices(&self) -> &[u32] { &self.indices }
    pub fn len(&self) -> usize { self.indices.len() }
    pub fn is_empty(&self) -> bool { self.indices.is_empty() }

    pub fn push(&mut self, index: u32) {
        self.indices.push(index);
        if index > 0xFFFF && self.index_type == IndexType::U16 {
            self.index_type = IndexType::U32;
        }
    }

    pub fn push_triangle(&mut self, a: u32, b: u32, c: u32) {
        self.indices.push(a);
        self.indices.push(b);
        self.indices.push(c);
        let mx = a.max(b).max(c);
        if mx > 0xFFFF && self.index_type == IndexType::U16 {
            self.index_type = IndexType::U32;
        }
    }

    pub fn push_restart(&mut self) {
        self.indices.push(self.index_type.restart_index());
    }

    /// Convert triangle strip to triangle list.
    pub fn strip_to_list(&self) -> IndexBuffer {
        if self.topology != Topology::TriangleStrip || self.indices.len() < 3 {
            return self.clone();
        }
        let mut out = Vec::new();
        let restart = if self.primitive_restart { Some(self.index_type.restart_index()) } else { None };
        let mut strip_start = 0;
        let mut i = 0;
        while i < self.indices.len() {
            if restart.map_or(false, |r| self.indices[i] == r) {
                strip_start = i + 1;
                i += 1;
                continue;
            }
            let local = i - strip_start;
            if local >= 2 {
                let (a, b, c) = if local % 2 == 0 {
                    (self.indices[i - 2], self.indices[i - 1], self.indices[i])
                } else {
                    (self.indices[i - 1], self.indices[i - 2], self.indices[i])
                };
                out.push(a);
                out.push(b);
                out.push(c);
            }
            i += 1;
        }
        let mut buf = IndexBuffer::from_indices(out, Topology::TriangleList);
        buf.index_type = self.index_type;
        buf
    }

    /// Convert triangle fan to triangle list.
    pub fn fan_to_list(&self) -> IndexBuffer {
        if self.topology != Topology::TriangleFan || self.indices.len() < 3 {
            return self.clone();
        }
        let mut out = Vec::new();
        let hub = self.indices[0];
        for i in 2..self.indices.len() {
            out.push(hub);
            out.push(self.indices[i - 1]);
            out.push(self.indices[i]);
        }
        let mut buf = IndexBuffer::from_indices(out, Topology::TriangleList);
        buf.index_type = self.index_type;
        buf
    }

    /// Expand triangles from the current topology into explicit triangle triples.
    pub fn expand_triangles(&self) -> Vec<[u32; 3]> {
        match self.topology {
            Topology::TriangleList => {
                let mut tris = Vec::new();
                let mut i = 0;
                while i + 2 < self.indices.len() {
                    tris.push([self.indices[i], self.indices[i + 1], self.indices[i + 2]]);
                    i += 3;
                }
                tris
            }
            Topology::TriangleStrip => {
                self.strip_to_list().expand_triangles()
            }
            Topology::TriangleFan => {
                self.fan_to_list().expand_triangles()
            }
            _ => Vec::new(),
        }
    }

    /// Detect degenerate triangles (two or more identical vertex indices).
    pub fn find_degenerate_triangles(&self) -> Vec<usize> {
        let tris = self.expand_triangles();
        let mut degenerate = Vec::new();
        for (i, tri) in tris.iter().enumerate() {
            if tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2] {
                degenerate.push(i);
            }
        }
        degenerate
    }

    /// Remove degenerate triangles. Only works on triangle list topology.
    pub fn remove_degenerate_triangles(&mut self) {
        if self.topology != Topology::TriangleList {
            return;
        }
        let mut new_indices = Vec::new();
        let mut i = 0;
        while i + 2 < self.indices.len() {
            let a = self.indices[i];
            let b = self.indices[i + 1];
            let c = self.indices[i + 2];
            if a != b && b != c && a != c {
                new_indices.push(a);
                new_indices.push(b);
                new_indices.push(c);
            }
            i += 3;
        }
        self.indices = new_indices;
    }

    /// Compute statistics.
    pub fn stats(&self) -> IndexBufferStats {
        let min_index = self.indices.iter().copied().min().unwrap_or(0);
        let max_index = self.indices.iter().copied().max().unwrap_or(0);
        let mut unique = self.indices.clone();
        unique.sort_unstable();
        unique.dedup();
        let unique_index_count = unique.len();
        let primitive_count = match self.topology {
            Topology::TriangleList => self.indices.len() / 3,
            Topology::LineList => self.indices.len() / 2,
            Topology::PointList => self.indices.len(),
            Topology::TriangleStrip | Topology::TriangleFan => {
                if self.indices.len() >= 3 { self.indices.len() - 2 } else { 0 }
            }
            Topology::LineStrip => {
                if self.indices.len() >= 2 { self.indices.len() - 1 } else { 0 }
            }
        };
        let degenerate_triangle_count = self.find_degenerate_triangles().len();
        IndexBufferStats {
            index_count: self.indices.len(),
            primitive_count,
            min_index,
            max_index,
            unique_index_count,
            degenerate_triangle_count,
            index_type: self.index_type,
            byte_size: self.indices.len() * self.index_type.byte_size(),
        }
    }

    /// Vertex-cache-friendly reordering using a Forsyth-like scoring heuristic.
    /// Operates on triangle lists only.
    pub fn optimize_vertex_cache(&mut self, vertex_count: usize) {
        if self.topology != Topology::TriangleList || self.indices.len() < 3 {
            return;
        }
        let cache_size: usize = 32;
        let tri_count = self.indices.len() / 3;

        // Build adjacency: for each vertex, which triangles reference it?
        let mut vert_tris: Vec<Vec<usize>> = vec![Vec::new(); vertex_count];
        for t in 0..tri_count {
            for k in 0..3 {
                let vi = self.indices[t * 3 + k] as usize;
                if vi < vertex_count {
                    vert_tris[vi].push(t);
                }
            }
        }

        // Score function for valence
        fn valence_score(active_tri_count: usize) -> f64 {
            if active_tri_count == 0 { return -1.0; }
            // Bonus for low valence
            let bonus = 2.0 / (active_tri_count as f64).sqrt();
            bonus
        }

        fn cache_score(cache_pos: Option<usize>, cache_size: usize) -> f64 {
            match cache_pos {
                None => 0.0,
                Some(pos) => {
                    if pos < 3 { 0.75 } // recently used
                    else {
                        let normalized = 1.0 - ((pos - 3) as f64) / ((cache_size - 3) as f64);
                        normalized.powf(1.5)
                    }
                }
            }
        }

        let mut active_count: Vec<usize> = (0..vertex_count)
            .map(|v| vert_tris[v].len())
            .collect();
        let mut emitted = vec![false; tri_count];
        // LRU cache
        let mut lru_cache: Vec<u32> = Vec::with_capacity(cache_size);

        fn find_in_cache(cache: &[u32], v: u32) -> Option<usize> {
            cache.iter().position(|c| *c == v)
        }

        let mut result = Vec::with_capacity(self.indices.len());

        for _ in 0..tri_count {
            // Score all candidate triangles (those touching cache verts first)
            let mut best_tri = None;
            let mut best_score = f64::NEG_INFINITY;

            // Check triangles adjacent to cached vertices first
            let cached_verts: Vec<u32> = lru_cache.clone();
            let mut candidates = Vec::new();
            for &cv in &cached_verts {
                let vi = cv as usize;
                if vi < vertex_count {
                    for &t in &vert_tris[vi] {
                        if !emitted[t] { candidates.push(t); }
                    }
                }
            }
            candidates.sort_unstable();
            candidates.dedup();

            if candidates.is_empty() {
                // Fall back to any unemitted triangle
                for t in 0..tri_count {
                    if !emitted[t] { candidates.push(t); break; }
                }
            }

            for t in &candidates {
                let mut score = 0.0;
                for k in 0..3 {
                    let vi = self.indices[t * 3 + k] as usize;
                    if vi < vertex_count {
                        let cp = find_in_cache(&lru_cache, vi as u32);
                        score += cache_score(cp, cache_size) + valence_score(active_count[vi]);
                    }
                }
                if score > best_score {
                    best_score = score;
                    best_tri = Some(*t);
                }
            }

            if let Some(t) = best_tri {
                emitted[t] = true;
                for k in 0..3 {
                    let vi = self.indices[t * 3 + k];
                    result.push(vi);
                    let viu = vi as usize;
                    if viu < vertex_count && active_count[viu] > 0 {
                        active_count[viu] -= 1;
                    }
                    // Update LRU
                    if let Some(pos) = find_in_cache(&lru_cache, vi) {
                        lru_cache.remove(pos);
                    }
                    lru_cache.insert(0, vi);
                    if lru_cache.len() > cache_size {
                        lru_cache.pop();
                    }
                }
            }
        }
        self.indices = result;
    }

    /// Convert to the opposite index type.
    pub fn convert_type(&mut self, new_type: IndexType) {
        if new_type == IndexType::U16 {
            for idx in &mut self.indices {
                if *idx > 0xFFFF { *idx = 0xFFFF; }
            }
        }
        self.index_type = new_type;
    }

    /// Serialize to bytes in the chosen format.
    pub fn to_bytes(&self) -> Vec<u8> {
        match self.index_type {
            IndexType::U16 => {
                let mut out = Vec::with_capacity(self.indices.len() * 2);
                for &i in &self.indices {
                    out.extend_from_slice(&(i as u16).to_le_bytes());
                }
                out
            }
            IndexType::U32 => {
                let mut out = Vec::with_capacity(self.indices.len() * 4);
                for &i in &self.indices {
                    out.extend_from_slice(&i.to_le_bytes());
                }
                out
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_list_stats() {
        let buf = IndexBuffer::from_indices(vec![0, 1, 2, 2, 3, 0], Topology::TriangleList);
        let s = buf.stats();
        assert_eq!(s.index_count, 6);
        assert_eq!(s.primitive_count, 2);
        assert_eq!(s.min_index, 0);
        assert_eq!(s.max_index, 3);
    }

    #[test]
    fn test_unique_indices() {
        let buf = IndexBuffer::from_indices(vec![0, 1, 2, 0, 2, 3], Topology::TriangleList);
        assert_eq!(buf.stats().unique_index_count, 4);
    }

    #[test]
    fn test_auto_type_u16() {
        let buf = IndexBuffer::from_indices(vec![0, 100, 200], Topology::TriangleList);
        assert_eq!(buf.index_type(), IndexType::U16);
    }

    #[test]
    fn test_auto_type_u32() {
        let buf = IndexBuffer::from_indices(vec![0, 70000, 100000], Topology::TriangleList);
        assert_eq!(buf.index_type(), IndexType::U32);
    }

    #[test]
    fn test_push_promotes_type() {
        let mut buf = IndexBuffer::new(Topology::TriangleList, IndexType::U16);
        buf.push(100);
        assert_eq!(buf.index_type(), IndexType::U16);
        buf.push(70000);
        assert_eq!(buf.index_type(), IndexType::U32);
    }

    #[test]
    fn test_strip_to_list() {
        let buf = IndexBuffer::from_indices(vec![0, 1, 2, 3], Topology::TriangleStrip);
        let list = buf.strip_to_list();
        assert_eq!(list.topology(), Topology::TriangleList);
        assert_eq!(list.len(), 6); // 2 triangles * 3
    }

    #[test]
    fn test_strip_winding() {
        let buf = IndexBuffer::from_indices(vec![0, 1, 2, 3], Topology::TriangleStrip);
        let list = buf.strip_to_list();
        let idx = list.indices();
        // First triangle: 0,1,2
        assert_eq!(idx[0], 0);
        assert_eq!(idx[1], 1);
        assert_eq!(idx[2], 2);
        // Second triangle (odd): swapped winding
        assert_eq!(idx[3], 2);
        assert_eq!(idx[4], 1);
        assert_eq!(idx[5], 3);
    }

    #[test]
    fn test_fan_to_list() {
        let buf = IndexBuffer::from_indices(vec![0, 1, 2, 3, 4], Topology::TriangleFan);
        let list = buf.fan_to_list();
        assert_eq!(list.len(), 9); // 3 triangles * 3
        let idx = list.indices();
        // All share vertex 0
        assert_eq!(idx[0], 0);
        assert_eq!(idx[3], 0);
        assert_eq!(idx[6], 0);
    }

    #[test]
    fn test_degenerate_detection() {
        let buf = IndexBuffer::from_indices(vec![0, 0, 1, 2, 3, 4], Topology::TriangleList);
        let degens = buf.find_degenerate_triangles();
        assert_eq!(degens.len(), 1);
        assert_eq!(degens[0], 0);
    }

    #[test]
    fn test_remove_degenerate() {
        let mut buf = IndexBuffer::from_indices(
            vec![0, 0, 1, 2, 3, 4], Topology::TriangleList,
        );
        buf.remove_degenerate_triangles();
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.indices()[0], 2);
    }

    #[test]
    fn test_no_degenerate() {
        let buf = IndexBuffer::from_indices(vec![0, 1, 2], Topology::TriangleList);
        assert_eq!(buf.find_degenerate_triangles().len(), 0);
    }

    #[test]
    fn test_line_list_stats() {
        let buf = IndexBuffer::from_indices(vec![0, 1, 2, 3], Topology::LineList);
        assert_eq!(buf.stats().primitive_count, 2);
    }

    #[test]
    fn test_line_strip_stats() {
        let buf = IndexBuffer::from_indices(vec![0, 1, 2, 3], Topology::LineStrip);
        assert_eq!(buf.stats().primitive_count, 3);
    }

    #[test]
    fn test_point_list_stats() {
        let buf = IndexBuffer::from_indices(vec![0, 1, 2], Topology::PointList);
        assert_eq!(buf.stats().primitive_count, 3);
    }

    #[test]
    fn test_to_bytes_u16() {
        let buf = IndexBuffer::from_indices(vec![1, 2, 3], Topology::TriangleList);
        let bytes = buf.to_bytes();
        assert_eq!(bytes.len(), 6);
        assert_eq!(u16::from_le_bytes([bytes[0], bytes[1]]), 1);
    }

    #[test]
    fn test_to_bytes_u32() {
        let mut buf = IndexBuffer::new(Topology::TriangleList, IndexType::U32);
        buf.push(100000);
        let bytes = buf.to_bytes();
        assert_eq!(bytes.len(), 4);
        assert_eq!(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]), 100000);
    }

    #[test]
    fn test_convert_type_clamp() {
        let mut buf = IndexBuffer::from_indices(vec![0, 70000, 2], Topology::TriangleList);
        buf.convert_type(IndexType::U16);
        assert_eq!(buf.indices()[1], 0xFFFF);
    }

    #[test]
    fn test_optimize_preserves_triangles() {
        let mut buf = IndexBuffer::from_indices(
            vec![0, 1, 2, 2, 3, 0, 3, 4, 5], Topology::TriangleList,
        );
        let original_count = buf.stats().primitive_count;
        buf.optimize_vertex_cache(6);
        assert_eq!(buf.stats().primitive_count, original_count);
    }

    #[test]
    fn test_optimize_preserves_indices() {
        let indices = vec![0, 1, 2, 3, 4, 5];
        let mut buf = IndexBuffer::from_indices(indices.clone(), Topology::TriangleList);
        buf.optimize_vertex_cache(6);
        let mut sorted_old = indices;
        sorted_old.sort();
        let mut sorted_new: Vec<u32> = buf.indices().to_vec();
        sorted_new.sort();
        assert_eq!(sorted_old, sorted_new);
    }

    #[test]
    fn test_empty_buffer() {
        let buf = IndexBuffer::new(Topology::TriangleList, IndexType::U16);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_primitive_restart_flag() {
        let mut buf = IndexBuffer::new(Topology::TriangleStrip, IndexType::U16);
        assert!(!buf.primitive_restart_enabled());
        buf.set_primitive_restart(true);
        assert!(buf.primitive_restart_enabled());
    }

    #[test]
    fn test_push_triangle() {
        let mut buf = IndexBuffer::new(Topology::TriangleList, IndexType::U16);
        buf.push_triangle(0, 1, 2);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.indices(), &[0, 1, 2]);
    }

    #[test]
    fn test_byte_size_stat() {
        let buf = IndexBuffer::from_indices(vec![0, 1, 2], Topology::TriangleList);
        assert_eq!(buf.stats().byte_size, 6); // 3 * 2 (u16)
    }

    #[test]
    fn test_strip_restart_to_list() {
        let mut buf = IndexBuffer::new(Topology::TriangleStrip, IndexType::U16);
        buf.set_primitive_restart(true);
        // First strip: 0,1,2,3
        for i in 0..4 { buf.push(i); }
        buf.push_restart();
        // Second strip: 4,5,6,7
        for i in 4..8 { buf.push(i); }
        let list = buf.strip_to_list();
        assert_eq!(list.stats().primitive_count, 4); // 2+2
    }
}
