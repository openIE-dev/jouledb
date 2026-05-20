//! WGSL compute shaders for database operations
//!
//! These shaders implement various database operations on the GPU:
//! - Aggregations (sum, count, min, max, avg)
//! - Hash functions (xxhash64, crc32, murmur3)
//! - Vector operations (similarity search, dot product)
//! - Hyperdimensional operations (bind, bundle, similarity)
//! - B-tree operations (range scan, filtering)
//! - Columnar operations (format conversion, aggregations)

/// Sum aggregation shader
pub const SUM_SHADER: &str = r#"
struct AggregationParams {
    num_elements: u32,
    _padding: u32,
    _padding2: u32,
    _padding3: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> params: AggregationParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_elements) {
        return;
    }
    
    // Atomic add to output
    atomicAdd(&output_data[0], input_data[idx]);
}
"#;

/// Count aggregation shader
pub const COUNT_SHADER: &str = r#"
struct AggregationParams {
    num_elements: u32,
    _padding: u32,
    _padding2: u32,
    _padding3: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<u32>;
@group(0) @binding(2) var<uniform> params: AggregationParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_elements) {
        return;
    }
    
    // Count non-null elements
    if (input_data[idx] != 0u) {
        atomicAdd(&output_data[0], 1u);
    }
}
"#;

/// Average aggregation shader
pub const AVG_SHADER: &str = r#"
struct AggregationParams {
    num_elements: u32,
    _padding: u32,
    _padding2: u32,
    _padding3: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> sum_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> count_data: array<u32>;
@group(0) @binding(3) var<uniform> params: AggregationParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_elements) {
        return;
    }
    
    // Sum and count
    atomicAdd(&sum_data[0], input_data[idx]);
    atomicAdd(&count_data[0], 1u);
}
"#;

/// Min aggregation shader
pub const MIN_SHADER: &str = r#"
struct AggregationParams {
    num_elements: u32,
    _padding: u32,
    _padding2: u32,
    _padding3: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> params: AggregationParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_elements) {
        return;
    }
    
    // Atomic min
    atomicMin(&output_data[0], input_data[idx]);
}
"#;

/// Max aggregation shader
pub const MAX_SHADER: &str = r#"
struct AggregationParams {
    num_elements: u32,
    _padding: u32,
    _padding2: u32,
    _padding3: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> params: AggregationParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_elements) {
        return;
    }
    
    // Atomic max
    atomicMax(&output_data[0], input_data[idx]);
}
"#;

/// XXHASH64 hash shader
/// Simplified implementation of XXHASH64 for GPU
pub const XXHASH64_SHADER: &str = r#"
struct HashParams {
    data_length: u32,
    seed: u64,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u8>;
@group(0) @binding(1) var<storage, read_write> output_hash: array<u64>;
@group(0) @binding(2) var<uniform> params: HashParams;

const PRIME64_1: u64 = 11400714785074694791u;
const PRIME64_2: u64 = 14029467366897019727u;
const PRIME64_3: u64 = 1609587929392839161u;
const PRIME64_4: u64 = 9650029242287828579u;
const PRIME64_5: u64 = 2870177450012600261u;

fn read_u64_le(data: array<u8>, offset: u32) -> u64 {
    var result: u64 = 0u;
    for (var i = 0u; i < 8u; i++) {
        if (offset + i < arrayLength(&data)) {
            result = result | (u64(data[offset + i]) << (i * 8u));
        }
    }
    return result;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x > 0u) {
        return; // Only first thread computes hash
    }
    
    var h64: u64 = params.seed + PRIME64_5;
    var offset: u32 = 0u;
    let end = params.data_length;
    
    // Process 32-byte chunks
    while (offset + 32u <= end) {
        var v1 = read_u64_le(input_data, offset) * PRIME64_2;
        var v2 = read_u64_le(input_data, offset + 8u) * PRIME64_2;
        var v3 = read_u64_le(input_data, offset + 16u) * PRIME64_2;
        var v4 = read_u64_le(input_data, offset + 24u) * PRIME64_2;
        
        h64 = rotateLeft(h64 ^ v1, 31u) * PRIME64_1;
        h64 = rotateLeft(h64 ^ v2, 31u) * PRIME64_1;
        h64 = rotateLeft(h64 ^ v3, 31u) * PRIME64_1;
        h64 = rotateLeft(h64 ^ v4, 31u) * PRIME64_1;
        
        offset += 32u;
    }
    
    // Finalize
    h64 = h64 ^ u64(end);
    h64 = h64 * PRIME64_1;
    h64 = h64 ^ (h64 >> 33u);
    h64 = h64 * PRIME64_2;
    h64 = h64 ^ (h64 >> 29u);
    h64 = h64 * PRIME64_3;
    h64 = h64 ^ (h64 >> 32u);
    
    output_hash[0] = h64;
}

fn rotateLeft(value: u64, amount: u32) -> u64 {
    return (value << amount) | (value >> (64u - amount));
}
"#;

/// CRC32 hash shader
/// Standard CRC32 implementation for GPU
pub const CRC32_SHADER: &str = r#"
struct HashParams {
    data_length: u32,
    _padding: u32,
    _padding2: u32,
    _padding3: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u8>;
@group(0) @binding(1) var<storage, read_write> output_hash: array<u32>;
@group(0) @binding(2) var<uniform> params: HashParams;

var<workgroup> crc_table: array<u32, 256>;

fn init_crc_table() {
    for (var i = 0u; i < 256u; i++) {
        var crc = i;
        for (var j = 0u; j < 8u; j++) {
            if ((crc & 1u) != 0u) {
                crc = (crc >> 1u) ^ 0xEDB88320u;
    } else {
                crc = crc >> 1u;
            }
        }
        crc_table[i] = crc;
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Initialize CRC table (first thread only)
    if (idx == 0u) {
        init_crc_table();
    }
    workgroupBarrier();
    
    if (idx >= params.data_length) {
        return;
    }
    
    // Compute CRC32
    var crc: u32 = 0xFFFFFFFFu;
    let chunk_size = (params.data_length + 255u) / 256u;
    let start = idx * chunk_size;
    let end = min(start + chunk_size, params.data_length);
    
    for (var i = start; i < end; i++) {
        let byte = u32(input_data[i]);
        let table_idx = (crc ^ byte) & 0xFFu;
        crc = (crc >> 8u) ^ crc_table[table_idx];
    }
    
    // Combine results (simplified - would use reduction in production)
    if (idx == 0u) {
        output_hash[0] = crc ^ 0xFFFFFFFFu;
    }
}
"#;

/// MURMUR3 hash shader
/// MurmurHash3 implementation for GPU
pub const MURMUR3_SHADER: &str = r#"
struct HashParams {
    data_length: u32,
    seed: u32,
    _padding: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u8>;
@group(0) @binding(1) var<storage, read_write> output_hash: array<u32>;
@group(0) @binding(2) var<uniform> params: HashParams;

fn read_u32_le(data: array<u8>, offset: u32) -> u32 {
    var result: u32 = 0u;
    for (var i = 0u; i < 4u; i++) {
        if (offset + i < arrayLength(&data)) {
            result = result | (u32(data[offset + i]) << (i * 8u));
        }
    }
    return result;
}

fn fmix32(h: u32) -> u32 {
    h = h ^ (h >> 16u);
    h = h * 0x85EBCA6Bu;
    h = h ^ (h >> 13u);
    h = h * 0xC2B2AE35u;
    h = h ^ (h >> 16u);
    return h;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x > 0u) {
        return; // Only first thread computes hash
    }
    
    var h1: u32 = params.seed;
    var offset: u32 = 0u;
    let end = params.data_length;
    
    // Process 4-byte chunks
    while (offset + 4u <= end) {
        var k1 = read_u32_le(input_data, offset);
        k1 = k1 * 0xCC9E2D51u;
        k1 = rotateLeft(k1, 15u);
        k1 = k1 * 0x1B873593u;
        
        h1 = h1 ^ k1;
        h1 = rotateLeft(h1, 13u);
        h1 = h1 * 5u + 0xE6546B64u;
        
        offset += 4u;
    }
    
    // Handle remaining bytes
    if (offset < end) {
        var k1: u32 = 0u;
        for (var i = 0u; i < (end - offset); i++) {
            k1 = k1 | (u32(input_data[offset + i]) << (i * 8u));
        }
        k1 = k1 * 0xCC9E2D51u;
        k1 = rotateLeft(k1, 15u);
        k1 = k1 * 0x1B873593u;
        h1 = h1 ^ k1;
    }
    
    // Finalize
    h1 = h1 ^ u32(end);
    h1 = fmix32(h1);
    
    output_hash[0] = h1;
}

fn rotateLeft(value: u32, amount: u32) -> u32 {
    return (value << amount) | (value >> (32u - amount));
}
"#;

/// B-tree range scan shader
/// Performs parallel range scan over B-tree nodes
pub const BTREE_RANGE_SCAN_SHADER: &str = r#"
struct BTreeParams {
    num_nodes: u32,
    keys_per_node: u32,
    start_key_len: u32,
    end_key_len: u32,
    include_start: u32,
    include_end: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> nodes: array<u8>;
@group(0) @binding(1) var<storage, read> start_key: array<u8>;
@group(0) @binding(2) var<storage, read> end_key: array<u8>;
@group(0) @binding(3) var<storage, read_write> result_keys: array<u32>;
@group(0) @binding(4) var<storage, read_write> result_values: array<u32>;
@group(0) @binding(5) var<storage, read_write> result_count: array<u32>;
@group(0) @binding(6) var<uniform> params: BTreeParams;

fn compare_keys(key1: array<u8>, offset1: u32, len1: u32, key2: array<u8>, offset2: u32, len2: u32) -> i32 {
    let min_len = min(len1, len2);
    for (var i = 0u; i < min_len; i++) {
        let b1 = u32(key1[offset1 + i]);
        let b2 = u32(key2[offset2 + i]);
        if (b1 < b2) {
            return -1;
        } else if (b1 > b2) {
            return 1;
        }
    }
    if (len1 < len2) {
        return -1;
    } else if (len1 > len2) {
        return 1;
    }
    return 0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;
    if (node_idx >= params.num_nodes) {
        return;
    }
    
    // Simplified B-tree node traversal
    // In production, would properly deserialize B-tree nodes
    // For now, assume each node is 16KB and contains keys_per_node entries
    
    let node_size = 16384u; // 16KB
    let node_offset = node_idx * node_size;
    let keys_per_node = params.keys_per_node;
    
    // Process each key in the node
    for (var key_idx = 0u; key_idx < keys_per_node; key_idx++) {
        let key_offset = node_offset + (key_idx * 512u); // Simplified key offset
        
        // Check if key is in range
        var in_range = true;
        
        // Check start bound
        if (params.start_key_len > 0u) {
            let cmp = compare_keys(nodes, key_offset, 256u, start_key, 0u, params.start_key_len);
            if (params.include_start == 1u) {
                in_range = in_range && (cmp >= 0);
            } else {
                in_range = in_range && (cmp > 0);
            }
        }
        
        // Check end bound
        if (params.end_key_len > 0u && in_range) {
            let cmp = compare_keys(nodes, key_offset, 256u, end_key, 0u, params.end_key_len);
            if (params.include_end == 1u) {
                in_range = in_range && (cmp <= 0);
            } else {
                in_range = in_range && (cmp < 0);
            }
        }
        
        if (in_range) {
            // Add to results (atomic increment)
            let result_idx = atomicAdd(&result_count[0], 1u);
            if (result_idx < arrayLength(&result_keys)) {
                result_keys[result_idx] = key_offset;
                result_values[result_idx] = key_offset + 256u; // Value offset
            }
        }
    }
}
"#;

/// Binary hyperdimensional bind shader
/// Binds two binary hypervectors using XOR
pub const BINARY_HD_BIND_SHADER: &str = r#"
struct BindParams {
    num_words: u32,
    _padding: u32,
    _padding2: u32,
    _padding3: u32,
}

@group(0) @binding(0) var<storage, read> vector1: array<u32>;
@group(0) @binding(1) var<storage, read> vector2: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> params: BindParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_words) {
        return;
    }
    
    // Binary bind: XOR operation
    output[idx] = vector1[idx] ^ vector2[idx];
}
"#;

/// Binary hyperdimensional bundle shader
/// Bundles multiple binary hypervectors using per-bit majority voting
pub const BINARY_HD_BUNDLE_SHADER: &str = r#"
struct BundleParams {
    num_vectors: u32,
    num_words: u32,
    _padding: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<storage, read> vectors: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: BundleParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let word_idx = global_id.x;
    if (word_idx >= params.num_words) {
        return;
    }

    let threshold = params.num_vectors / 2u;
    var result_word: u32 = 0u;

    // Per-bit majority voting across all vectors
    for (var bit = 0u; bit < 32u; bit++) {
        var ones_count: u32 = 0u;
        let bit_mask: u32 = 1u << bit;

        for (var vec_idx = 0u; vec_idx < params.num_vectors; vec_idx++) {
            let offset = vec_idx * params.num_words + word_idx;
            if ((vectors[offset] & bit_mask) != 0u) {
                ones_count += 1u;
            }
        }

        if (ones_count > threshold) {
            result_word |= bit_mask;
        }
    }

    output[word_idx] = result_word;
}
"#;

/// Binary hyperdimensional similarity shader
/// Computes Hamming similarity between query and vectors using hardware popcount
pub const BINARY_HD_SIMILARITY_SHADER: &str = r#"
struct SimilarityParams {
    num_vectors: u32,
    num_words: u32,
    _padding: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<storage, read> query: array<u32>;
@group(0) @binding(1) var<storage, read> vectors: array<u32>;
@group(0) @binding(2) var<storage, read_write> similarities: array<u32>;
@group(0) @binding(3) var<uniform> params: SimilarityParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vec_idx = global_id.x;
    if (vec_idx >= params.num_vectors) {
        return;
    }

    var distance: u32 = 0u;

    for (var word_idx = 0u; word_idx < params.num_words; word_idx++) {
        let query_word = query[word_idx];
        let vector_word = vectors[vec_idx * params.num_words + word_idx];
        distance += countOneBits(query_word ^ vector_word);
    }

    // Store similarity (total bits - distance)
    similarities[vec_idx] = (params.num_words * 32u) - distance;
}
"#;

/// Binary hyperdimensional batch bind shader
/// Binds N pairs of binary hypervectors using XOR in a single dispatch
pub const BINARY_HD_BATCH_BIND_SHADER: &str = r#"
struct BatchBindParams {
    num_pairs: u32,
    num_words: u32,
    _padding: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<storage, read> vectors_a: array<u32>;
@group(0) @binding(1) var<storage, read> vectors_b: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> params: BatchBindParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_words = params.num_pairs * params.num_words;
    if (idx >= total_words) {
        return;
    }

    // XOR corresponding words across all pairs
    output[idx] = vectors_a[idx] ^ vectors_b[idx];
}
"#;

/// Binary hyperdimensional multi-query similarity shader
/// Computes Hamming similarity for M queries against N vectors in a single dispatch
pub const BINARY_HD_MULTI_SIMILARITY_SHADER: &str = r#"
struct MultiSimilarityParams {
    num_queries: u32,
    num_vectors: u32,
    num_words: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> queries: array<u32>;
@group(0) @binding(1) var<storage, read> vectors: array<u32>;
@group(0) @binding(2) var<storage, read_write> similarities: array<u32>;
@group(0) @binding(3) var<uniform> params: MultiSimilarityParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_idx = global_id.x;
    let total_pairs = params.num_queries * params.num_vectors;
    if (pair_idx >= total_pairs) {
        return;
    }

    let query_idx = pair_idx / params.num_vectors;
    let vec_idx = pair_idx % params.num_vectors;

    var distance: u32 = 0u;
    for (var w = 0u; w < params.num_words; w++) {
        let q = queries[query_idx * params.num_words + w];
        let v = vectors[vec_idx * params.num_words + w];
        distance += countOneBits(q ^ v);
    }

    // Store similarity (total bits - distance)
    similarities[pair_idx] = (params.num_words * 32u) - distance;
}
"#;

/// Columnar format conversion shader (B-tree row to columnar)
pub const COLUMNAR_CONVERSION_SHADER: &str = r#"
struct ConversionParams {
    num_rows: u32,
    num_columns: u32,
    row_size: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> btree_data: array<u8>;
@group(0) @binding(1) var<storage, read_write> columnar_data: array<u8>;
@group(0) @binding(2) var<uniform> params: ConversionParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_idx = global_id.x;
    if (row_idx >= params.num_rows) {
        return;
    }
    
    let row_offset = row_idx * params.row_size;
    
    // Convert row to columnar format
    // Each thread processes one row, writing to appropriate column positions
    for (var col = 0u; col < params.num_columns; col++) {
        let col_offset = col * params.num_rows * 4u; // Assuming 4 bytes per value
        let value_offset = row_offset + col * 4u;
        
        // Copy value to columnar position
        if (value_offset + 4u <= arrayLength(&btree_data)) {
            columnar_data[col_offset + row_idx * 4u] = btree_data[value_offset];
            columnar_data[col_offset + row_idx * 4u + 1u] = btree_data[value_offset + 1u];
            columnar_data[col_offset + row_idx * 4u + 2u] = btree_data[value_offset + 2u];
            columnar_data[col_offset + row_idx * 4u + 3u] = btree_data[value_offset + 3u];
        }
    }
}
"#;

/// Columnar aggregation shader (SUM, AVG, etc.)
pub const COLUMNAR_AGGREGATE_SHADER: &str = r#"
struct AggregateParams {
    num_rows: u32,
    column_idx: u32,
    agg_type: u32, // 0=SUM, 1=AVG, 2=MIN, 3=MAX, 4=COUNT
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> columnar_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: AggregateParams;

var<workgroup> shared_sum: array<f32, 256>;
var<workgroup> shared_count: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;
    
    // Load data to shared memory
    if (gid < params.num_rows) {
        let col_offset = params.column_idx * params.num_rows;
        shared_sum[tid] = columnar_data[col_offset + gid];
        shared_count[tid] = 1u;
    } else {
        shared_sum[tid] = 0.0;
        shared_count[tid] = 0u;
    }
    workgroupBarrier();
    
    // Parallel reduction
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
            shared_count[tid] = shared_count[tid] + shared_count[tid + stride];
        }
        workgroupBarrier();
    }
    
    // Write result
    if (tid == 0u) {
        if (params.agg_type == 0u) { // SUM
            output[workgroup_id.x] = shared_sum[0];
        } else if (params.agg_type == 1u) { // AVG
            output[workgroup_id.x] = shared_sum[0] / f32(shared_count[0]);
        } else if (params.agg_type == 4u) { // COUNT
            output[workgroup_id.x] = f32(shared_count[0]);
        }
    }
}
"#;

/// Ternary hyperdimensional dot product shader
/// Computes dot product between a query ternary vector and N stored vectors.
/// Trits are packed 16 per u32 using 2-bit encoding: 00=-1, 01=0, 10=+1.
/// Each thread handles one stored vector.
pub const TERNARY_HD_DOT_PRODUCT_SHADER: &str = r#"
struct TernaryDotParams {
    num_vectors: u32,
    num_words: u32,
    dimension: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> query: array<u32>;
@group(0) @binding(1) var<storage, read> vectors: array<u32>;
@group(0) @binding(2) var<storage, read_write> dot_products: array<i32>;
@group(0) @binding(3) var<uniform> params: TernaryDotParams;

fn decode_trit(bits: u32) -> i32 {
    return i32(bits) - 1;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vec_idx = global_id.x;
    if (vec_idx >= params.num_vectors) {
        return;
    }

    var dot: i32 = 0;

    for (var w = 0u; w < params.num_words; w++) {
        let q_word = query[w];
        let v_word = vectors[vec_idx * params.num_words + w];

        for (var t = 0u; t < 16u; t++) {
            let shift = t * 2u;
            let q_trit = decode_trit((q_word >> shift) & 3u);
            let v_trit = decode_trit((v_word >> shift) & 3u);
            dot += q_trit * v_trit;
        }
    }

    dot_products[vec_idx] = dot;
}
"#;

/// Ternary hyperdimensional bundle shader
/// Bundles N ternary vectors using per-dimension majority voting.
/// Trits packed 16 per u32 using 2-bit encoding: 00=-1, 01=0, 10=+1.
/// Each thread handles one packed word across all input vectors.
pub const TERNARY_HD_BUNDLE_SHADER: &str = r#"
struct TernaryBundleParams {
    num_vectors: u32,
    num_words: u32,
    _padding: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<storage, read> vectors: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: TernaryBundleParams;

fn decode_trit(bits: u32) -> i32 {
    return i32(bits) - 1;
}

fn encode_trit(val: i32) -> u32 {
    return u32(val + 1);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let word_idx = global_id.x;
    if (word_idx >= params.num_words) {
        return;
    }

    var result_word: u32 = 0u;

    for (var t = 0u; t < 16u; t++) {
        let shift = t * 2u;
        var accum: i32 = 0;

        for (var vec_idx = 0u; vec_idx < params.num_vectors; vec_idx++) {
            let offset = vec_idx * params.num_words + word_idx;
            let trit = decode_trit((vectors[offset] >> shift) & 3u);
            accum += trit;
        }

        var result_trit: i32 = 0;
        if (accum > 0) {
            result_trit = 1;
        } else if (accum < 0) {
            result_trit = -1;
        }

        result_word |= encode_trit(result_trit) << shift;
    }

    output[word_idx] = result_word;
}
"#;

/// Ternary hyperdimensional bind shader
/// Element-wise multiplication of two ternary vectors (bind operation).
/// Trits packed 16 per u32 using 2-bit encoding: 00=-1, 01=0, 10=+1.
pub const TERNARY_HD_BIND_SHADER: &str = r#"
struct TernaryBindParams {
    num_words: u32,
    _padding: u32,
    _padding2: u32,
    _padding3: u32,
}

@group(0) @binding(0) var<storage, read> vector1: array<u32>;
@group(0) @binding(1) var<storage, read> vector2: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> params: TernaryBindParams;

fn decode_trit(bits: u32) -> i32 {
    return i32(bits) - 1;
}

fn encode_trit(val: i32) -> u32 {
    return u32(val + 1);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let word_idx = global_id.x;
    if (word_idx >= params.num_words) {
        return;
    }

    let w1 = vector1[word_idx];
    let w2 = vector2[word_idx];
    var result: u32 = 0u;

    for (var t = 0u; t < 16u; t++) {
        let shift = t * 2u;
        let t1 = decode_trit((w1 >> shift) & 3u);
        let t2 = decode_trit((w2 >> shift) & 3u);
        result |= encode_trit(t1 * t2) << shift;
    }

    output[word_idx] = result;
}
"#;
