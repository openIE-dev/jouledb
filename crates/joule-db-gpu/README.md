# joule-db-gpu

GPU compute backend for JouleDB using wgpu.

`joule-db-gpu` is the cross-platform GPU compute layer — wgpu underneath gives you WebGPU in browsers, Metal on Apple, Vulkan on Linux / Android, DirectX on Windows, and CUDA via translation. Column ops, B-tree serialization for GPU traversal, and HDC compute kernels all live here.

## Module map

| Module | Role |
|---|---|
| [`backend.rs`](src/backend.rs) | Top-level GPU backend — device init, command submission |
| [`shaders.rs`](src/shaders.rs) | Compiled WGSL shaders |
| [`btree_serialization.rs`](src/btree_serialization.rs) | Serialize B-tree pages into GPU-traversable layout |
| [`hdc_compute.rs`](src/hdc_compute.rs) | HDC primitives on GPU (bind, bundle, similarity) |

## Tests

30 tests in `src/`.

## See also

- [joule-db-core](../joule-db-core/) — the storage layer being read
- [joule-db-hdc](../joule-db-hdc/) — HDC primitives the GPU kernels accelerate
- [joule-db-browser](../joule-db-browser/) — uses this crate for the WebGPU path
- [joule-db-edge](../joule-db-edge/) — uses this crate for native GPU acceleration
