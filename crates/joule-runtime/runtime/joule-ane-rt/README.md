# joule-ane-rt

Apple Neural Engine (ANE) lane for JouleDB HDC similarity, through Core ML.
Core ML is the only sanctioned way onto the ANE. There is no public direct
ANE API, and this crate does not use the private `AppleNeuralEngine` or
Espresso frameworks.

## What it computes

`scores = Q · Mᵀ`, where `Q` is `n × d` and `M` is `k × d`, both bipolar
(+1/−1) and stored as FP16. The Hamming distance is `(d − score) / 2`.
Binary hypervectors are mapped bit 0 → +1 and bit 1 → −1. XOR/popcount is not
expressed, because it is not eligible for the ANE.

FP16 is exact here. When `d ≤ 2048`, every partial sum is an integer with
magnitude at most 2048, and FP16 represents those exactly. A larger `d` is
split into 2048-wide tiles, and the tile scores are summed in `i32` on the
host.

## How the model is produced

The model is generated at runtime in pure Rust (`src/model.rs`, `src/pb.rs`).
There is no Python, no coremltools, and no checked-in model artifact. For
each `(layout, n, d-tile, k, codebook)` the crate writes an `.mlpackage`:

```
Manifest.json
Data/com.apple.CoreML/model.mlmodel       CoreML.Specification.Model, specificationVersion 7, mlProgram, opset CoreML6
Data/com.apple.CoreML/weights/weight.bin  MIL blob storage v2, one FP16 blob (the codebook)
```

The field numbers come from Apple's published `Model.proto`,
`FeatureTypes.proto` and `MIL.proto` (coremltools `mlmodel/format`). The crate
then calls `MLModel compileModelAtURL` and caches the `.mlmodelc` under
`$TMPDIR/joule-ane-rt/`, keyed by a fingerprint. It loads the model with
`MLComputeUnits::CPUAndNeuralEngine` and predicts with Float16
`MLMultiArray`s.

There are two layouts:

- `Layout::MatMul`: `matmul(q[n,d], codebook[k,d], transpose_y=true)`.
- `Layout::ChannelsFirstConv`: the Apple ml-ane-transformers layout. The query
  is `(1, d, 1, n)` (B, C, 1, S), and a 1×1 `conv` uses weight `(k, d, 1, 1)`.

`hdc_similarity_auto` tries `MatMul` first. If `MLComputePlan` does not place
that op on the Neural Engine, it then tries `ChannelsFirstConv`.

## Placement

`MLComputePlan loadContentsOfURL:configuration:completionHandler:` is read for
the compiled model. For each ML Program operation, `Placement` holds the
preferred device, the supported devices and the estimated cost:
`MLNeuralEngineComputeDevice`, `MLGPUComputeDevice` or `MLCPUComputeDevice`.
Callers must decide whether the ANE ran from `Placement`, not from
assumptions.

Measured on an M5 Max running macOS 27.0.1 (26A434), with
`CPUAndNeuralEngine` and `d = 2048`:

- The op is planned on the CPU for small jobs: `n×k ≤ 64×256`.
- It is planned on the Neural Engine from `n=64, k=1024` upward.
- The CPU and Neural Engine are listed as supported for both layouts.

See `examples/ane_placement_probe.rs`.

## Energy

No public Core ML API returns joules. When `sudo -n true` succeeds (it never
prompts and never stores a password), the crate runs
`powermetrics --samplers cpu_power` while the prediction loops, and reads the
`ANE Power` line. On macOS 27, `--samplers ane_power` prints only headers.
Otherwise there is no measurement, and the fabric reports
`energy_source=estimate`.

## Verifying the generated artifact without a Mac

```
cargo run -p joule-ane-rt --example write_hdc_mlpackage -- /tmp/hdc.mlpackage matmul 4 1024 16
python3 scripts/verify_mlpackage.py /tmp/hdc.mlpackage matmul 4 1024 16   # needs `pip install coremltools`
```

The script parses the spec with coremltools' own protobuf classes and loads the
program through coremltools' milproto frontend, including the FP16 weight
blob.

On non-macOS targets every entry point returns `AneError::NotAvailable`.
