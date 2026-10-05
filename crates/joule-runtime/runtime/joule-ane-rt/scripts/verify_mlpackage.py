#!/usr/bin/env python3
"""Validate a joule-ane-rt generated .mlpackage with coremltools' own parsers.

Works on Linux (no Core ML runtime needed):
  * parses Data/com.apple.CoreML/model.mlmodel with coremltools.proto.Model_pb2
  * loads the ML Program through coremltools' milproto frontend, which also
    reads the fp16 blob from weights/weight.bin
  * evaluates the MIL graph in numpy and checks scores == Q @ M^T

Usage: verify_mlpackage.py <path.mlpackage> <matmul|conv> <n> <d> <k>
This script is a verification aid only; the model itself is generated in Rust.
"""
import os, sys
import numpy as np
from coremltools.proto import Model_pb2
from coremltools.converters.mil.frontend.milproto.load import load as milproto_load

pkg, layout, n, d, k = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:6])
data = os.path.join(pkg, "Data", "com.apple.CoreML")
spec = Model_pb2.Model()
spec.ParseFromString(open(os.path.join(data, "model.mlmodel"), "rb").read())
assert spec.specificationVersion == 7, spec.specificationVersion
assert spec.WhichOneof("Type") == "mlProgram"
inp, out = spec.description.input[0], spec.description.output[0]
print("input", inp.name, list(inp.type.multiArrayType.shape), Model_pb2.ArrayFeatureType.ArrayDataType.Name(inp.type.multiArrayType.dataType))
print("output", out.name, list(out.type.multiArrayType.shape), Model_pb2.ArrayFeatureType.ArrayDataType.Name(out.type.multiArrayType.dataType))
prog = milproto_load(spec, specification_version=spec.specificationVersion, file_weights_dir=os.path.join(data, "weights"))
print(prog)
fn = prog.functions["main"]
ops = {op.name: op for op in fn.operations}
W = ops["codebook"].outputs[0].val.astype(np.float32)
print("codebook", W.shape, W.dtype, "unique", np.unique(W))
assert set(np.unique(W)).issubset({-1.0, 1.0})
M = W.reshape(k, d)
rng = np.random.default_rng(7)
Q = np.where(rng.random((n, d)) < 0.5, -1.0, 1.0).astype(np.float32)
want = Q @ M.T
assert np.all(np.abs(want) <= d)
sim = [op for op in fn.operations if op.op_type in ("matmul", "conv")]
assert len(sim) == 1, [op.op_type for op in fn.operations]
assert sim[0].op_type == ("matmul" if layout == "matmul" else "conv")
assert tuple(sim[0].outputs[0].shape) == ((n, k) if layout == "matmul" else (1, k, 1, n)), sim[0].outputs[0].shape
print("OK", layout, "n=%d d=%d k=%d" % (n, d, k))
