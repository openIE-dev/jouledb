# `.jsyn` (`JSYN-v1`) Living Model Weights

A `.safetensors` or `.gguf` file is a frozen matrix slab that forces dense FFN parameters to act as a lossy key-value encyclopedia. **JouleDB** separates **stored facts** (MVCC SQL + `Engram` `O(1)` memory + `TriModal GraphRAG`) from **neural circuits** (`.jsyn` sub-1-bit bitplanes).

## Binary Layout & Factorization

Every `.jsyn` snapshot carries an IEEE `CRC-32` checksummed `JSYN` v1 header and `zstd` payload containing:
- **`littlebit_tensors`:** Coupled multi-scale binary low-rank factors `u_bits`, `v_bits` (`u64` FastLanes words), continuous `row_scale`/`col_scale`/`latent_scale`, sparse heavy-tailed `outliers`, and **Sakana `SVF`** adaptation vectors.
- **`vectors_1d`:** Exact `FP32` `RMSNorm`, `+1` Gemma offsets, `QK-Norm`, `Liquid ShortConv`, and `Mamba-2` SSM vectors (`0%` lossy compression).
- **`model_vocabs` + `model_configs` + `engram`:** Self-contained BPE/SentencePiece tokenizer, architecture graph, and `O(1)` conditional memory.

## 82 Public Models & Rank-Sweep Proof

Browse all **82 adapted `< 30 GB` open-weight models** at **[huggingface.co/dcharlot65-openie](https://huggingface.co/dcharlot65-openie)**, explore interactive rank-sweep (`r = 8 -> 32 -> 64 -> 128` + `AWQ ⊕ GPTQ SVF`) curves at **[jouledb.org/benchmarks](https://jouledb.org/benchmarks)**, and read the full binary specification in [`docs/JSYN-LIVING-WEIGHTS-SPEC.md`](https://github.com/openIE-dev/jouledb/blob/main/docs/JSYN-LIVING-WEIGHTS-SPEC.md).
