# `.jsyn` (`JSYN-v1`) Living Model Weights

A `.safetensors` or `.gguf` file is a frozen matrix slab that forces dense FFN parameters to act as a lossy key-value encyclopedia. **JouleDB** separates **stored facts** (MVCC SQL + `Engram` `O(1)` memory + `TriModal GraphRAG`) from **neural circuits** (`.jsyn` sub-1-bit bitplanes).

## Binary Layout & Factorization

Every `.jsyn` snapshot carries an IEEE `CRC-32` checksummed `JSYN` v1 header and `zstd` payload containing:
- **`littlebit_tensors`:** Coupled multi-scale binary low-rank factors `u_bits`, `v_bits` (`u64` FastLanes words), continuous `row_scale`/`col_scale`/`latent_scale`, sparse heavy-tailed `outliers`, and **Sakana `SVF`** adaptation vectors.
- **`vectors_1d`:** Exact `FP32` `RMSNorm`, `+1` Gemma offsets, `QK-Norm`, `Liquid ShortConv`, and `Mamba-2` SSM vectors (`0%` lossy compression).
- **`model_vocabs` + `model_configs` + `engram`:** Self-contained BPE/SentencePiece tokenizer, architecture graph, and `O(1)` conditional memory.

## Apple M5 Max Rank-Sweep Proof (`POST /api/v1/synapse/bench/rank_sweep`)

| Model / Layer | Rank | `BPW` | Shrink | Latency | UMA Energy | Quality (`cos` / `KL`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Gemma-4-E2B`** `q_proj` (`BF16`) | `8` | `0.028` | **`564.2×`** | `385 µs` | `49.8 nJ` (**`562×`**) | `0.271` / `3.87` |
| **`Gemma-4-E2B`** `q_proj` (`BF16`) | `32` | `0.056` | **`286.6×`** | `831 µs` | `108.5 nJ` (**`258×`**) | `0.341` / `3.69` |
| **`Gemma-4-E2B`** `q_proj` (`BF16`) | `64` | `0.092` | **`173.1×`** | `1,723 µs` | `186.7 nJ` (**`150×`**) | `0.392` / `3.37` |
| **`Gemma-4-E2B`** `q_proj` (`BF16`) | `128` | `0.166` | **`96.6×`** | `3,193 µs` | `343.2 nJ` (**`82×`**) | `0.455` / `2.72` |
| **`Inkling-2-Mini`** (`128`-Exp `NVFP4`) | `8 → 128` | `0.023–0.134` | **`701× → 120×`** | `466–4,156 µs` | `60–416 nJ` (**`699×`**) | `0.113 → 0.350` |
| **`SmolLM2-135M`** (`30L` Causal LLM) | `8 → 128` | `0.125–0.388` | **`65.2× → 27.4×`** | **`58.4–31.3 tok/s`** | `3.93–9.35 MB` | `w_cos 0.19 → 0.62` |

## 82 Public Models & Interactive Explorer

Browse all **82 adapted `< 30 GB` open-weight models** at **[huggingface.co/dcharlot65-openie](https://huggingface.co/dcharlot65-openie)**, explore interactive rank-sweep (`r = 8 → 32 → 64 → 128` + `AWQ ⊕ GPTQ SVF`) curves at **[jouledb.org/benchmarks](https://jouledb.org/benchmarks)**, and read the full binary specification in [`docs/JSYN-LIVING-WEIGHTS-SPEC.md`](https://github.com/openIE-dev/jouledb/blob/main/docs/JSYN-LIVING-WEIGHTS-SPEC.md).
