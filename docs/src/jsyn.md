# `.jsyn` (`JSYN-v1`) Experimental Low-Rank Bitplanes & Fidelity Boundaries

`.jsyn` (`crates/joule-db-weights`) is JouleDB's experimental low-bitwidth tensor container. It decomposes 2D weight projections via coupled binary sign bitplanes (`u_bits`, `v_bits`), `FP16` row/column/singular scales (`SVF`), and sparse outlier entries, while storing 1D normalization/bias vectors in exact `FP32`.

---

## Candid Quality vs. Compression Findings

Compression arithmetic alone (`20x–560x` smaller files) does not mean a compressed neural network behaves like the upstream model. Empirical quality depends strongly on architecture and bits-per-weight (`BPW`):

| Model Class | Representative Checkpoint | Avg `BPW` | Size (`FP16 -> .jsyn`) | Measured Fidelity vs Original | Practical Usability |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **6L Bidirectional Sentence Encoder** | [`all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit) (`Apache-2.0`) | **`0.4378`** | `43.32 MB -> 1.95 MB` (`22.2×`) | **`cos = 0.9441`** mean sentence-embedding cosine | **Usable** for compact CPU/edge semantic retrieval (`BM25 + RaBitQ + HDC`) |
| **Single Dense Projection (`BF16 q_proj`)** | Rank-sweep ablation (`rank = 8 -> 128`) | `0.028 -> 0.166` | `564× -> 96.6×` | Output `cos = 0.271 -> 0.455`, `KL = 3.87 -> 2.72` (`65.6%` Top-5) | **Degraded** — demonstrates Eckart-Young rank trade-off, not lossless LLM compression |
| **30L Autoregressive Causal LLM** | `SmolLM2-135M` rank sweep (`rank = 8 -> 128`) | `0.125 -> 0.388` | `65.2× -> 27.4×` | Mean weight `cos = 0.19 -> 0.62` | **Not a drop-in replacement** for the upstream instruct LLM |

> **November 2026 Hugging Face Cleanup:** 81 unverified ultra-low-BPW (`0.013–0.25 BPW`) and non-standard vendor-licensed `.jsyn` repositories were removed from `huggingface.co/dcharlot65-openie`. Only checkpoints with permissive licenses (`Apache-2.0` / `MIT`) and verified `>= 0.85` task fidelity (currently [`all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit)) are published publicly.
