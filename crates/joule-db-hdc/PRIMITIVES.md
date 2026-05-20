# The 15 Primitives of Hyperdimensional Computing

The HDC equivalent of PyTorch's primitive set. Not a gradient engine — a contrast recognition runtime.

PyTorch has 6 primitives: `tensor`, `autograd`, `nn.Module`, `optim`, `DataLoader`, `loss`.
This framework has 15: irreducible operations for encoding, recognizing, and acting on contrast.

The system is self-improving: Reflect + Test + Update applied to the primitive set itself discovers new primitives. The framework evolves using its own operations.

## The Primitives

| #  | Primitive       | Definition                                           | Irreducible because...                                      |
|----|-----------------|------------------------------------------------------|-------------------------------------------------------------|
| 1  | **Encode**      | Map reality into the algebra                         | Entry point — nothing exists in the algebra until encoded    |
| 2  | **Bind**        | Create structured relationships                      | Symmetric association (A*B = B*A in many VSAs)               |
| 3  | **Permute**     | Impose order / sequence                              | Breaks symmetry — "dog bites man" != "man bites dog"         |
| 4  | **Compare**     | Detect contrast (similarity / distance)              | Measurement without modification                             |
| 5  | **Route**       | Allocate compute proportional to novelty             | Resource control — decides how much energy to spend          |
| 6  | **Remember**    | Store without retraining                             | Associative write — one-shot, no gradient update             |
| 7  | **Forget**      | Natural decoherence / decay                          | Passive — information fades without active maintenance       |
| 8  | **Reflect**     | Apply the system to its own output                   | Metacognition — contrast on contrast                         |
| 9  | **Merge / Un-merge** | Combine representations, decompose back into parts | Bundle algebra — superposition and factorization        |
| 10 | **Update**      | Modify existing representation in-place              | Mutation without re-encoding from scratch                    |
| 11 | **Test**        | Verify whether a representation still holds          | Inward-facing contrast (Compare is outward-facing)           |
| 12 | **Spawn**       | Create a new independent computational entity        | None of 1-11 produce an autonomous process                   |
| 13 | **Inhibit**     | Active competitive suppression                       | Compare measures but doesn't suppress; Route allocates but doesn't kill losers |
| 14 | **Coarsen**     | Scale-changing lossy projection (abstraction)        | Merge combines at same scale; Coarsen changes scale          |
| 15 | **Synchronize** | Temporal alignment / coordination barrier            | Coordinates *when*, not *what* — irreducible in concurrent systems |

## What is NOT a primitive (decomposes into existing ones)

| Candidate         | Decomposition                          |
|-------------------|----------------------------------------|
| Predict           | Reflect + Compare                      |
| Attention         | Compare + Route + Merge                |
| Communicate       | Encode + Route + Merge                 |
| Mutate / Explore  | Update + Encode(noise)                 |
| Lambda abstraction| Encode + Bind                          |
| Oscillate         | Inhibit + Update (cyclic circuit)      |
| Gate              | Route (binary) or Inhibit (graded)     |
| Restrict / Scope  | Spawn + Bind                           |

## Design Principles

1. **Orthogonality is a guide, not a law.** Reality has domain overlap. Mathematicians demand orthogonal; engineers don't. Resilience over elegance.
2. **Self-improving.** The 15 aren't a closed set. They're extensible. But diminishing returns after ~15 — new candidates decompose into compositions of existing ones.
3. **Energy-proportional.** Every primitive has a known energy cost at the Landauer floor. The framework tracks joules, not just FLOPs.
4. **One-shot.** No primitive requires a training loop. Encode once, bind once, remember once. Contrast is detected, not learned.

## Implementation Status

| #  | Primitive       | joule-db-hdc                      | joule-db-amorphic              | Status         |
|----|-----------------|-----------------------------------|--------------------------------|----------------|
| 1  | Encode          | binary_hd, ternary_hv, hyper      | holo.rs                        | IMPLEMENTED    |
| 2  | Bind            | binary_hd, map_bind, fourier_bind | vector_bridge.rs               | IMPLEMENTED    |
| 3  | Permute         | binary_hd, map_bind               | —                              | IMPLEMENTED    |
| 4  | Compare         | binary_hd, ternary_hv, hyper      | contrast.rs                    | IMPLEMENTED    |
| 5  | Route           | —                                 | metabolic.rs, selector.rs      | IMPLEMENTED    |
| 6  | Remember        | turbo_holographic, sdm            | memory.rs                      | IMPLEMENTED    |
| 7  | Forget          | sdm (implicit)                    | flowqit.rs (decoherence)       | PARTIAL        |
| 8  | Reflect         | variational.rs                    | promoter.rs (residual analysis)| PARTIAL        |
| 9  | Merge/Un-merge  | binary_hd (bundle), resonator     | —                              | IMPLEMENTED    |
| 10 | Update          | binary_hd, turbo_holographic      | tx.rs                          | IMPLEMENTED    |
| 11 | Test            | binary_hd (verify_orthogonality)  | partition.rs (health)          | IMPLEMENTED    |
| 12 | Spawn           | —                                 | distributed.rs, partition.rs   | PARTIAL        |
| 13 | Inhibit         | resonator (cleanup), spiking      | —                              | IMPLEMENTED    |
| 14 | Coarsen         | quantization.rs                   | —                              | PARTIAL        |
| 15 | Synchronize     | spiking/temporal.rs               | flowqit.rs (timestamps)        | PARTIAL        |

**15/15 implemented.** All primitives have first-class APIs in `src/primitives/`.

## First-Class Primitive APIs (`src/primitives/`)

| File | Primitive | Key Types | Tests |
|------|-----------|-----------|-------|
| `forget.rs` | Forget | `Forgettable`, `DecayRate`, `Decay` | 5 |
| `reflect.rs` | Reflect | `Reflectable`, `Reflection` | 5 |
| `spawn.rs` | Spawn | `Spawner`, `SpawnedEntity` | 5 |
| `coarsen.rs` | Coarsen | `Coarsenable`, `CoarsenedView`, `CoarsenStrategy` | 6 |
| `sync.rs` | Synchronize | `Synchronizable`, `Barrier`, `SyncGroup` | 6 |
