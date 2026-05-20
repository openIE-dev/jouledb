//! # The 15 Primitives of Hyperdimensional Computing
//!
//! First-class APIs for the irreducible operations of a contrast recognition runtime.
//! This is the HDC equivalent of PyTorch's primitive set.
//!
//! ## Already implemented elsewhere in the crate:
//! - **Encode**: `BinaryHV::from_data()`, `from_bytes()`, `from_embedding()`
//! - **Bind**: `BinaryHV::bind()`, `map_bind`, `fourier_bind`
//! - **Permute**: `BinaryHV::permute()`
//! - **Compare**: `BinaryHV::similarity()`, `hamming_distance()`
//! - **Remember**: `TurboHolographic::put()`, `SparseDistributedMemory::write()`
//! - **Merge/Un-merge**: `BundleAccumulator`, `BinaryResonator::factorize()`
//! - **Update**: `BinaryHV::bind_inplace()`, `bundle_inplace()`
//! - **Inhibit**: `BinaryCodebook::cleanup()` (winner-take-all)
//!
//! ## Formalized here as first-class primitives:
//! - **Route**: `route` — energy-proportional compute allocation
//! - **Forget**: `forget` — explicit decoherence / decay
//! - **Reflect**: `reflect` — apply system to its own output
//! - **Test**: `test` — verify representation integrity
//! - **Spawn**: `spawn` — create independent computational entity
//! - **Coarsen**: `coarsen` — scale-changing lossy projection
//! - **Synchronize**: `sync` — temporal coordination barrier

pub mod forget;
pub mod reflect;
pub mod spawn;
pub mod coarsen;
pub mod sync;

pub use forget::{Decay, DecayRate, Forgettable};
pub use reflect::{Reflectable, Reflection};
pub use spawn::{SpawnedEntity, Spawner};
pub use coarsen::{Coarsenable, CoarsenedView, CoarsenStrategy};
pub use sync::{Barrier, SyncGroup, Synchronizable};
