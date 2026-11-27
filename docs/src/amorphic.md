# Amorphous Continuum, `MorphRule` DSL & 8 Fluid Lenses

Un-siloing the digital world requires **maintaining exact semantic meaning while ditching arbitrary rigid structure**.

Traditional databases force a false choice: rigid relational DDL (requiring brittle ETL pipelines whenever schemas drift) or untyped JSON blobs (losing units, identity, and algebraic queryability). `JouleDB v0.16.0` (`crates/joule-db-amorphic` & `crates/joule-db-viz`) explores an experimental alternative: the **Amorphous Semantic Continuum**.

---

## 1. Zero-DDL Semantic Extraction & `10,240`-Bit Kanerva VSA Holograms

Raw payloads from JSON APIs, CSV exports, nested XML telemetry, `k=v` traces, unstructured prose logs, and RDF N-Triples dissolve into typed `SemanticAtom`s and superimpose into a **`10,240`-bit (`160 × u64`) Hyperdimensional Hologram**:

- **Algebraic Role Unbinding (`H ⊗ ROLE⁻¹`):** Query any concept across heterogeneous records in $O(1)$ `SIMD POPCNT` (`100%` Top-1 unbinding accuracy) even when silos share **`0` field names** (`customer_ref` vs `subject_uri` vs prose mention).
- **Exact Scientific & Financial SI Harmonization (`0.00%` Skew):**
  - Temperature: `25°C == 77°F == 298.15 K` (`0.000 K` error)
  - Photonics / Length: `532 nm == 0.532 µm == 5.32e-7 m`
  - Mass & Currency: `500 mg == 0.5 g`; `\$14,250.00 == 1,425,000 cents`.

---

## 2. Programmable `MorphRule` DSL

Domain engineers can declare custom harmonization, unit conversion, logarithmic scaling, and cross-entity linking rules without altering storage schemas:

```text
MAP "temp_celsius" -> Quantity(Kelvin) USING offset_kelvin
MAP "wavelength_nm" -> Quantity(Meters) USING scale(1e-9)
DERIVE "p_intensity" FROM "flux_ratio" USING neg_log10
EQUATE "sensor_uid" == "device_serial"
LINK WHEN entity.kelvin > 310.0 -> ALERT "thermal_excursion"
```

---

## 3. Bounded-Memory $O(E)$ Multi-Gigabyte Streaming (`StreamingAmorphicContinuum`)

`jouledb amorphic stream-corpus` ingests multi-gigabyte heterogeneous directories in strictly bounded $O(E)$ RAM using 64 KiB streaming readers, reservoir sampling, and 16-bit SimHash LSH hologram buckets:

- **Single-Node Streaming Run (`Apple M5 Max`):** **`1.35 GB` (`47,488,952` semantic atoms across `.jsonl`, `.csv`, `.xml`, `.kv`, `.log`, `.nt`)** streamed in **`57.64 MB` OS Peak RSS**.
- **Cross-OS Bit-Exact Determinism:** Produces identical 128-bit FNV-1a stream fingerprints (`de6330b34ee42a9985a74261f86b431b`) across `macOS`, `Linux`, and `Windows 11`.

---

## 4. 8 Observer-Intent Fluid Visual Lenses

The same underlying continuum morphs instantaneously into **8 visual lenses** without copying data (try it live at **[jouledb.org/amorphic](https://jouledb.org/amorphic)**):

1. `EntityConstellation` (Force-directed cross-silo identity graph)
2. `TemporalRibbon` (Causal microsecond event timeline)
3. `SpatialManifold` (GeoJSON coordinate projection)
4. `MetricHorizon` (Normalized SI distribution bars)
5. `SynthesizedRelationalGrid` (On-the-fly SQL-like tabular projection)
6. `NarrativeBrief` (Grounded executive synthesis with provenance)
7. `ParetoScatter` (`CustomLensCompiler` 2D multi-objective Pareto frontier)
8. `SankeyLineage` (Source-format $\to$ Canonical-concept dissolution flow).
