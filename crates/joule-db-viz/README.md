# joule-db-viz

Database-native visualization for JouleDB.

`joule-db-viz` turns query results into chart hints, Vega-Lite specs, accessibility metadata, and natural-language summaries — without leaving the database. The query path emits `VizHint`s alongside the result rows; clients can render directly or pass the spec to any Vega-compatible renderer.

## Capabilities

- **Inference** — automatically pick the right chart type from query metadata + result statistics ([`VizInferencer`](src/infer/))
- **Rendering** — Vega-Lite JSON, text summaries, accessibility hints (alt-text, ARIA)
- **Types** — serializable hint types (`ChartType`, `SemanticType`, `AxisMapping`) that travel with query responses

## Module map

| Module | Role |
|---|---|
| [`hint.rs`](src/hint.rs) | Hint types — `VizHint`, `ChartType`, `SemanticType`, `AxisMapping`, `EnergyEfficiency`, `EnergyOverlay`, `AccessibilityHint`, `SonificationHint` |
| [`infer/`](src/infer/) | Inference engine — picks chart type from data shape |
| [`column_classifier.rs`](src/column_classifier.rs) | Classify columns (categorical / quantitative / temporal / spatial) |
| [`data_profile.rs`](src/data_profile.rs) | Statistical profile of a column for inference |
| [`render/`](src/render/) | Renderers — Vega-Lite, text summary, accessibility |
| [`error.rs`](src/error.rs) | `VizError` / `VizResult` |

## Feature flags

| Feature | Default | Purpose |
|---|---|---|
| `infer` | yes | Inference engine |
| `vega` | yes | Vega-Lite JSON renderer |
| `text-summary` | yes | Natural-language summaries |
| `accessibility` | yes | Alt-text + ARIA output |
| `sonification` | no | Auditory data mapping |
| `gpu` | no | GPU renderer stub (browser) |

## Server integration

Behind the `viz` feature flag of [`joule-db-server`](../joule-db-server/), every query response carries a `VizHint` derived from the result schema.

## See also

- [joule-db-server](../joule-db-server/) — the integration point
- [joule-db-browser/viz](../joule-db-browser/src/viz/) — browser-side renderer
- [joule-db-query](../joule-db-query/) — produces the metadata `VizInferencer` consumes
