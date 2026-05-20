# SigQL: Signal Query Language

> Query physical reality like a database.

SigQL is a domain-specific language for storing, querying, and processing **signals** (time-series data) with the same ease as SQL queries relational data.

## Key Concepts

1.  **First-Class Signals**: Data is not just rows of floats; it's a signal with sampling rate, uncertainty, and physical units.
2.  **Frequency Domain**: `TRANSFORM` clauses allow querying in the frequency domain (FFT, Power Spectrum) as easily as the time domain.
3.  **Uncertainty**: Every value tracks its confidence interval.

## Query Syntax

```sql
FROM source_signal
[TRANSFORM operation(args)...]
[WINDOW window_spec]
[AGGREGATE { name: op(args) }]
[RETURNING options]
```

### Examples

**1. Basic Filtering (Time Domain)**
```sql
FROM sensor.accelerometer
WHERE value > 9.8  -- Filter high-G events
WINDOW sliding(1s)
RETURNING value
```

**2. Frequency Analysis (The "Killer Feature")**
Analyze the power in a specific frequency band (4-12Hz) over time.

```sql
FROM controller.imu
TRANSFORM 
    bandpass(4Hz, 12Hz),  -- Apply physical filter
    fft()                 -- Switch to frequency domain
WINDOW sliding(2s, 500ms) -- 2s window, stepping every 500ms
AGGREGATE { 
    alpha_power: band_power(8Hz..12Hz) 
}
RETURNING confidence(0.95)
```

**3. Cross-Signal Correlation**
```sql
FROM eeg.frontal
CORRELATE WITH eeg.parietal USING coherence
RETURNING table
```

## Architecture

SigQL compiles to an **Execution Plan** that can run on:
*   **CPU** (SIMD-accelerated)
*   **GPU** (WebGPU / CUDA) - *Experimental*
*   **DSP** (Digital Signal Processors)

## Rust Usage

```rust
use sigql::prelude::*;

let query = sigql::parse("
    FROM telemetry.voltage 
    TRANSFORM lowpass(60Hz)
")?;

let plan = sigql::compile(&query, Target::Simd)?;
let result = runtime.execute(&plan)?;
```
