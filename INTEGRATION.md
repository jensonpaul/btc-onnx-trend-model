# Phase 3 — Wiring `OnnxTrendModel` into `polymarket-trading-terminal`

## 1. Add dependencies to `Cargo.toml`

```toml
[dependencies]
# ... existing deps ...
btc-onnx-trend-model = { path = "../btc-onnx-trend-model" }
# (or `git = "..."` once this crate has its own repo)
```

No change needed to the existing `btc_prediction_engine` git dependency —
`btc-onnx-trend-model` depends on the same engine version itself, and Cargo
will unify the two as long as the `branch`/version matches.

## 2. Place the trained model file

Copy the `.onnx` file produced by Phase 2
(`btc-model-trainer/python/model/direction_model.onnx`) into the terminal
repo, e.g.:

```
polymarket-trading-terminal/
└── models/
    └── direction_model.onnx
```

Add `models/*.onnx` to `.gitignore` if the model is large or retrained
frequently — or commit it if you want reproducible builds tied to a specific
model version. For the embedded (`include_bytes!`) approach in step 4, the
file must be present at compile time regardless.

## 3. Edit `src/prediction/service.rs`

### Import

```rust
use btc_onnx_trend_model::OnnxTrendModel;
use btc_prediction_engine::pipeline::PipelineConfig;
```

### Replace `EngineConfig::default()`

**Before:**

```rust
        let (btc_engine, _handles) =
            btc_prediction_engine::prelude::PredictionEngine::start(
                EngineConfig::default(),
            )
            .await;
```

**After:**

```rust
        // Load the trained directional classifier. This is a startup-time
        // failure (`expect`) by design: running with the heuristic fallback
        // silently would mask a deployment mistake (missing/corrupt model
        // file). If you want graceful degradation instead, change this to
        // log the error and fall through to `EngineConfig::default()`.
        let onnx_model = OnnxTrendModel::load("models/direction_model.onnx")
            .expect(
                "failed to load models/direction_model.onnx — \
                 run btc-model-trainer Phases 1-2 first, or set \
                 ext_trend to None to use the heuristic classifier",
            );

        let engine_config = EngineConfig {
            pipeline: PipelineConfig {
                ext_trend: Some(Box::new(onnx_model)),
                ..PipelineConfig::default()
            },
            ..EngineConfig::default()
        };

        let (btc_engine, _handles) =
            btc_prediction_engine::prelude::PredictionEngine::start(
                engine_config,
            )
            .await;
```

Everything below this point in `service.rs` (`add_feed`, `add_book_feed`,
`subscribe`, etc.) is unchanged — `stage_models` picks up `ext_trend`
automatically and calls `OnnxTrendModel::predict` for all four `TimeScale`
variants via `tokio::join!`.

## 4. (Recommended for production) Embed the model at compile time

Avoids a missing-file failure mode in deployed binaries. In `service.rs`:

```rust
const MODEL_BYTES: &[u8] = include_bytes!("../../models/direction_model.onnx");

// ... and in PredictionService::run():
let onnx_model = OnnxTrendModel::load_from_bytes(MODEL_BYTES)
    .expect("embedded direction_model.onnx failed to load — rebuild after retraining");
```

Trade-off: every model retrain now requires a rebuild + redeploy of the
terminal binary, rather than just dropping a new file next to it. Choose
based on how often you retrain vs. how strict your deployment process is.

## 5. Sanity-check before going live

1. `cargo build` — confirms `tract-onnx` compiles cleanly (pure Rust, no
   system deps, but it's a sizeable crate — first build will be slow).
2. Run the terminal against live feeds for a few minutes and log
   `PredictionSnapshot.short.confidence` — if it's always exactly `0.0`,
   `infer()` is hitting its fallback path (check `feature_array` shapes /
   ONNX output names match the table in `btc-onnx-trend-model/src/lib.rs`).
3. Compare `fused_direction` distribution against the heuristic baseline
   (`EngineConfig::default()` with `ext_trend: None`) over the same time
   window — the new model should differ meaningfully, not just rubber-stamp
   the heuristic.

## ⚠️ Pre-existing bug worth flagging separately

While reviewing `external_btc.rs`, I noticed the direction → side mapping
looks inverted:

```rust
let target_side = match pred.fused_direction {
    Direction::Bullish => PredictionSide::Down,
    Direction::Bearish => PredictionSide::Up,
    _ => return None,
};
```

A `Bullish` BTC signal mapping to a `Down` Polymarket side (and vice versa)
seems backwards — but this is **independent of the ONNX integration** and
existed before this change. Worth a second look, but I haven't touched it
here since it's out of scope for "plug in a trained model" and changing it
silently alongside a model swap would make it hard to attribute any P&L
shift to the right cause.
