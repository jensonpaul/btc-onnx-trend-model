//! ONNX-backed implementation of [`TrendModelExt`](btc_prediction_engine::models::TrendModelExt).
//!
//! Loads a LightGBM (or any sklearn-compatible) classifier exported with
//! `skl2onnx` (`zipmap=False`, `target_opset=17`) — see
//! `btc-model-trainer/python/train_direction_model.py` for the training side.
//!
//! Lives as a **separate crate** from `btc_prediction_engine` deliberately:
//! the upstream engine is a `git` dependency tracked by branch, and keeping
//! this wrapper external means upgrading the engine never requires
//! re-merging a fork.
//!
//! ## Expected ONNX I/O
//!
//! | Name                  | Shape    | Dtype | Meaning                  |
//! |------------------------|----------|-------|--------------------------|
//! | `float_input` (input)  | `[1,17]` | f32   | normalised feature array |
//! | `output_label`         | `[1]`    | i64   | predicted class (0/1/2)  |
//! | `output_probability`   | `[1,3]`  | f32   | class probabilities      |
//!
//! Class encoding: `0 = Bearish, 1 = Sideways, 2 = Bullish` — must match the
//! `label` column produced by `btc-model-trainer`'s collector.

use std::path::Path;

use anyhow::{Context, Result};
use ndarray::Array2;
use tract_onnx::prelude::*;

use btc_prediction_engine::features::FeatureVector;
use btc_prediction_engine::models::TrendModelExt;
use btc_prediction_engine::types::{TimeScale, TrendDirection, TrendSignal};

/// Number of input features. Must match the column count produced by
/// `btc-model-trainer` and `feature_array()` below, in the same order.
const N_FEATURES: usize = 17;

type OnnxPlan = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// A loaded ONNX directional classifier.
///
/// `Send + Sync` (required by [`TrendModelExt`]) — `tract`'s `SimplePlan` is
/// immutable after construction and safe to share across threads.
pub struct OnnxTrendModel {
    plan: OnnxPlan,
}

impl OnnxTrendModel {
    /// Load an ONNX model from a file path.
    ///
    /// # Errors
    /// Returns an error if the file is missing, malformed, or fails to
    /// optimize/compile into a runnable plan. Fails fast at startup rather
    /// than at first inference.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let plan = tract_onnx::onnx()
            .model_for_path(path)
            .with_context(|| format!("loading ONNX model from {}", path.display()))?
            .into_optimized()
            .context("optimizing ONNX model")?
            .into_runnable()
            .context("compiling ONNX model into a runnable plan")?;
        Ok(Self { plan })
    }

    /// Load an ONNX model from raw bytes (e.g. `include_bytes!` for
    /// build-time embedding — recommended for production so the binary has
    /// no external file dependency).
    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self> {
        let plan = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(bytes))
            .context("loading ONNX model from embedded bytes")?
            .into_optimized()
            .context("optimizing ONNX model")?
            .into_runnable()
            .context("compiling ONNX model into a runnable plan")?;
        Ok(Self { plan })
    }

    /// Convert a [`FeatureVector`] into the normalised `[f32; 17]` array the
    /// model expects.
    ///
    /// **Must stay in sync** with `feature_array()` in
    /// `btc-model-trainer/src/main.rs` — a mismatch here is a silent
    /// train/serve skew bug. Both copies are documented to cross-reference
    /// each other; if you change one, change both.
    fn feature_array(f: &FeatureVector) -> [f32; N_FEATURES] {
        [
            (f.rsi_14.unwrap_or(50.0) / 100.0) as f32,
            f.vwap_deviation.unwrap_or(0.0) as f32,
            f.momentum_micro.unwrap_or(0.0) as f32,
            f.momentum_short.unwrap_or(0.0) as f32,
            f.ewma_vol_tick.unwrap_or(0.001) as f32,
            (f.tick_velocity / 20.0) as f32,
            f.ofi_30s as f32,
            f.ofi_300s as f32,
            f.autocorr_lag1.unwrap_or(0.0) as f32,
            f.realised_vol_30s.unwrap_or(0.001) as f32,
            (f.inter_exchange_spread / 100.0) as f32,
            ((f.price - 30_000.0) / 70_000.0) as f32,
            f.ewma_variance as f32,
            f.book_imbalance_top5.unwrap_or(0.0) as f32,
            f.book_imbalance_full.unwrap_or(0.0) as f32,
            f.book_weighted_mid
                .map(|m| (m - 30_000.0) / 70_000.0)
                .unwrap_or(0.0) as f32,
            f.book_spread_usd.map(|s| s / 100.0).unwrap_or(0.0) as f32,
        ]
    }

    /// Run inference, returning `(direction, confidence)`.
    ///
    /// On any internal error (shape mismatch, unexpected dtype), falls back
    /// to `(Sideways, 0.0)` rather than panicking — a single bad tick must
    /// never take down the prediction pipeline. Errors are only checked at
    /// `load()` time during development; this defensive fallback covers the
    /// remaining runtime risk in production.
    fn infer(&self, features: &FeatureVector) -> (TrendDirection, f64) {
        let arr = Self::feature_array(features);

        let input: Tensor = match Array2::from_shape_vec((1, N_FEATURES), arr.to_vec()) {
            Ok(a) => a.into(),
            Err(_) => return (TrendDirection::Sideways, 0.0),
        };

        let outputs = match self.plan.run(tvec![input.into()]) {
            Ok(o) => o,
            Err(_) => return (TrendDirection::Sideways, 0.0),
        };

        let label = match outputs.get(0).and_then(|t| t.to_array_view::<i64>().ok()) {
            Some(view) => match view.get(0) {
                Some(v) => *v,
                None => return (TrendDirection::Sideways, 0.0),
            },
            None => return (TrendDirection::Sideways, 0.0),
        };

        let confidence = outputs
            .get(1)
            .and_then(|t| t.to_array_view::<f32>().ok())
            .and_then(|probs| probs.get([0, label as usize]).copied())
            .map(|p| p as f64)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);

        let direction = match label {
            0 => TrendDirection::Bearish,
            2 => TrendDirection::Bullish,
            // 1 (Sideways) and any unexpected label both map to Sideways.
            _ => TrendDirection::Sideways,
        };

        (direction, confidence)
    }
}

impl TrendModelExt for OnnxTrendModel {
    /// Note: a single model predicts the same direction regardless of
    /// `scale`. If you've trained per-`TimeScale` models (recommended —
    /// see `btc-model-trainer` README), use [`PerScaleOnnxTrendModel`]
    /// instead, which dispatches to a different model per scale.
    fn predict(&self, features: &FeatureVector, scale: TimeScale) -> TrendSignal {
        let (direction, confidence) = self.infer(features);
        TrendSignal {
            direction,
            confidence,
            scale,
            computed_at: features.ts_micros,
        }
    }
}

/// Variant of [`OnnxTrendModel`] that dispatches to a different ONNX model
/// per [`TimeScale`], for when you've trained separate models per horizon
/// (e.g. `direction_model_micro.onnx`, `..._short.onnx`, etc. — see
/// `btc-model-trainer` README §"Multiple time scales").
///
/// All four scales must be provided; if you've only trained one model for
/// some scales, load the same [`OnnxTrendModel`] into multiple slots.
pub struct PerScaleOnnxTrendModel {
    pub micro:  OnnxTrendModel,
    pub short:  OnnxTrendModel,
    pub medium: OnnxTrendModel,
    pub broad:  OnnxTrendModel,
}

impl TrendModelExt for PerScaleOnnxTrendModel {
    fn predict(&self, features: &FeatureVector, scale: TimeScale) -> TrendSignal {
        let model = match scale {
            TimeScale::Micro => &self.micro,
            TimeScale::Short => &self.short,
            TimeScale::Medium => &self.medium,
            TimeScale::Broad => &self.broad,
        };
        model.predict(features, scale)
    }
}
