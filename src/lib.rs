//! ONNX-backed implementation of [`TrendModelExt`](btc_prediction_engine::models::TrendModelExt).
//!
//! Loads a LightGBM (or any sklearn-compatible) classifier exported with
//! `skl2onnx` (`zipmap=False`, `target_opset=17`) -- see
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
//! | `float_input` (input)  | `[1,28]` | f32   | normalised feature array |
//! | `output_label`         | `[1]`    | i64   | predicted class (0/1/2)  |
//! | `output_probability`   | `[1,3]`  | f32   | class probabilities      |
//!
//! Class encoding: `0 = Bearish, 1 = Sideways, 2 = Bullish` -- must match the
//! `label` column produced by `btc-model-trainer`'s collector.
//!
//! ## Feature ordering (must stay in lock-step with `btc-model-trainer`'s `FEATURE_NAMES`)
//!
//! **A mismatch here is a silent train/serve skew bug.** The two copies
//! (`feature_array` below and `feature_array` in `btc-model-trainer/src/main.rs`)
//! are cross-referenced in their doc comments; changing one requires changing both.
//!
//! ```text
//!  1  return_5s        -- log-return over last 5 s
//!  2  return_30s       -- log-return over last 30 s
//!  3  return_300s      -- log-return over last 300 s
//!  4  vol_30s          -- realised vol (std of log-returns), 30 s
//!  5  vol_300s         -- realised vol, 300 s
//!  6  vol_1800s        -- realised vol, 1800 s (regime reference)
//!  7  vol_ratio        -- vol_30s / vol_1800s
//!  8  ofi_5s           -- order-flow imbalance, 5 s  in [-1, 1]
//!  9  ofi_30s          -- order-flow imbalance, 30 s  in [-1, 1]
//! 10  ofi_300s         -- order-flow imbalance, 300 s in [-1, 1]
//! 11  ofi_delta_30s    -- ofi_5s - ofi_30s
//! 12  buy_ratio_30s    -- buy_vol / total_vol, 30 s  in [0, 1]
//! 13  buy_ratio_300s   -- buy_vol / total_vol, 300 s in [0, 1]
//! 14  vwap_dev_30s     -- (price - VWAP_30s) / VWAP_30s / vol_1800s
//! 15  vwap_dev_300s    -- (price - VWAP_300s) / VWAP_300s / vol_1800s
//! 16  volume_ratio     -- volume_30s / volume_300s
//! 17  tick_velocity    -- rolling tick rate 30 s, divided by 20 (ticks/s / 20)
//! 18  activity_regime  -- tick_rate_30s / tick_rate_1800s
//! 19  spread_pct       -- cross-exchange (max-min) / mid
//! 20  book_imb5        -- top-5 bid/ask volume imbalance in [-1, 1]
//! 21  book_imb_full    -- full-depth bid/ask imbalance in [-1, 1]
//! 22  book_spread_pct  -- best bid-ask spread / mid
//! 23  book_pressure    -- micro-price (weighted_mid) deviation from mid
//! 24  trend_strength   -- abs(return_300s) / vol_1800s
//! 25  vol_regime       -- vol_300s / vol_1800s
//! 26  zreturn_30s      -- return_30s / vol_1800s
//! 27  zreturn_300s     -- return_300s / vol_1800s
//! ```

use std::path::Path;

use anyhow::{Context, Result};
use ndarray::Array2;
use tract_onnx::prelude::*;

use btc_prediction_engine::features::FeatureVector;
use btc_prediction_engine::models::TrendModelExt;
use btc_prediction_engine::types::{TimeScale, TrendDirection, TrendSignal};

/// Number of input features. Must match the column count produced by
/// `btc-model-trainer` and `feature_array()` below, in the same order.
const N_FEATURES: usize = 28;

type OnnxPlan = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// A loaded ONNX directional classifier.
///
/// `Send + Sync` (required by [`TrendModelExt`]) -- `tract`'s `SimplePlan` is
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
            .with_input_fact(
		0,
		InferenceFact::dt_shape(f32::datum_type(), tvec![1usize, 28usize]),
	    )
	    .context("setting input fact")?
            .into_optimized()
            .context("optimizing ONNX model")?
            .into_runnable()
            .context("compiling ONNX model into a runnable plan")?;
        Ok(Self { plan })
    }

    /// Load an ONNX model from raw bytes (e.g. `include_bytes!` for
    /// build-time embedding -- recommended for production so the binary has
    /// no external file dependency).
    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self> {
        let plan = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(bytes))
            .context("loading ONNX model from embedded bytes")?
            .with_input_fact(
		0,
		InferenceFact::dt_shape(f32::datum_type(), tvec![1usize, 28usize]),
	    )
	    .context("setting input fact")?
            .into_optimized()
            .context("optimizing ONNX model")?
            .into_runnable()
            .context("compiling ONNX model into a runnable plan")?;
        Ok(Self { plan })
    }

    /// Convert a [`FeatureVector`] into the normalised `[f32; N_FEATURES]` array
    /// the model expects.
    ///
    /// **Must stay in sync** with `feature_array()` in
    /// `btc-model-trainer/src/main.rs` -- a mismatch is a silent
    /// train/serve skew bug. Both copies cross-reference each other; if you
    /// change one, change both and bump the model version string.
    ///
    /// Default values during the accumulator warm-up period:
    /// - Signed features (returns, OFI, VWAP deviation, book metrics) -> 0.0
    /// - Volatility features -> small positive floor (0.001) to avoid
    ///   divide-by-zero in downstream z-score computations.
    /// - Ratio features (vol_ratio, volume_ratio, vol_regime, activity_regime)
    ///   -> 1.0 (neutral: short window == long window).
    /// - buy_ratio -> 0.5 (balanced flow).
    fn feature_array(f: &FeatureVector) -> [f32; N_FEATURES] {
        const VOL_FLOOR: f32 = 0.001;

        [
            // Returns -- time-horizon aligned, log-space.
            f.return_5s.unwrap_or(0.0)   as f32,   //  1
            f.return_30s.unwrap_or(0.0)  as f32,   //  2
            f.return_300s.unwrap_or(0.0) as f32,   //  3

            // Volatility -- three horizons + expansion ratio.
            f.vol_30s.unwrap_or(VOL_FLOOR as f64)   as f32,  //  4
            f.vol_300s.unwrap_or(VOL_FLOOR as f64)  as f32,  //  5
            f.vol_1800s.unwrap_or(VOL_FLOOR as f64) as f32,  //  6
            f.vol_ratio.unwrap_or(1.0)               as f32,  //  7

            // Order flow imbalance.
            f.ofi_5s          as f32,   //  8
            f.ofi_30s         as f32,   //  9
            f.ofi_300s        as f32,   // 10
            f.ofi_delta_30s   as f32,   // 11
            f.buy_ratio_30s   as f32,   // 12
            f.buy_ratio_300s  as f32,   // 13

            // VWAP deviation -- already z-scored by vol_1800s.
            f.vwap_dev_30s.unwrap_or(0.0)  as f32,  // 14
            f.vwap_dev_300s.unwrap_or(0.0) as f32,  // 15

            // Volume activity ratio.
            f.volume_ratio.unwrap_or(1.0) as f32,   // 16

            // Tick activity -- soft-clipped and dimensionless.
            (f.tick_velocity / 20.0) as f32,        // 17
            f.activity_regime        as f32,         // 18

            // Cross-exchange spread (already a percentage).
            f.spread_pct as f32,                     // 19

            // Order book features.
            f.book_imbalance_5.unwrap_or(0.0)    as f32,  // 20
            f.book_imbalance_full.unwrap_or(0.0) as f32,  // 21
            f.book_spread_pct.unwrap_or(0.0)     as f32,  // 22
            f.book_pressure.unwrap_or(0.0)       as f32,  // 23

            // Regime features.
            f.trend_strength.unwrap_or(0.0) as f32,  // 24
            f.vol_regime.unwrap_or(1.0)     as f32,  // 25

            // z-scored returns -- primary cross-regime generalisation signal.
            f.zreturn_30s.unwrap_or(0.0)  as f32,   // 26
            f.zreturn_300s.unwrap_or(0.0) as f32,   // 27

            // Padding slot reserved for future use (always 0.0 until assigned).
            0.0_f32,                                  // 28
        ]
    }

    /// Run inference, returning `(direction, confidence)`.
    ///
    /// On any internal error (shape mismatch, unexpected dtype), falls back
    /// to `(Sideways, 0.0)` rather than panicking -- a single bad tick must
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
    /// `scale`. If you've trained per-`TimeScale` models (recommended --
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
/// (e.g. `direction_model_micro.onnx`, `..._short.onnx`, etc. -- see
/// `btc-model-trainer` README section "Multiple time scales").
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
            TimeScale::Micro  => &self.micro,
            TimeScale::Short  => &self.short,
            TimeScale::Medium => &self.medium,
            TimeScale::Broad  => &self.broad,
        };
        model.predict(features, scale)
    }
}
