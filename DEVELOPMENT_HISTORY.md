# Flipkart Gridlock: Development History & Mathematical Optimizations

## 📖 Purpose of this Document
This document tracks the iterative evolution of the Machine Learning pipeline for the Flipkart Gridlock hackathon. It details the mathematical roadblocks we encountered, the "hackathon traps" we avoided, and the explicit reasoning behind our current architectural choices. 

---

## 🛑 Phase 1: Fixing Data Leakage (The Honest Baseline)
**The Problem:** The initial pipeline utilized `train_test_split` with a random shuffle. This resulted in an artificially high $R^2$ score (~0.94) because the model was looking at future traffic data to predict past traffic data (Data Leakage). 
**The Solution:** We enforced **Chronological Sorting** (`X.sort_values(by=["day", "hour", "minute"])`) before splitting the validation set. 
**The Result:** The local validation score dropped to ~0.75, giving us a true, leak-free baseline that accurately reflects how the model will perform on the hackathon's hidden test set.

---

## ⏱️ Phase 2: Spatio-Temporal Feature Engineering
Tree-based algorithms struggle with human concepts of time and space. We engineered explicit features to translate the physical world into math.

* **Cyclical Time Encoding:** The model originally viewed time linearly, treating Midnight (`0`) as infinitely far away from 11:00 PM (`23`). We converted `hour` and `day` into continuous circles using Sine and Cosine functions (`hour_sin`, `hour_cos`), allowing the algorithm to seamlessly predict overnight traffic flows.
* **Geohash Decoding:** We integrated the `pygeohash` library to decode alphanumeric strings into precise `lat` and `lon` coordinates, grounding the model in real-world spatial geography.
* **Human-Behavior Flags:** We added an explicit `is_rush_hour` flag for morning (7-10) and evening (16-19) peaks, saving the model from having to mathematically derive human work schedules.

---

## 🧠 Phase 3: The LightGBM Pivot & Native Categories
**The Problem:** The original `RandomForestRegressor` and `OneHotEncoder` were too slow and mathematically rigid to capture complex interactions between thousands of spatial coordinates.
**The Solution:**
1. **Upgraded to LightGBM:** We moved from a bagging approach to Gradient Boosting, allowing trees to iteratively correct each other's errors.
2. **Native Categorical Handling:** We stripped out the `OneHotEncoder`. By casting text columns (like `RoadType`, `Weather`, and `geohash`) to Pandas `"category"` dtypes, LightGBM evaluates them natively. This resulted in shallower, smarter decision trees and pushed our **Public Leaderboard score to ~85-88**.

---

## ☠️ Phase 4: Identifying & Removing "Silent Killers"
During our push to break the 90-point barrier, we attempted several advanced techniques that actually caused our validation score to drop. We diagnosed and removed these "Silent Killers":

1. **The Target Encoding Time-Leak:** We attempted to use `TargetEncoder` on `geohash`. However, it averaged future traffic jams into past data, subtly re-introducing time leakage. **Fix:** Removed the encoder and let LightGBM handle `geohash` as a native category.
2. **The "Weekend" Assumption Trap:** We created an `is_weekend` flag assuming `day >= 5` was Saturday/Sunday. Because the dataset days do not necessarily start on Monday, we were randomizing the weekend logic and confusing the model. **Fix:** Removed the flag entirely.
3. **The "Anchor Effect" of Ensembles:** We built a `VotingRegressor` combining LightGBM and XGBoost. Because XGBoost was completely untuned, it acted as a mathematical anchor, dragging down LightGBM's superior predictions. **Fix:** Benched XGBoost temporarily to allow LightGBM to operate at full capacity.

---

## 🏆 Phase 5: The "Goldilocks" Configuration
To stop the model from memorizing the training data (overfitting a ~60,000 row dataset), we dialed in a specific "Medium Data" hyperparameter setup for LightGBM:
* `num_leaves=63` and `max_depth=8`: Deep enough to learn complex spatial logic, shallow enough to prevent memorization.
* `n_estimators=800` & `learning_rate=0.03`: Allows the model to take tiny, precise steps to correct its errors.
* `subsample=0.8` & `colsample_bytree=0.8`: Injects mathematical randomness to force the model to generalize well to the hidden test set.

---

## 🚀 Phase 6: Pipeline Overhaul — Leaderboard 91.22
**Previous best leaderboard score:** ~85-88  
**New leaderboard score:** 91.22803  
**Local validation R²:** 0.750 (honest, leak-free baseline confirmed)

### What Changed

#### 1. Removed All Residual "Silent Killers" from Code
The codebase had diverged from the documented architecture. Several already-diagnosed problems were still present in the actual code:
- `TargetEncoder` was still imported and applied to `geohash` — **removed**
- `is_weekend` flag was still being computed and used — **removed**
- Dead imports (`XGBRegressor`, `VotingRegressor`, `OneHotEncoder`, `StandardScaler`) — **removed**
- Sklearn `ColumnTransformer` preprocessor wrapping LightGBM — **removed** (LightGBM handles NaN and categories natively)

#### 2. Richer Spatio-Temporal Feature Engineering
Added features that give the model more mathematical handles on time and space:

| Feature | What It Encodes |
|---|---|
| `minute_sin`, `minute_cos` | Cyclical encoding of minutes (was raw integer before) |
| `slot_of_day` | Absolute 15-min slot index within a day (0–95) — finer than hour alone |
| `is_late_night` | Flag for 23:00–05:00 low-traffic window |
| `is_off_peak` | Flag for 11:00–15:00 mid-day lull |
| `geohash_prefix4` | ~40 km geographic cluster (multi-resolution spatial view) |
| `geohash_prefix5` | ~5 km geographic cluster |
| `geohash` (native) | Full geohash added to feature set as LightGBM native category |
| `lat_hour_interaction` | `lat × hour_sin` — rush hour patterns differ by location |
| `lon_hour_interaction` | `lon × hour_cos` — spatial-temporal cross feature |
| `lat_lon_combined` | `lat × lon` — compact single spatial proxy |

#### 3. Submission Format Fixed
- Identified that test rows were being reordered during feature engineering, causing predictions to be written in the wrong order relative to the submission `Index` column
- Fixed by preserving `_original_index` through the pipeline and sorting back before saving

#### 4. Two-Phase Training Strategy
- **Phase 1:** Train on 80% with early stopping to find the optimal `n_estimators`
- **Phase 2:** Retrain on 100% of training data using exactly `best_iteration_` rounds
- This gives the model full temporal context (all training data ends at day 49, 2:00 — immediately before the test window starts at 2:15)

#### 5. Upgraded LightGBM Hyperparameters
| Parameter | Old | New | Reason |
|---|---|---|---|
| `num_leaves` | 63 | 255 | Richer spatial splits for 1249 geohashes |
| `max_depth` | 8 | 12 | Deeper trees for complex interactions |
| `n_estimators` | 600 | 2000 (+ early stop) | Auto-finds optimal rounds |
| `learning_rate` | 0.03 | 0.02 | Slower, more precise convergence |
| `reg_alpha` | — | 0.05 | L1 regularisation added |
| `reg_lambda` | — | 1.0 | L2 regularisation added |

#### 6. What We Tried That Didn't Work
- **Lag features via `shift()`:** 97% NaN rate because the test data is sparse per geohash (not every location appears at every time slot)
- **Slot-based lag lookup:** Correct values computed but added noise because test covers rush-hour slots (2:15–13:45) while lags pointed to early-morning low-traffic values
- **Demand stat features (`geo_mean_demand`, `slot_mean_demand`):** Computed from training data and merged into features, but caused overfitting because means encoded the average across all time windows including rush-hour patterns the model was trying to predict

---

## 🎯 Current Status
- **Local validation R²:** ~0.75 (honest chronological split)
- **Public Leaderboard R²:** 91.22803
- **Target:** 95–96
- **Gap remaining:** ~4–5 points

### Next Steps to Close the Gap
The remaining gap between local (0.75) and leaderboard (0.91) suggests the test set is more predictable than our val set. To push toward 95–96:
1. Better handling of the 2,495 missing `Temperature` values (geohash-aware imputation)
2. Road network features — `NumberofLanes × RoadType` interaction
3. Hyperparameter search (Optuna/BayesOpt) focused on `num_leaves` and `min_child_samples`
4. Ensemble: XGBoost re-tuned properly and combined with LightGBM

---
