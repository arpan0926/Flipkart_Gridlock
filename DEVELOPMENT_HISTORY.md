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

* **Cyclical Time Encoding:** The model originally viewed time linearly, treating Midnight (`0`) as infinitely far away from 11:00 PM (`23`). We converted `hour` and `day` into continuous circles using Sine and Cosine functions (`hour_sin`, `hour_cos`).
* **Geohash Decoding:** We integrated the `pygeohash` library to decode alphanumeric strings into precise `lat` and `lon` coordinates.
* **Human-Behavior Flags:** We added an explicit `is_rush_hour` flag for morning (7-10) and evening (16-19) peaks.

---

## 🧠 Phase 3: The LightGBM Pivot & Native Categories
**The Problem:** The original `RandomForestRegressor` and `OneHotEncoder` were too slow and mathematically rigid to capture complex interactions.
**The Solution:**
1. **Upgraded to LightGBM:** We moved from a bagging approach to Gradient Boosting.
2. **Native Categorical Handling:** We stripped out the `OneHotEncoder` and passed `geohash` natively. This pushed our **Public Leaderboard score to ~85-88**.

---

## ☠️ Phase 4: Identifying & Removing "Silent Killers"
During our push to break the 90-point barrier, we diagnosed and removed several "Silent Killers":
1. **The Target Encoding Time-Leak:** `TargetEncoder` averaged future traffic jams into past data. **Fix:** Removed the encoder.
2. **The "Weekend" Assumption Trap:** Assuming `day >= 5` was Saturday/Sunday randomized the weekend logic. **Fix:** Removed the flag entirely.

---

## 🚀 Phase 5 & 6: Feature Expansion — Leaderboard 91.22
We expanded our feature set to give the model more mathematical handles on time and space, including 15-minute `slot_of_day`, `is_late_night`, `is_off_peak`, and geohash prefixes. We also implemented a custom LightGBM parameter set with a high leaf count (`num_leaves=255`, `max_depth=12`) to capture complex geographic interactions.

---

## 📉 Phase 7: Overcoming "Validation Drift" & The Grand Blend — Leaderboard 91.58
**The Problem:** We attempted to create ultra-precise spatial clusters (rounding coordinates) and Road/Lane cross-features, and used Optuna to heavily tune the model on the last 20% of our data. Our local validation $R^2$ skyrocketed to ~0.85, but our public leaderboard score plummeted to **90.31**.
**The Diagnosis (Validation Drift):** The new features were too rigid, causing the model to memorize the specific traffic patterns of the validation set rather than learning general rules. The future test set behaved differently, and the over-tuned model crashed.

**The Solution (The 91.58 Pivot):**
To bypass the validation drift and reclaim our score, we executed a three-part "Grandmaster" strategy:

1. **Reverted Rigid Features:** We stripped out the noisy spatial clusters, returning to LightGBM's superior native handling of the raw `geohash`.
2. **The 100% Training Rule:** We stopped evaluating locally. For the final submission, we trained the models on **100% of the available data**, ensuring they had the maximum possible context right up until the test set began.
3. **The CatBoost Stabilizer (Ensemble):** LightGBM is highly aggressive and prone to variance. We introduced **CatBoost** (which uses mathematically stable "Symmetric Trees") to act as an anchor. We blended their predictions using a `60% LightGBM / 40% CatBoost` weight.
4. **Target Transformation (The Secret Weapon):** Traffic demand is extremely "right-skewed" (massive rush-hour spikes). We applied a **Logarithmic Transformation** (`np.log1p`) to the target variable before training. This mathematically compressed the extreme traffic spikes, making it easier for the trees to learn the underlying patterns. We then reversed the math (`np.expm1`) on the final predictions.

---

## 🎯 Current Status
- **Public Leaderboard R²:** 91.58054
- **Target:** 94–96 (Qualification Zone)
- **Gap remaining:** ~2.5–3.5 points

### Next Steps to Close the Gap
1. **Deeper Optuna Search:** Now that the pipeline is clean and stabilized by the log-transform, we can run a fresh hyperparameter search for both LightGBM and CatBoost, optimizing for a generalized fit rather than memorizing the local validation fold.
2. **Weather Interactions:** Engineer explicit cross-features between `Weather` and Time (e.g., *Is it raining during morning rush hour vs. midnight?*).
3. **Lag Proxy:** Since actual lag features fail due to missing data in the test set, create "Historical Average" features for each geohash/timeslot directly from the training data.
