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

## 🏆 Phase 5: The Current "Goldilocks" Configuration
To stop the model from memorizing the training data (overfitting a ~60,000 row dataset), we dialed in a specific "Medium Data" hyperparameter setup for LightGBM:
* `num_leaves=63` and `max_depth=8`: Deep enough to learn complex spatial logic, shallow enough to prevent memorization.
* `n_estimators=800` & `learning_rate=0.03`: Allows the model to take tiny, precise steps to correct its errors.
* `subsample=0.8` & `colsample_bytree=0.8`: Injects mathematical randomness to force the model to generalize well to the hidden test set.

---
