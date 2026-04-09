# Multi-Model Prediction of Win Outcomes in High-Elo Teamfight Tactics

**Authors:** Ziyan DAI, Ilyes BENZIANE

## Overview
This project explores the predictability of winning a game of Teamfight Tactics (TFT) based on a player's final board state. Because TFT is an auto-chess game heavily reliant on strategic resource management and probability, we built a multi-model machine learning pipeline to analyze challenger-level data. The project identifies the game's strict "meta" compositions, mathematically calculates the optimal items for specific units, and predicts match outcomes (Top 4 vs. Bot 4) with high accuracy.

---

## Features & Methodology

### 1. Data Collection & Preprocessing
* **Custom API Client:** Built a custom `RateLimiter` class in `tft_utils.py` to query Riot's API without exceeding request limits, with extraction handled via `data_extractor.py`.
* **Challenger-Tier & Patch Isolation:** Exclusively collected data from top-tier Challenger players within a single patch (e.g., 16.2) to prevent suboptimal gameplay from skewing the data and to ensure meta consistency.
* **Feature Engineering:** Extracted relative metrics like `ratio_valeur_plateau` (board value relative to the lobby) and `ratio_level` to contextualize player strength against their specific opponents.
* **Data Cleaning:** Removed "rage-sell" boards by setting minimum placement-specific gold thresholds (e.g., discarding top-two finishes with less than 60 gold). Handled data sparsity by dropping combinations occurring less than 0.3% of the time and exported the dataset as Parquet.

### 2. Unsupervised Learning: Meta Discovery
* **K-Means Clustering:** Applied K-Means exclusively on categorical synergy columns to force the algorithm to cluster based on board identity rather than macro variables like player wealth.
* **Finding the Optimal K:** Validated `K=10` as the optimal number of clusters, accurately mirroring the standard ~10 viable meta compositions typical of a TFT patch.
* **Cluster Interpretability:** Translated raw mathematical clusters into recognizable compositions (e.g., Ionia Yunara/Wukong, Demacia) by mapping synergies to their core champions.

### 3. Supervised Learning: Item Optimization
* **Linear Regression:** Used to isolate the marginal effect of specific items on specific champions (e.g., finding the Best-in-Slot items for Ziggs).
* **The "Delta" Metric:** The regression coefficient (beta) represents the expected change in final placement. A negative delta (e.g., -0.18) pushes a player toward 1st place and is considered highly beneficial.

### 4. Supervised Learning: Match Outcome Prediction
* **Random Forest Classifier:** Selected to capture complex, non-linear relationships between synergies and items, outperforming a baseline Naive Bayes model that collapsed under strict independence assumptions.
* **Preventing Data Leakage:** Implemented a Group Shuffle Split based on `Match ID` to ensure players from the same correlated lobby were not split across training and testing sets.

---

## Key Results
* **High Predictive Power:** The Random Forest model achieved a global accuracy of **83.92%** and an **AUC score of 0.92**, proving highly capable of distinguishing winning boards from losing ones.
* **Macro over Micro:** Feature importance analysis revealed that macro fundamentals—specifically a player's relative board value and level compared to their lobby—are the dominant predictors of victory.
* **The Cost of Deviation:** Cross-referencing clusters with win rates provided numerical proof of a "rigid meta." The Bilgewater/Peeba composition topped the patch at a 54.5% Top 4 rate, and players who deviated from established optimal clusters suffered significantly lower win rates. 

---

## Note on Missing Variables
Due to Riot Games removing augment data from their public API, the current model predicts outcomes without factoring in Augments. While board value and level remain the strongest available predictors, augments represent a crucial unquantifiable variable that can dramatically alter game outcomes.