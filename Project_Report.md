# NBA Player Performance Prediction: Project Report Outline

## I. Motivation
### Slide 1: Introduction
- **Overview**: Predicting NBA player prop outcomes (PTS, AST, REB).
- **The Challenge**: NBA performance is highly volatile, influenced by matchups, fatigue, and individual shooting fluctuations.
- **Project Goal**: Leverage both historical statistics and spatial shot metadata to beat professional betting lines.

### Slide 2: Why Multimodal?
- **Statistical Limits**: Traditional box scores lack spatial context (where are they shooting?).
- **Spatial Insights**: Shot heatmaps reveal player tendencies/efficiency not captured in simple averages.
- **Fusion Approach**: Combining temporal sequences (Transformer) with spatial history (CNN).

---

## II. Problem Formulation
### Slide 3: Task Definition
- **Input**: 
  - Last 7-10 games of statistical features (PTS, AST, REB, MIN, etc.).
  - Recent shot location history (converted to 2D heatmaps).
- **Target**: Predict the specific counts for PTS, AST, and REB in the **next** game.
- **Dataset**: NBA game/shot data from 2016 to 2025.

### Slide 4: Data Processing Pipeline
- **Temporal Alignment**: Sorting games by date per player, ensuring no data leakage across seasons.
- **Feature Engineering**: Standardizing box score stats; 2D Gaussian smoothing for shot heatmaps.
- **Target Normalization**: Using MinMax/Standard Scaling for deep learning convergence.

---

## III. Proposed Model
### Slide 5: Architecture Overview
- **Visual Branch**: CNN (Convolutional Neural Network) to extract features from spatial heatmaps.
- **Statistical Branch**: MLP Encoder to process historical box scores.
- **Transformer Decoder**: Attention mechanism to weigh the importance of past games in the sequence.

### Slide 6: Confidence Modeling (Quantile Regression)
- **Beyond Point Estimation**: Predicting a single number isn't enough for betting.
- **Quantile Loss**: Outputting 10th, 50th (median), and 90th percentiles.
- **Uncertainty**: The "Spread" between P10 and P90 indicates the model's confidence in the player's performance.

---

## IV. Experimental Results
### Slide 7: Baseline Comparison
- **Models Evaluated**:
  - Naive (Historical Mean)
  - Linear Regression (Baseline)
  - XGBoost (Tree-based Baseline)
- **Metric**: MAE (Mean Absolute Error).
- **Results**: Multimodal Transformer achieves significantly lower MAE (~3.08 Avg) vs. Naive models (~4.38 Avg).

### Slide 8: Betting Simulation Logic
- **Betting Strategy**: Only placing bets when the model's P50 prediction significantly deviates from the House Line (EV+).
- **Risk Management**: Kelly Criterion or Fixed Unit staking based on model confidence (Spread).

### Slide 9: Performance & ROI
- **Winning Rate**: Achieving over 55% win rate on selected player props.
- **PnL Tracking**: Visual demonstration of bankroll growth over the 2024-25 season.
- **ROI**: Demonstration of positive Return on Investment against commercial odds (including vig).

---

## V. Conclusions
### Slide 10: Summary & Future Work
- **Key Takeaways**:
  - Multimodal fusion (Stats + Heatmaps) provides a competitive edge.
  - Confidence-aware betting reduces drawdown.
- **Future Improvements**:
  - Incorporating injury reports and real-time lineup changes.
  - Expanding to "In-game" live prediction tracking.
