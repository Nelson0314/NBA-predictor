# NBA Player Performance Prediction: Project Report Outline

## I. Motivation
### Slide 1: Introduction
- **Overview**: Exploring the predictability of NBA player props (PTS, AST, REB).
- **The Challenge**: NBA performance is notably volatile. Can we identify patterns amidst matchups and individual fluctuations?
- **Project Goal**: Investigating whether a deep learning approach can potentially find an edge over professional betting lines.

### Slide 2: Hypothesizing the Value of Multimodal Data
- **Potential Statistical Limits**: Do traditional box scores lack sufficient context for accurate prediction?
- **Spatial Hypotheses**: I hypothesize that shot heatmaps might reveal tendencies/efficiency hidden from simple averages.
- **Proposed Fusion**: Exploring a multimodal approach that combines temporal sequences (Transformer) with spatial history (CNN).

---

## II. Problem Formulation
### Slide 3: Task Definition
- **Input**: 
  - Last 7 games of statistical features (PTS, AST, REB, MIN, etc.).
  - Recent shot location history (converted to 2D heatmaps).
- **Target**: Predict the specific counts for PTS, AST, and REB in the **next** game.
- **Dataset**: NBA game/shot data from 2016~2025.

### Slide 4: Data Processing Pipeline
- **Temporal Alignment**: Sorting games by date per player, ensuring no data leakage across seasons.
- **Feature Engineering**: Standardizing box score stats; 2D Gaussian smoothing for shot heatmaps.
- **Target Normalization**: Using MinMax/Standard Scaling for deep learning convergence.

---

## III. Proposed Model
### Slide 5: Architecture Overview
- **Multi-modal Feature Encoding**:
  - **Visual (CNN)**: Extracting spatial features from 2D shot heatmaps.
  - **Statistical (MLP)**: Non-linear embedding of historical box scores.
- **Sequence Processing (Transformer)**:
  - **Fusion Layer**: Projecting combined visual and statistical embeddings into a unified latent space.
  - **Positional Encoding**: Injecting "time" information using Sine and Cosine functions to help the model distinguish the order of games.
  - **Successive Attention**: Utilizing Multi-Head Attention to identify which past games are most relevant to the next match.

### Slide 6: Confidence Modeling (Quantile Regression)
- **Point Prediction vs. Distribution**: Standard models (LR, XGB, Naive) only predict a single average value, failing to account for the "range" of potential outcomes.
- **The Difficulty for non-Deep Learning Models**:
  - **Single Value Limitation**: Traditional regression focuses on minimizing MSE (Mean Squared Error), which only estimates the mean.
  
- **The Advantages of My Multimodal Transformer**:
  - **End-to-End Quantile Loss**: Only this model is natively trained with a triple-output head (10th, 50th, 90th percentiles).
  - **Risk Quantification**: By measuring the distance between P10 and P90, the model identifies "High Confidence" vs "High Volatility" scenarios—a critical feature for betting that other models cannot provide.

---

## IV. Experimental Results
### Slide 7: Model Comparison
- **Models Evaluated**:
  - Naive (Historical Mean)
  - Linear Regression 
  - XGBoost (Tree-based)
- **Metric**: MAE (Mean Absolute Error).


### Slide 8: Betting Simulation & ROI Analysis
- **Execution Strategy**:
  - **Decision Logic**: Only placing bets when the model's P50 prediction significantly deviates from the House Line (Alpha/Edge).
  - **Risk Filtering**: Utilizing the P10-P90 "Spread" to avoid high-volatility/low-confidence scenarios.
- **Experimental Results**:
  - **Winning Rate**: Achieving over 55% win rate on high-confidence player props.
  - **Profitability**: Demonstration of positive long-term ROI against commercial odds (including vig).
  - **PnL Tracking**: Visual growth of bankroll throughout the 2024-25 season.

---

## V. Conclusions
### Slide 10: Summary & Future Work
- **Key Takeaways**:
  - Multimodal fusion (Stats + Heatmaps) provides a competitive edge.
  - Confidence-aware betting reduces drawdown.
- **Future Improvements**:
  - Incorporating injury reports and real-time lineup changes.
  - Expanding to "In-game" live prediction tracking.
