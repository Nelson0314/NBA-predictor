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

### Slide 8: Result 1 - Model Performance vs. Data Scale
- **Observations (Reference: report_full.txt vs. report_7dim.txt)**:
  Scenario A: Low-Dimensional Data (Raw Features)

Observation: In environments with limited raw features, the Transformer significantly outperforms traditional models (XGBoost/Linear Regression).

Insight: This demonstrates the Transformer's superior capability for Automated Feature Extraction, successfully mining complex latent patterns from raw temporal sequences without human intervention.

Scenario B: High-Dimensional Data (Extensive Feature Engineering)

Observation: When extensive manually engineered features (e.g., rolling averages, lag features) are introduced, traditional models catch up to the Transformer's performance.

Key Conclusion: The Deep Learning architecture effectively replaces the need for laborious manual feature engineering, showcasing the efficiency of end-to-end learning in discovering interactions that otherwise require domain-expert coding.

### Slide 9: Result 2 - The Impact of Spatial Heatmaps (CNN)
- **Observations (Reference: With CNN vs. No CNN)**:
  | Model Config | No CNN (Stats only) | With CNN (Multimodal) | Result |
  | :--- | :--- | :--- | :--- |
  | **MAE (Lower is better)** | ~3.0827 | ~3.0866 | **No Gain** |
- **Inference & Hypotheses**:
  1. **Information Redundancy**: Traditional stats (FG%, 3P%) may already implicitly describe the "where" and "how" of a player's shot profile.
  2. **Relevance Gap**: Spatial patterns might be crucial for individual shot outcome prediction but are less critical for aggregate game-total counts (PTS/REB).
  3. **Compression Loss**: Compressing complex 2D image data to match statistical feature dimensions (for fusion) likely stripped away critical spatial information.

### Slide 10: Result 3 - The Value of Confidence (Quantile Regression)
- **Observations**: The introduction of quantile-based confidence metrics significantly elevates prediction quality for decision-making.
- **Proof**: This is clearly demonstrated in the **Betting Simulation Results**, where identifying "predictable" games (low spread) accounts for the majority of the realized profit.

### Slide 11: Betting Simulation & ROI Analysis
- **Execution Strategy**:
  - **Decision Logic**: Only placing bets when the model's P50 prediction significantly deviates from the House Line (Alpha/Edge).
  - **Risk Filtering**: Utilizing the P10-P90 "Spread" to avoid high-volatility/low-confidence scenarios.
- **Experimental Results**:
  - **Winning Rate**: Achieving over ?% win rate on high-confidence player props.
  - **Profitability**: Positive long-term ROI against commercial lines.

### Slide 12: Model Interpretability - Attention Map Analysis
- **Temporal Dynamics**: Visualizing how the Transformer weights each of the past 7 games.
- **Key Findings**:
  - **"Bell-Shaped" Attention**: Interestingly, the weights are often **highest in the middle** of the sequence (games 3-5) and lower at both the earliest and most recent games.
  - **Stability over Recency**: This suggests the model relies on a "stabilized average form" from the mid-sequence rather than overreacting to the very last game or focusing on outdated long-term history.
- **Value of Attention**: Identifies the "Anchor Games" that the model uses to stabilize its prediction against extreme outliers.

---

## V. Conclusions
### Slide 13: Summary & Key Takeaways
- **Architecture Efficiency**: Automated Feature Extraction: Deep Learning demonstrates the ability to mine complex patterns from raw data autonomously, whereas traditional models rely heavily on extensive manual feature engineering to achieve comparable performance.
- **Redundancy of Visual Data**: Spatial shot history may be redundant for predicting traditional box-score outcomes.
- **Strategic Success**: Quantile regression is the most critical component for practical application, transforming a prediction model into a risk-management tool.

Slide 13: Future Directions

Extending to NLP & Live Prediction:

Concept: Integrating Play-by-Play text streams and news sentiment analysis using Large Language Models (LLMs).

Goal: To capture contextual nuances that box scores cannot quantify—such as "player injury/discomfort," "defensive intensity," or "momentum shifts"—and advancing the prediction window from pre-game to In-Game (Live Betting).