# NBA Player Performance Predictor 🏀

## Overview
This project leverages **Multimodal Deep Learning** to predict NBA player performance (Points, Assists, Rebounds). By combining **Temporal Sequences** (historical stats processed by Transformers) with **Spatial History** (shot heatmaps processed by CNNs), the model captures both player form and shooting tendencies. Furthermore, it employs **Quantile Regression** (predicting P10, P50, P90) to estimate confidence intervals, providing a critical edge for risk assessment in betting scenarios.

## 📂 Repository Structure

### `src/` - Core Modules
- **`multiModel.py`**: Implementation of the **Multimodal Model** (CNN Encoder + Transformer).
- **`seqModel.py`**: Implementation of the **Sequence-only Model** (Transformer) for ablation studies.
- **`graphModel.py`**: Experimental GNN implementation.
- **`config.py`**: Central configuration for file paths and hyperparameters.
- **`simulation.py`**: Logic for betting simulations (ROI calculation, Kelly Criterion, etc.).

### `scripts/` - Execution Scripts
- **`train.py`**: Main entry point for training models.
- **`comparison.py`**: comprehensive evaluation script comparing Deep Learning models against Baselines (Linear Regression, XGBoost).
- **`update_live_2025.py`**: Fetches the latest game data via `nba_api` to keep the dataset current.
- **`predict_bets.py`**: Generates daily predictions for upcoming games and compares them with odds.

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Nelson0314/NBA-predictor.git
    cd NBA-predictor
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### 1. Training
To train the default Multimodal model with Quantile Regression:
```bash
python scripts/train.py --model multimodal --epochs 20
```
**Arguments:**
- `--model`: Choose architecture: `multimodal` (default), `seq`, or `graph`.
- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size (default: 32).

The best model checkpoints are saved to `savedModels_conf/`.

### 2. Evaluation
To evaluate model performance against baselines on the test set (2024-25 season):
```bash
python scripts/comparison.py
```
This script acts as the primary benchmark tool, generating:
- RMSE/MAE comparisons.
- Profit/Loss (PnL) analysis plots.
- Evaluation reports in `evaluation_report_*.txt`.

### 3. Live Prediction (Betting)
For daily usage during the NBA season:

**Step A: Update Data**
Fetch the last night's games to ensure the dataset is up-to-date.
```bash
python scripts/update_live_2025.py
```

**Step B: Generate Predictions**
Predict stats for upcoming games and compare with market odds.
```bash
python scripts/auto_bet.py
```
*Note: This requires an `event_odds_data.json` file containing current market lines.*

## 📄 Reference
For a deep dive into the architecture, hypothesis, and mathematical formulation, please refer to the [IEEE Project Report](docs/Project_Report_IEEE.tex).

## 👨‍💻 Author
**Nelson Weng**
National Yang Ming Chiao Tung University
