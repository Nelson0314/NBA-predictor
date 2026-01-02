# NBA Player Performance Predictor

## 🏀 Overiew
This project utilizes a multimodal Deep Learning approach to predict NBA player performance (Points, Assists, Rebounds). It combines **Temporal Sequences** (past game stats processed by Transformers) with **Spatial History** (shot heatmaps processed by CNNs) to identify patterns in player form and matchups. The model also calculates confidence intervals using **Quantile Regression** to aid in risk assessment for betting scenarios.

## 🚀 Features
- **Multimodal Architecture**: Fuses statistical time-series data with visual shot chart heatmaps.
- **Transformer-based Sequence Modeling**: Captures long-term dependencies in player form.
- **Quantile Regression**: Predicts P10, P50, and P90 to estimate volatility and confidence.
- **Betting Simulation**: Includes a strategy engine to backtest predictions against betting lines with ROI analysis.

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

## 🏗️ Usage

### Training
To train the multimodal model:
```bash
python scripts/train_with_conf.py
```
This script will:
- Load data from 2016-2025.
- Train the transformer model with quantile loss.
- Save the best model to `savedModels_conf/`.

### Prediction & Simulation
To evaluate the model and run the betting simulation:
```bash
python scripts/comparison.py
```
This will compare the model's predictions against a baseline and generate profit/loss reports.

## 📂 Dataset
The project uses NBA data fetched via `nba_api` covering seasons from 2016-17 to 2024-25. Data includes:
- **Game Logs**: Traditional box score stats.
- **Shot Charts**: Spatial coordinate data for every shot attempt.

## 📄 Reference
For a detailed technical explanation, please refer to the [IEEE Project Report](docs/Project_Report_IEEE.tex).

## 👨‍💻 Author
**Nelson Weng**
National Yang Ming Chiao Tung University
