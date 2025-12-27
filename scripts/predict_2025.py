
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from tqdm import tqdm
import os
import json
import warnings

import sys

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import DATA_DIR, SAVED_MODELS_DIR

# Import data utility
# We assume multiModel.py is in the same directory
try:
    from src.multiModel import loadAndPreprocessData, createMultimodalSequences, preloadHeatmaps, MultimodalDataset, NbaMultimodal
except ImportError as e:
    print(f"Error importing src modules: {e}")
    exit()

warnings.filterwarnings('ignore')

# ==========================================
# 1. Config & Paths
# ==========================================
MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "quant_ep40_seq5_dm64")
DATASET_DIR_OLD = DATA_DIR
DATASET_DIR_NEW = os.path.join(DATA_DIR, "live_2025")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "predictions_2025.csv")

# Hyperparameters (Must match the trained model)
# We load them from config.json
with open(os.path.join(MODEL_PATH, 'config.json'), 'r') as f:
    CONFIG = json.load(f)

SEQ_LENGTH = CONFIG['seqLength']
CONF_THRESH = 40.0 # From previous tuning
MAX_SPREADS = {'PTS': 30.0, 'AST': 10.0, 'REB': 12.0}
TARGET_COLS = ['PTS', 'AST', 'REB']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 2. Model Definition (Copied from train_with_conf.py)
# ==========================================
class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
    def forward(self, preds, target):
        loss = 0
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, :, i]
            loss += torch.max((q-1) * errors, q * errors).mean()
        return loss

class NbaMultimodalQuantile(NbaMultimodal):
    def __init__(self, numStatFeatures, seqLength, numTargets=3, cnnEmbedDim=64, statEmbedDim=128, dModel=128, nHead=4, numLayers=2, dropout=0.1):
        # Initialize Parent
        # Note: Parent __init__ takes (numStatFeatures, seqLength, outputDim, ...)
        # We pass numTargets as outputDim, but we will replace the head anyway.
        super().__init__(numStatFeatures, seqLength, numTargets, cnnEmbedDim, statEmbedDim, dModel, nHead, numLayers, dropout)
        
        # Override Prediction Head for Quantiles
        self.quantiles = [0.1, 0.5, 0.9]
        self.num_quantiles = len(self.quantiles)
        self.num_targets = numTargets
        
        # Head output dim = numTargets * numQuantiles
        self.head = nn.Sequential(
            nn.Linear(dModel, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, numTargets * self.num_quantiles)
        )

    def forward(self, imgSeq, statSeq):
        # We need to access the transformer output (lastState)
        # Parent forward() returns self.head(...)
        # We must re-implement forward or access intermediate?
        # Re-implementing forward logic using parent's modules
        
        # Parent modules:
        # self.cnnEncoder (Input: [B*S, 2, 50, 50]) -> [B*S, cnnDim]
        # self.statEncoder (Input: [B, S, F]) -> [B, S, statDim]
        # self.fusionProj (Input: [B, S, cnn+stat]) -> [B, S, dModel]
        # self.transformer (Input: [B, S, dModel]) -> [B, S, dModel]
        
        batchSize, seqLen, C, H, W = imgSeq.size()
        
        # 1. Visual Branch
        imgFlat = imgSeq.view(batchSize * seqLen, C, H, W)
        visualEmbeds = self.cnnEncoder(imgFlat) 
        visualEmbeds = visualEmbeds.view(batchSize, seqLen, -1)
        
        # 2. Stat Branch
        statEmbeds = self.statEncoder(statSeq) 
        
        # 3. Fusion & Transformer
        jointEmbeds = torch.cat([visualEmbeds, statEmbeds], dim=2) 
        transformerInput = self.fusionProj(jointEmbeds) 
        transformerOut = self.transformer(transformerInput)
        
        # Take last time step
        lastState = transformerOut[:, -1, :] 
        
        # 4. Quantile Head
        out = self.head(lastState)
        
        # Reshape to (Batch, NumTargets, NumQuantiles)
        out = out.view(batchSize, self.num_targets, self.num_quantiles)
        
        return out

# ==========================================
# 3. Training House Baselines (On OLD Data)
# ==========================================
def train_house_models():
    print("Step 1: Training House Models (LR + XGB) on Historical Data...")
    
    # Load OLD Data
    gamesPath = os.path.join(DATASET_DIR_OLD, 'games.csv')
    shotsPath = os.path.join(DATASET_DIR_OLD, 'shots.csv')
    teamsPath = os.path.join(DATASET_DIR_OLD, 'teams.csv')
    
    gamesData, shotsGrouped, featureCols, targetCols = loadAndPreprocessData(gamesPath, shotsPath, teamsPath, SEQ_LENGTH)
    # featureCols is already returned!
    # featureCols = [c for c in gamesData.columns if c not in ['Player_ID', 'Game_ID', 'GAME_DATE', 'SEASON_ID', 'Target_PTS', 'Target_AST', 'Target_REB', 'Target_MIN', 'MATCHUP', 'WL', 'PTS', 'AST', 'REB', 'MIN', 'Player_Name', 'Season', 'TEAM_ABBREVIATION', 'OPPONENT_ABBREVIATION']]

    
    # We need to replicate the exact Training Set used by train_with_conf
    trainSeasons = CONFIG['trainSeasons']
    trainGames = gamesData[gamesData['SEASON_ID'].isin(trainSeasons)]
    
    # Create Sequences (We only need Stats for House)
    # We can use createMultimodalSequences but ignore images
    print("  Generating Training Sequences...")
    _, _, xStatTrain, yTrain = createMultimodalSequences(trainGames, None, SEQ_LENGTH, featureCols, targetCols + ['MIN'])
    # yTrain: [PTS, AST, REB, MIN]
    
    N, S, F = xStatTrain.shape
    xTrainFlat = xStatTrain.reshape(N, S * F)
    yTrainPredict = yTrain[:, :3] # PTS, AST, REB
    
    # Scale Inputs for LR
    scalerX = StandardScaler()
    xTrainFlatScaled = scalerX.fit_transform(xTrainFlat)
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(xTrainFlatScaled, yTrainPredict)
    
    # To reduce memory and time for this script, we can skip XGB or use a lightweight ver?
    # User wants "Simulate Gambling", imply high quality. We stick to Hybrid.
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, n_jobs=-1, random_state=42)
    xgb.fit(xTrainFlat, yTrainPredict)
    
    print("  House Models Trained.")
    return lr, xgb, scalerX, featureCols

# ==========================================
# 4. Predicting 2025 Data
# ==========================================
def main():
    # 1. Train House
    house_lr, house_xgb, house_scaler, featureCols = train_house_models()
    
    # 2. Load New Data
    print("\nStep 2: Loading 2025 Data...")
    gamesPathNew = os.path.join(DATASET_DIR_NEW, 'games_2025.csv')
    shotsPathNew = os.path.join(DATASET_DIR_NEW, 'shots_2025.csv')
    teamsPathNew = os.path.join(DATASET_DIR_NEW, 'teams_2025.csv')
    
    # Note: loadAndPreprocessData handles feature engineering (Rolling stats etc)
    # Note: loadAndPreprocessData handles feature engineering (Rolling stats etc)
    # We assume 'games_2025.csv' has correct format.
    gamesDataNew, shotsGroupedNew, _, _ = loadAndPreprocessData(gamesPathNew, shotsPathNew, teamsPathNew, SEQ_LENGTH)
    # Note: We ignore returned featureCols/targetCols here as we must use the one from Training (Old) Data to ensure consistency?
    # Actually, they should be identical. But strict safety: use the one returned from Step 1.
    
    # Load Heatmaps (New Dir)
    heatmapDirNew = os.path.join(DATASET_DIR_NEW, 'heatmaps')
    heatmapCache = preloadHeatmaps(heatmapDirNew)
    
    print("  Generating 2025 Sequences...")
    # NOTE: Since 2025 is a new file, the first 'SEQ_LENGTH' games for each player 
    # will be skipped by createMultimodalSequences logic. This is unavoidable without 2024 data.
    xPlayer, xGame, xStat, y = createMultimodalSequences(gamesDataNew, shotsGroupedNew, SEQ_LENGTH, featureCols, TARGET_COLS + ['MIN'])
    
    if len(xPlayer) == 0:
        print("Error: No sequences generated. Data might be too short for seqLength.")
        return

    # 3. Load My Model
    print("\nStep 3: Loading Gambler Model...")
    
    # Re-instantiate Model
    model = NbaMultimodalQuantile(
        numStatFeatures=len(featureCols),
        seqLength=SEQ_LENGTH,
        numTargets=3,
        cnnEmbedDim=CONFIG['cnnEmbedDim'],
        statEmbedDim=CONFIG['statEmbedDim'],
        dModel=CONFIG['dModel'],
        nHead=CONFIG['nHead'],
        numLayers=CONFIG['numLayers'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'model.ckpt'), map_location=device))
    model.eval()
    
    # Scalers (Need to re-fit Scalers on TRAIN data or load them? 
    # Ideally should load. But train_with_conf didn't save scalers...
    # Critical: Use the scalers fitted on NEW data? NO. Distribution shift.
    # Must fit scaler on OLD data (Train).
    # Re-fitting scaler on Old Data...
    print("  Re-fitting Scalers on Historical Data...")
    gamesPathOld = os.path.join(DATASET_DIR_OLD, 'games.csv')
    gamesDataOld = pd.read_csv(gamesPathOld) 
    # We assume preprocess already ran on it inside train_house, but we need raw 'featureCols' values
    # Actually, simpler: loadAndPreprocessData returns dataframe with features.
    
    # To save time, let's look at how train_with_conf does it.
    # It fits StandardScaler on train data.
    # We must replicate this.
    
    # We already have loaded Old Data in step 1?
    # Wait, in step 1 I called loadAndPreprocessData but didn't keep the DF.
    # Let's fix Step 1 to return xStatTrain for scalar fitting.
    pass # In this script I will just re-load old data briefly to fit scaler.
    
    # Re-fitting Scalers on Global Data? No on OLD.
    teamsPath = os.path.join(DATASET_DIR_OLD, 'teams.csv')
    gamesDataOld, _, _, _ = loadAndPreprocessData(gamesPathOld, shotsPathNew, teamsPath, SEQ_LENGTH) # Shots path doesn't matter for features
    trainGamesOld = gamesDataOld[gamesDataOld['SEASON_ID'].isin(CONFIG['trainSeasons'])]
    # Extract features
    # This is getting heavy.
    # Shortcut: If we assume 2025 distribution is similar, we can fit on 2025. 
    # But that's risky.
    # Correct way: Fit on Old.
    
    scalerStat = StandardScaler()
    scalerStat.fit(trainGamesOld[featureCols].values)
    
    # Scale New Data
    # xStat is (N, S, F). Reshape, transform, reshape.
    N, S, F = xStat.shape
    xStatFlat = xStat.reshape(N * S, F)
    xStatScaledFlat = scalerStat.transform(xStatFlat)
    xStatScaled = xStatScaledFlat.reshape(N, S, F)
    
    # CRITICAL: Fit Target Scaler (MinMax) on Old Data to Inverse Transform Predictions
    print("  Fitting Target Scaler (MinMax)...")
    scalerY = MinMaxScaler(feature_range=(0, 1))
    # We need yTrain from train_house_models step.
    # To avoid reloading, we assume we can get it from trainGamesOld
    # yTrain was generated in train_house_models but not returned.
    # We must generate it here from trainGamesOld.
    # Actually, train_house_models only generated xStatTrain.
    # We need to re-generate yTrain to fit scalerY.
    _, _, _, yTrainOld = createMultimodalSequences(trainGamesOld, None, SEQ_LENGTH, featureCols, TARGET_COLS + ['MIN'])
    yTrainPredictOld = yTrainOld[:, :3]
    scalerY.fit(yTrainPredictOld)
    
    # Calculate Std Dev of Residuals (Pop Std of Targets) for Odds Calc
    # This is a Rough Estimate of "Variance" for Odds setting
    # In reality, House Variance is Error Variance.
    # We use global std dev of the target as a baseline scalar.
    target_std = np.std(yTrainPredictOld, axis=0) # [StdPTS, StdAST, StdREB]
    
    # 4. Inference
    print("\nStep 4: Running Inference...")
    
    # A. House
    # Flatten
    xStatHouse = xStat.reshape(N, S*F)
    xStatHouseScaled = house_scaler.transform(xStatHouse)
    
    pred_lr = house_lr.predict(xStatHouseScaled)
    pred_xgb = house_xgb.predict(xStatHouse)
    
    # Hybrid House
    indices = [featureCols.index(t) for t in ['PTS', 'AST', 'REB']]
    pred_naive = np.mean(xStat[:, :, indices], axis=1) # (N, 3)
    
    house_preds = 0.4 * pred_lr + 0.45 * pred_xgb + 0.15 * pred_naive
    
    # B. Gambler
    dataset = MultimodalDataset(xPlayer, xGame, xStatScaled, heatmapCache, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    model_preds_scaled = []
    
    with torch.no_grad():
        for xImg, xSt, _ in tqdm(loader, desc="Gambler Predicting"): 
            xImg, xSt = xImg.to(device), xSt.to(device)
            out = model(xImg, xSt)
            model_preds_scaled.append(out.cpu().numpy())
            
    model_preds_scaled = np.concatenate(model_preds_scaled, axis=0) # (N, 3, 3) 
    # model_preds_scaled is in [0, 1] range.
    
    # Inverse Transform
    model_preds = np.zeros_like(model_preds_scaled)
    for q in range(3): # For each quantile
        # Slice (N, 3)
        preds_q = model_preds_scaled[:, :, q]
        preds_inv = scalerY.inverse_transform(preds_q)
        model_preds[:, :, q] = preds_inv

    # Unpack targets
    actuals = y[:, :3]
    
    # 5. Compile Results
    print("\nStep 5: compiling Results...")
    
    # ... (Metadata logic same as before) ...
    meta_list = []
    groups = gamesDataNew.groupby(['Player_ID', 'SEASON_ID']) if 'SEASON_ID' in gamesDataNew.columns else gamesDataNew.groupby('Player_ID')
    
    for _, group in groups:
        if len(group) <= SEQ_LENGTH: continue
        g_ids = group['GAME_ID'].values
        dates = group['GAME_DATE'].values
        names = group['Player_Name'].values
        teams = group['TEAM_ABBREVIATION'].values
        opps = group['OPPONENT_ABBREVIATION'].values
        p_ids = group['Player_ID'].values
        
        for i in range(len(group) - SEQ_LENGTH):
            idx = i + SEQ_LENGTH
            meta_list.append({
                'Player_ID': p_ids[idx],
                'Player_Name': names[idx],
                'GAME_ID': g_ids[idx],
                'Date': dates[idx],
                'Team': teams[idx],
                'Opponent': opps[idx]
            })
            
    from scipy.stats import norm
    
    results = []
    targets = ['PTS', 'AST', 'REB']
    
    for i in range(len(meta_list)):
        row = meta_list[i]
        
        row['Actual_PTS'] = actuals[i, 0]
        row['Actual_AST'] = actuals[i, 1]
        row['Actual_REB'] = actuals[i, 2]
        
        # House (Lines) - Force .5
        line_pts = int(house_preds[i, 0]) + 0.5
        line_ast = int(house_preds[i, 1]) + 0.5
        line_reb = int(house_preds[i, 2]) + 0.5
        
        row['Line_PTS'] = line_pts
        row['Line_AST'] = line_ast
        row['Line_REB'] = line_reb
        
        # Gambler & Odds
        lines = [line_pts, line_ast, line_reb]
        
        for j, t in enumerate(targets):
            p10 = model_preds[i, j, 0]
            p50 = model_preds[i, j, 1] 
            p90 = model_preds[i, j, 2]
            spread = p90 - p10
            
            msp = MAX_SPREADS[t]
            conf_pct = max(0, 100 * (1.0 - spread / msp))
            
            row[f'Pred_{t}'] = round(p50, 1)
            row[f'Conf_{t}'] = round(spread, 1)
            row[f'ConfPct_{t}'] = round(conf_pct, 1)
            
            # --- Odds Calculation ---
            # Line is set to floor(HousePred) + 0.5.
            # HousePred is house_preds[i, j].
            # Deviation = (Line - HousePred).
            # Z-score relative to global StdDev.
            # We assume House is pricing based on a distribution centered at HousePred.
            # Prob(Over) = P(X > Line | Mean=HousePred, Std=target_std[j]*0.6) 
            # (We scale Std down because predictions are more accurate than raw variance)
            
            h_pred = house_preds[i, j]
            line = lines[j]
            std = target_std[j] * 0.7 # Empirical factor for implied volatility
            
            z = (line - h_pred) / std
            prob_over = 1.0 - norm.cdf(z)
            prob_under = norm.cdf(z)
            
            # Fair Odds (No Vig)
            odds_over = round(1.0 / prob_over, 2)
            odds_under = round(1.0 / prob_under, 2)
            
            # Cap odds for display (e.g. max 5.0)
            odds_over = min(5.0, max(1.1, odds_over))
            odds_under = min(5.0, max(1.1, odds_under))
            
            # Store Prediction-side Odds
            # We only care about the odds for the side we bet on?
            # Or store both? The app needs to know which odds apply.
            # Let's store Odds_Over and Odds_Under.
            row[f'OddsOver_{t}'] = odds_over
            row[f'OddsUnder_{t}'] = odds_under
            
            # Bet Logic
            bet_signal = "PASS"
            diff = p50 - line
            
            if conf_pct >= CONF_THRESH:
                 if diff >= 1.5:
                     bet_signal = "BET Over"
                 elif diff <= -1.5:
                     bet_signal = "BET Under"
            
            row[f'Bet_{t}'] = bet_signal
            row[f'Edge_{t}'] = round(diff, 1)

        results.append(row)
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone! Saved predictions to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
