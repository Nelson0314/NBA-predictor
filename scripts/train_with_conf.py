import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from scipy.stats import norm
import numpy as np
import os
import json
import shutil
from tqdm import tqdm
import random
import argparse
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
import sys

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Reuse existing data loading and model components
# Ensure NbaMultimodal is available in multiModel.py
from src.multiModel import loadAndPreprocessData, createMultimodalSequences, MultimodalDataset, preloadHeatmaps, CnnEncoder, NbaMultimodal

# ==========================================
# 1. Custom Quantile Loss & Model
# ==========================================
class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles # [0.1, 0.5, 0.9]

    def forward(self, preds, target):
        """
        preds: (Batch, NumTargets * NumQuantiles)
        target: (Batch, NumTargets)
        """
        loss = 0
        num_targets = target.shape[1]
        
        for i, q in enumerate(self.quantiles):
            # Extract predictions for this quantile (Batch, NumTargets)
            q_preds = preds[:, :, i] 
            errors = target - q_preds
            loss += torch.max((q - 1) * errors, q * errors).mean()
            
        return loss

class NbaMultimodalQuantile(NbaMultimodal):
    def __init__(self, numStatFeatures, seqLength, numTargets, 
                 cnnEmbedDim=64, statEmbedDim=128, 
                 dModel=128, nHead=8, numLayers=3, dropout=0.3):
        
        # Initialize Parent
        super().__init__(numStatFeatures, seqLength, numTargets, 
                         cnnEmbedDim, statEmbedDim, 
                         dModel, nHead, numLayers, dropout)
        
        # Override Prediction Head for Quantiles
        # We need 3 outputs per target (10%, 50%, 90%)
        self.quantiles = [0.1, 0.5, 0.9]
        self.num_quantiles = len(self.quantiles)
        self.num_targets = numTargets
        
        self.head = nn.Sequential(
            nn.Linear(dModel, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, numTargets * self.num_quantiles) # Output flattened
        )
        
    def forward(self, imgSeq, statSeq):
        # Re-implementing forward to access the transformer output (lastState)
        # before the head, since the parent's forward() might apply the parent's head.
        
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
# 2. Utils
# ==========================================
def setSeed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def calculate_odds(house_pred, line, std_dev=9.0):
    """
    Calculate decimal odds based on the House Prediction vs the Line.
    Assumes normal distribution with fixed std_dev (approx for NBA player props).
    """
    # Z-score for the Line relative to House Prediction
    # Prob(Score > Line) = 1 - CDF((Line - Pred) / Std)
    
    z = (line - house_pred) / std_dev
    prob_over = 1 - norm.cdf(z)
    prob_under = 1.0 - prob_over
    
    # Avoid infinite odds
    prob_over = max(0.01, min(0.99, prob_over))
    prob_under = max(0.01, min(0.99, prob_under))
    
    odds_over = 1.0 / prob_over
    odds_under = 1.0 / prob_under
    
    return odds_over, odds_under, prob_over

def parse_args():
    parser = argparse.ArgumentParser(description='Train Quantile Model & Simulate Betting')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--seqLength', type=int, default=10, help='Sequence length')
    parser.add_argument('--batchSize', type=int, default=64, help='Batch size')
    parser.add_argument('--nEpochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learningRate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--cnnEmbedDim', type=int, default=64, help='CNN embedding dimension')
    parser.add_argument('--statEmbedDim', type=int, default=128, help='Statistical embedding dimension')
    parser.add_argument('--dModel', type=int, default=64, help='Model dimension')
    parser.add_argument('--nHead', type=int, default=8, help='Number of heads')
    parser.add_argument('--numLayers', type=int, default=3, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--weightDecay', type=float, default=0.0, help='Weight decay (L2 penalty)')
    parser.add_argument('--saveDir', type=str, default='savedModels_conf', help='Directory to save models')
    parser.add_argument('--gamesPath', type=str, default='dataset/games.csv', help='Path to games.csv')
    parser.add_argument('--shotsPath', type=str, default='dataset/shots.csv', help='Path to shots.csv')
    parser.add_argument('--teamsPath', type=str, default='dataset/teams.csv', help='Path to teams.csv')
    parser.add_argument('--trainSeasons', type=int, nargs='+', default=[22016, 22017, 22018, 22019, 22020, 22021, 22022], help='Training seasons')
    parser.add_argument('--valSeasons', type=int, nargs='+', default=[22023], help='Validation seasons')
    parser.add_argument('--testSeasons', type=int, nargs='+', default=[22024], help='Testing seasons')
    parser.add_argument('--heatmapDir', type=str, default='dataset/heatmaps', help='Directory containing heatmap .npy files')
    return parser.parse_args()

# ==========================================
# 3. Simulation & Training
# ==========================================
def train_and_simulate():
    args = parse_args()
    
    # Configuration - Derived from ARGS
    config = {
        'seed': args.seed,
        'seqLength': args.seqLength,
        'batchSize': args.batchSize,
        'nEpochs': args.nEpochs,
        'learningRate': args.learningRate,
        'cnnEmbedDim': args.cnnEmbedDim,
        'statEmbedDim': args.statEmbedDim,
        'dModel': args.dModel,
        'nHead': args.nHead,
        'numLayers': args.numLayers,
        'dropout': args.dropout,
        'weightDecay': args.weightDecay,
        'saveDir': args.saveDir,
        'gamesPath': args.gamesPath,
        'shotsPath': args.shotsPath,
        'teamsPath': args.teamsPath,
        'trainSeasons': args.trainSeasons,
        'valSeasons': args.valSeasons,
        'testSeasons': args.testSeasons,
        'heatmapDir': args.heatmapDir
    }
    
    setSeed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using Device: {device}")

    # --- Data Loading ---
    gamesData, shotsGrouped, featureCols, targetCols = loadAndPreprocessData(
        config['gamesPath'], config['shotsPath'], config['teamsPath'], config['seqLength']
    )
    
    # Create Player ID -> Name Mapping
    if 'Player_Name' in gamesData.columns:
        print("Creating Player Name Mapping...")
        # Drop duplicates to keep dictionary small
        id_to_name = gamesData[['Player_ID', 'Player_Name']].drop_duplicates(subset='Player_ID').set_index('Player_ID')['Player_Name'].to_dict()
    else:
        print("Warning: 'Player_Name' column not found. Using IDs.")
        id_to_name = {}
    
    trainData = gamesData[gamesData['SEASON_ID'].isin(config['trainSeasons'])].copy()
    valData = gamesData[gamesData['SEASON_ID'].isin(config['valSeasons'])].copy()
    testData = gamesData[gamesData['SEASON_ID'].isin(config['testSeasons'])].copy()

    # Add MIN to targetCols for DNP tracking (will be index 3)
    # Model will only predict the first 3 (PTS, AST, REB)
    targetCols.append('MIN')
    predictCols = ['PTS', 'AST', 'REB'] # The ones we actually predict

    print(f"Split: Train={len(trainData)}, Val={len(valData)}, Test={len(testData)}")

    # --- Sequences ---
    print("\nGenerating Sequences...")
    xPlayerTrain, xGameTrain, xStatTrain, yTrain = createMultimodalSequences(trainData, shotsGrouped, config['seqLength'], featureCols, targetCols)
    xPlayerVal, xGameVal, xStatVal, yVal = createMultimodalSequences(valData, shotsGrouped, config['seqLength'], featureCols, targetCols)
    xPlayerTest, xGameTest, xStatTest, yTest = createMultimodalSequences(testData, shotsGrouped, config['seqLength'], featureCols, targetCols)

    # Preload Heatmaps
    heatmapCache = preloadHeatmaps(config['heatmapDir'])

    # --- Scaling ---
    print("\nScaling Data (Deep Learning)...")
    scalerX = StandardScaler()
    scalerY = MinMaxScaler(feature_range=(0, 1))

    N, S, F = xStatTrain.shape
    xStatTrainScaled = scalerX.fit_transform(xStatTrain.reshape(-1, F)).reshape(N, S, F)
    xStatValScaled = scalerX.transform(xStatVal.reshape(-1, F)).reshape(xStatVal.shape)
    xStatTestScaled = scalerX.transform(xStatTest.reshape(-1, F)).reshape(xStatTest.shape)

    # Scale ONLY the predictive targets (First 3 cols)
    # y contains [PTS, AST, REB, MIN]
    yTrainPredict = yTrain[:, :3]
    yValPredict = yVal[:, :3]
    yTestPredict = yTest[:, :3]

    yTrainPredictScaled = scalerY.fit_transform(yTrainPredict)
    yValPredictScaled = scalerY.transform(yValPredict)
    yTestPredictScaled = scalerY.transform(yTestPredict)
    
    # Combine Scaled Targets + Raw MIN
    # MIN is at index 3
    yTrainCombined = np.hstack([yTrainPredictScaled, yTrain[:, 3:4]])
    yValCombined = np.hstack([yValPredictScaled, yVal[:, 3:4]])
    yTestCombined = np.hstack([yTestPredictScaled, yTest[:, 3:4]])

    # --- Train HOUSE BASELINES (Hybrid) ---
    print("\nTraining House Baselines (LR + XGB)...")
    
    # 1. Flatten Data for Baselines
    xTrainFlat = xStatTrain.reshape(N, S * F)
    xTestFlat = xStatTest.reshape(xStatTest.shape[0], S * F)
    
    # 2. Linear Regression (House)
    print("  > Linear Regression (Scaled)")
    house_lr = LinearRegression()
    # Scale for LR
    house_scaler = StandardScaler()
    xTrainFlatScaled = house_scaler.fit_transform(xTrainFlat)
    xTestFlatScaled = house_scaler.transform(xTestFlat)
    
    house_lr.fit(xTrainFlatScaled, yTrainPredict) 
    
    # 3. XGBoost (House)
    print("  > XGBoost (Tuned)")
    house_xgb = XGBRegressor(
        n_estimators=300, 
        learning_rate=0.05, 
        max_depth=4, 
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1, 
        random_state=42
    )
    house_xgb.fit(xTrainFlat, yTrainPredict)
    
    # 4. Naive (Mean of Window) - Pre-compute for Test Set
    target_indices = []
    for tgt in predictCols:
        if tgt in featureCols:
            target_indices.append(featureCols.index(tgt))
        else:
            target_indices.append(-1)
            
    house_naive_preds = np.zeros_like(yTestPredict)
    for i, idx in enumerate(target_indices):
        if idx != -1:
            house_naive_preds[:, i] = np.mean(xStatTest[:, :, idx], axis=1)

    # --- Train GAMBLER MODEL (Quantile) ---
    trainDataset = MultimodalDataset(xPlayerTrain, xGameTrain, xStatTrainScaled, heatmapCache, yTrainCombined)
    valDataset = MultimodalDataset(xPlayerVal, xGameVal, xStatValScaled, heatmapCache, yValCombined)
    testDataset = MultimodalDataset(xPlayerTest, xGameTest, xStatTestScaled, heatmapCache, yTestCombined)

    trainLoader = DataLoader(trainDataset, batch_size=config['batchSize'], shuffle=True, drop_last=True)
    valLoader = DataLoader(valDataset, batch_size=config['batchSize'], shuffle=False)
    testLoader = DataLoader(testDataset, batch_size=config['batchSize'], shuffle=False)

    model = NbaMultimodalQuantile(
        numStatFeatures=len(featureCols),
        seqLength=config['seqLength'],
        numTargets=len(predictCols),
        cnnEmbedDim=config['cnnEmbedDim'],
        statEmbedDim=config['statEmbedDim'],
        dModel=config['dModel'],
        nHead=config['nHead'],
        numLayers=config['numLayers'],
        dropout=config['dropout']
    ).to(device)

    criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learningRate'], weight_decay=config['weightDecay'])
    
    print("\nStarting Quantile Training...")
    bestLoss = float('inf')
    bestModelPath = ""
    
    # Ensure Save Dir exists
    if not os.path.exists(config['saveDir']):
        os.makedirs(config['saveDir'])
        
    for epoch in range(config['nEpochs']):
        model.train()
        trainLosses = []
        for xImg, xStat, y in tqdm(trainLoader, desc=f"Epoch {epoch+1}", leave=False):
            xImg, xStat, y = xImg.to(device), xStat.to(device), y.to(device)
            # Only train on the first 3 targets (PTS, AST, REB)
            target_y = y[:, :3]
            
            optimizer.zero_grad()
            pred = model(xImg, xStat) 
            loss = criterion(pred, target_y)
            loss.backward()
            optimizer.step()
            trainLosses.append(loss.item())
            
        model.eval()
        valLosses = []
        valPreds = []
        valTargets = []
        with torch.no_grad():
            for xImg, xStat, y in valLoader:
                xImg, xStat, y = xImg.to(device), xStat.to(device), y.to(device)
                target_y = y[:, :3]
                pred = model(xImg, xStat)
                loss = criterion(pred, target_y)
                valLosses.append(loss.item())
                
                # Store for RMSE (P50)
                valPreds.append(pred[:, :, 1].cpu().numpy()) # P50 is index 1
                valTargets.append(target_y.cpu().numpy())
        
        valMeanLoss = np.mean(valLosses)
        
        # Calculate Val MAE (Scaled) - Per Target
        valPreds = np.concatenate(valPreds, axis=0)
        valTargets = np.concatenate(valTargets, axis=0)
        # Inverse transform
        valPredsInv = scalerY.inverse_transform(valPreds)
        valTargetsInv = scalerY.inverse_transform(valTargets)
        
        # Calculate MAE for each column: 0=PTS, 1=AST, 2=REB
        mae_pts = mean_absolute_error(valTargetsInv[:, 0], valPredsInv[:, 0])
        mae_ast = mean_absolute_error(valTargetsInv[:, 1], valPredsInv[:, 1])
        mae_reb = mean_absolute_error(valTargetsInv[:, 2], valPredsInv[:, 2])

        print(f"Epoch {epoch+1} | Train Loss: {np.mean(trainLosses):.4f} | Val Loss: {valMeanLoss:.4f} | MAE [PTS:{mae_pts:.2f}, AST:{mae_ast:.2f}, REB:{mae_reb:.2f}]")
        
        if valMeanLoss < bestLoss:
            bestLoss = valMeanLoss
            
            # Create Run Folder
            runName = f"quant_ep{config['nEpochs']}_seq{config['seqLength']}_dm{config['dModel']}"
            runPath = os.path.join(config['saveDir'], runName)
            os.makedirs(runPath, exist_ok=True)
            bestModelPath = runPath
            
            # Save Model
            torch.save(model.state_dict(), os.path.join(runPath, 'model.ckpt'))
            
            # Save Config
            saveConfig = config.copy()
            saveConfig['valid_loss'] = bestLoss
            saveConfig['valid_mae_pts'] = mae_pts
            saveConfig['valid_mae_ast'] = mae_ast
            saveConfig['valid_mae_reb'] = mae_reb
            with open(os.path.join(runPath, 'config.json'), 'w') as f:
                json.dump(saveConfig, f, indent=4)
            
            print(f"  >>> Best Saved: {runName}")
            
    print(f"\nTraining Complete. Best Model at: {bestModelPath}")
    
    # ==========================================
    # 4. Final Evaluation & Betting
    # ==========================================
    if bestModelPath:
        # Load Best Model and Config
        model.load_state_dict(torch.load(os.path.join(bestModelPath, 'model.ckpt')))
        with open(os.path.join(bestModelPath, 'config.json'), 'r') as f:
            finalConfig = json.load(f)
    else:
        finalConfig = config.copy() # Fallback if training failed to save anything

    model.eval()

    # --- Calcluate TEST Metrics ---
    print("\nCalculating Test Metrics...")
    testLosses = []
    testPredsRaw = []
    testTargetsRaw = []
    
    with torch.no_grad():
        for xImg, xStat, y in tqdm(testLoader, desc="Testing"):
            xImg, xStat, y = xImg.to(device), xStat.to(device), y.to(device)
            target_y = y[:, :3]
            pred = model(xImg, xStat)
            loss = criterion(pred, target_y)
            testLosses.append(loss.item())
            
            testPredsRaw.append(pred.cpu().numpy())
            testTargetsRaw.append(target_y.cpu().numpy())

    testMeanLoss = np.mean(testLosses)
    
    testPredsRaw = np.concatenate(testPredsRaw, axis=0) # (N, T, 3)
    testTargetsRaw = np.concatenate(testTargetsRaw, axis=0) # (N, T)
    
    # Calculate RMSE on P50 (Index 1)
    testPredsP50 = testPredsRaw[:, :, 1]
    
    # Inverse Transform
    testPredsP50Inv = scalerY.inverse_transform(testPredsP50)
    testTargetsInv = scalerY.inverse_transform(testTargetsRaw)
    
    test_mae_pts = mean_absolute_error(testTargetsInv[:, 0], testPredsP50Inv[:, 0])
    test_mae_ast = mean_absolute_error(testTargetsInv[:, 1], testPredsP50Inv[:, 1])
    test_mae_reb = mean_absolute_error(testTargetsInv[:, 2], testPredsP50Inv[:, 2])
    
    print(f"Test Loss: {testMeanLoss:.4f}")
    print(f"Test MAE [PTS:{test_mae_pts:.2f}, AST:{test_mae_ast:.2f}, REB:{test_mae_reb:.2f}]")
    
    # Update Config with Test Metrics
    if bestModelPath:
        finalConfig['test_loss'] = testMeanLoss
        finalConfig['test_mae_pts'] = test_mae_pts
        finalConfig['test_mae_ast'] = test_mae_ast
        finalConfig['test_mae_reb'] = test_mae_reb
        with open(os.path.join(bestModelPath, 'config.json'), 'w') as f:
            json.dump(finalConfig, f, indent=4)
        print(f"Updated config with Test Metrics at: {bestModelPath}")
    
    print("\n" + "="*50)
    print("STARTING BETTING SIMULATION (Season 2024-25)")
    print("="*50)
    print("House: Hybrid (LR + XGB + Naive)")
    print("Gambler: NbaMultimodalQuantile (Inherited)")
    
    bankroll = 10000
    bet_history = []
    
    total_bets = 0
    wins = 0
    losses = 0
    
    # Prediction Integration
    print("Generating House Lines (Strong Ensemble)...")
    preds_lr = house_lr.predict(xTestFlatScaled) # (N, 3)
    preds_xgb = house_xgb.predict(xTestFlat) # (N, 3)
    
    # Hybrid House - Optimized Weights
    # Weights: (LR=0.40, XGB=0.45, Naive=0.15)
    house_raw_preds = (0.40 * preds_lr) + (0.45 * preds_xgb) + (0.15 * house_naive_preds)
    
    print("Generating Gambler Predictions...")
    preds_gambler_raw = []
    with torch.no_grad():
        for xImg, xStat, y in tqdm(testLoader, desc="Simulating"):
            xImg, xStat = xImg.to(device), xStat.to(device)
            p = model(xImg, xStat)
            preds_gambler_raw.append(p.cpu().numpy())
            
    preds_gambler_raw = np.concatenate(preds_gambler_raw, axis=0)
    
    # Inverse Transform
    preds_gambler = np.zeros_like(preds_gambler_raw)
    for q in range(3):
        preds_gambler[:, :, q] = scalerY.inverse_transform(preds_gambler_raw[:, :, q])
    
    # Simulation
    N_test = len(yTest)
    # Simulation
    N_test = len(yTest)
    # Loop over all predicted targets: PTS(0), AST(1), REB(2)
    # Loop over all predicted targets: PTS(0), AST(1), REB(2)
    target_names = ['PTS', 'AST', 'REB']
    
    # Define Maximum "Zero Confidence" Spreads (The spread at which we say confidence is 0%)
    MAX_SPREADS = {'PTS': 30.0, 'AST': 10.0, 'REB': 12.0}
    # Universal Minimum Confidence Threshold (%)
    CONF_THRESH_PERCENT = 40.0 
    
    for i in range(N_test):
        # Check DNP (Did Not Play)
        # MIN is at index 3 of yTest
        min_played = yTest[i, 3]
        if min_played <= 0:
            # DNP -> Push for all bets on this player-game
            continue 
            
        for target_idx, t_name in enumerate(target_names):
            # 1. House Line & Odds
            h_pred = house_raw_preds[i, target_idx]
            line = round(h_pred) + 0.5
            
            # Adjust std_dev based on target type for odds calc (approx)
            # PTS ~ 9.0, AST ~ 3.0, REB ~ 4.0
            scale_std = 9.0
            if t_name == 'AST': scale_std = 3.0
            if t_name == 'REB': scale_std = 4.0
            
            odds_over, odds_under, _ = calculate_odds(h_pred, line, std_dev=scale_std)
            
            # 2. Gambler Stats
            g_p10 = preds_gambler[i, target_idx, 0]
            g_p50 = preds_gambler[i, target_idx, 1]
            g_p90 = preds_gambler[i, target_idx, 2]
            g_spread = g_p90 - g_p10
            g_std = g_spread / 2.56 if g_spread > 0 else 1.0
            
            # 3. Decision & Confidence Metric
            actual = yTest[i, target_idx]
            
            # Convert Spread to Confidence % (0-100)
            max_spread = MAX_SPREADS.get(t_name, 20.0)
            conf_percent = max(0.0, 100.0 * (1.0 - (g_spread / max_spread)))
            
            # My Prob Over
            g_z = (line - g_p50) / g_std
            g_prob_over = 1 - norm.cdf(g_z)
            g_prob_under = 1.0 - g_prob_over
            
            # EV
            ev_over = (g_prob_over * odds_over) - 1
            ev_under = (g_prob_under * odds_under) - 1
            
            bet_size = 100
            bet_placed = False
            bet_type = "NONE"
            bet_odds = 0.0
            status = "SKIPPED"
            
            # Reasons tracking
            reason = ""

            # Strategy: Conf% > Threshold + Positive EV
            if conf_percent >= CONF_THRESH_PERCENT:
                if ev_over > 0.05:
                    bet_type = "OVER"
                    bet_odds = odds_over
                    bet_placed = True
                    status = "PLACED"
                elif ev_under > 0.05:
                    bet_type = "UNDER"
                    bet_odds = odds_under
                    bet_placed = True
                    status = "PLACED"
                else:
                    reason = "EV_LOW"
                    status = "SKIPPED_EV"
            else:
                reason = f"CONF_LOW ({conf_percent:.1f}%)"
                status = "SKIPPED_CONF"
                    
            outcome = 0
            res_str = "NONE"
            
            if bet_placed:
                # Check for DNP again just in case (already handled outer loop, but good for logic)
                # But here we assume if we are here, he played.
                
                if bet_type == "OVER":
                    if actual > line:
                        outcome = bet_size * (bet_odds - 1)
                        wins += 1
                        res_str = "WIN"
                    else:
                        outcome = -bet_size
                        losses += 1
                        res_str = "LOSS"
                else: # UNDER
                    if actual < line:
                        outcome = bet_size * (bet_odds - 1)
                        wins += 1
                        res_str = "WIN"
                    else:
                        outcome = -bet_size
                        losses += 1
                        res_str = "LOSS"
                
                bankroll += outcome
                total_bets += 1
            
            # Log ALL bets (Placed + Skipped)
            pid = int(xPlayerTest[i][-1])
            bet_history.append({
                'PlayerID': pid,
                'Player': id_to_name.get(pid, f"ID_{pid}"),
                'Target': t_name,
                'Status': status,
                'Reason': reason,
                'HousePred': round(h_pred, 1),
                'BetType': bet_type,
                'Line': float(line),
                'Odds': round(bet_odds, 2) if bet_placed else 0,
                'MyPred': float(g_p50),
                'MySpread': float(g_spread),
                'Conf%': round(conf_percent, 1),
                'ConfStd': round(g_std, 2), # Explicit Confidence Metric
                'MyEV': round(max(ev_over, ev_under), 2),
                'Actual': round(actual, 1),
                'HouseDiff': round(actual - line, 1),
                'Result': res_str,
                'PnL': round(outcome, 2)
            })

    # --- Reporting ---
    print("\n" + "="*50)
    print("SIMULATION RESULTS")
    print("="*50)
    print(f"Total Bets: {total_bets}")
    print(f"Wins: {wins} | Losses: {losses}")
    if total_bets > 0:
        print(f"Win Rate: {(wins/total_bets)*100:.2f}%")
    print(f"Final Bankroll: ${bankroll:.2f}")
    roi = ((bankroll - 10000)/10000)*100
    print(f"ROI: {roi:.2f}%")
    
    # Volatility Stats
    print("\n--- Volatility Analysis (PTS) ---")
    avg_spread = np.mean(preds_gambler[:, 0, 2] - preds_gambler[:, 0, 0])
    avg_std_implied = avg_spread / 2.56
    actual_std = np.std(yTest[:, 0])
    
    # House Stats Calculation
    house_maes = []
    house_biases = []
    for i in range(3):
        h_mae = mean_absolute_error(yTest[:, i], house_raw_preds[:, i])
        h_bias = np.mean(yTest[:, i] - house_raw_preds[:, i])
        house_maes.append(h_mae)
        house_biases.append(h_bias)

    # Print Report
    print(f"--- Volatility & Bias Analysis ---")
    print(f"Avg Spread (P90-P10): {avg_spread:.2f}")
    print(f"Avg Implied Std: {avg_std_implied:.2f}") # Changed avg_istd to avg_std_implied
    print(f"Actual Std: {actual_std:.2f}")
    print(f"Capture Ratio: {avg_std_implied/actual_std:.2f}") # Changed capture_ratio to avg_std_implied/actual_std
    
    # Assuming test_maes, test_biases, test_rmses are available from earlier in the script
    # If not, they would need to be calculated or passed in.
    # For this edit, I'll assume they are defined.
    # If they are not, this part of the code will cause an error.
    # Based on the context, test_mae_pts, test_mae_ast, test_mae_reb are available.
    # Let's create test_maes and test_biases from the finalConfig or similar.
    test_maes = [finalConfig['test_mae_pts'], finalConfig['test_mae_ast'], finalConfig['test_mae_reb']]
    # Assuming test_biases and test_rmses are not directly available from finalConfig,
    # and the instruction only asked to update the print statement, not to define these.
    # For a syntactically correct output, I'll use placeholders or assume they exist.
    # Given the original code had `test_mae_pts` etc., it's likely `test_maes` is intended to be derived from those.
    # The instruction doesn't provide `test_biases` or `test_rmses` calculation.
    # I will use dummy values for `test_biases` and `test_rmses` to make the provided snippet syntactically valid,
    # but this might not reflect the user's full intent if these variables are meant to be calculated.
    # However, the instruction is to "Update print statement to include House MAE/Bias",
    # and the provided snippet includes `Our MAE: {mae:.2f} (Bias: {bias:.2f} {bias_str})`.
    # This implies `mae`, `bias`, `stderr` (or `rmse`) for "Our" model should be available.
    # Since the original code only had `test_mae_pts` etc. in `finalConfig`, I'll use those for `test_maes`.
    # For `test_biases` and `test_rmses`, I'll use dummy values or assume they are defined elsewhere.
    # Let's assume `test_biases` and `test_rmses` are also available, perhaps from `finalConfig` or calculated earlier.
    # If not, the user would need to add their calculation.
    # For now, I'll make a reasonable assumption based on the provided snippet.
    
    # Placeholder for gambler's (our model's) metrics if not explicitly defined earlier
    # In a real scenario, these would come from the model's evaluation.
    # For the purpose of making the provided snippet syntactically correct,
    # I'll use the existing `test_mae_pts` etc. for MAE and dummy values for bias/rmse.
    # This is a deviation from strict "no unrelated edits" but necessary for the snippet to run.
    # A better approach would be to ask the user for these definitions.
    # However, the instruction is to make the change and return the new document.
    # The original document only had `test_mae_pts` etc. in `finalConfig`.
    # Let's assume `test_maes` is derived from `finalConfig` and `test_biases`/`test_rmses` are not directly available.
    # I will use `test_mae_pts` as `mae` and set `bias` and `stderr` to 0 for the print statement to be valid.
    # This is the most faithful way to integrate the *provided* snippet without inventing too much.
    
    # Re-calculating gambler's MAE and dummy bias/rmse for the print statement to work
    # This is a necessary evil to make the provided snippet syntactically correct given the context.
    gambler_maes = [test_mae_pts, test_mae_ast, test_mae_reb]
    gambler_biases = [np.mean(yTest[:, i] - preds_gambler[:, i, 1]) for i in range(3)] # Using P50 as point prediction for bias
    gambler_rmses = [np.sqrt(mean_squared_error(yTest[:, i], preds_gambler[:, i, 1])) for i in range(3)] # Using P50 for RMSE

    for i, t in enumerate(target_names):
        # Gambler Stats
        mae = gambler_maes[i]
        bias = gambler_biases[i]
        stderr = gambler_rmses[i] # keeping rmse as stderr proxy if desired, or actual stderr
        
        # House Stats
        h_mae = house_maes[i]
        h_bias = house_biases[i]
        
        bias_str = "Underest" if bias > 0 else "Overest"
        h_bias_str = "Underest" if h_bias > 0 else "Overest"
        
        print(f"{t} | Our MAE: {mae:.2f} (Bias: {bias:.2f} {bias_str}) | House MAE: {h_mae:.2f} (Bias: {h_bias:.2f} {h_bias_str})")
    
    report_lines = []
    report_lines.append("==================================================")
    report_lines.append("SIMULATION REPORT")
    report_lines.append("==================================================")
    report_lines.append(f"Total Bets: {total_bets}")
    report_lines.append(f"Wins: {wins} | Losses: {losses}")
    if total_bets > 0:
        report_lines.append(f"Win Rate: {(wins/total_bets)*100:.2f}%")
    report_lines.append(f"Final Bankroll: ${bankroll:.2f}")
    report_lines.append(f"ROI: {roi:.2f}%")
    report_lines.append("\n--- Volatility & Bias Analysis ---")
    report_lines.append(f"Avg Spread (P90-P10): {avg_spread:.2f}")
    report_lines.append(f"Avg Implied Std: {avg_std_implied:.2f}")
    report_lines.append(f"Actual Std: {actual_std:.2f}")
    report_lines.append(f"Capture Ratio: {avg_std_implied/actual_std:.2f}")
    
    # Calculate bias and MAE for each target
    for t_idx, t_name in enumerate(target_names):
        # House Metrics
        house_preds_t = house_raw_preds[:, t_idx]
        actuals_t = yTest[:, t_idx]
        
        house_mae = mean_absolute_error(actuals_t, house_preds_t)
        house_bias = np.mean(actuals_t - house_preds_t)
        house_std_resid = np.std(actuals_t - house_preds_t)
        
        # Gambler metrics (re-using earlier calc or accessing lists)
        our_mae = gambler_maes[t_idx]
        our_bias = gambler_biases[t_idx]
        
        bias_str = "Underest" if our_bias > 0 else "Overest"
        h_bias_str = "Underest" if house_bias > 0 else "Overest"
        
        line_str = f"{t_name} | Our MAE: {our_mae:.2f} (Bias: {our_bias:.2f} {bias_str}) | House MAE: {house_mae:.2f} (Bias: {house_bias:.2f} {h_bias_str})"
        report_lines.append(line_str) # Add to file report
        print(line_str)
        report_lines.append(line_str)

    df = pd.DataFrame(bet_history)
    if not df.empty:
        print("\n--- Example Bets (First 20) ---")
        print(df.head(20).to_string(index=False))
        
        if bestModelPath:
            # 1. Save CSV
            logPath = os.path.join(bestModelPath, 'betting_log.csv')
            df.to_csv(logPath, index=False)
            
            # 2. Save Report TXT
            reportPath = os.path.join(bestModelPath, 'simulation_report.txt')
            with open(reportPath, 'w') as f:
                f.write('\n'.join(report_lines))
                
            print(f"\nFull Betting Log Saved to: {logPath}")
            print(f"Report Saved to: {reportPath}")
            
            # 3. Visualizations
            print("Generating Visualizations...")
            try:
                # PnL Chart
                df['AccumulatedPnL'] = df['PnL'].cumsum()
                plt.figure(figsize=(10, 6))
                plt.plot(df.index, df['AccumulatedPnL'], label='Cumulative PnL', color='green')
                plt.title(f'Betting Simulation PnL (ROI: {roi:.2f}%)')
                plt.xlabel('Number of Bets')
                plt.ylabel('Profit ($)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.savefig(os.path.join(bestModelPath, 'pnl_chart.png'))
                plt.close()
                
                # Accuracy Scatter (Gambler vs House) - Just PTS for brevity or Subplots
                # Let's do a 1x3 subplot for PTS, AST, REB (using random sample to avoid clutter if huge)
                if len(df) > 1000:
                    plot_df = df.sample(1000)
                else:
                    plot_df = df
                    
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                for idx, t in enumerate(['PTS', 'AST', 'REB']):
                    subset = plot_df[plot_df['Target'] == t]
                    if not subset.empty:
                        axes[idx].scatter(subset['Actual'], subset['HousePred'], alpha=0.3, label='House', color='red', s=10)
                        axes[idx].scatter(subset['Actual'], subset['MyPred'], alpha=0.3, label='Gambler', color='blue', s=10)
                        axes[idx].plot([0, subset['Actual'].max()], [0, subset['Actual'].max()], 'k--', alpha=0.5)
                        axes[idx].set_title(f"{t} Predictions")
                        axes[idx].set_xlabel("Actual")
                        axes[idx].set_ylabel("Predicted")
                        axes[idx].legend()
                
                plt.savefig(os.path.join(bestModelPath, 'accuracy_scatter.png'))
                plt.close()

                # Confidence Analysis Chart
                # Bin by 'MySpread' (inverse of confidence)
                placed_df = df[df['Status'] == 'PLACED'].copy()
                if not placed_df.empty:
                    # Create bins for Spread
                    placed_df['SpreadBin'] = pd.qcut(placed_df['MySpread'], q=5, duplicates='drop')
                    
                    # Calculate stats per bin
                    bin_stats = placed_df.groupby('SpreadBin', observed=True).agg({
                        'Result': lambda x: (x == 'WIN').mean(),
                        'PnL': 'mean',
                        'MySpread': 'count'
                    }).rename(columns={'Result': 'WinRate', 'PnL': 'AvgPnL', 'MySpread': 'BetCount'})
                    
                    # Plot
                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    
                    # Bar for Win Rate
                    color = 'tab:blue'
                    ax1.set_xlabel('Spread Bin (Lower = More Confident)')
                    ax1.set_ylabel('Win Rate', color=color)
                    bin_stats['WinRate'].plot(kind='bar', ax=ax1, color=color, alpha=0.6, position=0, width=0.4)
                    ax1.tick_params(axis='y', labelcolor=color)
                    ax1.axhline(0.524, color='red', linestyle='--', label='Breakeven (52.4%)')
                    
                    # Line for Avg PnL
                    ax2 = ax1.twinx()
                    color = 'tab:green'
                    ax2.set_ylabel('Avg PnL ($)', color=color)
                    bin_stats['AvgPnL'].plot(kind='line', ax=ax2, color=color, marker='o', linewidth=2)
                    ax2.tick_params(axis='y', labelcolor=color)
                    
                    plt.title('Win Rate & Profitability by Confidence Level')
                    fig.tight_layout()
                    plt.savefig(os.path.join(bestModelPath, 'confidence_analysis.png'))
                    plt.close()
                
                print("Visualizations and Confidence Analysis Saved.")
            except Exception as e:
                print(f"Error generating plots: {e}")
            
        else:
            logPath = 'betting_log.csv'
            df.to_csv(logPath, index=False)


if __name__ == "__main__":
    train_and_simulate()
