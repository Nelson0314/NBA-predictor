import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from scipy.stats import norm
import numpy as np
import os
import json
import shutil
from tqdm import tqdm
import random
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
import sys
import torch.multiprocessing as mp

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Reuse existing data loading and model components
# Ensure NbaMultimodal is available in multiModel.py
from src.multiModel import loadAndPreprocessData, createMultimodalSequences, MultimodalDataset, preloadHeatmaps, CnnEncoder, NbaMultimodal
from src.seqModel import train as train_seq
from src.odds import calculate_odds
from src.simulation import run_betting_simulation

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
    parser.add_argument('--models', nargs='+', default=['multi'], choices=['multi', 'seq'], help='Models to train (multi, seq)')
    return parser.parse_args()

# ==========================================
# 3. Simulation & Training
# ==========================================
def train_multimodal_quantile(config):
    setSeed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using Device: {device}")

    # --- Data Loading ---
    gamesData, shotsGrouped, featureCols, targetCols = loadAndPreprocessData(
        config['gamesPath'], config['shotsPath'], config['teamsPath'], config['seqLength']
    )
    
    # Create Player ID -> Name Mapping
    if 'Player_Name' in gamesData.columns:
        id_to_name = gamesData[['Player_ID', 'Player_Name']].drop_duplicates(subset='Player_ID').set_index('Player_ID')['Player_Name'].to_dict()
    else:
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
    # Loop over all predicted targets: PTS(0), AST(1), REB(2)
    target_names = ['PTS', 'AST', 'REB']
    
    # House Stats Calculation
    house_maes = []
    house_biases = []
    for i in range(3):
        h_mae = mean_absolute_error(yTest[:, i], house_raw_preds[:, i])
        h_bias = np.mean(yTest[:, i] - house_raw_preds[:, i])
        house_maes.append(h_mae)
        house_biases.append(h_bias)

    # Print Report
    report_lines = []
    
    # --- Calculate Comprehensive Baselines (Test Set) ---
    # 1. Naive
    naive_mae = mean_absolute_error(yTestPredict, house_naive_preds)
    naive_mse = mean_squared_error(yTestPredict, house_naive_preds)
    
    # 2. Linear Regression
    lr_preds = house_lr.predict(xTestFlatScaled)
    lr_mae = mean_absolute_error(yTestPredict, lr_preds)
    lr_mse = mean_squared_error(yTestPredict, lr_preds)
    
    # 3. XGBoost
    xgb_preds = house_xgb.predict(xTestFlat)
    xgb_mae = mean_absolute_error(yTestPredict, xgb_preds)
    xgb_mse = mean_squared_error(yTestPredict, xgb_preds)
    
    baseline_info = [
        "="*50,
        "BASELINE COMPARISON (TEST SET)",
        "="*50,
        f"Naive (Mean)      | MAE: {naive_mae:.4f} | MSE: {naive_mse:.4f}",
        f"Linear Regression | MAE: {lr_mae:.4f} | MSE: {lr_mse:.4f}",
        f"XGBoost           | MAE: {xgb_mae:.4f} | MSE: {xgb_mse:.4f}",
        "="*50
    ]
    
    for l in baseline_info:
        print(l)
        report_lines.append(l)

    # Prepare metrics for report
    test_maes = [finalConfig['test_mae_pts'], finalConfig['test_mae_ast'], finalConfig['test_mae_reb']]
    
    # Calculate gambler metrics for comparison
    gambler_maes = [test_mae_pts, test_mae_ast, test_mae_reb]
    gambler_biases = [np.mean(yTest[:, i] - preds_gambler[:, i, 1]) for i in range(3)] 
    gambler_rmses = [np.sqrt(mean_squared_error(yTest[:, i], preds_gambler[:, i, 1])) for i in range(3)]

    for i, t in enumerate(target_names):
        # Gambler Stats
        mae = gambler_maes[i]
        bias = gambler_biases[i]
        h_mae = house_maes[i]
        h_bias = house_biases[i]
        
        line_str = f"{t} | Our MAE: {mae:.2f} (Bias: {bias:.2f}) | House MAE: {h_mae:.2f} (Bias: {h_bias:.2f})"
        print(line_str)
        report_lines.append(line_str)
    
    print("MultiModel Training Completed. Returning predictions for Simulation...")
    return preds_gambler, house_raw_preds, yTest, metaTest, bestModelPath



def main():
    args = parse_args()
    
    # Build Config Dict
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
        'heatmapDir': args.heatmapDir,
        'models': args.models # New list
    }
    
    for model_name in config['models']:
        print(f"\n{'='*40}")
        print(f"Running Training: {model_name.upper()}")
        print(f"{'='*40}")
        
        result = None
        if model_name == 'multi':
            result = train_multimodal_quantile(config)
        elif model_name == 'seq':
            # seqModel train mostly relies on config passed
            config['datasetPath'] = config['gamesPath'] # Ensure backward compatibility if seqModel uses datasetPath
            result = train_seq(config)
        else:
            print(f"Unknown model: {model_name}")
            
        # Shared Simulation CALL
        if result:
            preds_gambler, house_preds, y_test, meta_test, run_path = result
            if run_path and os.path.exists(run_path):
                 run_betting_simulation(
                     gambler_preds=preds_gambler,
                     house_preds=house_preds,
                     y_true=y_test,
                     metadata=meta_test,
                     target_cols=['PTS', 'AST', 'REB'], # Assuming fixed
                     save_dir=run_path
                 )

if __name__ == "__main__":
    main()
