import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import os
import json
import shutil
from tqdm import tqdm
import random
import argparse
import sys

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.multiModel import NbaMultimodal, loadAndPreprocessData, createMultimodalSequences, MultimodalDataset, preloadHeatmaps
from src.seqModel import train as train_seq
from src.graphModel import train as train_cnn
from comparison import train_baselines_and_compare

# ==========================================
# 1. 固定隨機種子 (Set Seed)
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
    parser = argparse.ArgumentParser(description='Train NBA Multimodal Model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--seqLength', type=int, default=10, help='Sequence length')
    parser.add_argument('--batchSize', type=int, default=64, help='Batch size')
    parser.add_argument('--nEpochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learningRate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--cnnEmbedDim', type=int, default=64, help='CNN embedding dimension')
    parser.add_argument('--statEmbedDim', type=int, default=128, help='Statistical embedding dimension')
    parser.add_argument('--dModel', type=int, default=64, help='Model dimension (Transformer)')
    parser.add_argument('--nHead', type=int, default=8, help='Number of heads (Transformer)')
    parser.add_argument('--numLayers', type=int, default=3, help='Number of layers (Transformer)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--saveDir', type=str, default='savedModels', help='Directory to save models')
    # Default paths assume running from root or scripts, but passing relative to root is safer if we use os.path.join(ROOT, ...)
    # But here we rely on the user passing correct paths or default relative to CWD.
    # Given we set Config paths in src/config, we could use them as defaults?
    # For now, let's leave them relative and assume user runs from project root.
    parser.add_argument('--gamesPath', type=str, default='dataset/games.csv', help='Path to games.csv')
    parser.add_argument('--shotsPath', type=str, default='dataset/shots.csv', help='Path to shots.csv')
    parser.add_argument('--teamsPath', type=str, default='dataset/teams.csv', help='Path to teams.csv')
    parser.add_argument('--trainSeasons', type=int, nargs='+', default=[22016, 22017, 22018, 22019, 22020, 22021, 22022], help='Training seasons')
    parser.add_argument('--valSeasons', type=int, nargs='+', default=[22023], help='Validation seasons')
    parser.add_argument('--testSeasons', type=int, nargs='+', default=[22024], help='Testing seasons')
    
    # New argument for selecting models
    parser.add_argument('--models', nargs='+', default=['multi'], choices=['multi', 'seq', 'cnn'], 
                        help='List of models to train (multi, seq, cnn). Default: multi')
    
    return parser.parse_args()

# ==========================================
# 2. Multimodal Training Function
# ==========================================
def train_multi(config):
    setSeed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using Device: {device}")

    # --- Data Loading ---
    try:
        gamesData, shotsGrouped, featureCols, targetCols = loadAndPreprocessData(
            config['gamesPath'], config['shotsPath'], config['teamsPath'], config['seqLength']
        )

        trainData = gamesData[gamesData['SEASON_ID'].isin(config['trainSeasons'])].copy()
        valData = gamesData[gamesData['SEASON_ID'].isin(config['valSeasons'])].copy()
        testData = gamesData[gamesData['SEASON_ID'].isin(config['testSeasons'])].copy()

        print(f"Split: Train={len(trainData)}, Val={len(valData)}, Test={len(testData)}")

        # --- Generate Sequences ---
        print("\nGenerating Train Sequences...")
        xPlayerTrain, xGameTrain, xStatTrain, yTrain = createMultimodalSequences(trainData, shotsGrouped, config['seqLength'], featureCols, targetCols)
        
        print("Generating Val Sequences...")
        xPlayerVal, xGameVal, xStatVal, yVal = createMultimodalSequences(valData, shotsGrouped, config['seqLength'], featureCols, targetCols)
        
        print("Generating Test Sequences...")
        xPlayerTest, xGameTest, xStatTest, yTest = createMultimodalSequences(testData, shotsGrouped, config['seqLength'], featureCols, targetCols)

        if len(xStatTrain) == 0:
            raise ValueError("No training data generated!")
        
        # --- Preload Heatmaps into RAM (Fix Speed Issue) ---
        print("\nPre-loading Heatmaps into RAM...")
        heatmapCache = preloadHeatmaps('dataset/heatmaps')

        # --- Scaling (Stats Only) ---
        print("\nScaling Data...")
        scalerX = StandardScaler()
        scalerY = MinMaxScaler(feature_range=(0, 1))

        # Scale Features
        N, S, F = xStatTrain.shape
        xStatTrainScaled = scalerX.fit_transform(xStatTrain.reshape(-1, F)).reshape(N, S, F)
        xStatValScaled = scalerX.transform(xStatVal.reshape(-1, F)).reshape(xStatVal.shape)
        xStatTestScaled = scalerX.transform(xStatTest.reshape(-1, F)).reshape(xStatTest.shape)

        # Scale Targets
        yTrainScaled = scalerY.fit_transform(yTrain)
        yValScaled = scalerY.transform(yVal)
        yTestScaled = scalerY.transform(yTest)

        # --- Datasets & Loaders ---
        # Pass heatmapCache to Dataset
        trainDataset = MultimodalDataset(xPlayerTrain, xGameTrain, xStatTrainScaled, heatmapCache, yTrainScaled)
        valDataset = MultimodalDataset(xPlayerVal, xGameVal, xStatValScaled, heatmapCache, yValScaled)
        testDataset = MultimodalDataset(xPlayerTest, xGameTest, xStatTestScaled, heatmapCache, yTestScaled)

        if os.name == 'nt':
            num_workers = 0 # RAM Cache is fast enough, using 0 workers on Windows prevents spawn overhead
            persistent_workers = False
        else:
            num_workers = 4
            persistent_workers = True

        trainLoader = DataLoader(trainDataset, batch_size=config['batchSize'], shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)
        valLoader = DataLoader(valDataset, batch_size=config['batchSize'], shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)
        testLoader = DataLoader(testDataset, batch_size=config['batchSize'], shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)

        # --- Model Initialization ---
        model = NbaMultimodal(
            numStatFeatures=len(featureCols),
            seqLength=config['seqLength'],
            outputDim=len(targetCols),
            cnnEmbedDim=config['cnnEmbedDim'],
            statEmbedDim=config['statEmbedDim'],
            dModel=config['dModel'],
            nHead=config['nHead'],
            numLayers=config['numLayers'],
            dropout=config['dropout']
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learningRate'], weight_decay=1e-4)
        # Cosine Annealing Scheduler: T_max corresponds to total epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['nEpochs'], eta_min=1e-6)

        # --- Training Loop ---
        print("\nStep 3: Start Training (Multimodal)...")
        bestLoss = float('inf')
        bestModelPath = ""
        
        if not os.path.exists(config['saveDir']):
            os.makedirs(config['saveDir'])

        for epoch in range(config['nEpochs']):
            # Train
            model.train()
            trainLosses = []
            trainPbar = tqdm(trainLoader, desc=f"Epoch {epoch+1}/{config['nEpochs']}", leave=False)
            
            for xImg, xStat, y in trainPbar:
                xImg, xStat, y = xImg.to(device), xStat.to(device), y.to(device)
                
                optimizer.zero_grad()
                pred = model(xImg, xStat)
                loss = criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                trainLosses.append(loss.item())
            
            # Step Scheduler
            scheduler.step()
            trainMeanLoss = np.mean(trainLosses)

            # Val
            model.eval()
            valLosses = []
            valDiffSqSum = np.zeros(len(targetCols))
            valCount = 0
            
            with torch.no_grad():
                for xImg, xStat, y in valLoader:
                    xImg, xStat, y = xImg.to(device), xStat.to(device), y.to(device)
                    pred = model(xImg, xStat)
                    loss = criterion(pred, y)
                    valLosses.append(loss.item())

                    # Metrics (Original Scale)
                    predOrig = scalerY.inverse_transform(pred.cpu().numpy())
                    predOrig = np.maximum(predOrig, 0) # Force non-negative
                    yOrig = scalerY.inverse_transform(y.cpu().numpy())
                    diff = predOrig - yOrig
                    valDiffSqSum += np.sum(diff**2, axis=0)
                    valCount += len(y)

            valMeanLoss = np.mean(valLosses)
            
            if valCount > 0:
                valRmse = np.sqrt(valDiffSqSum / valCount)
            else:
                valRmse = np.zeros(len(targetCols))

            print(f"Epoch [{epoch+1}] | Train: {trainMeanLoss:.4f} | Val: {valMeanLoss:.4f}")
            print(f"  >>> Val RMSE: {', '.join([f'{c}={v:.4f}' for c, v in zip(targetCols, valRmse)])}")

            # Save Best
            if valMeanLoss < bestLoss:
                bestLoss = valMeanLoss
                runName = f"best_multimodal_ep{config['nEpochs']}_seq{config['seqLength']}_dm{config['dModel']}"
                runPath = os.path.join(config['saveDir'], runName)
                
                # Cleanup old best
                if bestModelPath and os.path.exists(bestModelPath) and bestModelPath != runPath:
                    try:
                        shutil.rmtree(bestModelPath)
                    except:
                        pass
                
                os.makedirs(runPath, exist_ok=True)
                
                # Save weights & config
                torch.save(model.state_dict(), os.path.join(runPath, 'model.ckpt'))
                
                saveConfig = config.copy()
                saveConfig['featureCols'] = featureCols
                saveConfig['targetCols'] = targetCols
                saveConfig['valid_mse'] = bestLoss
                saveConfig['valid_rmse'] = {c: v for c, v in zip(targetCols, valRmse)}
                
                with open(os.path.join(runPath, 'config.json'), 'w') as f:
                    json.dump(saveConfig, f, indent=4)
                
                bestModelPath = runPath
                print(f"  >>> New Best Saved: {runName}")

        print(f"\nTraining Complete. Best Val Loss: {bestLoss:.4f}")
        print(f"Model saved to: {bestModelPath}")

        # --- Test Phase ---
        print("\nStep 4: Start Testing with Best Model...")
        if bestModelPath:
            # Load Best Model
            model.load_state_dict(torch.load(os.path.join(bestModelPath, 'model.ckpt')))
            model.eval()

            testLosses = []
            testDiffSqSum = np.zeros(len(targetCols))
            testCount = 0

            with torch.no_grad():
                for xImg, xStat, y in testLoader:
                    xImg, xStat, y = xImg.to(device), xStat.to(device), y.to(device)
                    pred = model(xImg, xStat)
                    loss = criterion(pred, y)
                    testLosses.append(loss.item())

                    # Metrics (Original Scale)
                    predOrig = scalerY.inverse_transform(pred.cpu().numpy())
                    predOrig = np.maximum(predOrig, 0)
                    yOrig = scalerY.inverse_transform(y.cpu().numpy())
                    diff = predOrig - yOrig
                    testDiffSqSum += np.sum(diff**2, axis=0)
                    testCount += len(y)
            
            testMeanLoss = np.mean(testLosses) if testLosses else 0
            if testCount > 0:
                testRmse = np.sqrt(testDiffSqSum / testCount)
            else:
                testRmse = np.zeros(len(targetCols))
            
            print(f"Test Loss (MSE): {testMeanLoss:.4f}")
            print(f"Test RMSE: {', '.join([f'{c}={v:.4f}' for c, v in zip(targetCols, testRmse)])}")

            # Update Config
            configPath = os.path.join(bestModelPath, 'config.json')
            if os.path.exists(configPath):
                with open(configPath, 'r') as f:
                    finalConfig = json.load(f)
                
                finalConfig['test_mse'] = testMeanLoss
                finalConfig['test_rmse'] = {c: v for c, v in zip(targetCols, testRmse)}
                
                with open(configPath, 'w') as f:
                    json.dump(finalConfig, f, indent=4)
                print("  >>> Updated config.json with Test metrics.")
        else:
            print("No model was saved.")

    except Exception as e:
        print(f"An error occurred: {e}")

# ==========================================
# 3. Main Entry Point
# ==========================================
if __name__ == '__main__':
    args = parse_args()

    # Configuration
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
        'saveDir': args.saveDir,
        'gamesPath': args.gamesPath,
        'shotsPath': args.shotsPath,
        'teamsPath': args.teamsPath,
        'trainSeasons': args.trainSeasons,
        'valSeasons': args.valSeasons,
        'testSeasons': args.testSeasons,
        'models': args.models
    }

    print(f"Selected Models: {config['models']}")
    
    for model_name in config['models']:
        print(f"\n{'='*40}")
        print(f"Running Training for: {model_name.upper()}")
        print(f"{'='*40}")
        
        if model_name == 'multi':
            train_multi(config)
        elif model_name == 'seq':
            # Note: seqModel might expect 'datasetPath' mapping, but I handled it in seqModel.py
            # to default to 'gamesPath' if 'datasetPath' is missing.
            train_seq(config)
        elif model_name == 'cnn':
            train_cnn(config)
        else:
            print(f"Unknown model type: {model_name}")

    # After all requested models are trained, run the baseline comparison
    train_baselines_and_compare(config)