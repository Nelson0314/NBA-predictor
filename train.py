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

# Import local modules
from multimodalModel import NbaMultimodal, loadAndPreprocessData, createMultimodalSequences, MultimodalDataset

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

# ==========================================
# 2. 主程式 (Main)
# ==========================================
if __name__ == '__main__':
    # Configuration
    config = {
        'seed': 42,
        'seqLength': 10,
        'batchSize': 32,
        'nEpochs': 20,
        'learningRate': 1e-3,
        'cnnEmbedDim': 64,
        'statEmbedDim': 32,
        'dModel': 128,
        'nHead': 4,
        'numLayers': 3,
        'dropout': 0.2,
        'saveDir': 'savedMultimodalModels',
        'gamesPath': 'dataset/games.csv',
        'shotsPath': 'dataset/shots.csv',
        'teamsPath': 'dataset/teams.csv',
        'trainSeasons': [22016, 22017, 22018, 22019, 22020, 22021, 22022],
        'valSeasons': [22023],
        'testSeasons': [22024]
    }

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

        # --- Generate Sequences & Pre-compute Images ---
        # Note: creating sequences involves generating heatmaps which is slow.
        # We do it once here.
        
        print("\nGenerating Train Sequences...")
        xImgTrain, xStatTrain, yTrain = createMultimodalSequences(trainData, shotsGrouped, config['seqLength'], featureCols, targetCols)
        
        print("Generating Val Sequences...")
        xImgVal, xStatVal, yVal = createMultimodalSequences(valData, shotsGrouped, config['seqLength'], featureCols, targetCols)
        
        print("Generating Test Sequences...")
        xImgTest, xStatTest, yTest = createMultimodalSequences(testData, shotsGrouped, config['seqLength'], featureCols, targetCols)

        if len(xImgTrain) == 0:
            raise ValueError("No training data generated!")

        # --- Scaling (Stats Only) ---
        print("\nScaling Data...")
        scalerX = StandardScaler()
        scalerY = MinMaxScaler(feature_range=(0, 1))

        # Scale Features
        # xStatTrain: (N, Seq, Feat) -> Reshape -> Scale -> Reshape
        N, S, F = xStatTrain.shape
        xStatTrainScaled = scalerX.fit_transform(xStatTrain.reshape(-1, F)).reshape(N, S, F)
        xStatValScaled = scalerX.transform(xStatVal.reshape(-1, F)).reshape(xStatVal.shape)
        xStatTestScaled = scalerX.transform(xStatTest.reshape(-1, F)).reshape(xStatTest.shape)

        # Scale Targets
        yTrainScaled = scalerY.fit_transform(yTrain)
        yValScaled = scalerY.transform(yVal)
        yTestScaled = scalerY.transform(yTest)

        # --- Datasets & Loaders ---
        trainDataset = MultimodalDataset(xImgTrain, xStatTrainScaled, yTrainScaled)
        valDataset = MultimodalDataset(xImgVal, xStatValScaled, yValScaled)
        testDataset = MultimodalDataset(xImgTest, xStatTestScaled, yTestScaled)

        trainLoader = DataLoader(trainDataset, batch_size=config['batchSize'], shuffle=True, drop_last=True, num_workers=0)
        valLoader = DataLoader(valDataset, batch_size=config['batchSize'], shuffle=False, num_workers=0)
        testLoader = DataLoader(testDataset, batch_size=config['batchSize'], shuffle=False, num_workers=0)

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
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learningRate'], weight_decay=1e-5)
        # Cosine Annealing Scheduler: T_max corresponds to total epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['nEpochs'], eta_min=1e-6)

        # --- Training Loop ---
        print("\nStep 3: Start Training...")
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

    except Exception as e:
        print(f"An error occurred: {e}")
