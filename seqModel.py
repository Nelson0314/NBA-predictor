import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math
import random
import os
import json
import shutil
from tqdm import tqdm

# ==========================================
# 1. 固定隨機種子
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
# 2. 資料準備函式
# ==========================================
def loadAndPreprocessData(filePath, seqLength=10):
    print("Step 1: Loading and Cleaning Data...")
    
    # 讀取資料
    if not os.path.exists(filePath):
        raise FileNotFoundError(f"Data file not found at: {filePath}")
        
    gamesData = pd.read_csv(filePath, low_memory=False)

    # 定義特徵欄位與目標欄位
    featureCols = [
        'PTS', 'AST', 'REB', 
        'FGM', 'FGA', 'FG_PCT', 
        'FG3M', 'FG3A', 'FG3_PCT', 
        'FTM', 'FTA', 'FT_PCT', 
        'OREB', 'DREB', 
        'STL', 'BLK', 'TOV', 'PF', 
        'PLUS_MINUS', 'MIN', 'USG_PCT', 'OFF_RATING', 'DEF_RATING', 'PACE', 'TS_PCT'
    ]
    targetCols = ['PTS', 'AST', 'REB'] 

    # 強制將數值欄位轉為數字，無法轉換的變成 NaN (處理 Dirty Data)
    allCols = featureCols + targetCols
    for col in allCols:
        gamesData[col] = pd.to_numeric(gamesData[col], errors='coerce')

    # 移除髒資料
    gamesData = gamesData.dropna(subset=allCols)

    # 時間排序
    try:
        gamesData['GAME_DATE'] = pd.to_datetime(gamesData['GAME_DATE'])
    except: 
        pass
    gamesData = gamesData.sort_values(by=['Player_ID', 'GAME_DATE']).reset_index(drop=True)

    print(f"Data Loaded. Total Records: {len(gamesData)}")
    return gamesData, featureCols, targetCols

def createSequences(data, seqLength, featureCols, targetCols):
    """
    將資料轉換為序列 (Sliding Window)
    """
    print("Step 2: Generating Sequences...")
    xList, yList = [], []
    
    # 針對每個球員與賽季分組處理 (確保不跨賽季)
    if 'SEASON_ID' not in data.columns:
        print("Warning: 'SEASON_ID' not found in data. Grouping by 'Player_ID' only.")
        groups = data.groupby('Player_ID')
    else:
        groups = data.groupby(['Player_ID', 'SEASON_ID'])

    for groupKey, group in groups:
        if len(group) <= seqLength:
            continue
            
        features = group[featureCols].values
        targets = group[targetCols].values
        
        # 滑動視窗
        for i in range(len(group) - seqLength):
            x = features[i : i + seqLength]
            y = targets[i + seqLength]
            xList.append(x)
            yList.append(y)
            
    return np.array(xList), np.array(yList)

# ==========================================
# 3. Dataset 類別
# ==========================================
class NbaSequenceDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx]

    def __len__(self):
        return len(self.x)

# ==========================================
# 4. 模型架構 (Positional Encoding + Transformer)
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, dModel, maxLen=5000):
        super(PositionalEncoding, self).__init__()
        
        # 建立 (maxLen, dModel) 的矩陣
        pe = torch.zeros(maxLen, dModel)
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(1)
        
        # divTerm 計算: 1 / (10000 ^ (2i / dModel))
        divTerm = torch.exp(torch.arange(0, dModel, 2).float() * (-math.log(10000.0) / dModel))
        
        pe[:, 0::2] = torch.sin(position * divTerm)
        pe[:, 1::2] = torch.cos(position * divTerm)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batchSize, seqLen, dModel)
        x = x + self.pe[:, :x.size(1), :]
        return x

class NbaTransformer(nn.Module):
    def __init__(self, inputDim, dModel, nHead, numLayers, outputDim, dropout=0.1):
        super(NbaTransformer, self).__init__()
        
        self.embedding = nn.Linear(inputDim, dModel)
        self.posEncoder = PositionalEncoding(dModel)
        
        encoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHead, dropout=dropout, batch_first=True)
        self.transformerEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        
        self.decoder = nn.Sequential(
            nn.Linear(dModel, 32),
            nn.GELU(),
            nn.Linear(32, outputDim) 
        )

    def forward(self, x):
        x = self.embedding(x) 
        x = self.posEncoder(x)
        x = self.transformerEncoder(x) 
        lastTimeStep = x[:, -1, :] 
        out = self.decoder(lastTimeStep)
        return out

# ==========================================
# 5. 主程式 (Main Execution)
# ==========================================
if __name__ == '__main__':
    # 設定參數
    config = {
        'seed': 42,
        'seqLength': 10,
        'batchSize': 32,
        'nEpochs': 20,
        'learningRate': 0.001,
        'dModel': 64,
        'nHead': 4,
        'numLayers': 3,
        'dropout': 0.1,
        'saveDir': 'savedModels', # Changed from savePath to saveDir
        'datasetPath': 'dataset/games.csv',
        # 定義賽季切分
        'trainSeasons': [22016, 22017, 22018, 22019, 22020, 22021, 22022],
        'valSeasons': [22023], 
        'testSeasons': [22024]
    }

    # 1. 初始化
    setSeed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using Device: {device}")

    # 2. 資料處理
    try:
        gamesData, featureCols, targetCols = loadAndPreprocessData(config['datasetPath'], config['seqLength'])
        
        # 依照賽季 ID 切分 DataFrame
        trainData = gamesData[gamesData['SEASON_ID'].isin(config['trainSeasons'])].copy()
        valData = gamesData[gamesData['SEASON_ID'].isin(config['valSeasons'])].copy()
        testData = gamesData[gamesData['SEASON_ID'].isin(config['testSeasons'])].copy()

        print(f"Data Split Summary:")
        print(f"  Train Seasons: {config['trainSeasons']} | Records: {len(trainData)}")
        print(f"  Val Seasons:   {config['valSeasons']} | Records: {len(valData)}")
        print(f"  Test Seasons:  {config['testSeasons']} | Records: {len(testData)}")

        # 分別產生序列
        print("\nCreating Sequences for Training Set...")
        xTrain, yTrain = createSequences(trainData, config['seqLength'], featureCols, targetCols)
        
        print("Creating Sequences for Validation Set...")
        xVal, yVal = createSequences(valData, config['seqLength'], featureCols, targetCols)
        
        print("Creating Sequences for Test Set...")
        xTest, yTest = createSequences(testData, config['seqLength'], featureCols, targetCols)

        print(f"\nSequence Shapes:")
        print(f"  Train: x={xTrain.shape}, y={yTrain.shape}")
        print(f"  Val:   x={xVal.shape}, y={yVal.shape}")
        print(f"  Test:  x={xTest.shape}, y={yTest.shape}")

        if len(xTrain) == 0:
            raise ValueError("No training data generated! Check Season IDs.")

        # 標準化 (Features)
        scalerX = StandardScaler()
        
        # Fit on Train Features
        xTrainReshaped = xTrain.reshape(-1, len(featureCols))
        xTrainScaled = scalerX.fit_transform(xTrainReshaped).reshape(xTrain.shape)
        
        # Transform Valid Features
        xValReshaped = xVal.reshape(-1, len(featureCols))
        xValScaled = scalerX.transform(xValReshaped).reshape(xVal.shape)
        
        # Transform Test Features
        xTestReshaped = xTest.reshape(-1, len(featureCols))
        xTestScaled = scalerX.transform(xTestReshaped).reshape(xTest.shape)

        # 標準化 (Targets)
        scalerY = StandardScaler()
        
        # Fit on Train Targets
        yTrainScaled = scalerY.fit_transform(yTrain)
        
        # Transform Valid & Test Targets
        yValScaled = scalerY.transform(yVal)
        yTestScaled = scalerY.transform(yTest)

        # DataLoader
        trainDataset = NbaSequenceDataset(xTrainScaled, yTrainScaled)
        valDataset = NbaSequenceDataset(xValScaled, yValScaled)
        testDataset = NbaSequenceDataset(xTestScaled, yTestScaled)
        
        # num_workers=0 avoids Windows multiprocessing issues
        trainLoader = DataLoader(trainDataset, batch_size=config['batchSize'], shuffle=True, drop_last=True, num_workers=0)
        valLoader = DataLoader(valDataset, batch_size=config['batchSize'], shuffle=False, num_workers=0)
        testLoader = DataLoader(testDataset, batch_size=config['batchSize'], shuffle=False, num_workers=0)

        # 3. 建立模型
        model = NbaTransformer(
            inputDim=len(featureCols),
            dModel=config['dModel'],
            nHead=config['nHead'],
            numLayers=config['numLayers'],
            outputDim=len(targetCols),
            dropout=config['dropout']
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learningRate'], weight_decay=1e-5)

        # 4. 訓練迴圈
        bestLoss = float('inf')
        bestModelPath = "" # Store the path of the best model
        
        # Ensure save directory exists
        if not os.path.exists(config['saveDir']):
            os.makedirs(config['saveDir'])
            
        print("Step 3: Start Training...")

        for epoch in range(config['nEpochs']):
            # --- Training ---
            model.train()
            trainLossList = []
            
            # 使用 tqdm 顯示進度
            trainPbar = tqdm(trainLoader, desc=f"Epoch {epoch+1}/{config['nEpochs']}", leave=False)
            
            for x, y in trainPbar:
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                trainLossList.append(loss.item())

            trainMeanLoss = sum(trainLossList) / len(trainLossList)

            # --- Validation ---
            model.eval()
            valLossList = []
            valSquaredErrorSum = np.zeros(len(targetCols))
            valCount = 0

            with torch.no_grad():
                for x, y in valLoader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    loss = criterion(pred, y)
                    valLossList.append(loss.item())

                    # Calculate Original Scale Metrics
                    predOriginal = scalerY.inverse_transform(pred.cpu().numpy())
                    yOriginal = scalerY.inverse_transform(y.cpu().numpy())
                    
                    diff = predOriginal - yOriginal
                    valSquaredErrorSum += np.sum(diff ** 2, axis=0)
                    valCount += len(y)
                    
            valMeanLoss = sum(valLossList) / len(valLossList)
            
            # Avoid division by zero
            if valCount > 0:
                valMseOriginal = valSquaredErrorSum / valCount
                valRmseOriginal = np.sqrt(valMseOriginal)
            else:
                valRmseOriginal = np.zeros(len(targetCols))
            
            # Print 結果 (這會顯示在 Slurm 的 output file 中)
            print(f"Epoch [{epoch+1}/{config['nEpochs']}] | Train Loss: {trainMeanLoss:.4f} | Val Loss: {valMeanLoss:.4f}")
            print(f"  >>> Val RMSE (Original): {', '.join([f'{col}={val:.4f}' for col, val in zip(targetCols, valRmseOriginal)])}")


            # 儲存最佳模型
            if valMeanLoss < bestLoss:
                bestLoss = valMeanLoss
                
                # Define Run Name (Used as Folder Name)
                runName = f"best_run_ep{config['nEpochs']}_seq{config['seqLength']}_d{config['dModel']}_head{config['nHead']}_lr{config['learningRate']}_bs{config['batchSize']}"
                runPath = os.path.join(config['saveDir'], runName)
                
                # Cleanup previous best model folder if it exists and is different
                if bestModelPath and os.path.exists(bestModelPath) and bestModelPath != runPath:
                    try:
                        shutil.rmtree(bestModelPath)
                        print(f"  >>> Removed previous best run: {bestModelPath}")
                    except OSError as e:
                        print(f"  >>> Error removing previous run: {e}")

                # Create new run folder
                os.makedirs(runPath, exist_ok=True)
                
                # Save Model Checkpoint
                ckptPath = os.path.join(runPath, 'model.ckpt')
                torch.save(model.state_dict(), ckptPath)
                
                # Save Config as JSON
                configPath = os.path.join(runPath, 'config.json')
                # Add current featureCols to config for inference consistency
                saveConfig = config.copy()
                saveConfig['featureCols'] = featureCols
                saveConfig['targetCols'] = targetCols
                saveConfig['valid_mse'] = bestLoss
                saveConfig['valid_rmse_original'] = {col: val for col, val in zip(targetCols, valRmseOriginal)}
                
                with open(configPath, 'w') as f:
                    json.dump(saveConfig, f, indent=4)

                bestModelPath = runPath
                print(f"  >>> New Best Model & Config Saved to: {runPath}")
        
        # --- Testing Phase ---
        print("\nStep 4: Start Testing with Best Model...")
        
        if bestModelPath:
            # Load best model
            print(f"Loading best model from: {bestModelPath}")
            ckptLoadPath = os.path.join(bestModelPath, 'model.ckpt')
            model.load_state_dict(torch.load(ckptLoadPath))
            model.eval()
            
            testLossList = []
            testSquaredErrorSum = np.zeros(len(targetCols))
            testCount = 0

            with torch.no_grad():
                for x, y in testLoader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    loss = criterion(pred, y)
                    testLossList.append(loss.item())

                    # Calculate Original Scale Metrics
                    predOriginal = scalerY.inverse_transform(pred.cpu().numpy())
                    yOriginal = scalerY.inverse_transform(y.cpu().numpy())
                    
                    diff = predOriginal - yOriginal
                    testSquaredErrorSum += np.sum(diff ** 2, axis=0)
                    testCount += len(y)
            
            testMeanLoss = sum(testLossList) / len(testLossList) if testLossList else 0
            
            if testCount > 0:
                testMseOriginal = testSquaredErrorSum / testCount
                testRmseOriginal = np.sqrt(testMseOriginal)
            else:
                testRmseOriginal = np.zeros(len(targetCols))
            print(f"\nTraining Complete. Best Validation Loss: {bestLoss:.4f}")
            print(f"Test Loss (MSE): {testMeanLoss:.4f}")
            print(f"Test RMSE (Original): {', '.join([f'{col}={val:.4f}' for col, val in zip(targetCols, testRmseOriginal)])}")
            print(f"Model saved to: {bestModelPath}")

            # Append Test Metric to Config
            configPath = os.path.join(bestModelPath, 'config.json')
            if os.path.exists(configPath):
                try:
                    with open(configPath, 'r') as f:
                        finalConfig = json.load(f)
                    
                    finalConfig['test_mse'] = testMeanLoss
                    finalConfig['test_rmse_original'] = {col: val for col, val in zip(targetCols, testRmseOriginal)}
                    
                    with open(configPath, 'w') as f:
                        json.dump(finalConfig, f, indent=4)
                    print("  >>> Updated config.json with Test MSE.")
                except Exception as e:
                    print(f"  >>> Warning: Could not update config with test loss: {e}")
        else:
            print("No model was saved during training.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")