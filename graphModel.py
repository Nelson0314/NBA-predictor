import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage import gaussian_filter   
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
# 2. 資料準備函式 (熱力圖專用)
# ==========================================
def generateHeatmap(shotsDf, imgSize=50, sigma=1.0):
    """
    將投籃數據轉換為 (2, 50, 50) 的 Tensor
    Channel 0: 出手位置 (Attempts)
    Channel 1: 命中位置 (Made)
    """
    # 初始化
    heatmapTensor = np.zeros((2, imgSize, imgSize))
    
    if len(shotsDf) == 0:
        return torch.FloatTensor(heatmapTensor)

    # 座標範圍 (NBA半場: X -250~250, Y -50~420)
    xEdges = np.linspace(-250, 250, imgSize + 1)
    yEdges = np.linspace(-50, 450, imgSize + 1)

    x = shotsDf['LOC_X'].values
    y = shotsDf['LOC_Y'].values
    made = shotsDf['SHOT_MADE_FLAG'].values

    # Channel 0: Attempts
    histAttempts, _, _ = np.histogram2d(x, y, bins=[xEdges, yEdges])
    
    # Channel 1: Made
    madeMask = (made == 1)
    if np.sum(madeMask) > 0:
        histMade, _, _ = np.histogram2d(x[madeMask], y[madeMask], bins=[xEdges, yEdges])
    else:
        histMade = np.zeros_like(histAttempts)

    # 轉置與模糊化 (讓點變成區域)
    heatmapTensor[0] = gaussian_filter(histAttempts.T, sigma=sigma)
    heatmapTensor[1] = gaussian_filter(histMade.T, sigma=sigma)

    # 正規化 (Min-Max Scaling)
    maxVal = np.max(heatmapTensor) + 1e-9
    heatmapTensor /= maxVal

    return torch.FloatTensor(heatmapTensor)

def loadAndPreprocessData(gamesPath, shotsPath):
    print("Step 1: Loading and Cleaning Data...")
    
    if not os.path.exists(gamesPath) or not os.path.exists(shotsPath):
        raise FileNotFoundError("Files not found. Please check games.csv and shots.csv paths.")

    # 1. 讀取 Games (為了取得 Target: PTS, AST, REB)
    gamesData = pd.read_csv(gamesPath, low_memory=False)
    # 2. 讀取 Shots (為了取得 Input: LOC_X, LOC_Y)
    shotsData = pd.read_csv(shotsPath, low_memory=False)

    targetCols = ['PTS', 'AST', 'REB']
    
    # 清洗 Games Data
    gamesData['GAME_DATE'] = pd.to_datetime(gamesData['GAME_DATE'], errors='coerce')
    for col in targetCols:
        gamesData[col] = pd.to_numeric(gamesData[col], errors='coerce')
    gamesData = gamesData.dropna(subset=targetCols)
    gamesData = gamesData.sort_values(by=['Player_ID', 'GAME_DATE']).reset_index(drop=True)

    # 清洗 Shots Data
    shotsData = shotsData[['Player_ID', 'GAME_ID', 'LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG']].copy()
    shotsData = shotsData.dropna()

    # 建立快速索引 (為了在生成序列時快速抓取某場比賽的投籃)
    # Group by (Player_ID, GAME_ID)
    # 建立快速索引 (為了在生成序列時快速抓取某場比賽的投籃)
    # Group by (Player_ID, GAME_ID)
    # print("Indexing shots data...")
    # shotsGrouped = dict(list(shotsData.groupby(['Player_ID', 'GAME_ID'])))
    shotsGrouped = None # Use pre-generated heatmaps

    print(f"Data Loaded. Games: {len(gamesData)}, Shot Groups: (Loaded from disk)")
    return gamesData, shotsGrouped, targetCols

def createCnnSequences(gamesData, shotsGrouped, seqLength, targetCols):
    """
    Input: 過去 N 場比賽的累積投籃圖 (Aggregate Heatmap)
    Output: 下一場比賽的數據 (Target)
    """
    print("Step 2: Generating Heatmaps & Targets...")
    xList, yList = [], []
    
    groups = gamesData.groupby(['Player_ID', 'SEASON_ID']) if 'SEASON_ID' in gamesData.columns else gamesData.groupby('Player_ID')

    for groupKey, group in tqdm(groups, desc="Processing Players"):
        if len(group) <= seqLength:
            continue
        
        # 取得該球員該賽季的所有 GameIDs 和 Targets
        gameIds = group['GAME_ID'].values
        playerId = group['Player_ID'].values[0]
        targets = group[targetCols].values
        
        # 滑動視窗
        for i in range(len(group) - seqLength):
            target = targets[i + seqLength]
            pastGameIds = gameIds[i : i + seqLength]
            
            # Result Shape: (seqLength * 2, 50, 50)
            dailyHeatmaps = []
            
            for gid in pastGameIds:
                # Optimized: Load from disk
                # key = (playerId, gid)
                # if key in shotsGrouped:
                #     # 單場比賽熱圖 (2, 50, 50)
                #     h = generateHeatmap(shotsGrouped[key])
                # else:
                #     # 缺少投籃數據，補零圖
                #     h = torch.zeros((2, 50, 50), dtype=torch.float32)
                
                heatmapPath = os.path.join('dataset/heatmaps', f"{int(playerId)}_{str(gid).zfill(10)}.npy")
                if os.path.exists(heatmapPath):
                    h = np.load(heatmapPath) # (2, 50, 50)
                else:
                    h = np.zeros((2, 50, 50), dtype=np.float32)
                
                dailyHeatmaps.append(h)

            # Stack along channel dimension (axis=0)
            # List of N items of shape (2, 50, 50) -> (2*N, 50, 50)
            stackedInput = np.concatenate(dailyHeatmaps, axis=0)

            xList.append(stackedInput)
            yList.append(target)
            
    return np.array(xList), np.array(yList)

# ==========================================
# 3. Dataset 類別
# ==========================================
class NbaCnnDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.FloatTensor(x) # Shape: (N, 2, 50, 50)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx]

    def __len__(self):
        return len(self.x)

# ==========================================
# 4. 模型架構 (CNN)
# ==========================================
class NbaCnn(nn.Module):
    def __init__(self, outputDim, inputChannels=2):
        super(NbaCnn, self).__init__()
        
        # Input: (Batch, inputChannels, 50, 50)
        # inputChannels = seqLength * 2 (because of Channel Stacking)
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(inputChannels, 16, kernel_size=3, padding=1), # -> (16, 50, 50)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # -> (16, 25, 25)
            
            # Conv Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # -> (32, 25, 25)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # -> (32, 12, 12)
            
            # Conv Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> (64, 12, 12)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # -> (64, 6, 6)
        )
        
        self.flatten = nn.Flatten()
        
        # Calculate size after convolutions: 64 * 6 * 6 = 2304
        self.regressor = nn.Sequential(
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, outputDim) # Predict PTS, AST, REB
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.regressor(x)
        return x

# ==========================================
# 5. 主程式 (Main Execution)
# ==========================================

def train(config):
    # 1. 初始化
    setSeed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using Device: {device}")

    # 2. 資料處理
    try:
        gamesData, shotsGrouped, targetCols = loadAndPreprocessData(config['gamesPath'], config['shotsPath'])
        
        # 依照賽季 ID 切分 DataFrame (只切分 GamesData，Shots 透過 Grouped 字典查詢)
        trainGames = gamesData[gamesData['SEASON_ID'].isin(config['trainSeasons'])].copy()
        valGames = gamesData[gamesData['SEASON_ID'].isin(config['valSeasons'])].copy()
        testGames = gamesData[gamesData['SEASON_ID'].isin(config['testSeasons'])].copy()

        print(f"Data Split Summary:")
        print(f"  Train Records: {len(trainGames)}")
        print(f"  Val Records:   {len(valGames)}")
        print(f"  Test Records:  {len(testGames)}")

        # 產生圖像序列
        # 注意：這一步會比純數值運算慢，因為要動態生成圖片
        print("\\nCreating Heatmaps for Training Set...")
        xTrain, yTrain = createCnnSequences(trainGames, shotsGrouped, config['seqLength'], targetCols)
        
        print("Creating Heatmaps for Validation Set...")
        xVal, yVal = createCnnSequences(valGames, shotsGrouped, config['seqLength'], targetCols)
        
        print("Creating Heatmaps for Test Set...")
        xTest, yTest = createCnnSequences(testGames, shotsGrouped, config['seqLength'], targetCols)

        print(f"\\nTensor Shapes:")
        print(f"  Train: x={xTrain.shape}, y={yTrain.shape}") # Expect (N, seq*2, 50, 50)
        
        if len(xTrain) == 0:
            raise ValueError("No training data generated!")

        # 標準化 (Targets) - 圖片不需要 StandardScale，已經在 generateHeatmap 做過 MinMax
        # 標準化 (Targets) - 圖片不需要 StandardScale，已經在 generateHeatmap 做過 MinMax
        scalerY = MinMaxScaler(feature_range=(0, 1))
        yTrainScaled = scalerY.fit_transform(yTrain)
        yValScaled = scalerY.transform(yVal)
        yTestScaled = scalerY.transform(yTest)

        # DataLoader
        trainDataset = NbaCnnDataset(xTrain, yTrainScaled)
        valDataset = NbaCnnDataset(xVal, yValScaled)
        testDataset = NbaCnnDataset(xTest, yTestScaled)
        
        trainLoader = DataLoader(trainDataset, batch_size=config['batchSize'], shuffle=True, drop_last=True, num_workers=0)
        valLoader = DataLoader(valDataset, batch_size=config['batchSize'], shuffle=False, num_workers=0)
        testLoader = DataLoader(testDataset, batch_size=config['batchSize'], shuffle=False, num_workers=0)

        # 3. 建立模型 (CNN)
        # Input Channels = seqLength * 2 (每場比賽 2 個 channel: Attempts, Made)
        inputCh = config['seqLength'] * 2
        print(f"Model Input Channels: {inputCh}")
        model = NbaCnn(outputDim=len(targetCols), inputChannels=inputCh).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learningRate'])

        # 4. 訓練迴圈
        bestLoss = float('inf')
        bestModelPath = "" 
        
        if not os.path.exists(config['saveDir']):
            os.makedirs(config['saveDir'])
            
        print("Step 3: Start Training (CNN)...")

        for epoch in range(config['nEpochs']):
            # --- Training ---
            model.train()
            trainLossList = []
            
            trainPbar = tqdm(trainLoader, desc=f"Epoch {epoch+1}/{config['nEpochs']}", leave=False)
            
            for x, y in trainPbar:
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                
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

                    predOriginal = scalerY.inverse_transform(pred.cpu().numpy())
                    predOriginal = np.maximum(predOriginal, 0)
                    yOriginal = scalerY.inverse_transform(y.cpu().numpy())
                    
                    diff = predOriginal - yOriginal
                    valSquaredErrorSum += np.sum(diff ** 2, axis=0)
                    valCount += len(y)
                    
            valMeanLoss = sum(valLossList) / len(valLossList)
            
            if valCount > 0:
                valMseOriginal = valSquaredErrorSum / valCount
                valRmseOriginal = np.sqrt(valMseOriginal)
            else:
                valRmseOriginal = np.zeros(len(targetCols))
            
            print(f"Epoch [{epoch+1}/{config['nEpochs']}] | Train Loss: {trainMeanLoss:.4f} | Val Loss: {valMeanLoss:.4f}")
            print(f"  >>> Val RMSE: {', '.join([f'{col}={val:.4f}' for col, val in zip(targetCols, valRmseOriginal)])}")

            # 儲存最佳模型
            if valMeanLoss < bestLoss:
                bestLoss = valMeanLoss
                
                runName = f"best_cnn_ep{config['nEpochs']}_seq{config['seqLength']}_lr{config['learningRate']}"
                runPath = os.path.join(config['saveDir'], runName)
                
                if bestModelPath and os.path.exists(bestModelPath) and bestModelPath != runPath:
                    try:
                        shutil.rmtree(bestModelPath)
                    except OSError: pass

                os.makedirs(runPath, exist_ok=True)
                
                torch.save(model.state_dict(), os.path.join(runPath, 'model.ckpt'))
                
                # Config
                saveConfig = config.copy()
                saveConfig['valid_mse'] = bestLoss
                saveConfig['valid_rmse'] = {c: v for c, v in zip(targetCols, valRmseOriginal)}
                
                with open(os.path.join(runPath, 'config.json'), 'w') as f:
                    json.dump(saveConfig, f, indent=4)

                bestModelPath = runPath
                print(f"  >>> New Best Model Saved to: {runPath}")
        
        # --- Testing Phase ---
        print("\\nStep 4: Start Testing with Best Model...")
        
        if bestModelPath:
            model.load_state_dict(torch.load(os.path.join(bestModelPath, 'model.ckpt')))
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

                    predOriginal = scalerY.inverse_transform(pred.cpu().numpy())
                    predOriginal = np.maximum(predOriginal, 0)
                    yOriginal = scalerY.inverse_transform(y.cpu().numpy())
                    
                    diff = predOriginal - yOriginal
                    testSquaredErrorSum += np.sum(diff ** 2, axis=0)
                    testCount += len(y)
            
            testMeanLoss = sum(testLossList) / len(testLossList) if testLossList else 0
            
            if testCount > 0:
                testRmseOriginal = np.sqrt(testSquaredErrorSum / testCount)
            else:
                testRmseOriginal = np.zeros(len(targetCols))
                
            print(f"Test Loss (MSE): {testMeanLoss:.4f}")
            print(f"Test RMSE: {', '.join([f'{col}={val:.4f}' for col, val in zip(targetCols, testRmseOriginal)])}")
            print(f"Model saved to: {bestModelPath}")

            # Update Config with Test Results
            configPath = os.path.join(bestModelPath, 'config.json')
            if os.path.exists(configPath):
                with open(configPath, 'r') as f:
                    finalConfig = json.load(f)
                
                finalConfig['test_mse'] = testMeanLoss
                finalConfig['test_rmse'] = {c: v for c, v in zip(targetCols, testRmseOriginal)}
                
                with open(configPath, 'w') as f:
                    json.dump(finalConfig, f, indent=4)
                print("  >>> Updated config.json with Test metrics.")

    except Exception as e:
        print(f"\\nAn error occurred: {e}")

if __name__ == '__main__':
    # 設定參數
    config = {
        'seed': 42,
        'seqLength': 5, # 累積過去 5 場的熱圖
        'batchSize': 32,
        'nEpochs': 20,
        'learningRate': 0.001,
        'saveDir': 'savedModels',
        'gamesPath': 'dataset/games.csv',
        'shotsPath': 'dataset/shots.csv', # 必須有 shots.csv
        'trainSeasons': [22016, 22017, 22018, 22019, 22020, 22021, 22022],
        'valSeasons': [22023], 
        'testSeasons': [22024]
    }
    train(config)