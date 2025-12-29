import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import random
import os
import json
import shutil
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt

# ==========================================
# 0. Utils & Loss
# ==========================================
def calculate_odds(house_pred, line, std_dev=9.0):
    """
    Calculate decimal odds based on the House Prediction vs the Line.
    Assumes normal distribution with fixed std_dev (approx for NBA player props).
    """
    z = (line - house_pred) / std_dev
    prob_over = 1 - norm.cdf(z)
    prob_under = 1.0 - prob_over
    
    # Avoid infinite odds
    prob_over = max(0.01, min(0.99, prob_over))
    prob_under = max(0.01, min(0.99, prob_under))
    
    odds_over = 1.0 / prob_over
    odds_under = 1.0 / prob_under
    
    return odds_over, odds_under, prob_over

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles # [0.1, 0.5, 0.9]

    def forward(self, preds, target):
        loss = 0
        for i, q in enumerate(self.quantiles):
            q_preds = preds[:, :, i] 
            errors = target - q_preds
            loss += torch.max((q - 1) * errors, q * errors).mean()
        return loss

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
def loadAndPreprocessData(filePath, teamsPath="dataset/teams.csv", seqLength=10):
    print("Step 1: Loading and Cleaning Data...")
    
    # 讀取資料
    if not os.path.exists(filePath):
        raise FileNotFoundError(f"Data file not found at: {filePath}")
        
    gamesData = pd.read_csv(filePath, low_memory=False, dtype={'GAME_ID': str}) # Force GAME_ID as str
    gamesData = gamesData.loc[:, ~gamesData.columns.duplicated()]

    # Load Teams Data
    if not os.path.exists(teamsPath):
        # Fallback if just filename provided
        teamsPath = os.path.join(os.path.dirname(filePath), 'teams.csv')
        
    teamsData = pd.read_csv(teamsPath, low_memory=False, dtype={'GAME_ID': str}) # Force GAME_ID as str

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
    allCols = list(set(featureCols + targetCols))
    for col in allCols:
        if col in gamesData.columns:
            gamesData[col] = pd.to_numeric(gamesData[col], errors='coerce')

    # 移除髒資料
    gamesData = gamesData.dropna(subset=allCols)

    # 時間排序
    gamesData['GAME_DATE'] = pd.to_datetime(gamesData['GAME_DATE'], errors='coerce')
    gamesData = gamesData.dropna(subset=['GAME_DATE'])
    gamesData = gamesData.sort_values(by=['Player_ID', 'GAME_DATE']).reset_index(drop=True)

    print(f"Data Loaded. Total Records: {len(gamesData)}")
    
    # ---------------------------------------------------------
    # Feature Engineering (Season-to-Date)
    # ---------------------------------------------------------
    print("Calculating Rolling Features...")
    
    # Parse Matchup for Opponent
    def parse_matchup(m):
        if pd.isna(m): return None, None
        if ' vs. ' in m:
            parts = m.split(' vs. ')
            return parts[0], parts[1]
        elif ' @ ' in m:
            parts = m.split(' @ ')
            return parts[0], parts[1]
        return None, None

    matchups = gamesData['MATCHUP'].apply(parse_matchup)
    gamesData['TEAM_ABBREVIATION'] = [x[0] for x in matchups]
    gamesData['OPPONENT_ABBREVIATION'] = [x[1] for x in matchups]
    
    # ---------------------------------------------------------
    # Merge Team & Opponent Rolling Stats (from teams.csv) - ADDED
    # ---------------------------------------------------------
    print("Merging Team Stats (Own + Opponent)...")
    
    # Normalize GAME_ID to 10-digit string
    gamesData['GAME_ID'] = pd.to_numeric(gamesData['GAME_ID'], errors='coerce').fillna(0).astype('int64').astype(str).str.zfill(10)
    teamsData['GAME_ID'] = pd.to_numeric(teamsData['GAME_ID'], errors='coerce').fillna(0).astype('int64').astype(str).str.zfill(10)

    # Identify stats columns in teams.csv (AVG_*)
    teamStatsCols = [c for c in teamsData.columns if c.startswith('AVG_')]
    
    # Subset for merging
    teamsSubset = teamsData[['GAME_ID', 'TEAM_ABBREVIATION'] + teamStatsCols].copy()

    # Safety Fix: Shift features to avoid leakage (Just like MultiModel)
    teamsSubset = teamsSubset.sort_values(by=['TEAM_ABBREVIATION', 'GAME_ID'])
    stats_cols_only = [c for c in teamsSubset.columns if c.startswith('AVG_')]
    teamsSubset[stats_cols_only] = teamsSubset.groupby('TEAM_ABBREVIATION')[stats_cols_only].shift(1).fillna(0)
    
    # 1. Merge Own Team Stats
    renameOwn = {c: f'TEAM_{c}' for c in teamStatsCols}
    teamsSubsetOwn = teamsSubset.rename(columns=renameOwn)
    
    gamesData = pd.merge(
        gamesData, 
        teamsSubsetOwn, 
        how='left', 
        on=['GAME_ID', 'TEAM_ABBREVIATION']
    )
    
    # 2. Merge Opponent Team Stats
    renameOpp = {c: f'OPP_{c}' for c in teamStatsCols}
    teamsSubsetOpp = teamsSubset.rename(columns=renameOpp)
    
    gamesData = pd.merge(
        gamesData,
        teamsSubsetOpp,
        how='left',
        left_on=['GAME_ID', 'OPPONENT_ABBREVIATION'],
        right_on=['GAME_ID', 'TEAM_ABBREVIATION'],
        suffixes=('', '_opp_merge')
    )
    
    if 'TEAM_ABBREVIATION_opp_merge' in gamesData.columns:
        gamesData = gamesData.drop(columns=['TEAM_ABBREVIATION_opp_merge'])

    newTeamCols = list(renameOwn.values()) + list(renameOpp.values())
    for col in newTeamCols:
        if col not in gamesData.columns:
            gamesData[col] = 0.0
        gamesData[col] = gamesData[col].fillna(0)
        
    featureCols.extend(newTeamCols)
    print(f"Team Features Added: {len(newTeamCols)} columns.")

    # Sort Back
    gamesData = gamesData.sort_values(by=['Player_ID', 'SEASON_ID', 'GAME_DATE'])
    
    # Player Rolling
    p_cols = ['PTS', 'AST', 'REB']
    new_p_cols = [f'PLAYER_AVG_{c}' for c in p_cols]
    
    for c, new_c in zip(p_cols, new_p_cols):
        gamesData[new_c] = gamesData.groupby(['Player_ID', 'SEASON_ID'])[c].transform(lambda x: x.expanding().mean().shift(1)).fillna(0)
        
    featureCols.extend(new_p_cols)

    # ---------------------------------------------------------
    # Feature: Days Since Last Game
    # ---------------------------------------------------------
    print("Calculating Days Since Last Game...")
    # Calculate days diff for each player
    gamesData['DAYS_SINCE_LAST_GAME'] = gamesData.groupby('Player_ID')['GAME_DATE'].diff().dt.days
    
    # Fill NaN with default (e.g., 7 days)
    gamesData['DAYS_SINCE_LAST_GAME'] = gamesData['DAYS_SINCE_LAST_GAME'].fillna(7)
    
    featureCols.append('DAYS_SINCE_LAST_GAME')
    
    return gamesData, featureCols, targetCols

def createSequences(data, seqLength, featureCols, targetCols):
    """
    將資料轉換為序列 (Sliding Window)
    """
    print("Step 2: Generating Sequences...")
    xList, yList, metaList = [], [], []
    
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
        player_ids = group['Player_ID'].values
        
        # 滑動視窗
        for i in range(len(group) - seqLength):
            x = features[i : i + seqLength]
            y = targets[i + seqLength]
            p_id = player_ids[i + seqLength] # ID of the target game
            
            xList.append(x)
            yList.append(y)
            metaList.append(p_id)
            
    return np.array(xList), np.array(yList), np.array(metaList)

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
    def __init__(self, inputDim, dModel, nHead, numLayers, outputDim, statEmbedDim=128, dropout=0.1):
        super(NbaTransformer, self).__init__()
        
        # 1. Stat Encoder (MLP) - Matching MultiModel exactly
        # Multi: Linear(In, 128) -> GELU -> Drop -> Linear(128, statEmbedDim) -> GELU
        self.statEncoder = nn.Sequential(
            nn.Linear(inputDim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, statEmbedDim),
            nn.GELU()
        )
        
        # 2. Projection to dModel
        self.projection = nn.Linear(statEmbedDim, dModel)
        
        self.posEncoder = PositionalEncoding(dModel)
        
        encoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHead, dropout=dropout, batch_first=True)
        self.transformerEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        
        # 3. Head - Matching MultiModel exactly
        # Multi: Linear(dModel, 64) -> ReLU -> Drop -> Linear(64, Out)
        self.decoder = nn.Sequential(
            nn.Linear(dModel, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, outputDim * 3) # Output 3 quantiles per target
        )

    def forward(self, x):
        x = self.statEncoder(x)
        x = self.projection(x)
        x = self.posEncoder(x)
        x = self.transformerEncoder(x) 
        lastTimeStep = x[:, -1, :] 
        out = self.decoder(lastTimeStep)
        
        # Reshape to (Batch, NumTargets, 3)
        return out.view(x.size(0), -1, 3)

# ==========================================
# 5. 主程式 (Main Execution)
# ==========================================
def train(config):
    # 1. 初始化
    setSeed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using Device: {device}")
    
    datasetPath = config.get('datasetPath', config.get('gamesPath'))

    # 2. 資料處理
    try:
        teamsPath = config.get('teamsPath', "dataset/teams.csv")
        gamesData, featureCols, targetCols = loadAndPreprocessData(datasetPath, teamsPath, config['seqLength'])
        
        # 依照賽季 ID 切分 DataFrame
        trainData = gamesData[gamesData['SEASON_ID'].isin(config['trainSeasons'])].copy()
        valData = gamesData[gamesData['SEASON_ID'].isin(config['valSeasons'])].copy()
        testData = gamesData[gamesData['SEASON_ID'].isin(config['testSeasons'])].copy()

        # Create Player ID -> Name Mapping
        if 'Player_Name' in gamesData.columns:
            id_to_name = gamesData[['Player_ID', 'Player_Name']].drop_duplicates(subset='Player_ID').set_index('Player_ID')['Player_Name'].to_dict()
        else:
            id_to_name = {}

        print(f"Data Split Summary:")
        print(f"  Train Seasons: {config['trainSeasons']} | Records: {len(trainData)}")
        print(f"  Val Seasons:   {config['valSeasons']} | Records: {len(valData)}")
        print(f"  Test Seasons:  {config['testSeasons']} | Records: {len(testData)}")

        # 分別產生序列
        print("\\nCreating Sequences for Training Set...")
        xTrain, yTrain, metaTrain = createSequences(trainData, config['seqLength'], featureCols, targetCols)
        
        print("Creating Sequences for Validation Set...")
        xVal, yVal, metaVal = createSequences(valData, config['seqLength'], featureCols, targetCols)
        
        print("Creating Sequences for Test Set...")
        xTest, yTest, metaTest = createSequences(testData, config['seqLength'], featureCols, targetCols)

        print(f"\\nSequence Shapes:")
        print(f"  Train: x={xTrain.shape}, y={yTrain.shape}")
        print(f"  Val:   x={xVal.shape}, y={yVal.shape}")
        print(f"  Test:  x={xTest.shape}, y={yTest.shape}")

        if len(xTrain) == 0:
            raise ValueError("No training data generated! Check Season IDs.")

        # --- Comprehensive Baselines ---
        print("\nCalculating Baselines (Naive, LR, XGBoost)...")
        
        # 1. Naive (Mean)
        trainMean = np.mean(yTrain, axis=0)
        naiveError = yVal - trainMean
        naiveMSE = np.mean(naiveError ** 2)
        naiveMAE = np.mean(np.abs(naiveError))
        print(f"  [Naive] Val MSE: {naiveMSE:.4f} | Val MAE: {naiveMAE:.4f}")

        # Flatten for LR/XGB
        N_tr, S_tr, F_tr = xTrain.shape
        xTrainFlat = xTrain.reshape(N_tr, -1)
        
        N_val, S_val, F_val = xVal.shape
        xValFlat = xVal.reshape(N_val, -1)
        
        # 2. Linear Regression
        try:
            lr = LinearRegression()
            lr.fit(xTrainFlat, yTrain)
            lrPred = lr.predict(xValFlat)
            lrMSE = np.mean((yVal - lrPred) ** 2)
            lrMAE = np.mean(np.abs(yVal - lrPred))
            print(f"  [Linear] Val MSE: {lrMSE:.4f} | Val MAE: {lrMAE:.4f}")
        except Exception as e:
            print(f"  [Linear] Error: {e}")

        # 3. XGBoost
        try:
            xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, n_jobs=-1, random_state=42)
            # XGBoost handles multi-output naturally? 
            # Standard XGBRegressor supports multi-class but for multi-target regression it might need MultiOutputRegressor wrapper 
            # or it might handle it if output dim > 1. 
            # Recent XGBoost versions support multi-output regression natively.
            xgb.fit(xTrainFlat, yTrain)
            xgbPred = xgb.predict(xValFlat)
            xgbMSE = np.mean((yVal - xgbPred) ** 2)
            xgbMAE = np.mean(np.abs(yVal - xgbPred))
            print(f"  [XGBoost] Val MSE: {xgbMSE:.4f} | Val MAE: {xgbMAE:.4f}")
        except Exception as e:
            print(f"  [XGBoost] Error: {e}")
            
        print("-" * 30)

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

        # Initialize Model
        # Get statEmbedDim from config
        statEmbedDim = config.get('statEmbedDim', 128) # Default 128 if not set
        
        model = NbaTransformer(
            inputDim=len(featureCols),
            dModel=config['dModel'],
            nHead=config['nHead'],
            numLayers=config['numLayers'],
            outputDim=len(targetCols),
            statEmbedDim=statEmbedDim,
            dropout=config['dropout']
        ).to(device)

        criterion = QuantileLoss([0.1, 0.5, 0.9])
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
            valAbsoluteErrorSum = np.zeros(len(targetCols))
            valCount = 0

            with torch.no_grad():
                for x, y in valLoader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x) # (B, 3, 3)
                    loss = criterion(pred, y)
                    valLossList.append(loss.item())

                    # Calculate Original Scale Metrics using P50 (Index 1)
                    predP50 = pred[:, :, 1]
                    predOriginal = scalerY.inverse_transform(predP50.cpu().numpy())
                    yOriginal = scalerY.inverse_transform(y.cpu().numpy())
                    
                    diff = predOriginal - yOriginal
                    valAbsoluteErrorSum += np.sum(np.abs(diff), axis=0) # Sum Absolute Errors
                    valCount += len(y)
                    
            valMeanLoss = sum(valLossList) / len(valLossList)
            
            # Avoid division by zero
            if valCount > 0:
                valMaeOriginal = valAbsoluteErrorSum / valCount
            else:
                valMaeOriginal = np.zeros(len(targetCols))
            
            # Print 結果 (這會顯示在 Slurm 的 output file 中)
            print(f"Epoch [{epoch+1}/{config['nEpochs']}] | Train Loss: {trainMeanLoss:.4f} | Val Loss: {valMeanLoss:.4f}")
            print(f"  >>> Val MAE (Original): {', '.join([f'{col}={val:.4f}' for col, val in zip(targetCols, valMaeOriginal)])}")


            # 儲存最佳模型
            if valMeanLoss < bestLoss:
                bestLoss = valMeanLoss
                
                # Define Run Name (Used as Folder Name)
                runName = f"best_seq_ep{config['nEpochs']}_seq{config['seqLength']}_d{config['dModel']}_head{config['nHead']}_lr{config['learningRate']}_bs{config['batchSize']}"
                runPath = os.path.join(config['saveDir'], runName)
                
                # Cleanup previous best model folder if it exists and is different
                if bestModelPath and os.path.exists(bestModelPath) and bestModelPath != runPath:
                    try:
                        shutil.rmtree(bestModelPath)
                        # print(f"  >>> Removed previous best run: {bestModelPath}")
                    except OSError as e:
                        pass
                        # print(f"  >>> Error removing previous run: {e}")

                # Create new run folder
                os.makedirs(runPath, exist_ok=True)
                
                # Save Model Checkpoint
                ckptPath = os.path.join(runPath, 'model.ckpt')
                torch.save(model.state_dict(), ckptPath)
                
                # Save Config as JSON
                configPath = os.path.join(runPath, 'config.json')
                saveConfig = config.copy()
                saveConfig['featureCols'] = featureCols
                saveConfig['valid_loss'] = valMeanLoss
                # Flatten Valid Metrics
                for col, val in zip(targetCols, valMaeOriginal):
                    saveConfig[f'valid_mae_{col.lower()}'] = val
                
                with open(os.path.join(runPath, 'config.json'), 'w') as f:
                    json.dump(saveConfig, f, indent=4)

                bestModelPath = runPath
                print(f"  >>> New Best Model & Config Saved to: {runPath}")
        
        # --- Testing Phase ---
        print("\\nStep 4: Start Testing with Best Model...")
        
        if bestModelPath:
            # Load best model
            print(f"Loading best model from: {bestModelPath}")
            ckptLoadPath = os.path.join(bestModelPath, 'model.ckpt')
            model.load_state_dict(torch.load(ckptLoadPath))
            model.eval()
            
            testLossList = []
            allGamblerPreds = []
            
            with torch.no_grad():
                for x, y in testLoader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x) # (B, 3, 3)
                    loss = criterion(pred, y)
                    testLossList.append(loss.item())
                    allGamblerPreds.append(pred.cpu().numpy())

            testMeanLoss = sum(testLossList) / len(testLossList) if testLossList else 0
            
            # Concatenate
            gamblerPredsScaled = np.concatenate(allGamblerPreds, axis=0) # (N, 3, 3)
            
            # Inverse Transform (P10, P50, P90)
            preds_gambler = np.zeros_like(gamblerPredsScaled)
            for q in range(3):
                preds_gambler[:, :, q] = scalerY.inverse_transform(gamblerPredsScaled[:, :, q])
            
            # Extract P50 for Metrics
            predsP50 = preds_gambler[:, :, 1]
            
            # --- House Baselines (TEST SET) ---
            print("Generating House Lines (Hybrid)...")
            # --- House Baselines Predictions (TEST SET) ---
            print("Generating House Lines (Hybrid)...")
            N_test, S_test, F_test = xTest.shape
            xTestFlat = xTest.reshape(N_test, -1)
            
            # 1. Naive (Window Mean)
            house_naive_preds = np.zeros_like(yTest)
            for i, tgt in enumerate(targetCols):
                if tgt in featureCols:
                    idx = featureCols.index(tgt)
                    house_naive_preds[:, i] = np.mean(xTest[:, :, idx], axis=1)
                else:
                    house_naive_preds[:, i] = trainMean[i]
            
            # 2. LR
            try:
                preds_lr = lr.predict(xTestFlat)
            except:
                preds_lr = house_naive_preds
                
            # 3. XGB
            try:
                preds_xgb = xgb.predict(xTestFlat)
            except:
                preds_xgb = house_naive_preds
                
            # Hybrid House Line
            # Weights: (LR=0.40, XGB=0.45, Naive=0.15)
            house_preds = (0.40 * preds_lr) + (0.45 * preds_xgb) + (0.15 * house_naive_preds)
            
            print("Seq Model Training Completed. Returning predictions for Simulation...")
            
            # metaTest is numpy array of IDs
            
            # --- Update Config with Test Metrics ---
            # Calculate Test Metrics
            test_maes = []
            for i in range(len(targetCols)):
                 test_maes.append(mean_absolute_error(yTest[:, i], predsP50[:, i]))
            
            with open(os.path.join(runPath, 'config.json'), 'r') as f:
                updateConfig = json.load(f)
            
            updateConfig['test_loss'] = testMeanLoss
            for col, val in zip(targetCols, test_maes):
                updateConfig[f'test_mae_{col.lower()}'] = val
                
            with open(os.path.join(runPath, 'config.json'), 'w') as f:
                json.dump(updateConfig, f, indent=4)
            print("Updated config with Test Metrics.")

            return preds_gambler, house_preds, yTest, metaTest, runPath

            # --- Reporting ---
            roi = ((bankroll - 10000)/10000)*100
            
            # Calc Metrics
            test_maes = []
            for i in range(len(targetCols)):
                 test_maes.append(mean_absolute_error(yTest[:, i], predsP50[:, i]))
            
            avg_spread = np.mean(preds_gambler[:, :, 2] - preds_gambler[:, :, 0])
             
            report_lines = []
            report_lines.append(f"Training Run: {runName}")
            report_lines.append(f"Best Validation Loss: {bestLoss:.4f}")
            report_lines.append(f"Test Loss (MSE): {testMeanLoss:.4f}")
            report_lines.append(f"Test MAE: {test_maes}")
            report_lines.append(f"Avg Spread: {avg_spread:.2f}")
            
            # Baseline Comparision
            naive_mae = mean_absolute_error(yTest, house_naive_preds)
            lr_mae = mean_absolute_error(yTest, preds_lr)
            xgb_mae = mean_absolute_error(yTest, preds_xgb)
            
            baseline_info = [
                "="*50,
                "BASELINE COMPARISON (TEST SET)",
                "="*50,
                f"Naive (Window)    | MAE: {naive_mae:.4f}",
                f"Linear Regression | MAE: {lr_mae:.4f}",
                f"XGBoost           | MAE: {xgb_mae:.4f}",
                "="*50
            ]
            report_lines.extend(baseline_info)
            for l in baseline_info: print(l)
            
            report_lines.append("SIMULATION REPORT")
            report_lines.append("="*50)
            report_lines.append(f"Total Bets: {total_bets}")
            report_lines.append(f"Wins: {wins} | Losses: {losses}")
            if total_bets > 0:
                report_lines.append(f"Win Rate: {(wins/total_bets)*100:.2f}%")
            report_lines.append(f"ROI: {roi:.2f}%")
            report_lines.append(f"Final Bankroll: ${bankroll:.2f}")
            
            # Save Report
            reportPath = os.path.join(bestModelPath, 'simulation_report.txt')
            with open(reportPath, 'w') as f:
                f.write('\n'.join(report_lines))
            print(f"Report Saved to: {reportPath}")
            
            # Save CSV
            df = pd.DataFrame(bet_history)
            logPath = os.path.join(bestModelPath, 'betting_log.csv')
            df.to_csv(logPath, index=False)
            print(f"Betting Log Saved to: {logPath}")
            
            # Visualizations
            if not df.empty:
                # PnL Chart
                df['AccumulatedPnL'] = df['PnL'].cumsum()
                plt.figure(figsize=(10, 6))
                plt.plot(df.index, df['AccumulatedPnL'], label='Cumulative PnL', color='green')
                plt.title(f'Betting Simulation PnL (ROI: {roi:.2f}%)')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(bestModelPath, 'pnl_chart.png'))
                plt.close()
                print("Visualizations Saved.")
            
            # Update Config
            configPath = os.path.join(bestModelPath, 'config.json')
            if os.path.exists(configPath):
                try:
                    with open(configPath, 'r') as f:
                        finalConfig = json.load(f)
                    finalConfig['test_mse'] = testMeanLoss
                    finalConfig['test_mae'] = {col: val for col, val in zip(targetCols, test_maes)}
                    with open(configPath, 'w') as f:
                        json.dump(finalConfig, f, indent=4)
                except:
                    pass
        else:
            print("No model was saved.")
            
    except Exception as e:
        print(f"\\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

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
        'saveDir': 'savedModels',
        'datasetPath': 'dataset/games.csv',
        # 定義賽季切分
        'trainSeasons': [22016, 22017, 22018, 22019, 22020, 22021, 22022],
        'valSeasons': [22023], 
        'testSeasons': [22024]
    }
    train(config)