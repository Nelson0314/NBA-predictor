import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
import os
from tqdm import tqdm
import gc
import concurrent.futures
from .config import DATA_DIR

def load_single_heatmap(args):
    """Parallel helper function to load a single heatmap"""
    fpath, key = args
    try:
        # Load and cast to float32 to save RAM
        data = np.load(fpath).astype(np.float32)
        return (key, data)
    except:
        return None

# ==========================================
# 1. 輔助函式 (Helper Functions)
# ==========================================
def preloadHeatmaps(heatmapDir):
    """
    Load all heatmaps into RAM using Parallel Processing.
    Returns: dict { 'playerID_gameID': np.array }
    """
    print(f"Pre-loading heatmaps from {heatmapDir} into RAM (Parallel)...")
    cache = {}
    
    if not os.path.exists(heatmapDir):
        print("Heatmap directory not found! Returning empty cache.")
        return cache

    files = [f for f in os.listdir(heatmapDir) if f.endswith('.npy')]
    
    # Prepare tasks: (file_path, key_name)
    tasks = []
    for f in files:
        key = f.replace('.npy', '')
        path = os.path.join(heatmapDir, f)
        tasks.append((path, key))
        
    # Parallel Load (I/O Bound -> Use Threads)
    # Using 16 workers usually saturates I/O nicely without blocking Python GIL too much
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(load_single_heatmap, t): t for t in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(files), desc="Loading .npy files"):
            res = future.result()
            if res:
                k, data = res
                cache[k] = data
            
    print(f"Loaded {len(cache)} heatmaps into RAM.")
    return cache

def loadAndPreprocessData(gamesPath, shotsPath, teamsPath, seqLength=10):
    # Fix for Argument Mismatch (Backward Compatibility)
    if isinstance(teamsPath, int):
        print(f"Warning: Detected 3-argument call. Assuming teamsPath='dataset/teams.csv' and seqLength={teamsPath}")
        seqLength = teamsPath
        teamsPath = os.path.join(DATA_DIR, 'teams.csv')

    print("Step 1: Loading and Cleaning Data...")
    
    if not os.path.exists(gamesPath):
        raise FileNotFoundError("Files not found.")

    gamesData = pd.read_csv(gamesPath, low_memory=False, dtype={'GAME_ID': str})
    gamesData = gamesData.loc[:, ~gamesData.columns.duplicated()]
    # shotsData = pd.read_csv(shotsPath, low_memory=False) # Optimized: Not loading shots here
    teamsData = pd.read_csv(teamsPath, low_memory=False, dtype={'GAME_ID': str})
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
    
    # Clean Games
    gamesData['GAME_DATE'] = pd.to_datetime(gamesData['GAME_DATE'], errors='coerce')
    allCols = list(set(featureCols + targetCols))
    for col in allCols:
        if col in gamesData.columns:
            gamesData[col] = pd.to_numeric(gamesData[col], errors='coerce')
    gamesData = gamesData.dropna(subset=allCols)
    gamesData = gamesData.sort_values(by=['Player_ID', 'GAME_DATE']).reset_index(drop=True)

    shotsGrouped = None # Placeholder since we load from disk

    print(f"Data Loaded. Games: {len(gamesData)}, Shot Groups: (Loaded from disk)")
    
    # ---------------------------------------------------------
    # Feature Engineering: Season-to-Date Averages
    # ---------------------------------------------------------
    print("Calculating Season-to-Date Features...")
    
    # 1. Parse Matchup
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
    # Merge Team & Opponent Rolling Stats (from teams.csv)
    # ---------------------------------------------------------
    print("Merging Team Stats...")
    # Normalize GAME_ID to 10-digit string
    gamesData['GAME_ID'] = pd.to_numeric(gamesData['GAME_ID'], errors='coerce').fillna(0).astype('int64').astype(str).str.zfill(10)
    teamsData['GAME_ID'] = pd.to_numeric(teamsData['GAME_ID'], errors='coerce').fillna(0).astype('int64').astype(str).str.zfill(10)

    # Identify stats columns in teams.csv (AVG_*)
    teamStatsCols = [c for c in teamsData.columns if c.startswith('AVG_')]
    
    # Subset for merging
    teamsSubset = teamsData[['GAME_ID', 'TEAM_ABBREVIATION'] + teamStatsCols].copy()

    # Safety Fix: Shift features to avoid leakage
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
        gamesData[col] = gamesData[col].fillna(0)
        
    featureCols.extend(newTeamCols)
    print(f"Team Features Added: {len(newTeamCols)} columns.")
    
    # 2. Player Rolling Stats
    gamesData = gamesData.sort_values(by=['Player_ID', 'SEASON_ID', 'GAME_DATE'])
    
    player_rolling_cols = ['PTS', 'AST', 'REB']
    new_player_cols = [f'PLAYER_AVG_{c}' for c in player_rolling_cols]
    
    for col in player_rolling_cols:
        new_col = f'PLAYER_AVG_{col}'
        gamesData[new_col] = gamesData.groupby(['Player_ID', 'SEASON_ID'])[col].transform(lambda x: x.expanding().mean().shift(1)).fillna(0)
        
    featureCols.extend(new_player_cols)
    
    # Feature 3: Days Since Last Game
    gamesData['GAME_DATE'] = pd.to_datetime(gamesData['GAME_DATE'])
    gamesData['DAYS_SINCE_LAST_GAME'] = gamesData.groupby('Player_ID')['GAME_DATE'].diff().dt.days
    gamesData['DAYS_SINCE_LAST_GAME'] = gamesData['DAYS_SINCE_LAST_GAME'].fillna(7)
    
    featureCols.append('DAYS_SINCE_LAST_GAME')

    return gamesData, shotsGrouped, featureCols, targetCols

def createMultimodalSequences(gamesData, shotsGrouped, seqLength, featureCols, targetCols):
    """
    Returns:
        playerSequences: (N, seqLength) - Player IDs
        gameSequences:   (N, seqLength) - Game IDs
        statSequences:   (N, seqLength, numFeatures)
        targets:         (N, numTargets)
    """
    print("Step 2: Generating Multimodal Sequences...")
    playerSequences = []
    gameSequences = []
    statSequences = []
    targets = []
    
    if 'SEASON_ID' in gamesData.columns:
        groups = gamesData.groupby(['Player_ID', 'SEASON_ID'])
    else:
        groups = gamesData.groupby('Player_ID')

    for groupKey, group in tqdm(groups, desc="Processing Players"):
        if len(group) <= seqLength:
            continue
        
        gameIds = group['GAME_ID'].values
        playerId = group['Player_ID'].values[0]
        feats = group[featureCols].values
        targs = group[targetCols].values
        
        for i in range(len(group) - seqLength):
            y = targs[i + seqLength]
            currentGameIds = gameIds[i : i + seqLength]
            xStat = feats[i : i + seqLength] 
            
            playerSequences.append(np.full(seqLength, playerId))
            gameSequences.append(currentGameIds)
            statSequences.append(xStat)
            targets.append(y)
            
    return np.array(playerSequences), np.array(gameSequences), np.array(statSequences), np.array(targets)

# ==========================================
# 2. Dataset 類別
# ==========================================
class MultimodalDataset(Dataset):
    def __init__(self, playerIds, gameIds, xStat, heatmapCache, y=None):
        """
        heatmapCache: Dict { 'pid_gid': np.array }
        """
        self.playerIds = playerIds
        self.gameIds = gameIds
        self.xStat = torch.FloatTensor(xStat)
        self.y = torch.FloatTensor(y) if y is not None else None
        self.heatmapCache = heatmapCache # Pass the RAM cache here
        
        # Pre-allocate a zero tensor for missing data to avoid creating it every time
        self.emptyHeatmap = np.zeros((2, 50, 50), dtype=np.float32)

    def __getitem__(self, idx):
        # Fast Lookup from RAM
        currentPlayers = self.playerIds[idx] # (SeqLen,)
        currentGames = self.gameIds[idx]     # (SeqLen,)
        
        xImgSeq = []
        for pid, gid in zip(currentPlayers, currentGames):
            key = f"{int(pid)}_{str(gid).zfill(10)}"
            # Dictionary lookup is O(1) - Extremely Fast
            h = self.heatmapCache.get(key, self.emptyHeatmap)
            xImgSeq.append(h)
            
        xImgSeq = np.array(xImgSeq) # (SeqLen, 2, 50, 50)
        
        xImgTensor = torch.FloatTensor(xImgSeq)
        xStatTensor = self.xStat[idx]
        
        if self.y is not None:
            return xImgTensor, xStatTensor, self.y[idx]
        else:
            return xImgTensor, xStatTensor

    def __len__(self):
        return len(self.xStat)

# ==========================================
# 3. 模型架構 (Multimodal CRNN)
# ==========================================
class CnnEncoder(nn.Module):
    def __init__(self, outputDim):
        super(CnnEncoder, self).__init__()
        # Input: (Batch, Channels=2, H=50, W=50)
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> (16, 25, 25)
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> (32, 12, 12)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # -> (64, 6, 6)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64 * 6 * 6, outputDim),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class NbaMultimodal(nn.Module):
    def __init__(self, numStatFeatures, seqLength, outputDim, 
                 cnnEmbedDim=64, statEmbedDim=128, 
                 dModel=128, nHead=4, numLayers=2, dropout=0.1):
        super(NbaMultimodal, self).__init__()
        
        # 1. Visual Encoder (CNN)
        self.cnnEncoder = CnnEncoder(outputDim=cnnEmbedDim)
        
        # 2. Stat Encoder (MLP)
        self.statEncoder = nn.Sequential(
            nn.Linear(numStatFeatures, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, statEmbedDim),
            nn.GELU()
        )
        
        # 3. Fusion & Transformer
        # Joint Embedding Size
        self.jointDim = cnnEmbedDim + statEmbedDim
        
        # Projection to Transformer Dimension (dModel)
        self.fusionProj = nn.Linear(self.jointDim, dModel)
        
        # Transformer
        encoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        
        # 4. Prediction Head
        self.head = nn.Sequential(
            nn.Linear(dModel, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, outputDim)
        )
        
    def forward(self, imgSeq, statSeq):
        batchSize, seqLen, C, H, W = imgSeq.size()
        
        imgFlat = imgSeq.view(batchSize * seqLen, C, H, W)
        visualEmbeds = self.cnnEncoder(imgFlat) 
        visualEmbeds = visualEmbeds.view(batchSize, seqLen, -1)
        
        statEmbeds = self.statEncoder(statSeq) 
        
        jointEmbeds = torch.cat([visualEmbeds, statEmbeds], dim=2) 
        
        transformerInput = self.fusionProj(jointEmbeds) 
        
        transformerOut = self.transformer(transformerInput)
        
        lastState = transformerOut[:, -1, :] 
        
        out = self.head(lastState)
        return out