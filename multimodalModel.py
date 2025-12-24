import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
import os
from tqdm import tqdm

# ==========================================
# 1. 輔助函式 (Helper Functions)
# ==========================================
def generateHeatmap(shotsDf, imgSize=50, sigma=1.0):
    """
    Generate 2-channel heatmap (Attempts, Made).
    """
    heatmapTensor = np.zeros((2, imgSize, imgSize))
    
    if len(shotsDf) == 0:
        return torch.FloatTensor(heatmapTensor)

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

    heatmapTensor[0] = gaussian_filter(histAttempts.T, sigma=sigma)
    heatmapTensor[1] = gaussian_filter(histMade.T, sigma=sigma)

    GLOBAL_MAX_VAL = 5.0
    heatmapTensor = np.clip(heatmapTensor, 0, GLOBAL_MAX_VAL)
    heatmapTensor /= GLOBAL_MAX_VAL

    return torch.FloatTensor(heatmapTensor)

def loadAndPreprocessData(gamesPath, shotsPath, teamsPath, seqLength=10):
    print("Step 1: Loading and Cleaning Data...")
    
    if not os.path.exists(gamesPath):
        raise FileNotFoundError("Files not found.")

    gamesData = pd.read_csv(gamesPath, low_memory=False)
    gamesData = gamesData.loc[:, ~gamesData.columns.duplicated()]
    gamesData = pd.read_csv(gamesPath, low_memory=False)
    gamesData = gamesData.loc[:, ~gamesData.columns.duplicated()]
    # shotsData = pd.read_csv(shotsPath, low_memory=False) # Optimized: Not loading shots here
    teamsData = pd.read_csv(teamsPath, low_memory=False)

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
    gamesData['GAME_DATE'] = pd.to_datetime(gamesData['GAME_DATE'], format='mixed')
    gamesData['GAME_DATE'] = pd.to_datetime(gamesData['GAME_DATE'], format='mixed')
    allCols = list(set(featureCols + targetCols))
    for col in allCols:
        if col in gamesData.columns:
            gamesData[col] = pd.to_numeric(gamesData[col], errors='coerce')
    gamesData = gamesData.dropna(subset=allCols)
    gamesData = gamesData.sort_values(by=['Player_ID', 'GAME_DATE']).reset_index(drop=True)

    # Clean Shots
    # shotsData = shotsData[['Player_ID', 'GAME_ID', 'LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG']].copy()
    # shotsData = shotsData.dropna()

    # print("Indexing shots data...")
    # shotsGrouped = dict(list(shotsData.groupby(['Player_ID', 'GAME_ID'])))
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
    gamesData['GAME_ID'] = gamesData['GAME_ID'].astype(str).str.zfill(10)
    teamsData['GAME_ID'] = teamsData['GAME_ID'].astype(str).str.zfill(10)
    
    # Identify stats columns in teams.csv (AVG_*)
    teamStatsCols = [c for c in teamsData.columns if c.startswith('AVG_')]
    
    # Subset for merging
    teamsSubset = teamsData[['GAME_ID', 'TEAM_ABBREVIATION'] + teamStatsCols].copy()

    # Safety Fix: Shift features to avoid leakage
    teamsSubset = teamsSubset.sort_values(by=['TEAM_ABBREVIATION', 'GAME_ID'])
    stats_cols_only = [c for c in teamsSubset.columns if c.startswith('AVG_')]
    teamsSubset[stats_cols_only] = teamsSubset.groupby('TEAM_ABBREVIATION')[stats_cols_only].shift(1).fillna(0)
    
    # 1. Merge Own Team Stats
    # Rename columns: AVG_PTS -> TEAM_AVG_PTS
    renameOwn = {c: f'TEAM_{c}' for c in teamStatsCols}
    teamsSubsetOwn = teamsSubset.rename(columns=renameOwn)
    
    # Merge on GAME_ID and TEAM_ABBREVIATION
    gamesData = pd.merge(
        gamesData, 
        teamsSubsetOwn, 
        how='left', 
        on=['GAME_ID', 'TEAM_ABBREVIATION']
    )
    
    # 2. Merge Opponent Team Stats
    # Rename columns: AVG_PTS -> OPP_AVG_PTS
    renameOpp = {c: f'OPP_{c}' for c in teamStatsCols}
    teamsSubsetOpp = teamsSubset.rename(columns=renameOpp)
    
    # Merge on GAME_ID and MATCH with OPPONENT_ABBREVIATION
    # The teams table has the opponent's stats under their own abbreviation
    # So we match gamesData['OPPONENT_ABBREVIATION'] == teamsSubsetOpp['TEAM_ABBREVIATION']
    gamesData = pd.merge(
        gamesData,
        teamsSubsetOpp,
        how='left',
        left_on=['GAME_ID', 'OPPONENT_ABBREVIATION'],
        right_on=['GAME_ID', 'TEAM_ABBREVIATION'],
        suffixes=('', '_opp_merge')
    )
    
    # Drop the extra key column from merge if it exists
    if 'TEAM_ABBREVIATION_opp_merge' in gamesData.columns:
        gamesData = gamesData.drop(columns=['TEAM_ABBREVIATION_opp_merge'])

    # Check for merge failures
    missing_opp_stats = gamesData[list(renameOpp.values())[0]].isna().sum()
    if missing_opp_stats > 0:
        print(f"Warning: {missing_opp_stats} games failed to match Opponent Stats based on Abbreviation.")
        
    # Fill NaNs with 0 (for first games of seasons or missing data)
    newTeamCols = list(renameOwn.values()) + list(renameOpp.values())
    for col in newTeamCols:
        gamesData[col] = gamesData[col].fillna(0)
        
    featureCols.extend(newTeamCols)
    print(f"Team Features Added: {len(newTeamCols)} columns.")
    
    # 2. Player Rolling Stats
    # Sort for rolling calculation
    gamesData = gamesData.sort_values(by=['Player_ID', 'SEASON_ID', 'GAME_DATE'])
    
    player_rolling_cols = ['PTS', 'AST', 'REB']
    new_player_cols = [f'PLAYER_AVG_{c}' for c in player_rolling_cols]
    
    # Group by Player+Season
    # Calculate one by one to avoid ambiguity
    for col in player_rolling_cols:
        new_col = f'PLAYER_AVG_{col}'
        gamesData[new_col] = gamesData.groupby(['Player_ID', 'SEASON_ID'])[col].transform(lambda x: x.expanding().mean().shift(1)).fillna(0)
        
    featureCols.extend(new_player_cols)
    
    print(f"Features Added: {new_player_cols}")
    
    return gamesData, shotsGrouped, featureCols, targetCols

def createMultimodalSequences(gamesData, shotsGrouped, seqLength, featureCols, targetCols):
    """
    Returns:
        imageSequences: (N, seqLength, 2, 50, 50)
        statSequences:  (N, seqLength, numFeatures)
        targets:        (N, numTargets)
    """
    print("Step 2: Generating Multimodal Sequences...")
    imageSequences = []
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
            # Target is the game AFTER the sequence
            y = targs[i + seqLength]
            
            # Sequence Data
            currentGameIds = gameIds[i : i + seqLength]
            xStat = feats[i : i + seqLength] # Shape (seqLength, numFeatures)
            
            # Generate Image Sequence
            xImgSeq = []
            for gid in currentGameIds:
                # Optimized: Load from disk
                # key = (playerId, gid)
                # if key in shotsGrouped:
                #     h = generateHeatmap(shotsGrouped[key]) # (2, 50, 50)
                # else:
                #     h = torch.zeros((2, 50, 50), dtype=torch.float32)
                
                heatmapPath = os.path.join('dataset/heatmaps', f"{int(playerId)}_{str(gid).zfill(10)}.npy")
                if os.path.exists(heatmapPath):
                    h = np.load(heatmapPath) # (2, 50, 50)
                else:
                    h = np.zeros((2, 50, 50), dtype=np.float32)

                xImgSeq.append(h)
            
            imageSequences.append(np.array(xImgSeq)) # (seqLength, 2, 50, 50)
            statSequences.append(xStat)
            targets.append(y)
            
    return np.array(imageSequences), np.array(statSequences), np.array(targets)

# ==========================================
# 2. Dataset 類別
# ==========================================
class MultimodalDataset(Dataset):
    def __init__(self, xImg, xStat, y=None):
        self.xImg = torch.FloatTensor(xImg)
        self.xStat = torch.FloatTensor(xStat)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __getitem__(self, idx):
        if self.y is not None:
            return self.xImg[idx], self.xStat[idx], self.y[idx]
        else:
            return self.xImg[idx], self.xStat[idx]

    def __len__(self):
        return len(self.xImg)

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
                 cnnEmbedDim=64, statEmbedDim=32, 
                 dModel=128, nHead=4, numLayers=2, dropout=0.1):
        super(NbaMultimodal, self).__init__()
        
        # 1. Visual Encoder (CNN)
        self.cnnEncoder = CnnEncoder(outputDim=cnnEmbedDim)
        
        # 2. Stat Encoder (MLP)
        self.statEncoder = nn.Sequential(
            nn.Linear(numStatFeatures, statEmbedDim),
            nn.ReLU()
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
        # imgSeq: (Batch, SeqLen, 2, 50, 50)
        # statSeq: (Batch, SeqLen, NumStatFeatures)
        
        batchSize, seqLen, C, H, W = imgSeq.size()
        
        # --- Visual Encoding ---
        # Flatten Sequence dimension into Batch for parallel CNN processing
        # (Batch * SeqLen, 2, 50, 50)
        imgFlat = imgSeq.view(batchSize * seqLen, C, H, W)
        visualEmbeds = self.cnnEncoder(imgFlat) # (Batch * SeqLen, cnnEmbedDim)
        # Reshape back to Sequence
        visualEmbeds = visualEmbeds.view(batchSize, seqLen, -1)
        
        # --- Stat Encoding ---
        statEmbeds = self.statEncoder(statSeq) # (Batch, SeqLen, statEmbedDim)
        
        # --- Fusion ---
        # Concatenate along feature dimension
        jointEmbeds = torch.cat([visualEmbeds, statEmbeds], dim=2) # (Batch, SeqLen, jointDim)
        
        # Project to dModel
        transformerInput = self.fusionProj(jointEmbeds) # (Batch, SeqLen, dModel)
        
        # --- Temporal Modeling ---
        transformerOut = self.transformer(transformerInput)
        
        # Take last time step
        lastState = transformerOut[:, -1, :] # (Batch, dModel)
        
        # --- Prediction ---
        out = self.head(lastState)
        return out
