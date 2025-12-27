import json
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import norm
from tqdm import tqdm
import warnings
import sys

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import DATA_DIR, SAVED_MODELS_DIR, GAMES_PATH, TEAMS_PATH
try:
    from src.multiModel import loadAndPreprocessData, createMultimodalSequences, preloadHeatmaps, NbaMultimodal
except ImportError as e:
    print(f"Error importing src modules: {e}")
    exit()

warnings.filterwarnings('ignore')

# Config
MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "quant_ep40_seq5_dm64")
DATASET_DIR_OLD = DATA_DIR
DATASET_DIR_NEW = os.path.join(DATA_DIR, "live_2025")
ODDS_FILE = os.path.join(os.path.dirname(__file__), "..", "event_odds_data.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "bets_1226.csv")
SEQ_LENGTH = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Config
with open(os.path.join(MODEL_PATH, 'config.json'), 'r') as f:
    CONFIG = json.load(f)

# Feature Cols (Must match training)
# We need to replicate the 'train_house_models' feature extraction just to get the list of columns and the scaler.
# To avoid slow re-training, we will just use the same logic to load data and fit scaler.
FEATURE_COLS = [] # Will be populated

# ==========================================
# 1. Parsing Odds
# ==========================================
def parse_odds(file_path):
    print(f"Parsing {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rows = []
    # Structure: List[Game] -> bookmakers -> markets -> outcomes
    for game in data:
        home = game.get('home_team')
        away = game.get('away_team')
        commence = game.get('commence_time')
        
        # We only look for 'fanduel' or first available
        bookie = next((b for b in game['bookmakers'] if b['key'] == 'fanduel'), None)
        if not bookie:
            bookie = game['bookmakers'][0] if game['bookmakers'] else None
        
        if not bookie:
            continue
            
        for market in bookie['markets']:
            m_key = market['key'] # e.g. player_points
            
            # Target mapping
            target_map = {
                'player_points': 'PTS',
                'player_assists': 'AST',
                'player_rebounds': 'REB'
            }
            if m_key not in target_map:
                continue
            
            tgt = target_map[m_key]
            
            # Outcomes grouped by player? No, list of all outcomes.
            # Usually pairs of Over/Under for same player & line.
            # We need to group them.
            outcomes = market['outcomes']
            
            # Helper dict to pair them: (Player, Point) -> {Over_Price, Under_Price}
            lines_dict = {}
            for out in outcomes:
                p_name = out['description']
                side = out['name'] # Over / Under
                price = out['price']
                point = out['point']
                
                k = (p_name, point)
                if k not in lines_dict:
                    lines_dict[k] = {}
                lines_dict[k][side] = price
            
            # Now process the pairs
            for (p_name, point), prices in lines_dict.items():
                if 'Over' in prices and 'Under' in prices:
                    o_price = prices['Over']
                    u_price = prices['Under']
                    
                    # Calculate No Vig
                    p_over = 1 / o_price
                    p_under = 1 / u_price
                    margin = p_over + p_under
                    
                    fair_p_over = p_over / margin
                    fair_p_under = p_under / margin
                    
                    fair_o_price = 1 / fair_p_over
                    fair_u_price = 1 / fair_p_under
                    
                    rows.append({
                        'Player_Name': p_name,
                        'Target': tgt,
                        'Line': point,
                        'Odds_Over': o_price,
                        'Odds_Under': u_price,
                        'FairPro_Over': round(fair_p_over, 3),
                        'FairPro_Under': round(fair_p_under, 3),
                        'FairOdds_Over': round(fair_o_price, 2),
                        'FairOdds_Under': round(fair_u_price, 2)
                    })
    
    df = pd.DataFrame(rows)
    print(f"  Found {len(df)} props.")
    return df

# ==========================================
# 2. Model Definition (Reused)
# ==========================================
class NbaMultimodalQuantile(NbaMultimodal):
    def __init__(self, numStatFeatures, seqLength, numTargets=3, cnnEmbedDim=64, statEmbedDim=128, dModel=128, nHead=4, numLayers=2, dropout=0.1):
        super().__init__(numStatFeatures, seqLength, numTargets, cnnEmbedDim, statEmbedDim, dModel, nHead, numLayers, dropout)
        self.quantiles = [0.1, 0.5, 0.9]
        self.num_quantiles = len(self.quantiles)
        self.num_targets = numTargets
        self.head = nn.Sequential(
            nn.Linear(dModel, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, numTargets * self.num_quantiles)
        )

    def forward(self, imgSeq, statSeq):
        batchSize, seqLen, C, H, W = imgSeq.size()
        imgFlat = imgSeq.view(batchSize * seqLen, C, H, W)
        visualEmbeds = self.cnnEncoder(imgFlat) 
        visualEmbeds = visualEmbeds.view(batchSize, seqLen, -1)
        statEmbeds = self.statEncoder(statSeq) 
        fusionInput = torch.cat([visualEmbeds, statEmbeds], dim=2)
        fusionEmbeds = self.fusionProj(fusionInput)
        transformerOut = self.transformer(fusionEmbeds)
        lastState = transformerOut[:, -1, :] 
        return self.head(lastState)

# ==========================================
# 3. Predict & Bet
# ==========================================
def main():
    odds_df = parse_odds(ODDS_FILE)
    if odds_df.empty:
        print("No odds found.")
        return

    # Load 2025 Data
    # Note: update_live_2025.py now generates clean data directly.
    print("Loading 2025 Data for Context...")
    gamesPathNew = os.path.join(DATASET_DIR_NEW, 'games_2025.csv')
    shotsPathNew = os.path.join(DATASET_DIR_NEW, 'shots_2025.csv')
    teamsPathNew = os.path.join(DATASET_DIR_NEW, 'teams_2025.csv')
    
    # We load using multiModel logic to get features
    # But we need FEATURE_COLS first. 
    # To get correct FEATURE_COLS and SCALER, we MUST load OLD data first.
    print("Loading Historic Data to Fit Scaler...")
    gamesPathOld = os.path.join(DATASET_DIR_OLD, 'games.csv')
    teamsPathOld = os.path.join(DATASET_DIR_OLD, 'teams.csv')
    
    # Load Old to get columns and fit scaler
    # Note: loadAndPreprocessData calculates Rolling stats.
    # We rely on it to define the feature set.
    # Returns: gamesData, shotsGrouped, featureCols, targetCols
    gamesOld, _, featureCols, _ = loadAndPreprocessData(gamesPathOld, None, teamsPathOld, SEQ_LENGTH)
    
    print(f"DEBUG: featureCols length: {len(featureCols)}")
    
    # Fit Scaler
    print("Fitting Scaler...")
    scaler = StandardScaler()
    scaler.fit(gamesOld[featureCols].values)
    
    # Load New (using same feature cols logic)
    print("Processing 2025 Data...")
    gamesNew, shotsNew, _, _ = loadAndPreprocessData(gamesPathNew, shotsPathNew, teamsPathNew, SEQ_LENGTH)
    
    # Load Heatmaps
    heatmapDirNew = os.path.join(DATASET_DIR_NEW, 'heatmaps')
    heatmapCache = preloadHeatmaps(heatmapDirNew)
    
    # Load Model
    print("Loading Model...")
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
    ).to(DEVICE)
    
    print(f"DEBUG: Model Initialized.")
    print(f"DEBUG: Model Head: {model.head}")
    print(f"DEBUG: statEncoder In_Features: {len(featureCols)}")
    
    try:
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'model.ckpt'), map_location=DEVICE))
    except Exception as e:
        print(f"CRITICAL ERROR LOADING MODEL: {e}")
        # Attempt to inspect checkpoint keys for guidance
        print("Falling back to strict=False for debugging (DO NOT USE IN PROD)...")
        # model.load_state_dict(..., strict=False)
        return

    model.eval()
    
    # Target Scaler (MinMax) - Needed to Inverse Transform output?
    # Wait, the model output (from train_with_conf) is SCALED [0,1].
    # We need the Target Scaler fitted on Old Data too.
    # createMultimodalSequences in train_house_models does this.
    # We need to replicate fitting scalerY.
    print("Fitting Target Scaler...")
    TARGET_COLS = ['PTS', 'AST', 'REB']
    # Get y from Old Data
    # We can just take the target columns from gamesOld
    # Note: createMultimodalSequences shifts things.
    # But for scaling, fitting on raw distribution of targets is standard?
    # Or specifically the 'y' used in training?
    # In `train_with_conf`, yTrain is used. 
    # yTrain comes from `createMultimodalSequences`.
    # Let's fit on raw gamesOld[TARGET_COLS] for simplicity/approximation, 
    # assuming roughly same distribution.
    scalerY = MinMaxScaler()
    scalerY.fit(gamesOld[TARGET_COLS].values)
    
    # --- Prediction Loop ---
    # For each player in odds, get last 5 games
    print("Generating Predictions...")
    
    results = []
    
    # Cache player data to avoid repeated lookups
    player_groups = gamesNew.sort_values('GAME_DATE').groupby('Player_Name')
    
    for idx, row in tqdm(odds_df.iterrows(), total=len(odds_df)):
        p_name = row['Player_Name']
        target_name = row['Target']
        line = row['Line']
        
        if p_name not in player_groups.groups:
            # Name mismatch check? 
            # Simple check: try matching without case? 
            # For now skip.
            continue
            
        p_data = player_groups.get_group(p_name)
        
        if len(p_data) < SEQ_LENGTH:
            continue
            
        # Take last N games
        seq_data = p_data.iloc[-SEQ_LENGTH:].copy()
        
        # Prepare Features
        # Scale stats
        stats = seq_data[featureCols].values # (S, F)
        stats_scaled = scaler.transform(stats)
        stats_tensor = torch.tensor(stats_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE) # (1, S, F)
        
        # Prepare Images
        img_list = []
        for i in range(SEQ_LENGTH):
            g_id = seq_data.iloc[i]['GAME_ID'] # Patched to GAME_ID
            p_id = seq_data.iloc[i]['Player_ID']
            t_id = seq_data.iloc[i]['TEAM_ID'] # Patched to TEAM_ID
            # Key format matching generate_heatmaps_2025.py
            # Format: "{int(pid)}_{str(gid).zfill(10)}"
            h_key = f"{int(p_id)}_{str(g_id).zfill(10)}"
            
            if h_key in heatmapCache:
                img_list.append(heatmapCache[h_key])
            else:
                img_list.append(np.zeros((2, 50, 50), dtype=np.float32)) # Pad if missing
        
        img_seq = np.array(img_list) # (S, 2, 50, 50)
        img_tensor = torch.tensor(img_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            preds_scaled = model(img_tensor, stats_tensor) # (1, 3 targets * 3 quantiles)
            
        preds_scaled = preds_scaled.cpu().numpy().reshape(1, 3, 3) # (1, Targets, Quantiles)
        
        # Inverse Transform
        # We need to inverse transform each quantile?
        # scalerY expects (N, 3). We have (1, 3, 3).
        # We can treat quantiles as separate items.
        
        pred_vals = np.zeros((3, 3)) # (PTS/AST/REB, P10/P50/P90)
        for q in range(3):
            # Shape (1, 3)
            sq = preds_scaled[:, :, q]
            inv = scalerY.inverse_transform(sq)
            pred_vals[:, q] = inv[0]
            
        # Extract specific target
        t_idx = list(TARGET_COLS).index(target_name)
        
        p10, p50, p90 = pred_vals[t_idx]
        
        # Logic:
        # Mean ~ P50
        # Std ~ (P90 - P10) / 2.56 (assuming normalish)
        pred_mean = p50
        pred_std = (p90 - p10) / 2.56
        if pred_std < 0.1: pred_std = 0.1 # Safety
        
        # Calculate Edge
        # Z-score of Line
        z = (line - pred_mean) / pred_std
        prob_under = norm.cdf(z)
        prob_over = 1.0 - prob_under
        
        # Determine Bet
        # EV calc (Using Real Odds)
        ev_over = (prob_over * row['Odds_Over']) - 1.0
        ev_under = (prob_under * row['Odds_Under']) - 1.0
        
        choice = "SKIP"
        confidence = 0.0
        chosen_ev = 0.0
        chosen_odds = 0.0
        
        # Threshold (e.g., 55% conf or positive EV)
        # Strategy: Bet if EV > 0.05 (5% edge)
        if ev_over > 0.05:
            choice = "OVER"
            confidence = prob_over
            chosen_ev = ev_over
            chosen_odds = row['Odds_Over']
        elif ev_under > 0.05:
            choice = "UNDER"
            confidence = prob_under
            chosen_ev = ev_under
            chosen_odds = row['Odds_Under']
            
        row['Pred_Mean'] = round(pred_mean, 1)
        row['Pred_Std'] = round(pred_std, 2)
        row['MyProb_Over'] = round(prob_over, 3)
        row['MyProb_Under'] = round(prob_under, 3)
        row['Pick'] = choice
        row['Pick_Conf'] = round(confidence, 3)
        row['Pick_EV'] = round(chosen_ev, 3)
        row['Pick_Odds'] = chosen_odds
        
        results.append(row)
        
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        # Reorder columns
        cols = ['Player_Name', 'Target', 'Line', 'Odds_Over', 'Odds_Under', 'FairOdds_Over', 'FairOdds_Under', 
                'Pred_Mean', 'Pred_Std', 'MyProb_Over', 'MyProb_Under', 'Pick', 'Pick_Conf', 'Pick_EV', 'Pick_Odds']
        res_df = res_df[cols]
        print("\nAll Bets Generated:")
        print(res_df[res_df['Pick'] != 'SKIP'].to_string())
        res_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSaved to {OUTPUT_FILE}")
    else:
        print("No bets generated (maybe no matching players?).")

if __name__ == "__main__":
    main()
