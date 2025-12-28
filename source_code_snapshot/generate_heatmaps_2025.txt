import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch
import sys

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import DATA_DIR
try:
    from src.graphModel import generateHeatmap
except ImportError as e:
    print(f"Error importing src modules: {e}")
    exit()

# Config
GAMES_FILE = os.path.join(DATA_DIR, 'live_2025', 'games_2025.csv')
SHOTS_FILE = os.path.join(DATA_DIR, 'live_2025', 'shots_2025.csv')
HEATMAP_DIR = os.path.join(DATA_DIR, 'live_2025', 'heatmaps')

if not os.path.exists(HEATMAP_DIR):
    os.makedirs(HEATMAP_DIR)

def main():
    print("Loading 2025 Data...")
    games = pd.read_csv(GAMES_FILE)
    shots = pd.read_csv(SHOTS_FILE)
    
    # Ensure ID formats
    # Check column names: games likely has 'GAME_ID', 'Player_ID'. 
    # shots likely has 'GAME_ID', 'Player_ID', 'LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG'
    
    print(f"Games: {len(games)}")
    print(f"Shots: {len(shots)}")
    
    # Group shots by (GAME_ID, Player_ID) for faster access
    print("Grouping shots...")
    shots_grouped = shots.groupby(['GAME_ID', 'Player_ID'])
    
    # Iterate all player-games
    # We only generate heatmaps for players in the game log
    tasks = []
    
    # Pre-check existing to skip? (User said 'new fetched data', maybe overwrite to be safe or skip existing)
    # Let's overwrite to ensure 12/26 data is fresh.
    
    success_count = 0
    skip_count = 0
    
    print("Generating Heatmaps...")
    for idx, row in tqdm(games.iterrows(), total=len(games)):
        gid = row['GAME_ID'] # e.g. 22500414 (might be int or str)
        pid = row['Player_ID']
        
        # Standardize ID format for filename: PID_GID (10 digit GID)
        # multiModel expects: f"{int(pid)}_{str(gid).zfill(10)}"
        
        gid_str = str(gid).zfill(10)
        pid_str = str(pid)
        
        filename = f"{pid_str}_{gid_str}.npy"
        filepath = os.path.join(HEATMAP_DIR, filename)
        
        if os.path.exists(filepath):
            # print(f"Skipping {filename} (Exists)")
            skip_count += 1
            continue
        
        try:
            # Get shots for this specific game & player
            # Note: shots_grouped.get_group throws KeyError if empty
            if (gid, pid) in shots_grouped.groups:
                player_game_shots = shots_grouped.get_group((gid, pid))
            else:
                player_game_shots = pd.DataFrame() # Empty
            
            # Generate
            # generateHeatmap returns Tensor (2, 50, 50)
            tensor = generateHeatmap(player_game_shots, imgSize=50)
            
            # Save as numpy
            np_array = tensor.numpy() # Convert to numpy
            np.save(filepath, np_array)
            success_count += 1
            
        except Exception as e:
            # print(f"Error {pid}_{gid}: {e}")
            pass

    print(f"Done. Generated {success_count} heatmaps.")

if __name__ == "__main__":
    main()
