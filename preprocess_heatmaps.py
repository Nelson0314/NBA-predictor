import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch
from multimodalModel import generateHeatmap

# Configuration
SHOTS_PATH = 'dataset/shots.csv'
OUTPUT_DIR = 'dataset/heatmaps'

def preprocess_heatmaps():
    print(f"Loading shots from {SHOTS_PATH}...")
    if not os.path.exists(SHOTS_PATH):
        raise FileNotFoundError(f"{SHOTS_PATH} not found.")

    shotsData = pd.read_csv(SHOTS_PATH, low_memory=False)
    
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # Group by Player and Game
    print("Grouping shots data...")
    shotsGrouped = shotsData.groupby(['Player_ID', 'GAME_ID'])
    
    print(f"Found {len(shotsGrouped)} unique player-game combinations.")
    print("Generating and saving heatmaps...")
    
    count = 0
    for (playerId, gameId), group in tqdm(shotsGrouped):
        # Generate heatmap using the function from multimodalModel (ensures consistency)
        # generateHeatmap returns a torch.FloatTensor (2, 50, 50)
        heatmapTensor = generateHeatmap(group)
        
        # Convert to numpy for saving
        heatmapNp = heatmapTensor.numpy()
        
        # Define filename: {PlayerID}_{GameID}.npy
        # Ensure IDs are strings to avoid format issues
        savePath = os.path.join(OUTPUT_DIR, f"{int(playerId)}_{str(gameId).zfill(10)}.npy")
        
        # Save as compressed numpy to save space (optional, save is faster)
        np.save(savePath, heatmapNp)
        count += 1
        
    print(f"Done! Saved {count} heatmaps to {OUTPUT_DIR}")

if __name__ == "__main__":
    preprocess_heatmaps()
