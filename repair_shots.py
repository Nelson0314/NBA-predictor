import pandas as pd
import os
import time
import random
from tqdm import tqdm
from nba_api.stats.endpoints import shotchartdetail
from requests.exceptions import ReadTimeout, ConnectTimeout
import urllib3

# ==========================================
# Configuration
# ==========================================
GAMES_PATH = 'dataset/games.csv'
OUTPUT_SHOTS_PATH = 'dataset/shots_fixed.csv'

def fetchWithRetry(apiFunc, maxRetries=3, **kwargs):
    kwargs['timeout'] = 25
    for i in range(maxRetries):
        try:
            time.sleep(random.uniform(0.4, 0.8)) 
            return apiFunc(**kwargs)
        except (ReadTimeout, ConnectTimeout, urllib3.exceptions.ReadTimeoutError, ConnectionResetError):
            time.sleep(3)
        except Exception:
            break
    return None

def clean_duplicates(df, subset):
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep='last')
    after = len(df)
    return df, before - after

def main():
    if not os.path.exists(GAMES_PATH):
        print("❌ games.csv not found!")
        return

    print("Step 1: Reading and Deduplicating Games Data...")
    dfGames = pd.read_csv(GAMES_PATH, low_memory=False)
    
    # 1. Deduplicate Games
    dfGames, removed = clean_duplicates(dfGames, subset=['Game_ID', 'Player_ID'])
    if removed > 0:
        print(f"   🧹 Removed {removed} duplicate rows from games data.")
        # Optional: save back cleaned games
        # dfGames.to_csv(GAMES_PATH, index=False) 
    
    # Clean Player_ID
    dfGames['Player_ID'] = dfGames['Player_ID'].astype(str).apply(lambda x: x.split('.')[0])
    dfGames['Game_ID'] = dfGames['Game_ID'].astype(str).apply(lambda x: x.split('.')[0])
    
    # Identify unique Tasks (Player, Season)
    tasks = dfGames[['Player_ID', 'Season']].drop_duplicates().values.tolist()
    print(f"📋 Found {len(tasks)} unique (Player, Season) pairs.")
    print(f"📋 Total Games expecting shots: {len(dfGames)}")

    # 2. Check Resume / Existing Data
    existingShots = pd.DataFrame()
    processed_pid_season = set()
    
    if os.path.exists(OUTPUT_SHOTS_PATH):
        print(f"⚠️ Found existing {OUTPUT_SHOTS_PATH}, reading to resume...")
        try:
            existingShots = pd.read_csv(OUTPUT_SHOTS_PATH, low_memory=False)
            existingShots['Player_ID'] = existingShots['Player_ID'].astype(str).apply(lambda x: x.split('.')[0])
            existingShots['GAME_ID'] = existingShots['GAME_ID'].astype(str).apply(lambda x: x.split('.')[0])
            
            # Identify which PID/Season are already "likely" complete
            # Heuristic: If we have shots for Player P, we *might* assume his season is done?
            # Safer: Look at the Games DataFrame vs Existing Shots match rate
            # But "Resume" by season is faster. 
            # Let's map downloaded GameIDs back to seasons to mark tasks as done.
            downloadedGames = set(existingShots['GAME_ID'].unique())
            
            # Simple check: If a Player-Season pair has > 90% of its games in downloadedGames, skip it
            # Pre-calculate counts
            task_counts = dfGames.groupby(['Player_ID', 'Season']).size().to_dict()
            
            # Check overlap
            for pid, seas in tasks:
                # Get games for this task
                task_game_ids = dfGames[(dfGames['Player_ID'] == pid) & (dfGames['Season'] == seas)]['Game_ID'].values
                matches = sum(1 for gid in task_game_ids if gid in downloadedGames)
                total = len(task_game_ids)
                
                if total > 0 and (matches / total) > 0.9: # 90% tolerance for resume
                    processed_pid_season.add((pid, seas))
                    
            print(f"⏩ Skipping {len(processed_pid_season)} tasks that seem complete.")
            
        except Exception as e:
            print(f"⚠️ Error reading existing file: {e}. Starting fresh.")
            existingShots = pd.DataFrame()

    tasksToRun = [t for t in tasks if (str(t[0]), str(t[1])) not in processed_pid_season]
    
    batchShots = []
    saveInterval = 20
    count = 0

    print(f"Step 2: Start Fetching Shost for {len(tasksToRun)} tasks...")
    
    for pid, season in tqdm(tasksToRun, desc="Fetching"):
        try:
            shotApi = fetchWithRetry(
                shotchartdetail.ShotChartDetail,
                team_id=0, player_id=pid, 
                context_measure_simple='FGA', season_nullable=season
            )
            
            if shotApi:
                dfNew = shotApi.get_data_frames()[0]
                if not dfNew.empty:
                    # CRITICAL FIX: Add Player_ID
                    dfNew['Player_ID'] = str(pid)
                    
                    # Ensure cols exist
                    cols = ['Player_ID', 'GAME_ID', 'LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG', 'SHOT_TYPE', 'ACTION_TYPE']
                    validCols = [c for c in cols if c in dfNew.columns]
                    
                    dfNew = dfNew[validCols]
                    batchShots.append(dfNew)
                    count += 1
            
            # Incremental Save (Optimized: Append to list, then concat occasionally)
            if len(batchShots) >= saveInterval:
                if not existingShots.empty:
                    dfCurrentBatch = pd.concat(batchShots, ignore_index=True)
                    existingShots = pd.concat([existingShots, dfCurrentBatch], ignore_index=True)
                else:
                    existingShots = pd.concat(batchShots, ignore_index=True)
                
                # Save checkpoint
                existingShots.to_csv(OUTPUT_SHOTS_PATH, index=False)
                batchShots = [] # Clear memory
                
        except Exception as e:
            print(f"Error fetching {pid} {season}: {e}")

    # Final Merge
    if batchShots:
        if not existingShots.empty:
            dfCurrentBatch = pd.concat(batchShots, ignore_index=True)
            existingShots = pd.concat([existingShots, dfCurrentBatch], ignore_index=True)
        else:
            existingShots = pd.concat(batchShots, ignore_index=True)

    print("Step 3: Verification & Dedup...")
    if not existingShots.empty:
        # Standardize ID formats
        existingShots['Player_ID'] = existingShots['Player_ID'].astype(str).apply(lambda x: x.split('.')[0])
        existingShots['GAME_ID'] = existingShots['GAME_ID'].astype(str).apply(lambda x: x.split('.')[0])
        
        # Deduplication
        existingShots, removed = clean_duplicates(existingShots, subset=['GAME_ID', 'LOC_X', 'LOC_Y', 'SHOT_TYPE'])
        print(f"🧹 Removed {removed} duplicate shots.")
        
        # Coverage Check
        uniqueGamesInShots = set(existingShots['GAME_ID'].unique())
        uniqueGamesInGames = set(dfGames['Game_ID'].unique())
        
        missing = uniqueGamesInGames - uniqueGamesInShots
        coverage = 100 * (1 - len(missing) / len(uniqueGamesInGames))
        
        print(f"📊 Coverage Report: {len(uniqueGamesInShots)}/{len(uniqueGamesInGames)} games found ({coverage:.2f}%)")
        
        if len(missing) > 0 and len(missing) < 50:
            print(f"⚠️ Missing Game IDs: {list(missing)}")
        elif len(missing) >= 50:
            print(f"⚠️ Missing {len(missing)} games. (Use 'missing' list to debug if needed)")

        # Save Final
        existingShots.to_csv(OUTPUT_SHOTS_PATH, index=False)
        print(f"✅ Final validated data saved to {OUTPUT_SHOTS_PATH}")
    else:
        print("❌ No shots data found.")

if __name__ == "__main__":
    main()
