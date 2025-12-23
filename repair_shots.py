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
            print(f"   ⚠️ Timeout/Connection error. Retrying {i+1}/{maxRetries}...")
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
    # Ensure Game_ID and Player_ID are strings
    dfGames['Game_ID'] = dfGames['Game_ID'].astype(str).str.split('.').str[0]
    dfGames['Player_ID'] = dfGames['Player_ID'].astype(str).str.split('.').str[0]
    
    dfGames, removed = clean_duplicates(dfGames, subset=['Game_ID', 'Player_ID'])
    if removed > 0:
        print(f"   🧹 Removed {removed} duplicate rows from games data (in-memory).")
    
    # Identify unique Tasks (Player, Season)
    # Season might be int or str, normalize to str
    dfGames['Season'] = dfGames['Season'].astype(str)
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
            if not existingShots.empty:
                existingShots['Player_ID'] = existingShots['Player_ID'].astype(str).str.split('.').str[0]
                existingShots['GAME_ID'] = existingShots['GAME_ID'].astype(str).str.split('.').str[0]
                
                downloadedGames = set(existingShots['GAME_ID'].unique())
                
                # Check which tasks are complete
                for pid, seas in tasks:
                    task_game_ids = dfGames[(dfGames['Player_ID'] == pid) & (dfGames['Season'] == seas)]['Game_ID'].values
                    if len(task_game_ids) == 0: continue
                    
                    matches = sum(1 for gid in task_game_ids if gid in downloadedGames)
                    # If we found at least 90% of games, assume this season fetch was done
                    if (matches / len(task_game_ids)) > 0.9:
                        processed_pid_season.add((pid, seas))
                        
                print(f"⏩ Skipping {len(processed_pid_season)} tasks that ensure >90% coverage.")
        except Exception as e:
            print(f"⚠️ Error reading existing file: {e}. Starting fresh.")
            existingShots = pd.DataFrame()

    tasksToRun = [t for t in tasks if (str(t[0]), str(t[1])) not in processed_pid_season]
    
    batchShots = []
    saveInterval = 20 # Save every 20 queries
    
    print(f"Step 2: Start Fetching Shots for {len(tasksToRun)} tasks...")
    
    try:
        with tqdm(total=len(tasksToRun), desc="Fetching") as pbar:
            for pid, season in tasksToRun:
                pbar.set_description(f"Fetching {pid} - {season}")
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
                        dfNew['GAME_ID'] = dfNew['GAME_ID'].astype(str)
                        
                        # Ensure cols exist
                        cols = ['Player_ID', 'GAME_ID', 'LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG', 'SHOT_TYPE', 'ACTION_TYPE']
                        validCols = [c for c in cols if c in dfNew.columns]
                        
                        dfNew = dfNew[validCols]
                        batchShots.append(dfNew)
                
                # Incremental Update to Memory & Disk
                if len(batchShots) >= saveInterval:
                    dfCurrentBatch = pd.concat(batchShots, ignore_index=True)
                    if not existingShots.empty:
                        existingShots = pd.concat([existingShots, dfCurrentBatch], ignore_index=True)
                    else:
                        existingShots = dfCurrentBatch
                    
                    # Deduplicate before save to keep size manageable
                    existingShots, _ = clean_duplicates(existingShots, subset=['GAME_ID', 'LOC_X', 'LOC_Y', 'SHOT_TYPE'])
                    existingShots.to_csv(OUTPUT_SHOTS_PATH, index=False)
                    batchShots = [] # Clear batch LIST, but keep main DF
                    
                pbar.update(1)
                
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Saving progress...")

    # Final Merge & Save
    if batchShots:
        dfCurrentBatch = pd.concat(batchShots, ignore_index=True)
        if not existingShots.empty:
            existingShots = pd.concat([existingShots, dfCurrentBatch], ignore_index=True)
        else:
            existingShots = dfCurrentBatch

    print("\nStep 3: Final Verification & Deduplication...")
    if not existingShots.empty:
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
            print(f"⚠️ Missing {len(missing)} games. (Check network or missing data source)")

        # Save Final
        existingShots.to_csv(OUTPUT_SHOTS_PATH, index=False)
        print(f"✅ Final validated data saved to {OUTPUT_SHOTS_PATH}")
    else:
        print("❌ No shots data collected.")

if __name__ == "__main__":
    main()
