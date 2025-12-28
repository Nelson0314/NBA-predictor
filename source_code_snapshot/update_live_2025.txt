import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
from nba_api.stats.endpoints import leaguegamelog, shotchartdetail, leaguedashplayerstats, boxscoretraditionalv2
from nba_api.stats.static import players, teams
from tqdm import tqdm

import sys
# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import DATA_DIR

from datetime import datetime
# Config
SEASON = '2025-26'
CUTOFF_DATE = datetime.today().strftime('%Y-%m-%d') # Default to Today
# Can override if needed via args, but for auto-bet, today is good.
OUTPUT_DIR = os.path.join(DATA_DIR, 'live_2025')
TOP_N_PLAYERS = 150

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Fetching data for {SEASON} up to {CUTOFF_DATE}...")

# 1. Fetch Games (Game Log)
print("1. Fetching Game Log...")
gl = leaguegamelog.LeagueGameLog(season=SEASON, player_or_team_abbreviation='P').get_data_frames()[0]

# Filter Date
gl['GAME_DATE'] = pd.to_datetime(gl['GAME_DATE'])
cutoff_dt = pd.to_datetime(CUTOFF_DATE)
gl = gl[gl['GAME_DATE'] <= cutoff_dt]
print(f"  Found {len(gl)} player-game records.")

# Rename basics to match our convention strictly
rename_map = {
    'PLAYER_ID': 'Player_ID',
    'GAME_ID': 'GAME_ID', # Uppercase per multiModel expectation
    'TEAM_ID': 'TEAM_ID', # Uppercase
    'PLAYER_NAME': 'Player_Name',
    'MATCHUP': 'MATCHUP' # Ensure exists
}
# Only rename columns that exist
gl = gl.rename(columns={k:v for k,v in rename_map.items() if k in gl.columns})

# Add Defaults for Advanced Stats (Missing in basic log)
defaults = {
    'OFF_RATING': 112.0,
    'DEF_RATING': 112.0,
    'PACE': 99.0,
    'USG_PCT': 0.20,
    'TS_PCT': 0.57,
    'PLUS_MINUS': 0.0,
    'MIN': 24.0
}
for col, val in defaults.items():
    if col not in gl.columns:
        gl[col] = val

# MIN cleanup
def convert_min(x):
    if isinstance(x, str):
        if ':' in x:
            parts = x.split(':')
            return float(parts[0]) + float(parts[1])/60
        return float(x)
    return x

if 'MIN' in gl.columns:
    gl['MIN'] = gl['MIN'].apply(convert_min)

gl.to_csv(os.path.join(OUTPUT_DIR, 'games_2025.csv'), index=False)
print(f"  Saved games_2025.csv ({len(gl)} rows)")

# 2. Identify Top Players for Shots
print("2. Identifying Top Scorers for Shot Data...")
top_scorers = gl.groupby('Player_ID')['PTS'].sum().sort_values(ascending=False).head(TOP_N_PLAYERS).index.tolist()
print(f"  Selected top {len(top_scorers)} players.")

# 3. Fetch Shots
# 3. Fetch Shots
print("3. Fetching Shot Charts...")

shots_file = os.path.join(OUTPUT_DIR, 'shots_2025.csv')
existing_shots = pd.DataFrame()
start_date_str = ''

if os.path.exists(shots_file):
    print(f"  Loading existing shots from {shots_file}...")
    existing_shots = pd.read_csv(shots_file)
    if 'GAME_DATE' in existing_shots.columns and not existing_shots.empty:
        # Check max date
        # Ensure datetime
        existing_shots['GAME_DATE'] = pd.to_datetime(existing_shots['GAME_DATE'])
        max_date = existing_shots['GAME_DATE'].max()
        # Start from next day
        start_date = max_date + pd.Timedelta(days=1)
        start_date_str = start_date.strftime('%m/%d/%Y') # API format MM/DD/YYYY usually
        
        # Stop if start_date > cutoff
        if start_date > cutoff_dt:
             print("  Data up to date. Skipping shot fetch.")
             all_shots = [] # Trigger skip
        else:
             print(f"  Fetching shots starting from {start_date_str}...")

all_shots = []

# If we are fetching (i.e. we didn't determine it's up to date)
# Note: start_date_str might be empty if file doesn't exist -> fetch all
should_fetch = True
if start_date_str and start_date > cutoff_dt:
    should_fetch = False

if should_fetch:
    for pid in tqdm(top_scorers):
        try:
            params = {
                'team_id': 0,
                'player_id': pid,
                'season_nullable': SEASON,
                'context_measure_simple': 'FGA'
            }
            if start_date_str:
                params['date_from_nullable'] = start_date_str
                
            shot_df = shotchartdetail.ShotChartDetail(**params).get_data_frames()[0]
            
            if not shot_df.empty:
                # Filter Date (redundant if API works but safe)
                # API format might differ, standardizing
                if 'GAME_DATE' in shot_df.columns:
                     shot_df['GAME_DATE'] = pd.to_datetime(shot_df['GAME_DATE'], format='%Y%m%d')
                     shot_df = shot_df[shot_df['GAME_DATE'] <= cutoff_dt]
                     
                all_shots.append(shot_df)
            time.sleep(0.6) # Rate limit
        except Exception as e:
            pass

if all_shots:
    new_shots = pd.concat(all_shots, ignore_index=True)
    new_shots = new_shots.rename(columns={'PLAYER_ID': 'Player_ID', 'PLAYER_NAME': 'Player_Name', 'GAME_ID': 'GAME_ID', 'TEAM_ID': 'TEAM_ID'})
    
    if not existing_shots.empty:
        # Merge
        # Ensure existing has consistent cols
        # Concat
        shots_final = pd.concat([existing_shots, new_shots], ignore_index=True)
        # Drop duplicates just in case
        shots_final = shots_final.drop_duplicates(subset=['GAME_ID', 'Player_ID', 'GAME_EVENT_ID'])
    else:
        shots_final = new_shots
        
    shots_final.to_csv(shots_file, index=False)
    print(f"  Saved shots_2025.csv ({len(shots_final)} rows) - Added {len(new_shots)} new rows.")
else:
    if not existing_shots.empty:
         # Just save existing back (maybe filtered by date? No, keep history)
         print("  No new shots fetched.")
    else:
         print("  No shots fetched at all.")

# 4. Generate Teams Metadata (Rolling inputs)
print("4. Generating Team Stats (for teams.csv)...")
# Aggregate from player logs to Team-Game level
# Use mapped names (GAME_ID, TEAM_ID)
# Ensure columns exist before grouping
group_cols = ['GAME_ID', 'TEAM_ID', 'GAME_DATE']
if 'TEAM_ABBREVIATION' in gl.columns: group_cols.append('TEAM_ABBREVIATION')

# Stats to sum
sum_cols = ['PTS', 'AST', 'REB', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS']
# Filter sum_cols to existing
sum_cols = [c for c in sum_cols if c in gl.columns]

team_stats = gl.groupby(group_cols)[sum_cols].sum().reset_index()

print("  Calculating Rolling Averages & Derived Stats...")
team_stats = team_stats.sort_values(['TEAM_ID', 'GAME_DATE'])

# Derived AVG Cols
for col in sum_cols:
    team_stats[f'AVG_{col}'] = team_stats.groupby('TEAM_ID')[col].transform(lambda x: x.expanding().mean())

# Derived Percentages (needed for featureCols in multiModel)
if 'AVG_FGM' in team_stats.columns and 'AVG_FGA' in team_stats.columns:
    team_stats['AVG_FG_PCT'] = team_stats['AVG_FGM'] / (team_stats['AVG_FGA'] + 1e-6)
else:
    team_stats['AVG_FG_PCT'] = 0.45

if 'AVG_FG3M' in team_stats.columns and 'AVG_FG3A' in team_stats.columns:
    team_stats['AVG_FG3_PCT'] = team_stats['AVG_FG3M'] / (team_stats['AVG_FG3A'] + 1e-6)
else:
    team_stats['AVG_FG3_PCT'] = 0.35

if 'AVG_FTM' in team_stats.columns and 'AVG_FTA' in team_stats.columns:
    team_stats['AVG_FT_PCT'] = team_stats['AVG_FTM'] / (team_stats['AVG_FTA'] + 1e-6)
else:
    team_stats['AVG_FT_PCT'] = 0.75
    
team_stats['AVG_TS_PCT'] = 0.57 # Default

# Save
team_stats.to_csv(os.path.join(OUTPUT_DIR, 'teams_2025.csv'), index=False)
print(f"  Saved teams_2025.csv ({len(team_stats)} rows)")

print("Done.")
