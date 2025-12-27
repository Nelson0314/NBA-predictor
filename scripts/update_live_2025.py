import pandas as pd
import numpy as np
import time
import os
from nba_api.stats.endpoints import leaguegamelog, shotchartdetail, leaguedashplayerstats, boxscoretraditionalv2
from nba_api.stats.static import players, teams
from tqdm import tqdm

import sys
# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import DATA_DIR

# Config
# Config
SEASON = '2025-26'
CUTOFF_DATE = '2025-12-26' # YYYY-MM-DD
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
print("3. Fetching Shot Charts...")
all_shots = []

for pid in tqdm(top_scorers):
    try:
        shot_df = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=pid,
            season_nullable=SEASON,
            context_measure_simple='FGA' # or 'PTS'
        ).get_data_frames()[0]
        
        # Filter Date
        shot_df['GAME_DATE'] = pd.to_datetime(shot_df['GAME_DATE'], format='%Y%m%d')
        shot_df = shot_df[shot_df['GAME_DATE'] <= cutoff_dt]
        
        all_shots.append(shot_df)
        time.sleep(0.6) # Rate limit
    except Exception as e:
        # print(f"  Failed for {pid}: {e}")
        pass

if all_shots:
    shots_final = pd.concat(all_shots, ignore_index=True)
    shots_final = shots_final.rename(columns={'PLAYER_ID': 'Player_ID', 'PLAYER_NAME': 'Player_Name', 'GAME_ID': 'GAME_ID', 'TEAM_ID': 'TEAM_ID'})
    shots_final.to_csv(os.path.join(OUTPUT_DIR, 'shots_2025.csv'), index=False)
    print(f"  Saved shots_2025.csv ({len(shots_final)} rows)")
else:
    print("  No shots fetched.")

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
