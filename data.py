import os
import time
import random
import pandas as pd
from tqdm.notebook import tqdm
from requests.exceptions import ReadTimeout, ConnectionError
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, shotchartdetail, leaguedashplayerstats

# ================= Configuration =================
SEASONS = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24']
TOP_N_PLAYERS = 150
DATASET_ROOT = 'dataset_numeric'
os.makedirs(DATASET_ROOT, exist_ok=True)
# =================================================

# 1. 取得球員清單 (跟之前一樣)
def get_top_scorers(season, top_n=100):
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season, per_mode_detailed='PerGame')
        df = stats.get_data_frames()[0]
        return df.sort_values(by='PTS', ascending=False).head(top_n)[['PLAYER_ID', 'PLAYER_NAME']].to_dict('records')
    except: return []

target_player_ids = {}
print("Building Player List...")
for season in SEASONS:
    for p in get_top_scorers(season, TOP_N_PLAYERS):
        target_player_ids[p['PLAYER_ID']] = p['PLAYER_NAME']
    time.sleep(1)

print(f"Total Players: {len(target_player_ids)}")

# 2. 準備容器
all_games = []
all_shots = []

# 3. 開始爬取
print("Starting Data Collection...")
for pid, pname in tqdm(target_player_ids.items(), desc="Players"):
    for season in SEASONS:
        try:
            # A. 抓 Game Log (數值)
            time.sleep(random.uniform(0.5, 0.8)) # 隨機延遲
            gamelog = playergamelog.PlayerGameLog(player_id=pid, season=season)
            df_log = gamelog.get_data_frames()[0]
            if df_log.empty: continue
            
            # 處理日期與 Target
            df_log['GAME_DATE'] = pd.to_datetime(df_log['GAME_DATE'], format='%b %d, %Y')
            df_log = df_log.sort_values('GAME_DATE').reset_index(drop=True)
            df_log['TARGET_PTS'] = df_log['PTS'].shift(-1)
            
            # 存入 Game List
            # 只要存需要的欄位就好，節省空間
            cols = ['Game_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'PTS', 'REB', 'AST', 'MIN', 'TARGET_PTS']
            # 把 Player ID 加上去
            df_subset = df_log[cols].copy()
            df_subset['Player_ID'] = pid
            df_subset['Season'] = season
            all_games.append(df_subset)
            
            # B. 抓 Shot Chart (座標) - 這是重點，只存座標不畫圖
            time.sleep(random.uniform(0.5, 0.8))
            shot_api = shotchartdetail.ShotChartDetail(
                team_id=0, player_id=pid, 
                context_measure_simple='FGA', season_nullable=season
            )
            df_shots = shot_api.get_data_frames()[0]
            
            if not df_shots.empty:
                # 只保留我們需要的座標資訊
                shot_cols = ['GAME_ID', 'LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG', 'SHOT_TYPE']
                all_shots.append(df_shots[shot_cols])
                
        except Exception as e:
            print(f"Error: {e}")
            continue

# 4. 存檔
print("Saving CSVs...")
if all_games:
    df_games_master = pd.concat(all_games, ignore_index=True)
    # 去掉沒有 Target 的最後一場
    df_games_master = df_games_master.dropna(subset=['TARGET_PTS'])
    df_games_master.to_csv(os.path.join(DATASET_ROOT, 'games.csv'), index=False)

if all_shots:
    df_shots_master = pd.concat(all_shots, ignore_index=True)
    # 使用 Parquet 格式存座標數據會快非常多 (因為資料量大)，如果不行就用 CSV
    try:
        df_shots_master.to_parquet(os.path.join(DATASET_ROOT, 'shots.parquet'))
    except:
        df_shots_master.to_csv(os.path.join(DATASET_ROOT, 'shots.csv'), index=False)

print("Done! Only raw data saved.")