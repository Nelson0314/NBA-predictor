import os
import time
import random
import pandas as pd
from tqdm import tqdm
from requests.exceptions import ReadTimeout, ConnectTimeout
import urllib3
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, shotchartdetail, leaguedashplayerstats, playergamelogs

# ==========================================
# 1. 設定
# ==========================================
SEASONS = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24']
TOP_N_PLAYERS = 150
DATASET_DIR = 'dataset'

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

GAMES_CSV_PATH = os.path.join(DATASET_DIR, 'games.csv')
SHOTS_CSV_PATH = os.path.join(DATASET_DIR, 'shots.csv')

# ==========================================
# 2. 輔助函式
# ==========================================
def fetch_with_retry(api_func, max_retries=3, **kwargs):
    kwargs['timeout'] = 25
    for i in range(max_retries):
        try:
            time.sleep(random.uniform(1.5, 3.0)) 
            return api_func(**kwargs)
        except (ReadTimeout, ConnectTimeout, urllib3.exceptions.ReadTimeoutError, ConnectionResetError):
            time.sleep(10)
        except Exception:
            break
    return None

def get_top_scorers(season, top_n=100):
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season, per_mode_detailed='PerGame', timeout=30)
        df = stats.get_data_frames()[0]
        return df.sort_values(by='PTS', ascending=False).head(top_n)[['PLAYER_ID', 'PLAYER_NAME']].to_dict('records')
    except: return []

# 用來清理重複資料的函式
def clean_duplicates(filepath, subset_cols):
    if os.path.exists(filepath):
        print(f"🧹 正在清理重複資料: {filepath} ...", end='\r')
        df = pd.read_csv(filepath)
        original_len = len(df)
        # 針對特定欄位去重 (保留最後一筆)
        df = df.drop_duplicates(subset=subset_cols, keep='last')
        df.to_csv(filepath, index=False)
        print(f"✅ 清理完成: {filepath} (移除 {original_len - len(df)} 筆重複)")

# ==========================================
# 3. 準備任務與檢查續傳
# ==========================================
print("步驟 1/3: 建立球員名單...")
target_player_ids = {} 
for season in SEASONS:
    for p in get_top_scorers(season, TOP_N_PLAYERS):
        target_player_ids[p['PLAYER_ID']] = p['PLAYER_NAME']

# 建立所有任務列表
all_tasks = []
for pid, pname in target_player_ids.items():
    for season in SEASONS:
        all_tasks.append((pid, pname, season))

# --- 🟢 關鍵改良：嚴格的續傳檢查 ---
processed_tasks = set()
if os.path.exists(GAMES_CSV_PATH):
    try:
        # 強制將 Player_ID 讀取為 string，避免 int/str 混淆
        existing_df = pd.read_csv(GAMES_CSV_PATH, dtype={'Player_ID': str, 'Season': str})
        
        # 建立已完成的 (ID, Season) 集合
        for _, row in existing_df.iterrows():
            processed_tasks.add((str(row['Player_ID']), str(row['Season'])))
            
        print(f"🔄 讀取舊檔成功，已完成 {len(processed_tasks)} 個任務。")
    except Exception as e: 
        print(f"⚠️ 讀取舊檔失敗 (可能是空檔): {e}")

# 過濾任務 (確保比對時也轉成 string)
tasks_to_run = [
    t for t in all_tasks 
    if (str(t[0]), str(t[2])) not in processed_tasks
]
print(f"🚀 總任務: {len(all_tasks)} | 剩餘任務: {len(tasks_to_run)}")

# ==========================================
# 4. 執行爬蟲
# ==========================================

if not tasks_to_run:
    print("所有任務已完成！跳至清理步驟。")
else:
    with tqdm(total=len(tasks_to_run), desc="初始化中", dynamic_ncols=True, unit="task") as pbar:
        
        for pid, pname, season in tasks_to_run:
            pbar.set_description(f"正在抓取: {pname} ({season})")
            
            # --- 雙重檢查：防止在本次執行期間重複 ---
            # (雖然 tasks_to_run 已經濾過了，但這是保險)
            if (str(pid), str(season)) in processed_tasks:
                pbar.update(1)
                continue

            batch_games = []
            batch_shots = []

            # --- A. 基礎數據 ---
            base_api = fetch_with_retry(playergamelog.PlayerGameLog, player_id=pid, season=season)
            if not base_api: 
                pbar.update(1); continue
            df_base = base_api.get_data_frames()[0]
            if df_base.empty: 
                pbar.update(1); continue

            # --- B. 進階數據 ---
            adv_api = fetch_with_retry(
                playergamelogs.PlayerGameLogs, 
                player_id_nullable=pid, season_nullable=season,
                measure_type_player_game_logs_nullable='Advanced'
            )
            df_merged = df_base
            if adv_api:
                df_adv = adv_api.get_data_frames()[0]
                if not df_adv.empty:
                    df_base['Game_ID'] = df_base['Game_ID'].astype(str)
                    df_adv['GAME_ID'] = df_adv['GAME_ID'].astype(str)
                    adv_cols = ['GAME_ID', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'AST_PCT', 'AST_TO', 
                                'OREB_PCT', 'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT', 'USG_PCT', 'PACE', 'PIE']
                    valid_cols = [c for c in adv_cols if c in df_adv.columns]
                    df_merged = pd.merge(df_base, df_adv[valid_cols], left_on='Game_ID', right_on='GAME_ID', how='left')

            # --- C. 整理 ---
            try: df_merged['GAME_DATE'] = pd.to_datetime(df_merged['GAME_DATE'])
            except: pass
            df_merged = df_merged.sort_values('GAME_DATE').reset_index(drop=True)
            df_merged['TARGET_PTS'] = df_merged['PTS'].shift(-1)
            df_merged['Player_ID'] = pid
            df_merged['Player_Name'] = pname
            df_merged['Season'] = season
            df_merged = df_merged.dropna(subset=['TARGET_PTS'])
            batch_games.append(df_merged)

            # --- D. 投籃圖 ---
            shot_api = fetch_with_retry(
                shotchartdetail.ShotChartDetail,
                team_id=0, player_id=pid, 
                context_measure_simple='FGA', season_nullable=season
            )
            if shot_api:
                df_shots = shot_api.get_data_frames()[0]
                if not df_shots.empty:
                    s_cols = ['GAME_ID', 'LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG', 'SHOT_TYPE', 'ACTION_TYPE']
                    valid = [c for c in s_cols if c in df_shots.columns]
                    batch_shots.append(df_shots[valid])

            # --- E. 存檔 ---
            if batch_games:
                df_g = pd.concat(batch_games, ignore_index=True)
                df_g.to_csv(GAMES_CSV_PATH, mode='a', header=not os.path.exists(GAMES_CSV_PATH), index=False)
            
            if batch_shots:
                df_s = pd.concat(batch_shots, ignore_index=True)
                df_s.to_csv(SHOTS_CSV_PATH, mode='a', header=not os.path.exists(SHOTS_CSV_PATH), index=False)

            # 更新狀態與進度條
            processed_tasks.add((str(pid), str(season)))
            pbar.update(1)

# ==========================================
# 5. 收尾：資料去重清理 (Final Cleanup)
# ==========================================
print("\n步驟 3/3: 正在進行最終資料庫去重與清理...")

# 針對 Games 表，如果有重複的 (Player_ID, Game_ID)，只留一筆
clean_duplicates(GAMES_CSV_PATH, subset_cols=['Player_ID', 'Game_ID'])

# 針對 Shots 表，如果有重複的 (GAME_ID, EVENT_ID 或座標)，這比較難判斷，通常用 GAME_ID + LOC_X + LOC_Y 
# 但最簡單是去掉完全重複的行
clean_duplicates(SHOTS_CSV_PATH, subset_cols=None) # None 代表檢查所有欄位是否完全一樣

print("\n🎉 全部作業完成！資料庫已保證乾淨無重複。")