import requests
import json
from scipy.stats import norm

# 你的API金鑰
API_KEY = "507566ed32ab4d902263ce5a407351a7"

def calculate_odds(house_pred, line, std_dev=9.0):
    """
    Calculate decimal odds based on the House Prediction vs the Line.
    Assumes normal distribution with fixed std_dev (approx for NBA player props).
    """
    # Z-score for the Line relative to House Prediction
    # Prob(Score > Line) = 1 - CDF((Line - Pred) / Std)
    
    z = (line - house_pred) / std_dev
    prob_over = 1 - norm.cdf(z)
    prob_under = 1.0 - prob_over
    
    # Avoid infinite odds
    prob_over = max(0.01, min(0.99, prob_over))
    prob_under = max(0.01, min(0.99, prob_under))
    
    odds_over = 1.0 / prob_over
    odds_under = 1.0 / prob_under
    
    return odds_over, odds_under, prob_over

def fetch_odds():
    """
    Fetch NBA odds from the-odds-api and save to 'event_odds_data.json'.
    Returns the data list.
    """
    print("Fetching Live Odds from API...")
    # 獲取NBA即將開始的比賽
    odds_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?apiKey={API_KEY}&regions=us&bookmakers=fanduel&markets?"
    odds_response = requests.get(odds_url)

    # 檢查HTTP響應狀態碼
    if odds_response.status_code != 200:
        print(f"API請求失敗，狀態碼: {odds_response.status_code}")
        print(odds_response.text)
        return []
    
    odds_data = odds_response.json()

    # 確保odds_data是列表
    if isinstance(odds_data, list) and odds_data:
        # 提取比賽ID和賭盤資料
        all_event_odds = []
        for event in odds_data:
            game_id = event['id']
            print(f"比賽ID: {game_id}, 主隊: {event['home_team']}, 客隊: {event['away_team']}")

            # 定義要獲取的市場列表
            markets = [
                'player_points', 
                'player_assists', 
                'player_rebounds',
                'player_points_rebounds_assists',
                'player_points_rebounds',
                'player_points_assists',
                'player_rebounds_assists'
            ]
            
            for market in markets:
                event_odds_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{game_id}/odds?apiKey={API_KEY}&regions=us&bookmakers=fanduel&markets={market}"
                event_odds_response = requests.get(event_odds_url)

                if event_odds_response.status_code == 200:
                     event_odds_data = event_odds_response.json()
                     all_event_odds.append(event_odds_data)
                else:
                    # Silent or debug print to avoid spam if market missing
                    # print(f"獲取賭盤資料失敗 ({market})")
                    pass

        # 將所有賭盤資料存儲到JSON文件
        with open('event_odds_data.json', 'w', encoding='utf-8') as f:
            json.dump(all_event_odds, f, ensure_ascii=False, indent=4)
        print("賭盤資料已存儲到 event_odds_data.json")
        return all_event_odds
    else:
        print("未找到即將開始的比賽")
        return []

if __name__ == "__main__":
    fetch_odds()
