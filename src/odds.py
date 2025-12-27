import requests
import json

# 你的API金鑰
api_key = "a9d6aa1d61e1eff2b15c5f7152a3b480"

# 獲取NBA即將開始的比賽
odds_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?apiKey={api_key}&regions=us&bookmakers=fanduel&markets?"
odds_response = requests.get(odds_url)

# 檢查HTTP響應狀態碼
if odds_response.status_code != 200:
    print(f"API請求失敗，狀態碼: {odds_response.status_code}")
    print(odds_response.text)  # 打印錯誤信息
else:
    odds_data = odds_response.json()

    # 確保odds_data是列表
    if isinstance(odds_data, list) and odds_data:
        # 提取比賽ID和賭盤資料
        all_event_odds = []  # 用於存儲所有賭盤資料
        for event in odds_data:
            game_id = event['id']  # 獲取比賽ID
            print(f"比賽ID: {game_id}, 主隊: {event['home_team']}, 客隊: {event['away_team']}")

            # 使用game_id獲取賭盤資料
            event_odds_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{game_id}/odds?apiKey={api_key}&regions=us&bookmakers=fanduel&markets=player_points"
            event_odds_response = requests.get(event_odds_url)

            if event_odds_response.status_code == 200:
                event_odds_data = event_odds_response.json()
                all_event_odds.append(event_odds_data)  # 存儲賭盤資料
            else:
                print(f"獲取賭盤資料失敗，狀態碼: {event_odds_response.status_code}")
                print(event_odds_response.text)  # 打印錯誤信息
            # 使用game_id獲取賭盤資料
            event_odds_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{game_id}/odds?apiKey={api_key}&regions=us&bookmakers=fanduel&markets=player_assists"
            event_odds_response = requests.get(event_odds_url)

            if event_odds_response.status_code == 200:
                event_odds_data = event_odds_response.json()
                all_event_odds.append(event_odds_data)  # 存儲賭盤資料
            else:
                print(f"獲取賭盤資料失敗，狀態碼: {event_odds_response.status_code}")
                print(event_odds_response.text)  # 打印錯誤信息
            # 使用game_id獲取賭盤資料
            event_odds_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{game_id}/odds?apiKey={api_key}&regions=us&bookmakers=fanduel&markets=player_rebounds"
            event_odds_response = requests.get(event_odds_url)

            if event_odds_response.status_code == 200:
                event_odds_data = event_odds_response.json()
                all_event_odds.append(event_odds_data)  # 存儲賭盤資料
            else:
                print(f"獲取賭盤資料失敗，狀態碼: {event_odds_response.status_code}")
                print(event_odds_response.text)  # 打印錯誤信息
            # 使用game_id獲取賭盤資料
            event_odds_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{game_id}/odds?apiKey={api_key}&regions=us&bookmakers=fanduel&markets=player_points_rebounds_assists"
            event_odds_response = requests.get(event_odds_url)

            if event_odds_response.status_code == 200:
                event_odds_data = event_odds_response.json()
                all_event_odds.append(event_odds_data)  # 存儲賭盤資料
            else:
                print(f"獲取賭盤資料失敗，狀態碼: {event_odds_response.status_code}")
                print(event_odds_response.text)  # 打印錯誤信息
            # 使用game_id獲取賭盤資料
            event_odds_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{game_id}/odds?apiKey={api_key}&regions=us&bookmakers=fanduel&markets=player_points_rebounds_assists"
            event_odds_response = requests.get(event_odds_url)

            if event_odds_response.status_code == 200:
                event_odds_data = event_odds_response.json()
                all_event_odds.append(event_odds_data)  # 存儲賭盤資料
            else:
                print(f"獲取賭盤資料失敗，狀態碼: {event_odds_response.status_code}")
                print(event_odds_response.text)  # 打印錯誤信息
            # 使用game_id獲取賭盤資料
            event_odds_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{game_id}/odds?apiKey={api_key}&regions=us&bookmakers=fanduel&markets=player_points_rebounds"
            event_odds_response = requests.get(event_odds_url)

            if event_odds_response.status_code == 200:
                event_odds_data = event_odds_response.json()
                all_event_odds.append(event_odds_data)  # 存儲賭盤資料
            else:
                print(f"獲取賭盤資料失敗，狀態碼: {event_odds_response.status_code}")
                print(event_odds_response.text)  # 打印錯誤信息
            # 使用game_id獲取賭盤資料
            event_odds_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{game_id}/odds?apiKey={api_key}&regions=us&bookmakers=fanduel&markets=player_points_assists"
            event_odds_response = requests.get(event_odds_url)

            if event_odds_response.status_code == 200:
                event_odds_data = event_odds_response.json()
                all_event_odds.append(event_odds_data)  # 存儲賭盤資料
            else:
                print(f"獲取賭盤資料失敗，狀態碼: {event_odds_response.status_code}")
                print(event_odds_response.text)  # 打印錯誤信息
            # 使用game_id獲取賭盤資料
            event_odds_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{game_id}/odds?apiKey={api_key}&regions=us&bookmakers=fanduel&markets=player_rebounds_assists"
            event_odds_response = requests.get(event_odds_url)

            if event_odds_response.status_code == 200:
                event_odds_data = event_odds_response.json()
                all_event_odds.append(event_odds_data)  # 存儲賭盤資料
            else:
                print(f"獲取賭盤資料失敗，狀態碼: {event_odds_response.status_code}")
                print(event_odds_response.text)  # 打印錯誤信息
        
        # 將所有賭盤資料存儲到JSON文件
        with open('event_odds_data.json', 'w', encoding='utf-8') as f:
            json.dump(all_event_odds, f, ensure_ascii=False, indent=4)
        print("賭盤資料已存儲到 event_odds_data.json")
    else:
        print("未找到即將開始的比賽")
