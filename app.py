
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page Config
st.set_page_config(page_title="NBA AI Bettor (2025)", layout="wide")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("predictions_2025.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# Sidebar
st.sidebar.title("🏀 NBA AI Predictor")
page = st.sidebar.radio("Navigation", ["Player Review", "Betting Room (Simulation)"])

# ==========================================
# Page 1: Player Review
# ==========================================
if page == "Player Review":
    st.title("Player Performance Review (2025-26)")
    
    # Filters
    teams = sorted(df['Team'].unique())
    selected_team = st.sidebar.selectbox("Select Team", teams)
    
    players = sorted(df[df['Team'] == selected_team]['Player_Name'].unique())
    selected_player = st.sidebar.selectbox("Select Player", players)
    
    # Data for Player
    player_df = df[df['Player_Name'] == selected_player].sort_values(by='Date')
    
    # Metrics
    targets = ['PTS', 'AST', 'REB']
    
    for target in targets:
        st.subheader(f"{target} Performance")
        
        # Chart
        fig = go.Figure()
        
        # Actual
        fig.add_trace(go.Scatter(x=player_df['Date'], y=player_df[f'Actual_{target}'], mode='lines+markers', name='Actual', line=dict(color='black', width=2)))
        
        # House Line
        fig.add_trace(go.Scatter(x=player_df['Date'], y=player_df[f'Line_{target}'], mode='lines', name='House Line', line=dict(color='orange', dash='dot')))
        
        # Prediction (with Confidence Interval if enabled)
        fig.add_trace(go.Scatter(x=player_df['Date'], y=player_df[f'Pred_{target}'], mode='lines', name='AI Prediction', line=dict(color='blue', width=2)))
        
        # Highlight Bets
        # Identify dates where we bet
        bet_indices = player_df[player_df[f'Bet_{target}'] != 'PASS'].index
        bet_dates = player_df.loc[bet_indices, 'Date']
        bet_vals = player_df.loc[bet_indices, f'Pred_{target}']
        
        fig.add_trace(go.Scatter(x=bet_dates, y=bet_vals, mode='markers', name='Bet Signal', marker=dict(color='red', size=10, symbol='star')))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # DataFrame
        cols = ['Date', 'Opponent', f'Actual_{target}', f'Line_{target}', f'Pred_{target}', f'ConfPct_{target}', f'Bet_{target}']
        st.dataframe(player_df[cols].style.applymap(lambda x: 'background-color: #ffcccc' if 'BET' in str(x) else '', subset=[f'Bet_{target}']))

# ==========================================
# Page 2: Betting Room
# ==========================================
elif page == "Betting Room (Simulation)":
    st.title("🎰 Daily Betting Simulation")
    
    # Date Picker
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    selected_date = st.sidebar.date_input("Select Date", max_date, min_value=min_date, max_value=max_date)
    selected_date = pd.to_datetime(selected_date)
    
    # Filter Data
    day_df = df[df['Date'] == selected_date]
    
    if len(day_df) == 0:
        st.warning("No games found for this date.")
    else:
        st.write(f"### Games for {selected_date.strftime('%Y-%m-%d')}")
        
        # Summary metrics
        total_games = len(day_df)
        st.metric("Total Players Tracked", total_games)
        
        # Columns for Targets
        t1, t2, t3 = st.tabs(["PTS Bets", "AST Bets", "REB Bets"])
        
        targets = ['PTS', 'AST', 'REB']
        tabs = [t1, t2, t3]
        
        for i, target in enumerate(targets):
            with tabs[i]:
                # Filter for Active Bets
                active_bets = day_df[day_df[f'Bet_{target}'] != 'PASS'].copy()
                
                if len(active_bets) == 0:
                    st.info("No AI Bets for this category today.")
                else:
                    st.success(f"AI suggests {len(active_bets)} bets.")
                    
                    # Calculate Daily PnL (Simulation)
                    # Win if (Bet Over AND Actual > Line) OR (Bet Under And Actual < Line)
                    # Assume Odds -110 (1.91)
                    wins = 0
                    ps = 0
                    
                    for idx, row in active_bets.iterrows():
                        bet = row[f'Bet_{target}'] # "BET Over"
                        actual = row[f'Actual_{target}']
                        line = row[f'Line_{target}']
                        
                        result = "PUSH"
                        profit = 0
                        used_odds = 0
                        
                        # Variable Odds Logic
                        if "Over" in bet:
                            used_odds = row[f'OddsOver_{target}']
                            if actual > line: 
                                result = "WIN"
                                wins += 1
                                profit = used_odds - 1.0 # Profit = (Odds - 1)
                            elif actual < line: 
                                result = "LOSS"
                                profit = -1.0
                        elif "Under" in bet:
                            used_odds = row[f'OddsUnder_{target}']
                            if actual < line: 
                                result = "WIN"
                                wins += 1
                                profit = used_odds - 1.0
                            elif actual > line: 
                                result = "LOSS"
                                profit = -1.0
                                
                        active_bets.loc[idx, 'Result'] = result
                        active_bets.loc[idx, 'Profit'] = profit
                        active_bets.loc[idx, 'Odds'] = used_odds
                        
                    # Display PnL
                    total_profit = active_bets['Profit'].sum()
                    roi = (total_profit / len(active_bets)) * 100
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Win Rate", f"{wins}/{len(active_bets)} ({wins/len(active_bets):.1%})")
                    c2.metric("Total Profit", f"{total_profit:.2f}u")
                    c3.metric("ROI", f"{roi:.1f}%")
                    
                    # Display Table
                    display_cols = ['Player_Name', 'Team', 'Opponent', f'Line_{target}', 'Odds', f'Pred_{target}', f'ConfPct_{target}', f'Bet_{target}', f'Actual_{target}', 'Result']
                    
                    st.dataframe(
                        active_bets[display_cols].style.applymap(
                            lambda x: 'color: green; font-weight: bold' if x == 'WIN' else ('color: red; font-weight: bold' if x == 'LOSS' else ''), 
                            subset=['Result']
                        )
                    )
