import pandas as pd
df = pd.read_csv('bets_1226_results.csv')
profit = df['Profit'].sum()
wins = len(df[df['Result']=='WIN'])
total = len(df[df['Result'].isin(['WIN','LOSS'])])
win_rate = (wins/total)*100 if total > 0 else 0
print(f"PROFIT: {profit:.2f}")
print(f"WINRATE: {win_rate:.1f}")
