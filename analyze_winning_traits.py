import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

def analyze_traits():
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client['ipo_scanner_v2']
    signals = list(db.signals_v2.find({"scanner_version": "backfill-1.0"}))
    
    if not signals:
        print("No backfilled signals found for analysis.")
        return

    # Convert to DataFrame for easy analysis
    data = []
    for s in signals:
        outcome = s.get('outcome', {})
        market = s.get('market_context', {})
        features = s.get('features', {})
        
        # Flatten the record
        row = {
            'symbol': s.get('symbol'),
            'pnl_pct': outcome.get('max_runup_pct', 0), # Use Max Runup for "Potential" analysis
            'max_drawdown': outcome.get('max_drawdown_pct', 0),
            'market_state': market.get('market_state'),
            'nifty_slope': market.get('nifty_trend_slope'),
            'nifty_dist_ma20': market.get('nifty_20ma_dist'),
            'scanner': s.get('scanner'),
            'sector': s.get('sector', 'Unknown'),
            'industry': s.get('industry', 'Unknown'),
            'status': outcome.get('status', 'UNKNOWN')
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    
    print("-" * 60)
    print("           ALFA TRAIT DISCOVERY REPORT (V2)")
    print("-" * 60)
    print(f"Total Trades Analyzed: {len(df)}")
    
    # Winners vs Losers based on Synthetic Outcome
    winners = df[df['status'] == 'SUCCESS']
    super_winners = df[df['pnl_pct'] >= 40]
    losers = df[df['status'].isin(['STOPPED_OUT', 'FAILED'])]
    
    print(f"\nWinners (Success): {len(winners)}")
    print(f"SUPER WINNERS (>40%): {len(super_winners)}")
    print(f"Losers (Stopped):  {len(losers)}")
    print(f"Scratch/Pending:   {len(df) - len(winners) - len(losers)}")
    
    if len(super_winners) > 0:
        print("\n[SUPER WINNER TRAITS]")
        print(f"Avg Nifty Slope:      {super_winners['nifty_slope'].mean():.3f}")
        print(f"Avg Dist from 20MA:   {super_winners['nifty_dist_ma20'].mean():.2f}%")
        print("Market State Dist:")
        print(super_winners['market_state'].value_counts())
        print("Sector Distribution:")
        print(super_winners['sector'].value_counts())
        print("Symbols:", ", ".join(super_winners['symbol'].tolist()))

    if len(winners) > 0:
        print("\n[WINNING TRAITS]")
        print(f"Avg Nifty Slope:      {winners['nifty_slope'].mean():.3f}")
        print(f"Avg Dist from 20MA:   {winners['nifty_dist_ma20'].mean():.2f}%")
        print("Market State Dist:")
        print(winners['market_state'].value_counts())
        
    if len(losers) > 0:
        print("\n[LOSING TRAITS]")
        print(f"Avg Nifty Slope:      {losers['nifty_slope'].mean():.3f}")
        print(f"Avg Dist from 20MA:   {losers['nifty_dist_ma20'].mean():.2f}%")
        print("Market State Dist:")
        print(losers['market_state'].value_counts())

    print("\n" + "-" * 60)
    print("Scanner Distribution of Winners:")
    if len(winners) > 0:
        print(winners['scanner'].value_counts())
    
    # Summary Insight
    print("\n[ALPHA INSIGHT]")
    if len(winners) > 0 and len(losers) > 0:
        slope_diff = winners['nifty_slope'].mean() - losers['nifty_slope'].mean()
        if slope_diff > 0.1:
            print(f"👉 Winning trades correlate strongly with a HIGHER Nifty Slope ({winners['nifty_slope'].mean():.3f} vs {losers['nifty_slope'].mean():.3f}).")
        
        bull_winners = len(winners[winners['market_state'] == 'BULL_CONFIRMED'])
        bear_winners = len(winners[winners['market_state'] == 'BEAR_CONFIRMED'])
        if bull_winners > bear_winners:
            print(f"👉 Setup success is {bull_winners/len(winners)*100:.1f}% more likely in BULL_CONFIRMED states.")

if __name__ == "__main__":
    analyze_traits()
