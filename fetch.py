import pandas as pd
import requests
from datetime import datetime, timedelta
from io import StringIO

def fetch_recent_ipo_symbols(years_back=1):
    """Dynamic IPO symbol fetching with multiple fallback methods"""
    try:
        print(f"🔄 Fetching recent IPO symbols for last {years_back} year(s)...")
        
        # Method 1: Try NSE API
        try:
            print("📡 Trying NSE API...")
            url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
            headers = {'User-Agent': 'Mozilla/5.0'}
            session = requests.Session()
            session.headers.update(headers)
            session.get("https://www.nseindia.com", timeout=10)
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            
            df = pd.read_csv(StringIO(resp.text))
            print(f"📊 NSE API returned {len(df)} records")
            print(f"📋 Columns: {list(df.columns)}")
            
            # Handle different column names
            date_col = None
            symbol_col = None
            name_col = None
            
            for col in df.columns:
                col_upper = col.upper()
                if 'DATE' in col_upper and 'LISTING' in col_upper:
                    date_col = col
                elif 'SYMBOL' in col_upper:
                    symbol_col = col
                elif 'NAME' in col_upper and 'COMPANY' in col_upper:
                    name_col = col
            
            if date_col and symbol_col:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                cutoff = datetime.now() - timedelta(days=365 * years_back)
                recent_ipos = df[df[date_col] > cutoff]
                
                symbols = recent_ipos[symbol_col].tolist()
                companies = recent_ipos[name_col].tolist() if name_col else symbols
                dates = recent_ipos[date_col].dt.strftime('%Y-%m-%d').tolist()
                
                print(f"✅ NSE API: Found {len(symbols)} recent IPOs")
                
                # Create DataFrame
                df_symbols = pd.DataFrame({
                    'symbol': symbols,
                    'company': companies,
                    'listing_date': dates
                })
                
                # Save files (overwrite existing)
                df_symbols.to_csv("recent_ipo_symbols.csv", index=False)
                
                with open("recent_ipo_symbols.txt", "w") as f:
                    for sym in symbols:
                        f.write(f"{sym}\n")
                
                print(f"💾 Saved to: recent_ipo_symbols.csv")
                return df_symbols
            else:
                print("⚠️ NSE API: Could not find required columns")
                raise Exception("Column mapping failed")
                
        except Exception as e:
            print(f"⚠️ NSE API failed: {e}")
            print("🔄 Falling back to CSV method...")
            
            # Method 2: Fallback to CSV
            ipo_df = pd.read_csv("IPO-PastIssue-01-01-2024-to-23-09-2025.csv")
            ipo_df['DATE OF LISTING'] = pd.to_datetime(ipo_df['DATE OF LISTING'], format="%d-%b-%Y", errors='coerce')
            ipo_df = ipo_df[ipo_df['SECURITY TYPE'].str.contains("EQ", case=False)]
            
            cutoff_date = datetime.now() - timedelta(days=365 * years_back)
            recent = ipo_df[ipo_df['DATE OF LISTING'] >= cutoff_date]
            
            print(f"📊 CSV fallback: {len(recent)} recent IPOs from {len(ipo_df)} total")
            
            if len(recent) > 0:
                symbols = recent['Symbol'].str.strip().tolist()
                companies = recent['COMPANY NAME'].tolist()
                dates = recent['DATE OF LISTING'].dt.strftime('%Y-%m-%d').tolist()
                
                df_symbols = pd.DataFrame({
                    'symbol': symbols,
                    'company': companies,
                    'listing_date': dates
                })
                
                # Save files (overwrite existing)
                df_symbols.to_csv("recent_ipo_symbols.csv", index=False)
                
                with open("recent_ipo_symbols.txt", "w") as f:
                    for sym in symbols:
                        f.write(f"{sym}\n")
                
                print(f"💾 Saved to: recent_ipo_symbols.csv")
                return df_symbols
            else:
                print("❌ No recent IPOs found")
                return None
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    symbols_df = fetch_recent_ipo_symbols(years_back=1)
    if symbols_df is not None:
        print(f"🎯 Successfully fetched {len(symbols_df)} recent IPO symbols")
    else:
        print("❌ Failed to fetch IPO symbols")