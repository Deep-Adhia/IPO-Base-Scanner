import pandas as pd
import requests
from datetime import datetime, timedelta
from io import StringIO

def fetch_recent_ipo_symbols(years_back=1):
    """Dynamic IPO symbol fetching with multiple fallback methods"""
    try:
        print(f"🔄 Fetching recent IPO symbols for last {years_back} year(s)...")
        
        # Method 1: Use NSE EQUITY_L.csv with robust validation
        try:
            print("📡 Fetching NSE equity list with validation...")
            url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
            headers = {'User-Agent': 'Mozilla/5.0'}
            session = requests.Session()
            session.headers.update(headers)
            session.get("https://www.nseindia.com", timeout=10)
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            
            df = pd.read_csv(StringIO(resp.text))
            print(f"📊 NSE EQUITY_L returned {len(df)} records")
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
                
                # ROBUST VALIDATION: Multiple layers of filtering
                print(f"📅 Initial data: {len(df)} companies")
                
                # 1. Remove companies with invalid dates
                valid_dates_mask = df[date_col].notna()
                df = df[valid_dates_mask]
                print(f"📅 After removing invalid dates: {len(df)} companies")
                
                # 2. Remove companies with future dates (more than 7 days)
                current_date = datetime.now()
                max_future_date = current_date + timedelta(days=7)
                valid_future_mask = df[date_col] <= max_future_date
                df = df[valid_future_mask]
                print(f"📅 After removing future dates: {len(df)} companies")
                
                # 3. Remove companies with very old dates (before 2020)
                min_date = datetime(2020, 1, 1)
                valid_old_mask = df[date_col] >= min_date
                df = df[valid_old_mask]
                print(f"📅 After removing very old dates: {len(df)} companies")
                
                # 4. Filter for recent IPOs only
                recent_mask = df[date_col] > cutoff
                recent_ipos = df[recent_mask]
                print(f"📅 After recent IPO filter: {len(recent_ipos)} companies")
                
                # 5. Additional validation: Check for suspicious patterns
                # Remove companies with suspicious names or symbols
                suspicious_patterns = [
                    'RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICI', 'SBI', 'BHARTI', 'ITC', 'LT',
                    'RNBDENIMS', 'R&B', 'Denims', 'BANK', 'FINANCE', 'STEEL', 'CEMENT'
                ]
                
                # Create a mask to exclude suspicious companies
                suspicious_mask = recent_ipos[name_col].str.contains('|'.join(suspicious_patterns), case=False, na=False)
                recent_ipos = recent_ipos[~suspicious_mask]
                print(f"📅 After removing suspicious companies: {len(recent_ipos)} companies")
                
                print(f"📅 Final date range: {recent_ipos[date_col].min()} to {recent_ipos[date_col].max()}")
                
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
                
                # Manual blacklist of known old companies that might appear with wrong dates
                blacklisted_symbols = {
                    'RNBDENIMS',  # R&B Denims Ltd. - IPO was in 2014
                    'RELIANCE',   # Reliance Industries - very old
                    'TCS',        # Tata Consultancy Services - very old
                    'INFY',       # Infosys - very old
                    'HDFCBANK',   # HDFC Bank - very old
                    'ICICIBANK',  # ICICI Bank - very old
                    'SBIN',       # State Bank of India - very old
                    'BHARTIARTL', # Bharti Airtel - very old
                    'ITC',        # ITC Limited - very old
                    'LT',         # Larsen & Toubro - very old
                }
                
                # Remove blacklisted symbols
                before_blacklist = len(df_symbols)
                df_symbols = df_symbols[~df_symbols['symbol'].isin(blacklisted_symbols)]
                after_blacklist = len(df_symbols)
                
                if before_blacklist != after_blacklist:
                    print(f"🚫 Removed {before_blacklist - after_blacklist} blacklisted old companies")
                
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