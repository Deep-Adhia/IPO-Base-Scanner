import pandas as pd
import requests
from datetime import datetime, timedelta
from io import StringIO

def fetch_recent_ipo_symbols(years_back=1):
    """Dynamic IPO symbol fetching with multiple fallback methods"""
    try:
        print(f"🔄 Fetching recent IPO symbols for last {years_back} year(s)...")
        
        # Method 1: Try NSE API with retry
        for attempt in range(3):
            try:
                print(f"📡 Fetching NSE equity list... (Attempt {attempt + 1}/3)")
                url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                session = requests.Session()
                session.headers.update(headers)
                
                # Establish session first
                session.get("https://www.nseindia.com", timeout=15)
                resp = session.get(url, timeout=45)
                resp.raise_for_status()
                
                print("✅ NSE API connection successful")
                
                df = pd.read_csv(StringIO(resp.text))
                print(f"📊 NSE EQUITY_L returned {len(df)} records")
                
                # Find the right columns
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
                    
                    # Filter for recent IPOs
                    recent_mask = df[date_col] > cutoff
                    recent_ipos = df[recent_mask]
                    
                    # Remove suspicious companies
                    suspicious_patterns = ['RNBDENIMS'] # Kept one for example, but removed major groups to allow subsidiaries
                    if name_col:
                        suspicious_mask = recent_ipos[name_col].str.contains('|'.join(suspicious_patterns), case=False, na=False)
                        recent_ipos = recent_ipos[~suspicious_mask]
                        
                    # Remove RE and SME shares
                    if symbol_col:
                        re_sme_mask = recent_ipos[symbol_col].str.contains('-RE|-SM|RE1', case=False, na=False)
                        recent_ipos = recent_ipos[~re_sme_mask]
                    
                    symbols = recent_ipos[symbol_col].tolist()
                    companies = recent_ipos[name_col].tolist() if name_col else symbols
                    dates = recent_ipos[date_col].dt.strftime('%Y-%m-%d').tolist()
                    
                    print(f"✅ NSE API: Found {len(symbols)} recent IPOs")
                    
                    df_symbols = pd.DataFrame({
                        'symbol': symbols,
                        'company': companies,
                        'listing_date': dates
                    })
                    
                    # Save files
                    df_symbols.to_csv("recent_ipo_symbols.csv", index=False, encoding='utf-8')
                    
                    with open("recent_ipo_symbols.txt", "w", encoding='utf-8') as f:
                        for sym in symbols:
                            f.write(f"{sym}\n")
                    
                    print(f"💾 Saved to: recent_ipo_symbols.csv")

                    # MongoDB dual-write: upsert discovered IPOs
                    try:
                        from db import upsert_ipo, ensure_indexes
                        ensure_indexes()
                        for _, row in df_symbols.iterrows():
                            upsert_ipo(
                                symbol=row['symbol'],
                                listing_date=row['listing_date'],
                                name=row['company']
                            )
                        print(f"✅ [MongoDB] Upserted {len(df_symbols)} IPO records")
                    except Exception as db_e:
                        print(f"⚠️ [MongoDB] IPO write FAILED (CSV write succeeded): {db_e}")
                        try:
                            from db import db_metrics
                            db_metrics["failures"] = db_metrics.get("failures", 0) + 1
                        except Exception:
                            pass

                    return df_symbols
                else:
                    print("⚠️ NSE API: Could not find required columns")
                    raise Exception("Column mapping failed")
                    
            except Exception as e:
                print(f"⚠️ NSE API attempt {attempt + 1} failed: {e}")
                if attempt == 2:  # Last attempt
                    print("❌ All NSE API attempts failed")
                    break
                else:
                    print("🔄 Retrying in 5 seconds...")
                    import time
                    time.sleep(5)
        
        # Method 2: Fallback to existing CSV
        print("🔄 Falling back to existing CSV...")
        try:
            if os.path.exists("recent_ipo_symbols.csv"):
                print("📁 Using existing recent_ipo_symbols.csv as fallback")
                df_symbols = pd.read_csv("recent_ipo_symbols.csv")
                if not df_symbols.empty:
                    print(f"📊 CSV fallback: {len(df_symbols)} symbols from existing file")
                    return df_symbols
            print("❌ No valid existing CSV file found or file is empty")
        except Exception as csv_error:
            print(f"⚠️ CSV fallback failed: {csv_error}")
            print("🔄 Creating minimal fallback data...")
            
            # Method 3: Create minimal fallback
            fallback_symbols = [
                'SWIGGY', 'BLACKBUCK', 'STALLION', 'BHARATSE', 
                'NATCAPSUQ', 'MOSCHIP', 'TRAVELFOOD', 'OCCLLTD', 'GARUDA',
                'CEWATER', 'RACLGEAR', 'ORCHASP', 'OSWALPUMPS', 'IGIL',
                'VIKRAN', 'AFCONS', 'MOBIKWIK', 'MASTERTR', 'JAINREC'
            ]
            
            df_symbols = pd.DataFrame({
                'symbol': fallback_symbols,
                'company': [f"{sym} Ltd" for sym in fallback_symbols],
                'listing_date': [datetime.now().strftime('%Y-%m-%d')] * len(fallback_symbols)
            })
            
            df_symbols.to_csv("recent_ipo_symbols.csv", index=False)
            
            with open("recent_ipo_symbols.txt", "w", encoding='utf-8') as f:
                for sym in fallback_symbols:
                    f.write(f"{sym}\n")
            
            print(f"💾 Created fallback with {len(fallback_symbols)} symbols")
            return df_symbols
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
