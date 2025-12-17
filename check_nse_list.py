
import pandas as pd
import requests
from io import StringIO

def check_nse_list():
    try:
        url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        session = requests.Session()
        session.headers.update(headers)
        
        print("Fetching EQUITY_L.csv...")
        session.get("https://www.nseindia.com", timeout=15)
        resp = session.get(url, timeout=45)
        resp.raise_for_status()
        
        df = pd.read_csv(StringIO(resp.text))
        print(f"Total records: {len(df)}")
        print("Columns:", df.columns.tolist())
        
        terms = ["MEESHO", "AEQ", "AQU", "FASHNEAR"]
        print(f"Searching for terms: {terms}")
        
        found = False
        for col in df.columns:
            # Check symbol and name columns
            if 'SYMBOL' in col.upper() or 'NAME' in col.upper():
                for term in terms:
                    matches = df[df[col].astype(str).str.contains(term, case=False, na=False)]
                    if not matches.empty:
                        print(f"\n--- Matches for '{term}' in column '{col}' ---")
                        print(matches.to_string())
                        found = True
                        
        if not found:
            print("\nNo matches found.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_nse_list()
