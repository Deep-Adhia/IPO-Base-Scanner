
from fetch import fetch_recent_ipo_symbols
print("Running fetch_recent_ipo_symbols...")
df = fetch_recent_ipo_symbols(years_back=1)
if df is not None:
    print(f"Update complete. Total rows: {len(df)}")
else:
    print("Update failed.")
