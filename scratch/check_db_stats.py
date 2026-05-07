import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client["ipo_scanner_v2"]

v1_count = db.signals.count_documents({})
v2_count = db.signals_v2.count_documents({})
logs_count = db.logs.count_documents({})

print(f"Signals (v1):    {v1_count}")
print(f"Signals (v2):    {v2_count}")
print(f"Logs:            {logs_count}")

# Check if any listing day signals are in v2
listing_v2 = db.signals_v2.count_documents({"scanner": "listing_day"})
print(f"Listing Day (v2): {listing_v2}")
