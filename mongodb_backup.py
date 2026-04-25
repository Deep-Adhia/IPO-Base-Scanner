"""
mongodb_backup.py
Export all MongoDB collections to local JSON files as a disaster recovery path.
Run this weekly or before major cutovers.
"""
import os
import json
import logging
from datetime import datetime, timezone
from pymongo import MongoClient
from dotenv import load_dotenv
from bson import json_util

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI")
BACKUP_DIR = "backups"

def run_backup():
    if not MONGO_URI:
        logger.error("MONGO_URI not set in .env")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_backup_path = os.path.join(BACKUP_DIR, timestamp)
    os.makedirs(current_backup_path, exist_ok=True)

    try:
        client = MongoClient(MONGO_URI)
        db = client["ipo_scanner_v2"]
        collections = db.list_collection_names()

        logger.info(f"🚀 Starting backup for {len(collections)} collections...")

        for coll_name in collections:
            if coll_name.startswith("_"): continue # skip internal test collections
            
            logger.info(f"📂 Exporting {coll_name}...")
            cursor = db[coll_name].find({})
            data = list(cursor)
            
            file_path = os.path.join(current_backup_path, f"{coll_name}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                # Use bson.json_util to handle MongoDB-specific types like ObjectId and ISODate
                f.write(json_util.dumps(data, indent=2))
            
            logger.info(f"✅ Saved {len(data)} records to {file_path}")

        logger.info(f"🎉 Backup complete! Path: {current_backup_path}")
        
    except Exception as e:
        logger.error(f"❌ Backup failed: {e}")

if __name__ == "__main__":
    run_backup()
