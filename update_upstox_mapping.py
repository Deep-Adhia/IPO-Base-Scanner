#!/usr/bin/env python3
"""
update_upstox_mapping.py

Automatically update ipo_upstox_mapping.csv with new IPO symbols.
This script:
1. Reads recent IPOs from recent_ipo_symbols.csv
2. Uses Upstox API to find instrument keys for missing symbols
3. Updates ipo_upstox_mapping.csv automatically
"""

import os
import pandas as pd
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
import logging

# Load environment
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
MAPPING_CSV = "ipo_upstox_mapping.csv"
RECENT_IPO_CSV = "recent_ipo_symbols.csv"
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN')

def get_isin_from_nse(symbol):
    """Get ISIN code from NSE equity list"""
    try:
        url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        session = requests.Session()
        session.headers.update(headers)
        
        # Establish session
        session.get("https://www.nseindia.com", timeout=15)
        resp = session.get(url, timeout=45)
        resp.raise_for_status()
        
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        
        # Find symbol column
        symbol_col = None
        isin_col = None
        
        for col in df.columns:
            col_upper = col.upper()
            if 'SYMBOL' in col_upper:
                symbol_col = col
            elif 'ISIN' in col_upper:
                isin_col = col
        
        if symbol_col and isin_col:
            match = df[df[symbol_col] == symbol]
            if not match.empty:
                isin = match[isin_col].iloc[0]
                return isin
        
        return None
    except Exception as e:
        logger.debug(f"Error getting ISIN from NSE for {symbol}: {e}")
        return None

def search_instrument_key(symbol, listing_date=None):
    """Search for instrument key using Upstox API - requires ISIN code"""
    if not UPSTOX_ACCESS_TOKEN:
        logger.warning("UPSTOX_ACCESS_TOKEN not found - cannot search for instrument keys")
        return None
    
    try:
        headers = {
            'Accept': 'application/json',
            'Api-Version': '2.0',
            'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}'
        }
        
        logger.info(f"Searching for {symbol}...")
        
        # Step 1: Get ISIN code from NSE (same as existing mappings use)
        isin = get_isin_from_nse(symbol)
        
        if not isin:
            logger.warning(f"Could not find ISIN for {symbol} from NSE - skipping")
            return None
        
        # Use ISIN in instrument key format (same as existing mappings: NSE_EQ|INE...)
        instrument_key = f"NSE_EQ|{isin}"
        logger.info(f"Found ISIN for {symbol}: {isin}")
        
        # Step 2: Get company name from NSE data (if available)
        name = symbol  # Default to symbol
        
        try:
            url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            session = requests.Session()
            session.headers.update(headers)
            session.get("https://www.nseindia.com", timeout=15)
            resp = session.get(url, timeout=45)
            resp.raise_for_status()
            
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            
            # Find columns
            symbol_col = None
            name_col = None
            for col in df.columns:
                col_upper = col.upper()
                if 'SYMBOL' in col_upper:
                    symbol_col = col
                elif 'NAME' in col_upper and 'COMPANY' in col_upper:
                    name_col = col
            
            if symbol_col and name_col:
                match = df[df[symbol_col] == symbol]
                if not match.empty:
                    name = match[name_col].iloc[0]
        except:
            pass  # Use symbol as name if we can't fetch
        
        # Return mapping in same format as existing ones (match_type: 'exact')
        logger.info(f"Created mapping for {symbol}: {instrument_key}")
        return {
            'ipo_symbol': symbol,
            'upstox_symbol': symbol,
            'name': name,
            'instrument_key': instrument_key,
            'match_type': 'exact'  # Same as existing mappings
        }
        
    except Exception as e:
        logger.error(f"Error searching for {symbol}: {e}")
        return None

def update_mapping_csv():
    """Update mapping CSV with new IPO symbols"""
    try:
        # Load existing mapping
        if os.path.exists(MAPPING_CSV):
            existing_mapping = pd.read_csv(MAPPING_CSV, encoding='utf-8')
            existing_symbols = set(existing_mapping['ipo_symbol'].tolist())
        else:
            # Create new mapping file
            existing_mapping = pd.DataFrame(columns=[
                'ipo_symbol', 'upstox_symbol', 'name', 'instrument_key', 'match_type'
            ])
            existing_symbols = set()
            existing_mapping.to_csv(MAPPING_CSV, index=False, encoding='utf-8')
            logger.info(f"Created new mapping file: {MAPPING_CSV}")
        
        # Load recent IPOs
        if not os.path.exists(RECENT_IPO_CSV):
            logger.warning(f"{RECENT_IPO_CSV} not found")
            return
        
        recent_ipos = pd.read_csv(RECENT_IPO_CSV, encoding='utf-8')
        new_mappings = []
        updated_count = 0
        
        for _, row in recent_ipos.iterrows():
            symbol = row['symbol']
            
            # Skip if already in mapping
            if symbol in existing_symbols:
                continue
            
            logger.info(f"Processing {symbol}...")
            
            # Get listing date if available
            listing_date = None
            if 'listing_date' in row and pd.notna(row['listing_date']):
                listing_date = row['listing_date']
            
            # Try to find instrument key
            mapping = search_instrument_key(symbol, listing_date)
            
            if mapping:
                new_mappings.append(mapping)
                updated_count += 1
                logger.info(f"✅ Added mapping for {symbol}")
            else:
                logger.warning(f"⚠️ Could not find mapping for {symbol} - skipping")
            
            # Rate limiting
            time.sleep(0.2)
        
        # Update mapping CSV
        if new_mappings:
            new_df = pd.DataFrame(new_mappings)
            updated_mapping = pd.concat([existing_mapping, new_df], ignore_index=True)
            updated_mapping.to_csv(MAPPING_CSV, index=False, encoding='utf-8')
            logger.info(f"✅ Updated {MAPPING_CSV} with {updated_count} new mappings")
        else:
            logger.info("No new mappings to add")
        
        return updated_count
        
    except Exception as e:
        logger.error(f"Error updating mapping CSV: {e}")
        return 0

def main():
    """Main function"""
    try:
        print("="*60)
        print("Upstox Mapping Updater")
        print("="*60)
    except:
        print("="*60)
        print("Upstox Mapping Updater")
        print("="*60)
    
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print()
    
    if not UPSTOX_ACCESS_TOKEN:
        print("WARNING: UPSTOX_ACCESS_TOKEN not found!")
        print("   Mapping update will be limited.")
        print("   Set UPSTOX_ACCESS_TOKEN in .env file for full functionality.")
        print()
    
    updated = update_mapping_csv()
    
    print()
    print("="*60)
    print(f"Update Complete: {updated} new mappings added")
    print("="*60)

if __name__ == "__main__":
    main()

