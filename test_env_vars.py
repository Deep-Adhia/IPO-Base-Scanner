#!/usr/bin/env python3
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Get environment variables
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

print(f"BOT_TOKEN: {'Set' if BOT_TOKEN else 'Missing'}")
print(f"CHAT_ID: {CHAT_ID}")
print(f"BOT_TOKEN length: {len(BOT_TOKEN) if BOT_TOKEN else 0}")

if BOT_TOKEN and CHAT_ID:
    print("✅ Both credentials are set")
    
    # Test Telegram API
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    response = requests.post(url, json={
        "chat_id": CHAT_ID, 
        "text": "🧪 Environment test from GitHub Actions", 
        "parse_mode": "HTML"
    }, timeout=10)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("✅ Telegram message sent successfully!")
    else:
        print("❌ Failed to send Telegram message")
else:
    print("❌ Missing credentials!")
