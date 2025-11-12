# 🚀 IPO Breakout Scanner

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Automated](https://img.shields.io/badge/automation-GitHub%20Actions-green.svg)](https://github.com/features/actions)

An intelligent, automated IPO breakout detection system for Indian markets using systematic signal generation.

## 🎯 **System Overview**

This scanner implements multiple strategies to identify IPO breakout opportunities based on price action and volume analysis.

### **Key Features**
- 🔍 **Automated IPO Detection** - Daily scanning of newly listed stocks
- 📱 **Telegram Alerts** - Real-time signal notifications
- 📊 **Multiple Strategies** - Consolidation-based and listing day breakout detection
- ⚡ **Manual Execution** - Generates signals for manual review (compliance-friendly)
- 📈 **Simple Tracking** - Clean CSV-based portfolio management

## 🛠️ **Quick Setup**

### **1. Create Your Repository**
```bash
git clone https://github.com/yourusername/ipo-breakout-scanner.git
cd ipo-breakout-scanner
```

### **2. Set Up Telegram Bot (Optional)**
1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Send `/newbot` and follow instructions
3. Save your bot token
4. Start a chat with your bot and send any message
5. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
6. Find your chat ID in the response

**Note**: Scanner works without Telegram - notifications will appear in logs instead

### **3. Configure Environment Variables (Optional)**

Create a `.env` file in the project root (copy from `.env.example`):

```bash
# Copy example file
cp .env.example .env
# Edit .env with your values
```

**Required for local development:**
- None (system works without any tokens, uses NSE fallback)

**Optional (for better performance):**
- `UPSTOX_ACCESS_TOKEN` - Upstox API token for faster data fetching
  - Get from: https://account.upstox.com/developer/apps
  - Without this: System uses NSE (jugaad-data) which is slower but works fine
- `TELEGRAM_BOT_TOKEN` - Telegram bot token for alerts
- `TELEGRAM_CHAT_ID` - Your Telegram chat ID

**For GitHub Actions:**
Go to **Settings > Secrets and variables > Actions** and add:

| Secret Name | Description | Example | Required |
|-------------|-------------|---------|----------|
| `TELEGRAM_BOT_TOKEN` | Your bot token from BotFather | `1234567890:ABCdef...` | Optional |
| `TELEGRAM_CHAT_ID` | Your Telegram chat ID | `987654321` | Optional |
| `UPSTOX_ACCESS_TOKEN` | Upstox API access token | `your_token_here` | Optional |
| `IPO_YEARS_BACK` | Years of IPO data to scan | `2` | Optional |
| `LOG_LEVEL` | Logging verbosity | `INFO` | Optional |

### **4. Deploy & Activate**
```bash
git add .
git commit -m "🚀 Deploy IPO Scanner"
git push origin main
```

The scanners will start automatically and run:

**Active Scanners:**
- **Consolidation-Based Scanner** - Daily scan at 2:15 PM IST (Weekdays) ✅
  - Detects consolidation patterns and breakouts
  - Signal Type: "Consolidation-Based Breakout"
- **Listing Day Breakout Scanner** - Hourly during market hours (9:15 AM - 3:30 PM IST) ✅
  - Checks for breakouts above listing day high
  - Signal Type: "Listing Day Breakout"
- **Watchlist Scanner** - Hourly during market hours (if watchlist has active symbols) ✅
  - Intraday breakout detection for watchlist stocks

## 🎯 **How It Works**

### **Strategy 1: Consolidation-Based Detection Algorithm**
- **Pattern Detection:** Identifies consolidation patterns with price and volume analysis
- **Signal Type:** "Consolidation-Based Breakout" (marked in alerts and CSV)
- **Grading System:**
  - **Grade A+**: Perfect setup, highest allocation
  - **Grade B**: Good setup, moderate allocation  
  - **Grade C**: Acceptable setup, smaller allocation
- **Risk Management:**
  - **Stop losses** with trailing functionality
  - **Position sizing by grade** for risk management
  - **Quick loser rejection** (5-day, -5% rule)
- **Conflict Prevention:** Checks for existing active positions before creating new signals

### **Strategy 2: Listing Day High Breakout**
- **Core Principle:** IPO stocks breaking listing day high with volume show strong momentum
- **Signal Type:** "Listing Day Breakout" (marked in alerts and CSV)
- **Entry:** When price breaks listing day high + volume spike (1.5x average)
- **Stop Loss:** Listing day low (critical support)
- **Target:** Listing day high + 50% of listing day range
- **Advantage:** Simple, clear entry/exit rules based on key price levels
- **No complex indicators** - Just price and volume confirmation
- **Conflict Prevention:** Checks for existing active positions before creating new signals

### **Strategy 3: Intraday Watchlist Monitoring**
- **Purpose:** Real-time monitoring of specific symbols
- **Data:** 5-minute candles from Upstox API
- **Detection:** Intraday breakouts with volume confirmation
- **Use Case:** Quick scalping opportunities on watchlist stocks

## 📱 **Sample Telegram Alerts**

### **Consolidation-Based Breakout Signal:**
```
🎯 CONSOLIDATION BREAKOUT SIGNAL

📊 Symbol: KAYNES
📋 Signal Type: Consolidation-Based Breakout
⭐ Grade: A+
💰 Entry: ₹1,247.50
🛑 Stop Loss: ₹1,120.15
📈 Target: ₹1,450.00
```

### **Listing Day Breakout Signal:**
```
🎯 LISTING DAY HIGH BREAKOUT!

📊 Symbol: ABCOTS
📋 Signal Type: Listing Day Breakout
📅 Listing Date: 2024-09-24
💰 Entry: ₹1,500.00
🛑 Stop Loss: ₹1,200.00 (Listing Day Low)
📈 Target: ₹1,800.00

📅 Date: 2025-09-27 20:30

Manual review recommended
```

## 📊 **File Structure & Flow**

### **Core Python Scripts**

```
ipo-breakout-scanner/
├── streamlined-ipo-scanner.py      # Main consolidation-based scanner
├── listing_day_breakout_scanner.py # Listing day high breakout scanner
├── hourly_breakout_scanner.py      # Intraday breakout scanner for watchlist
├── fetch.py                        # IPO symbol fetcher from NSE
├── utils.py                        # Upstox API utility functions
└── requirements.txt                # Python dependencies
```

### **Data Files (Auto-generated)**

```
├── recent_ipo_symbols.csv          # List of recent IPOs with listing dates
├── ipo_listing_data.csv            # Listing day high/low for each IPO
├── ipo_signals.csv                 # All generated signals
├── ipo_positions.csv               # Active and closed positions
├── ipo_upstox_mapping.csv          # IPO symbol to Upstox instrument mapping
└── watchlist.csv                   # Symbols to monitor for intraday breakouts
```

### **Automation Workflows**

```
.github/workflows/
├── ipo-scanner.yml                 # Daily consolidation-based scanning
├── listing-day-breakout.yml        # Hourly listing day breakout checks
└── watchlist-hourly-scanner.yml    # Hourly intraday breakout checks
```

## 🔄 **System Flow & Strategies**

### **Strategy 1: Consolidation-Based Breakout Scanner** 
*(Main Scanner - `streamlined-ipo-scanner.py`)*
**Status:** ✅ ACTIVE - Runs daily at 2:15 PM IST (Weekdays)

**Flow:**
1. **IPO Discovery** → `fetch.py` fetches recent IPOs from NSE
2. **Data Collection** → Fetches historical price data (Upstox API or NSE fallback)
3. **Pattern Detection** → Identifies consolidation patterns with:
   - Price consolidation analysis
   - Volume analysis
   - Breakout confirmation
4. **Signal Generation** → Creates signals with grades (A+, B, C)
5. **Position Tracking** → Monitors active positions with trailing stops
6. **Alerts** → Sends Telegram notifications

**Entry Logic:**
- Detects breakout from consolidation base
- Entry: Next day opening price after breakout
- Stop Loss: Grade-based (5-15% below entry)
- Target: Based on consolidation range and grade

**Automation:** Runs daily via `ipo-scanner.yml` workflow

---

### **Strategy 2: Listing Day High Breakout Scanner**
*(`listing_day_breakout_scanner.py`)*

**Flow:**
1. **Listing Data Collection** → Fetches and stores listing day high/low for each IPO
2. **Tracking** → Stores in `ipo_listing_data.csv`:
   - Listing day high
   - Listing day low
   - Listing day close
   - Listing day volume
3. **Breakout Detection** → Checks if current price breaks listing day high
4. **Volume Confirmation** → Requires 1.5x average volume spike
5. **Signal Generation** → Creates entry signal when breakout confirmed

**Entry Logic:**
- **Entry:** Current price when breaks listing day high with volume
- **Stop Loss:** Listing day low (critical support level)
- **Target:** Listing day high + 50% of listing day range

**Key Principle:** IPO stocks that break their listing day high with volume often continue upward momentum.

**Automation:** Runs hourly during market hours via `listing-day-breakout.yml` workflow

---

### **Strategy 3: Intraday Watchlist Scanner**
*(`hourly_breakout_scanner.py`)*

**Flow:**
1. **Watchlist Loading** → Reads symbols from `watchlist.csv` (status=ACTIVE)
2. **Intraday Data** → Fetches 5-minute candles from Upstox API
3. **Real-time Detection** → Monitors for intraday breakouts
4. **Breakout Criteria:**
   - Price breaks recent high
   - Volume spike (1.5x average)
   - RSI > 60 (momentum confirmation)
5. **Immediate Alerts** → Sends Telegram alerts for quick action

**Use Case:** Monitor specific symbols for intraday trading opportunities

**Automation:** Runs hourly during market hours via `watchlist-hourly-scanner.yml` workflow (only if watchlist has active symbols)

---

## 📋 **Data Flow Diagram**

```
┌─────────────────────────────────────────────────────────────┐
│                    IPO Discovery Layer                       │
│  fetch.py → recent_ipo_symbols.csv                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼──────────┐      ┌──────────▼──────────────┐
│ Strategy 1:      │      │ Strategy 2:             │
│ Consolidation    │      │ Listing Day Breakout    │
│ Scanner          │      │ Scanner                 │
│                  │      │                         │
│ • Pattern detect │      │ • Track listing high/low│
│ • Grade signals  │      │ • Breakout detection    │
│ • Daily scan     │      │ • Hourly scan           │
└───────┬──────────┘      └──────────┬──────────────┘
        │                             │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │   Signal & Position Storage │
        │  • ipo_signals.csv          │
        │  • ipo_positions.csv        │
        │  • ipo_listing_data.csv     │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │      Telegram Alerts        │
        │  • Entry signals            │
        │  • Exit alerts              │
        │  • Position updates         │
        └─────────────────────────────┘
```

## 🎯 **Key Files Explained**

### **Core Scripts**

| File | Purpose | When It Runs |
|------|---------|--------------|
| `streamlined-ipo-scanner.py` | Main consolidation-based scanner | Daily (8:45 AM IST) |
| `listing_day_breakout_scanner.py` | Listing day high breakout detection | Hourly (9:15 AM - 3:30 PM IST) |
| `hourly_breakout_scanner.py` | Intraday breakout for watchlist | Hourly (if watchlist active) |
| `fetch.py` | Fetches IPO symbols from NSE | On demand / when scanner runs |

### **Data Files**

| File | Purpose | Auto-updated |
|------|---------|--------------|
| `recent_ipo_symbols.csv` | List of recent IPOs | Yes (via fetch.py) |
| `ipo_listing_data.csv` | Listing day high/low for each IPO | Yes (via listing scanner) |
| `ipo_signals.csv` | All generated trading signals | Yes (all scanners) |
| `ipo_positions.csv` | Active and closed positions | Yes (position updates) |
| `watchlist.csv` | Symbols to monitor intraday | Manual |
| `ipo_upstox_mapping.csv` | Symbol to Upstox instrument mapping | Manual/External |

### **Workflow Files**

| File | Schedule | Purpose | Status |
|------|----------|---------|--------|
| `ipo-scanner.yml` | Daily 2:15 PM IST (Weekdays) | Consolidation-based scanning | ✅ ACTIVE |
| `listing-day-breakout.yml` | Hourly 9:15 AM - 3:30 PM IST | Listing day breakout checks | ✅ ACTIVE |
| `watchlist-hourly-scanner.yml` | Hourly 9:15 AM - 3:30 PM IST | Watchlist intraday scanning | ✅ ACTIVE |

## 🚀 **Quick Start Guide**

### **1. Initial Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create .env file with:
# TELEGRAM_BOT_TOKEN=your_token
# TELEGRAM_CHAT_ID=your_chat_id
# UPSTOX_ACCESS_TOKEN=your_token (optional)
```

### **2. Run Scanners Manually**

```bash
# Consolidation-based scanner (main)
python streamlined-ipo-scanner.py scan

# Listing day breakout scanner
python listing_day_breakout_scanner.py

# Watchlist intraday scanner
python hourly_breakout_scanner.py
```

### **3. Add Symbols to Watchlist**

Edit `watchlist.csv`:
```csv
symbol,added_date,notes,status
NATCAPSUQ,2025-11-12,IPO breakout candidate,ACTIVE
SUPREME,2025-11-12,High volume pattern,ACTIVE
```

### **4. Monitor Results**

- **Signals:** Check `ipo_signals.csv`
- **Positions:** Check `ipo_positions.csv`
- **Listing Data:** Check `ipo_listing_data.csv`
- **Alerts:** Telegram notifications (if configured)

## 🔧 **Local Development**

### **Setup Local Environment**
```bash
# Clone repository
git clone <your-repo-url>
cd ipo-breakout-scanner

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.template .env
# Edit .env with your credentials

# Run scanner manually
python streamlined-ipo-scanner.py scan
```

### **Manual Commands**
```bash
# Daily IPO scan
python streamlined-ipo-scanner.py scan

# Weekly summary
python streamlined-ipo-scanner.py weekly_summary

# Monthly review
python streamlined-ipo-scanner.py monthly_review
```

## 📈 **Trading Approach**

### **Signal Generation**
- **Automated detection** of IPO breakout patterns
- **Quality filtering** through multi-indicator confluence
- **Grade-based prioritization** for optimal allocation

### **Manual Execution** (Regulatory Compliance)
- **Signals for review** - not automated trading
- **Human oversight** for final entry decisions
- **Flexible position sizing** based on market conditions

### **Portfolio Management**
- **Real-time tracking** via CSV files
- **Position monitoring** with current prices and P&L
- **Risk monitoring** with stop-loss alerts

## ⚠️ **Important Notes**

### **Regulatory Compliance**
- **Manual execution only** - system generates signals for review
- **No automated trading** - you control all buy/sell decisions
- **Educational/analytical tool** - not investment advice

### **Risk Disclaimer**
- **Past performance doesn't guarantee future results**
- **Trading involves risk** - never risk more than you can afford to lose
- **Systematic approach reduces but doesn't eliminate risk**

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **NSE** for providing market data access
- **jugaad-data** library for data fetching
- **Telegram Bot API** for notification system
- **GitHub Actions** for free automation platform

## 📧 **Support**

For questions, issues, or suggestions:
- 🐛 [Open an Issue](https://github.com/yourusername/ipo-breakout-scanner/issues)
- 💬 [Start a Discussion](https://github.com/yourusername/ipo-breakout-scanner/discussions)

---

**⚡ Ready to systematize your IPO trading? Deploy this scanner and start receiving high-quality breakout signals automatically!** 🚀

---

<sub>Built with ❤️ for systematic IPO traders | Automated via GitHub Actions | Powered by proven technical analysis</sub>