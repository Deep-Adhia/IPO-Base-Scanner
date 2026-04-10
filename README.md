# 🚀 IPO Breakout Qualification Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Automated](https://img.shields.io/badge/automation-GitHub%20Actions-green.svg)](https://github.com/features/actions)

This is **not** a simple breakout scanner.

It is a behavior-driven IPO momentum qualification system that trades **only confirmed breakouts** with structural and volume validation. The system ruthlessly filters out market noise, grading high-quality setups while explicitly tracking its own rejections to continuously refine its edge.

## 🧭 **Strategy Philosophy**

This system **does not predict breakouts — it validates them**. 

It only participates in moves where demand is already proven through sustained price behavior and volume expansion. By forcing the market to show its hand first, we eliminate the false signals that plague generic scanners.

## 🎯 **System Overview & Edge**

### 🧠 **Confirmation Engine (Core Edge)**
Breakouts are *never* traded immediately upon crossing a line.

**Flow:**
1. **Breakout Detected** → Symbol enters `PENDING` state.
2. **Behavior Observation** → The system observes the price action for 45–60 minutes.
3. **Rejection** → Fails instantly if the price falls back below the breakout level or if the rejection tail exceeds the mathematical threshold.
4. **Execution** → Only mathematically `CONFIRMED` breakouts generate actionable trade signals.

*This eliminates the vast majority of false breakouts and intraday traps.*

### 🚫 **Rejection Logic (Critical Filters)**
The system rejects aggressively. It is built to avoid weak setups. A setup is terminated if:
- **IPO Age is too high** (loss of momentum).
- **Volume is insufficient** (lacks institutional footprints).
- **Base structure is loose** (high-volatility chop rather than accumulation).
- **Breakout fails confirmation** (inability to hold the high).
- **Time Decay Filter**: If a confirmed breakout fails to push ≥1.5% away from the breakout level within a defined time window (e.g., 60–90 minutes), it is treated as a "dead breakout" and rejected. (Post-confirm momentum failure).

*Most signals are rejected — only the absolute highest-quality trades survive to the Execution stage.*

### 🔁 **Learning & Feedback Loop**
The system continuously improves its own configurations through:
- **Rejection Audits (`ipo_rejections.csv`)**: A strict log tracking exactly *why* a trade was filtered (e.g., "Volume was 1.7x, needed 1.8x").
- **Trade Outcome Tracking**: Position and result history logged to `ipo_positions.csv`.
- **Data-driven refinement**: You can review the rejection logs 30 days later to see if you missed massive runners because your rules were too tight, enabling continuous optimization of your strategy parameters.

---

## 🏗️ **Tiered Decision Architecture**

The scanner relies on a strict capital allocation engine that sizes positions directly according to the structural quality of the base and breakout metrics.

- **Tier A+ (`100% Size`)**: The perfect storm. IPO age ≤ 60d + tight structural base + massive volume spike (≥ 1.8x).
- **Tier A (`60% Size`)**: Pure momentum. Excellent volume (≥ 2.0x) + extreme youth (IPO age ≤ 45d).
- **Tier B (`40% Size`)**: Highly structured accumulation base breakout, catching the move *before* it hits the listing high.
- **Controlled Fallback (`50% Size`)**: Used sparingly for valid momentum setups that miss ideal A/A+ conditions. These trades are explicitly marked as lower-conviction and should be managed more aggressively.

**Note:**
- `WATCHLIST` signals are NOT trades. They simply prime the system.
- Tier assignment and sizing calculation occurs ONLY after the confirmation engine validates the breakout.
- Unconfirmed signals are ignored entirely.

---

## 🛠️ **Installation & Configuration**

### **1. Setup Environment**
```bash
git clone https://github.com/yourusername/ipo-breakout-scanner.git
cd ipo-breakout-scanner
pip install -r requirements.txt
cp .env.template .env
```

### **2. Configure Data Sources**
⚠️ **IMPORTANT:** The Upstox API is **REQUIRED** for accurate intraday confirmation and volume validation. Fallback sources (NSE/YFinance) are **NOT** suitable for real-time decision-making and are strictly for non-critical/historical use out of market hours.

In your `.env` file:
```bash
# Get this by creating an "Analytics App" in the Upstox Developer Dashboard 
# (Permanent 1-year token. No 2FA or daily logins required!)
UPSTOX_ACCESS_TOKEN=your_analytics_token_here

# Telegram Alerts (Optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### **3. Automation Deployment**
To deploy to GitHub Actions for fully automated cloud processing:
1. Add `UPSTOX_ACCESS_TOKEN`, `TELEGRAM_BOT_TOKEN`, and `TELEGRAM_CHAT_ID` to your GitHub Repository Secrets.
2. Push to the `main` branch.
3. The system will automatically spawn scanning jobs based on the cron schedules in `.github/workflows/`.

---

## 📂 **Data Infrastructure Flow**

```text
ipo-breakout-scanner/
├── ipo_signals.csv                 # All mathematically approved, confirmed trades (with tier sizes)
├── ipo_rejections.csv              # Audit log of potential setups blocked by validation (False-negative tracking)
├── ipo_positions.csv               # Active/Closed portfolio tracking and trailing stops
├── ipo_listing_data.csv            # Upstox-sourced listing metrics (High, Low, Close, Volume)
├── recent_ipo_symbols.csv          # Discovery layer (NSE) output
└── logs/                           # Raw system runtime logs
```

---

## 📱 **Alert Standard (Telegram)**

When a trade survives the confirmation engine and is mathematically assigned a tier, the system fires a highly readable standard alert:

```text
🎯 LISTING DAY HIGH BREAKOUT!

📊 KAYNES
📋 Signal Type: Listing Day Breakout

🏆 TIER: A+  |  💰 Position Size: 100%
📌 Perfect Base + High Volume Breakout

⏰ Context & Timing:
• Age: 12 days old
• Post-Confirm Move: +2.15%
• ✅ Perfect Base Detected

💰 Trade Details:
• Current Price: ₹1,247.50 (Live)
• Entry Target: ₹1,245.00
• Stop Loss: ₹1,120.15 (-10%)
• Target Obj: ₹1,450.00
• Risk:Reward: 1:2.0

📈 Metrics:
• Listing Day High: ₹1,200.00 (BROKEN!)
• Base High: ₹1,195.00

📊 Confirmation:
• Volume Spike: 2.5x avg
• Vol vs Listing: 1.1x ✅

⚡ Action Required: Consider entry based on tier size.
```

---

## ⚠️ **System Discipline & Compliance**

### **Risk Disclaimer**
- **Conflict Prevention:** The system ensures absolute discipline: `1 stock = 1 position`. If a stock is currently held in your active portfolio, new signals on that symbol are rigorously ignored.
- **Manual Execution Only:** This engine calculates, sizes, grades, and alerts. It **does not** place API trades automatically. It acts as an analytical overlay, requiring human oversight for execution compliance and final risk approval.
- **Educational/Analytical Tool:** Not financial or investment advice.

---
<sub>Built for systematic IPO momentum trading | Automated via GitHub Actions | Driven by mathematical price action</sub>