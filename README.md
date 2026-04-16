# 🚀 IPO Breakout Qualification Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-2.1.0-orange.svg)](https://github.com/Deep-Adhia/IPO-Base-Scanner)
[![Automated](https://img.shields.io/badge/automation-GitHub%20Actions-green.svg)](https://github.com/features/actions)

This is **not** a simple breakout scanner.

It is a behavior-driven IPO momentum qualification system that participates **only in confirmed breakouts** with structural and volume validation. The system ruthlessly filters out market noise, grading high-quality setups while explicitly tracking every rejection — so the data can be analysed 30 days later to continuously refine the edge.

---

## 🧭 Strategy Philosophy

This system **does not predict breakouts — it validates them.**

It only participates in moves where demand is already proven through sustained price behaviour and volume expansion. By forcing the market to show its hand first, false signals are eliminated structurally rather than by intuition.

---

## 🎯 Dual-Scanner Architecture

The system runs **two independent scanners** that cover different phases of the IPO lifecycle.

### 1. 📅 Listing Day Breakout Scanner (`listing_day_breakout_scanner.py`)

Targets the **listing day** and the days immediately following, when momentum is freshest and institutional footprints are most visible.

**Flow:**
1. **Symbol detected on listing day** → enters `PENDING` observation state.
2. **Behaviour observation** → system monitors price action for 45–60 minutes post-breakout.
3. **Rejection** → terminated instantly if price falls back below breakout level or rejection tail exceeds the mathematical threshold.
4. **`CONFIRMED` execution** → only mathematically confirmed breakouts generate actionable signals.

> **Time Decay Filter**: If a confirmed breakout fails to sustain ≥1.5% away from the breakout level within a defined window (60–90 min), it is treated as a "dead breakout" and silently rejected.

---

### 2. 🔁 Consolidation Breakout Scanner (`streamlined-ipo-scanner.py`)

Targets IPOs **10–200 days post-listing** that have built a proper base structure and are breaking out of that base.

**Scan windows:** `10, 20, 40, 80, 120` days (configurable)

**Flow:**
1. Symbol must be within `8–35%` below listing-day high (base formation range).
2. Consolidation range must be `≤60%` (tight base, not chop).
3. Breakout candle must **close** above the base high (no wick fakes).
4. Volume must confirm via one of: `2.5x avg burst`, `VOL_MULT (1.2x)` rolling, or absolute `3M+ value`.
5. Follow-through filter: next candle must hold base high ±2% **or** show 80%+ continuation volume.
6. Grade and R:R checks filter further before signal emission.

---

## 🚫 Rejection Logic (Critical Filters)

The system rejects aggressively. A setup is terminated at the first failing condition:

| Filter | Reason Logged |
|---|---|
| Price below `8%` or above `35%` of listing high | Outside base formation range |
| Consolidation range `>60%` | `loose_base` — chop, not accumulation |
| Failed follow-through | `failed_follow_through` |
| Grade below minimum (`C` by default) | `low_grade` |
| Risk:Reward ratio `< 1.3` | `poor_risk_reward` |
| Entry `>8%` above breakout level | `too_extended` |
| Breakout `>10` days old | `stale_breakout` |
| Cooldown (`<10` days since last signal for same symbol) | `cooldown` |
| Symbol already in active position | Silent skip (no duplicate positions) |
| Market holiday (NSE calendar enforced) | Scanner exits cleanly with Telegram notification |

*Most symbols are rejected. Only the highest-quality setups generate signals.*

---

## 📊 Grading System

Grades are assigned by the `compute_grade_hybrid()` scoring function (5 criteria, max score 5):

**Note on terminology**
- `Grade` (consolidation scanner) and `Tier` (listing breakout engine) are independent scoring systems.
- `Grade` measures consolidation/base quality.
- `Tier` measures breakout quality and position sizing allocation.

| Grade | Score | Min Confidence | Position Bias |
|---|---|---|---|
| **A+** | 4–5 | Very High (91%) | Full size |
| **B** | 2–3 | Medium-High (75%) | Reduced + smart filters |
| **C** | 1 | Medium (65%) | Min size — monitor closely |
| **D** | 0 | Rejected | ❌ Never traded |

**5 scoring criteria:**
1. Consolidation range `≤18%` (tight base = institutional accumulation)
2. Massive volume — breakout day `≥2.5x` avg + 3-day sum `≥4x` avg
3. Momentum percentile — 20-day return in top 85th percentile
4. Technical alignment — MACD bullish + RSI `>65` + EMA20 above EMA50
5. Gap-up confirmation — next open `≥4%` above breakout close

---

## 🔁 Learning & Feedback Loop

Every rejection and signal is written to a **structured daily JSONL log**, building the dataset for algorithm tuning:

```
logs/
  YYYY-MM-DD/
    consolidation.jsonl    ← REJECTED_BREAKOUT + SIGNAL_GENERATED events
    listing_day.jsonl      ← PENDING / CONFIRMED / BREAKOUT_SIGNAL events
    watchlist.jsonl        ← Hourly watchlist SIGNAL_GENERATED + REJECTED_BREAKOUT + SCAN_COMPLETED
    positions.jsonl        ← POSITION_CLOSED + DAILY_SNAPSHOT + TRAILING_STOP_UPDATED
    scanner.jsonl          ← SCAN_COMPLETED funnel totals
```

Each JSONL entry is structured for machine parsing:
```json
{
  "timestamp": "2026-04-15 15:46:06 IST",
  "version": "2.1.0",
  "scanner": "consolidation",
  "symbol": "ARSSBL",
  "action": "REJECTED_BREAKOUT",
  "details": { "reason": "poor_risk_reward", "ratio": 0.83, "min_required": 1.3 }
}
```

Daily summary snapshots may also be generated as:

```text
logs/YYYY-MM-DD/daily_summary.json
```

This file is **derived output**, not a source-of-truth input. The source remains JSONL logs.

After 30 days this dataset allows answering:
- Which grades actually hit their targets (win rate per grade)?
- Are we exiting too early (trailing stop too tight)?
- Which rejection reason filters out the most candidates?
- Do low-volume breakouts (`LISTING_BREAKOUT_LOW_VOL`) underperform full confirms?

---

## 📂 Data Infrastructure

```text
IPO-Base-Scanner/
├── streamlined-ipo-scanner.py      # Consolidation scanner (v2.1.0)
├── listing_day_breakout_scanner.py # Listing day scanner
│
├── ipo_signals.csv                 # All confirmed signals (active + closed)
├── ipo_positions.csv               # Portfolio tracker (pnl, trailing stops, status)
├── ipo_listing_data.csv            # Listing day metrics per symbol (High, Low, Vol)
├── ipo_upstox_mapping.csv          # NSE symbol → Upstox instrument_key mapping
├── recent_ipo_symbols.csv          # Discovery layer output (symbols + listing dates)
│
└── logs/                           # Structured daily JSONL logs (per scanner per day)
    └── YYYY-MM-DD/
        ├── consolidation.jsonl
        ├── listing_day.jsonl
        ├── watchlist.jsonl
        ├── positions.jsonl
        └── scanner.jsonl
```

> **No `ipo_rejections.csv`** — rejections are tracked in the daily JSONL logs, not a flat CSV, making them queryable by date, scanner, version, and reason.

---

## 🛠️ Installation & Configuration

### 1. Setup Environment
```bash
git clone https://github.com/Deep-Adhia/IPO-Base-Scanner.git
cd IPO-Base-Scanner
pip install -r requirements.txt
cp .env.template .env
```

### 2. Configure Data Sources

⚠️ **IMPORTANT**: The Upstox API is **required** for live price confirmation during market hours. Fallback sources (NSE/YFinance) are used only for historical data outside market hours.

```bash
# .env — Upstox Analytics token (permanent, no daily login required)
UPSTOX_ACCESS_TOKEN=your_analytics_token_here

# Telegram alerts
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Tunable parameters (see .env.template for all options)
MIN_LIVE_GRADE=C         # Minimum grade for signal emission (D/C/B/A/A+)
MIN_RISK_REWARD=1.3      # Minimum R:R ratio
MIN_DAYS_BETWEEN_SIGNALS=10   # Cooldown window per symbol
CONSOL_WINDOWS=10,20,40,80,120
```

### 3. Run Manually
```bash
# Run consolidation scan
python streamlined-ipo-scanner.py scan

# Update stop-losses on active positions
python streamlined-ipo-scanner.py stop_loss_update

# Weekly / monthly summaries (Telegram)
python streamlined-ipo-scanner.py weekly_summary
python streamlined-ipo-scanner.py monthly_review
```

### 4. Automation Deployment (GitHub Actions)
Add to GitHub Repository Secrets: `UPSTOX_ACCESS_TOKEN`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`

Primary workflows:
- `ipo-scanner-v2.yml` — consolidation scanner (`scan`, `stop_loss_update`, weekly/monthly summaries)
- `listing-day-breakout.yml` — listing-day breakout scanner
- `watchlist-hourly-scanner.yml` — hourly watchlist breakout scanner

Automated schedules (IST):
| Job | Time | Cron (UTC) |
|---|---|---|
| Daily scan + stop-loss update | 2:15 PM weekdays | `45 08 * * 1-5` |
| Weekly summary | Sunday 2:45 PM | `15 09 * * 0` |
| Monthly review | 1st of month 2:45 PM | `15 09 1 * *` |

> **NSE Holiday Guard**: The scanner automatically skips NSE public holidays (full 2025–2026 calendar enforced in code). A Telegram notification is sent when a day is skipped.

---

## 📈 Quantitative Analysis (30-Day)

Run:

```bash
python analyze_30d_data.py
```

The analysis script now uses a resilient read order for rejection metrics:

1. Prefer `logs/YYYY-MM-DD/daily_summary.json` when available.
2. If missing/empty (common on fresh local pull), automatically fallback to parsing:
   - `logs/YYYY-MM-DD/consolidation.jsonl`
   - `logs/YYYY-MM-DD/listing_day.jsonl`
   - `logs/YYYY-MM-DD/watchlist.jsonl`

This means you can run analysis locally even if CI-generated summary files are not present in your branch.

---

## 📱 Alert Format (Telegram)

```text
🎯 CONSOLIDATION BREAKOUT SIGNAL

📊 Symbol: SAATVIKGL
📋 Signal Type: Consolidation-Based Breakout
📈 Grade: C (Medium Confidence)

💰 Price Information:
• Current/Live Price: ₹464.00 (🚀 Upstox Live)
• Entry Reference: ₹464.00 (Next Day Opening)

🛑 Stop Loss: ₹408.32 (12.0% risk)
🎯 Target: ₹589.58 (27.1% reward)
📊 Risk:Reward: 1:2.3

📋 Pattern Details:
• Consolidation: ₹408.32 – ₹446.90
• Breakout: ₹446.90
• Score: 1.0/5
• Consolidation Window: 80 days

🤖 Scanner v2.1.0 | 2026-04-13 10:14 IST
```

---

## ⚠️ System Discipline & Compliance

- **1 stock = 1 position**: If a symbol already has an active position, new signals for that symbol are ignored at the scan level.
- **NSE holiday-aware**: Will not scan on market holidays — no stale-price signals.
- **Manual execution only**: The engine calculates, grades, and alerts. It does **not** place API orders. Human oversight is required for every execution.
- **Educational/Analytical Tool**: Not financial or investment advice.

---

<sub>Built for systematic IPO momentum trading | v2.1.0 | Automated via GitHub Actions | Dual-scanner: Listing Day + Consolidation</sub>