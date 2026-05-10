import sys, os, pandas as pd, yfinance as yf

SYMBOLS = {
    "KWIL":   "2026-02-16",
    "MEESHO": "2025-12-10",
    "TRUALT": "2025-10-03",
}
CONSOL_WINDOWS = [10, 20, 30, 60, 90, 120]
MAX_PRNG = 25.0
VOL_MULT = 1.2
ABS_VOL_MIN = 3_000_000

for sym, listing in SYMBOLS.items():
    ticker = f"{sym}.NS"
    df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        print(f"[SKIP] {sym}")
        continue
    df = df.reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.rename(columns={"Date":"DATE","Open":"OPEN","High":"HIGH","Low":"LOW","Close":"CLOSE","Volume":"VOLUME"})
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
    df = df.sort_values("DATE").reset_index(drop=True)

    lhigh = float(df["HIGH"].iloc[0])
    print(f"\n{'='*65}")
    print(f"  {sym}  Listed: {listing}  Listing High: Rs{lhigh:.2f}  Rows: {len(df)}")
    print(f"{'='*65}")
    print(f"  {'W':>4}  {'Date':>12}  {'BaseL':>8}  {'BaseH':>8}  {'PRNG%':>6}  {'VOL':>5}  PRNG_OK  VOL_OK  STATUS")

    found_any = False
    for w in CONSOL_WINDOWS:
        if len(df) < w + 1:
            continue
        for j in range(w, len(df)):
            base = df.iloc[j-w:j]
            low   = float(base["LOW"].min())
            high2 = float(base["HIGH"].max())
            perf  = (low - lhigh) / lhigh

            # Context rule — base must sit 8-50% below listing high (50% wider than scanner for research)
            if not (0.08 <= -perf <= 0.50):
                continue

            prng = (high2 - low) / low * 100
            avgv = float(base["VOLUME"].mean())
            if avgv <= 0:
                continue

            vol_ratio = float(df["VOLUME"].iat[j]) / avgv
            close_j   = float(df["CLOSE"].iat[j])

            # Only log actual breakout candles
            if close_j <= high2:
                continue

            prng_ok = prng <= MAX_PRNG
            vol_ok  = vol_ratio >= VOL_MULT or (float(df["VOLUME"].iat[j]) * close_j) >= ABS_VOL_MIN
            caught  = prng_ok and vol_ok

            miss = ""
            if not prng_ok: miss += f"PRNG={prng:.1f}>{MAX_PRNG} "
            if not vol_ok:  miss += f"VOL={vol_ratio:.1f}x<{VOL_MULT}x"

            status = "CAUGHT" if caught else f"MISSED ({miss.strip()})"
            print(f"  {w:>4}  {str(df['DATE'].iat[j]):>12}  {low:>8.2f}  {high2:>8.2f}  {prng:>6.1f}  {vol_ratio:>5.1f}x  {'Y':>7}  {'Y' if vol_ok else 'N':>6}  {status}")
            found_any = True

    if not found_any:
        print("  [!] No breakout bases detected under current scanner parameters")
