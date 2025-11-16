# Listing Day Breakout Scanner - Logic Summary

## Requirements (Final Implementation):
1. ✅ Detect shares near listing day high (within 5% limit)
2. ✅ Listing day high and listing day low are important levels (reference)
3. ✅ Stop loss is fixed at 8% below entry (not based on listing day low)
4. ✅ No time filter - IPOs that correct for months and then break listing day high are valid
5. ✅ Volume confirmation required (1.2x listing day volume)

## Current Implementation:

### 1. Entry Detection (Within 5% of Listing Day High)
- **Filter**: `MAX_ENTRY_ABOVE_HIGH_PCT = 5.0%`
- Only generates signals when current price is within 5% above listing day high
- Rejects signals if price is too far from breakout level
- **Rationale**: Prevents late entries after breakout has already moved significantly

### 2. Stop Loss Calculation (Fixed 8% Below Entry)
- **Fixed Percentage**: Always 8% below entry price
- **Formula**: `stop_loss = entry_price * 0.92`
- **NOT based on listing day low** - listing day low is reference only
- **Rationale**: Consistent risk management regardless of listing day range

### 3. Listing Day Low (Reference Only)
- **Purpose**: Last support level (informational/reference)
- **NOT used in stop loss calculation**
- Displayed in alerts for context
- **Rationale**: Important level to be aware of, but stop loss is entry-based

### 4. No Time Filter
- **Removed**: No maximum days since listing filter
- **Rationale**: IPOs can correct for months before breaking listing day high - these are valid signals
- Days since listing is still calculated and displayed for information

### 5. Volume Confirmation
- **Filter**: `MIN_VOLUME_VS_LISTING_DAY = 1.2x`
- Current volume must be at least 1.2x (20% higher) than listing day volume
- **Rationale**: Ensures breakout has sufficient volume support

### 6. Risk/Reward Validation
- **Minimum**: 1:1 risk/reward ratio required
- Rejects signals with poor risk/reward
- **Rationale**: Ensures potential reward justifies the risk

### 7. Target Calculation
- **Based on Entry Price**: `target = entry_price + (listing_range * multiplier)`
- **Multiplier**: 
  - Entry ≤2% above listing high → 100% of range
  - Entry 2-5% above listing high → 75% of range
- **Rationale**: Ensures target is always above entry price

## Filter Order (Applied Sequentially):

1. **Entry Distance Filter**: Entry must be within 5% of listing day high
2. **Volume Confirmation**: Current volume ≥1.2x listing day volume
3. **Stop Loss Calculation**: Fixed 8% below entry
4. **Risk/Reward Filter**: Minimum 1:1 ratio required

## Example Scenarios:

### Scenario 1: Fresh Breakout (Accepted)
- Listing High: ₹100
- Listing Low: ₹95 (5% range)
- Entry: ₹102 (2% above listing high)
- Days Since Listing: 3 days
- Current Volume: 1.5x listing day volume
- Stop Loss: ₹93.84 (8% below entry)
- Target: ₹107 (entry + 100% of range)
- Result: ✅ Accepted - Fresh breakout with proper volume

### Scenario 2: Extended Correction (Accepted)
- Listing High: ₹100
- Listing Low: ₹80 (20% range - wide range accepted)
- Entry: ₹103 (3% above listing high)
- Days Since Listing: 120 days (4 months - no time filter)
- Current Volume: 1.3x listing day volume
- Stop Loss: ₹94.76 (8% below entry)
- Target: ₹108.25 (entry + 75% of range)
- Result: ✅ Accepted - Extended correction breakout is valid

### Scenario 3: Entry Too Far (Rejected)
- Listing High: ₹100
- Entry: ₹110 (10% above listing high)
- Result: ❌ Rejected - Entry >5% above listing high

### Scenario 4: Insufficient Volume (Rejected)
- Listing High: ₹100
- Entry: ₹102 (2% above listing high)
- Current Volume: 1.0x listing day volume (below 1.2x minimum)
- Result: ❌ Rejected - Insufficient volume confirmation

### Scenario 5: Poor Risk/Reward (Rejected)
- Listing High: ₹100
- Listing Low: ₹99 (1% range - very tight)
- Entry: ₹102 (2% above listing high)
- Stop Loss: ₹93.84 (8% below entry)
- Target: ₹103 (entry + 100% of range = only ₹1 above entry)
- Risk: ₹8.16, Reward: ₹1
- Risk/Reward: 1:0.12
- Result: ❌ Rejected - Risk/Reward <1:1

## Key Points:

✅ **Stop Loss**: Always 8% below entry (fixed, not variable)
✅ **No Range Rejection**: Wide listing day ranges are accepted
✅ **No Time Filter**: IPOs correcting for months are valid
✅ **Entry Filter**: Must be within 5% of listing day high
✅ **Volume Required**: Must have 1.2x listing day volume
✅ **Listing Day Low**: Reference only (not used in calculations)
✅ **Target**: Always above entry (based on entry + % of range)

## Summary:
The system now:
- ✅ Detects shares within 5% of listing day high
- ✅ Uses fixed 8% stop loss below entry
- ✅ Accepts IPOs regardless of time since listing
- ✅ Requires volume confirmation (1.2x listing day volume)
- ✅ Shows listing day low as reference (not used in stop loss)
- ✅ Ensures minimum 1:1 risk/reward ratio
- ✅ Calculates target based on entry price (always above entry)
