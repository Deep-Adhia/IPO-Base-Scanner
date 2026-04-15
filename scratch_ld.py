def check_listing_day_breakout(symbol, listing_info, pending_breakouts=None):
    """Check if symbol has broken listing day high with volume"""
    try:
        listing_day_high = listing_info['listing_day_high']
        listing_day_low = listing_info['listing_day_low']
        listing_day_close = listing_info.get('listing_day_close', listing_day_high)  # Fallback to high if close not available
        listing_day_volume = float(listing_info.get('listing_day_volume', 0))  # CRITICAL: was missing, caused NameError
        listing_date = listing_info['listing_date']
        last_updated = listing_info.get('last_updated', 'N/A')  # When listing data was captured
        
        # Convert listing_date to date object if needed (from CSV it might be string)
        if isinstance(listing_date, str):
            listing_date = pd.to_datetime(listing_date).date()
        elif hasattr(listing_date, 'date'):
            listing_date = listing_date.date()
        elif isinstance(listing_date, pd.Timestamp):
            listing_date = listing_date.date()
        
        # Fetch current data
        df = fetch_data(symbol, listing_date)
        
        if df is None or df.empty:
            return None
        
        # Get latest data from historical
        latest = df.iloc[-1]
        historical_high = float(latest['HIGH'])
        current_volume = float(latest['VOLUME'])
        current_date = latest['DATE']
        
        # Validate data freshness - check if latest data is from today
        latest_date = latest['DATE']
        if isinstance(latest_date, pd.Timestamp):
            latest_date = latest_date.date()
        elif hasattr(latest_date, 'date'):
            latest_date = latest_date.date()
        else:
            latest_date = pd.to_datetime(latest_date).date()
        
        today_date = datetime.today().date()
        days_old = (today_date - latest_date).days
        
        # CRITICAL: Get LIVE price FIRST for accurate breakout detection
        current_price = None
        current_high = None
        price_source = "Historical Close"
        
        try:
            live_price, live_source = get_live_price(symbol)
            if live_price is not None and live_price > 0:
                current_price = live_price
                current_high = live_price  # Use live price as current high for breakout detection
                price_source = f"Live ({live_source})"
                logger.info(f"✅ Using live price for {symbol} breakout detection: ₹{current_price:.2f} from {live_source}")
        except Exception as e:
            logger.debug(f"Could not get live price for {symbol}: {e}")
        
        # Fallback to historical data if live price unavailable
        if current_price is None:
            current_price = float(latest['CLOSE'])
            current_high = historical_high  # Use historical high
            price_source = f"Historical Close ({latest_date.strftime('%Y-%m-%d')})"
            
            # Warn if data is stale
            if days_old > 1:
                logger.warning(f"⚠️ Using stale data for {symbol}: {days_old} days old ({latest_date})")
            elif days_old == 0:
                logger.info(f"✅ Using today's historical close for {symbol}: ₹{current_price:.2f}")
            else:
                logger.info(f"⚠️ Using yesterday's close for {symbol}: ₹{current_price:.2f} (market may be closed)")
        
        # Log breakout level comparison
        logger.info(f"📊 {symbol} Breakout Level Check:")
        logger.info(f"   Listing Day High: ₹{listing_day_high:.2f}")
        logger.info(f"   Current High: ₹{current_high:.2f} ({price_source})")
        logger.info(f"   Breakout Required: Current High > ₹{listing_day_high:.2f}")
        
        # Calculate average volume (last 10 days excluding listing day)
        if len(df) > 1:
            recent_df = df.tail(10)
            avg_volume = recent_df['VOLUME'].mean()
        else:
            avg_volume = current_volume
        
        # Check for breakout
        is_breakout = False
        breakout_conditions = []
        rejection_reason = None
        volume_warnings = []  # Track volume-related warnings
        
        if current_high > listing_day_high:
            is_breakout = True
            signal_type = 'BREAKOUT'
            breakout_conditions.append(f"Price broke listing day high ({current_high:.2f} > {listing_day_high:.2f})")
        elif current_high >= listing_day_high * 0.95:
            # Watchlist condition: Within 5% of listing high
            is_breakout = True  # We set this to True to proceed with calculations, but mark type as WATCHLIST
            signal_type = 'WATCHLIST'
            breakout_conditions.append(f"Near Breakout: {current_high:.2f} is within 5% of {listing_day_high:.2f}")
            logger.info(f"👀 {symbol}: Detected as WATCHLIST candidate (High: {current_high:.2f}, Trigger: {listing_day_high:.2f})")
        elif LISTING_TIER_B_ENABLED:
            # Tier B candidate: > 5% below listing high — validate tight base below
            is_breakout = True
            signal_type = 'BASE_BREAKOUT'
            logger.info(f"📦 {symbol}: Checking Tier B base breakout (live high {current_high:.2f}, listing high {listing_day_high:.2f})")
        else:
            rejection_reason = f"Price ({current_high:.2f}) below listing day high ({listing_day_high:.2f})"
        
        # Condition 2: Volume confirmation (now a warning, not a rejection)
        volume_spike = current_volume >= avg_volume * MIN_VOLUME_MULTIPLIER
        if volume_spike:
            breakout_conditions.append(f"Volume spike ({current_volume:,.0f} vs avg {avg_volume:,.0f})")
        elif is_breakout:
            # Price broke but volume insufficient - add warning instead of rejecting
            volume_warnings.append(f"Low volume spike: {current_volume:,.0f} vs avg {avg_volume:,.0f} (need {MIN_VOLUME_MULTIPLIER}x)")
        
        # Proceed if price broke listing day high OR is watchlist
        if is_breakout:
            # Tracking vars for tier classification (populated below per path)
            base_range_high: float = 0.0
            perfect_base_ok: bool = False

            # Calculate entry, stop loss, and target
            # For Watchlist, use Listing High as the hypothetical entry price
            if signal_type == 'WATCHLIST':
                entry_price = listing_day_high
            else:
                entry_price = current_price  # For confirmed breakout, use current price
            
            # CRITICAL FIX: Calculate target based on ENTRY price, not listing day high
            # This ensures target is always above entry price
            listing_range = listing_day_high - listing_day_low
            listing_range_pct = (listing_range / listing_day_high * 100) if listing_day_high > 0 else 0
            
            # Note: Listing day range is not used for rejection - listing day low is last support level
            # Stop loss is purely percentage-based (8% below entry), not based on listing day low
            
            # Calculate how far above listing high the entry is
            entry_above_high = entry_price - listing_day_high
            entry_above_high_pct = (entry_above_high / listing_day_high * 100) if listing_day_high > 0 else 0
            
            # FILTER 2: Only generate signals if entry is within reasonable distance of listing high
            # This prevents generating signals when breakout happened long ago
            # Only apply for actual BREAKOUTs, not WATCHLIST
            if signal_type == 'BREAKOUT' and entry_above_high_pct > MAX_ENTRY_ABOVE_HIGH_PCT:
                rejection_reason = f"Entry ({entry_price:.2f}) is {entry_above_high_pct:.1f}% above listing high ({listing_day_high:.2f}) - too far from breakout level"
                logger.info(f"⏭️ Skipping {symbol}: {rejection_reason}")
                return None
            
            # Days since listing (used for display + strict watchlist / breakout gates)
            today_date = datetime.today().date()
            if isinstance(listing_date, str):
                listing_date_obj = pd.to_datetime(listing_date).date()
            elif hasattr(listing_date, 'date'):
                listing_date_obj = listing_date.date()
            else:
                listing_date_obj = listing_date
            
            days_since_listing = (today_date - listing_date_obj).days
            vol_vs_avg = (current_volume / avg_volume) if avg_volume > 0 else 0.0

            # --- Strict watchlist gate: same strategic lens as IPO momentum (no stale / dead-volume radar) ---
            if LISTING_STRICT_QUALITY and signal_type == 'WATCHLIST':
                if days_since_listing > LISTING_WATCHLIST_ABSOLUTE_MAX_AGE_DAYS:
                    rejection_reason = (
                        f"Strict watchlist: IPO age {days_since_listing}d > absolute max "
                        f"{LISTING_WATCHLIST_ABSOLUTE_MAX_AGE_DAYS}d (not IPO-base regime)"
                    )
                    logger.info(f"⏭️ Skipping {symbol}: {rejection_reason}")
                    return None
                if days_since_listing > LISTING_WATCHLIST_MAX_DAYS_SINCE_LISTING:
                    rejection_reason = (
                        f"Strict watchlist: {days_since_listing}d since listing "
                        f"(max {LISTING_WATCHLIST_MAX_DAYS_SINCE_LISTING}d)"
                    )
                    logger.info(f"⏭️ Skipping {symbol}: {rejection_reason}")
                    return None
                if LISTING_WATCHLIST_MIN_VOLUME_MULT > 0 and vol_vs_avg < LISTING_WATCHLIST_MIN_VOLUME_MULT:
                    rejection_reason = (
                        f"Strict watchlist: volume {vol_vs_avg:.2f}x avg "
                        f"(need ≥{LISTING_WATCHLIST_MIN_VOLUME_MULT}x)"
                    )
                    logger.info(f"⏭️ Skipping {symbol}: {rejection_reason}")
                    return None
                if LISTING_WATCHLIST_MAX_VOL_VS_AVG > 0 and vol_vs_avg > LISTING_WATCHLIST_MAX_VOL_VS_AVG:
                    rejection_reason = (
                        f"Strict watchlist: volume {vol_vs_avg:.2f}x avg "
                        f"(need ≤{LISTING_WATCHLIST_MAX_VOL_VS_AVG}x for dry-up)"
                    )
                    logger.info(f"⏭️ Skipping {symbol}: {rejection_reason}")
                    return None

                base_ok, base_reason, base_metrics = _evaluate_watchlist_perfect_base(
                    df,
                    LISTING_WATCHLIST_BASE_LOOKBACK,
                    float(listing_day_high),
                    float(current_high),
                )
                if not base_ok:
                    rejection_reason = base_reason
                    logger.info(f"⏭️ Skipping {symbol}: {rejection_reason} {base_metrics}")
                    return None
                perfect_base_ok = True  # watchlist passed — track for debug (WATCHLIST is not a trade)

            # --- Tier B gate: validate base breakout below listing high ---
            if signal_type == 'BASE_BREAKOUT':
                dist_below_high_pct = (
                    (listing_day_high - current_high) / listing_day_high * 100.0
                    if listing_day_high > 0 else 0.0
                )
                if days_since_listing > LISTING_TIER_B_MAX_AGE_DAYS:
                    rejection_reason = (
                        f"Tier B: IPO age {days_since_listing}d > max {LISTING_TIER_B_MAX_AGE_DAYS}d"
                    )
                    logger.info(f"⏭️ Skipping {symbol}: {rejection_reason}")
                    return None
                if dist_below_high_pct > LISTING_TIER_B_MAX_DISTANCE_FROM_HIGH_PCT:
                    rejection_reason = (
                        f"Tier B: {dist_below_high_pct:.1f}% below listing high "
                        f"(max {LISTING_TIER_B_MAX_DISTANCE_FROM_HIGH_PCT}%)"
                    )
                    logger.info(f"⏭️ Skipping {symbol}: {rejection_reason}")
                    return None
                # Base quality (skip proximity — intentionally below listing high)
                base_ok_b, base_reason_b, _ = _evaluate_watchlist_perfect_base(
                    df,
                    LISTING_WATCHLIST_BASE_LOOKBACK,
                    float(listing_day_high),
                    float(current_high),
                    proximity_check=False,
                )
                if not base_ok_b:
                    rejection_reason = f"Tier B base check: {base_reason_b}"
                    logger.info(f"⏭️ Skipping {symbol}: {rejection_reason}")
                    return None
                # Confirm current_high actually broke above the recent base range high
                _tail_b = df.tail(LISTING_WATCHLIST_BASE_LOOKBACK).copy()
                _tail_b["HIGH"] = pd.to_numeric(_tail_b["HIGH"], errors="coerce")
                _tail_b = _tail_b.dropna(subset=["HIGH"])
                # exclude today's bar when computing historical base high
                base_range_high = (
                    float(_tail_b["HIGH"].iloc[:-1].max())
                    if len(_tail_b) > 1
                    else float(_tail_b["HIGH"].max())
                )
                if current_high <= base_range_high:
                    rejection_reason = (
                        f"Tier B: current high {current_high:.2f} ≤ base high {base_range_high:.2f}"
                    )
                    logger.info(f"⏭️ Skipping {symbol}: {rejection_reason}")
                    return None
                perfect_base_ok = True
                breakout_conditions.append(
                    f"Base breakout: high {current_high:.2f} > base range high {base_range_high:.2f} "
                    f"({dist_below_high_pct:.1f}% below listing high)"
                )
                logger.info(
                    f"✅ {symbol}: Tier B BASE_BREAKOUT validated "
                    f"(base {base_range_high:.2f} → {current_high:.2f}, listing high {listing_day_high:.2f})"
                )

            volume_vs_listing_day = current_volume / listing_day_volume if listing_day_volume > 0 else 0
            if volume_vs_listing_day < MIN_VOLUME_VS_LISTING_DAY:
                volume_warnings.append(f"Low volume vs listing day: {volume_vs_listing_day:.1f}x (need {MIN_VOLUME_VS_LISTING_DAY:.1f}x)")
                if signal_type == 'BREAKOUT' and not LISTING_STRICT_QUALITY:
                    logger.warning(f"⚠️ {symbol}: Low volume vs listing day ({volume_vs_listing_day:.1f}x, need {MIN_VOLUME_VS_LISTING_DAY:.1f}x) - sending signal with caution")

            # --- Strict quality gate (default): only persist / alert full-quality breakouts ---
            if LISTING_STRICT_QUALITY and signal_type == 'BREAKOUT':
                if days_since_listing > MAX_DAYS_SINCE_LISTING_FOR_BREAKOUT:
                    rejection_reason = (
                        f"Strict: {days_since_listing}d since listing (max {MAX_DAYS_SINCE_LISTING_FOR_BREAKOUT}d)"
                    )
                    logger.info(f"⏭️ Skipping {symbol}: {rejection_reason}")
                    return None
                if not volume_spike:
                    rejection_reason = (
                        f"Strict: volume spike required (current {current_volume:,.0f} vs avg {avg_volume:,.0f}, need {MIN_VOLUME_MULTIPLIER}x)"
                    )
                    logger.info(f"⏭️ Skipping {symbol}: {rejection_reason}")
                    return None
                if listing_day_volume > 0:
                    if volume_vs_listing_day < MIN_VOLUME_VS_LISTING_DAY:
                        rejection_reason = (
                            f"Strict: volume vs listing day {volume_vs_listing_day:.2f}x < {MIN_VOLUME_VS_LISTING_DAY}x"
                        )
                        logger.info(f"⏭️ Skipping {symbol}: {rejection_reason}")
                        return None
                else:
                    if current_volume < avg_volume * MIN_VOL_MULT_WHEN_NO_LISTING_VOL:
                        rejection_reason = (
                            f"Strict: listing day volume missing/0 — need current vol ≥ {MIN_VOL_MULT_WHEN_NO_LISTING_VOL}x avg ({avg_volume:,.0f})"
                        )
                        logger.info(f"⏭️ Skipping {symbol}: {rejection_reason}")
                        return None
                # Passed strict checks — treat as high-quality (no LOW_VOL grade)
                volume_warnings = []
            
            # Stop loss % below entry (configurable via LISTING_STOP_LOSS_PCT)
            stop_loss_pct = STOP_LOSS_PCT / 100.0
            
            # Calculate stop loss purely based on entry price percentage
            stop_loss = entry_price * (1 - stop_loss_pct)
            
            # Target calculation: Use entry price + percentage of listing range
            if entry_above_high_pct <= 2.0:
                target_multiplier = 1.0
            elif entry_above_high_pct <= 5.0:
                target_multiplier = 0.75
            else:
                target_multiplier = 0.5
            
            target_price = entry_price + (listing_range * target_multiplier)
            
            # Risk/Reward
            risk = entry_price - stop_loss
            reward = target_price - entry_price
            risk_reward = reward / risk if risk > 0 else 0
            
            # FILTER: Minimum risk/reward ratio (reward must be at least equal to risk)
            if risk_reward < MIN_RISK_REWARD:
                rejection_reason = f"Risk/Reward ratio ({risk_reward:.2f}) below minimum ({MIN_RISK_REWARD:.1f})"
                logger.info(f"⏭️ Skipping {symbol}: Risk/Reward ratio ({risk_reward:.2f}) is below minimum ({MIN_RISK_REWARD:.1f})")
                return None

            # Leader score gate (selection quality)
            leader_score = _leader_score(
                entry_above_high_pct=entry_above_high_pct,
                volume_spike=(current_volume / avg_volume) if avg_volume > 0 else 0,
                risk_reward=risk_reward,
                current_price=current_price,
                listing_high=listing_day_high,
                listing_range_pct=listing_range_pct
            )
            min_leader_for_signal = None
            if signal_type == 'BREAKOUT':
                min_leader_for_signal = LISTING_MIN_LEADER_SCORE
            elif LISTING_STRICT_QUALITY and signal_type == 'WATCHLIST':
                min_leader_for_signal = LISTING_WATCHLIST_MIN_LEADER_SCORE
            if min_leader_for_signal is not None and leader_score < min_leader_for_signal:
                logger.info(
                    f"⏭️ Skipping {symbol}: Leader score {leader_score} < {min_leader_for_signal} "
                    f"({'breakout' if signal_type == 'BREAKOUT' else 'watchlist'})"
                )
                return None

            # --- Detect perfect base for BREAKOUT signals (used for A+ tier eligibility) ---
            if signal_type == 'BREAKOUT':
                _pb_ok, _, _pb_metrics = _evaluate_watchlist_perfect_base(
                    df,
                    LISTING_WATCHLIST_BASE_LOOKBACK,
                    float(listing_day_high),
                    float(current_high),
                    proximity_check=False,  # price is above listing high — proximity irrelevant
                )
                perfect_base_ok = _pb_ok
                if _pb_ok:
                    logger.info(f"✅ {symbol}: BREAKOUT has tight base — A+ tier eligible")
                    write_daily_log("listing_day", symbol, "PERFECT_BASE_DETECTED", _pb_metrics)

            # Intraday confirmation engine: PENDING -> CONFIRMED -> ENTER
            if signal_type == 'BREAKOUT' and LISTING_CONFIRMATION_MINUTES > 0 and _market_is_open_ist():
                if pending_breakouts is None:
                    pending_breakouts = {}
                state = pending_breakouts.get(symbol)
                now_ts = _now_ist()
                now_iso = now_ts.isoformat()
                if not state:
                    pending_breakouts[symbol] = {
                        "started_at": now_iso,
                        "breakout_level": float(listing_day_high),
                        "max_price_seen": float(current_price),
                        "last_price": float(current_price)
                    }
                    logger.info(f"⏳ {symbol}: breakout moved to PENDING for {LISTING_CONFIRMATION_MINUTES}m confirmation")
                    write_daily_log("listing_day", symbol, "PENDING_STARTED", {
                        "breakout_level": float(listing_day_high),
                        "confirm_minutes": LISTING_CONFIRMATION_MINUTES,
                        "price": round(float(current_price), 2),
                        "leader_score": int(leader_score),
                    })
                    return {
                        "symbol": symbol,
                        "type": "PENDING",
                        "current_price": round(current_price, 2),
                        "listing_day_high": listing_day_high,
                        "confirm_minutes": LISTING_CONFIRMATION_MINUTES,
                    }
                # update state
                started = datetime.fromisoformat(state["started_at"])
                state["max_price_seen"] = max(float(state.get("max_price_seen", current_price)), float(current_price))
                state["last_price"] = float(current_price)
                pending_breakouts[symbol] = state

                # rejection filter during observation
                max_seen = float(state["max_price_seen"])
                rejection_pct = ((max_seen - current_price) / max_seen * 100) if max_seen > 0 else 0
                if current_price < listing_day_high or rejection_pct > 2.5:
                    rej_reason = "below_breakout" if current_price < listing_day_high else "rejection_from_high"
                    pending_breakouts.pop(symbol, None)
                    logger.info(f"⏭️ {symbol}: pending confirmation rejected (price hold/rejection failed)")
                    write_daily_log("listing_day", symbol, "PENDING_REJECTED", {
                        "reason": rej_reason,
                        "current_price": round(float(current_price), 2),
                        "breakout_level": float(listing_day_high),
                        "rejection_pct": round(float(rejection_pct), 2),
                        "max_price_seen": round(float(max_seen), 2),
                        "elapsed_minutes": int((now_ts - started).total_seconds() // 60),
                    })
                    return None

                elapsed_min = int((now_ts - started).total_seconds() // 60)
                if elapsed_min < LISTING_CONFIRMATION_MINUTES:
                    logger.info(f"⏳ {symbol}: pending {elapsed_min}/{LISTING_CONFIRMATION_MINUTES}m confirmed hold")
                    return {
                        "symbol": symbol,
                        "type": "PENDING",
                        "current_price": round(current_price, 2),
                        "listing_day_high": listing_day_high,
                        "confirm_minutes": LISTING_CONFIRMATION_MINUTES,
                    }
                # Confirmed
                pending_breakouts.pop(symbol, None)
                breakout_conditions.append(f"Confirmed hold {LISTING_CONFIRMATION_MINUTES}m above breakout")
                write_daily_log("listing_day", symbol, "PENDING_CONFIRMED", {
                    "breakout_level": float(listing_day_high),
                    "confirm_minutes": LISTING_CONFIRMATION_MINUTES,
                    "entry_reference": round(float(current_price), 2),
                    "leader_score": int(leader_score),
                    "elapsed_minutes": elapsed_min,
                })
            
            # Calculate gain from listing day close
            gain_from_listing_close = (
                (current_price - listing_day_close) / listing_day_close * 100
            ) if listing_day_close > 0 else 0

            # Post-confirm move: % price has moved beyond its breakout reference level
            if signal_type == 'BASE_BREAKOUT':
                post_confirm_move_pct = (
                    (current_high - base_range_high) / base_range_high * 100.0
                    if base_range_high > 0 else 0.0
                )
                # Target for BASE_BREAKOUT: the listing high is the natural objective
                if listing_day_high > entry_price:
                    target_price = listing_day_high
                    reward = target_price - entry_price
                    risk_reward = reward / risk if risk > 0 else 0
            else:
                post_confirm_move_pct = float(entry_above_high_pct)

            # --- Tier assignment (WATCHLIST always returns None — never a trade) ---
            vol_ratio_for_tier = (current_volume / avg_volume) if avg_volume > 0 else 0.0
            tier, position_size_pct, tier_rationale = _assign_breakout_tier(
                signal_type=signal_type,
                confirmed=True,  # PENDING returns early above; here we are always confirmed
                perfect_base=perfect_base_ok,
                volume_ratio=vol_ratio_for_tier,
                days_since_listing=days_since_listing,
                post_confirm_move_pct=post_confirm_move_pct,
            )
            if tier is None and signal_type != 'WATCHLIST':
                logger.info(f"⏭️ {symbol}: No tier assigned — {tier_rationale}")
                return None

            return {
                'symbol': symbol,
                'listing_date': listing_date,
                'listing_day_high': listing_day_high,
                'listing_day_low': listing_day_low,
                'listing_day_close': listing_day_close,
                'current_price': current_price,
                'current_high': current_high,
                'entry_price': round(entry_price, 2),
                'stop_loss': round(stop_loss, 2),
                'target_price': round(target_price, 2),
                'volume_spike': round(current_volume / avg_volume, 2),
                'volume_vs_listing_day': round(volume_vs_listing_day, 2),
                'listing_range_pct': round(listing_range_pct, 2),
                'risk_reward': round(risk_reward, 2),
                'breakout_date': current_date,
                'breakout_conditions': ' | '.join(breakout_conditions),
                'price_source': price_source,
                'days_since_listing': days_since_listing,
                'gain_from_listing_close': round(gain_from_listing_close, 2),
                'entry_above_high_pct': round(entry_above_high_pct, 2),
                'target_multiplier': round(target_multiplier, 2),
                'last_updated': last_updated,
                'volume_warnings': volume_warnings,
                'has_volume_caution': len(volume_warnings) > 0,
                'leader_score': int(leader_score),
                'type': signal_type,
                # --- Tier fields ---
                'tier': tier,
                'position_size_pct': position_size_pct,
                'tier_rationale': tier_rationale,
                'perfect_base': perfect_base_ok,
                'post_confirm_move_pct': round(post_confirm_move_pct, 2),
                'base_range_high': round(base_range_high, 2) if base_range_high > 0 else None,
            }

        
        # Log rejection reason if available
        if rejection_reason:
            logger.info(f"⏭️ {symbol}: Breakout rejected - {rejection_reason}")
        
        return None
    
    except Exception as e:
        logger.error(f"Error checking breakout for {symbol}: {e}")
        return None
