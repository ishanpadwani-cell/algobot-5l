import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from bot_core import (
    ADDON_BUY_RS,
    COOLDOWN_DAYS,
    CLOSED_COLS,
    FEE_PCT,
    FEATURES,
    FIRST_BUY_RS,
    IPO_MAX_HOLD_DAYS,
    MAX_OPEN_STOCKS,
    NSELiveClient,
    OPEN_COLS,
    PENDING_CONFIRM_FILE,
    PENDING_EXITS_FILE,
    STOP_LOSS_PCT,
    append_row,
    build_dashboard,
    compute_baseline_from_df,
    download_in_chunks,
    extract_ohlcv,
    fetch_quotes_batch,
    fetch_recent_ipos,
    git_sync,
    inspect_company_stack,
    is_market_open_now,
    is_trading_day,
    load_brain,
    load_json,
    load_wallet,
    log_ai_memory_entry,
    market_day_progress,
    now_ist,
    save_json,
    save_wallet,
    scan_news,
    score_with_brain,
    send_telegram,
    stamp_ai_memory_result,
    to_bool,
    to_float,
    to_int,
    trading_days_between,
)
from train_and_audit import run_holiday_routine

warnings.filterwarnings("ignore")

POLL_SECONDS = 180   # Live scan every 3 minutes
CONFIRM_CHECKS = 2   # 2 consecutive checks (~3-6 mins) to block 1-min fakeout spikes


def load_pending_confirm(today_str: str) -> dict:
    data = load_json(PENDING_CONFIRM_FILE, {"date": today_str, "counts": {}})
    if data.get("date") != today_str:
        return {}
    return dict(data.get("counts", {}))


def save_pending_confirm(today_str: str, counts: dict):
    save_json(PENDING_CONFIRM_FILE, {"date": today_str, "counts": counts})


def run_live_shift(shift: str = "auto", once: bool = False):
    t_now = now_ist()

    # 1. AUTOMATIC HOLIDAY / WEEKEND SWITCH
    if not once and not is_trading_day(t_now):
        if shift in ("1", "auto"):
            print(f"[Holiday/Weekend Detected: {t_now.strftime('%Y-%m-%d')}] Switching to Self-Train + Holding Audit!")
            run_holiday_routine(force_train=False)
        else:
            print(f"[Holiday/Weekend: {t_now.strftime('%Y-%m-%d')}] Shift 2 skipped (morning shift handled training & audit).")
        return

    # Ensure a Champion Brain exists before live trading
    brain = load_brain()
    if brain is None:
        print("[First Run] No champion_brain.pkl found — training initial brain now...")
        run_holiday_routine(force_train=True)
        brain = load_brain()
        if brain is None:
            raise RuntimeError("Failed to create initial champion_brain.pkl.")

    ai_min_prob = float(brain.get("threshold", 0.45))
    ai_active = bool(brain.get("ai_active", False)) and (brain.get("calibrator") is None)
    brain_auc = float(brain.get("auc", 0.0))

    # 2. PRE-OPEN AUTO-WAIT (8:30 AM -> 9:15:20 AM IST)
    if not once and is_trading_day(t_now):
        cur_mins = t_now.hour * 60 + t_now.minute
        if cur_mins < (9 * 60 + 15):
            open_target = t_now.replace(hour=9, minute=15, second=20, microsecond=0)
            wait_secs = max(5, int((open_target - t_now).total_seconds()))
            print(f"[Pre-Open Ready] Woke up at {t_now.strftime('%H:%M IST')}. Waiting {wait_secs}s for 9:15 AM NSE open...")
            time.sleep(wait_secs)

    live_session = is_market_open_now() and (not once)
    today_ist_ts = pd.Timestamp(now_ist().date())
    nse = NSELiveClient()

    nifty_raw = extract_ohlcv(yf.download("^NSEI", period="3mo", interval="1d", auto_adjust=True, progress=False), "^NSEI")
    if nifty_raw.empty or len(nifty_raw) < 25:
        raise RuntimeError("Could not download Nifty 50 baseline.")

    session_dt = today_ist_ts if (is_trading_day(now_ist()) and now_ist().hour >= 9) else nifty_raw.index[-1]
    nifty_prior = nifty_raw[nifty_raw.index < session_dt]
    nifty_prev_closes = nifty_prior["Close"].tail(19).values
    nifty_5d_ago = float(nifty_prior["Close"].iloc[-5])

    # 3. ONE-TIME SHIFT STARTUP: CACHE 1-YEAR BASELINES
    state, open_df, closed_df = load_wallet()
    ipo_pool = fetch_recent_ipos(session_dt)
    res_pool = nse.get_upcoming_board_meetings(session_dt)
    radar_pool = {**ipo_pool}
    for sym, meta in res_pool.items():
        if sym in radar_pool:
            meta["AnchorLock"] = radar_pool[sym]["AnchorLock"]
        radar_pool[sym] = meta

    open_syms = open_df["Symbol"].astype(str).str.strip().tolist() if not open_df.empty else []
    startup_syms = list(dict.fromkeys(list(radar_pool.keys()) + open_syms))
    print(f"[Shift {shift} Startup] Session {session_dt.date()} | {len(res_pool)} Results + {len(ipo_pool)} IPOs + {len(open_syms)} Open")

    raw_charts = download_in_chunks([f"{s}.NS" for s in startup_syms], period="1y", chunk_size=40)
    history_cache = {}
    for sym in startup_syms:
        df = raw_charts.get(f"{sym}.NS", pd.DataFrame())
        if not df.empty:
            base = compute_baseline_from_df(df, session_dt)
            if base is not None:
                history_cache[sym] = base

    liquid_syms = {
        sym for sym, b in history_cache.items()
        if (b["prev_close"] >= 45.0 and b["med_turn_cr"] >= 5.0) or (sym in res_pool) or (sym in open_syms)
    }
    known_res_syms = set(res_pool.keys())
    pending_confirm = load_pending_confirm(str(session_dt.date()))
    last_live_prices = {}
    last_sync_time = time.time()

    # Shift cutoff times (IST) so Shift 1 hands off cleanly to Shift 2, and Shift 2 closes after 3:30 PM
    shift_end_mins = (12 * 60 + 12) if shift == "1" else (15 * 60 + 38)
    cycle = 0
    latest_nifty_str = "Nifty 50 — Live"

    while True:
        cycle += 1
        try:
            state, open_df, closed_df = load_wallet()
            pending_exits = load_json(PENDING_EXITS_FILE, {})
            live_now = is_market_open_now()
            day_prog = market_day_progress() if live_now else 1.0
            action_log = []

            # A. Live Nifty 50 Traffic Light
            nq = nse.get_nifty_live()
            nifty_c, nifty_src = (nq["ltp"], nq["source"]) if nq else (float(nifty_raw["Close"].iloc[-1]), "YF-CLOSED")
            nifty_20 = float(np.mean(np.append(nifty_prev_closes, nifty_c)))
            nifty_ret5 = (nifty_c / nifty_5d_ago) - 1.0
            nifty_ok = nifty_c >= nifty_20
            tranche1_rs = FIRST_BUY_RS if nifty_ok else (FIRST_BUY_RS * 0.5)
            latest_nifty_str = f"Nifty 50 [{nifty_src}]: {nifty_c:,.1f} vs 20-DMA {nifty_20:,.1f} ({'HEALTHY' if nifty_ok else 'WEAK'})"

            print(f"\n--- Cycle #{cycle} | {now_ist().strftime('%H:%M:%S IST')} | {latest_nifty_str} ---")

            # B. Midday Results-Date Watcher
            if cycle > 1:
                fresh_res = nse.get_upcoming_board_meetings(session_dt)
                new_filings = [s for s in fresh_res if s not in known_res_syms]
                if new_filings:
                    alert_txt = f"📢 New Midday NSE Results Filing: {', '.join(new_filings)}"
                    print("  " + alert_txt)
                    send_telegram(alert_txt)
                    new_raw = download_in_chunks([f"{s}.NS" for s in new_filings], period="1y", chunk_size=20)
                    for sym in new_filings:
                        meta = fresh_res[sym]
                        if sym in radar_pool:
                            meta["AnchorLock"] = radar_pool[sym]["AnchorLock"]
                        radar_pool[sym] = meta
                        known_res_syms.add(sym)
                        if sym not in history_cache:
                            df = new_raw.get(f"{sym}.NS", pd.DataFrame())
                            if not df.empty:
                                b = compute_baseline_from_df(df, session_dt)
                                if b is not None:
                                    history_cache[sym] = b
                        if sym in history_cache:
                            liquid_syms.add(sym)

            # C. Direct NSE Live Quotes (strictly zero stale fallback during live market hours)
            current_open_syms = open_df["Symbol"].astype(str).str.strip().tolist() if not open_df.empty else []
            poll_syms = list(dict.fromkeys(current_open_syms + sorted(liquid_syms)))
            quotes = fetch_quotes_batch(nse, poll_syms, history_cache, allow_closed_fallback=(not live_session))

            # D. Manage Open Positions + Execute Any Holiday-Audit Queued 9:15 AM Exits
            updated_rows = []
            open_val = 0.0
            pending_exits_changed = False

            for _, pos in open_df.iterrows():
                sym = str(pos["Symbol"]).strip()
                entry_p = to_float(pos["EntryPrice"])
                qty = to_int(pos["Qty"])
                sl = to_float(pos["StopLoss"])
                invested = to_float(pos["Invested"])
                realized = to_float(pos["RealizedPnL"])
                addon = to_bool(pos["BlueSkyAddOn"])
                booked_half = to_bool(pos["BookedHalf"])
                exit_dt = pd.Timestamp(pos["MandatoryExitDate"]).normalize()
                days_held = to_int(pos["DaysHeld"])
                bought_today = (str(pos["EntryDate"]) == str(session_dt.date()))

                q = quotes.get(sym)
                if q is None:
                    mark_p = last_live_prices.get(sym, entry_p)
                    updated_rows.append(pos.to_dict())
                    open_val += mark_p * qty
                    continue

                ltp, open_p, high_p, low_p = q["ltp"], q["open"], q["high"], q["low"]
                last_live_prices[sym] = ltp

                # D1. Holiday-Audit Queued Exit at 9:15 AM Open
                if sym in pending_exits and live_now:
                    proceeds = qty * ltp * (1 - FEE_PCT)
                    state["cash"] += proceeds
                    pnl = realized + (proceeds - invested)
                    reason = f"Holiday Audit Exit @ ₹{ltp:.1f} ({pending_exits[sym].get('reason', 'Thesis broken')})"
                    closed_df = append_row(closed_df, {
                        "Symbol": sym, "Catalyst": pos["Catalyst"], "EntryDate": pos["EntryDate"],
                        "ExitDate": str(session_dt.date()), "ResultsDate": pos["ResultsDate"],
                        "BlueSkyAddOn": addon, "NetPnL": round(pnl, 0), "WhyItExited": reason,
                    }, CLOSED_COLS)
                    stamp_ai_memory_result(sym, str(pos["EntryDate"]), pnl)
                    pending_exits.pop(sym, None)
                    pending_exits_changed = True
                    action_log.append(f"🔴 SELL {sym} (AUDIT EXIT) | P&L: ₹{pnl:+,.0f} | {reason}")
                    continue

                # D2. Stop-Loss Check
                sl_hit = (ltp <= sl) if bought_today else (low_p <= sl)
                if sl_hit:
                    exit_p = ltp if bought_today else min(open_p, sl)
                    proceeds = qty * exit_p * (1 - FEE_PCT)
                    state["cash"] += proceeds
                    pnl = realized + (proceeds - invested)
                    reason = f"Hit {'intraday' if bought_today else ('breakeven' if addon else '5% hard')} Stop-Loss @ ₹{exit_p:.1f}"
                    closed_df = append_row(closed_df, {
                        "Symbol": sym, "Catalyst": pos["Catalyst"], "EntryDate": pos["EntryDate"],
                        "ExitDate": str(session_dt.date()), "ResultsDate": pos["ResultsDate"],
                        "BlueSkyAddOn": addon, "NetPnL": round(pnl, 0), "WhyItExited": reason,
                    }, CLOSED_COLS)
                    stamp_ai_memory_result(sym, str(pos["EntryDate"]), pnl)
                    action_log.append(f"🛑 SELL {sym} (STOP-LOSS) | P&L: ₹{pnl:+,.0f} | {reason}")
                    continue

                if not bought_today:
                    if str(pos["LastUpdatedDate"]) != str(session_dt.date()):
                        days_held += 1

                    # D3. Partial Book 50% at +12%
                    if not booked_half and high_p >= entry_p * 1.12 and qty >= 2:
                        sq = qty // 2
                        sell_p = entry_p * 1.12
                        proc = sq * sell_p * (1 - FEE_PCT)
                        cpart = invested * (sq / qty)
                        state["cash"] += proc
                        realized += (proc - cpart)
                        invested -= cpart
                        qty -= sq
                        booked_half = True
                        sl = max(sl, entry_p * 1.03)
                        action_log.append(f"💰 PARTIAL BOOK 50% {sym} @ ₹{sell_p:.1f} (+12% hit, SL trailed to +3%)")

                    # D4. Open Blue Sky Add-On (+₹40k) for winners
                    b = history_cache.get(sym)
                    if b and not addon and ltp > entry_p and session_dt < exit_dt and state["cash"] >= ADDON_BUY_RS:
                        trapped_pct = float(np.mean(b["prior_highs_252"] > ltp))
                        if ltp >= b["prev_high_252"] * 0.995 or trapped_pct <= 0.02:
                            aq = int(ADDON_BUY_RS / ltp)
                            cost = aq * ltp * (1 + FEE_PCT)
                            if aq >= 1 and cost <= state["cash"]:
                                state["cash"] -= cost
                                total_qty = qty + aq
                                entry_p = ((entry_p * qty) + (ltp * aq)) / total_qty
                                qty = total_qty
                                invested += cost
                                addon = True
                                sl = max(sl, entry_p)
                                action_log.append(f"🚀 ADD-ON {sym} (+{aq} sh @ ₹{ltp:.1f} | Open Blue Sky, SL ₹{sl:.1f})")

                    # D5. Mandatory Exit (Pre-Results or Day-6 IPO cap)
                    ready_to_exit = (not live_now) or ((now_ist().hour * 60 + now_ist().minute) >= (15 * 60 + 10))
                    if session_dt >= exit_dt and ready_to_exit:
                        proceeds = qty * ltp * (1 - FEE_PCT)
                        state["cash"] += proceeds
                        pnl = realized + (proceeds - invested)
                        reason = (
                            f"Sold 1 trading day BEFORE {pos['ResultsDate']} results"
                            if pos["Catalyst"] == "Pre-Results"
                            else f"Day-{IPO_MAX_HOLD_DAYS} IPO time exit @ ₹{ltp:.1f}"
                        )
                        closed_df = append_row(closed_df, {
                            "Symbol": sym, "Catalyst": pos["Catalyst"], "EntryDate": pos["EntryDate"],
                            "ExitDate": str(session_dt.date()), "ResultsDate": pos["ResultsDate"],
                            "BlueSkyAddOn": addon, "NetPnL": round(pnl, 0), "WhyItExited": reason,
                        }, CLOSED_COLS)
                        stamp_ai_memory_result(sym, str(pos["EntryDate"]), pnl)
                        action_log.append(f"⏰ SELL {sym} (MANDATORY EXIT) | P&L: ₹{pnl:+,.0f} | {reason}")
                        continue

                pdict = pos.to_dict()
                pdict.update({
                    "EntryPrice": round(entry_p, 2), "Qty": qty, "StopLoss": round(sl, 2),
                    "Invested": round(invested, 2), "RealizedPnL": round(realized, 2),
                    "BlueSkyAddOn": addon, "BookedHalf": booked_half,
                    "DaysHeld": days_held, "LastUpdatedDate": str(session_dt.date()),
                })
                updated_rows.append(pdict)
                open_val += ltp * qty

            open_df = pd.DataFrame(updated_rows, columns=OPEN_COLS)
            if pending_exits_changed:
                save_json(PENDING_EXITS_FILE, pending_exits)

            already_open = set(open_df["Symbol"].astype(str).str.strip().tolist()) if not open_df.empty else set()
            cooldown_syms = set()
            if not closed_df.empty:
                for _, cr in closed_df.iterrows():
                    exd = pd.to_datetime(cr["ExitDate"], errors="coerce")
                    if pd.notna(exd) and trading_days_between(exd, session_dt) < COOLDOWN_DAYS:
                        cooldown_syms.add(str(cr["Symbol"]).strip())

            # E. Scan New Breakouts & Score with Champion AI
            eligible = []
            for sym in liquid_syms:
                meta = radar_pool.get(sym)
                b = history_cache.get(sym)
                q = quotes.get(sym)
                if not meta or not b or not q:
                    continue

                ltp = q["ltp"]
                prev_c = q["prev_close"] if q["prev_close"] > 0 else b["prev_close"]
                if ltp < 50 or prev_c <= 0 or q["volume"] <= 0:
                    continue

                move_1d = (ltp - prev_c) / prev_c
                raw_vol_x = q["volume"] / b["vol_med_15"]
                pace_vol_x = min(raw_vol_x / day_prog, 15.0)
                vol_ok = (pace_vol_x >= 1.5) and (raw_vol_x >= (0.45 if live_now and day_prog < 0.85 else 1.5))

                is_breakout = (0.025 <= move_1d <= 0.085) and vol_ok and (b["ret_10d"] <= 0.25)
                if not is_breakout:
                    pending_confirm.pop(sym, None)
                    continue
                if sym in already_open or sym in cooldown_syms or "BLOCK:" in meta["AnchorLock"]:
                    continue

                pct_from_high = (ltp - b["prev_high_252"]) / b["prev_high_252"]
                trapped_pct = float(np.mean(b["prior_highs_252"] > ltp))
                is_blue_sky = int((ltp >= b["prev_high_252"] * 0.995) or (trapped_pct <= 0.02 and pace_vol_x >= 1.3))
                hi_lo = q["high"] - q["low"]
                feat = {
                    "today_move_pct": move_1d, "vol_surge_med": pace_vol_x, "flat_base_4d": b["flat_4d"],
                    "vol_dry_4d": b["vol_dry_4d"], "pct_from_high": pct_from_high, "trapped_sellers_pct": trapped_pct,
                    "is_blue_sky": is_blue_sky, "close_strength": (ltp - q["low"]) / hi_lo if hi_lo > 0 else 0.5,
                    "ret_10d_prior": b["ret_10d"], "atr_pct": b["atr_pct"],
                    "nifty_above_20dma": int(nifty_ok), "nifty_ret_5d": nifty_ret5,
                    "is_preresults": 1 if meta["Catalyst"] == "Pre-Results" else 0,
                }
                if not all(np.isfinite(float(feat[k])) for k in FEATURES):
                    continue

                ai_prob = score_with_brain(brain, feat)
                if ai_active and ai_prob < ai_min_prob:
                    continue

                eligible.append({
                    "Symbol": sym, "Catalyst": meta["Catalyst"], "Feed": q["source"],
                    "Price": round(ltp, 1), "1D%": round(move_1d * 100, 1), "VolPace": round(pace_vol_x, 2),
                    "AI_Odds": ai_prob, "ResultsDate": meta["ResultsDate"], "ExitDate": meta["ExitDate"],
                    "IsHotQIB": meta["QIBTag"] != "-", "Features": feat,
                })

            eligible.sort(
                key=lambda x: (x["Catalyst"] == "Pre-Results", x["IsHotQIB"], x["AI_Odds"], x["VolPace"]),
                reverse=True,
            )

            for c in eligible:
                if len(open_df) >= MAX_OPEN_STOCKS:
                    break
                sym = c["Symbol"]
                fund = inspect_company_stack(sym)
                news = scan_news(sym)
                if not fund["FundPass"] or not news["NewsPass"]:
                    continue

                if live_now and CONFIRM_CHECKS > 1:
                    cnt = pending_confirm.get(sym, 0) + 1
                    pending_confirm[sym] = cnt
                    if cnt < CONFIRM_CHECKS:
                        print(f"    [Confirming 1/2] {sym} @ ₹{c['Price']} (+{c['1D%']}%, AI {c['AI_Odds']*100:.1f}%)")
                        continue

                price = c["Price"]
                qty = int(tranche1_rs / price)
                cost = qty * price * (1 + FEE_PCT)
                if qty >= 1 and cost <= state["cash"]:
                    state["cash"] -= cost
                    sl_p = round(price * (1 - STOP_LOSS_PCT), 2)
                    reason = f"{c['Catalyst']} | AI {c['AI_Odds']*100:.1f}% | +{c['1D%']}% on {c['VolPace']}x vol | {fund['MarginTrend']}"
                    open_df = append_row(open_df, {
                        "Symbol": sym, "Catalyst": c["Catalyst"], "EntryDate": str(session_dt.date()),
                        "EntryPrice": price, "Qty": qty, "StopLoss": sl_p, "Invested": round(cost, 2),
                        "RealizedPnL": 0.0, "BlueSkyAddOn": False, "BookedHalf": False,
                        "ResultsDate": c["ResultsDate"], "MandatoryExitDate": c["ExitDate"],
                        "DaysHeld": 0, "LastUpdatedDate": str(session_dt.date()), "EntryReason": reason,
                    }, OPEN_COLS)
                    log_ai_memory_entry(sym, str(session_dt.date()), c["Features"])
                    pending_confirm.pop(sym, None)
                    already_open.add(sym)
                    last_live_prices[sym] = price
                    open_val += price * qty
                    action_log.append(
                        f"🟢 BUY {sym}: {qty} sh @ ₹{price} | SL ₹{sl_p} | Exit {c['ExitDate']} | AI {c['AI_Odds']*100:.1f}%"
                    )

            save_wallet(state, open_df, closed_df)
            save_pending_confirm(str(session_dt.date()), pending_confirm)

            total_eq = state["cash"] + open_val
            net_pnl = total_eq - state["starting_cash"]
            print(f"  Wallet: ₹{total_eq:,.0f} (P&L ₹{net_pnl:+,.0f}) | Open: {len(open_df)}/{MAX_OPEN_STOCKS} | Quotes: {len(quotes)}/{len(poll_syms)}")

            # Push dashboard & alert immediately on any trade, or refresh web dashboard every 30 mins
            if action_log:
                build_dashboard(live_prices=last_live_prices, nifty_str=latest_nifty_str, status_mode="Live Trading Active")
                send_telegram("⚡ AlgoBot Trade Alert:\n" + "\n".join(action_log) + f"\n\nTotal Equity: ₹{total_eq:,.0f} ({net_pnl:+,.0f})")
                git_sync(f"Live Trade Update ({now_ist().strftime('%Y-%m-%d %H:%M IST')})")
                last_sync_time = time.time()
            elif (time.time() - last_sync_time) >= 1800:
                build_dashboard(live_prices=last_live_prices, nifty_str=latest_nifty_str, status_mode="Live Monitoring")
                git_sync(f"Live Portfolio Refresh ({now_ist().strftime('%H:%M IST')})")
                last_sync_time = time.time()

        except Exception as exc:
            print(f"[Cycle #{cycle} recovered from error: {exc}]")

        # Check exit conditions for this shift
        cur_t = now_ist()
        cur_mins = cur_t.hour * 60 + cur_t.minute
        if once or (not live_session) or (cur_mins >= shift_end_mins):
            break
        time.sleep(POLL_SECONDS)

    # Final shift wrap-up: update dashboard & send End-of-Day Telegram report after 3:30 PM
    state, open_df, closed_df = load_wallet()
    build_dashboard(
        live_prices=last_live_prices,
        nifty_str=latest_nifty_str,
        status_mode="Market Closed — EOD Synced" if shift == "2" else "Shift 1 Handoff",
    )
    git_sync(f"Shift {shift} Complete ({now_ist().strftime('%Y-%m-%d %H:%M IST')})")

    if shift == "2" or once:
        open_val = sum(
            to_float(last_live_prices.get(str(r["Symbol"]).strip()), to_float(r["EntryPrice"])) * to_int(r["Qty"])
            for _, r in open_df.iterrows()
        )
        total_eq = to_float(state["cash"]) + open_val
        net_pnl = total_eq - to_float(state["starting_cash"])
        pos_lines = [
            f"• {r['Symbol']}: {to_int(r['Qty'])} sh @ ₹{to_float(r['EntryPrice']):.1f} "
            f"(Live ₹{to_float(last_live_prices.get(str(r['Symbol']).strip()), to_float(r['EntryPrice'])):.1f}, SL ₹{to_float(r['StopLoss']):.1f})"
            for _, r in open_df.iterrows()
        ]
        summary = (
            f"📊 AlgoBot Daily Summary ({now_ist().strftime('%d %b %Y')})\n"
            f"Total Equity: ₹{total_eq:,.0f} (Net P&L: ₹{net_pnl:+,.0f})\n"
            f"Cash: ₹{to_float(state['cash']):,.0f} | Open: {len(open_df)}/{MAX_OPEN_STOCKS}\n\n"
            + ("\n".join(pos_lines) if pos_lines else "100% Virtual Cash.")
        )
        send_telegram(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shift", default="auto", choices=["1", "2", "auto"])
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run_live_shift(shift=args.shift, once=args.once)
