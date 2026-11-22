import io
import os
import pickle
import warnings
from datetime import timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.metrics import roc_auc_score

from bot_core import (
    AUDIT_FILE,
    BRAIN_FILE,
    EVOLUTION_LOG,
    FEATURES,
    HAS_CFFI,
    HEADERS,
    MEMORY_FILE,
    META_FILE,
    MIN_LIVE_AUC,
    NSELiveClient,
    PENDING_EXITS_FILE,
    build_dashboard,
    compute_baseline_from_df,
    download_in_chunks,
    extract_ohlcv,
    fetch_recent_ipos,
    git_sync,
    inspect_company_stack,
    load_brain,
    load_json,
    load_wallet,
    now_ist,
    save_json,
    save_wallet,
    scan_news,
    score_with_brain,
    send_telegram,
    to_float,
    to_int,
    trading_days_between,
)

if HAS_CFFI:
    from curl_cffi import requests as cffi_requests

warnings.filterwarnings("ignore")

CORE_UNIVERSE = [
    "MAZDOCK", "RVNL", "BDL", "HAL", "BEL", "COCHINSHIP", "IRFC", "IRCTC", "HUDCO", "NBCC",
    "NCC", "KPIL", "GMRAIRPORT", "KEC", "IRB", "ENGINERSIN", "SUZLON", "BHEL", "TATAPOWER",
    "TDPOWERSYS", "INOXWIND", "SJVN", "NHPC", "HINDCOPPER", "NATIONALUM", "SAIL", "NMDC",
    "JINDALSAW", "APLAPOLLO", "IEX", "IDFCFIRSTB", "RBLBANK", "BANDHANBNK", "FEDERALBNK",
    "AUBANK", "ABCAPITAL", "CDSL", "BSE", "MCX", "ANGELONE", "PAYTM", "PNBHOUSING",
    "MANAPPURAM", "POONAWALLA", "LTF", "DIXON", "POLYCAB", "KEI", "KAYNES", "AMBER",
    "KPITTECH", "COFORGE", "PERSISTENT", "TATAELXSI", "SONATSOFTW", "TANLA", "BSOFT",
    "CYIENT", "AFFLE", "LAURUSLABS", "GRANULES", "BIOCON", "GLENMARK", "NATCOPHARM",
    "SYNGENE", "LALPATHLAB", "DEEPAKNTR", "TATACHEM", "AARTIIND", "NAVINFLUOR", "CLEAN",
    "CHAMBLFERT", "KALYANKJIL", "HONASA", "INDHOTEL", "VOLTAS", "CROMPTON", "BATAINDIA",
    "PVRINOX", "JUBLFOOD", "DEVYANI", "TRENT", "DELHIVERY", "UNOMINDA", "SONACOMS",
    "EXIDEIND", "BHARATFORG", "OBEROIRLTY", "PRESTIGE", "ASTRAL",
]


def fetch_recent_ipo_symbols(days_back: int = 270, max_count: int = 60) -> set:
    try:
        r = requests.get("https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv", headers=HEADERS, timeout=15)
        r.raise_for_status()
        eq = pd.read_csv(io.StringIO(r.text))
        eq.columns = eq.columns.str.strip()
        eq["SYMBOL"] = eq["SYMBOL"].astype(str).str.strip()
        eq["SERIES"] = eq["SERIES"].astype(str).str.strip()
        eq["LIST_DATE"] = pd.to_datetime(eq["DATE OF LISTING"].astype(str).str.strip(), format="%d-%b-%Y", errors="coerce")
        today_ts = pd.Timestamp(now_ist().date())
        ipos = eq[
            (eq["SERIES"] == "EQ")
            & (eq["LIST_DATE"] >= today_ts - pd.Timedelta(days=days_back))
            & (eq["LIST_DATE"] <= today_ts - pd.Timedelta(days=25))
        ]
        return set(ipos.sort_values("LIST_DATE", ascending=False).head(max_count)["SYMBOL"].tolist())
    except Exception:
        return set()


def fetch_historical_results(symbols_clean: set) -> dict:
    events = {s: [] for s in symbols_clean}
    try:
        sess = cffi_requests.Session(impersonate="chrome") if HAS_CFFI else requests.Session()
        sess.get("https://www.nseindia.com", headers=HEADERS, timeout=8)
        sess.get("https://www.nseindia.com/companies-listing/corporate-filings-board-meetings", headers=HEADERS, timeout=8)
        end_dt = now_ist()
        for step in range(6):
            to_d = end_dt - timedelta(days=step * 90)
            from_d = to_d - timedelta(days=90)
            url = (
                "https://www.nseindia.com/api/corporate-board-meetings"
                f"?index=equities&from_date={from_d.strftime('%d-%m-%Y')}&to_date={to_d.strftime('%d-%m-%Y')}"
            )
            r = sess.get(
                url,
                headers={
                    **HEADERS,
                    "Accept": "application/json, text/plain, */*",
                    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-board-meetings",
                },
                timeout=10,
            )
            if r.status_code != 200 or not isinstance(r.json(), list):
                continue
            for item in r.json():
                sym = str(item.get("bm_symbol") or item.get("symbol") or "").upper().strip()
                purpose = str(item.get("bm_purpose", "") + " " + item.get("bm_desc", "")).lower()
                if sym in events and "result" in purpose:
                    mdt = pd.to_datetime(item.get("bm_date"), errors="coerce")
                    adt = pd.to_datetime(item.get("bm_timestamp"), errors="coerce")
                    if pd.notna(mdt):
                        mdt = mdt.tz_localize(None).normalize()
                        adt = adt.tz_localize(None).normalize() if pd.notna(adt) else (mdt - pd.Timedelta(days=10))
                        events[sym].append((adt, mdt))
    except Exception:
        pass
    return events


def calc_trapped_sellers(highs: np.ndarray, closes: np.ndarray, window: int = 252) -> np.ndarray:
    n = len(closes)
    out = np.full(n, np.nan)
    for i in range(5, n):
        out[i] = np.mean(highs[max(0, i - window) : i] > closes[i])
    return out


def build_features_df(df: pd.DataFrame, nifty: pd.DataFrame, sym_events: list) -> pd.DataFrame:
    d = df.copy()
    prev_c = d["Close"].shift(1).replace(0, np.nan)
    d["today_move_pct"] = (d["Close"] - prev_c) / prev_c
    vol_med = d["Volume"].rolling(15, min_periods=4).median().shift(1).replace(0, np.nan)
    d["vol_surge_med"] = d["Volume"] / vol_med
    d["flat_base_4d"] = (d["High"].rolling(4).max().shift(1) - d["Low"].rolling(4).min().shift(1)) / prev_c
    d["vol_dry_4d"] = (d["Volume"].rolling(4).median().shift(1) / vol_med).fillna(1.0)

    prev_high = d["High"].rolling(252, min_periods=5).max().shift(1).replace(0, np.nan)
    d["pct_from_high"] = (d["Close"] - prev_high) / prev_high
    d["trapped_sellers_pct"] = calc_trapped_sellers(d["High"].values, d["Close"].values)
    d["is_blue_sky"] = (
        (d["Close"] >= prev_high * 0.995) | ((d["trapped_sellers_pct"] <= 0.02) & (d["vol_surge_med"] >= 1.3))
    ).astype(int)

    rng = (d["High"] - d["Low"]).replace(0, np.nan)
    d["close_strength"] = ((d["Close"] - d["Low"]) / rng).fillna(0.5)
    d["ret_10d_prior"] = prev_c.pct_change(10).fillna(0.0)

    tr = pd.concat([d["High"] - d["Low"], (d["High"] - prev_c).abs(), (d["Low"] - prev_c).abs()], axis=1).max(axis=1)
    d["atr_pct"] = tr.rolling(14, min_periods=4).mean().shift(1) / prev_c

    d["nifty_above_20dma"] = nifty["above_20dma"].reindex(d.index, method="ffill").fillna(1)
    d["nifty_ret_5d"] = nifty["ret_5d"].reindex(d.index, method="ffill").fillna(0.0)

    pre_flag = np.zeros(len(d), dtype=int)
    for idx_i, dt in enumerate(d.index):
        for adt, mdt in sym_events:
            if adt <= dt < mdt and (mdt - dt).days <= 10:
                pre_flag[idx_i] = 1
                break
    d["is_preresults"] = pre_flag

    med_turn_cr = (d["Close"] * d["Volume"]).rolling(10, min_periods=5).median().shift(1) / 1e7
    d["is_candidate"] = (
        d["today_move_pct"].between(0.025, 0.085)
        & (d["vol_surge_med"] >= 1.5)
        & (d["Close"] >= 50)
        & (med_turn_cr >= 5.0)
    )
    return d.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)


def grade_outcome(df: pd.DataFrame, idx: int, hold_days: int = 6) -> int:
    entry = float(df.iloc[idx]["Close"])
    sl, tp = entry * 0.95, entry * 1.07
    end_idx = min(idx + hold_days, len(df) - 1)
    for j in range(idx + 1, end_idx + 1):
        if float(df.iloc[j]["Low"]) <= sl:
            return 0
        if float(df.iloc[j]["High"]) >= tp:
            return 1
    return 1 if float(df.iloc[end_idx]["Close"]) >= entry * 1.02 else 0


def make_lgbm():
    return lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.025, max_depth=3, num_leaves=7,
        min_child_samples=35, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.5, reg_lambda=3.0, random_state=42, verbose=-1,
    )


def run_self_training(nifty: pd.DataFrame, force: bool = False) -> str:
    latest_session = str(nifty.index[-1].date())
    meta = load_json(META_FILE, {})
    if not force and os.path.exists(BRAIN_FILE) and meta.get("last_trained_session") == latest_session:
        return f"Skipped retrain (already trained on latest session {latest_session})"

    print("[Stage 1/2] Retraining Challenger AI on 90 Core NSE Stocks + Recent IPOs + Live Memory...")
    ipo_syms = fetch_recent_ipo_symbols(days_back=270, max_count=60)
    all_syms = sorted(set(CORE_UNIVERSE) | ipo_syms)
    stock_map = download_in_chunks([f"{s}.NS" for s in all_syms], period="3y", chunk_size=35, min_bars=25)
    events_map = fetch_historical_results({s.replace(".NS", "") for s in stock_map.keys()})

    samples = []
    for sym_ns, df in stock_map.items():
        sym_clean = sym_ns.replace(".NS", "")
        is_ipo = sym_clean in ipo_syms
        fdf = build_features_df(df, nifty, events_map.get(sym_clean, []))
        for i in range(len(fdf) - 6):
            if bool(fdf.iloc[i]["is_candidate"]):
                rec = fdf.iloc[i][FEATURES].to_dict()
                rec["Date"] = fdf.index[i]
                rec["label"] = grade_outcome(fdf, i)
                rec["weight"] = 3.0 if rec["is_preresults"] == 1 else (2.0 if is_ipo else 1.0)
                samples.append(rec)

    if not samples:
        return "Retrain skipped (could not pull historical samples)"

    ds = pd.DataFrame(samples).sort_values("Date").reset_index(drop=True)
    dates_sorted = sorted(ds["Date"].dropna().unique())
    cut_train = dates_sorted[int(len(dates_sorted) * 0.75)]
    train_df, exam_df = ds[ds["Date"] < cut_train], ds[ds["Date"] >= cut_train]

    live_df = pd.DataFrame(columns=FEATURES + ["Date", "label", "weight"])
    if os.path.exists(MEMORY_FILE):
        mdf = pd.read_csv(MEMORY_FILE)
        if not mdf.empty and all(c in mdf.columns for c in ["EntryDate", "Label"] + FEATURES):
            mdf["Label"] = pd.to_numeric(mdf["Label"], errors="coerce")
            cm = mdf[mdf["Label"].isin([0, 1])].copy()
            for col in FEATURES:
                cm[col] = pd.to_numeric(cm[col], errors="coerce")
            cm["Date"] = pd.to_datetime(cm["EntryDate"], errors="coerce")
            cm = cm.dropna(subset=["Date", "Label"] + FEATURES)
            if not cm.empty:
                cm["label"] = cm["Label"].astype(int)
                cm["weight"] = 4.0
                live_df = cm[FEATURES + ["Date", "label", "weight"]].copy()

    exam_model = make_lgbm()
    exam_model.fit(train_df[FEATURES], train_df["label"], sample_weight=train_df["weight"])
    chal_probs = exam_model.predict_proba(exam_df[FEATURES])[:, 1]
    chal_auc = float(roc_auc_score(exam_df["label"], chal_probs)) if exam_df["label"].nunique() == 2 else 0.50
    smart_threshold = round(float(np.quantile(chal_probs, 0.55)), 3)
    ai_passed = chal_auc >= MIN_LIVE_AUC

    old_brain = load_brain()
    old_auc, old_ver, promoted = None, 0, False
    if old_brain:
        old_ver = int(old_brain.get("version", 1))
        valid_old = old_brain.get("features") == FEATURES and old_brain.get("calibrator") is None and old_brain.get("ai_active", False)
        if valid_old:
            old_auc = float(old_brain.get("auc", 0.50))
            if ai_passed and (chal_auc >= old_auc + 0.002 or len(live_df) > int(old_brain.get("live_trades_learned", 0))):
                promoted = True
        else:
            promoted = True
    else:
        promoted = True

    new_ver = (old_ver + 1) if promoted else old_ver
    if promoted:
        full_train = pd.concat([ds, live_df], ignore_index=True) if len(live_df) > 0 else ds
        final_model = make_lgbm()
        final_model.fit(full_train[FEATURES], full_train["label"], sample_weight=full_train["weight"])
        with open(BRAIN_FILE, "wb") as f:
            pickle.dump({
                "version": new_ver, "trained_on": str(now_ist().date()), "model": final_model,
                "calibrator": None, "features": FEATURES, "threshold": smart_threshold,
                "auc": round(chal_auc, 4), "ai_active": ai_passed, "live_trades_learned": len(live_df),
            }, f)
        decision = f"UPGRADED to v{new_ver} (AUC {chal_auc:.4f})" if ai_passed else f"SAVED v{new_ver} in SAFE MODE (AUC {chal_auc:.4f})"
    else:
        decision = f"KEPT Champion v{old_ver} (Old {old_auc:.4f} vs New {chal_auc:.4f})"

    evo_row = pd.DataFrame([{
        "Date": str(now_ist().date()), "Version": new_ver,
        "Old_AUC": round(old_auc, 4) if old_auc is not None else "-",
        "New_AUC": round(chal_auc, 4), "AI_Active": ai_passed,
        "Threshold": smart_threshold, "LiveTradesLearned": len(live_df), "Decision": decision,
    }])
    if os.path.exists(EVOLUTION_LOG):
        evo_row = pd.concat([pd.read_csv(EVOLUTION_LOG), evo_row], ignore_index=True)
    evo_row.to_csv(EVOLUTION_LOG, index=False)

    meta["last_trained_session"] = latest_session
    save_json(META_FILE, meta)
    print(f"    -> {decision}")
    return decision


def audit_open_positions(nifty: pd.DataFrame) -> tuple[list, dict]:
    """5-Point 'Is My Current Investment Still Worth It?' Audit for every open trade."""
    print("[Stage 2/2] Running 5-Point Holding Worth-It Audit on Open Positions...")
    state, open_df, closed_df = load_wallet()
    brain = load_brain()
    pending_exits = load_json(PENDING_EXITS_FILE, {})
    audit_map = {}
    live_prices = {}
    lines = []

    if open_df.empty:
        save_json(AUDIT_FILE, audit_map)
        return ["No open positions to audit (100% cash)."], live_prices

    latest_dt = nifty.index[-1]
    nifty_ok = int(nifty["above_20dma"].iloc[-1])
    nifty_ret5 = float(nifty["ret_5d"].iloc[-1])

    nse = NSELiveClient()
    upcoming_res = nse.get_upcoming_board_meetings(latest_dt, days_ahead=10)
    recent_ipos = fetch_recent_ipos(latest_dt)

    syms = open_df["Symbol"].astype(str).str.strip().tolist()
    charts = download_in_chunks([f"{s}.NS" for s in syms], period="1y", chunk_size=20)

    updated_rows = []
    for _, pos in open_df.iterrows():
        pdict = pos.to_dict()
        sym = str(pos["Symbol"]).strip()
        entry_p = to_float(pos["EntryPrice"])
        sl = to_float(pos["StopLoss"])
        days_held = to_int(pos["DaysHeld"])
        df = charts.get(f"{sym}.NS", pd.DataFrame())

        q = nse.get_live_quote(sym)
        ltp = q["ltp"] if q else (float(df["Close"].iloc[-1]) if not df.empty else entry_p)
        live_prices[sym] = ltp
        pnl_pct = ((ltp - entry_p) / entry_p * 100) if entry_p > 0 else 0.0

        # 1. Overnight News Red-Flag Scan
        news = scan_news(sym, use_cache=False)
        # 2. Fresh Quarterly Margin Check
        fund = inspect_company_stack(sym, use_cache=False)
        # 3. Surprise Results Tomorrow or Anchor Lock-In Collision
        res_info = upcoming_res.get(sym)
        ipo_info = recent_ipos.get(sym, {})
        surprise_res = bool(res_info and res_info.get("TradingDaysAway", 99) <= 1)
        lock_collision = "BLOCK:" in str(ipo_info.get("AnchorLock", ""))

        # 4 & 5. Fresh Champion AI Re-Score + Dead-Money / Regime Check
        ai_score = None
        if brain and not df.empty and len(df) >= 6:
            base = compute_baseline_from_df(df, latest_dt)
            if base:
                row = df.iloc[-1]
                prev_c = base["prev_close"]
                move_1d = (ltp - prev_c) / prev_c
                vol_x = float(row["Volume"]) / base["vol_med_15"]
                pct_high = (ltp - base["prev_high_252"]) / base["prev_high_252"]
                trapped = float(np.mean(base["prior_highs_252"] > ltp))
                hi_lo = float(row["High"] - row["Low"])
                feat = {
                    "today_move_pct": move_1d, "vol_surge_med": vol_x, "flat_base_4d": base["flat_4d"],
                    "vol_dry_4d": base["vol_dry_4d"], "pct_from_high": pct_high, "trapped_sellers_pct": trapped,
                    "is_blue_sky": int(ltp >= base["prev_high_252"] * 0.995 or trapped <= 0.02),
                    "close_strength": (ltp - float(row["Low"])) / hi_lo if hi_lo > 0 else 0.5,
                    "ret_10d_prior": base["ret_10d"], "atr_pct": base["atr_pct"],
                    "nifty_above_20dma": nifty_ok, "nifty_ret_5d": nifty_ret5,
                    "is_preresults": 1 if str(pos["Catalyst"]) == "Pre-Results" else 0,
                }
                if all(np.isfinite(float(feat[k])) for k in FEATURES):
                    ai_score = score_with_brain(brain, feat)

        verdict = "HOLD"
        reason = f"Thesis intact | P&L {pnl_pct:+.1f}% | Margins: {fund['MarginTrend']}"
        if ai_score is not None:
            reason += f" | AI {ai_score*100:.1f}%"

        # Strict Decision Ladder: Only force-exit on genuine thesis breaks; tighten SL on mild weakness
        if not news["NewsPass"]:
            verdict = "QUEUE 9:15 AM EXIT"
            reason = f"Thesis broken: {news['Tag']}"
            pending_exits[sym] = {"reason": reason, "queued_on": str(now_ist().date())}
        elif not fund["FundPass"]:
            verdict = "QUEUE 9:15 AM EXIT"
            reason = f"Thesis broken: {fund['FundNote']}"
            pending_exits[sym] = {"reason": reason, "queued_on": str(now_ist().date())}
        elif surprise_res:
            verdict = "QUEUE 9:15 AM EXIT"
            reason = f"Surprise results on {res_info['ResultsDate']} — exiting before event"
            pending_exits[sym] = {"reason": reason, "queued_on": str(now_ist().date())}
        elif lock_collision:
            verdict = "QUEUE 9:15 AM EXIT"
            reason = f"IPO lock-in collision: {ipo_info['AnchorLock']}"
            pending_exits[sym] = {"reason": reason, "queued_on": str(now_ist().date())}
        elif (days_held >= 3 and pnl_pct < 1.0 and nifty_ok == 0) or (ai_score is not None and ai_score < 0.35):
            new_sl = round(max(sl, ltp * 0.975), 2)
            if new_sl > sl and new_sl < ltp:
                pdict["StopLoss"] = new_sl
                verdict = "TIGHTEN SL"
                reason = f"Momentum/AI soft (AI {ai_score*100:.1f}% if scored, Nifty {'OK' if nifty_ok else 'WEAK'}) — SL raised ₹{sl:.1f}->₹{new_sl:.1f}"
            else:
                verdict = "WATCH CLOSELY"
                reason = f"Soft score ({ai_score*100:.1f}% if scored); keeping SL ₹{sl:.1f}"

        audit_map[sym] = {"verdict": verdict, "reason": reason, "audited_on": str(now_ist().date())}
        updated_rows.append(pdict)
        lines.append(f"• {sym} (₹{ltp:.1f}, {pnl_pct:+.1f}%): {verdict} — {reason}")
        print(f"    -> {lines[-1]}")

    open_df = pd.DataFrame(updated_rows, columns=list(open_df.columns))
    save_wallet(state, open_df, closed_df)
    save_json(PENDING_EXITS_FILE, pending_exits)
    save_json(AUDIT_FILE, audit_map)
    return lines, live_prices


def run_holiday_routine(force_train: bool = False):
    print("=" * 78)
    print(f"  HOLIDAY / WEEKEND BRAIN EVOLUTION + HOLDING AUDIT ({now_ist().strftime('%Y-%m-%d %H:%M IST')})")
    print("=" * 78)
    nifty = extract_ohlcv(yf.download("^NSEI", period="3y", interval="1d", auto_adjust=True, progress=False), "^NSEI")
    if nifty.empty or len(nifty) < 25:
        raise RuntimeError("Could not download Nifty 50 history.")
    nifty["above_20dma"] = (nifty["Close"] > nifty["Close"].rolling(20).mean()).astype(int)
    nifty["ret_5d"] = nifty["Close"].pct_change(5).fillna(0.0)

    train_msg = run_self_training(nifty, force=force_train)
    audit_lines, live_prices = audit_open_positions(nifty)

    nifty_c = float(nifty["Close"].iloc[-1])
    nifty_20 = float(nifty["Close"].rolling(20).mean().iloc[-1])
    nifty_str = f"Nifty 50: {nifty_c:,.1f} vs 20-DMA {nifty_20:,.1f} ({'HEALTHY' if nifty_c >= nifty_20 else 'WEAK'})"
    build_dashboard(live_prices=live_prices, nifty_str=nifty_str, status_mode="Holiday Self-Train & Audit Complete")
    git_sync(f"Holiday AI Train & Holding Audit ({now_ist().strftime('%Y-%m-%d')})")

    report = (
        f"🧠 AlgoBot Holiday Brain & Portfolio Audit ({now_ist().strftime('%d %b %Y')})\n\n"
        f"1. AI Evolution: {train_msg}\n\n"
        f"2. Holding Check:\n" + "\n".join(audit_lines)
    )
    send_telegram(report)
    print("\n" + report)


if __name__ == "__main__":
    run_holiday_routine(force_train=True)
