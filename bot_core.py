import concurrent.futures
import html
import io
import json
import os
import pickle
import random
import subprocess
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

try:
    from curl_cffi import requests as cffi_requests
    HAS_CFFI = True
except ImportError:
    HAS_CFFI = False

# ==================== SHARED CONFIG ====================
STARTING_CASH = 500000.0
FIRST_BUY_RS = 60000.0
ADDON_BUY_RS = 40000.0
MAX_OPEN_STOCKS = 5
STOP_LOSS_PCT = 0.05
IPO_MAX_HOLD_DAYS = 6
COOLDOWN_DAYS = 12
FEE_PCT = 0.0015
MIN_LIVE_AUC = 0.52

KNOWN_HOT_QIB_IPOS = {
    "BLSPOLY": "200.25x QIB",
    "SSRETAIL": "203.61x QIB",
}

# Add upcoming NSE weekday holidays here as announced
NSE_HOLIDAYS = {
    "2026-11-24",  # Guru Nanak Jayanti
    "2026-12-25",  # Christmas
    "2027-01-26",  # Republic Day
}

IST = timezone(timedelta(hours=5, minutes=30))

FEATURES = [
    "today_move_pct", "vol_surge_med", "flat_base_4d", "vol_dry_4d",
    "pct_from_high", "trapped_sellers_pct", "is_blue_sky", "close_strength",
    "ret_10d_prior", "atr_pct", "nifty_above_20dma", "nifty_ret_5d", "is_preresults",
]

DATA_DIR = os.environ.get("ALGOBOT_DATA_DIR", "data")
DOCS_DIR = os.environ.get("ALGOBOT_DOCS_DIR", "docs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

BRAIN_FILE = os.path.join(DATA_DIR, "champion_brain.pkl")
MEMORY_FILE = os.path.join(DATA_DIR, "ai_memory_log.csv")
EVOLUTION_LOG = os.path.join(DATA_DIR, "evolution_history.csv")
WALLET_FILE = os.path.join(DATA_DIR, "wallet_state.json")
OPEN_FILE = os.path.join(DATA_DIR, "open_positions.csv")
CLOSED_FILE = os.path.join(DATA_DIR, "closed_trades.csv")
PENDING_EXITS_FILE = os.path.join(DATA_DIR, "pending_exits.json")
PENDING_CONFIRM_FILE = os.path.join(DATA_DIR, "pending_confirm.json")
AUDIT_FILE = os.path.join(DATA_DIR, "holding_audit.json")
META_FILE = os.path.join(DATA_DIR, "bot_meta.json")
DASHBOARD_FILE = os.path.join(DOCS_DIR, "index.html")

OPEN_COLS = [
    "Symbol", "Catalyst", "EntryDate", "EntryPrice", "Qty", "StopLoss",
    "Invested", "RealizedPnL", "BlueSkyAddOn", "BookedHalf", "ResultsDate",
    "MandatoryExitDate", "DaysHeld", "LastUpdatedDate", "EntryReason",
]
CLOSED_COLS = [
    "Symbol", "Catalyst", "EntryDate", "ExitDate", "ResultsDate",
    "BlueSkyAddOn", "NetPnL", "WhyItExited",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36",
    "Accept": "*/*",
}


# ==================== TIME & TYPE HELPERS ====================
def now_ist() -> datetime:
    return datetime.now(IST)


def is_trading_day(dt_obj: datetime = None) -> bool:
    t = dt_obj or now_ist()
    return t.weekday() < 5 and t.strftime("%Y-%m-%d") not in NSE_HOLIDAYS


def is_market_open_now() -> bool:
    t = now_ist()
    if not is_trading_day(t):
        return False
    mins = t.hour * 60 + t.minute
    return (9 * 60 + 15) <= mins <= (15 * 60 + 30)


def market_day_progress() -> float:
    t = now_ist()
    elapsed = (t.hour * 60 + t.minute) - (9 * 60 + 15)
    if elapsed <= 0:
        return 0.15
    return min(max(elapsed / 375.0, 0.15), 1.0)


def prev_trading_day(dt: pd.Timestamp) -> pd.Timestamp:
    d = pd.Timestamp(dt).normalize() - pd.Timedelta(days=1)
    while d.weekday() >= 5 or d.strftime("%Y-%m-%d") in NSE_HOLIDAYS:
        d -= pd.Timedelta(days=1)
    return d


def add_trading_days(dt: pd.Timestamp, n: int) -> pd.Timestamp:
    d, added = pd.Timestamp(dt).normalize(), 0
    while added < n:
        d += pd.Timedelta(days=1)
        if d.weekday() < 5 and d.strftime("%Y-%m-%d") not in NSE_HOLIDAYS:
            added += 1
    return d


def trading_days_between(d1: pd.Timestamp, d2: pd.Timestamp) -> int:
    d1, d2 = pd.Timestamp(d1).normalize(), pd.Timestamp(d2).normalize()
    if d2 <= d1:
        return 0
    cnt, cur = 0, d1
    while cur < d2:
        cur += pd.Timedelta(days=1)
        if cur.weekday() < 5 and cur.strftime("%Y-%m-%d") not in NSE_HOLIDAYS:
            cnt += 1
    return cnt


def to_bool(val) -> bool:
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    return str(val).strip().lower() in ("true", "1", "yes", "y")


def to_float(val, default: float = 0.0) -> float:
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return float(default)
        s = str(val).strip()
        return float(s) if s != "" else float(default)
    except Exception:
        return float(default)


def to_int(val, default: int = 0) -> int:
    return int(round(to_float(val, float(default))))


def load_json(path: str, default):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return default


def save_json(path: str, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ==================== TELEGRAM & GIT SYNC ====================
def send_telegram(message: str):
    """Sends alerts via Telegram using GitHub Secrets (silently skips if not set)."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(
            url,
            json={"chat_id": chat_id, "text": message[:4000], "disable_web_page_preview": True},
            timeout=10,
        )
    except Exception as e:
        print(f"[Telegram skipped: {e}]")


def git_sync(commit_msg: str):
    """Commits & pushes updated data/ and docs/ back to GitHub when running inside Actions."""
    if os.environ.get("GITHUB_ACTIONS", "").lower() != "true":
        return
    try:
        subprocess.run(["git", "config", "user.name", "AlgoBot AutoPilot"], check=False)
        subprocess.run(["git", "config", "user.email", "algobot@users.noreply.github.com"], check=False)
        subprocess.run(["git", "add", DATA_DIR, DOCS_DIR], check=False)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if diff.returncode != 0:
            subprocess.run(["git", "commit", "-m", commit_msg], check=False)
            subprocess.run(["git", "pull", "--rebase", "--autostash"], check=False)
            subprocess.run(["git", "push"], check=False)
            print(f"[Git Sync] Pushed: {commit_msg}")
    except Exception as e:
        print(f"[Git Sync skipped: {e}]")


# ==================== THREAD-SAFE DIRECT NSE LIVE CLIENT ====================
class NSELiveClient:
    def __init__(self):
        self._local = threading.local()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
        }

    def _get_session(self):
        sess = getattr(self._local, "sess", None)
        last_warm = getattr(self._local, "last_warm", 0.0)
        if sess is None:
            sess = cffi_requests.Session(impersonate="chrome") if HAS_CFFI else requests.Session()
            self._local.sess = sess
            self._warmup_session(sess)
        elif time.time() - last_warm > 240:
            self._warmup_session(sess)
        return sess

    def _warmup_session(self, sess):
        try:
            sess.get("https://www.nseindia.com", headers=self.headers, timeout=8)
            sess.get(
                "https://www.nseindia.com/companies-listing/corporate-filings-board-meetings",
                headers=self.headers, timeout=8,
            )
            self._local.last_warm = time.time()
        except Exception:
            self._local.last_warm = time.time()

    def _get_json(self, url: str, referer: str = "https://www.nseindia.com/", timeout: int = 8):
        sess = self._get_session()
        hdr = {**self.headers, "Referer": referer}
        for attempt in range(2):
            try:
                r = sess.get(url, headers=hdr, timeout=timeout)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (401, 403) and attempt == 0:
                    self._warmup_session(sess)
                    time.sleep(0.4)
            except Exception:
                if attempt == 0:
                    time.sleep(0.3)
        return None

    @staticmethod
    def _num(val, default: float = 0.0) -> float:
        if val is None or val == "" or val == "-":
            return float(default)
        try:
            return float(str(val).replace(",", "").strip())
        except Exception:
            return float(default)

    def get_nifty_live(self):
        data = self._get_json("https://www.nseindia.com/api/allIndices")
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            for row in data["data"]:
                if str(row.get("index", "")).upper() == "NIFTY 50":
                    ltp = self._num(row.get("last"), 0.0)
                    if ltp > 0:
                        return {
                            "ltp": ltp,
                            "open": self._num(row.get("open"), ltp),
                            "high": self._num(row.get("high"), ltp),
                            "low": self._num(row.get("low"), ltp),
                            "prev_close": self._num(row.get("previousClose"), ltp),
                            "source": "NSE-LIVE",
                        }
        return None

    def get_upcoming_board_meetings(self, ref_dt: pd.Timestamp, days_ahead: int = 18) -> dict:
        out = {}
        from_d = now_ist().strftime("%d-%m-%Y")
        to_d = (now_ist() + timedelta(days=days_ahead)).strftime("%d-%m-%Y")
        url = f"https://www.nseindia.com/api/corporate-board-meetings?index=equities&from_date={from_d}&to_date={to_d}"
        rows = self._get_json(
            url, referer="https://www.nseindia.com/companies-listing/corporate-filings-board-meetings", timeout=10
        )
        if not isinstance(rows, list):
            return out
        for item in rows:
            sym = str(item.get("bm_symbol") or item.get("symbol") or "").upper().strip()
            purpose = str(item.get("bm_purpose", "") + " " + item.get("bm_desc", "")).lower()
            if not sym or "result" not in purpose:
                continue
            mdt = pd.to_datetime(item.get("bm_date"), errors="coerce")
            if pd.isna(mdt):
                continue
            mdt = mdt.tz_localize(None).normalize()
            tdays = trading_days_between(ref_dt, mdt)
            if 1 <= tdays <= 12:
                if sym in out and pd.Timestamp(out[sym]["ResultsDate"]) <= mdt:
                    continue
                out[sym] = {
                    "Catalyst": "Pre-Results",
                    "ResultsDate": str(mdt.date()),
                    "ExitDate": str(prev_trading_day(mdt).date()),
                    "TradingDaysAway": tdays,
                    "AnchorLock": "Clear",
                    "QIBTag": KNOWN_HOT_QIB_IPOS.get(sym, "-"),
                }
        return out

    def get_live_quote(self, sym: str, hist_vol_med: float = 0.0):
        sym_enc = urllib.parse.quote(sym.upper().strip())
        ref = f"https://www.nseindia.com/get-quotes/equity?symbol={sym_enc}"
        url = f"https://www.nseindia.com/api/NextApi/apiClient/GetQuoteApi?functionName=getSymbolData&marketType=N&series=EQ&symbol={sym_enc}"
        payload = self._get_json(url, referer=ref)
        if not isinstance(payload, dict):
            return None
        eq_list = payload.get("equityResponse")
        if not isinstance(eq_list, list) or not eq_list:
            return None

        eq = eq_list[0] or {}
        meta = eq.get("metaData") or {}
        pinfo = eq.get("priceInfo") or {}
        obook = eq.get("orderBook") or {}
        tinfo = eq.get("tradeInfo") or (eq.get("marketDeptOrderBook") or {}).get("tradeInfo") or {}
        sec_dp = eq.get("securityWiseDP") or {}
        hl = pinfo.get("intraDayHighLow") or {}

        ltp = self._num(pinfo.get("lastPrice") or obook.get("lastPrice") or meta.get("closePrice") or meta.get("lastPrice"))
        if ltp <= 0:
            return None

        prev_c = self._num(pinfo.get("previousClose") or meta.get("previousClose"))
        open_p = self._num(pinfo.get("open") or meta.get("open"), ltp)
        high_p = self._num(hl.get("max") or meta.get("dayHigh") or pinfo.get("high"), max(ltp, open_p))
        low_p = self._num(hl.get("min") or meta.get("dayLow") or pinfo.get("low"), min(ltp, open_p))

        vol = self._num(
            tinfo.get("totalTradedVolume")
            or meta.get("totalTradedVolume")
            or pinfo.get("totalTradedVolume")
            or sec_dp.get("quantityTraded")
            or obook.get("totalTradedVolume"),
            0.0,
        )
        if 0 < vol < 5000 and hist_vol_med > 20000 and (vol * 1e5) <= (hist_vol_med * 60):
            vol *= 1e5

        return {
            "ltp": ltp,
            "open": open_p if open_p > 0 else ltp,
            "high": max(high_p, ltp),
            "low": min(low_p, ltp) if low_p > 0 else ltp,
            "prev_close": prev_c,
            "volume": vol,
            "source": "NSE-LIVE",
        }


# ==================== CHART, WALLET & FUNDAMENTAL HELPERS ====================
def extract_ohlcv(raw_df: pd.DataFrame, sym_ns: str) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    if isinstance(raw_df.columns, pd.MultiIndex):
        if sym_ns in raw_df.columns.get_level_values(0):
            df = raw_df[sym_ns].copy()
        elif sym_ns in raw_df.columns.get_level_values(1):
            df = raw_df.xs(sym_ns, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        df = raw_df.copy()
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if len(cols) < 5:
        return pd.DataFrame()
    df = df[cols].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


def download_in_chunks(tickers: list, period: str = "1y", chunk_size: int = 40, min_bars: int = 1) -> dict:
    out = {}
    unique_t = list(dict.fromkeys(tickers))
    for i in range(0, len(unique_t), chunk_size):
        chunk = unique_t[i : i + chunk_size]
        try:
            raw = yf.download(
                chunk, period=period, interval="1d", auto_adjust=True,
                group_by="ticker", threads=True, progress=False,
            )
            for sym_ns in chunk:
                df = extract_ohlcv(raw, sym_ns)
                if len(df) >= min_bars:
                    out[sym_ns] = df
        except Exception:
            pass
    return out


def compute_baseline_from_df(df: pd.DataFrame, cutoff_dt: pd.Timestamp):
    hist = df[df.index < cutoff_dt]
    if len(hist) < 5:
        return None
    prev_c = float(hist["Close"].iloc[-1])
    vol_med_15 = float(hist["Volume"].tail(15).median())
    if prev_c <= 0 or vol_med_15 <= 0:
        return None

    med_turn_cr = float((hist["Close"] * hist["Volume"]).tail(10).median()) / 1e7
    prior_4 = hist.tail(4)
    flat_4d = (float(prior_4["High"].max()) - float(prior_4["Low"].min())) / prev_c
    vol_dry_4d = float(prior_4["Volume"].median()) / vol_med_15

    prior_highs_252 = hist["High"].tail(252).values
    prev_high_252 = float(np.max(prior_highs_252))
    if prev_high_252 <= 0:
        return None

    base_10_c = float(hist["Close"].iloc[-11]) if len(hist) >= 11 else float(hist["Close"].iloc[0])
    ret_10d = ((prev_c / base_10_c) - 1) if base_10_c > 0 else 0.0

    tr_tail = pd.concat([
        hist["High"] - hist["Low"],
        (hist["High"] - hist["Close"].shift(1)).abs(),
        (hist["Low"] - hist["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1).tail(14)
    atr_pct = float(tr_tail.mean()) / prev_c
    if not np.isfinite(atr_pct):
        atr_pct = 0.03

    after_hours_fb = None
    if cutoff_dt in df.index:
        fb_row = df.loc[cutoff_dt]
        after_hours_fb = {
            "ltp": float(fb_row["Close"]), "open": float(fb_row["Open"]),
            "high": float(fb_row["High"]), "low": float(fb_row["Low"]),
            "prev_close": prev_c, "volume": float(fb_row["Volume"]), "source": "YF-CLOSED",
        }

    return {
        "prev_close": prev_c, "vol_med_15": vol_med_15, "med_turn_cr": med_turn_cr,
        "flat_4d": flat_4d, "vol_dry_4d": vol_dry_4d, "prev_high_252": prev_high_252,
        "prior_highs_252": prior_highs_252, "ret_10d": ret_10d, "atr_pct": atr_pct,
        "after_hours_fb": after_hours_fb,
    }


def fetch_recent_ipos(ref_dt: pd.Timestamp) -> dict:
    ipos = {}
    try:
        r = requests.get("https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv", headers=HEADERS, timeout=15)
        r.raise_for_status()
        eq = pd.read_csv(io.StringIO(r.text))
        eq.columns = eq.columns.str.strip()
        eq["SYMBOL"] = eq["SYMBOL"].astype(str).str.strip()
        eq["SERIES"] = eq["SERIES"].astype(str).str.strip()
        eq["LIST_DATE"] = pd.to_datetime(eq["DATE OF LISTING"].astype(str).str.strip(), format="%d-%b-%Y", errors="coerce")
        recent = eq[(eq["SERIES"] == "EQ") & (eq["LIST_DATE"] >= ref_dt - pd.Timedelta(days=180))]
        for _, m in recent.iterrows():
            sym = m["SYMBOL"]
            dl = int((ref_dt - m["LIST_DATE"]).days)
            d30, d90 = 28 - dl, 88 - dl
            lock = (
                f"BLOCK: 30d Unlock ({d30:+d}d)" if -3 <= d30 <= 5
                else (f"BLOCK: 90d Unlock ({d90:+d}d)" if -3 <= d90 <= 7 else "Clear")
            )
            ipos[sym] = {
                "Catalyst": "Recent-IPO", "ResultsDate": "-",
                "ExitDate": str(add_trading_days(ref_dt, IPO_MAX_HOLD_DAYS).date()),
                "AnchorLock": lock, "QIBTag": KNOWN_HOT_QIB_IPOS.get(sym, "-"),
            }
    except Exception as e:
        print(f"    (EQUITY_L skipped: {e})")
    return ipos


def append_row(df: pd.DataFrame, row_dict: dict, cols: list) -> pd.DataFrame:
    row_df = pd.DataFrame([row_dict], columns=cols)
    return row_df if df.empty else pd.concat([df, row_df], ignore_index=True)[cols]


def load_wallet():
    state = load_json(WALLET_FILE, {"starting_cash": STARTING_CASH, "cash": STARTING_CASH})
    open_df = pd.read_csv(OPEN_FILE) if os.path.exists(OPEN_FILE) else pd.DataFrame(columns=OPEN_COLS)
    closed_df = pd.read_csv(CLOSED_FILE) if os.path.exists(CLOSED_FILE) else pd.DataFrame(columns=CLOSED_COLS)

    num_defaults = {"EntryPrice": 0.0, "Qty": 0, "StopLoss": 0.0, "Invested": 0.0, "RealizedPnL": 0.0, "DaysHeld": 0}
    bool_defaults = {"BlueSkyAddOn": False, "BookedHalf": False}
    for c in OPEN_COLS:
        if c not in open_df.columns:
            open_df[c] = num_defaults.get(c, bool_defaults.get(c, ""))
    for c in CLOSED_COLS:
        if c not in closed_df.columns:
            closed_df[c] = 0.0 if c == "NetPnL" else (False if c == "BlueSkyAddOn" else "")
    return state, open_df[OPEN_COLS], closed_df[CLOSED_COLS]


def save_wallet(state, open_df, closed_df):
    save_json(WALLET_FILE, state)
    open_df.to_csv(OPEN_FILE, index=False)
    closed_df.to_csv(CLOSED_FILE, index=False)


def log_ai_memory_entry(sym: str, entry_date: str, feat_dict: dict):
    cols = ["Symbol", "EntryDate", "Label"] + FEATURES
    row = {"Symbol": sym, "EntryDate": entry_date, "Label": -1}
    for k in FEATURES:
        row[k] = round(float(feat_dict[k]), 6)
    mdf = pd.read_csv(MEMORY_FILE) if os.path.exists(MEMORY_FILE) else pd.DataFrame(columns=cols)
    if not mdf.empty and ((mdf["Symbol"].astype(str) == str(sym)) & (mdf["EntryDate"].astype(str) == str(entry_date))).any():
        return
    append_row(mdf, row, cols).to_csv(MEMORY_FILE, index=False)


def stamp_ai_memory_result(sym: str, entry_date: str, net_pnl: float):
    if not os.path.exists(MEMORY_FILE):
        return
    mdf = pd.read_csv(MEMORY_FILE)
    mask = (mdf["Symbol"].astype(str) == str(sym)) & (mdf["EntryDate"].astype(str) == str(entry_date)) & (mdf["Label"] == -1)
    if mask.any():
        mdf.loc[mask, "Label"] = 1 if net_pnl > 0 else 0
        mdf.to_csv(MEMORY_FILE, index=False)


def load_brain():
    if not os.path.exists(BRAIN_FILE):
        return None
    with open(BRAIN_FILE, "rb") as f:
        return pickle.load(f)


def score_with_brain(brain: dict, feat_dict: dict) -> float:
    feats = brain["features"]
    row_df = pd.DataFrame([{k: float(feat_dict[k]) for k in feats}], columns=feats)
    return float(brain["model"].predict_proba(row_df)[0, 1])


FUND_CACHE, NEWS_CACHE = {}, {}


def inspect_company_stack(sym: str, use_cache: bool = True) -> dict:
    if use_cache and sym in FUND_CACHE:
        return FUND_CACHE[sym]
    out = {"Promoter%": "-", "MarginTrend": "N/A", "FundPass": True, "FundNote": "Clean"}
    try:
        tk = yf.Ticker(f"{sym}.NS")
        info = tk.info or {}
        if info.get("heldPercentInsiders") is not None:
            out["Promoter%"] = f"{info['heldPercentInsiders'] * 100:.1f}%"
        q_inc = tk.quarterly_income_stmt
        if q_inc is not None and not q_inc.empty and q_inc.shape[1] >= 2:
            rev_row = next((r for r in ["Total Revenue", "Operating Revenue"] if r in q_inc.index), None)
            op_row = next((r for r in ["Operating Income", "EBIT", "EBITDA"] if r in q_inc.index), None)
            if rev_row and op_row:
                r0, r1 = float(q_inc.loc[rev_row].iloc[0]), float(q_inc.loc[rev_row].iloc[1])
                o0, o1 = float(q_inc.loc[op_row].iloc[0]), float(q_inc.loc[op_row].iloc[1])
                if pd.notna(r0) and pd.notna(r1) and r0 > 0 and r1 > 0 and pd.notna(o0) and pd.notna(o1):
                    m0, m1 = (o0 / r0) * 100, (o1 / r1) * 100
                    diff = m0 - m1
                    rel_drop = (diff / abs(m1)) if m1 != 0 else 0.0
                    if diff >= 0.75:
                        out["MarginTrend"] = f"Expanding ({diff:+.1f}%)"
                    elif (m0 < 0 and m1 > 0) or (diff <= -2.0 and rel_drop <= -0.12):
                        out["MarginTrend"] = f"Shrinking ({diff:+.1f}%)"
                        out["FundPass"] = False
                        out["FundNote"] = f"Blocked: Margins shrinking ({m1:.1f}%->{m0:.1f}%)"
                    else:
                        out["MarginTrend"] = f"Steady ({diff:+.1f}%)"
    except Exception:
        pass
    FUND_CACHE[sym] = out
    return out


def scan_news(sym: str, use_cache: bool = True) -> dict:
    if use_cache:
        cached = NEWS_CACHE.get(sym)
        if cached and (time.time() - cached[0] < 1800):
            return cached[1]
    red_words = ["sebi", "fraud", "raid", "resign", "downgrade", "penalty", "default", "pledge", "slumps"]
    pos_words = ["goldman", "morgan stanley", "jefferies", "motilal", "icici", "target", "upgrade", "order", "bagged"]
    res = {"NewsPass": True, "Tag": "-"}
    try:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(sym + ' NSE share')}&hl=en-IN&gl=IN&ceid=IN:en"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        if resp.status_code == 200:
            items = ET.fromstring(resp.content).findall(".//item")[:5]
            for it in items:
                tl = it.findtext("title", "").lower()
                for rw in red_words:
                    if rw in tl:
                        res = {"NewsPass": False, "Tag": f"RED FLAG: {rw.upper()}"}
                        NEWS_CACHE[sym] = (time.time(), res)
                        return res
                for pw in pos_words:
                    if pw in tl:
                        res["Tag"] = f"News: {pw.title()}"
    except Exception:
        pass
    NEWS_CACHE[sym] = (time.time(), res)
    return res


def fetch_quotes_batch(nse: NSELiveClient, sym_list: list, baselines: dict, allow_closed_fallback: bool) -> dict:
    def _one(sym):
        time.sleep(random.uniform(0.05, 0.18))
        b = baselines.get(sym)
        hv = b["vol_med_15"] if b else 0.0
        q = nse.get_live_quote(sym, hv)
        if q is None and allow_closed_fallback and b is not None and b.get("after_hours_fb") is not None:
            q = b["after_hours_fb"]
        return sym, q

    quotes = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for sym, q in ex.map(_one, sym_list):
            if q is not None:
                quotes[sym] = q
    return quotes


# ==================== AUTO-GENERATED GITHUB PAGES WEB DASHBOARD ====================
def build_dashboard(live_prices: dict = None, nifty_str: str = "Nifty 50 — Standby", status_mode: str = "Active"):
    state, open_df, closed_df = load_wallet()
    brain = load_brain() or {}
    audit_map = load_json(AUDIT_FILE, {})
    pending_exits = load_json(PENDING_EXITS_FILE, {})
    live_prices = live_prices or {}

    cash = to_float(state.get("cash"), STARTING_CASH)
    start_cash = to_float(state.get("starting_cash"), STARTING_CASH)
    open_val = 0.0
    open_rows_html = ""

    for _, pos in open_df.iterrows():
        sym = str(pos["Symbol"]).strip()
        entry_p = to_float(pos["EntryPrice"])
        qty = to_int(pos["Qty"])
        sl = to_float(pos["StopLoss"])
        ltp = to_float(live_prices.get(sym), entry_p)
        val = ltp * qty
        open_val += val
        unr_pnl = (ltp - entry_p) * qty
        unr_pct = ((ltp - entry_p) / entry_p * 100) if entry_p > 0 else 0.0
        sl_dist = ((ltp - sl) / ltp * 100) if ltp > 0 else 0.0
        pnl_cls = "pos" if unr_pnl >= 0 else "neg"

        aud = audit_map.get(sym, {})
        verdict = "EXIT QUEUED (9:15)" if sym in pending_exits else aud.get("verdict", "HOLD")
        badge_cls = "bad" if "EXIT" in verdict else ("warn" if "TIGHTEN" in verdict else "ok")
        reason_tip = html.escape(str(aud.get("reason", pos.get("EntryReason", ""))))

        open_rows_html += f"""
        <tr>
          <td><strong>{html.escape(sym)}</strong><div class="sub">{html.escape(str(pos['Catalyst']))}</div></td>
          <td>{qty}</td>
          <td>₹{entry_p:,.2f}</td>
          <td><strong>₹{ltp:,.2f}</strong></td>
          <td class="{pnl_cls}">₹{unr_pnl:+,.0f} ({unr_pct:+.1f}%)</td>
          <td>₹{sl:,.2f}<div class="sub">{sl_dist:.1f}% buffer</div></td>
          <td>{html.escape(str(pos['MandatoryExitDate']))}</td>
          <td><span class="badge {badge_cls}" title="{reason_tip}">{html.escape(verdict)}</span></td>
        </tr>"""

    if not open_rows_html:
        open_rows_html = '<tr><td colspan="8" class="empty">100% Virtual Cash — Waiting for next high-conviction breakout</td></tr>'

    total_eq = cash + open_val
    net_pnl = total_eq - start_cash
    net_pct = (net_pnl / start_cash) * 100 if start_cash > 0 else 0.0
    pnl_cls = "pos" if net_pnl >= 0 else "neg"

    # Closed trades stats
    win_rate_str = "-"
    closed_rows_html = ""
    if not closed_df.empty:
        pnls = pd.to_numeric(closed_df["NetPnL"], errors="coerce").fillna(0.0)
        wins = int((pnls > 0).sum())
        win_rate_str = f"{(wins / len(pnls)) * 100:.0f}% ({wins}/{len(pnls)})"
        for _, cr in closed_df.tail(15).iloc[::-1].iterrows():
            cpnl = to_float(cr["NetPnL"])
            ccls = "pos" if cpnl >= 0 else "neg"
            closed_rows_html += f"""
            <tr>
              <td><strong>{html.escape(str(cr['Symbol']))}</strong></td>
              <td>{html.escape(str(cr['EntryDate']))} → {html.escape(str(cr['ExitDate']))}</td>
              <td class="{ccls}">₹{cpnl:+,.0f}</td>
              <td>{html.escape(str(cr['WhyItExited']))}</td>
            </tr>"""
    else:
        closed_rows_html = '<tr><td colspan="4" class="empty">No closed trades yet.</td></tr>'

    # AI Evolution History
    evo_rows_html = ""
    if os.path.exists(EVOLUTION_LOG):
        edf = pd.read_csv(EVOLUTION_LOG).tail(8).iloc[::-1]
        for _, er in edf.iterrows():
            evo_rows_html += f"""
            <tr>
              <td>{html.escape(str(er.get('Date', '-')))}</td>
              <td>v{html.escape(str(er.get('Version', '-')))}</td>
              <td>{html.escape(str(er.get('New_AUC', '-')))}</td>
              <td>{html.escape(str(er.get('LiveTradesLearned', '0')))}</td>
              <td>{html.escape(str(er.get('Decision', '-')))}</td>
            </tr>"""
    else:
        evo_rows_html = '<tr><td colspan="5" class="empty">Waiting for first scheduled self-training run.</td></tr>'

    ai_ver = brain.get("version", "-")
    ai_auc = brain.get("auc", 0.0)
    ai_gate = brain.get("threshold", 0.45) * 100
    ai_state = "LIVE AI ACTIVE" if brain.get("ai_active", False) else "SAFE RULES MODE"
    updated_at = now_ist().strftime("%d %b %Y, %I:%M %p IST")

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AlgoBot ₹5L Live Portfolio</title>
<style>
:root{{--bg:#0b1020;--card:#131a30;--border:#232f52;--text:#e8eefc;--muted:#8fa0c9;--pos:#22c55e;--neg:#ef4444;--warn:#f59e0b;--accent:#3b82f6;}}
*{{box-sizing:border-box;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;}}
body{{margin:0;background:var(--bg);color:var(--text);padding:20px;}}
.wrap{{max-width:1100px;margin:0 auto;}}
header{{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:12px;margin-bottom:20px;}}
h1{{margin:0;font-size:22px;}} .meta{{color:var(--muted);font-size:13px;}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-bottom:22px;}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px;}}
.label{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.5px;}}
.val{{font-size:24px;font-weight:700;margin-top:6px;}} .sub{{font-size:12px;color:var(--muted);margin-top:3px;}}
.pos{{color:var(--pos);}} .neg{{color:var(--neg);}}
h2{{font-size:16px;margin:22px 0 10px;color:#c7d5f7;}}
.table-wrap{{overflow-x:auto;background:var(--card);border:1px solid var(--border);border-radius:12px;}}
table{{width:100%;border-collapse:collapse;font-size:14px;}}
th,td{{padding:12px 14px;text-align:left;border-bottom:1px solid var(--border);}}
th{{color:var(--muted);font-size:12px;text-transform:uppercase;}}
tr:last-child td{{border-bottom:none;}}
.empty{{text-align:center;color:var(--muted);padding:22px;}}
.badge{{display:inline-block;padding:4px 9px;border-radius:999px;font-size:11px;font-weight:600;}}
.badge.ok{{background:rgba(34,197,94,.16);color:#4ade80;}}
.badge.warn{{background:rgba(245,158,11,.18);color:#fbbf24;}}
.badge.bad{{background:rgba(239,68,68,.18);color:#f87171;}}
</style></head><body><div class="wrap">
<header>
  <div><h1>AlgoBot ₹5,00,000 Paper Portfolio</h1>
  <div class="meta">{html.escape(nifty_str)} &bull; Mode: {html.escape(status_mode)}</div></div>
  <div class="meta">Last updated: {updated_at}</div>
</header>
<div class="grid">
  <div class="card"><div class="label">Total Equity</div><div class="val">₹{total_eq:,.0f}</div>
    <div class="sub {pnl_cls}">Net P&amp;L: ₹{net_pnl:+,.0f} ({net_pct:+.2f}%)</div></div>
  <div class="card"><div class="label">Invested Value</div><div class="val">₹{open_val:,.0f}</div>
    <div class="sub">{len(open_df)}/{MAX_OPEN_STOCKS} Open Positions</div></div>
  <div class="card"><div class="label">Available Cash</div><div class="val">₹{cash:,.0f}</div>
    <div class="sub">{MAX_OPEN_STOCKS - len(open_df)} Slots Ready</div></div>
  <div class="card"><div class="label">Champion AI Brain</div><div class="val">v{ai_ver} (AUC {ai_auc:.3f})</div>
    <div class="sub">{ai_state} &bull; Gate &ge; {ai_gate:.1f}% &bull; Win Rate: {win_rate_str}</div></div>
</div>
<h2>Open Positions &amp; Latest Holding Audit</h2>
<div class="table-wrap"><table><thead><tr>
  <th>Stock</th><th>Qty</th><th>Avg Entry</th><th>Live Price</th><th>Unrealized P&amp;L</th><th>Stop-Loss</th><th>Exit By</th><th>Audit Status</th>
</tr></thead><tbody>{open_rows_html}</tbody></table></div>
<h2>Recent Closed Trades</h2>
<div class="table-wrap"><table><thead><tr>
  <th>Stock</th><th>Dates</th><th>Net P&amp;L</th><th>Exit Reason</th>
</tr></thead><tbody>{closed_rows_html}</tbody></table></div>
<h2>AI Self-Evolution History</h2>
<div class="table-wrap"><table><thead><tr>
  <th>Date</th><th>Version</th><th>Exam AUC</th><th>Live Trades Learned</th><th>Decision</th>
</tr></thead><tbody>{evo_rows_html}</tbody></table></div>
</div></body></html>"""

    with open(DASHBOARD_FILE, "w", encoding="utf-8") as f:
        f.write(page)
