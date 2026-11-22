import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import yfinance as yf

BULLISH_WORDS = {
    "order win": 0.45,
    "bags order": 0.45,
    "contract": 0.25,
    "profit jumps": 0.40,
    "net profit rises": 0.35,
    "strong q": 0.30,
    "expansion": 0.25,
    "upgrade": 0.30,
    "buy rating": 0.30,
    "dividend": 0.15,
    "bonus": 0.20,
}

BEARISH_WORDS = {
    "profit falls": -0.40,
    "net loss": -0.45,
    "loss widens": -0.50,
    "downgrade": -0.35,
    "weak q": -0.35,
    "sell rating": -0.35,
    "promoter sells": -0.45,
    "stake sale": -0.30,
    "penalty": -0.35,
    "tax notice": -0.30,
}

HARD_VETO_WORDS = [
    "sebi ban",
    "sebi notice",
    "fraud",
    "ed raid",
    "income tax raid",
    "default",
    "insolvency",
    "nclt",
    "auditor resigns",
    "cfo resigns",
    "pledge",
    "asm stage",
    "esm stage",
    "lower circuit",
]


def check_stock_news(symbol: str, company_name: str = "") -> dict:
    """
    Fetches latest India headlines from Google News RSS + yfinance.
    Returns sentiment score (-1.0 to +1.0), red-flag veto boolean, and top 3 headlines.
    """
    headlines = []
    query = urllib.parse.quote(f"{company_name or symbol} NSE stock")
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

    # 1. Google News RSS (catches mid/small caps like GERMAN & XTRANET)
    try:
        req = urllib.request.Request(
            rss_url, headers={"User-Agent": "Mozilla/5.0 (compatible; AlgoBot5L/1.0)"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            root = ET.fromstring(resp.read())
        for item in root.findall(".//item")[:6]:
            title = item.findtext("title")
            if title:
                headlines.append(title.strip())
    except Exception:
        pass

    # 2. Yahoo Finance backup
    try:
        yf_news = yf.Ticker(f"{symbol}.NS").news or []
        for item in yf_news[:4]:
            title = item.get("title") or item.get("content", {}).get("title")
            if title and title not in headlines:
                headlines.append(title.strip())
    except Exception:
        pass

    score = 0.0
    is_red_flag = False
    reason = ""

    for h in headlines:
        text = h.lower()
        for bad in HARD_VETO_WORDS:
            if bad in text:
                is_red_flag = True
                reason = f"Red flag '{bad}': {h}"
                break
        for word, weight in BULLISH_WORDS.items():
            if word in text:
                score += weight
        for word, weight in BEARISH_WORDS.items():
            if word in text:
                score += weight

    return {
        "symbol": symbol,
        "score": max(-1.0, min(1.0, round(score, 2))),
        "is_red_flag": is_red_flag,
        "reason": reason,
        "headlines": headlines[:3],
    }


def format_news_briefing(holdings: list, watchlist: list = None) -> str:
    """
    Builds a Telegram-ready pre-market news summary.
    Pass lists of (symbol, company_name) tuples.
    """
    lines = ["📰 *AlgoBot 5L — News Scan*"]

    for sym, name in holdings:
        res = check_stock_news(sym, name)
        status = "🚨 RED FLAG" if res["is_red_flag"] else f"Score: {res['score']:+.2f}"
        top = res["headlines"][0] if res["headlines"] else "No recent news"
        lines.append(f"• *{sym}* ({status})\n  _{top}_")

    if watchlist:
        lines.append("\n🔎 *Shortlist Check:*")
        for sym, name in watchlist[:5]:
            res = check_stock_news(sym, name)
            status = "⛔ BLOCKED" if res["is_red_flag"] else f"Score: {res['score']:+.2f}"
            lines.append(f"• *{sym}* — {status}")

    return "\n".join(lines)
