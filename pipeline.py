"""Swing Lab pipeline.
fetch : download NSE daily price files (EQ series only) and store them in data/
check : run data-quality checks, write docs/data/health.json
build : compute market overview numbers, write docs/data/market.json
all   : fetch, then check, then build (build is skipped if a critical check fails)
"""
import argparse, glob, io, json, os, sys, time
from datetime import date, datetime, timedelta, timezone
import pandas as pd
import requests

CFG = json.load(open("config.json"))
DATA, OUT = "data", "docs/data"
URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d}.csv"
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
       "Accept": "text/csv,*/*", "Referer": "https://www.nseindia.com/all-reports"}
COLS = ["SYMBOL", "SERIES", "DATE1", "PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE",
        "CLOSE_PRICE", "TTL_TRD_QNTY", "TURNOVER_LACS", "NO_OF_TRADES", "DELIV_PER"]
IST = timezone(timedelta(hours=5, minutes=30))
os.makedirs(DATA, exist_ok=True); os.makedirs(OUT, exist_ok=True)


def parse(text):
    df = pd.read_csv(io.StringIO(text), skipinitialspace=True)
    df.columns = [c.strip().upper() for c in df.columns]
    miss = [c for c in COLS if c not in df.columns]
    if miss:
        raise ValueError(f"NSE file format has changed. Missing columns: {miss}")
    df = df[COLS].copy()
    for c in ("SYMBOL", "SERIES"):
        df[c] = df[c].astype(str).str.strip()
    df = df[df.SERIES.isin(CFG["series"])].copy()
    df["DATE1"] = pd.to_datetime(df["DATE1"].astype(str).str.strip(), format="%d-%b-%Y").dt.strftime("%Y-%m-%d")
    for c in COLS[3:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fetch_day(sess, d, src_dir=None):
    name = d.strftime("%d%m%Y")
    if src_dir:  # test mode: read local files instead of NSE
        p = f"{src_dir}/sec_bhavdata_full_{name}.csv"
        return parse(open(p).read()) if os.path.exists(p) else None
    last = "no response"
    for attempt in range(3):
        try:
            r = sess.get(URL.format(d=name), headers=HDR, timeout=40)
            if r.status_code == 404:
                return None  # holiday, weekend, or file not published yet
            if r.status_code == 200:
                return parse(r.text)
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = str(e)[:120]
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Could not download {d}: {last}")


def save(df):
    for yr, g in df.groupby(df.DATE1.str[:4]):
        p = f"{DATA}/prices_{yr}.csv.gz"
        old = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=COLS)
        new = pd.concat([old, g]).drop_duplicates(["DATE1", "SYMBOL"], keep="last").sort_values(["DATE1", "SYMBOL"])
        new.to_csv(p, index=False, compression="gzip")


def load(last_files=None):
    files = sorted(glob.glob(f"{DATA}/prices_*.csv.gz"))
    if not files:
        sys.exit("No price data stored yet. Run the fetch step first.")
    if last_files:
        files = files[-last_files:]
    return pd.concat(pd.read_csv(f) for f in files).sort_values(["SYMBOL", "DATE1"]).reset_index(drop=True)


def do_fetch(start, end, src_dir=None):
    today = datetime.now(IST).date()
    end = date.fromisoformat(end) if end else today
    if start:
        start = date.fromisoformat(start)
    else:
        files = sorted(glob.glob(f"{DATA}/prices_*.csv.gz"))
        start = date.fromisoformat(load(1).DATE1.max()) + timedelta(days=1) if files else today - timedelta(days=45)
    sess, frames, got, skipped = requests.Session(), [], 0, 0
    if not src_dir:
        try: sess.get("https://www.nseindia.com", headers=HDR, timeout=20)
        except requests.RequestException: pass
    d = start
    while d <= end:
        if d.weekday() < 5:
            df = fetch_day(sess, d, src_dir)
            if df is None: skipped += 1
            else: frames.append(df); got += 1
            if len(frames) >= 20:
                save(pd.concat(frames)); frames = []
            if not src_dir: time.sleep(1.2)
        d += timedelta(days=1)
    if frames:
        save(pd.concat(frames))
    print(f"Fetch done: {got} trading days saved, {skipped} weekdays with no file (holidays or not published yet).")


def adjusted(df):
    """Adds ADJ (split/bonus-adjusted close) and flags corporate-action days using PREV_CLOSE."""
    df = df.sort_values(["SYMBOL", "DATE1"]).copy()
    sess = {d: i for i, d in enumerate(sorted(df.DATE1.unique()))}
    df["S"] = df.DATE1.map(sess)
    g = df.groupby("SYMBOL")
    ratio = df.PREV_CLOSE / g.CLOSE_PRICE.shift(1)
    consecutive = (df.S - g.S.shift(1)) == 1
    df["CA"] = (consecutive & ((ratio - 1).abs() > CFG["ca_threshold"])).fillna(False)
    f = ratio.where(df.CA, 1.0).fillna(1.0)
    rev = f.groupby(df.SYMBOL).transform(lambda s: s[::-1].cumprod()[::-1])
    df["ADJ"] = df.CLOSE_PRICE * (rev / f)
    return df


def do_check(df):
    res = []
    def add(name, ok, detail, critical=True):
        res.append({"check": name, "status": "Pass" if ok else ("Fail" if critical else "Warn"), "detail": detail})
    sessions = sorted(df.DATE1.unique()); last = sessions[-1]
    age = (datetime.now(IST).date() - date.fromisoformat(last)).days
    add("Latest price date is recent", age <= 4, f"Latest data is for {last} ({age} days ago)")
    recent = sessions[-60:]
    gaps = [f"{a} to {b}" for a, b in zip(recent, recent[1:]) if (date.fromisoformat(b) - date.fromisoformat(a)).days > 5]
    add("No gaps longer than 5 days (last 60 sessions)", not gaps, "None" if not gaps else "; ".join(gaps))
    w = df[df.DATE1.isin(recent)]
    cnt = w.groupby("DATE1").size()
    if len(cnt) > 5:
        med = cnt.iloc[:-1].median()
        add("Stock count on latest day is normal", abs(cnt.iloc[-1] / med - 1) <= 0.15, f"{cnt.iloc[-1]} stocks against a typical {int(med)}")
    add("Only EQ series stocks included", set(df.SERIES.unique()) <= set(CFG["series"]), "Series found: " + ", ".join(sorted(df.SERIES.unique())))
    n = int(df.duplicated(["DATE1", "SYMBOL"]).sum())
    add("No duplicate rows", n == 0, f"{n} duplicates")
    bad = w[(w.CLOSE_PRICE <= 0) | (w.OPEN_PRICE <= 0) | w.CLOSE_PRICE.isna() | w.HIGH_PRICE.isna() | w.LOW_PRICE.isna()]
    add("No zero, negative or missing prices (last 60 sessions)", bad.empty, f"{len(bad)} bad rows" + ("" if bad.empty else ": " + ", ".join(bad.SYMBOL.unique()[:8])))
    inc = w[(w.HIGH_PRICE < w.LOW_PRICE) | (w.CLOSE_PRICE > w.HIGH_PRICE * 1.001) | (w.CLOSE_PRICE < w.LOW_PRICE * 0.999)]
    add("Close lies between day low and high", inc.empty, f"{len(inc)} rows to review", critical=False)
    day = df[df.DATE1 == last]
    mv = day[((day.CLOSE_PRICE / day.PREV_CLOSE) - 1).abs() > 0.25]
    add("No unexplained one-day moves above 25% on latest day", mv.empty, "None" if mv.empty else "Review: " + ", ".join(mv.SYMBOL.head(8)), critical=False)
    a = adjusted(w)
    ca = a[a.CA & (a.DATE1 == last)]
    add("Corporate actions detected and auto-adjusted today", True, "None today" if ca.empty else f"{len(ca)} stocks: " + ", ".join(ca.SYMBOL.head(8)))
    res.append({"check": "Closing prices match a second source", "status": "Pending", "detail": "Needs a second data source. Planned for a later phase."})
    overall = "FAIL" if any(r["status"] == "Fail" for r in res) else "PASS"
    json.dump({"as_of": last, "generated_at": datetime.now(IST).isoformat(timespec="minutes"), "overall": overall, "checks": res},
              open(f"{OUT}/health.json", "w"), indent=1)
    for r in res: print(f"[{r['status']}] {r['check']}: {r['detail']}")
    print("Overall:", overall)
    return overall == "PASS"


def do_build(df):
    df = adjusted(df)
    sessions = sorted(df.DATE1.unique()); last = sessions[-1]
    t20 = df[df.DATE1.isin(sessions[-20:])].groupby("SYMBOL").TURNOVER_LACS.mean() / 100
    liquid = set(t20[t20 >= CFG["min_daily_value_cr"]].index)
    L = df[df.SYMBOL.isin(liquid)]
    latest = L[L.DATE1 == last].set_index("SYMBOL")
    win = L[L.DATE1.isin(sessions[-252:])].groupby("SYMBOL").ADJ
    hi, lo = win.max().reindex(latest.index), win.min().reindex(latest.index)
    out = {"as_of": last, "source": "NSE daily price file (EQ series)", "generated_at": datetime.now(IST).isoformat(timespec="minutes"),
           "liquid_universe": len(liquid), "min_daily_value_cr": CFG["min_daily_value_cr"],
           "total_traded_value_cr": round(float(df[df.DATE1 == last].TURNOVER_LACS.sum() / 100), 1),
           "closing_52w_highs": int((latest.ADJ >= hi).sum()), "closing_52w_lows": int((latest.ADJ <= lo).sum()), "periods": {}}
    for key, days in {"1D": 1, "1W": 7, "1M": 30, "3M": 90, "1Y": 365}.items():
        if key == "1D" and len(sessions) > 1: base = sessions[-2]
        else:
            c = [s for s in sessions if s <= (date.fromisoformat(last) - timedelta(days=days)).isoformat()]
            if not c: continue
            base = c[-1]
        b = L[L.DATE1 == base].set_index("SYMBOL").ADJ
        pct = ((latest.ADJ / b.reindex(latest.index) - 1) * 100).dropna()
        rows = lambda s: [{"symbol": k, "close": round(float(latest.CLOSE_PRICE[k]), 2), "change_pct": round(float(v), 2),
                           "traded_cr": round(float(latest.TURNOVER_LACS[k] / 100), 1)} for k, v in s.items()]
        out["periods"][key] = {"from": base, "advances": int((pct > 0).sum()), "declines": int((pct < 0).sum()),
                               "gainers": rows(pct.nlargest(10)), "losers": rows(pct.nsmallest(10))}
    json.dump(out, open(f"{OUT}/market.json", "w"), indent=1)
    print(f"Built market.json for {last}: {len(liquid)} liquid EQ stocks.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "check", "build", "all"])
    ap.add_argument("--start"); ap.add_argument("--end"); ap.add_argument("--from-dir")
    a = ap.parse_args()
    if a.cmd in ("fetch", "all"):
        do_fetch(a.start or None, a.end or None, a.from_dir)
    if a.cmd in ("check", "build", "all"):
        data = load(3)
        ok = do_check(data) if a.cmd in ("check", "all") else True
        if a.cmd in ("build", "all"):
            if ok: do_build(data)
            else: print("Critical check failed: market.json not updated."); sys.exit(1)
        elif not ok: sys.exit(1)
