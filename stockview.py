"""Writes docs/data/stocks.json: one compact record per liquid stock for the 'Stock check' tab.
Every number is computed from stored NSE prices; no outside or estimated data."""
import json
import numpy as np
import pandas as pd


def run(df_adj, tech_json_path="docs/data/techniques.json", out_path="docs/data/stocks.json", min_cr=5):
    C = df_adj.pivot(index="DATE1", columns="SYMBOL", values="ADJ").sort_index()
    TV = df_adj.pivot(index="DATE1", columns="SYMBOL", values="TURNOVER_LACS").sort_index() / 100
    last = C.index[-1]
    liquid = TV.tail(20).mean()
    liquid = liquid[liquid >= min_cr].index
    flags = {}
    try:
        for t in json.load(open(tech_json_path))["techniques"]:
            for s in t["signals"]:
                flags.setdefault(s["symbol"], []).append(t["name"])
    except Exception:
        pass
    out = {}
    for s in liquid:
        p = C[s].dropna()
        n = len(p)
        if n < 130 or p.index[-1] != last:
            continue
        v = p.values
        rets = np.diff(v[-61:]) / v[-61:-1]
        g = lambda k: round(float((v[-1] / v[-1 - k] - 1) * 100), 1) if n > k else None
        rec = {"px": round(float(v[-1]), 2), "r1m": g(21), "r3m": g(63), "r6m": g(126), "r1y": g(251),
               "vs50": round(float((v[-1] / v[-50:].mean() - 1) * 100), 1) if n >= 50 else None,
               "vs200": round(float((v[-1] / v[-200:].mean() - 1) * 100), 1) if n >= 200 else None,
               "off_hi": round(float((v[-1] / v[-252:].max() - 1) * 100), 1),
               "vol": round(float(rets.std() * np.sqrt(252) * 100), 0), "tv": round(float(TV[s].tail(20).mean()), 1),
               "days": n, "flags": flags.get(s, [])}
        f = (p.shift(-21) / p - 1).dropna().tail(756) * 100          # what happened over each 21-session (about 1 month) hold
        if len(f) >= 250:
            rec["m"] = {"n": int(len(f)), "med": round(float(f.median()), 1), "p10": round(float(f.quantile(.1)), 1),
                        "p90": round(float(f.quantile(.9)), 1), "worst": round(float(f.min()), 1),
                        "best": round(float(f.max()), 1), "up": round(float((f > 0).mean() * 100), 0)}
        rec["hist"] = [round(float(x), 1) for x in v[::-1][::5][:52][::-1]]   # weekly closes, last ~year
        out[s] = rec
    json.dump({"as_of": last, "stocks": out}, open(out_path, "w"), separators=(",", ":"))
    print(f"Stock check: {len(out)} stocks written.")
