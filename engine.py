"""Swing Lab engine. Replays every technique over the stored history as a paper-trading simulation
and writes docs/data/techniques.json. Numbers come only from code and stored prices."""
import json, os
from datetime import date, datetime, timedelta, timezone
import numpy as np
import pandas as pd
import tech_momentum, tech_breakout52, tech_dma_cross, tech_swing_range

TECHS = [tech_momentum, tech_breakout52, tech_dma_cross, tech_swing_range]
IST = timezone(timedelta(hours=5, minutes=30))
DEFAULTS = dict(capital=500000, risk_pct=1.0, max_positions=10, max_position_pct=15.0,
                cost_per_side_pct=0.13, slippage_per_side_pct=0.10, min_daily_value_cr=5,
                min_trades=100, vanish_days=5)
WINDOWS = {"all": None, "3Y": 1096, "1Y": 365, "3M": 91, "1M": 30}


def make_panel(df, cfg):
    f = df.ADJ / df.CLOSE_PRICE
    d = pd.DataFrame({"D": df.DATE1.values, "S": df.SYMBOL.values,
                      "open": (df.OPEN_PRICE * f).values, "high": (df.HIGH_PRICE * f).values,
                      "low": (df.LOW_PRICE * f).values, "close": df.ADJ.values,
                      "vol": df.TTL_TRD_QNTY.values, "tv": (df.TURNOVER_LACS / 100).values})
    P = {k: d.pivot(index="D", columns="S", values=k) for k in ["open", "high", "low", "close", "vol", "tv"]}
    idx, cols = P["close"].index, P["close"].columns
    P = {k: v.reindex(index=idx, columns=cols) for k, v in P.items()}
    for k in ("open", "high", "low", "close"):
        P[k] = P[k].where(P[k] > 0)
    prev = P["close"].shift(1)
    tr = np.maximum(P["high"] - P["low"], np.maximum((P["high"] - prev).abs(), (P["low"] - prev).abs()))
    P["atr"] = tr.rolling(14, min_periods=14).mean()
    P["liquid"] = P["tv"].rolling(20, min_periods=20).mean() >= cfg["min_daily_value_cr"]
    return P


def simulate(P, entry, score, exitf, meta, cfg, stopf=None):
    """Daily paper-trading loop. Signals form at a close and trade at the next open."""
    dates, cols = list(P["close"].index), list(P["close"].columns)
    n = len(dates)
    ref = P["close"]
    O, Lw, C, A = P["open"].values, P["low"].values, P["close"].values, P["atr"].values
    E = entry.reindex_like(ref).fillna(False).values.astype(bool)
    Sc = score.reindex_like(ref).values.astype(float)
    X = (exitf.reindex_like(ref).fillna(False).values.astype(bool) if exitf is not None
         else np.zeros_like(E))
    SL = stopf.reindex_like(ref).values.astype(float) if stopf is not None else None
    cost, slip = cfg["cost_per_side_pct"] / 100, cfg["slippage_per_side_pct"] / 100
    k, maxh = meta["stop_atr"], meta["max_hold"]
    cash, pos, trades, eq = float(cfg["capital"]), {}, [], []
    pend_in, pend_out = [], set()

    def close_pos(j, px, t, why):
        nonlocal cash
        p = pos.pop(j)
        proceeds = p["qty"] * px * (1 - cost)
        cash += proceeds
        invested = p["qty"] * p["entry_px"] * (1 + cost)
        trades.append({"symbol": cols[j], "entry_date": dates[p["t"]], "exit_date": dates[t],
                       "entry_px": p["entry_px"], "exit_px": px, "qty": p["qty"],
                       "pnl": proceeds - invested, "ret_pct": (proceeds / invested - 1) * 100,
                       "days": t - p["t"], "reason": why})

    for t in range(n):
        for j in list(pend_out):                                   # 1. exit signals from yesterday
            if j in pos and np.isfinite(O[t, j]):
                close_pos(j, O[t, j] * (1 - slip), t, "Exit signal")
        eq_prev = eq[-1] if eq else cash
        for j in pend_in:                                          # 2. entries at today's open
            if len(pos) >= cfg["max_positions"]:
                break
            if j in pos or t == 0 or not np.isfinite(O[t, j]):
                continue
            if SL is not None:
                if not np.isfinite(SL[t - 1, j]):
                    continue
                lvl = SL[t - 1, j]                      # technique's own stop level (for example below a range low)
            else:
                if not np.isfinite(A[t - 1, j]):
                    continue
                lvl = None
            px = O[t, j] * (1 + slip)
            stop = lvl if lvl is not None else px - k * A[t - 1, j]
            if stop <= 0 or stop >= px:
                continue
            qty = int(min(eq_prev * cfg["risk_pct"] / 100 / (px - stop),
                          eq_prev * cfg["max_position_pct"] / 100 / px,
                          cash / (px * (1 + cost))))
            if qty < 1:
                continue
            cash -= qty * px * (1 + cost)
            pos[j] = {"t": t, "entry_px": px, "qty": qty, "stop": stop, "last": px, "miss": 0}
        for j in list(pos):                                        # 3. stop-loss using the day's low
            p = pos[j]
            if np.isfinite(Lw[t, j]) and Lw[t, j] <= p["stop"]:
                base = min(O[t, j], p["stop"]) if np.isfinite(O[t, j]) else p["stop"]
                close_pos(j, base * (1 - slip), t, "Stop-loss")
        for j in list(pos):                                        # 4. time exit, vanished stocks
            p = pos[j]
            if np.isfinite(C[t, j]):
                p["last"], p["miss"] = C[t, j], 0
            else:
                p["miss"] += 1
            if p["miss"] >= cfg["vanish_days"]:
                close_pos(j, p["last"] * (1 - slip), t, "No price for 5 days (left EQ series or suspended)")
            elif t - p["t"] >= maxh and np.isfinite(C[t, j]):
                close_pos(j, C[t, j] * (1 - slip), t, "Time exit")
        eq.append(cash + sum(p["qty"] * p["last"] for p in pos.values()))
        pend_out = {j for j in pos if X[t, j]}                     # 5. signals for tomorrow
        idx = np.where(E[t])[0]
        idx = [j for j in idx if j not in pos]
        idx.sort(key=lambda j: -(Sc[t, j] if np.isfinite(Sc[t, j]) else -np.inf))
        pend_in = idx[:cfg["max_positions"]]
    last = n - 1
    held = []
    for j, p in pos.items():
        held.append({"symbol": cols[j], "entry_date": dates[p["t"]], "entry_px": p["entry_px"], "last": p["last"],
                     "qty": p["qty"], "stop": p["stop"], "days": last - p["t"],
                     "ret_pct": (p["last"] * (1 - cost) / (p["entry_px"] * (1 + cost)) - 1) * 100,
                     "exit_planned": j in pend_out})
    eq_now = eq[-1] if eq else cfg["capital"]
    sigs = []
    for j in pend_in:
        if SL is not None:
            stop = SL[last, j]
        else:
            stop = C[last, j] - k * A[last, j] if np.isfinite(A[last, j]) else np.nan
        if not np.isfinite(stop) or stop >= C[last, j]:
            continue
        if stop <= 0:
            continue
        qty = int(min(eq_now * cfg["risk_pct"] / 100 / (C[last, j] - stop),
                      eq_now * cfg["max_position_pct"] / 100 / C[last, j]))
        sigs.append({"symbol": cols[j], "ref_close": C[last, j], "stop": stop, "qty": qty,
                     "score": float(Sc[last, j]) if np.isfinite(Sc[last, j]) else None})
    return {"equity": pd.Series(eq, index=dates), "trades": trades, "open": held, "signals": sigs}


def metrics(eq, trades, days, min_trades):
    last = eq.index[-1]
    start = None if days is None else (date.fromisoformat(last) - timedelta(days=days)).isoformat()
    e = eq if start is None else eq[eq.index >= start]
    tr = [t for t in trades if start is None or t["exit_date"] >= start]
    out = {"trades": len(tr), "low_sample": len(tr) < (min_trades if days is None else 30)}
    if len(e) < 2:
        return out
    span = (date.fromisoformat(e.index[-1]) - date.fromisoformat(e.index[0])).days
    tot = e.iloc[-1] / e.iloc[0] - 1
    out["from"] = e.index[0]
    out["return_pct"] = tot * 100
    out["cagr_pct"] = ((1 + tot) ** (365.25 / span) - 1) * 100 if span >= 365 and tot > -1 else None
    out["max_drawdown_pct"] = (e / e.cummax() - 1).min() * 100
    r = e.pct_change().dropna()
    out["sharpe"] = float(r.mean() / r.std() * np.sqrt(252)) if len(r) >= 20 and r.std() > 0 else None
    if tr:
        rets = np.array([t["ret_pct"] for t in tr])
        wins = [t["pnl"] for t in tr if t["pnl"] > 0]
        loss = [t["pnl"] for t in tr if t["pnl"] <= 0]
        out["win_rate_pct"] = float((rets > 0).mean() * 100)
        out["avg_win_pct"] = float(rets[rets > 0].mean()) if (rets > 0).any() else None
        out["avg_loss_pct"] = float(rets[rets <= 0].mean()) if (rets <= 0).any() else None
        out["profit_factor"] = (sum(wins) / -sum(loss)) if loss and sum(loss) < 0 else None
        out["expectancy_pct"] = float(rets.mean())
        out["avg_days"] = float(np.mean([t["days"] for t in tr]))
    return out


def screen_rows(sig, P):
    """Stocks sitting near the low of their range today (a watchlist, whether or not a buy signal has fired)."""
    sc = sig.get("screen")
    if not sc:
        return None
    pos = sc["pos"].iloc[-1].dropna().sort_values().head(15)
    trig = sig["entry"].iloc[-1]
    return [{"symbol": s, "close": P["close"][s].iloc[-1], "range_low": sc["low"][s].iloc[-1],
             "range_high": sc["high"][s].iloc[-1], "pos_pct": v * 100, "buy_signal": bool(trig[s])}
            for s, v in pos.items()]


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return round(float(o), 3) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run(df_adj, user_cfg=None, out_dir="docs/data", data_dir="data"):
    cfg = {**DEFAULTS, **{k: v for k, v in (user_cfg or {}).items() if k in DEFAULTS}}
    os.makedirs(out_dir, exist_ok=True)
    P = make_panel(df_adj, cfg)
    dates = list(P["close"].index)
    as_of = dates[-1]
    ret = P["close"].pct_change().clip(-0.5, 0.5).where(P["liquid"].shift(1, fill_value=False))
    bench = (1 + ret.mean(axis=1).fillna(0)).cumprod() * cfg["capital"]
    pts = sorted(set(range(0, len(dates), 5)) | {len(dates) - 1})
    result = {"as_of": as_of, "generated_at": datetime.now(IST).isoformat(timespec="minutes"),
              "config": cfg, "history_start": dates[0], "sessions": len(dates),
              "benchmark_label": "Average liquid stock (equal weight, no costs)", "techniques": []}
    log_rows = []
    for T in TECHS:
        sig = T.signals(P, cfg)
        res = simulate(P, sig["entry"], sig["score"], sig.get("exit"), T.META, cfg, sig.get("stop"))
        eq = res["equity"]
        result["techniques"].append({
            "id": T.META["id"], "name": T.META["name"], "hold": T.META["hold"], "rules": T.META["rules"],
            "total_trades": len(res["trades"]),
            "windows": {w: metrics(eq, res["trades"], d, cfg["min_trades"]) for w, d in WINDOWS.items()},
            "equity": [[dates[i], eq.iloc[i], bench.iloc[i]] for i in pts],
            "open": res["open"], "signals": res["signals"], "screen": screen_rows(sig, P),
            "closed": sorted(res["trades"], key=lambda t: t["exit_date"], reverse=True)[:150]})
        for s in res["signals"]:
            log_rows.append({"as_of": as_of, "technique": T.META["id"], "symbol": s["symbol"], "ref_close": round(s["ref_close"], 2)})
    json.dump(clean(result), open(f"{out_dir}/techniques.json", "w"), separators=(",", ":"))
    # permanent log of each day's signals, so past signals cannot be quietly rewritten
    lp = f"{data_dir}/signal_log.csv"
    old = pd.read_csv(lp) if os.path.exists(lp) else pd.DataFrame(columns=["as_of", "technique", "symbol", "ref_close"])
    new = pd.concat([old[old.as_of != as_of], pd.DataFrame(log_rows)]) if log_rows else old
    new.to_csv(lp, index=False)
    print(f"Engine done for {as_of}: " + ", ".join(f"{t['name']}: {t['total_trades']} trades" for t in result["techniques"]))
