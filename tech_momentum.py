"""Technique 1: Momentum (6-month strength), monthly rebalance.
Idea: stocks that have risen the most over the past 6 months (skipping the latest month) tend to keep outperforming for a while.
"""
import numpy as np
import pandas as pd

META = {
    "id": "momentum", "name": "Momentum (6-month strength)", "hold": "M",
    "max_hold": 45, "stop_atr": 2.5,
    "rules": {
        "entry": "On the last trading day of each month, buy the 10 strongest liquid stocks by return over months 2 to 6 back, if the stock is also above its 200-day average. Entry at the next day's open.",
        "exit": "On the last trading day of each month, sell if the stock has dropped out of the top 20 or fell below its 200-day average. Also sells after 45 trading days.",
        "stop": "2.5 times the 14-day average true range below the entry price.",
    },
}


def month_end_flags(index):
    """True on the last trading day of each month. The latest day counts if the next weekday is in a new month."""
    d = pd.to_datetime(pd.Index(index))
    nxt = np.append(d[1:].month.values, (d[-1] + pd.offsets.BDay(1)).month)
    return (d.month.values != nxt)


def signals(P, cfg):
    c, liq = P["close"], P["liquid"]
    ret = c.shift(21) / c.shift(147) - 1
    elig = liq & (c > c.rolling(200, min_periods=200).mean()) & ret.notna()
    r = ret.where(elig)
    rank = r.rank(axis=1, ascending=False)
    me = pd.Series(month_end_flags(c.index), index=c.index)
    me_df = pd.DataFrame(np.repeat(me.values[:, None], c.shape[1], axis=1), index=c.index, columns=c.columns)
    return {"entry": (rank <= 10) & me_df, "score": r, "exit": (~(rank <= 20)) & me_df}
