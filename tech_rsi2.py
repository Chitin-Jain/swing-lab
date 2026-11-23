"""Technique 5: Oversold bounce in an uptrend (2-day RSI)."""
import numpy as np
import pandas as pd
from regime import market_up

META = {"id": "rsi2_bounce", "name": "Oversold bounce (RSI-2)", "hold": "W", "max_hold": 7, "stop_atr": 3.0,
        "rules": {
            "entry": "Liquid stock closing above its 200-day average (long-term uptrend) while its 2-day RSI is below 10, meaning it has dropped sharply over the last couple of days. Only taken while the average liquid stock is above its own 100-day average (a healthy market). Entry at the next day's open. If many qualify, the most oversold are taken first.",
            "exit": "Sells when the close is back above the 5-day average (the bounce has happened). Also sells after 7 trading days.",
            "stop": "3 times the 14-day average true range below the entry price."}}


def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def signals(P, cfg):
    c, liq = P["close"], P["liquid"]
    r2 = rsi(c, 2)
    s200, s5 = c.rolling(200, min_periods=200).mean(), c.rolling(5, min_periods=5).mean()
    entry = liq & (c > s200) & (r2 < 10)
    entry = entry.mul(market_up(P), axis=0).astype(bool)
    return {"entry": entry.fillna(False), "score": -r2, "exit": (c > s5).fillna(False)}
