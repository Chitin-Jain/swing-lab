"""Technique 8: Three-day sell-off bounce in an uptrend."""
import numpy as np
import pandas as pd
from regime import market_up

META = {"id": "down_streak", "name": "Three-day sell-off bounce", "hold": "W", "max_hold": 6, "stop_atr": 3.0,
        "rules": {
            "entry": "Liquid stock above its 200-day average that has closed lower three days in a row and lost at least 6% over those three days. Only taken while the average liquid stock is above its own 100-day average (a healthy market). Entry at the next day's open. Biggest drops first.",
            "exit": "Sells when the close is back above the 5-day average. Also sells after 6 trading days.",
            "stop": "3 times the 14-day average true range below the entry price."}}


def signals(P, cfg):
    c, liq = P["close"], P["liquid"]
    s200, s5 = c.rolling(200, min_periods=200).mean(), c.rolling(5, min_periods=5).mean()
    down3 = (c < c.shift(1)) & (c.shift(1) < c.shift(2)) & (c.shift(2) < c.shift(3))
    drop = c / c.shift(3) - 1
    entry = liq & (c > s200) & down3 & (drop <= -0.06)
    entry = entry.mul(market_up(P), axis=0).astype(bool)
    return {"entry": entry.fillna(False), "score": -drop, "exit": (c > s5).fillna(False)}
