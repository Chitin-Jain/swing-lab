"""Technique 2: 52-week-high breakout with volume confirmation."""
import pandas as pd

META = {
    "id": "breakout52", "name": "52-week-high breakout", "hold": "M",
    "max_hold": 30, "stop_atr": 2.0,
    "rules": {
        "entry": "Buy when a liquid stock closes above its highest close of the previous 252 trading days, on volume at least 1.5 times its 20-day average, and not more than 10% above that old high. Entry at the next day's open.",
        "exit": "Sell when the close falls below the lowest close of the previous 10 days. Also sells after 30 trading days.",
        "stop": "2 times the 14-day average true range below the entry price.",
    },
}


def signals(P, cfg):
    c, v, liq = P["close"], P["vol"], P["liquid"]
    hi = c.shift(1).rolling(252, min_periods=200).max()
    volr = v / v.shift(1).rolling(20, min_periods=20).mean()
    entry = liq & (c > hi) & (volr >= 1.5) & (c / hi - 1 <= 0.10)
    exit_ = c < c.shift(1).rolling(10, min_periods=10).min()
    return {"entry": entry.fillna(False), "score": volr, "exit": exit_.fillna(False)}
