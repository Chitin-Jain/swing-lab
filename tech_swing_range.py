"""Technique 4: Swing range bounce (buy near the floor of a sideways range, sell near the ceiling)."""
import numpy as np
import pandas as pd

META = {
    "id": "swing_range", "name": "Swing range bounce", "hold": "W",
    "max_hold": 15, "stop_atr": 1.5,
    "rules": {
        "entry": "Looks at the highest high and lowest low of the previous 40 trading days. Buys when a liquid stock in a sideways range (range 8% to 35% wide, 50-day average roughly flat) closes in the bottom 20% of that range, still above the old low, and closes higher than the day before. Entry at the next day's open.",
        "exit": "Sells when the close reaches the top 25% of the range. Also sells after 15 trading days.",
        "stop": "Half a 14-day average true range below the range's lowest low. A close below the range low means the range has broken, so the trade is cut.",
    },
}
LOOK = 40


def signals(P, cfg):
    c, hi_, lo_, liq, atr = P["close"], P["high"], P["low"], P["liquid"], P["atr"]
    lo = lo_.shift(1).rolling(LOOK, min_periods=LOOK).min()
    hi = hi_.shift(1).rolling(LOOK, min_periods=LOOK).max()
    width = (hi - lo) / lo
    pos = (c - lo) / (hi - lo)
    s50 = c.rolling(50, min_periods=50).mean()
    flat = (s50 / s50.shift(20) - 1).abs() <= 0.08
    base = liq & (width >= 0.08) & (width <= 0.35) & flat
    entry = base & (pos >= 0) & (pos <= 0.20) & (c > c.shift(1))
    watch = base & (pos >= 0) & (pos <= 0.25)
    return {"entry": entry.fillna(False), "score": width, "exit": (pos >= 0.75).fillna(False),
            "stop": lo - 0.5 * atr,
            "screen": {"pos": pos.where(watch), "low": lo, "high": hi}}
