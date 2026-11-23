"""Technique 3: 50/200-day moving-average crossover (the 'golden cross')."""
import pandas as pd

META = {
    "id": "dma_cross", "name": "50/200 DMA crossover", "hold": "M",
    "max_hold": 80, "stop_atr": 3.0,
    "rules": {
        "entry": "Buy when the 50-day average closes above the 200-day average for the first time (it was below or equal the day before) and the price is above the 50-day average. Entry at the next day's open.",
        "exit": "Sell when the 50-day average drops below the 200-day average. Also sells after 80 trading days.",
        "stop": "3 times the 14-day average true range below the entry price.",
    },
}


def signals(P, cfg):
    c, liq = P["close"], P["liquid"]
    s50 = c.rolling(50, min_periods=50).mean()
    s200 = c.rolling(200, min_periods=200).mean()
    cross = (s50 > s200) & (s50.shift(1) <= s200.shift(1))
    entry = liq & cross & (c > s50)
    return {"entry": entry.fillna(False), "score": P["tv"].rolling(20, min_periods=20).mean(),
            "exit": (s50 < s200).fillna(False)}
