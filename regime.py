"""Market-trend check shared by techniques: is the average liquid stock above its own 100-day average?"""
import pandas as pd


def market_up(P, n=100):
    ret = P["close"].pct_change().clip(-0.5, 0.5).where(P["liquid"].shift(1, fill_value=False)).mean(axis=1).fillna(0)
    idx = (1 + ret).cumprod()
    return (idx > idx.rolling(n, min_periods=n).mean())
