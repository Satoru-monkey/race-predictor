"""表にしたデータを月ごとのファイルで保存・読み込みする（GitHubに保存して、自動の学び直しで使う）。

data/store/boat/2026-11.parquet、data/store/keiba/2026-11.parquet のように置く。
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "store"


def load(kind: str) -> pd.DataFrame:
    files = sorted((STORE / kind).glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def save(df: pd.DataFrame, kind: str, months: set[str] | None = None):
    """月ごとに保存する。months を渡すと、その月だけ書き直す。"""
    d = STORE / kind
    d.mkdir(parents=True, exist_ok=True)
    for month, g in df.groupby(df["date"].str[:7]):
        if months is None or month in months:
            g.reset_index(drop=True).to_parquet(d / f"{month}.parquet", index=False)


def merge(old: pd.DataFrame, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if old.empty:
        return new
    if new.empty:
        return old
    return (pd.concat([old, new], ignore_index=True)
            .drop_duplicates(keys, keep="last")
            .sort_values(["date"] + keys).reset_index(drop=True))
