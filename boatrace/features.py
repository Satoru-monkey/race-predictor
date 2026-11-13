"""予想に使う特徴量を作る。

学習時は「その日より前のデータだけ」で選手の過去成績を集計し、未来の情報が混ざらないようにする。
"""
import numpy as np
import pandas as pd

KIND_CODE = {"優勝": 7, "準優": 6, "ドリーム": 5, "特選": 4, "選抜": 4, "特賞": 3, "予選": 2, "一般": 1, "その他": 0}

BASE_COLS = [
    "boat", "class", "age", "weight", "nat_win", "nat_2", "loc_win", "loc_2",
    "motor_2", "boat_2", "series_n", "series_avg", "distance", "wind", "wave",
    "jcd_i", "kind_i", "rno",
]
REL_SRC = ["nat_win", "nat_2", "loc_win", "motor_2", "boat_2", "class", "series_avg"]
HIST_COLS = [
    "h_st_avg", "h_n", "h_win", "h_top3", "h_lane_win", "h_lane_top3", "h_lane_n",
    "v_lane_win",
]
EX_COLS = ["ex_time", "ex_diff", "ex_rank"]

FEATURES_NO_EX = BASE_COLS + [f"{c}_rel" for c in REL_SRC] + [f"{c}_rk" for c in REL_SRC] + HIST_COLS
FEATURES_EX = FEATURES_NO_EX + EX_COLS


def _smooth(hit, n, prior, k=5.0):
    return (hit + prior * k) / (n + k)


def add_basic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["jcd_i"] = df["jcd"].astype(int)
    df["kind_i"] = df["kind"].map(KIND_CODE).fillna(0).astype(int)
    for c in ("wind", "wave"):
        if c not in df:
            df[c] = np.nan
    g = df.groupby("race_id")
    for c in REL_SRC:
        df[f"{c}_rel"] = df[c] - g[c].transform("mean")
        df[f"{c}_rk"] = g[c].rank(ascending=False, method="min")
    # 展示（学習時は成績ファイルの展示タイム、当日は直前情報）
    if "ex_time" not in df:
        df["ex_time"] = np.nan
    if "st_ex" not in df:
        df["st_ex"] = np.nan  # スタート展示のST（当日のみ取得できる。学習時は欠損）
    df["ex_diff"] = df["ex_time"] - g["ex_time"].transform("min")
    df["ex_rank"] = g["ex_time"].rank(method="min")
    df["st_ex_rank"] = g["st_ex"].rank(method="min")
    return df


# ------------------------------------------------------------ 過去成績（学習用）
def add_history(df: pd.DataFrame) -> pd.DataFrame:
    """各行に「その日より前」の選手成績・場の枠別成績を付ける。"""
    df = df.copy()
    df["_win"] = (df["rank"] == 1).astype(float)
    df["_top3"] = (df["rank"] <= 3).astype(float)
    df["_st"] = df["st"]
    df["_st_n"] = df["st"].notna().astype(float)

    def cum_before(keys, cols):
        daily = df.groupby(keys + ["date"])[cols].sum().reset_index().sort_values(keys + ["date"])
        cs = daily.groupby(keys)[cols].cumsum() - daily[cols]  # 当日分を除いた累計
        out = daily[keys + ["date"]].copy()
        out[cols] = cs.values
        return out

    r = cum_before(["toban"], ["_win", "_top3", "_st", "_st_n"])
    r["h_n"] = r["_st_n"]
    one = df.assign(_one=1.0)
    n_all = one.groupby(["toban", "date"])["_one"].sum().groupby(level=0).cumsum() - one.groupby(["toban", "date"])["_one"].sum()
    r = r.merge(n_all.rename("h_runs").reset_index(), on=["toban", "date"])
    p_win, p_top3 = df["_win"].mean(), df["_top3"].mean()
    r["h_win"] = _smooth(r["_win"], r["h_runs"], p_win)
    r["h_top3"] = _smooth(r["_top3"], r["h_runs"], p_top3)
    r["h_st_avg"] = np.where(r["_st_n"] > 0, r["_st"] / r["_st_n"].replace(0, np.nan), np.nan)
    df = df.merge(r[["toban", "date", "h_st_avg", "h_n", "h_win", "h_top3"]], on=["toban", "date"], how="left")

    lane_prior = df.groupby("boat")[["_win", "_top3"]].mean()
    l = cum_before(["toban", "boat"], ["_win", "_top3"])
    ln = one.groupby(["toban", "boat", "date"])["_one"].sum()
    ln = (ln.groupby(level=[0, 1]).cumsum() - ln).rename("h_lane_n").reset_index()
    l = l.merge(ln, on=["toban", "boat", "date"])
    l["h_lane_win"] = _smooth(l["_win"], l["h_lane_n"], l["boat"].map(lane_prior["_win"]))
    l["h_lane_top3"] = _smooth(l["_top3"], l["h_lane_n"], l["boat"].map(lane_prior["_top3"]))
    df = df.merge(l[["toban", "boat", "date", "h_lane_win", "h_lane_top3", "h_lane_n"]],
                  on=["toban", "boat", "date"], how="left")

    v = cum_before(["jcd", "boat"], ["_win"])
    vn = one.groupby(["jcd", "boat", "date"])["_one"].sum()
    vn = (vn.groupby(level=[0, 1]).cumsum() - vn).rename("_vn").reset_index()
    v = v.merge(vn, on=["jcd", "boat", "date"])
    v["v_lane_win"] = _smooth(v["_win"], v["_vn"], v["boat"].map(lane_prior["_win"]), k=50)
    df = df.merge(v[["jcd", "boat", "date", "v_lane_win"]], on=["jcd", "boat", "date"], how="left")
    return df.drop(columns=["_win", "_top3", "_st", "_st_n"])


# ------------------------------------------------------------ 過去成績（当日の予想用）
def history_tables(hist: pd.DataFrame) -> dict:
    """手元の全データから、選手・枠・場ごとの最新の成績表を作る。"""
    h = hist.copy()
    h["_win"] = (h["rank"] == 1).astype(float)
    h["_top3"] = (h["rank"] <= 3).astype(float)
    p_win, p_top3 = h["_win"].mean(), h["_top3"].mean()
    lane_prior = h.groupby("boat")[["_win", "_top3"]].mean()

    r = h.groupby("toban").agg(_win=("_win", "sum"), _top3=("_top3", "sum"),
                               h_runs=("_win", "size"), h_st_avg=("st", "mean"), h_n=("st", "count"))
    r["h_win"] = _smooth(r["_win"], r["h_runs"], p_win)
    r["h_top3"] = _smooth(r["_top3"], r["h_runs"], p_top3)

    l = h.groupby(["toban", "boat"]).agg(_win=("_win", "sum"), _top3=("_top3", "sum"), h_lane_n=("_win", "size")).reset_index()
    l["h_lane_win"] = _smooth(l["_win"], l["h_lane_n"], l["boat"].map(lane_prior["_win"]))
    l["h_lane_top3"] = _smooth(l["_top3"], l["h_lane_n"], l["boat"].map(lane_prior["_top3"]))

    v = h.groupby(["jcd", "boat"]).agg(_win=("_win", "sum"), _vn=("_win", "size")).reset_index()
    v["v_lane_win"] = _smooth(v["_win"], v["_vn"], v["boat"].map(lane_prior["_win"]), k=50)
    return {
        "racer": r[["h_st_avg", "h_n", "h_win", "h_top3"]].reset_index(),
        "lane": l[["toban", "boat", "h_lane_win", "h_lane_top3", "h_lane_n"]],
        "venue": v[["jcd", "boat", "v_lane_win"]],
    }


def apply_history(df: pd.DataFrame, tables: dict) -> pd.DataFrame:
    df = df.merge(tables["racer"], on="toban", how="left")
    df = df.merge(tables["lane"], on=["toban", "boat"], how="left")
    df = df.merge(tables["venue"], on=["jcd", "boat"], how="left")
    return df
