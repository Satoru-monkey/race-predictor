"""競馬の特徴量。馬・騎手・調教師の成績は「そのレースより前」のものだけを使う。"""
import numpy as np
import pandas as pd

RACE_COLS = ["jcd_i", "rno", "surface", "direction", "distance", "going", "class",
             "handicap", "female", "heads"]
HORSE_COLS = ["waku", "umaban", "sex", "age", "kinryo", "weight", "weight_diff"]
HIST_COLS = [
    "h_n", "h_days", "h_last_rank_r", "h_avg_rank_r3", "h_best_rank_r3", "h_last_margin",
    "h_avg_margin3", "h_last_l3f_r", "h_avg_l3f_r3", "h_last_pop", "h_avg_front3",
    "h_class_diff", "h_dist_diff", "h_surface_same_rank_r", "h_win_rate", "h_top3_rate",
    "j_win", "j_top3", "j_n", "t_win", "t_top3",
]
# スピード指数・適性・相手の強さ
FIG_COLS = [
    "h_last_fig", "h_best_fig3", "h_avg_fig3", "h_last_l3fig", "h_avg_l3fig3", "h_surface_fig",
    "h_dist_rank_r", "h_dist_n", "h_venue_rank_r", "h_max_class3", "h_last_class_perf",
]
REL_SRC = ["h_avg_rank_r3", "h_avg_margin3", "h_avg_l3f_r3", "j_win", "kinryo", "h_last_pop",
           "h_best_fig3", "h_avg_fig3", "h_avg_l3fig3", "h_surface_fig"]
HIGH_IS_GOOD = {"j_win", "h_best_fig3", "h_avg_fig3", "h_avg_l3fig3", "h_surface_fig"}
FEATURES = (RACE_COLS + HORSE_COLS + HIST_COLS + FIG_COLS
            + [f"{c}_rk" for c in REL_SRC] + [f"{c}_rel" for c in REL_SRC])
# オッズ（市場の見立て）も材料にするモデル用
MKT_COLS = ["mkt_logp", "mkt_rank", "mkt_logp_vs_h"]
FEATURES_MKT = FEATURES + MKT_COLS


def _smooth(hit, n, prior, k=20.0):
    return (hit + prior * k) / (n + k)


def add_figures(df: pd.DataFrame) -> pd.DataFrame:
    """スピード指数：走破タイムを、同じ競馬場・芝ダート・距離の基準タイム（3着以内の馬の中央値）と比べ、
    馬場状態の影響を差し引いて、1000mあたり何秒速いかにしたもの（大きいほど速い）。上がり3ハロンも同様。"""
    df = df.copy()
    key = ["jcd", "surface", "distance"]
    top = df[df["rank"] <= 3]
    par = top.groupby(key)["time"].median().rename("_par")
    par3 = top.groupby(key)["last3f"].median().rename("_par3")
    df = df.join(par, on=key).join(par3, on=key)
    per1000 = (df["time"] - df["_par"]) / df["distance"] * 1000
    goadj = per1000[df["rank"] <= 3].groupby([df["surface"], df["going"]]).median().rename("_go")
    df = df.join(goadj, on=["surface", "going"])
    df["fig"] = -(per1000 - df["_go"].fillna(0))
    l3 = df["last3f"] - df["_par3"]
    l3go = l3[df["rank"] <= 3].groupby([df["surface"], df["going"]]).median().rename("_go3")
    df = df.join(l3go, on=["surface", "going"])
    df["l3fig"] = -(l3 - df["_go3"].fillna(0))
    df.loc[df["rank"] >= 99, ["fig", "l3fig"]] = np.nan  # 競走中止などは使わない
    df["fig"] = df["fig"].clip(-5, 5)
    df["l3fig"] = df["l3fig"].clip(-5, 5)
    return df.drop(columns=["_par", "_par3", "_go", "_go3"])


def per_race_stats(df: pd.DataFrame) -> pd.DataFrame:
    """各レースの中での相対的な成績（着順の割合・勝ち馬とのタイム差・上がりの順位・スピード指数）。"""
    df = add_figures(df)
    g = df.groupby("race_id")
    df["rank_r"] = np.where(df["rank"] < 99, (df["rank"] - 1) / (df["heads"] - 1).clip(lower=1), 1.0)
    df["margin"] = (df["time"] - g["time"].transform("min")).clip(upper=5)
    df["l3f_r"] = (g["last3f"].rank(method="min") - 1) / (df["heads"] - 1).clip(lower=1)
    df["front_r"] = (df["pass_first"] - 1) / (df["heads"] - 1).clip(lower=1)
    return df


def add_history(df: pd.DataFrame) -> pd.DataFrame:
    """学習用：各馬の過去走、騎手・調教師の過去成績をつける。"""
    df = per_race_stats(df).sort_values(["date", "race_id", "umaban"]).reset_index(drop=True)
    h = df.groupby("horse_id", sort=False)
    prev = lambda c: h[c].shift(1)
    roll3 = lambda c, f="mean": h[c].transform(lambda s: getattr(s.shift(1).rolling(3, min_periods=1), f)())
    df["h_n"] = h.cumcount()
    df["h_days"] = (pd.to_datetime(df["date"]) - pd.to_datetime(prev("date"))).dt.days
    df["h_last_rank_r"] = prev("rank_r")
    df["h_avg_rank_r3"] = roll3("rank_r")
    df["h_best_rank_r3"] = roll3("rank_r", "min")
    df["h_last_margin"] = prev("margin")
    df["h_avg_margin3"] = roll3("margin")
    df["h_last_l3f_r"] = prev("l3f_r")
    df["h_avg_l3f_r3"] = roll3("l3f_r")
    df["h_last_pop"] = prev("pop")
    df["h_avg_front3"] = roll3("front_r")
    df["h_class_diff"] = df["class"] - prev("class")
    df["h_dist_diff"] = df["distance"] - prev("distance")
    df["h_last_fig"] = prev("fig")
    df["h_best_fig3"] = roll3("fig", "max")
    df["h_avg_fig3"] = roll3("fig")
    df["h_last_l3fig"] = prev("l3fig")
    df["h_avg_l3fig3"] = roll3("l3fig")
    df["h_max_class3"] = roll3("class", "max")
    df["h_last_class_perf"] = prev("class") - 3 * prev("rank_r")  # 高いクラスで上位ほど大きい
    # 同じ芝ダートでのスピード指数の平均、同じくらいの距離（200m刻み）・同じ競馬場での着順
    hsf = df.groupby(["horse_id", "surface"], sort=False)["fig"]
    df["h_surface_fig"] = hsf.transform(lambda x: x.shift(1).expanding().mean())
    df["_db"] = (df["distance"] / 200).round()
    hd = df.groupby(["horse_id", "surface", "_db"], sort=False)["rank_r"]
    df["h_dist_rank_r"] = hd.transform(lambda x: x.shift(1).expanding().mean())
    df["h_dist_n"] = hd.cumcount()
    hv = df.groupby(["horse_id", "jcd"], sort=False)["rank_r"]
    df["h_venue_rank_r"] = hv.transform(lambda x: x.shift(1).expanding().mean())
    del df["_db"]  # df を作り直さない（上の groupby が同じ df を指したままにする）
    df["_win"] = (df["rank"] == 1).astype(float)
    df["_top3"] = (df["rank"] <= 3).astype(float)
    df["h_win_rate"] = _smooth(h["_win"].cumsum() - df["_win"], df["h_n"], 0.07, k=3)
    df["h_top3_rate"] = _smooth(h["_top3"].cumsum() - df["_top3"], df["h_n"], 0.21, k=3)
    # 同じ芝・ダートでの平均着順割合
    hs = df.groupby(["horse_id", "surface"], sort=False)
    cnt = hs.cumcount()
    df["h_surface_same_rank_r"] = np.where(cnt > 0, (hs["rank_r"].cumsum() - df["rank_r"]) / cnt.replace(0, np.nan), np.nan)
    # 騎手・調教師：前日までの累計
    for key, pre in (("jockey_id", "j"), ("trainer_id", "t")):
        daily = df.groupby([key, "date"])[["_win", "_top3"]].agg(["sum", "count"])
        daily.columns = ["w", "n", "t3", "_"]
        daily = daily.drop(columns="_").reset_index().sort_values([key, "date"])
        cs = daily.groupby(key)[["w", "n", "t3"]].cumsum() - daily[["w", "n", "t3"]]
        daily[f"{pre}_win"] = _smooth(cs["w"], cs["n"], 0.07)
        daily[f"{pre}_top3"] = _smooth(cs["t3"], cs["n"], 0.21)
        daily[f"{pre}_n"] = cs["n"]
        cols = [f"{pre}_win", f"{pre}_top3"] + ([f"{pre}_n"] if pre == "j" else [])
        df = df.merge(daily[[key, "date"] + cols], on=[key, "date"], how="left")
    return df.drop(columns=["_win", "_top3"])


def add_relative(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["jcd_i"] = df["jcd"].astype(int)
    g = df.groupby("race_id")
    for c in REL_SRC:
        asc = c not in HIGH_IS_GOOD  # 小さいほど良い指標は昇順
        df[f"{c}_rk"] = g[c].rank(ascending=asc, method="min")
        df[f"{c}_rel"] = df[c] - g[c].transform("mean")
    # 単勝オッズから見た確率（オッズがあるときだけ）
    if "odds" in df:
        inv = 1.0 / df["odds"].where(df["odds"] > 0)
        p = inv / inv.groupby(df["race_id"]).transform("sum")
        df["mkt_logp"] = np.log(p.clip(lower=1e-4))
        df["mkt_rank"] = df.groupby("race_id")["odds"].rank(method="min")
        # 近走の成績のわりに人気があるか（市場の見立てと近走のズレ）
        df["mkt_logp_vs_h"] = df["mkt_logp"] - df.groupby("race_id")["mkt_logp"].transform("mean") + df["h_avg_rank_r3_rel"].fillna(0) * 2
    return df


# ------------------------------------------------------------ 当日の予想用
def latest_tables(hist: pd.DataFrame) -> dict:
    """手元の全データから、馬ごとの直近成績・騎手/調教師の成績表を作る。"""
    d = per_race_stats(hist).sort_values(["date", "race_id"])
    d["_win"] = (d["rank"] == 1).astype(float)
    d["_top3"] = (d["rank"] <= 3).astype(float)
    g = d.groupby("horse_id")
    last3 = g.tail(3)
    l3 = last3.groupby("horse_id")
    horse = pd.DataFrame({
        "h_n": g.size(), "last_date": g["date"].last(),
        "h_last_rank_r": g["rank_r"].last(), "h_avg_rank_r3": l3["rank_r"].mean(),
        "h_best_rank_r3": l3["rank_r"].min(), "h_last_margin": g["margin"].last(),
        "h_avg_margin3": l3["margin"].mean(), "h_last_l3f_r": g["l3f_r"].last(),
        "h_avg_l3f_r3": l3["l3f_r"].mean(), "h_last_pop": g["pop"].last(),
        "h_avg_front3": l3["front_r"].mean(), "last_class": g["class"].last(),
        "last_distance": g["distance"].last(),
        "h_last_fig": g["fig"].last(), "h_best_fig3": l3["fig"].max(), "h_avg_fig3": l3["fig"].mean(),
        "h_last_l3fig": g["l3fig"].last(), "h_avg_l3fig3": l3["l3fig"].mean(),
        "h_max_class3": l3["class"].max(),
        "h_last_class_perf": g["class"].last() - 3 * g["rank_r"].last(),
        "h_win_rate": _smooth(g["_win"].sum(), g.size(), 0.07, k=3),
        "h_top3_rate": _smooth(g["_top3"].sum(), g.size(), 0.21, k=3),
    }).reset_index()
    surf = d.groupby(["horse_id", "surface"]).agg(h_surface_same_rank_r=("rank_r", "mean"),
                                                  h_surface_fig=("fig", "mean")).reset_index()
    d["_db"] = (d["distance"] / 200).round()
    dist = d.groupby(["horse_id", "surface", "_db"]).agg(h_dist_rank_r=("rank_r", "mean"),
                                                         h_dist_n=("rank_r", "size")).reset_index()
    venue = d.groupby(["horse_id", "jcd"])["rank_r"].mean().rename("h_venue_rank_r").reset_index()
    out = {"horse": horse, "surface": surf, "dist": dist, "venue": venue}
    for key, pre in (("jockey_id", "j"), ("trainer_id", "t")):
        a = d.groupby(key).agg(w=("_win", "sum"), t3=("_top3", "sum"), n=("_win", "size"))
        t = pd.DataFrame({f"{pre}_win": _smooth(a["w"], a["n"], 0.07),
                          f"{pre}_top3": _smooth(a["t3"], a["n"], 0.21)})
        if pre == "j":
            t["j_n"] = a["n"]
        out[pre] = t.reset_index()
    return out


def apply_latest(df: pd.DataFrame, tables: dict) -> pd.DataFrame:
    df = df.merge(tables["horse"], on="horse_id", how="left")
    df["h_n"] = df["h_n"].fillna(0)
    df["h_days"] = (pd.to_datetime(df["date"]) - pd.to_datetime(df["last_date"])).dt.days
    df["h_class_diff"] = df["class"] - df["last_class"]
    df["h_dist_diff"] = df["distance"] - df["last_distance"]
    df = df.merge(tables["surface"], on=["horse_id", "surface"], how="left")
    if "dist" in tables:
        df["_db"] = (df["distance"] / 200).round()
        df = df.merge(tables["dist"], on=["horse_id", "surface", "_db"], how="left").drop(columns="_db")
        df["h_dist_n"] = df["h_dist_n"].fillna(0)
        df = df.merge(tables["venue"], on=["horse_id", "jcd"], how="left")
    df = df.merge(tables["j"], on="jockey_id", how="left")
    df = df.merge(tables["t"], on="trainer_id", how="left")
    return df
