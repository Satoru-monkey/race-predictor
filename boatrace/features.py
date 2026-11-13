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
# 直近の調子（選手・モーター）とフライング
FORM_COLS = ["f_rank10", "f_st_std20", "f_fly180", "m_top3_20", "m_ex_rel20", "m_n20",
             "f_rank10_rk", "m_top3_20_rk", "m_ex_rel20_rk"]
# 進入コース（学習時は実際の進入、当日はスタート展示の進入）
COURSE_COLS = ["course_in", "front", "h_course_win", "h_course_top3"]

FEATURES_NO_EX = BASE_COLS + [f"{c}_rel" for c in REL_SRC] + [f"{c}_rk" for c in REL_SRC] + HIST_COLS + FORM_COLS
FEATURES_EX = FEATURES_NO_EX + EX_COLS + COURSE_COLS


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
    for c in ("f_rank10", "m_top3_20", "m_ex_rel20", "course_in"):
        if c not in df:
            df[c] = np.nan
    df["f_rank10_rk"] = g["f_rank10"].rank(method="min")
    df["m_top3_20_rk"] = g["m_top3_20"].rank(ascending=False, method="min")
    df["m_ex_rel20_rk"] = g["m_ex_rel20"].rank(method="min")
    df["front"] = df["boat"] - df["course_in"]  # プラスなら枠より内のコースに入った（前付け）
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


# ------------------------------------------------------------ 直近の調子・フライング・コース別（学習用）
def _ex_rel(df: pd.DataFrame) -> pd.Series:
    """展示タイムがそのレースの平均よりどれだけ速いか（マイナスほど速い）。"""
    return df["ex_time"] - df.groupby("race_id")["ex_time"].transform("mean")


def add_form(df: pd.DataFrame) -> pd.DataFrame:
    """選手の直近10走・スタートのばらつき・直近180日のフライング、モーターの直近20走、
    選手のコース別成績を付ける。どれも「その日より前」のレースだけで計算する。"""
    df = df.copy()
    df["_rank"] = df["rank"].clip(upper=6.5)
    df["_top3"] = (df["rank"] <= 3).astype(float)
    df["_win"] = (df["rank"] == 1).astype(float)
    df["_fly"] = df["flag"].astype(str).str.startswith("F").astype(float)
    df["_exrel"] = _ex_rel(df)
    df = df.sort_values(["date", "jcd", "rno", "boat"]).reset_index(drop=True)

    def recent(keys, col, n, func, minp):
        g = df.groupby(keys, sort=False)[col]
        v = g.transform(lambda x: getattr(x.shift(1).rolling(n, min_periods=minp), func)())
        # 同じ日のうちに走った分は使わない（当日の予想と条件をそろえる）
        return v.groupby([df[k] for k in keys] + [df["date"]]).transform("first")

    df["f_rank10"] = recent(["toban"], "_rank", 10, "mean", 3)
    df["f_st_std20"] = recent(["toban"], "st", 20, "std", 5)
    df["m_top3_20"] = recent(["jcd", "motor_no"], "_top3", 20, "mean", 3)
    df["m_ex_rel20"] = recent(["jcd", "motor_no"], "_exrel", 20, "mean", 3)
    df["m_n20"] = df.groupby(["jcd", "motor_no"]).cumcount().clip(upper=20)
    df["m_n20"] = df["m_n20"].groupby([df["jcd"], df["motor_no"], df["date"]]).transform("first")

    # 直近180日のフライング回数（前日まで）
    daily = df.groupby(["toban", "date"])["_fly"].sum().reset_index()
    daily["_d"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["toban", "_d"]).set_index("_d")
    daily["f_fly180"] = (daily.groupby("toban")["_fly"]
                         .rolling("180D", closed="left").sum().reset_index(level=0, drop=True))
    df = df.merge(daily.reset_index()[["toban", "date", "f_fly180"]], on=["toban", "date"], how="left")
    df["f_fly180"] = df["f_fly180"].fillna(0)

    # コース別成績（前日まで）
    df["course_in"] = df["course"]
    cprior = df.groupby("course")[["_win", "_top3"]].mean()
    d = df.dropna(subset=["course"]).groupby(["toban", "course", "date"])[["_win", "_top3"]].agg(["sum", "count"])
    d.columns = ["w", "n", "t3", "_"]
    d = d.drop(columns="_").reset_index().sort_values(["toban", "course", "date"])
    cs = d.groupby(["toban", "course"])[["w", "n", "t3"]].cumsum() - d[["w", "n", "t3"]]
    d["h_course_win"] = _smooth(cs["w"], cs["n"], d["course"].map(cprior["_win"]))
    d["h_course_top3"] = _smooth(cs["t3"], cs["n"], d["course"].map(cprior["_top3"]))
    df = df.merge(d[["toban", "course", "date", "h_course_win", "h_course_top3"]],
                  on=["toban", "course", "date"], how="left")
    return df.drop(columns=["_rank", "_top3", "_win", "_fly", "_exrel"])


def form_tables(hist: pd.DataFrame) -> dict:
    """当日の予想用：選手・モーターの直近の調子と、コース別成績の最新の表。"""
    h = hist.sort_values(["date", "jcd", "rno", "boat"]).copy()
    h["_rank"] = h["rank"].clip(upper=6.5)
    h["_top3"] = (h["rank"] <= 3).astype(float)
    h["_win"] = (h["rank"] == 1).astype(float)
    h["_fly"] = h["flag"].astype(str).str.startswith("F").astype(float)
    h["_exrel"] = _ex_rel(h)
    last = pd.to_datetime(h["date"].max())
    since = (last - pd.Timedelta(days=179)).strftime("%Y-%m-%d")
    g = h.groupby("toban")
    racer = pd.DataFrame({
        "f_rank10": g["_rank"].apply(lambda x: x.tail(10).mean() if len(x) >= 3 else np.nan),
        "f_st_std20": g["st"].apply(lambda x: x.dropna().tail(20).std() if x.notna().sum() >= 5 else np.nan),
        "f_fly180": h[h["date"] >= since].groupby("toban")["_fly"].sum(),
    }).reset_index()
    racer["f_fly180"] = racer["f_fly180"].fillna(0)
    gm = h.groupby(["jcd", "motor_no"])
    motor = pd.DataFrame({
        "m_top3_20": gm["_top3"].apply(lambda x: x.tail(20).mean() if len(x) >= 3 else np.nan),
        "m_ex_rel20": gm["_exrel"].apply(lambda x: x.dropna().tail(20).mean() if x.notna().sum() >= 3 else np.nan),
        "m_n20": gm.size().clip(upper=20),
    }).reset_index()
    cprior = h.groupby("course")[["_win", "_top3"]].mean()
    c = h.dropna(subset=["course"]).groupby(["toban", "course"]).agg(
        w=("_win", "sum"), t3=("_top3", "sum"), n=("_win", "size")).reset_index()
    c["h_course_win"] = _smooth(c["w"], c["n"], c["course"].map(cprior["_win"]))
    c["h_course_top3"] = _smooth(c["t3"], c["n"], c["course"].map(cprior["_top3"]))
    return {"form": racer, "motor": motor,
            "course": c[["toban", "course", "h_course_win", "h_course_top3"]]}


def apply_form(df: pd.DataFrame, t: dict) -> pd.DataFrame:
    df = df.merge(t["form"], on="toban", how="left")
    df = df.merge(t["motor"], on=["jcd", "motor_no"], how="left")
    if "course_in" in df and df["course_in"].notna().any():
        df = df.merge(t["course"].rename(columns={"course": "course_in"}), on=["toban", "course_in"], how="left")
    return df


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
