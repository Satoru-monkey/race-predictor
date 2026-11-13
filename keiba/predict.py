"""競馬のレース予想。"""
import datetime as dt
import json
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from boatrace.model import top_n_probs, trifecta_probs
from boatrace.predict import trio_view

from . import live
from .features import add_relative, apply_latest, latest_tables
from .parse import VENUES

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

LABELS = {
    "jcd_i": "競馬場", "rno": "レース番号", "surface": "芝・ダート", "direction": "回り", "distance": "距離",
    "going": "馬場状態", "class": "クラス", "handicap": "ハンデ戦", "female": "牝馬限定", "heads": "頭数",
    "waku": "枠", "umaban": "馬番", "sex": "性別", "age": "年齢", "kinryo": "斤量", "weight": "馬体重",
    "weight_diff": "馬体重の増減", "h_n": "出走数", "h_days": "前走からの間隔", "h_last_rank_r": "前走の着順",
    "h_avg_rank_r3": "近3走の着順", "h_best_rank_r3": "近3走の最高着順", "h_last_margin": "前走の勝ち馬とのタイム差",
    "h_avg_margin3": "近3走の勝ち馬とのタイム差", "h_last_l3f_r": "前走の上がり順位", "h_avg_l3f_r3": "近3走の上がり順位",
    "h_last_pop": "前走の人気", "h_avg_front3": "近3走の位置取り", "h_class_diff": "前走からのクラス変化",
    "h_dist_diff": "前走からの距離変化", "h_surface_same_rank_r": "同じ芝・ダートでの着順", "h_win_rate": "勝率",
    "h_top3_rate": "3着内率", "j_win": "騎手の勝率", "j_top3": "騎手の3着内率", "j_n": "騎手の騎乗数",
    "t_win": "調教師の勝率", "t_top3": "調教師の3着内率",
}
for _c in ("h_avg_rank_r3", "h_avg_margin3", "h_avg_l3f_r3", "j_win", "kinryo", "h_last_pop"):
    LABELS[f"{_c}_rk"] = LABELS[_c] + "（レース内の順位）"
    LABELS[f"{_c}_rel"] = LABELS[_c] + "（レース内の平均との差）"


def label(f):
    return LABELS.get(f, f)


@lru_cache(maxsize=1)
def meta():
    return json.loads((MODELS / "keiba_meta.json").read_text())


@lru_cache(maxsize=1)
def booster():
    return lgb.Booster(model_file=str(MODELS / "keiba.txt"))


@lru_cache(maxsize=1)
def tables():
    names = ("horse", "surface", "j", "t")
    if all((MODELS / f"keiba_hist_{n}.parquet").exists() for n in names):
        return {n: pd.read_parquet(MODELS / f"keiba_hist_{n}.parquet") for n in names}
    hist = pd.read_parquet(ROOT / "data" / "keiba" / "races.parquet")
    return latest_tables(hist[hist["surface"] != 2])


def day_races(day: dt.date) -> list[dict]:
    out = []
    for r in live.race_list(f"{day:%Y%m%d}"):
        rid = r["race_id"]
        out.append({**r, "venue": VENUES.get(rid[4:6], rid[4:6]), "rno": int(rid[10:12])})
    return out


def predict_race(race_id: str, day: dt.date) -> dict:
    race, df = live.shutuba(race_id)
    if df.empty:
        raise ValueError("出馬表が見つかりません")
    if race.get("surface") == 2:
        raise ValueError("障害レースは予想の対象外です")
    for k, v in race.items():
        if k not in df:
            df[k] = v
    df["date"] = day.isoformat()
    df = add_relative(apply_latest(df, tables()))
    feats = meta()["features"]
    X = df[feats].astype(float)
    m = booster()
    raw = m.predict(X)
    contrib = m.predict(X, pred_contrib=True)[:, :-1]
    p_model = raw / raw.sum()

    errors = []
    try:
        ow = live.odds_win(race_id)
        ot = live.odds_trifecta(race_id)
        of = live.odds_trio(race_id)
    except Exception as e:
        ow, ot, of = {}, {}, {}
        errors.append(f"オッズを取得できませんでした（{e}）")
    df["odds"] = df["umaban"].map(ow)
    have_odds = df["odds"].notna().sum() >= max(2, len(df) - 1)
    if have_odds:
        inv = (1 / df["odds"]).fillna((1 / df["odds"]).min() / 2)
        p_mkt = (inv / inv.sum()).values
        a, b = meta()["blend"]["a"], meta()["blend"]["b"]
        z = np.exp(a * np.log(np.clip(p_model, 1e-6, 1)) + b * np.log(np.clip(p_mkt, 1e-6, 1)))
        p = z / z.sum()
        lam = meta()["lambda_blend"]
        mode = "blend"
    else:
        p_mkt = None
        p = p_model
        lam = meta()["lambda_model"]
        mode = "model"

    pw = dict(zip(df["umaban"].astype(int), p))
    tri = trifecta_probs(pw, lam)
    tops = top_n_probs(tri)
    mean_c = contrib.mean(axis=0)
    horses = []
    for i, row in df.reset_index(drop=True).iterrows():
        u = int(row["umaban"])
        c = pd.Series(contrib[i] - mean_c, index=feats)
        horses.append({
            "umaban": u, "waku": int(row["waku"]) if pd.notna(row["waku"]) else None, "horse": row["horse"],
            "jockey": row["jockey"], "kinryo": row["kinryo"],
            "sexage": f"{({0: '牡', 1: '牝', 2: 'セ'}).get(row['sex'], '')}{int(row['age']) if pd.notna(row['age']) else ''}",
            "weight": f"{int(row['weight'])}({int(row['weight_diff']):+d})" if pd.notna(row.get("weight")) and pd.notna(row.get("weight_diff")) else "",
            "runs": int(row["h_n"]) if pd.notna(row["h_n"]) else 0,
            "p_model": float(p_model[i]), "p_mkt": float(p_mkt[i]) if p_mkt is not None else None,
            "win": tops[u]["win"], "top2": tops[u]["top2"], "top3": tops[u]["top3"],
            "odds": ow.get(u), "ev": tops[u]["win"] * ow[u] if u in ow else None,
            "plus": [label(f) for f in c.sort_values(ascending=False).index[:3] if c[f] > 0.03],
            "minus": [label(f) for f in c.sort_values().index[:2] if c[f] < -0.03],
        })
    horses.sort(key=lambda h: h["win"], reverse=True)
    ranked = sorted(tri.items(), key=lambda kv: kv[1], reverse=True)
    top_combos = [{"combo": "-".join(map(str, k)), "p": v, "odds": ot.get(k),
                   "ev": v * ot[k] if k in ot else None} for k, v in ranked[:10]]
    top = horses[0]
    return {
        "race_id": race_id, "date": day.isoformat(), "venue": VENUES.get(race["jcd"]), "rno": race["rno"],
        "title": race["title"], "start": race["start"], "race": race, "mode": mode,
        "horses": horses, "top_combos": top_combos, "errors": errors, "confidence": top["win"],
        "trio_conf": next(x["hit_p"] for x in trio_view(tri, {}, None)["plans"] if x["key"] == "top3"),
        "confidence_label": "高" if top["win"] >= 0.4 else ("中" if top["win"] >= 0.25 else "低"),
        "wakus": {h["umaban"]: h["waku"] for h in horses},
        "trio": trio_view(tri, of, meta()["backtest"][mode].get("trio")),
        "value_wins": [h for h in horses if h["ev"] and h["ev"] >= 1.1 and h["win"] >= 0.03],
    }
