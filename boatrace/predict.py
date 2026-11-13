"""学習済みモデルで、レースの順位を予想する。"""
import datetime as dt
import json
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import live
from .features import add_basic, apply_history, history_tables
from .model import TRIO_PLANS, top_n_probs, trifecta_probs, trio_plan_combos, trio_probs
from .parse import VENUES

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

LABELS = {
    "boat": "枠番", "class": "級別", "age": "年齢", "weight": "体重", "nat_win": "全国勝率",
    "nat_2": "全国2連率", "loc_win": "当地勝率", "loc_2": "当地2連率", "motor_2": "モーター2連率",
    "boat_2": "ボート2連率", "series_n": "今節の出走数", "series_avg": "今節の平均着順",
    "distance": "距離", "wind": "風速", "wave": "波高", "jcd_i": "レース場", "kind_i": "レースの種類",
    "rno": "レース番号", "h_st_avg": "平均スタートタイミング", "h_n": "スタート記録数",
    "h_win": "最近の1着率", "h_top3": "最近の3着内率", "h_lane_win": "この枠での1着率",
    "h_lane_top3": "この枠での3着内率", "h_lane_n": "この枠での出走数", "v_lane_win": "この場・この枠の1着率",
    "ex_time": "展示タイム", "ex_diff": "展示タイム（最速との差）", "ex_rank": "展示タイム順位",
}
for _c in ("nat_win", "nat_2", "loc_win", "motor_2", "boat_2", "class", "series_avg"):
    LABELS[f"{_c}_rel"] = LABELS[_c] + "（レース内の平均との差）"
    LABELS[f"{_c}_rk"] = LABELS[_c] + "（レース内の順位）"


def label(f: str) -> str:
    return LABELS.get(f, f)


@lru_cache(maxsize=1)
def meta() -> dict:
    return json.loads((MODELS / "meta.json").read_text())


@lru_cache(maxsize=2)
def booster(name: str) -> lgb.Booster:
    return lgb.Booster(model_file=str(MODELS / f"boat_{name}.txt"))


@lru_cache(maxsize=1)
def hist_tables() -> dict:
    """選手・枠・場ごとの成績表。学習時に保存した models/ の集計表を使う。"""
    names = ("racer", "lane", "venue")
    if all((MODELS / f"boat_hist_{n}.parquet").exists() for n in names):
        return {n: pd.read_parquet(MODELS / f"boat_hist_{n}.parquet") for n in names}
    hist = pd.read_parquet(ROOT / "data" / "races.parquet",
                           columns=["toban", "boat", "jcd", "rank", "st"])
    return history_tables(hist)


def _features(prog: pd.DataFrame, before: dict | None = None) -> pd.DataFrame:
    df = prog.copy()
    if before:
        w = before.get("weather", {})
        df["wind"] = w.get("wind")
        df["wave"] = w.get("wave")
        df["ex_time"] = df["boat"].map(lambda b: before["boats"].get(b, {}).get("ex_time"))
    df = apply_history(df, hist_tables())
    return add_basic(df)


def _probs(df: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    m, info = booster(name), meta()[name]
    feats = info["features"]
    X = df[feats].astype(float)
    raw = m.predict(X)
    contrib = m.predict(X, pred_contrib=True)[:, :-1]
    s = pd.Series(raw, index=df.index)
    p = (s / s.groupby(df["race_id"]).transform("sum")).values
    return p, contrib, feats


def day_overview(day: dt.date) -> list[dict]:
    """その日の全レースを番組表だけで予想し、確信度つきの一覧にする。"""
    prog = live.program(day)
    if prog.empty:
        return []
    df = _features(prog)
    p, _, _ = _probs(df, "noex")
    df["p"] = p
    lam = meta()["noex"]["lambda"]
    out = []
    for rid, g in df.groupby("race_id", sort=False):
        g = g.sort_values("p", ascending=False)
        top = g.iloc[0]
        tri = trifecta_probs(dict(zip(g["boat"].astype(int), g["p"])), lam)
        trio = trio_probs(tri)
        top3 = sorted(trio, key=trio.get, reverse=True)[:3]
        out.append({
            "race_id": rid, "jcd": top["jcd"], "venue": VENUES.get(top["jcd"], top["jcd"]),
            "rno": int(top["rno"]), "title": top["title"], "deadline": top["deadline"],
            "top_boat": int(top["boat"]), "top_name": top["name"], "top_p": float(top["p"]),
            "order": [int(b) for b in g["boat"]],
            "trio_p": sum(trio[c] for c in top3), "trio_top": ["=".join(map(str, c)) for c in top3],
        })
    return out


def predict_race(day: dt.date, jcd: str, rno: int, use_live: bool = True) -> dict:
    prog = live.program(day)
    prog = prog[(prog["jcd"] == jcd) & (prog["rno"] == rno)]
    if prog.empty:
        raise ValueError("このレースの出走表が見つかりません")
    before, errors = None, []
    if use_live:
        try:
            before = live.beforeinfo(jcd, rno, day)
        except Exception as e:  # 取得できなくても番組表だけで予想する
            errors.append(f"直前情報を取得できませんでした（{e}）")
    name = "ex" if before and before.get("ready") else "noex"
    df = _features(prog, before)
    p, contrib, feats = _probs(df, name)
    lam = meta()[name]["lambda"]
    pw = dict(zip(df["boat"].astype(int), p))
    tri = trifecta_probs(pw, lam)
    tops = top_n_probs(tri)

    odds_w, odds_t, odds_f = {}, {}, {}
    if use_live:
        try:
            odds_w = live.odds_win(jcd, rno, day)
            odds_t = live.odds_trifecta(jcd, rno, day)
            odds_f = live.odds_trio(jcd, rno, day)
        except Exception as e:
            errors.append(f"オッズを取得できませんでした（{e}）")

    boats = []
    for i, row in enumerate(df.itertuples()):
        b = int(row.boat)
        c = pd.Series(contrib[i], index=feats)
        # レース内の平均との差で「この艇が有利・不利な理由」を出す
        c = c - pd.Series(contrib.mean(axis=0), index=feats)
        pos = [label(f) for f in c.sort_values(ascending=False).index[:3] if c[f] > 0.02]
        neg = [label(f) for f in c.sort_values().index[:2] if c[f] < -0.02]
        bi = (before or {}).get("boats", {}).get(b, {})
        boats.append({
            "boat": b, "name": row.name, "class": "",
            "toban": int(row.toban), "nat_win": row.nat_win, "loc_win": row.loc_win,
            "motor_2": row.motor_2, "ex_time": bi.get("ex_time"), "st_ex": bi.get("st_ex_text"),
            "course_ex": bi.get("course_ex"), "tilt": bi.get("tilt"),
            "win": tops[b]["win"], "top2": tops[b]["top2"], "top3": tops[b]["top3"],
            "odds": odds_w.get(b), "ev": (tops[b]["win"] * odds_w[b]) if b in odds_w else None,
            "plus": pos, "minus": neg,
        })
    cls_map = dict(zip(prog["boat"].astype(int), prog["class"]))
    for bo in boats:
        bo["class"] = {4: "A1", 3: "A2", 2: "B1", 1: "B2"}.get(int(cls_map.get(bo["boat"], 0)), "")
    boats.sort(key=lambda x: x["win"], reverse=True)

    ranked = sorted(tri.items(), key=lambda kv: kv[1], reverse=True)
    top_combos = [{"combo": "-".join(map(str, k)), "p": v, "odds": odds_t.get(k),
                   "ev": v * odds_t[k] if k in odds_t else None} for k, v in ranked[:10]]
    value = [{"combo": "-".join(map(str, k)), "p": v, "odds": odds_t[k], "ev": v * odds_t[k]}
             for k, v in ranked if k in odds_t and v >= 0.02 and v * odds_t[k] >= 1.2]
    value.sort(key=lambda x: x["ev"], reverse=True)

    top = boats[0]
    return {
        "date": day.isoformat(), "jcd": jcd, "venue": VENUES.get(jcd, jcd), "rno": rno,
        "title": prog["title"].iloc[0], "deadline": prog["deadline"].iloc[0],
        "model": name, "before": before, "boats": boats, "top_combos": top_combos,
        "value_bets": value[:8], "errors": errors,
        "trio": trio_view(tri, odds_f, meta()[name]["backtest"].get("trio")),
        "confidence": top["win"],
        "trio_conf": next(x["hit_p"] for x in trio_view(tri, {}, None)["plans"] if x["key"] == "top3"),
        "confidence_label": "高" if top["win"] >= 0.7 else ("中" if top["win"] >= 0.55 else "低"),
    }


def trio_view(tri: dict, odds: dict, bt_trio: list | None) -> dict:
    """画面用の3連複の情報（上位10点と、買い方ごとの買い目・過去の成績）。"""
    trio = trio_probs(tri)
    ranked = sorted(trio.items(), key=lambda kv: kv[1], reverse=True)
    top = [{"combo": "=".join(map(str, k)), "p": v, "odds": odds.get(k),
            "ev": v * odds[k] if k in odds else None} for k, v in ranked[:10]]
    plans = []
    past = {r["plan"]: r for r in (bt_trio or [])}
    for key, combos in trio_plan_combos(tri).items():
        if key == "top10":
            continue
        hit_p = sum(trio[c] for c in combos)
        exp = sum(trio[c] * odds[c] for c in combos if c in odds) / len(combos) if all(c in odds for c in combos) else None
        plans.append({"key": key, "label": TRIO_PLANS[key][0], "combos": ["=".join(map(str, c)) for c in combos],
                      "hit_p": hit_p, "ev": exp, "past": past.get(key)})
    return {"top": top, "plans": plans}
