"""学習・答え合わせ（バックテスト）・確率計算。

    python -m boatrace.model      # 学習して models/ と reports/ を作る
"""
import itertools
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import (FEATURES_EX, FEATURES_NO_EX, add_basic, add_day_trend, add_form, add_history,
                       add_wind, form_tables, history_tables, results_long)

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)


# ------------------------------------------------------------ 確率
def normalize(df: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    """艇ごとの勝ちやすさを、レース内で合計1の「1着になる確率」に直す。"""
    s = pd.Series(raw, index=df.index)
    return (s / s.groupby(df["race_id"]).transform("sum")).values


def trifecta_probs(p: dict[int, float], lam: float = 1.0) -> dict[tuple, float]:
    """1着確率から3連単の確率を出す（2・3着は確率をlam乗して弱めたハーヴィル式）。"""
    boats = list(p)
    q = {b: max(p[b], 1e-9) ** lam for b in boats}
    out = {}
    for a, b, c in itertools.permutations(boats, 3):
        pa = p[a]
        pb = q[b] / (sum(q.values()) - q[a])
        pc = q[c] / (sum(q.values()) - q[a] - q[b])
        out[(a, b, c)] = pa * pb * pc
    return out


def top_n_probs(tri: dict[tuple, float]) -> dict[int, dict[str, float]]:
    """3連単確率から、各艇の1着・2着内・3着内の確率を出す。"""
    res = {}
    for (a, b, c), v in tri.items():
        for boat, pos in ((a, 1), (b, 2), (c, 3)):
            d = res.setdefault(boat, {"win": 0.0, "top2": 0.0, "top3": 0.0})
            if pos == 1:
                d["win"] += v
            if pos <= 2:
                d["top2"] += v
            d["top3"] += v
    return res


def trio_probs(tri: dict[tuple, float]) -> dict[tuple, float]:
    """3連単の確率から3連複の確率を出す（同じ3つの並び6通りを足す）。キーは小さい順。"""
    out: dict[tuple, float] = {}
    for k, v in tri.items():
        key = tuple(sorted(k))
        out[key] = out.get(key, 0.0) + v
    return out


TRIO_PLANS = {  # 名前: (説明, 点数)
    "top1": ("3連複 上位1点", 1), "top3": ("3連複 上位3点", 3), "top5": ("3連複 上位5点", 5),
    "top10": ("3連複 上位10点", 10), "box4": ("3着内に来やすい4つのボックス", 4),
    "box5": ("3着内に来やすい5つのボックス", 10),
}


def trio_plan_combos(tri: dict[tuple, float]) -> dict[str, list[tuple]]:
    """3連複の買い方ごとの買い目。"""
    trio = trio_probs(tri)
    ranked = sorted(trio, key=trio.get, reverse=True)
    top3 = top_n_probs(tri)
    order = sorted(top3, key=lambda b: top3[b]["top3"], reverse=True)
    return {
        "top1": ranked[:1], "top3": ranked[:3], "top5": ranked[:5], "top10": ranked[:10],
        "box4": [tuple(sorted(c)) for c in itertools.combinations(order[:4], 3)],
        "box5": [tuple(sorted(c)) for c in itertools.combinations(order[:5], 3)],
    }


class TrioScore:
    """3連複の買い方ごとに、的中数と払い戻しを数える。"""

    BUCKETS = (0.7, 0.6, 0.5, 0.4, 0.3, 0.0)

    def __init__(self):
        self.n = 0
        self.hit = {k: 0 for k in TRIO_PLANS}
        self.ret = {k: 0.0 for k in TRIO_PLANS}
        self.conf: dict[float, dict] = {}

    def add(self, tri: dict[tuple, float], actual: str | None, pay):
        try:
            act = tuple(sorted(int(x) for x in str(actual).split("-")))
        except ValueError:
            return
        if len(act) != 3:
            return
        self.n += 1
        plans = trio_plan_combos(tri)
        for k, combos in plans.items():
            if act in combos:
                self.hit[k] += 1
                self.ret[k] += pay if pd.notna(pay) else 0
        # 3連複の確信度：上位3点のどれかが当たる確率
        trio = trio_probs(tri)
        pc = sum(trio[c] for c in plans["top3"])
        key = next(b for b in self.BUCKETS if pc >= b)
        c = self.conf.setdefault(key, {"races": 0, "prob": 0.0, "hit1": 0, "ret1": 0.0, "hit3": 0, "ret3": 0.0, "hit5": 0, "ret5": 0.0})
        c["races"] += 1
        c["prob"] += pc
        for name, k in (("1", "top1"), ("3", "top3"), ("5", "top5")):
            if act in plans[k]:
                c["hit" + name] += 1
                c["ret" + name] += pay if pd.notna(pay) else 0

    def conf_result(self) -> list[dict]:
        out = []
        for k in sorted(self.conf, reverse=True):
            c = self.conf[k]
            r = c["races"]
            out.append({"min_prob": k, "races": r, "share": r / self.n, "avg_prob": c["prob"] / r,
                        "hit1": c["hit1"] / r, "roi1": c["ret1"] / (100 * r),
                        "hit3": c["hit3"] / r, "roi3": c["ret3"] / (300 * r),
                        "hit5": c["hit5"] / r, "roi5": c["ret5"] / (500 * r)})
        return out

    def result(self) -> list[dict]:
        return [{"plan": k, "label": TRIO_PLANS[k][0], "points": TRIO_PLANS[k][1],
                 "hit_rate": self.hit[k] / self.n if self.n else None,
                 "roi": self.ret[k] / (100 * TRIO_PLANS[k][1] * self.n) if self.n else None}
                for k in TRIO_PLANS]


def trio_md(rows: list[dict], conf: list[dict] | None = None) -> list[str]:
    pct = lambda x: "-" if x is None else f"{x * 100:.1f}%"
    out = ["", "### 3連複の買い方別", "", "| 買い方 | 点数 | 的中率 | 回収率 |", "|---|---|---|---|"]
    out += [f"| {r['label']} | {r['points']} | {pct(r['hit_rate'])} | {pct(r['roi'])} |" for r in rows]
    if conf:
        out += ["", "### 3連複の確信度別（上位3点のどれかが当たる確率で分けたもの）", "",
                "| 上位3点の的中確率 | レース数 | 割合 | 予想の平均 | 上位3点 的中率 | 上位3点 回収率 | 上位1点 的中率 | 上位1点 回収率 | 上位5点 的中率 | 上位5点 回収率 |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        out += [f"| {int(c['min_prob'] * 100)}%以上 | {c['races']:,} | {pct(c['share'])} | {pct(c['avg_prob'])} | "
                f"{pct(c['hit3'])} | {pct(c['roi3'])} | {pct(c['hit1'])} | {pct(c['roi1'])} | {pct(c['hit5'])} | {pct(c['roi5'])} |"
                for c in conf]
    return out


# ------------------------------------------------------------ 学習
def load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "races.parquet")
    df = add_history(df)
    df = add_form(df)
    df = add_basic(df)
    df = add_wind(df)
    df = add_day_trend(df, results_long(df))
    df["y"] = (df["rank"] == 1).astype(int)
    return df


def fit(train: pd.DataFrame, feats: list[str], valid: pd.DataFrame | None = None, rounds: int | None = None):
    dtr = lgb.Dataset(train[feats], train["y"])
    if valid is not None:
        dva = lgb.Dataset(valid[feats], valid["y"])
        m = lgb.train(PARAMS, dtr, 3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        m = lgb.train(PARAMS, dtr, rounds)
    return m


def tune_lambda(df: pd.DataFrame, p: np.ndarray) -> float:
    """実際の3連単の結果が最も起こりやすくなるlamを選ぶ。"""
    races = _race_dicts(df, p)
    best, best_ll = 1.0, -1e18
    for lam in np.arange(0.5, 1.21, 0.05):
        ll = 0.0
        for r in races:
            if r["tri"] is None:
                continue
            ll += np.log(trifecta_probs(r["p"], lam).get(r["tri"], 1e-9))
        if ll > best_ll:
            best, best_ll = float(round(lam, 2)), ll
    return best


def _race_dicts(df: pd.DataFrame, p: np.ndarray) -> list[dict]:
    d = df.assign(_p=p)
    out = []
    for rid, g in d.groupby("race_id", sort=False):
        tri = g["tri"].iloc[0]
        try:
            tri_t = tuple(int(x) for x in tri.split("-")) if isinstance(tri, str) else None
        except ValueError:
            tri_t = None
        out.append({
            "race_id": rid, "p": dict(zip(g["boat"], g["_p"])), "tri": tri_t,
            "pay_tri": g["pay_tri"].iloc[0], "pay_win": g["pay_win"].iloc[0],
            "trio": g["trio"].iloc[0] if "trio" in g else None,
            "pay_trio": g["pay_trio"].iloc[0] if "pay_trio" in g else None,
            "win_boat": int(g.loc[g["rank"] == 1, "boat"].iloc[0]),
        })
    return out


# ------------------------------------------------------------ 答え合わせ
def backtest(df: pd.DataFrame, p: np.ndarray, lam: float) -> dict:
    races = _race_dicts(df, p)
    n = len(races)
    hit1 = sum(max(r["p"], key=r["p"].get) == r["win_boat"] for r in races)
    base1 = sum(r["win_boat"] == 1 for r in races)
    # 単勝：予想1位を100円ずつ
    win_ret = sum(r["pay_win"] for r in races if max(r["p"], key=r["p"].get) == r["win_boat"] and pd.notna(r["pay_win"]))
    # 3連単：上位1点・上位5点
    tri_hit1 = tri_ret1 = tri_hit5 = tri_ret5 = 0
    buckets = {}
    trio_all = TrioScore()
    for r in races:
        tri = trifecta_probs(r["p"], lam)
        trio_all.add(tri, r["trio"], r["pay_trio"])
        ranked = sorted(tri, key=tri.get, reverse=True)
        pay = r["pay_tri"] if pd.notna(r["pay_tri"]) else 0
        if r["tri"] == ranked[0]:
            tri_hit1 += 1
            tri_ret1 += pay
        if r["tri"] in ranked[:5]:
            tri_hit5 += 1
            tri_ret5 += pay
        # 確信度（予想1位の1着確率）ごとの的中率
        top = max(r["p"], key=r["p"].get)
        pt = r["p"][top]
        key = next(k for k in (0.8, 0.7, 0.6, 0.5, 0.4, 0.0) if pt >= k)
        b = buckets.setdefault(key, {"races": 0, "hit": 0, "prob_sum": 0.0, "win_ret": 0, "tri_hit": 0, "tri_ret": 0})
        b["races"] += 1
        b["prob_sum"] += pt
        if top == r["win_boat"]:
            b["hit"] += 1
            b["win_ret"] += r["pay_win"] if pd.notna(r["pay_win"]) else 0
        if r["tri"] == ranked[0]:
            b["tri_hit"] += 1
            b["tri_ret"] += pay
    # 対数損失（小さいほど確率がよく当たっている）
    ll = -np.mean([np.log(max(r["p"][r["win_boat"]], 1e-9)) for r in races])
    lane_rate = df.groupby("boat")["y"].mean()
    ll_base = -np.mean([np.log(lane_rate[r["win_boat"]]) for r in races])
    conf = []
    for k in sorted(buckets, reverse=True):
        b = buckets[k]
        conf.append({
            "min_prob": k, "races": b["races"], "share": b["races"] / n,
            "hit_rate": b["hit"] / b["races"], "avg_prob": b["prob_sum"] / b["races"],
            "win_roi": b["win_ret"] / (100 * b["races"]),
            "tri_hit_rate": b["tri_hit"] / b["races"], "tri_roi": b["tri_ret"] / (100 * b["races"]),
        })
    return {
        "races": n,
        "period": [df["date"].min(), df["date"].max()],
        "win_hit_rate": hit1 / n, "baseline_boat1_hit_rate": base1 / n,
        "win_roi": win_ret / (100 * n),
        "tri_top1_hit_rate": tri_hit1 / n, "tri_top1_roi": tri_ret1 / (100 * n),
        "tri_top5_hit_rate": tri_hit5 / n, "tri_top5_roi": tri_ret5 / (500 * n),
        "logloss": ll, "logloss_lane_only": ll_base, "lambda": lam,
        "by_confidence": conf, "trio": trio_all.result(), "trio_conf": trio_all.conf_result(),
    }


def report_md(name: str, bt: dict) -> str:
    pct = lambda x: f"{x * 100:.1f}%"
    lines = [
        f"## {name}", "",
        f"- 期間：{bt['period'][0]} 〜 {bt['period'][1]}（{bt['races']:,}レース。学習に使っていない期間）",
        f"- 1着的中率：**{pct(bt['win_hit_rate'])}**（いつも1号艇を選んだ場合：{pct(bt['baseline_boat1_hit_rate'])}）",
        f"- 単勝（予想1位を毎レース100円）回収率：{pct(bt['win_roi'])}",
        f"- 3連単 予想1点 的中率：{pct(bt['tri_top1_hit_rate'])}／回収率：{pct(bt['tri_top1_roi'])}",
        f"- 3連単 予想上位5点 的中率：{pct(bt['tri_top5_hit_rate'])}／回収率：{pct(bt['tri_top5_roi'])}",
        f"- 対数損失：{bt['logloss']:.4f}（枠だけで予想した場合：{bt['logloss_lane_only']:.4f}。小さいほど良い）",
        "", "### 確信度別（予想1位の1着確率で分けたもの）", "",
        "| 1着確率 | レース数 | 割合 | 予想した確率の平均 | 実際の的中率 | 単勝回収率 | 3連単1点的中率 | 3連単1点回収率 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in bt["by_confidence"]:
        lines.append(
            f"| {int(c['min_prob'] * 100)}%以上 | {c['races']:,} | {pct(c['share'])} | {pct(c['avg_prob'])} | "
            f"{pct(c['hit_rate'])} | {pct(c['win_roi'])} | {pct(c['tri_hit_rate'])} | {pct(c['tri_roi'])} |"
        )
    lines += trio_md(bt["trio"], bt["trio_conf"])
    return "\n".join(lines) + "\n"


def save_history_tables():
    """予想に必要な集計表（選手・枠・場ごとの成績）だけを models/ に保存する。
    これがあれば、過去データを持っていないPCでも予想できる。"""
    hist = pd.read_parquet(ROOT / "data" / "races.parquet",
                           columns=["race_id", "date", "rno", "toban", "boat", "jcd", "rank", "st",
                                    "flag", "course", "motor_no", "ex_time"])
    tables = history_tables(hist) | form_tables(hist)
    for name, t in tables.items():
        t.to_parquet(MODELS / f"boat_hist_{name}.parquet", index=False)


def main(test_days: int = 60, valid_days: int = 30):
    df = load_dataset()
    dates = sorted(df["date"].unique())
    test_start = dates[-test_days]
    valid_start = dates[-(test_days + valid_days)]
    tr = df[df["date"] < valid_start]
    va = df[(df["date"] >= valid_start) & (df["date"] < test_start)]
    te = df[df["date"] >= test_start]
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    summary = {}
    md = ["# 答え合わせ（バックテスト）の結果", "",
          "学習に使っていない直近の期間で、予想がどれくらい当たったかを確かめた結果です。",
          "回収率は払い戻し÷購入額です。100%を超えると黒字ですが、超えなくても普通です。",
          "展示タイムありモデルは、答え合わせでは本番の実際の進入コースを使っています。当日はスタート展示の進入で予想するので、",
          "本番で並びが変わったレースの分だけ、実際は少し下がります（全員枠なりと仮定した場合、3連複上位3点の的中率は約1ポイント下がりました）。", ""]
    for name, feats, label in (("ex", FEATURES_EX, "展示タイムありモデル（直前情報が出たあと）"),
                               ("noex", FEATURES_NO_EX, "展示タイムなしモデル（番組表だけ）")):
        m = fit(tr, feats, va)
        best = m.best_iteration
        lam = tune_lambda(va, normalize(va, m.predict(va[feats], num_iteration=best)))
        p_te = normalize(te, m.predict(te[feats], num_iteration=best))
        bt = backtest(te, p_te, lam)
        md.append(report_md(label, bt))
        # 本番用：全期間で学び直す
        full = fit(df, feats, rounds=max(best, 50))
        full.save_model(str(MODELS / f"boat_{name}.txt"))
        imp = pd.Series(full.feature_importance("gain"), index=feats).sort_values(ascending=False)
        summary[name] = {"backtest": bt, "rounds": best, "lambda": lam, "features": feats,
                         "importance": (imp / imp.sum()).round(4).head(15).to_dict()}
        print(label, json.dumps({k: v for k, v in bt.items() if k != "by_confidence"}, ensure_ascii=False, default=float))
    summary["trained_until"] = dates[-1]
    save_history_tables()
    (MODELS / "meta.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    (REPORTS / "backtest.md").write_text("\n".join(md))
    print("reports/backtest.md を保存しました")


if __name__ == "__main__":
    main()
