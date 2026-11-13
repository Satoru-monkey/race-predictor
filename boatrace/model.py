"""学習・答え合わせ（バックテスト）・確率計算。

    python -m boatrace.model      # 学習して models/ と reports/ を作る
"""
import itertools
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import FEATURES_EX, FEATURES_NO_EX, add_basic, add_history

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


# ------------------------------------------------------------ 学習
def load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "races.parquet")
    df = add_history(df)
    df = add_basic(df)
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
    for r in races:
        tri = trifecta_probs(r["p"], lam)
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
        "by_confidence": conf,
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
    return "\n".join(lines) + "\n"


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
          "回収率は払い戻し÷購入額です。100%を超えると黒字ですが、超えなくても普通です。", ""]
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
    (MODELS / "meta.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    (REPORTS / "backtest.md").write_text("\n".join(md))
    print("reports/backtest.md を保存しました")


if __name__ == "__main__":
    main()
