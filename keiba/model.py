"""競馬の学習・答え合わせ。

    python -m keiba.model

予想は2段階で出す。
1. 能力モデル：馬・騎手・コースのデータだけで、各馬の1着確率を出す（オッズは使わない）
2. 市場との合成：能力モデルの確率と、単勝オッズから計算した確率を組み合わせる
   （組み合わせの比率は過去データで決める）。オッズがまだ出ていないときは1だけを使う
"""
import itertools
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from boatrace.model import trifecta_probs

from .features import FEATURES, add_history, add_relative

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=80,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1)


def load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "keiba" / "races.parquet")
    df = df[df["surface"] != 2]  # 障害レースは除く
    df = add_relative(add_history(df))
    df["y"] = (df["rank"] == 1).astype(int)
    return df


def norm(df, v):
    s = pd.Series(v, index=df.index)
    return (s / s.groupby(df["race_id"]).transform("sum")).values


def market_prob(df) -> np.ndarray:
    inv = 1.0 / df["odds"].where(df["odds"] > 0)
    inv = inv.fillna(inv.groupby(df["race_id"]).transform("min") / 2).fillna(1e-3)
    return norm(df, inv.values)


def blend(df, p_model, p_mkt, a, b):
    return norm(df, np.exp(a * np.log(np.clip(p_model, 1e-6, 1)) + b * np.log(np.clip(p_mkt, 1e-6, 1))))


def fit_blend(df, p_model, p_mkt) -> tuple[float, float]:
    """1着の馬の確率が最も高くなる組み合わせ比率を探す。"""
    win = df["y"].values == 1
    best = (1.0, 0.0, -1e18)
    for a in np.arange(0.0, 1.61, 0.1):
        for b in np.arange(0.0, 1.61, 0.1):
            if a == 0 and b == 0:
                continue
            ll = np.log(blend(df, p_model, p_mkt, a, b)[win]).sum()
            if ll > best[2]:
                best = (round(float(a), 2), round(float(b), 2), ll)
    return best[0], best[1]


def tune_lambda(df, p) -> float:
    d = df.assign(_p=p)
    races = [(dict(zip(g["umaban"], g["_p"])), g["tri"].iloc[0]) for _, g in d.groupby("race_id")]
    best, best_ll = 1.0, -1e18
    for lam in np.arange(0.5, 1.21, 0.05):
        ll = 0.0
        for pw, tri in races:
            try:
                key = tuple(int(x) for x in str(tri).split("-"))
            except ValueError:
                continue
            ll += np.log(trifecta_probs(pw, lam).get(key, 1e-9))
        if ll > best_ll:
            best, best_ll = float(round(lam, 2)), ll
    return best


def backtest(df: pd.DataFrame, p: np.ndarray, lam: float, value_th=(1.0, 1.1, 1.2, 1.3)) -> dict:
    d = df.assign(_p=p)
    n = hit = win_ret = fav_hit = 0
    tri1 = tri1_ret = tri5 = tri5_ret = 0
    val = {th: [0, 0.0] for th in value_th}
    buckets = {}
    ll = []
    for _, g in d.groupby("race_id", sort=False):
        n += 1
        top = g.loc[g["_p"].idxmax()]
        winner = g[g["rank"] == 1].iloc[0]
        ll.append(-np.log(max(winner["_p"], 1e-9)))
        ok = top["umaban"] == winner["umaban"]
        hit += ok
        win_ret += g["pay_win"].iloc[0] if ok else 0
        fav = g.loc[g["odds"].idxmin()] if g["odds"].notna().any() else top
        fav_hit += fav["umaban"] == winner["umaban"]
        # 単勝：期待値（確率×オッズ）がしきい値以上の馬をすべて買う
        for th in value_th:
            sel = g[(g["_p"] * g["odds"] >= th) & g["odds"].notna()]
            val[th][0] += len(sel)
            val[th][1] += g["pay_win"].iloc[0] if (sel["umaban"] == winner["umaban"]).any() else 0
        pw = dict(zip(g["umaban"], g["_p"]))
        tri = trifecta_probs(pw, lam)
        ranked = sorted(tri, key=tri.get, reverse=True)[:5]
        try:
            actual = tuple(int(x) for x in str(g["tri"].iloc[0]).split("-"))
        except ValueError:
            actual = None
        pay = g["pay_tri"].iloc[0] if pd.notna(g["pay_tri"].iloc[0]) else 0
        if actual == ranked[0]:
            tri1 += 1
            tri1_ret += pay
        if actual in ranked:
            tri5 += 1
            tri5_ret += pay
        pt = top["_p"]
        key = next(k for k in (0.6, 0.5, 0.4, 0.3, 0.2, 0.0) if pt >= k)
        bk = buckets.setdefault(key, {"races": 0, "hit": 0, "prob_sum": 0.0, "win_ret": 0})
        bk["races"] += 1
        bk["prob_sum"] += pt
        bk["hit"] += ok
        bk["win_ret"] += g["pay_win"].iloc[0] if ok else 0
    conf = [{"min_prob": k, "races": b["races"], "share": b["races"] / n, "avg_prob": b["prob_sum"] / b["races"],
             "hit_rate": b["hit"] / b["races"], "win_roi": b["win_ret"] / (100 * b["races"])}
            for k, b in sorted(buckets.items(), reverse=True)]
    return {
        "races": n, "period": [df["date"].min(), df["date"].max()],
        "win_hit_rate": hit / n, "favorite_hit_rate": fav_hit / n, "win_roi": win_ret / (100 * n),
        "tri_top1_hit_rate": tri1 / n, "tri_top1_roi": tri1_ret / (100 * n),
        "tri_top5_hit_rate": tri5 / n, "tri_top5_roi": tri5_ret / (500 * n),
        "logloss": float(np.mean(ll)), "lambda": lam, "by_confidence": conf,
        "value_bets": [{"ev_min": th, "bets": v[0], "roi": (v[1] / (100 * v[0])) if v[0] else None}
                       for th, v in val.items()],
    }


def report_md(label: str, bt: dict) -> str:
    pct = lambda x: "-" if x is None else f"{x * 100:.1f}%"
    out = [f"## {label}", "",
           f"- 期間：{bt['period'][0]} 〜 {bt['period'][1]}（{bt['races']:,}レース）",
           f"- 1着的中率：**{pct(bt['win_hit_rate'])}**（1番人気を選んだ場合：{pct(bt['favorite_hit_rate'])}）",
           f"- 単勝（予想1位を毎レース100円）回収率：{pct(bt['win_roi'])}",
           f"- 3連単 予想1点 的中率：{pct(bt['tri_top1_hit_rate'])}／回収率：{pct(bt['tri_top1_roi'])}",
           f"- 3連単 予想上位5点 的中率：{pct(bt['tri_top5_hit_rate'])}／回収率：{pct(bt['tri_top5_roi'])}",
           f"- 対数損失：{bt['logloss']:.4f}（小さいほど良い）", "",
           "| 単勝の期待値がこれ以上の馬を全部買う | 買った数 | 回収率 |", "|---|---|---|"]
    out += [f"| {v['ev_min']:.1f} | {v['bets']:,} | {pct(v['roi'])} |" for v in bt["value_bets"]]
    out += ["", "| 予想1位の1着確率 | レース数 | 予想の平均 | 実際の的中率 | 単勝回収率 |", "|---|---|---|---|---|"]
    out += [f"| {int(c['min_prob'] * 100)}%以上 | {c['races']:,} | {pct(c['avg_prob'])} | {pct(c['hit_rate'])} | {pct(c['win_roi'])} |"
            for c in bt["by_confidence"]]
    return "\n".join(out) + "\n"


def main(test_days: int = 90, valid_days: int = 60):
    df = load_dataset()
    last = pd.to_datetime(df["date"].max())
    test_start = (last - pd.Timedelta(days=test_days)).strftime("%Y-%m-%d")
    valid_start = (last - pd.Timedelta(days=test_days + valid_days)).strftime("%Y-%m-%d")
    tr = df[df["date"] < valid_start]
    va = df[(df["date"] >= valid_start) & (df["date"] < test_start)]
    te = df[df["date"] >= test_start]

    m = lgb.train(PARAMS, lgb.Dataset(tr[FEATURES], tr["y"]), 3000,
                  valid_sets=[lgb.Dataset(va[FEATURES], va["y"])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    best = m.best_iteration
    pm_va = norm(va, m.predict(va[FEATURES], num_iteration=best))
    pk_va = market_prob(va)
    a, b = fit_blend(va, pm_va, pk_va)
    lam_model = tune_lambda(va, pm_va)
    lam_blend = tune_lambda(va, blend(va, pm_va, pk_va, a, b))

    pm_te = norm(te, m.predict(te[FEATURES], num_iteration=best))
    pk_te = market_prob(te)
    bt = {
        "model": backtest(te, pm_te, lam_model),
        "market": backtest(te, pk_te, lam_blend),
        "blend": backtest(te, blend(te, pm_te, pk_te, a, b), lam_blend),
    }
    md = ["# 競馬 答え合わせ（バックテスト）", "",
          "学習に使っていない直近の期間での結果です。オッズは確定オッズを使っているため、"
          "締切前のオッズで買う実際よりも少し良く出ます。", ""]
    for k, label in (("blend", "能力モデル＋オッズの合成（オッズ発表後の予想）"),
                     ("model", "能力モデルだけ（オッズを使わない予想）"),
                     ("market", "比較用：オッズだけ（人気順）")):
        md.append(report_md(label, bt[k]))
        print(label, {x: y for x, y in bt[k].items() if x not in ("by_confidence", "value_bets")})

    full = lgb.train(PARAMS, lgb.Dataset(df[FEATURES], df["y"]), max(best, 50))
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    full.save_model(str(MODELS / "keiba.txt"))
    imp = pd.Series(full.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    meta = {"backtest": bt, "rounds": best, "blend": {"a": a, "b": b},
            "lambda_model": lam_model, "lambda_blend": lam_blend, "features": FEATURES,
            "importance": (imp / imp.sum()).round(4).head(15).to_dict(), "trained_until": df["date"].max()}
    (MODELS / "keiba_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))
    (REPORTS / "keiba_backtest.md").write_text("\n".join(md))
    print("合成の比率", a, b, "／ reports/keiba_backtest.md を保存しました")


if __name__ == "__main__":
    main()
