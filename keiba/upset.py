"""競馬の「荒れ度」：3連複の払い戻しが1万円以上になる確率を、レースごとに予想する。

    python -m keiba.upset      # 学習と答え合わせ（models/keiba_upset.txt と keiba_upset_meta.json を作る）

材料は、単勝オッズの割れ方（1番人気の確率、上位人気に票がどれだけ集まっているかなど）と、
レース条件（頭数・クラス・芝ダート・距離・馬場・競馬場・ハンデ戦・2歳戦など）。
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from boatrace.model import trifecta_probs, trio_probs

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
BIG = 10000  # 「荒れた」＝3連複の払い戻しが1万円以上

FEATURES = ["fav_p", "p2", "p3", "top3_sum", "top5_sum", "entropy", "fav_odds", "heads",
            "jcd_i", "surface", "distance", "going", "class", "handicap", "female", "age2", "rno"]
BANDS = ["荒れにくい", "やや荒れにくい", "ふつう", "やや荒れやすい", "荒れやすい"]
PARAMS = dict(objective="binary", learning_rate=0.02, num_leaves=15, min_data_in_leaf=60,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1)


def race_features(race: dict, odds: dict[int, float], ages: list | None = None) -> dict:
    """1レース分の材料。race はレース条件、odds は {馬番: 単勝オッズ}。"""
    o = np.array([v for v in odds.values() if v and v > 0], dtype=float)
    p = np.sort((1 / o) / (1 / o).sum())[::-1]
    return {
        "fav_p": p[0], "p2": p[1] if len(p) > 1 else np.nan, "p3": p[2] if len(p) > 2 else np.nan,
        "top3_sum": p[:3].sum(), "top5_sum": p[:5].sum(), "entropy": float(-(p * np.log(p)).sum()),
        "fav_odds": float(o.min()), "heads": race.get("heads") or len(o),
        "jcd_i": int(race["jcd"]), "surface": race.get("surface"), "distance": race.get("distance"),
        "going": race.get("going"), "class": race.get("class"), "handicap": race.get("handicap"),
        "female": race.get("female"),
        "age2": float(bool(ages) and all(a == 2 for a in ages if a is not None)), "rno": race.get("rno"),
    }


def build_races(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, g in df.groupby("race_id"):
        if g["odds"].isna().all() or pd.isna(g["pay_trio"].iloc[0]):
            continue
        r = g.iloc[0]
        f = race_features({"jcd": r["jcd"], "heads": len(g), "surface": r["surface"], "distance": r["distance"],
                           "going": r["going"], "class": r["class"], "handicap": r["handicap"],
                           "female": r["female"], "rno": r["rno"]},
                          dict(zip(g["umaban"], g["odds"])), list(g["age"]))
        f.update({"race_id": rid, "date": r["date"], "pay_trio": r["pay_trio"], "trio": r["trio"],
                  "big": int(r["pay_trio"] >= BIG)})
        rows.append(f)
    return pd.DataFrame(rows)


def band_of(p: float, cuts: list[float]) -> int:
    return int(np.searchsorted(cuts, p, side="right"))


def main(test_days: int = 150, valid_days: int = 90):
    df = pd.read_parquet(ROOT / "data" / "keiba" / "races.parquet")
    df = df[df["surface"] != 2]
    races = df
    R = build_races(df)
    last = pd.to_datetime(R["date"].max())
    ts = (last - pd.Timedelta(days=test_days)).strftime("%Y-%m-%d")
    vs = (last - pd.Timedelta(days=test_days + valid_days)).strftime("%Y-%m-%d")
    tr, va, te = R[R.date < vs], R[(R.date >= vs) & (R.date < ts)], R[R.date >= ts]
    m = lgb.train(PARAMS, lgb.Dataset(tr[FEATURES], tr["big"]), 2000,
                  valid_sets=[lgb.Dataset(va[FEATURES], va["big"])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    best = m.best_iteration
    # 荒れ度の区切りは、調整用の期間の予想を5等分して決める（答え合わせの期間は使わない）
    cuts = list(np.quantile(m.predict(va[FEATURES], num_iteration=best), [0.2, 0.4, 0.6, 0.8]))
    te = te.assign(pred=m.predict(te[FEATURES], num_iteration=best))
    te["band"] = te["pred"].map(lambda p: band_of(p, cuts))

    # 荒れ度ごとに、人気順の3連複（上位1点・3点・5点）を買った場合の成績
    stats = []
    for b in range(5):
        t = te[te.band == b]
        hit = {k: 0 for k in (1, 3, 5)}
        ret = {k: 0.0 for k in (1, 3, 5)}
        for rid in t["race_id"]:
            g = races[races.race_id == rid]
            pw = dict(zip(g["umaban"], (1 / g["odds"]) / (1 / g["odds"]).sum()))
            trio = trio_probs(trifecta_probs(pw, 0.7))
            ranked = sorted(trio, key=trio.get, reverse=True)
            act = tuple(sorted(int(x) for x in str(g["trio"].iloc[0]).split("-")))
            for k in hit:
                if act in ranked[:k]:
                    hit[k] += 1
                    ret[k] += g["pay_trio"].iloc[0]
        n = len(t)
        stats.append({
            "band": BANDS[b], "races": n, "pred": float(t["pred"].mean()), "big_rate": float(t["big"].mean()),
            "median_pay": float(t["pay_trio"].median()),
            **{f"hit{k}": hit[k] / n for k in hit}, **{f"roi{k}": ret[k] / (100 * k * n) for k in ret},
        })
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(te["big"], te["pred"]))
    except ImportError:
        auc = None

    full = lgb.train(PARAMS, lgb.Dataset(R[FEATURES], R["big"]), max(best, 50))
    MODELS.mkdir(exist_ok=True)
    full.save_model(str(MODELS / "keiba_upset.txt"))
    meta = {"features": FEATURES, "cuts": cuts, "bands": BANDS, "stats": stats, "auc": auc, "big": BIG,
            "base_rate": float(R["big"].mean()), "period": [te["date"].min(), te["date"].max()], "races": len(te)}
    (MODELS / "keiba_upset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))

    pct = lambda x: f"{x * 100:.1f}%"
    md = ["# 競馬の荒れ度（3連複の払い戻しが1万円以上になる確率）", "",
          f"答え合わせ：{meta['period'][0]} 〜 {meta['period'][1]}（{len(te):,}レース、学習に使っていない期間）。"
          f"全体で1万円以上になった割合は {pct(meta['base_rate'])}。"
          + (f"見分ける力（AUC）は {auc:.3f}（0.5ででたらめ、1.0で完全）。" if auc else ""),
          "オッズは確定オッズを使っているため、締切前のオッズで判断する実際よりも少し良く出ます。", "",
          "| 荒れ度 | レース数 | 予想した確率 | 実際に1万円以上 | 払い戻しの中央値 | 人気順の上位3点 的中率 | 回収率 | 上位1点 的中率 | 回収率 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in stats:
        md.append(f"| {s['band']} | {s['races']} | {pct(s['pred'])} | {pct(s['big_rate'])} | {s['median_pay']:,.0f}円 | "
                  f"{pct(s['hit3'])} | {pct(s['roi3'])} | {pct(s['hit1'])} | {pct(s['roi1'])} |")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "keiba_upset.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


# ------------------------------------------------------------ 当日の予想
_model = None


def predict(race: dict, odds: dict[int, float], ages: list | None = None) -> dict | None:
    """荒れ度を返す。オッズがまだなければ None。"""
    global _model
    path = MODELS / "keiba_upset.txt"
    if len([v for v in odds.values() if v]) < 3 or not path.exists():
        return None
    if _model is None:
        _model = lgb.Booster(model_file=str(path))
    meta = json.loads((MODELS / "keiba_upset_meta.json").read_text())
    f = race_features(race, odds, ages)
    p = float(_model.predict(pd.DataFrame([f])[meta["features"]].astype(float))[0])
    b = band_of(p, meta["cuts"])
    s = meta["stats"][b]
    advice = ("荒れにくいレースです。人気順の3連複上位3点が当たりやすい条件です" if b <= 1 else
              "荒れやすいレースです。人気薄が絡みやすく、人気順の3連複は当たりにくい条件です（見送りも検討）" if b >= 3 else
              "荒れ方はふつうです")
    return {"prob": p, "band": meta["bands"][b], "band_index": b, "advice": advice, "past": s}


if __name__ == "__main__":
    main()
