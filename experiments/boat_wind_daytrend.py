"""新しい特徴量（風向き・当日の場の流れ）を足して、答え合わせの成績を比べる。"""
import json
import sys

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from boatrace import features as F, model as M, store

df = store.load("boat")
df = F.add_history(df)
df = F.add_form(df)
df = F.add_basic(df)
df = F.add_wind(df)
df = F.add_day_trend(df, F.results_long(df))
df["y"] = (df["rank"] == 1).astype(int)

dates = sorted(df["date"].unique())
test_start, valid_start = dates[-60], dates[-90]
tr = df[df["date"] < valid_start]
va = df[(df["date"] >= valid_start) & (df["date"] < test_start)]
te = df[df["date"] >= test_start]

variants = {
    "noex": {"今まで": F.FEATURES_NO_EX, "+風向き": F.FEATURES_NO_EX + F.WIND_COLS,
             "+当日の流れ": F.FEATURES_NO_EX + F.DAY_COLS,
             "+両方": F.FEATURES_NO_EX + F.WIND_COLS + F.DAY_COLS},
    "ex": {"今まで": F.FEATURES_EX, "+風向き": F.FEATURES_EX + F.WIND_COLS,
           "+当日の流れ": F.FEATURES_EX + F.DAY_COLS,
           "+両方": F.FEATURES_EX + F.WIND_COLS + F.DAY_COLS},
}
out = {}
for model_name, vs in variants.items():
    for vname, feats in vs.items():
        m = M.fit(tr, feats, va)
        best = m.best_iteration
        lam = M.tune_lambda(va, M.normalize(va, m.predict(va[feats], num_iteration=best)))
        p = M.normalize(te, m.predict(te[feats], num_iteration=best))
        bt = M.backtest(te, p, lam)
        trio = {r["plan"]: r for r in bt["trio"]}
        # 平和島だけ・後半レース（7R以降）だけの上位3点的中率も見る
        sub = {}
        for sname, mask in (("平和島", te["jcd"] == "04"), ("7R以降", te["rno"] >= 7)):
            s = te[mask]
            ps = M.normalize(s, m.predict(s[feats], num_iteration=best))
            b2 = M.backtest(s, ps, lam)
            sub[sname] = {r["plan"]: r["hit_rate"] for r in b2["trio"]}["top3"]
        row = {"logloss": bt["logloss"], "win_hit": bt["win_hit_rate"],
               "trio1": trio["top1"]["hit_rate"], "trio3": trio["top3"]["hit_rate"],
               "trio3_roi": trio["top3"]["roi"], "trio5": trio["top5"]["hit_rate"],
               "box4": trio["box4"]["hit_rate"], "trio3_平和島": sub["平和島"], "trio3_7R以降": sub["7R以降"],
               "races": bt["races"], "rounds": best}
        out[f"{model_name} {vname}"] = row
        print(model_name, vname, json.dumps(row, ensure_ascii=False), flush=True)
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
