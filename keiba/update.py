"""競馬のデータを最新にして、モデルを学び直す。

    python -m keiba.update                       # 新しい開催日を取得 → 表に追加 → 学習
    python -m keiba.update --backfill 90         # さらに、まだない古い月（2025年1月〜）を最大90分取得
    python -m keiba.update --no-train

GitHub に保存した月ごとの表（data/store/keiba/）にある日は取り直さない。
"""
import datetime as dt
import sys
import time

from boatrace import store

from . import download, model, parse

FIRST_MONTH = "2025-01"
KEYS = ["race_id", "umaban"]


def main(argv):
    old = store.load("keiba")
    have = set(old["date"].str.replace("-", "")) if not old.empty else set()
    today = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).date()  # 日本時間
    if not old.empty:
        print(f"保存済みの表：{old['race_id'].nunique():,} レース（{old['date'].min()}〜{old['date'].max()}）")
        newest_month = old["date"].max()[:7]
    else:
        newest_month = FIRST_MONTH
    print(f"{newest_month} 以降の新しい開催日を取得します")
    download.main([newest_month, f"{today:%Y-%m}"], skip=have)
    if "--backfill" in argv:
        minutes = float(argv[argv.index("--backfill") + 1])
        oldest = old["date"].min()[:7] if not old.empty else f"{today:%Y-%m}"
        if oldest > FIRST_MONTH:
            print(f"古い月（{FIRST_MONTH}〜{oldest}）のうち、まだない日を最大{minutes:.0f}分取得します")
            download.main([FIRST_MONTH, oldest], skip=have, deadline=time.time() + minutes * 60)
    print("表に変換しています…")
    new = parse.build()
    df = store.merge(old, new, KEYS)
    store.save(df, "keiba", months=set(new["date"].str[:7]) if not new.empty else set())
    out = parse.ROOT / "data" / "keiba" / "races.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"{df.race_id.nunique():,} レース分のデータになりました（{df['date'].min()}〜{df['date'].max()}）")
    if "--no-train" not in argv:
        print("学習と答え合わせをしています…")
        model.main()


if __name__ == "__main__":
    main(sys.argv[1:])
