"""競艇のデータを最新にして、モデルを学び直す。

    python -m boatrace.update            # 足りない日の分だけ取得 → 表に追加 → 学習
    python -m boatrace.update --no-train # 取得と表の更新だけ

GitHub に保存した月ごとの表（data/store/boat/）があれば、その続きの日だけを取得する。
"""
import datetime as dt
import sys

from . import download, model, parse, store

FIRST_DAY = "2025-08-01"
KEYS = ["race_id", "boat"]


def main(argv):
    old = store.load("boat")
    end = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).date() - dt.timedelta(days=1)  # 日本時間の昨日
    if not old.empty:
        start = dt.date.fromisoformat(old["date"].max()) + dt.timedelta(days=1)
        print(f"保存済みの表：{old['race_id'].nunique():,} レース（{old['date'].min()}〜{old['date'].max()}）")
    else:
        start = dt.date.fromisoformat(FIRST_DAY)
    if start <= end:
        print(f"{start} 〜 {end} のデータを取得します")
        download.main([start.isoformat(), end.isoformat()])
    print("表に変換しています…")
    new = parse.build()
    if not old.empty and not new.empty:
        new = new[new["date"] >= start.isoformat()]
    df = store.merge(old, new, KEYS)
    store.save(df, "boat", months=set(new["date"].str[:7]) if not new.empty else set())
    df.to_parquet(parse.ROOT / "data" / "races.parquet")
    print(f"{df.race_id.nunique():,} レース分のデータになりました（{df['date'].min()}〜{df['date'].max()}）")
    if "--no-train" not in argv:
        print("学習と答え合わせをしています…")
        model.main()


if __name__ == "__main__":
    main(sys.argv[1:])
