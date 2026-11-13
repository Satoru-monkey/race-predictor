"""データを最新にして、モデルを学び直す。

    python -m boatrace.update            # 足りない日の分だけ取得 → 表に変換 → 学習
    python -m boatrace.update --no-train # 取得と変換だけ
"""
import datetime as dt
import sys

from . import download, model, parse

FIRST_DAY = "2025-08-01"


def main(argv):
    end = dt.date.today() - dt.timedelta(days=1)
    start = dt.date.fromisoformat(argv[0]) if argv and not argv[0].startswith("-") else dt.date.fromisoformat(FIRST_DAY)
    print(f"{start} 〜 {end} のうち、まだ持っていない日のデータを取得します（1日あたり数秒〜十数秒）")
    download.main([start.isoformat(), end.isoformat()])
    print("表に変換しています…")
    df = parse.build()
    df.to_parquet(parse.ROOT / "data" / "races.parquet")
    print(f"{df.race_id.nunique():,} レース分のデータになりました")
    if "--no-train" not in argv:
        print("学習と答え合わせをしています（数分かかります）…")
        model.main()


if __name__ == "__main__":
    main(sys.argv[1:])
