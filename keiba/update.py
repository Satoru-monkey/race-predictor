"""競馬のデータを最新にして、モデルを学び直す。

    python -m keiba.update              # 2025年1月から先月・今月まで、足りない分を取得 → 学習
    python -m keiba.update 2024-01      # 取得を始める月を指定
"""
import sys

from . import download, model, parse


def main(argv):
    start = argv[0] if argv and not argv[0].startswith("-") else "2025-01"
    print(f"{start} から今月までのレースのうち、まだ持っていないものを取得します（1レース数秒）")
    download.main([start])
    print("表に変換しています…")
    df = parse.build()
    df.to_parquet(parse.ROOT / "data" / "keiba" / "races.parquet")
    print(f"{df.race_id.nunique():,} レース分のデータになりました")
    if "--no-train" not in argv:
        print("学習と答え合わせをしています…")
        model.main()


if __name__ == "__main__":
    main(sys.argv[1:])
