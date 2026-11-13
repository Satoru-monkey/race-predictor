"""ボートレース振興会の公開データ（番組表 B / 競走成績 K）をダウンロードして展開する。

使い方:
    python -m boatrace.download 2025-11-01 2026-11-12
"""
import datetime as dt
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lhafile
import requests

BASE = "https://www1.mbrace.or.jp/od2"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
HEADERS = {"User-Agent": "race-predictor (personal use)"}


def url_for(kind: str, day: dt.date) -> str:
    k = kind.upper()
    return f"{BASE}/{k}/{day:%Y%m}/{k.lower()}{day:%y%m%d}.lzh"


def fetch(kind: str, day: dt.date, session: requests.Session) -> Path | None:
    """1日分を取得して TXT を保存する。すでにあれば何もしない。"""
    out = RAW / kind.upper() / f"{kind.upper()}{day:%y%m%d}.TXT"
    if out.exists():
        return out
    r = session.get(url_for(kind, day), headers=HEADERS, timeout=30)
    if r.status_code != 200 or b"-lh" not in r.content[:10]:
        return None
    arc = lhafile.Lhafile(io.BytesIO(r.content))
    out.parent.mkdir(parents=True, exist_ok=True)
    for name in arc.namelist():
        out.write_bytes(arc.read(name))
    return out


def daterange(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def download_kind(kind: str, days: list[dt.date]):
    s = requests.Session()
    for day in days:
        out = RAW / kind / f"{kind}{day:%y%m%d}.TXT"
        if out.exists():
            continue
        try:
            got = fetch(kind, day, s)
            print(day, kind, "ok" if got else "なし", flush=True)
        except Exception as e:  # 通信エラーは飛ばして次へ
            print(day, kind, "エラー", e, flush=True)
        time.sleep(1.0)  # サーバーに負担をかけない


def main(argv):
    start = dt.date.fromisoformat(argv[0])
    end = dt.date.fromisoformat(argv[1]) if len(argv) > 1 else dt.date.today()
    days = list(daterange(start, end))[::-1]  # 新しい日付から取る
    # 番組表と成績を1本ずつ並行で取る（同時接続は2本まで）
    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(lambda k: download_kind(k, days), ("B", "K")))


if __name__ == "__main__":
    main(sys.argv[1:])
