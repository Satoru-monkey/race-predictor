"""netkeiba から JRA の過去レース成績ページを取得して保存する。

    python -m keiba.download 2025-01 2026-11     # 月の範囲（新しい月から取る）

個人利用の範囲で、2.5秒以上の間隔をあけて取得する。取得済みのページは取り直さない。
"""
import datetime as dt
import gzip
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "keiba" / "raw"
HEADERS = {"User-Agent": "Mozilla/5.0 (race-predictor; personal use)"}
JRA_VENUES = {f"{i:02d}" for i in range(1, 11)}
INTERVAL = 2.5

_session = requests.Session()
_last = [0.0]


def get(url: str) -> bytes:
    wait = INTERVAL - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    for pause in (0, 10, 30, 60, 120):  # ときどき400が返るので、間をあけて取り直す
        if pause:
            time.sleep(pause)
        r = _session.get(url, headers=HEADERS, timeout=30)
        _last[0] = time.time()
        if r.status_code == 200:
            break
    r.raise_for_status()
    return r.content


def kaisai_dates(year: int, month: int) -> list[str]:
    t = get(f"https://race.netkeiba.com/top/calendar.html?year={year}&month={month}").decode("euc-jp", "replace")
    return sorted(d for d in set(re.findall(r"kaisai_date=(\d{8})", t)) if d.startswith(f"{year}{month:02d}"))


def race_ids(date: str) -> list[str]:
    t = get(f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date}").decode("utf-8", "replace")
    ids = set(re.findall(r"race_id=(\d{12})", t))
    return sorted(i for i in ids if i[4:6] in JRA_VENUES)


def race_path(race_id: str) -> Path:
    return RAW / race_id[:4] / f"{race_id}.html.gz"


def fetch_race(race_id: str) -> Path:
    p = race_path(race_id)
    if p.exists():
        return p
    html = get(f"https://db.netkeiba.com/race/{race_id}/")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(gzip.compress(html))
    return p


def months(start: str, end: str):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out[::-1]


def main(argv, skip: set[str] | None = None, deadline: float | None = None, newest_first: bool = True):
    """skip：取得しない日付（すでに表にある日）。deadline：この時刻（time.time()）を過ぎたら止める。"""
    today = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).date()  # 日本時間
    start = argv[0] if argv else "2025-01"
    end = argv[1] if len(argv) > 1 else f"{today:%Y-%m}"
    index = ROOT / "data" / "keiba" / "dates.txt"
    done = set(index.read_text().split()) if index.exists() else set()
    done |= skip or set()
    for y, m in (months(start, end) if newest_first else months(start, end)[::-1]):
        if deadline and time.time() > deadline:
            print("時間になったので取得を止めます（次回は続きから）", flush=True)
            return
        try:
            dates = kaisai_dates(y, m)
        except Exception as e:
            print(y, m, "カレンダー取得エラー", e, flush=True)
            continue
        for d in sorted(dates, reverse=True):
            if d >= f"{today:%Y%m%d}" or d in done:
                continue  # 今日以降はまだ結果がない
            if deadline and time.time() > deadline:
                print("時間になったので取得を止めます（次回は続きから）", flush=True)
                return
            try:
                ids = race_ids(d)
                for rid in ids:
                    if not race_path(rid).exists():
                        fetch_race(rid)
                done.add(d)
                index.parent.mkdir(parents=True, exist_ok=True)
                index.write_text("\n".join(sorted(done)))
                print(d, f"{len(ids)}レース ok", flush=True)
            except Exception as e:
                print(d, "エラー", e, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
