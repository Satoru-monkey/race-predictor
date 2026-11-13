"""当日の情報（番組表・直前情報・オッズ）を取得する。

公式サイトに負担をかけないよう、取得は最低1.5秒あけ、結果はしばらく使い回す（キャッシュ）。
"""
import datetime as dt
import itertools
import re
import threading
import time
import warnings

import pandas as pd
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from .download import RAW, fetch
from .parse import parse_b_lines, read_lines

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

SITE = "https://www.boatrace.jp/owpc/pc/race"
HEADERS = {"User-Agent": "race-predictor (personal use)"}
_lock = threading.Lock()
_last = [0.0]
_cache: dict[str, tuple[float, object]] = {}
_session = requests.Session()


def _get(url: str, ttl: float) -> str:
    now = time.time()
    hit = _cache.get(url)
    if hit and now - hit[0] < ttl:
        return hit[1]
    with _lock:
        wait = 1.5 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        r = _session.get(url, headers=HEADERS, timeout=30)
        _last[0] = time.time()
    r.raise_for_status()
    r.encoding = "utf-8"
    _cache[url] = (time.time(), r.text)
    return r.text


def _soup(url: str, ttl: float) -> BeautifulSoup:
    return BeautifulSoup(_get(url, ttl), "lxml")


def _num(s: str):
    m = re.search(r"-?(?:\d+\.?\d*|\.\d+)", s or "")
    return float(m.group()) if m else None


# ------------------------------------------------------------ 番組表
def program(day: dt.date) -> pd.DataFrame:
    """その日の全レースの出走表。公式の番組表ファイルを使う。"""
    path = RAW / "B" / f"B{day:%y%m%d}.TXT"
    if not path.exists():
        fetch("B", day, _session)
    if not path.exists():
        return pd.DataFrame()
    df = pd.DataFrame(parse_b_lines(read_lines(path), day.isoformat()))
    if df.empty:
        return df
    df["race_id"] = df["date"].str.replace("-", "") + df["jcd"] + df["rno"].astype(str).str.zfill(2)
    return df


# ------------------------------------------------------------ 直前情報
def beforeinfo(jcd: str, rno: int, day: dt.date) -> dict:
    """展示タイム・チルト・スタート展示・気象。まだ出ていなければ ex_time は None。"""
    s = _soup(f"{SITE}/beforeinfo?rno={rno}&jcd={jcd}&hd={day:%Y%m%d}", ttl=120)
    boats = {}
    tbl = s.select_one("table.is-w748")
    if tbl:
        for tb in tbl.select("tbody"):
            tds = [td.get_text(strip=True) for td in tb.select("tr")[0].select("td")]
            if len(tds) < 6 or not tds[0].isdigit():
                continue
            boats[int(tds[0])] = {"ex_time": _num(tds[4]), "tilt": _num(tds[5]),
                                  "weight_today": _num(tds[3])}
    st_tbl = s.select_one("table.is-w238")
    if st_tbl:
        for course, d in enumerate(st_tbl.select(".table1_boatImage1"), start=1):
            num = d.select_one("[class*=boatImage1Number]")
            if not num:
                continue
            m = re.search(r"is-type(\d)", " ".join(num.get("class", [])))
            if not m:
                continue
            st_txt = d.select_one("[class*=boatImage1Time]")
            st_txt = st_txt.get_text(strip=True) if st_txt else ""
            b = boats.setdefault(int(m.group(1)), {})
            b["course_ex"] = course
            b["st_ex_text"] = st_txt
            b["st_ex"] = (-_num(st_txt) if st_txt.startswith("F") else _num(st_txt)) if _num(st_txt) is not None else None
    weather = {}
    w = s.select_one(".weather1")
    if w:
        for u in w.select(".weather1_bodyUnit"):
            cls, txt = " ".join(u.get("class", [])), u.get_text(" ", strip=True)
            if "is-wind" in cls and "Direction" not in cls:
                weather["wind"] = _num(txt)
            elif "is-wave" in cls:
                weather["wave"] = _num(txt)
            elif "is-weather" in cls:
                weather["weather"] = txt
    ready = bool(boats) and all(b.get("ex_time") for b in boats.values() if "ex_time" in b)
    return {"boats": boats, "weather": weather, "ready": ready}


# ------------------------------------------------------------ オッズ
def odds_win(jcd: str, rno: int, day: dt.date) -> dict[int, float]:
    s = _soup(f"{SITE}/oddstf?rno={rno}&jcd={jcd}&hd={day:%Y%m%d}", ttl=60)
    out = {}
    for tbl in s.select("table"):
        if "単勝オッズ" not in tbl.get_text():
            continue
        for tr in tbl.select("tbody tr"):
            tds = [td.get_text(strip=True) for td in tr.select("td")]
            if len(tds) >= 3 and tds[0].isdigit():
                v = _num(tds[-1])
                if v:
                    out[int(tds[0])] = v
        break
    return out


def odds_trifecta(jcd: str, rno: int, day: dt.date) -> dict[tuple, float]:
    """3連単オッズ。表は「列＝1着の艇、行＝2着・3着の組み合わせ」の順に並んでいる。"""
    s = _soup(f"{SITE}/odds3t?rno={rno}&jcd={jcd}&hd={day:%Y%m%d}", ttl=60)
    vals = [_num(td.get_text(strip=True)) for td in s.select("td.oddsPoint")]
    if len(vals) != 120:
        return {}
    rows_by_first = {
        f: [(a, b) for a in range(1, 7) if a != f for b in range(1, 7) if b not in (f, a)]
        for f in range(1, 7)
    }
    out = {}
    for i, v in enumerate(vals):
        row, col = divmod(i, 6)
        first = col + 1
        a, b = rows_by_first[first][row]
        if v:
            out[(first, a, b)] = v
    return out


def all_trifectas():
    return list(itertools.permutations(range(1, 7), 3))
