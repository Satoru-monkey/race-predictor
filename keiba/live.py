"""競馬の当日情報（レース一覧・出馬表・オッズ）を取得する。間隔をあけ、結果は使い回す。"""
import json
import re
import threading
import time
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .download import HEADERS, JRA_VENUES
from .parse import SURFACE, class_level

_lock = threading.Lock()
_last = [0.0]
_cache: dict[str, tuple[float, str]] = {}
_session = requests.Session()


def nfkc(s):
    return unicodedata.normalize("NFKC", s or "")


def _get(url: str, ttl: float) -> str:
    hit = _cache.get(url)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    with _lock:
        for pause in (0, 5, 15):  # ときどき400が返るので、少し待って取り直す
            time.sleep(max(pause, 2.0 - (time.time() - _last[0])))
            r = _session.get(url, headers=HEADERS, timeout=30)
            _last[0] = time.time()
            if r.status_code == 200:
                break
    r.raise_for_status()
    raw = r.content
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("euc-jp", "replace")
    _cache[url] = (time.time(), text)
    return text


def race_list(date: str) -> list[dict]:
    """その日のJRAのレース一覧（date は 20261115 の形）。"""
    s = BeautifulSoup(_get(f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date}", 600), "lxml")
    out, seen = [], set()
    for li in s.select("li.RaceList_DataItem"):
        a = li.select_one("a[href*='race_id=']")
        if not a:
            continue
        rid = re.search(r"race_id=(\d{12})", a["href"]).group(1)
        if rid in seen or rid[4:6] not in JRA_VENUES:
            continue
        seen.add(rid)
        title = li.select_one(".ItemTitle")
        tm = li.select_one(".RaceList_Itemtime")
        out.append({"race_id": rid, "title": nfkc(title.get_text(strip=True)) if title else "",
                    "start": nfkc(tm.get_text(strip=True)) if tm else None})
    return out


def _grade(s) -> int | None:
    """重賞マーク（G1〜G3・リステッド）からレースの格を読む。"""
    box = s.select_one(".RaceName")
    m = re.search(r"Icon_GradeType(\d+)", str(box)) if box else None
    return {"1": 9, "2": 8, "3": 7, "15": 6}.get(m.group(1)) if m else None


def shutuba(race_id: str) -> tuple[dict, pd.DataFrame]:
    s = BeautifulSoup(_get(f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}", 300), "lxml")
    d1 = nfkc((s.select_one(".RaceData01") or s).get_text(" ", strip=True))
    d2 = nfkc((s.select_one(".RaceData02") or s).get_text(" ", strip=True))
    name = nfkc((s.select_one(".RaceName") or s).get_text(" ", strip=True))
    m = re.search(r"(芝|ダ|障)(\d{3,4})m\s*\(?\s*(右|左|直線)?", d1)
    going = re.search(r"馬場:(\S+)", d1)
    from .parse import GOING
    race = {
        "race_id": race_id, "jcd": race_id[4:6], "rno": int(race_id[10:12]), "title": name,
        "surface": SURFACE.get(m.group(1)) if m else None, "distance": int(m.group(2)) if m else None,
        "direction": {"右": 0, "左": 1, "直線": 2}.get(m.group(3) if m else None),
        "going": GOING.get(going.group(1)) if going else None,
        "start": (re.search(r"(\d{1,2}:\d{2})発走", d1) or [None, None])[1],
        "class": _grade(s) or class_level(name + " " + d2), "handicap": int("ハンデ" in d2),
        "female": int("牝" in d2.split("日目")[-1]),
    }
    rows = []
    for tr in s.select("tr.HorseList"):
        if "Cancel" in " ".join(tr.get("class", [])):
            continue  # 出走取消
        tds = tr.select("td")
        txt = [nfkc(td.get_text(" ", strip=True)) for td in tds]
        if len(txt) < 9 or not txt[1].isdigit():
            continue
        href = lambda k: (re.search(rf"/{k}/(?:result/recent/)?(\w+)", " ".join(a["href"] for a in tr.select("a[href]"))) or [None, None])[1]
        sexage = txt[4] if len(txt) > 4 else ""
        wt = re.match(r"(\d+)\(([+-]?\d+)\)", txt[8].replace(" ", "")) if len(txt) > 8 else None
        rows.append({
            "race_id": race_id, "waku": int(txt[0]) if txt[0].isdigit() else None,
            "umaban": int(txt[1]) if txt[1].isdigit() else None,
            "horse": tr.select_one(".HorseName").get_text(strip=True) if tr.select_one(".HorseName") else txt[3],
            "horse_id": href("horse"), "jockey_id": href("jockey"), "trainer_id": href("trainer"),
            "jockey": txt[6] if len(txt) > 6 else "",
            "sex": {"牡": 0, "牝": 1, "セ": 2}.get(sexage[:1]),
            "age": int(sexage[1:]) if sexage[1:].isdigit() else None,
            "kinryo": float(txt[5]) if re.fullmatch(r"\d+(\.\d)?", txt[5] if len(txt) > 5 else "") else None,
            "weight": int(wt.group(1)) if wt else None, "weight_diff": int(wt.group(2)) if wt else None,
        })
    df = pd.DataFrame(rows)
    race["heads"] = len(df)
    return race, df


def _odds(race_id: str, typ: int) -> dict:
    t = _get(f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type={typ}&action=update", 60)
    try:
        d = json.loads(t)
        return (d.get("data") or {}).get("odds", {}).get(str(typ), {}) or {}
    except (ValueError, AttributeError):
        return {}


def odds_win(race_id: str) -> dict[int, float]:
    out = {}
    for k, v in _odds(race_id, 1).items():
        try:
            out[int(k)] = float(v[0])
        except (ValueError, IndexError):
            pass
    return out


def odds_trifecta(race_id: str) -> dict[tuple, float]:
    out = {}
    for k, v in _odds(race_id, 8).items():
        try:
            out[(int(k[0:2]), int(k[2:4]), int(k[4:6]))] = float(v[0])
        except (ValueError, IndexError):
            pass
    return out


def odds_trio(race_id: str) -> dict[tuple, float]:
    out = {}
    for k, v in _odds(race_id, 7).items():
        try:
            out[tuple(sorted((int(k[0:2]), int(k[2:4]), int(k[4:6]))))] = float(v[0])
        except (ValueError, IndexError):
            pass
    return out
