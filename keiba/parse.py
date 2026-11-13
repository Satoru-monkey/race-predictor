"""netkeiba の成績ページを表(DataFrame)に変換する。

    python -m keiba.parse     # data/keiba/raw/ → data/keiba/races.parquet
"""
import gzip
import re
import unicodedata
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "keiba" / "raw"

VENUES = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
          "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
GOING = {"良": 0, "稍重": 1, "稍": 1, "重": 2, "不良": 3, "不": 3}
SURFACE = {"芝": 0, "ダ": 1, "障": 2}


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")


def decode(raw: bytes) -> str:
    for enc in ("utf-8", "euc-jp"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", "replace")


def class_level(text: str) -> int:
    """レースの格を数字に（大きいほど上）。"""
    t = nfkc(text)
    if re.search(r"G1|GI\b|\(GI\)", t):
        return 9
    if re.search(r"G2|GII\b|\(GII\)", t):
        return 8
    if re.search(r"G3|GIII\b|\(GIII\)", t):
        return 7
    if "(L)" in t or "リステッド" in t:
        return 6
    if "オープン" in t or "OP" in t:
        return 5
    if "3勝" in t or "1600万" in t:
        return 4
    if "2勝" in t or "1000万" in t:
        return 3
    if "1勝" in t or "500万" in t:
        return 2
    if "未勝利" in t:
        return 1
    if "新馬" in t:
        return 0
    return 2


def time_sec(s: str):
    m = re.fullmatch(r"(?:(\d+):)?(\d+)\.(\d)", s.strip())
    if not m:
        return None
    return int(m.group(1) or 0) * 60 + int(m.group(2)) + int(m.group(3)) / 10


def _id(tr, kind: str):
    a = tr.select_one(f'a[href*="/{kind}/"]')
    if not a:
        return None
    m = re.search(rf"/{kind}/(?:result/recent/)?(\w+)/?", a["href"])
    return m.group(1) if m else None


def parse_race_html(html: str, race_id: str) -> tuple[dict, list[dict]]:
    s = BeautifulSoup(html, "lxml")
    intro = s.select_one(".data_intro")
    if intro is None:
        return {}, []
    head = nfkc(intro.get_text(" ", strip=True))
    small = nfkc((s.select_one("p.smalltxt") or intro).get_text(" ", strip=True))
    title = nfkc((intro.select_one("h1") or intro).get_text(" ", strip=True))
    m = re.search(r"(芝|ダ|障)\S*?(右|左|直線)?\S*?(\d{3,4})m", head)
    date = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", small)
    race = {
        "race_id": race_id, "jcd": race_id[4:6], "rno": int(race_id[10:12]),
        "date": f"{date.group(1)}-{int(date.group(2)):02d}-{int(date.group(3)):02d}" if date else None,
        "surface": SURFACE.get(m.group(1)) if m else None,
        "direction": {"右": 0, "左": 1, "直線": 2}.get(m.group(2) if m else None),
        "distance": int(m.group(3)) if m else None,
        "weather": (re.search(r"天候\s*:\s*(\S+)", head) or [None, None])[1],
        "going": GOING.get((re.search(r"(?:芝|ダート)\s*:\s*(\S+)", head) or [None, ""])[1], None),
        "start": (re.search(r"発走\s*:\s*(\d{1,2}:\d{2})", head) or [None, None])[1],
        "title": title, "class": class_level(title + " " + small),
        "handicap": int("ハンデ" in small), "female": int("牝" in small.split("日目")[-1]),
    }
    table = s.select_one("table.race_table_01")
    if table is None:
        return race, []
    trs = table.select("tr")
    header = [nfkc(th.get_text(strip=True)) for th in trs[0].select("th,td")]

    def col(name):
        for i, h in enumerate(header):
            if h.startswith(name):
                return i
        return None

    idx = {k: col(k) for k in ("着順", "枠番", "馬番", "馬名", "性齢", "斤量", "騎手", "タイム",
                               "通過", "上り", "単勝", "人気", "馬体重")}
    rows = []
    for tr in trs[1:]:
        td = [nfkc(x.get_text(strip=True)) for x in tr.select("td")]
        if len(td) < len(header) - 2:
            continue
        g = lambda k: td[idx[k]] if idx[k] is not None and idx[k] < len(td) else ""
        rank_txt = g("着順")
        if rank_txt in ("取", "除"):
            continue  # 出走取消・除外
        rank = int(re.sub(r"\D", "", rank_txt)) if re.search(r"\d", rank_txt) else 99
        sexage = g("性齢")
        wt = re.match(r"(\d+)\(([+-]?\d+)\)", g("馬体重"))
        passing = [int(x) for x in g("通過").split("-") if x.isdigit()]
        try:
            odds = float(g("単勝"))
        except ValueError:
            odds = None
        rows.append({
            "race_id": race_id, "rank": rank, "waku": int(g("枠番") or 0), "umaban": int(g("馬番") or 0),
            "horse_id": _id(tr, "horse"), "horse": g("馬名"),
            "sex": {"牡": 0, "牝": 1, "セ": 2}.get(sexage[:1]), "age": int(sexage[1:]) if sexage[1:].isdigit() else None,
            "kinryo": float(g("斤量")) if re.fullmatch(r"\d+(\.\d)?", g("斤量")) else None,
            "jockey_id": _id(tr, "jockey"), "jockey": g("騎手"), "trainer_id": _id(tr, "trainer"),
            "time": time_sec(g("タイム")), "last3f": float(g("上り")) if re.fullmatch(r"\d+\.\d", g("上り")) else None,
            "pass_first": passing[0] if passing else None, "pass_last": passing[-1] if passing else None,
            "odds": odds, "pop": int(g("人気")) if g("人気").isdigit() else None,
            "weight": int(wt.group(1)) if wt else None, "weight_diff": int(wt.group(2)) if wt else None,
        })
    # 払い戻し
    for t in s.select("table.pay_table_01"):
        for tr in t.select("tr"):
            th = tr.select_one("th")
            tds = tr.select("td")
            if not th or len(tds) < 2:
                continue
            kind = nfkc(th.get_text(strip=True))
            combos = [nfkc(x) for x in tds[0].stripped_strings]
            pays = [int(nfkc(x).replace(",", "")) for x in tds[1].stripped_strings if nfkc(x).replace(",", "").isdigit()]
            if kind == "単勝" and pays:
                race["pay_win"] = pays[0]
            elif kind == "三連単" and pays:
                race["tri"], race["pay_tri"] = combos[0].replace(" ", ""), pays[0]
            elif kind == "馬連" and pays:
                race["quin"], race["pay_quin"] = combos[0].replace(" ", ""), pays[0]
    race["heads"] = len(rows)
    return race, rows


def build(raw_dir: Path = RAW) -> pd.DataFrame:
    races, rows = [], []
    for p in sorted(raw_dir.glob("*/*.html.gz")):
        rid = p.name[:12]
        try:
            r, rr = parse_race_html(decode(gzip.decompress(p.read_bytes())), rid)
        except Exception as e:
            print("読み取りエラー", rid, e)
            continue
        if r and rr:
            races.append(r)
            rows += rr
    df = pd.DataFrame(rows).merge(pd.DataFrame(races), on="race_id")
    ok = df.groupby("race_id")["rank"].transform(lambda s: (s == 1).any())
    df = df[ok & df["date"].notna()].copy()
    return df.sort_values(["date", "race_id", "umaban"]).reset_index(drop=True)


if __name__ == "__main__":
    out = ROOT / "data" / "keiba" / "races.parquet"
    df = build()
    df.to_parquet(out)
    print(f"{df.race_id.nunique():,} レース / {len(df):,} 頭 を {out} に保存しました")
