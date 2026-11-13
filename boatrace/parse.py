"""番組表(B)・競走成績(K)のテキストを表(DataFrame)に変換する。

    python -m boatrace.parse      # data/raw/ を読んで data/races.parquet を作る
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

VENUES = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島", "05": "多摩川", "06": "浜名湖",
    "07": "蒲郡", "08": "常滑", "09": "津", "10": "三国", "11": "びわこ", "12": "住之江",
    "13": "尼崎", "14": "鳴門", "15": "丸亀", "16": "児島", "17": "宮島", "18": "徳山",
    "19": "下関", "20": "若松", "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村",
}
CLASS_NUM = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}


def z2h(s: str) -> str:
    """全角英数字を半角に。"""
    return unicodedata.normalize("NFKC", s)


def _f(s: str):
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return None


def race_kind(title: str) -> str:
    t = z2h(title)
    for key in ("優勝", "準優", "ドリーム", "特選", "選抜", "特賞", "予選", "一般"):
        if key in t:
            return key
    return "その他"


# ---------------------------------------------------------------- 番組表
def parse_b_lines(lines: list[bytes], date: str) -> list[dict]:
    rows, jcd, rno, title, dist, deadline = [], None, None, "", None, None
    race_re = re.compile(r"^\s*(\d{1,2})R\s+(.*?)\s+H(\d+)m")
    for raw in lines:
        if len(raw) >= 6 and raw[2:6] == b"BBGN":
            jcd = raw[:2].decode()
            continue
        text = raw.decode("cp932", errors="replace")
        m = race_re.match(z2h(text))
        if m:
            rno, title, dist = int(m.group(1)), m.group(2).strip(), int(m.group(3))
            dl = re.search(r"締切予定\s*(\d{1,2}:\d{2})", z2h(text))
            deadline = dl.group(1) if dl else None
            continue
        if jcd and rno and len(raw) >= 64 and raw[:1] in b"123456" and raw[1:2] == b" " and raw[2:6].isdigit():
            g = lambda a, b: raw[a:b].decode("cp932", errors="replace")
            series = re.findall(r"[1-6]", z2h(g(64, 76)))
            rows.append({
                "date": date, "jcd": jcd, "rno": rno, "title": title, "kind": race_kind(title),
                "distance": dist, "deadline": deadline, "boat": int(g(0, 1)), "toban": int(g(2, 6)),
                "name": g(6, 14).replace("　", "").strip(), "age": _f(g(14, 16)),
                "branch": g(16, 20).strip(), "weight": _f(g(20, 22)),
                "class": CLASS_NUM.get(g(22, 24), 0), "nat_win": _f(g(24, 29)),
                "nat_2": _f(g(29, 35)), "loc_win": _f(g(35, 40)), "loc_2": _f(g(40, 46)),
                "motor_no": _f(g(46, 49)), "motor_2": _f(g(49, 55)), "boat_no": _f(g(55, 58)),
                "boat_2": _f(g(58, 64)),
                "series_n": len(series),
                "series_avg": (sum(map(int, series)) / len(series)) if series else None,
            })
    return rows


# ---------------------------------------------------------------- 競走成績
RES_RE = re.compile(
    r"^\s+(\S{1,2})\s+([1-6])\s+(\d{4})\s+\S.*?\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)"
)
HEAD_RE = re.compile(r"^\s+(\d{1,2})R\s+.*?H(\d+)m\s+(\S+)\s+風\s+(\S+)\s+(\d+)m\s+波\s+(\d+)cm")


def parse_k_lines(lines: list[bytes], date: str) -> tuple[list[dict], list[dict]]:
    results, races, jcd, cur = [], [], None, None
    for raw in lines:
        if len(raw) >= 6 and raw[2:6] == b"KBGN":
            jcd = raw[:2].decode()
            continue
        if not jcd:
            continue
        text = z2h(raw.decode("cp932", errors="replace"))
        m = HEAD_RE.match(text)
        if m:
            cur = {"date": date, "jcd": jcd, "rno": int(m.group(1)), "weather": m.group(3),
                   "wind_dir": m.group(4), "wind": int(m.group(5)), "wave": int(m.group(6))}
            races.append(cur)
            continue
        if cur is None:
            continue
        m = RES_RE.match(text)
        if m and (m.group(1).isdigit() or m.group(1)[0] in "FLKS"):
            rank = int(m.group(1)) if m.group(1).isdigit() else 7  # 失格・欠場などは7扱い
            ex = _f(m.group(6))
            course = int(m.group(7)) if m.group(7).isdigit() else None
            st = m.group(8)
            results.append({
                "date": date, "jcd": jcd, "rno": cur["rno"], "boat": int(m.group(2)),
                "toban": int(m.group(3)), "rank": rank, "flag": m.group(1),
                "ex_time": ex if ex and 6.0 < ex < 8.0 else None, "course": course,
                "st": _f(st) if re.fullmatch(r"\d?\.\d+", st) else None,
            })
            continue
        s = text.strip()
        if s.startswith("単勝"):
            p = s.split()
            if len(p) >= 3 and p[2].isdigit():
                cur["pay_win"], cur["win_boat"] = int(p[2]), p[1]
        elif s.startswith("3連単"):
            p = s.split()
            if len(p) >= 3 and p[2].isdigit():
                cur["tri"], cur["pay_tri"] = p[1], int(p[2])
                if "人気" in p:
                    cur["pop_tri"] = int(p[p.index("人気") + 1])
        elif s.startswith("3連複"):
            p = s.split()
            if len(p) >= 3 and p[2].isdigit():
                cur["trio"], cur["pay_trio"] = p[1], int(p[2])
        elif s.startswith("2連単"):
            p = s.split()
            if len(p) >= 3 and p[2].isdigit():
                cur["exa"], cur["pay_exa"] = p[1], int(p[2])
    return results, races


def date_from_name(name: str) -> str:
    y, m, d = name[1:3], name[3:5], name[5:7]
    return f"20{y}-{m}-{d}"


def read_lines(path: Path) -> list[bytes]:
    return path.read_bytes().split(b"\r\n")


def build(raw_dir: Path = RAW) -> pd.DataFrame:
    prog, res, races = [], [], []
    for p in sorted((raw_dir / "B").glob("B*.TXT")):
        prog += parse_b_lines(read_lines(p), date_from_name(p.name))
    for p in sorted((raw_dir / "K").glob("K*.TXT")):
        r, rc = parse_k_lines(read_lines(p), date_from_name(p.name))
        res += r
        races += rc
    prog, res, races = pd.DataFrame(prog), pd.DataFrame(res), pd.DataFrame(races)
    races = races.drop_duplicates(["date", "jcd", "rno"])
    df = prog.merge(res, on=["date", "jcd", "rno", "boat", "toban"], how="inner")
    df = df.merge(races, on=["date", "jcd", "rno"], how="left")
    # 1着がいないレース（中止・不成立）は除く
    ok = df.groupby(["date", "jcd", "rno"])["rank"].transform(lambda s: (s == 1).any())
    df = df[ok].copy()
    df["race_id"] = df["date"].str.replace("-", "") + df["jcd"] + df["rno"].astype(str).str.zfill(2)
    return df.sort_values(["date", "jcd", "rno", "boat"]).reset_index(drop=True)


if __name__ == "__main__":
    out = ROOT / "data" / "races.parquet"
    df = build()
    df.to_parquet(out)
    print(f"{df.race_id.nunique()} レース / {len(df)} 行 を {out} に保存しました")
