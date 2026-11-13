"""Claude がチャットで予想を答えるためのコマンド。結果を Markdown の文章で出す。

    python ask.py boat list [日付] [--top 10]          競艇：3連複の確信度が高いレース一覧
    python ask.py boat race 福岡 12 [日付]             競艇：1レースの予想（直前情報・オッズ込み）
    python ask.py keiba list [日付]                     競馬：その日のレース一覧
    python ask.py keiba race 京都 11 [日付]            競馬：1レースの予想
    python ask.py report                                答え合わせの要約

日付は 2026-11-15 / 11-15 / 今日 / 明日 のどれでもよい（省略すると今日。競馬は次の土日）。
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JST = dt.timezone(dt.timedelta(hours=9))


def today() -> dt.date:
    return dt.datetime.now(JST).date()


def parse_date(s: str | None, weekend: bool = False) -> dt.date:
    t = today()
    if not s or s == "今日":
        if weekend:
            while t.weekday() < 5:
                t += dt.timedelta(days=1)
        return t
    if s == "明日":
        return t + dt.timedelta(days=1)
    if s == "あさって":
        return t + dt.timedelta(days=2)
    s = s.replace("/", "-")
    if s.count("-") == 1:
        s = f"{t.year}-{s}"
    y, m, d = map(int, s.split("-"))
    return dt.date(y, m, d)


def pct(x) -> str:
    return "-" if x is None else f"{x * 100:.0f}%"


def venue_code(name: str, table: dict) -> str:
    for code, v in table.items():
        if name in (code, v, v + "競艇場", v + "競馬場"):
            return code
    raise SystemExit(f"会場「{name}」が見つかりません。候補：{'、'.join(table.values())}")


# ------------------------------------------------------------ 競艇
def boat_list(day: dt.date, top: int):
    from boatrace import predict
    races = predict.day_overview(day)
    if not races:
        print(f"{day} の番組表が見つかりません（番組表は前日の夕方ごろに出ます）。")
        return
    races.sort(key=lambda r: -r["trio_p"])
    print(f"## 競艇 {day}　3連複の確信度が高いレース（全{len(races)}レース中 上位{top}）\n")
    print("番組表だけの予想です。展示タイム・オッズ込みの予想は、レースを指定して聞いてください。\n")
    print("| 会場 | R | 締切 | 3連複 上位3点 | どれかが当たる確率 | 1着予想 | 1着確率 |")
    print("|---|---|---|---|---|---|---|")
    for r in races[:top]:
        print(f"| {r['venue']} | {r['rno']} | {r['deadline'] or '-'} | {'、'.join(r['trio_top'])} | "
              f"**{pct(r['trio_p'])}** | {r['top_boat']}号艇 {r['top_name']} | {pct(r['top_p'])} |")


def boat_race(venue: str, rno: int, day: dt.date):
    from boatrace import predict
    from boatrace.parse import VENUES
    r = predict.predict_race(day, venue_code(venue, VENUES), rno)
    model = "展示タイムあり" if r["model"] == "ex" else "番組表だけ（展示タイムはまだ出ていません）"
    print(f"## 競艇 {r['venue']}{r['rno']}R {r['title']}（{r['date']} 締切 {r['deadline'] or '-'}）\n")
    print(f"- 予想に使ったモデル：{model}")
    w = (r.get("before") or {}).get("weather") or {}
    if w and w.get("wind") is not None:
        print(f"- 気象：{w.get('weather', '')} 風{w.get('wind', '-')}m 波{w.get('wave', '-')}cm")
    print(f"- **3連複の確信度：上位3点のどれかが当たる確率 {pct(r['trio_conf'])}**")
    print(f"- 1着の確信度：予想1位の1着確率 {pct(r['confidence'])}")
    for e in r["errors"]:
        print(f"- 注意：{e}")
    print("\n### 予想順位\n")
    print("| 順 | 艇 | 選手 | 1着 | 2着内 | 3着内 | 展示 | 根拠（＋押し上げ／－押し下げ） |")
    print("|---|---|---|---|---|---|---|---|")
    for i, b in enumerate(r["boats"], 1):
        why = " ".join([f"＋{t}" for t in b["plus"]] + [f"－{t}" for t in b["minus"]])
        print(f"| {i} | {b['boat']} | {b['name']}（{b['class']}） | {pct(b['win'])} | {pct(b['top2'])} | "
              f"{pct(b['top3'])} | {b['ex_time'] or '-'} | {why} |")
    _trio(r["trio"])


# ------------------------------------------------------------ 競馬
def keiba_list(day: dt.date):
    from keiba import predict
    races = predict.day_races(day)
    if not races:
        print(f"{day} のJRAのレースは見つかりません（出馬表は開催の2〜3日前に出ます）。")
        return
    print(f"## 競馬 {day} のレース一覧\n")
    by = {}
    for r in races:
        by.setdefault(r["venue"], []).append(r)
    for v, rs in by.items():
        print(f"**{v}**：" + "、".join(f"{r['rno']}R {r['title']}（{r['start'] or '-'}）" for r in sorted(rs, key=lambda x: x["rno"])))
    print("\n予想は「京都11R」のように、会場とレース番号を指定して聞いてください。")


def keiba_race(venue: str, rno: int, day: dt.date):
    from keiba import predict
    from keiba.parse import VENUES
    code = venue_code(venue, VENUES)
    rid = next((r["race_id"] for r in predict.day_races(day) if r["race_id"][4:6] == code and int(r["race_id"][10:12]) == rno), None)
    if not rid:
        raise SystemExit(f"{day} の {VENUES[code]}{rno}R が見つかりません。")
    r = predict.predict_race(rid, day)
    surface = {0: "芝", 1: "ダート"}.get(r["race"].get("surface"), "")
    print(f"## 競馬 {r['venue']}{r['rno']}R {r['title']}（{r['date']} 発走 {r['start'] or '-'}・{surface}{r['race'].get('distance') or ''}m・{r['race']['heads']}頭）\n")
    print(f"- 予想のしかた：{'能力モデル＋単勝オッズ' if r['mode'] == 'blend' else '能力モデルだけ（オッズ未発表）'}")
    print(f"- **3連複の確信度：上位3点のどれかが当たる確率 {pct(r['trio_conf'])}**")
    print(f"- 1着の確信度：予想1位の1着確率 {pct(r['confidence'])}")
    for e in r["errors"]:
        print(f"- 注意：{e}")
    print("\n### 予想順位（上位8頭）\n")
    print("| 順 | 馬番 | 馬名 | 騎手 | 1着 | 3着内 | 単勝 | 根拠 |")
    print("|---|---|---|---|---|---|---|---|")
    for i, h in enumerate(r["horses"][:8], 1):
        why = " ".join([f"＋{t}" for t in h["plus"]] + [f"－{t}" for t in h["minus"]])
        print(f"| {i} | {h['umaban']} | {h['horse']} | {h['jockey']} | {pct(h['win'])} | {pct(h['top3'])} | {h['odds'] or '-'} | {why} |")
    _trio(r["trio"])


def _trio(t: dict):
    print("\n### 3連複 上位5点\n")
    print("| 買い目 | 確率 | オッズ | 期待値 |")
    print("|---|---|---|---|")
    for c in t["top"][:5]:
        ev = f"{c['ev']:.2f}" if c["ev"] else "-"
        print(f"| {c['combo']} | {c['p'] * 100:.1f}% | {c['odds'] or '-'} | {ev} |")
    print("\n### 3連複の買い方ごと\n")
    print("| 買い方 | 買い目 | このレースで当たる確率 | 期待値 | 過去の的中率／回収率 |")
    print("|---|---|---|---|---|")
    for p in t["plans"]:
        past = f"{pct(p['past']['hit_rate'])}／{pct(p['past']['roi'])}" if p.get("past") and p["past"].get("hit_rate") is not None else "-"
        ev = f"{p['ev']:.2f}" if p["ev"] else "-"
        print(f"| {p['label']}（{len(p['combos'])}点） | {'、'.join(p['combos'])} | {pct(p['hit_p'])} | {ev} | {past} |")


def report():
    for name, path in (("競艇（展示タイムありモデル）", ROOT / "models" / "meta.json"),
                       ("競馬（能力モデル＋オッズ）", ROOT / "models" / "keiba_meta.json")):
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        b = m["ex"]["backtest"] if "ex" in m else m["backtest"]["blend"]
        print(f"## {name}　{b['period'][0]}〜{b['period'][1]}（{b['races']:,}レース、学習に使っていない期間）\n")
        print(f"- 1着的中率 {pct(b['win_hit_rate'])}")
        for t in b.get("trio", []):
            print(f"- {t['label']}（{t['points']}点）：的中率 {pct(t['hit_rate'])}／回収率 {pct(t['roi'])}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["boat", "keiba", "report"])
    ap.add_argument("action", nargs="?", default="list")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    if a.kind == "report":
        return report()
    if a.action == "list":
        day = parse_date(a.args[0] if a.args else None, weekend=a.kind == "keiba")
        return boat_list(day, a.top) if a.kind == "boat" else keiba_list(day)
    if a.action == "race":
        if len(a.args) < 2:
            raise SystemExit("会場とレース番号を指定してください（例：福岡 12）")
        venue, rno = a.args[0], int(a.args[1].rstrip("RrＲ"))
        day = parse_date(a.args[2] if len(a.args) > 2 else None, weekend=a.kind == "keiba")
        return boat_race(venue, rno, day) if a.kind == "boat" else keiba_race(venue, rno, day)
    raise SystemExit("action は list か race です")


if __name__ == "__main__":
    sys.exit(main())
