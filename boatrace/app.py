"""ブラウザで使う予想画面。

    python -m boatrace.app        # http://localhost:8000 を開く
"""
import datetime as dt
import json
from pathlib import Path

from flask import Flask, abort, render_template, request

from keiba import live as keiba_live
from keiba import predict as keiba_predict

from . import live, predict
from .download import RAW, fetch
from .parse import VENUES, parse_k_lines, read_lines

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data" / "log"
JST = dt.timezone(dt.timedelta(hours=9))
app = Flask(__name__)
app.jinja_env.globals["labels"] = predict.label


def today() -> dt.date:
    return dt.datetime.now(JST).date()


def conf_label(p: float) -> str:
    return "高" if p >= 0.7 else ("中" if p >= 0.55 else "低")


@app.route("/")
def index():
    day = dt.date.fromisoformat(request.args.get("date") or today().isoformat())
    sort = request.args.get("sort", "conf")
    high = request.args.get("high") == "1"
    races = predict.day_overview(day)
    for r in races:
        r["label"] = conf_label(r["top_p"])
    if high:
        races = [r for r in races if r["label"] == "高"]
    if sort == "time":
        races.sort(key=lambda r: (r["deadline"] or "99", r["venue"]))
    else:
        races.sort(key=lambda r: -r["top_p"])
    return render_template("index.html", races=races, date=day.isoformat(), sort=sort, high=high)


@app.route("/race/<date>/<jcd>/<int:rno>")
def race(date, jcd, rno):
    if jcd not in VENUES or not 1 <= rno <= 12:
        abort(404)
    day = dt.date.fromisoformat(date)
    if request.args.get("refresh"):
        live._cache.clear()
    try:
        r = predict.predict_race(day, jcd, rno)
    except ValueError as e:
        return f"<p>{e}</p><p><a href='/'>戻る</a></p>", 404
    _save_log(r)
    return render_template("race.html", r=r)


# ------------------------------------------------------------ 予想の記録と答え合わせ
def _save_log(r: dict):
    """オッズが取れた予想を記録する（同じレースは最新で上書き）。"""
    if not any(b["odds"] for b in r["boats"]):
        return
    LOG.mkdir(parents=True, exist_ok=True)
    path = LOG / f"{r['date']}.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    key = f"{r['jcd']}-{r['rno']}"
    data[key] = {
        "jcd": r["jcd"], "rno": r["rno"], "model": r["model"],
        "saved_at": dt.datetime.now(JST).strftime("%H:%M"), "deadline": r["deadline"],
        "boats": [{"boat": b["boat"], "win": float(b["win"]), "odds": b["odds"]} for b in r["boats"]],
        "top_combo": r["top_combos"][0]["combo"],
        "value_bets": [{"combo": v["combo"], "odds": v["odds"], "p": float(v["p"])} for v in r["value_bets"]],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))


def _results(day: dt.date) -> dict:
    path = RAW / "K" / f"K{day:%y%m%d}.TXT"
    if not path.exists() and day < today():
        try:
            fetch("K", day, live._session)
        except Exception:
            pass
    if not path.exists():
        return {}
    _, races = parse_k_lines(read_lines(path), day.isoformat())
    return {f"{r['jcd']}-{r['rno']}": r for r in races}


@app.route("/log")
def log():
    rows, tot = [], {"races": 0, "hit": 0, "tri_hit": 0, "tri_ret": 0, "val_bets": 0, "val_ret": 0,
                     "ev_bets": 0, "ev_ret": 0}
    for path in sorted(LOG.glob("*.json"), reverse=True) if LOG.exists() else []:
        day = dt.date.fromisoformat(path.stem)
        res = _results(day)
        for key, p in json.loads(path.read_text()).items():
            r = res.get(key)
            row = {"date": path.stem, "venue": VENUES.get(p["jcd"]), "rno": p["rno"], "p": p,
                   "top": p["boats"][0]["boat"], "result": None}
            if r and r.get("tri"):
                win = int(r["tri"].split("-")[0])
                row["result"] = r["tri"]
                row["hit"] = row["top"] == win
                row["tri_hit"] = p["top_combo"] == r["tri"]
                row["val_ret"] = sum(v.get("odds", 0) * 100 for v in p["value_bets"] if v["combo"] == r["tri"])
                ev_boats = [b for b in p["boats"] if b["odds"] and b["win"] * b["odds"] >= 1.0]
                tot["ev_bets"] += len(ev_boats)
                tot["ev_ret"] += sum(r.get("pay_win", 0) for b in ev_boats if b["boat"] == win)
                tot["races"] += 1
                tot["hit"] += row["hit"]
                tot["tri_hit"] += row["tri_hit"]
                tot["tri_ret"] += r.get("pay_tri", 0) if row["tri_hit"] else 0
                tot["val_bets"] += len(p["value_bets"])
                tot["val_ret"] += row["val_ret"]
            rows.append(row)
    return render_template("log.html", rows=rows, t=tot)


@app.route("/report")
def report():
    try:
        km = keiba_predict.meta()
    except FileNotFoundError:
        km = None
    return render_template("report.html", m=predict.meta(), km=km)


# ------------------------------------------------------------ 競馬
@app.route("/keiba")
def keiba_index():
    day = dt.date.fromisoformat(request.args.get("date") or _next_weekend().isoformat())
    error, races = None, []
    try:
        races = keiba_predict.day_races(day)
    except Exception as e:
        error = f"レース一覧を取得できませんでした（{e}）"
    groups = {}
    for r in races:
        groups.setdefault(r["venue"], []).append(r)
    for v in groups.values():
        v.sort(key=lambda r: r["rno"])
    return render_template("keiba_index.html", groups=list(groups.items()), races=races,
                           date=day.isoformat(), error=error)


@app.route("/keiba/race/<race_id>")
def keiba_race(race_id):
    if not race_id.isdigit() or len(race_id) != 12:
        abort(404)
    day = dt.date.fromisoformat(request.args.get("date") or today().isoformat())
    if request.args.get("refresh"):
        keiba_live._cache.clear()
    try:
        r = keiba_predict.predict_race(race_id, day)
    except ValueError as e:
        return f"<p>{e}</p><p><a href='/keiba'>戻る</a></p>", 404
    except FileNotFoundError:
        return ("<p>競馬のモデルがまだありません。「競馬データ更新.bat」を実行してください。</p>"
                "<p><a href='/keiba'>戻る</a></p>"), 503
    return render_template("keiba_race.html", r=r)


def _next_weekend() -> dt.date:
    """今日が土日ならその日、平日なら次の土曜日。"""
    d = today()
    while d.weekday() < 5:
        d += dt.timedelta(days=1)
    return d


def main():
    import socket
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        ip = "このPCのIPアドレス"
    print("ブラウザで http://localhost:8000 を開いてください")
    print(f"同じWi-Fiのスマホからは http://{ip}:8000")
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()
