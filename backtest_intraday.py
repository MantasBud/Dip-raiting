#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intraday backtestas — testuoja TIKRA sandori: ijejimas dienos viduryje, isejimas ta pacia diena.

Paleidimas:
    pip install yfinance pandas numpy tzdata
    python backtest_intraday.py                # 60 d., visos sarasos akcijos
    python backtest_intraday.py --target 3

KUO SKIRIASI NUO backtest.py
backtest.py naudoja dienos barus: ijejimas uzdarymo kaina, rezultatas kita diena.
Sis naudoja 5 min. barus ir vertina tuos pacius rodiklius, kuriuos mato realus
modulis: VWAP, RSI(5min), RVOL pagal valanda, 1 ir 3 val. krypti, dienos maksimuma.
Ijejimo taskai tikrinami kelis kartus per diena, kaip ir realiai ziurint i moduli.

APRIBOJIMAI
1. Yahoo duoda tik 60 dienu 5 min. istorijos — tai riboja imti.
2. Kai tame paciame bare paliecti ir tikslas, ir stop, laikoma pralaimejimu.
3. Mokesciai ir spread'as neiskaiciuoti.
4. Tie patys keli ijejimo taskai per diena yra susije tarpusavyje, todel
   efektyvus nepriklausomu atveju skaicius mazesnis nei eiluciu skaicius.
"""

import argparse
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

import dip_reitingas as dr

CHECKPOINTS_5M = [18, 30, 42, 54, 66]   # ~10:30, 11:30, 12:30, 13:30, 14:30
CHECKPOINTS_60M = [1, 3, 5]             # valandiniai barai: ~10:00, 12:00, 14:00


def session_frames(intra, min_bars):
    """Suskaido duomenis i sesijas pagal data."""
    return [(d, g) for d, g in intra.groupby(intra.index.date) if len(g) >= min_bars]


def rvol_at(sessions, day_idx, k):
    """Apyvarta iki k-tojo baro, palyginti su tomis paciomis valandomis anksciau."""
    today = sessions[day_idx][1]
    cur = float(today["Volume"].iloc[:k + 1].sum())
    prev = []
    for i in range(max(0, day_idx - 10), day_idx):
        g = sessions[i][1]
        if len(g) > k:
            prev.append(float(g["Volume"].iloc[:k + 1].sum()))
    if not prev or cur <= 0:
        return None
    base = float(np.median(prev))
    return cur / base if base > 0 else None


def build_snapshot(sessions, day_idx, k, daily_hist, rsi_series, target, bph=12):
    """Atkuria tiksliai ta vaizda, kuri modulis matytu ta minute."""
    day, today = sessions[day_idx]
    bars = today.iloc[:k + 1]
    price = float(bars["Close"].iloc[-1])
    high = float(bars["High"].max())
    low = float(bars["Low"].min())
    if not price or high <= low:
        return None

    tp = (bars["High"] + bars["Low"] + bars["Close"]) / 3
    vol = bars["Volume"].replace(0, np.nan)
    vwap = float((tp * vol).sum() / vol.sum()) if vol.sum() > 0 else None

    try:
        rsi = float(rsi_series.loc[bars.index[-1]])
    except Exception:
        rsi = None

    a = dr.atr_pct(daily_hist)
    sup, res = dr.levels(daily_hist, price)
    ctx = dr.multiday_context(daily_hist, price)
    mom = dr.short_momentum(bars, bph=bph)
    v5 = dr.intraday_vol(bars)

    prev_close = float(daily_hist["Close"].iloc[-1])
    day_chg = (price - prev_close) / prev_close * 100 if prev_close else None

    res_intra = high if high > price * 1.001 else None
    sup_intra = low if low < price * 0.999 else None
    hi20 = float(daily_hist["High"].tail(20).max())
    cands = sorted(x for x in (res_intra, res, hi20) if x and x > price * 1.001)

    return dict(
        price=price, dayHigh=high, dayLow=low, vwap=vwap, rsi=rsi, atrPct=a,
        support=sup, resistance=res, sup_intra=sup_intra, res_intra=res_intra,
        res_list=cands, rvol=rvol_at(sessions, day_idx, k),
        sma20=float(daily_hist["Close"].tail(20).mean()),
        sma50=float(daily_hist["Close"].tail(50).mean()),
        earnings=False, day_chg=day_chg,
        down_days=ctx["down_days"], dd5=ctx["dd5"], chg3d=ctx["chg3d"],
        avgVolume=float(daily_hist["Volume"].tail(20).mean()),
        cur="EUR", cur_sym="\u20ac", vol5m=v5,
        exp_move=dr.expected_move(v5, dr.HOLD_HOURS, bph=bph),
        m1h=mom["m1h"], m3h=mom["m3h"], pos1h=mom["pos1h"],
        span_h=mom["span_h"], mom_partial=mom["partial"])


def outcome(sessions, day_idx, k, entry, stop, target_price, hold_hours):
    """Ka kaina padare po ijejimo: tikslas, stop ar nei viena."""
    bars = [sessions[day_idx][1].iloc[k + 1:]]
    if hold_hours > 8 and day_idx + 1 < len(sessions):
        bars.append(sessions[day_idx + 1][1])
    future = pd.concat(bars) if bars else None
    if future is None or future.empty:
        return None, 0.0

    for _, b in future.iterrows():
        hit_t = float(b["High"]) >= target_price
        hit_s = float(b["Low"]) <= stop
        if hit_t and hit_s:
            return "stop", (stop - entry) / entry * 100
        if hit_t:
            return "tikslas", (target_price - entry) / entry * 100
        if hit_s:
            return "stop", (stop - entry) / entry * 100

    last = float(future["Close"].iloc[-1])
    return "be rezultato", (last - entry) / entry * 100


def main():
    ap = argparse.ArgumentParser(description="Intraday backtestas")
    ap.add_argument("--target", type=float, default=dr.TARGET_PCT)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--interval", default="5m", choices=["5m", "60m"],
                    help="5m = tikslus, bet tik 60 d.; 60m = mazesnis tikslumas, bet iki 2 metu")
    ap.add_argument("--sweep", action="store_true",
                    help="Isbandyti skirtingus stop ir tikslo derinius, ir parodyti, kuris geriausias")
    args = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Paleisk: pip install yfinance pandas numpy tzdata")

    symbols = [s for _, s, _ in dr.WATCHLIST]
    if args.interval == "60m":
        bph, checkpoints, min_bars = 1, CHECKPOINTS_60M, 5
        days = min(args.days, 720) if args.days > 60 else 720
    else:
        bph, checkpoints, min_bars = 12, CHECKPOINTS_5M, 40
        days = min(args.days, 60)

    print(f"Siunciama {len(symbols)} akciju {args.interval} istorija ({days} d.)…")
    intraday = yf.download(symbols, period=f"{days}d", interval=args.interval,
                           group_by="ticker", progress=False, auto_adjust=False, threads=True)
    daily = yf.download(symbols, period="max" if args.interval == "60m" else "12mo",
                        interval="1d", group_by="ticker", progress=False,
                        auto_adjust=False, threads=True)

    rows = []
    for tag, sym, _ in dr.WATCHLIST:
        try:
            intra = dr.flatten(intraday, sym).dropna(subset=["Close"])
            dhist = dr.flatten(daily, sym).dropna(subset=["Close", "High", "Low"])
            if intra.empty or len(dhist) < 60:
                continue
            sessions = session_frames(intra, min_bars)
            rsi_series = dr.rsi(intra["Close"])

            for di, (day, _) in enumerate(sessions):
                if di < 1 or di + 1 >= len(sessions):
                    continue
                hist = dhist[dhist.index.date < day]
                if len(hist) < 55:
                    continue

                for k in checkpoints:
                    if k + max(2, bph // 2) >= len(sessions[di][1]):
                        continue
                    d = build_snapshot(sessions, di, k, hist, rsi_series, args.target, bph)
                    if not d:
                        continue
                    s = dr.score_stock(d, args.target, "neutral")
                    res, pnl = outcome(sessions, di, k, d["price"], s["stop"],
                                       s["tp"], dr.HOLD_HOURS)
                    if res is None:
                        continue
                    rows.append(dict(tag=tag, score=s["score"], grade=s["grade"],
                                     tradeable=bool(s.get("tradeable")),
                                     setup=s.get("setup"), result=res, pnl=pnl,
                                     **{f"c_{key}": s["parts"][key] for key, _, _ in dr.CRITERIA},
                                     _day=sessions[di][0],
                                     _sym=sym, _di=di, _k=k, _price=d["price"],
                                     _atr=d.get("atrPct") or 3.0, _sessions=sessions))
        except Exception as e:
            print(f"  {tag}: praleista ({str(e)[:50]})")

    if not rows:
        sys.exit("Nepavyko surinkti duomenu.")

    df = pd.DataFrame(rows)
    print(f"\nIstirta ijejimo tasku: {len(df)}  |  tikslas {args.target}%  "
          f"|  laikymas {dr.HOLD_HOURS} val.\n")

    # --- Pagal balo intervala (ne pagal pakopa: taip matosi tikra priklausomybe) ---
    bins = [0, 40, 50, 60, 70, 80, 101]
    labels = ["<40", "40-50", "50-60", "60-70", "70-80", "80+"]
    df["bucket"] = pd.cut(df["score"], bins=bins, labels=labels, right=False)

    base = df["pnl"].mean()
    print(f"BAZINE LINIJA (atsitiktinis ijejimas, visi {len(df)} taskai): {base:+.3f}% sandoriui")
    print("Bet kuris intervalas turi jа iveikti, kad atranka turetu verte.\n")

    print(f"{'BALAS':<8} {'ATVEJU':>7} {'TIKSLAS':>9} {'STOP':>8} {'VIDUT. REZ.':>12} {'PRIES BAZE':>11}")
    print("-" * 50)
    for lab in labels:
        g = df[df["bucket"] == lab]
        if len(g) < 10:
            continue
        print(f"{lab:<8} {len(g):>7} {(g['result']=='tikslas').mean()*100:>8.1f}% "
              f"{(g['result']=='stop').mean()*100:>7.1f}% {g['pnl'].mean():>11.2f}% "
              f"{g['pnl'].mean()-base:>+10.2f}%")

    # --- Ar tinkamumo zyma ka nors reiskia ---
    print(f"\n{'TINKAMUMAS':<14} {'ATVEJU':>7} {'TIKSLAS':>9} {'VIDUT. REZ.':>12}")
    print("-" * 46)
    for val, name in [(True, "atitinka"), (False, "netinkama")]:
        g = df[df["tradeable"] == val]
        if len(g) >= 10:
            print(f"{name:<14} {len(g):>7} {(g['result']=='tikslas').mean()*100:>8.1f}% "
                  f"{g['pnl'].mean():>11.2f}%")

    # --- Pagal scenariju ---
    print(f"\n{'SCENARIJUS':<14} {'ATVEJU':>7} {'TIKSLAS':>9} {'VIDUT. REZ.':>12}")
    print("-" * 46)
    for st in df["setup"].dropna().unique():
        g = df[df["setup"] == st]
        if len(g) >= 10:
            print(f"{st:<14} {len(g):>7} {(g['result']=='tikslas').mean()*100:>8.1f}% "
                  f"{g['pnl'].mean():>11.2f}%")

    # --- Ar kuris nors ATSKIRAS kriterijus turi verte? ---
    print(f"\n{'KRITERIJUS':<26} {'ZEMAS (<50)':>13} {'AUKSTAS (>75)':>15} {'SKIRTUMAS':>11}")
    print("-" * 68)
    signals = []
    for key, label, _ in dr.CRITERIA:
        col = f"c_{key}"
        if col not in df:
            continue
        lo = df[df[col] < 50]["pnl"]
        hi = df[df[col] > 75]["pnl"]
        if len(lo) < 200 or len(hi) < 200:
            continue
        diff = hi.mean() - lo.mean()
        signals.append((abs(diff), label, diff, len(lo), len(hi)))
        print(f"{label:<26} {lo.mean():>+12.3f}% {hi.mean():>+14.3f}% {diff:>+10.3f}%")

    if signals:
        signals.sort(reverse=True)
        _, lab, diff, _, _ = signals[0]
        print(f"\nStipriausias atskiras signalas: {lab} ({diff:+.3f} p. p.)")
        print("Kad butu vertas demesio, tas pats turi kartotis IR kitame laikotarpyje.")

    # --- V2 hipoteze: balas, sudarytas TIK is to, kas kartojosi abiejuose imtyse ---
    # Zenklai paimti is kriteriju analizes: teigiamas tik "vienadienis ar testinis",
    # visi kiti veikia atvirksciai, todel apverciami. RSI ir trendas ismesti (triuksmas).
    def score_v2(r):
        def inv(key):
            v = r.get(f"c_{key}")
            return None if v is None else 100 - v
        parts = [(r.get("c_multiday"), 35), (inv("dip"), 15), (inv("rvol"), 15),
                 (inv("vwap"), 12), (inv("support"), 12), (inv("atr"), 11)]
        vals = [(v, w) for v, w in parts if v is not None]
        if not vals:
            return None
        tot = sum(w for _, w in vals)
        return sum(v * w for v, w in vals) / tot

    df["score2"] = df.apply(score_v2, axis=1)

    if df["score2"].notna().sum() > 500 and "_day" in df:
        df["_day"] = pd.to_datetime(df["_day"])
        mid = df["_day"].median()
        halves = [("1-oji puse", df[df["_day"] <= mid]), ("2-oji puse", df[df["_day"] > mid])]

        print("\n" + "=" * 68)
        print("V2 HIPOTEZE: balas tik is to, kas kartojosi. Tikrinama per dvi laiko puses,")
        print("kad matytusi, ar veikia UZ tos imties, pagal kuria sudarytas.")
        print("=" * 68)
        print(f"{'LAIKOTARPIS':<14} {'BAZE':>8} {'DABARTINIS top10%':>18} {'V2 top10%':>12} {'V2 PRIES BAZE':>15}")
        print("-" * 68)
        for name, g in halves:
            if len(g) < 200:
                continue
            b = g["pnl"].mean()
            # Virsutinis decilis, o ne fiksuota riba: skales skiriasi, todel
            # fiksuota 75 riba vienam balui duoda tukstancius atveju, kitam - nulio
            cur = g[g["score"] >= g["score"].quantile(0.9)]["pnl"]
            v2 = g[g["score2"] >= g["score2"].quantile(0.9)]["pnl"]
            cur_s = f"{cur.mean():+.3f}%" if len(cur) >= 50 else "per maza"
            v2_s = f"{v2.mean():+.3f}%" if len(v2) >= 50 else "per maza"
            diff = f"{v2.mean() - b:+.3f}%" if len(v2) >= 50 else "-"
            print(f"{name:<14} {b:>+7.3f}% {cur_s:>16} {v2_s:>10} {diff:>15}")
        print("\nKad V2 butu vertas, jis turi iveikti baze ABIEJOSE pusese.")

    # --- Ar balo verte priklauso nuo rinkos krypties? ---
    # Rinkos rodiklis: visu 19 akciju mediana 5 dienu pokytis tuo metu.
    try:
        df["_day"] = pd.to_datetime(df["_day"])
        daily_mkt = df.groupby("_day")["pnl"].size()          # tik dienu sarasui
        day_list = sorted(df["_day"].unique())
        idx = {d: i for i, d in enumerate(day_list)}
        # kiekvienai dienai - vidutinis visu akciju rezultatas kaip rinkos artinys
        day_mean = df.groupby("_day")["pnl"].mean()
        roll = day_mean.rolling(5, min_periods=3).mean()
        df["_mkt"] = df["_day"].map(roll)

        sub = df.dropna(subset=["_mkt"])
        if len(sub) > 1000:
            q1, q2 = sub["_mkt"].quantile(0.33), sub["_mkt"].quantile(0.67)
            regimes = [("Rinka krenta", sub[sub["_mkt"] <= q1]),
                       ("Rinka soninе", sub[(sub["_mkt"] > q1) & (sub["_mkt"] < q2)]),
                       ("Rinka kyla", sub[sub["_mkt"] >= q2])]
            print("\n" + "=" * 68)
            print("AR BALAS PRIKLAUSO NUO RINKOS KRYPTIES?")
            print("=" * 68)
            print(f"{'REZIMAS':<16} {'ATVEJU':>8} {'BAZE':>9} {'TOP 10% BALAS':>15} {'PRIES BAZE':>12}")
            print("-" * 68)
            for name, g in regimes:
                if len(g) < 200:
                    continue
                b = g["pnl"].mean()
                top = g[g["score"] >= g["score"].quantile(0.9)]["pnl"]
                if len(top) < 30:
                    continue
                print(f"{name:<16} {len(g):>8} {b:>+8.3f}% {top.mean():>+14.3f}% "
                      f"{top.mean()-b:>+11.3f}%")
            print("\nJei skirtumas teigiamas tik kylanciai rinkai — balas nera pranasumas,")
            print("o rinkos krypties stiprintuvas, ir ji reikia naudoti tik su filtru.")
    except Exception as e:
        print(f"(rezimo analize praleista: {str(e)[:60]})")

    # --- Parametru paieska: koks stop ir koks tikslas realiai veikia ---
    if args.sweep:
        print("\n" + "=" * 64)
        print("PARAMETRU PAIESKA: vidutinis rezultatas vienam sandoriui (%)")
        print("Stop = daugiklis x dienos ATR;  eilutes = stop, stulpeliai = tikslas")
        print("=" * 64)
        targets = [1.5, 2.0, 3.0, 4.0]
        mults = [0.25, 0.4, 0.55, 0.75, 1.0, 1.5]

        header = "STOP".ljust(10) + "".join(f"{t:>10.1f}%" for t in targets)
        print(header)
        print("-" * len(header))
        best = None
        for m in mults:
            cells = []
            for t in targets:
                pnls = []
                for r in rows:
                    price, atr = r["_price"], r["_atr"]
                    stop_pct = max(0.5, m * atr)
                    stop = price * (1 - stop_pct / 100)
                    tp = price * (1 + t / 100)
                    res, pnl = outcome(r["_sessions"], r["_di"], r["_k"], price, stop, tp,
                                       dr.HOLD_HOURS)
                    if res:
                        pnls.append(pnl)
                avg = float(np.mean(pnls)) if pnls else 0.0
                cells.append(avg)
                if best is None or avg > best[0]:
                    best = (avg, m, t)
            print(f"{m:>4.2f}xATR  " + "".join(f"{c:>+10.2f}" for c in cells))

        print("-" * len(header))
        print(f"Geriausias derinys: stop {best[1]:.2f} x ATR, tikslas {best[2]:.1f}% "
              f"-> {best[0]:+.2f}% sandoriui")
        print("Nepamirsk: mokesciai ir spread'as cia neiskaiciuoti (~0.05-0.1% sandoriui).")

    # --- Isvada ---
    valid = [(lab, df[df["bucket"] == lab]) for lab in labels]
    valid = [(l, g) for l, g in valid if len(g) >= 20]
    if len(valid) >= 2:
        lo_pnl = valid[0][1]["pnl"].mean()
        hi_pnl = valid[-1][1]["pnl"].mean()
        print(f"\nZemiausias intervalas ({valid[0][0]}): {lo_pnl:+.2f}% vidutiniskai")
        print(f"Auksciausias intervalas ({valid[-1][0]}): {hi_pnl:+.2f}% vidutiniskai")
        d = hi_pnl - lo_pnl
        if d > 0.15:
            print(f"→ Balas veikia teisinga kryptimi: skirtumas {d:+.2f} proc. punkto sandoriui.")
        elif d < -0.15:
            print(f"→ Balas veikia ATVIRKSCIAI ({d:+.2f} p. p.). Svorius butina perziureti.")
        else:
            print(f"→ Balas kol kas neskiria gerų sandoriu nuo blogu ({d:+.2f} p. p.).")

    print("\nSVARBU: be mokesciu ir spread'o; kai bare paliesti abu lygiai — "
          "\nlaikoma pralaimejimu; keli ijejimo taskai per diena yra susije tarpusavyje.")


if __name__ == "__main__":
    main()
