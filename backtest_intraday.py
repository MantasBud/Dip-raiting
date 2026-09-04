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

CHECKPOINTS = [18, 30, 42, 54, 66]      # 5 min. barai: ~10:30, 11:30, 12:30, 13:30, 14:30


def session_frames(intra):
    """Suskaido 5 min. duomenis i sesijas pagal data."""
    return [(d, g) for d, g in intra.groupby(intra.index.date) if len(g) >= 40]


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


def build_snapshot(sessions, day_idx, k, daily_hist, rsi_series, target):
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
    mom = dr.short_momentum(bars)
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
        exp_move=dr.expected_move(v5, dr.HOLD_HOURS),
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
    args = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Paleisk: pip install yfinance pandas numpy tzdata")

    symbols = [s for _, s, _ in dr.WATCHLIST]
    print(f"Siunciama {len(symbols)} akciju 5 min. istorija ({args.days} d.)…")
    intraday = yf.download(symbols, period=f"{args.days}d", interval="5m",
                           group_by="ticker", progress=False, auto_adjust=False, threads=True)
    daily = yf.download(symbols, period="12mo", interval="1d",
                        group_by="ticker", progress=False, auto_adjust=False, threads=True)

    rows = []
    for tag, sym, _ in dr.WATCHLIST:
        try:
            intra = dr.flatten(intraday, sym).dropna(subset=["Close"])
            dhist = dr.flatten(daily, sym).dropna(subset=["Close", "High", "Low"])
            if intra.empty or len(dhist) < 60:
                continue
            sessions = session_frames(intra)
            rsi_series = dr.rsi(intra["Close"])

            for di, (day, _) in enumerate(sessions):
                if di < 1 or di + 1 >= len(sessions):
                    continue
                hist = dhist[dhist.index.date < day]
                if len(hist) < 55:
                    continue

                for k in CHECKPOINTS:
                    if k + 6 >= len(sessions[di][1]):
                        continue
                    d = build_snapshot(sessions, di, k, hist, rsi_series, args.target)
                    if not d:
                        continue
                    s = dr.score_stock(d, args.target, "neutral")
                    res, pnl = outcome(sessions, di, k, d["price"], s["stop"],
                                       s["tp"], dr.HOLD_HOURS)
                    if res is None:
                        continue
                    rows.append(dict(tag=tag, score=s["score"], grade=s["grade"],
                                     tradeable=bool(s.get("tradeable")),
                                     setup=s.get("setup"), result=res, pnl=pnl))
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

    print(f"{'BALAS':<8} {'ATVEJU':>7} {'TIKSLAS':>9} {'STOP':>8} {'VIDUT. REZ.':>12}")
    print("-" * 50)
    for lab in labels:
        g = df[df["bucket"] == lab]
        if len(g) < 10:
            continue
        print(f"{lab:<8} {len(g):>7} {(g['result']=='tikslas').mean()*100:>8.1f}% "
              f"{(g['result']=='stop').mean()*100:>7.1f}% {g['pnl'].mean():>11.2f}%")

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
