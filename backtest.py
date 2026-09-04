#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dip reitingo backtest'as — ar aukštesnis balas realiai reiškia geresnį sandorį?

Paleidimas:
    pip install yfinance pandas numpy
    python backtest.py                 # 12 mėn., visos sąrašo akcijos
    python backtest.py --months 24

KĄ TIKRINA
Kiekvienai dienai kiekvienai akcijai suskaičiuoja balą (ta pačia logika kaip
dip_reitingas.py) ir tikrina, kas nutiko KITĄ dieną: ar kaina pasiekė +TARGET%
prieš atsitrenkdama į stop.

APRIBOJIMAI, KURIUOS BŪTINA ŽINOTI SKAITANT REZULTATĄ
1. Naudojami dienos barai, ne 5 min. Todėl VWAP, intraday RSI ir RVOL pagal
   valandą čia NEPRIEINAMI — jų dedamosios pakeistos neutraliu 50. Realus
   modulis turi daugiau informacijos nei šis testas.
2. Kai per dieną pasiekiamas ir tikslas, ir stop, nežinome, kuris buvo pirmas.
   Toks atvejis skaičiuojamas kaip PRALAIMĖJIMAS (konservatyvu).
3. Neįskaičiuoti mokesčiai, spread'as ir prasilenkimas su kaina (slippage).
   Realus rezultatas bus prastesnis nei rodo šis testas.
4. Tai istorija, ne prognozė.
"""

import argparse
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

import dip_reitingas as dr


def daily_score(hist, i, target):
    """Balas naudojant tik dienos barus iki i-tosios dienos imtinai."""
    window = hist.iloc[: i + 1]
    if len(window) < 55:
        return None

    row = window.iloc[-1]
    price = float(row["Close"])
    high, low = float(row["High"]), float(row["Low"])
    if not price or not high or high <= low:
        return None

    c = window["Close"]
    sma20, sma50 = float(c.tail(20).mean()), float(c.tail(50).mean())
    a = dr.atr_pct(window)
    sup, res = dr.levels(window, price)
    ctx = dr.multiday_context(window, price)

    vol = window["Volume"]
    rv = float(vol.iloc[-1] / vol.tail(20).mean()) if vol.tail(20).mean() > 0 else None

    prev_close = float(c.iloc[-2]) if len(c) > 1 else None
    day_chg = (price - prev_close) / prev_close * 100 if prev_close else None
    hi20 = float(window["High"].tail(20).max())
    res_intra = high if high > price * 1.001 else None
    cands = sorted(x for x in (res_intra, res, hi20) if x and x > price * 1.001)

    d = dict(price=price, dayHigh=high, dayLow=low, vwap=None, rsi=None,
             atrPct=a, support=sup, resistance=res, rvol=rv,
             sup_intra=low if low < price * 0.999 else None, res_intra=res_intra,
             res_list=cands, day_chg=day_chg,
             sma20=sma20, sma50=sma50, earnings=False,
             avgVolume=float(window["Volume"].tail(20).mean()),
             down_days=ctx["down_days"], dd5=ctx["dd5"], chg3d=ctx["chg3d"])
    return d, dr.score_stock(d, target, "neutral")


def evaluate_next_day(hist, i, entry, stop, target_price):
    """Kas nutiko kitą dieną: tikslas, stop, ar nei viena."""
    if i + 1 >= len(hist):
        return None
    nxt = hist.iloc[i + 1]
    hi, lo = float(nxt["High"]), float(nxt["Low"])

    hit_target = hi >= target_price
    hit_stop = lo <= stop

    if hit_target and hit_stop:
        return "loss"          # neaišku kuris pirmas — laikome pralaimėjimu
    if hit_target:
        return "win"
    if hit_stop:
        return "loss"
    # nei viena riba nepasiekta — uždarome dienos pabaigoje
    return "flat_win" if float(nxt["Close"]) > entry else "flat_loss"


def main():
    ap = argparse.ArgumentParser(description="Dip reitingo backtest'as")
    ap.add_argument("--months", type=int, default=12, help="Kiek mėnesių istorijos")
    ap.add_argument("--target", type=float, default=dr.TARGET_PCT, help="Tikslas %%")
    args = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Paleisk: pip install yfinance pandas numpy")

    symbols = [s for _, s, _ in dr.WATCHLIST]
    period = f"{max(args.months + 4, 12)}mo"
    print(f"Siunčiama {len(symbols)} akcijų istorija ({period})…")
    data = yf.download(symbols, period=period, interval="1d",
                       group_by="ticker", progress=False, auto_adjust=False, threads=True)

    buckets = defaultdict(lambda: {"win": 0, "loss": 0, "flat_win": 0, "flat_loss": 0})
    per_symbol = defaultdict(lambda: {"n": 0, "win": 0})
    total = 0

    for tag, sym, _ in dr.WATCHLIST:
        try:
            hist = dr.flatten(data, sym).dropna(subset=["Close"])
        except Exception:
            continue
        if len(hist) < 60:
            continue

        for i in range(55, len(hist) - 1):
            res = daily_score(hist, i, args.target)
            if not res:
                continue
            d, s = res
            if s["rr"] <= 0 or s["shares"] == 0:
                continue

            outcome = evaluate_next_day(hist, i, d["price"], s["stop"], s["tp"])
            if outcome is None:
                continue

            buckets[s["grade"]][outcome] += 1
            per_symbol[tag]["n"] += 1
            if outcome in ("win", "flat_win"):
                per_symbol[tag]["win"] += 1
            total += 1

    if not total:
        sys.exit("Nepakako duomenų. Patikrink interneto ryšį.")

    print(f"\nIštirta sandorio galimybių: {total}\n")
    print(f"{'BALAS':<7} {'ATVEJŲ':>7} {'PASIEKĖ TIKSLĄ':>15} {'STOP':>8} {'BAIGĖ +':>9} {'BAIGĖ −':>9}")
    print("-" * 60)

    for g in ["A", "B", "C", "D"]:
        b = buckets[g]
        n = sum(b.values())
        if not n:
            continue
        print(f"{g:<7} {n:>7} {b['win']/n*100:>14.1f}% {b['loss']/n*100:>7.1f}% "
              f"{b['flat_win']/n*100:>8.1f}% {b['flat_loss']/n*100:>8.1f}%")

    print("\nAr aukštesnis balas = geresnis rezultatas?")
    rates = {}
    for g in ["A", "B", "C", "D"]:
        b = buckets[g]
        n = sum(b.values())
        if n >= 20:
            rates[g] = b["win"] / n * 100

    if len(rates) >= 2:
        order = [rates[g] for g in ["A", "B", "C", "D"] if g in rates]
        spread = order[0] - order[-1]
        monotonic = all(order[i] >= order[i + 1] - 2 for i in range(len(order) - 1))
        for g, r in rates.items():
            print(f"  {g}: {r:.1f}% pasiekė +{args.target}%")
        print()
        if monotonic and spread > 5:
            print(f"  → Reitingas veikia teisinga kryptimi: skirtumas tarp geriausio "
                  f"ir blogiausio {spread:.1f} proc. punkto.")
        elif spread > 5:
            print(f"  → Skirtumas yra ({spread:.1f} p. p.), bet ne nuoseklus per visas pakopas.")
        else:
            print(f"  → Reitingas praktiškai neskiria gerų sandorių nuo blogų "
                  f"(skirtumas tik {spread:.1f} p. p.). Svorius verta perskaičiuoti.")
    else:
        print("  Per mažai atvejų išvadai.")

    print("\nPagal akciją (bent 15 atvejų):")
    for tag, v in sorted(per_symbol.items(), key=lambda x: -x[1]["win"] / max(x[1]["n"], 1)):
        if v["n"] >= 15:
            print(f"  {tag:<7} {v['win']/v['n']*100:>5.1f}%  ({v['n']} atvejų)")

    print("\nSVARBU: naudoti dienos barai, be VWAP/RSI/RVOL dedamųjų; kai per dieną "
          f"\npasiekiamas ir tikslas, ir stop — skaičiuojama kaip pralaimėjimas; "
          f"\nmokesčiai ir spread'as neįskaičiuoti. Realūs rezultatai bus prastesni.")


if __name__ == "__main__":
    main()
