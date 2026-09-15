#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universo kainu pasiskirstymas — kiek akciju telpa i pozicija.

KAM TAI
Su 18 000 EUR pozicija akcijos kaina lemia, kiek vienetu gali nupirkti:
  kaina   50 EUR -> 360 akciju
  kaina  500 EUR ->  36 akcijos
  kaina 1400 EUR ->  12 akciju
Kuo maziau vienetu, tuo grubiau gali derinti pozicijos dydi ir tuo didesne
dalis pinigu lieka nepanaudota. Prie 1400 EUR kainos vienos akcijos kaina
yra 7.8% tavo pozicijos — negali nupirkti "puses pozicijos" tiksliai.

Paleidimas:
    python kainos.py
"""

import sys

import pandas as pd

import universas as U

POZICIJA = 18000.0


def flatten(df, sym):
    if isinstance(df.columns, pd.MultiIndex):
        if sym in df.columns.get_level_values(0):
            return df[sym].dropna(how="all")
        if sym in df.columns.get_level_values(-1):
            return df.xs(sym, axis=1, level=-1).dropna(how="all")
    return df.dropna(how="all")


def main():
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Paleisk: pip install yfinance pandas")

    syms = U.visi_tikeriai()
    sekt = U.sektoriai()
    print(f"Universas: {len(syms)} instrumentu. Siunciama…\n")
    raw = yf.download(syms, period="5d", interval="1d", group_by="ticker",
                      progress=False, auto_adjust=False, threads=True)

    eil, be_d = [], []
    for s in syms:
        try:
            d = flatten(raw, s).dropna(subset=["Close"])
            if d.empty:
                be_d.append(s)
                continue
            px = float(d["Close"].iloc[-1])
            eil.append(dict(sym=s, sekt=sekt.get(s, "kita"), kaina=px,
                            vnt=int(POZICIJA / px) if px > 0 else 0))
        except Exception:
            be_d.append(s)

    if not eil:
        sys.exit("Nepavyko gauti kainu.")
    df = pd.DataFrame(eil).sort_values("kaina")
    print(f"Kainu turi: {len(df)}; be duomenu: {len(be_d)}")
    if be_d:
        print(f"  be duomenu: {', '.join(be_d[:12])}"
              + (" …" if len(be_d) > 12 else ""))

    print("\n" + "=" * 78)
    print(f"KAINU PASISKIRSTYMAS (pozicija {POZICIJA:,.0f} EUR)")
    print("=" * 78)
    print(f"{'KAINU RUOZAS':<20} {'KIEK':>6} {'DALIS':>8} {'VNT. UZ POZICIJA':>20}")
    print("-" * 78)
    ruozai = [(0, 25, "iki 25 EUR"), (25, 50, "25-50"), (50, 100, "50-100"),
              (100, 200, "100-200"), (200, 400, "200-400"),
              (400, 800, "400-800"), (800, 10 ** 9, "virs 800")]
    for lo, hi, lab in ruozai:
        sub = df[(df["kaina"] >= lo) & (df["kaina"] < hi)]
        if sub.empty:
            continue
        vnt = f"{int(POZICIJA / sub['kaina'].max()):,}-{int(POZICIJA / sub['kaina'].min()):,}"
        print(f"{lab:<20} {len(sub):>6} {len(sub) / len(df) * 100:>7.1f}% {vnt:>20}")

    zemiau = df[df["kaina"] < 100]
    print("-" * 78)
    print(f"{'ZEMIAU 100 EUR':<20} {len(zemiau):>6} {len(zemiau) / len(df) * 100:>7.1f}%")
    print(f"{'100 EUR IR DAUGIAU':<20} {len(df) - len(zemiau):>6} "
          f"{(len(df) - len(zemiau)) / len(df) * 100:>7.1f}%")

    print("\n" + "=" * 78)
    print("BRANGIAUSI (mazai vienetu — grubus pozicijos derinimas)")
    print("=" * 78)
    for _, r in df.tail(12).iloc[::-1].iterrows():
        dalis = r["kaina"] / POZICIJA * 100
        zyma = "  <- viena akcija > 5% pozicijos" if dalis > 5 else ""
        print(f"  {r['sym']:<10} {r['kaina']:>9.2f} EUR  {r['vnt']:>5} vnt.  "
              f"({dalis:.1f}% pozicijos){zyma}")

    print("\n" + "=" * 78)
    print("ZEMIAU 100 EUR PAGAL SEKTORIU")
    print("=" * 78)
    for s, g in zemiau.groupby("sekt"):
        print(f"  {s:<32} {len(g):>3} is {len(df[df['sekt'] == s]):>3}")

    print(f"\nMediana: {df['kaina'].median():.2f} EUR "
          f"({int(POZICIJA / df['kaina'].median()):,} vnt. uz pozicija)")


if __name__ == "__main__":
    main()
