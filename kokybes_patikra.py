#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kokybes patikra — kurios universo bendroves neatitinka ismatuoto filtro.

KODEL SIS FILTRAS
Backtestas (2026-09-15, 8 metai, abi rinkos) parode, kad NUOSTOLINGOSE
bendrovese dipas baigiasi blogiau butent blogiausiais atvejais:

                        Europa            JAV
  pelninga, 5% blog.    -7.17%           -6.14%
  nuostolinga, 5% blog. -10.48%          -7.28%
  pelninga, < -10%       2.80%            1.62%
  nuostolinga, < -10%    5.44%            2.54%

Giliai nuostolingu sandoriu nuostolingose bendrovese beveik DVIGUBAI
daugiau, ir tai pasikartojo abiejose rinkose. Stipriausias vienas rodiklis —
PELNO MARZA: Europoje blogiausias sandoris geriausiu kvintilyje -44.2%,
prasciausiu -54.5%; JAV -36.3% pries -55.8%.

Skola ir likvidumo rodiklis skirtumo NEDAVE, nors intuicija sake kitaip.
Todel filtras remiasi tik tuo, kas ismatuota.

SVARBU: tai ne pranasumo siekis, o UODEGOS ribojimas. Vidurkiais remtis
negalima — fundamentai siandieniniai, tai zvilgsnis i ateiti. Bet uodegos
skirtumas islieka ir su tuo salismu, todel juo remtis galima.

Paleidimas:
    python kokybes_patikra.py
"""

import sys

import numpy as np
import pandas as pd

import universas as U

# Ribos: tik tos, kurias backtestas patvirtino
MIN_EPS = 0.0            # pelninga bendrove
MIN_MARZA = 0.0          # teigiama pelno marza


def main():
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Paleisk: pip install yfinance pandas numpy")

    syms = U.visi_tikeriai()
    sekt = U.sektoriai()
    print(f"Universas: {len(syms)} instrumentu. Siunciami fundamentai…\n")

    eil, be_d = [], []
    for i, s in enumerate(syms):
        try:
            inf = yf.Ticker(s).info or {}

            def g(k):
                v = inf.get(k)
                try:
                    return float(v)
                except Exception:
                    return np.nan

            eps, marza = g("trailingEps"), g("profitMargins")
            # ETF ir ETC pelno rodikliu neturi — jie NEVERTINAMI
            tipas = (inf.get("quoteType") or "").upper()
            etf = tipas in ("ETF", "MUTUALFUND") or (eps != eps and marza != marza)
            eil.append(dict(
                sym=s, sekt=sekt.get(s, "kita"),
                pav=(inf.get("shortName") or "")[:28],
                eps=eps, marza=marza,
                bruto_marza=g("grossMargins"), roe=g("returnOnEquity"),
                skola=g("debtToEquity"), etf=etf))
        except Exception:
            be_d.append(s)
        if (i + 1) % 60 == 0:
            print(f"  …{i + 1}/{len(syms)}")

    if not eil:
        sys.exit("Nepavyko gauti fundamentu.")
    df = pd.DataFrame(eil)
    print(f"\nFundamentu turi: {len(df)}; nepavyko: {len(be_d)}")
    if be_d:
        print(f"  nepavyko: {', '.join(be_d[:10])}"
              + (" …" if len(be_d) > 10 else ""))

    akcijos = df[~df["etf"]]
    etf = df[df["etf"]]
    print(f"  akciju: {len(akcijos)}, ETF ir ETC (nevertinami): {len(etf)}")

    # --- Kas neatitinka ---
    nuostol = akcijos[(akcijos["eps"] <= MIN_EPS) & akcijos["eps"].notna()]
    neig_marza = akcijos[(akcijos["marza"] <= MIN_MARZA) & akcijos["marza"].notna()]
    blogos = akcijos[akcijos["sym"].isin(
        set(nuostol["sym"]) | set(neig_marza["sym"]))]
    nezinoma = akcijos[akcijos["eps"].isna() & akcijos["marza"].isna()]

    print("\n" + "=" * 92)
    print("NEATITINKA FILTRO (nuostolinga arba neigiama marza)")
    print("=" * 92)
    if blogos.empty:
        print("  Nei viena — visas universas pelningas.")
    else:
        print(f"{'TIKERIS':<12} {'PAVADINIMAS':<30} {'EPS':>8} {'MARZA':>8} "
              f"{'ROE':>8}  SEKTORIUS")
        print("-" * 92)
        for _, r in blogos.sort_values("marza").iterrows():
            m = f"{r['marza'] * 100:.1f}%" if r["marza"] == r["marza"] else "—"
            e = f"{r['eps']:.2f}" if r["eps"] == r["eps"] else "—"
            ro = f"{r['roe'] * 100:.1f}%" if r["roe"] == r["roe"] else "—"
            print(f"{r['sym']:<12} {r['pav']:<30} {e:>8} {m:>8} {ro:>8}  {r['sekt']}")
        print(f"\n  Is viso: {len(blogos)} is {len(akcijos)} akciju "
              f"({len(blogos) / len(akcijos) * 100:.1f}%)")

    if not nezinoma.empty:
        print(f"\n  Be duomenu (nevertinam, paliekam): "
              f"{', '.join(nezinoma['sym'].tolist()[:12])}")

    # --- Silpniausios, bet dar pelningos ---
    silpnos = akcijos[(akcijos["marza"] > 0) & (akcijos["marza"] < 0.03)]
    if not silpnos.empty:
        print("\n" + "=" * 92)
        print("SILPNOS, BET PELNINGOS (marza 0-3%) — sprendimas tavo")
        print("=" * 92)
        for _, r in silpnos.sort_values("marza").iterrows():
            print(f"  {r['sym']:<12} {r['pav']:<30} marza "
                  f"{r['marza'] * 100:>5.1f}%  {r['sekt']}")

    # --- Kodas salinimui ---
    if not blogos.empty:
        print("\n" + "=" * 92)
        print("KA DARYTI")
        print("=" * 92)
        print("  Siuos tikerius verta pasalinti is universas.py:")
        print("  " + " ".join(f'"{s}"' for s in sorted(blogos["sym"])))
        print("\n  Priminimas: tai UODEGOS ribojimas, ne pranasumo siekis.")
        print("  Backteste nuostolingose bendrovese giliai nuostolingu sandoriu")
        print("  buvo beveik dvigubai daugiau — Europoje 5.44% pries 2.80%.")

    print(f"\nSEKTORIU SUVESTINE:")
    for s, g_ in akcijos.groupby("sekt"):
        bl = len(g_[g_["sym"].isin(blogos["sym"])])
        if bl:
            print(f"  {s:<32} {bl} is {len(g_)} neatitinka")


if __name__ == "__main__":
    main()
