#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dienos laiko backtestas — KADA matuoti ir ar tarpas testiasi.

TRYS KLAUSIMAI, VISI UZRASYTI PRIES MATAVIMA

A. KADA MATUOTI IBS
   Modulis visa diena rodo ta pati sarasa. Bet IBS 11:00 ir 17:00 yra
   skirtingi dydziai: rytа dienos diapazonas dar siauras. Jei pranasumas
   naktinis, tai svarbu, kuriuo metu matuojam. Tikrinam IBS kas valandа ir
   ziurim, kuris geriausiai prognozuoja nakties grazа.

B. AR NAKTIES TARPAS TESIASI PER DIENA ("gap and go")
   Vartotojo pastebejimas: buna poziciju, kurios per nakti kilsteli ir tada
   nuo ryto iki uzdarymo stabiliai juda i virsu. Tikrinam tiesiogiai: po dipo
   signalo, jei naktis buvo teigiama, ar diena testia kilimа.
   ANKSTESNIS KONTEKSTAS: nakties tarpas ir atidarymo diapazono pramusimas
   atskirai netiko. Bet sios salygos (dipas -> teigiama naktis -> diena)
   netikrinom ne karto.

C. AR PIRMA VALANDA PROGNOZUOJA LIKUSIA DIENA
   Jei taip, uztektu palaukti iki 10:00 ir spresti. Jei ne — dienos kryptis
   nenuspejama, ir tai patvirtina, kad verta prekiauti tik nakties langu.

Paleidimas:
    python backtest_laikas.py --rinka eu
"""

import argparse
import sys

import numpy as np
import pandas as pd

import universas as U

MIN_IVYKIU = 200
MIN_DIENU = 40
BARU_SESIJOJE = 9          # valandiniai barai: 9:00..17:00


def flatten(df, sym):
    if isinstance(df.columns, pd.MultiIndex):
        if sym in df.columns.get_level_values(0):
            return df[sym].dropna(how="all")
        if sym in df.columns.get_level_values(-1):
            return df.xs(sym, axis=1, level=-1).dropna(how="all")
    return df.dropna(how="all")


def week_bootstrap(vals, dienos, n=2000, seed=13):
    if len(vals) < 15 or float(np.std(vals)) < 1e-12:
        return None
    rng = np.random.default_rng(seed)
    sav = pd.Series([pd.Timestamp(d).to_period("W") for d in dienos])
    gr = [np.asarray(vals)[(sav == w).to_numpy()] for w in sav.unique()]
    gr = [g for g in gr if len(g)]
    if len(gr) < 10:
        return None
    boot = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(gr), len(gr))
        boot[i] = np.concatenate([gr[j] for j in pick]).mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(mean=float(np.mean(vals)), lo=float(lo), hi=float(hi),
                dienos=len(vals))


def ivertink(sub, col):
    s = sub.dropna(subset=[col])
    if len(s) < MIN_IVYKIU:
        return None, len(s)
    pd_ = s.groupby("data")[col].mean()
    if len(pd_) < MIN_DIENU:
        return None, len(s)
    r = week_bootstrap(pd_.to_numpy(), pd_.index)
    if r:
        r["n"] = len(s)
    return r, len(s)


def main():
    ap = argparse.ArgumentParser(description="Dienos laiko backtestas")
    ap.add_argument("--rinka", default="eu", choices=["eu", "us"])
    ap.add_argument("--pozicija", type=float, default=18000.0)
    ap.add_argument("--sanaudos-eur", type=float, default=5.0)
    a = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Paleisk: pip install yfinance pandas numpy")

    uni = U.UNIVERSAS_US if a.rinka == "us" else U.UNIVERSAS
    syms = U.visi_tikeriai(uni)
    print(f"{'JAV' if a.rinka == 'us' else 'EUROPOS'} universas: {len(syms)}")
    print("Siunciama valandine istorija (720 d.)…")
    raw = yf.download(syms, period="720d", interval="60m", group_by="ticker",
                      progress=False, auto_adjust=False, threads=True)

    eil, be_d = [], 0
    for s in syms:
        try:
            d = flatten(raw, s).dropna(subset=["Open", "High", "Low", "Close"])
            if len(d) < 500:
                be_d += 1
                continue
            d = d.copy()
            d["_d"] = [i.date() for i in d.index]
            sesijos = [(dd, g) for dd, g in d.groupby("_d") if len(g) >= 6]
            for i in range(1, len(sesijos) - 1):
                dd, g = sesijos[i]
                pr_dd, pr_g = sesijos[i - 1]
                kt_dd, kt_g = sesijos[i + 1]
                try:
                    uzdar = float(g["Close"].iloc[-1])
                    pr_uzdar = float(pr_g["Close"].iloc[-1])
                    atidar = float(g["Open"].iloc[0])
                    kt_atidar = float(kt_g["Open"].iloc[0])
                    kt_uzdar = float(kt_g["Close"].iloc[-1])
                    if min(uzdar, pr_uzdar, atidar, kt_atidar) <= 0:
                        continue

                    rec = dict(tag=s, data=pd.Timestamp(dd))
                    rec["dienos_pok"] = (uzdar / pr_uzdar - 1) * 100
                    # IBS kas valandа: kaupiamas dienos diapazonas iki to baro
                    for k in range(2, min(len(g), BARU_SESIJOJE) + 1):
                        dalis = g.iloc[:k]
                        hi_, lo_ = float(dalis["High"].max()), float(dalis["Low"].min())
                        c_ = float(dalis["Close"].iloc[-1])
                        rec[f"ibs_{k}"] = ((c_ - lo_) / (hi_ - lo_)
                                           if hi_ > lo_ else np.nan)
                    rec["ibs_gal"] = rec.get(f"ibs_{min(len(g), BARU_SESIJOJE)}")
                    # Langai
                    rec["naktis"] = (kt_atidar / uzdar - 1) * 100
                    rec["kita_diena"] = (kt_uzdar / kt_atidar - 1) * 100
                    rec["visa_para"] = (kt_uzdar / uzdar - 1) * 100
                    # Pirma valanda kitos dienos ir likusi diena
                    if len(kt_g) >= 3:
                        po_1val = float(kt_g["Close"].iloc[0])
                        rec["kt_1val"] = (po_1val / kt_atidar - 1) * 100
                        rec["kt_likusi"] = (kt_uzdar / po_1val - 1) * 100
                    eil.append(rec)
                except Exception:
                    continue
        except Exception:
            be_d += 1

    if not eil:
        sys.exit("Nepavyko surinkti duomenu.")
    df = pd.DataFrame(eil)
    df = df[(df["dienos_pok"] <= -0.8) & (df["dienos_pok"] >= -6.0)].copy()
    for c in ("naktis", "kita_diena", "visa_para", "kt_1val", "kt_likusi"):
        if c in df:
            df[c + "_dm"] = df[c] - df.groupby("data")[c].transform("mean")
    print(f"  akciju: {df['tag'].nunique()}, ijejimo tasku: {len(df):,}, "
          f"be duomenu: {be_d}\n")

    riba = df["data"].quantile(0.5)
    p1 = df["data"] <= riba
    IBS_RIBA = float(df.loc[p1, "ibs_gal"].quantile(0.20))
    kand = df[df["ibs_gal"] <= IBS_RIBA]

    # ---------- A. KADA MATUOTI IBS ----------
    print("=" * 100)
    print("A. KADA MATUOTI IBS — kuris laikas geriausiai prognozuoja nakti")
    print("=" * 100)
    print(f"{'MATUOTA':<16} {'N':>7} {'NAKTIES GRAZA':>15} {'95% INTERVALAS':>22} "
          f"{'EUR':>9}")
    print("-" * 100)
    for k in range(2, BARU_SESIJOJE + 1):
        c = f"ibs_{k}"
        if c not in df or df[c].notna().sum() < 1000:
            continue
        q = float(df.loc[p1, c].quantile(0.20))
        sub = df[df[c] <= q]
        r, n = ivertink(sub, "naktis_dm")
        if r is None:
            continue
        val = 8 + k          # 9:00 + (k-1) valandu
        eur = sub["naktis"].mean() / 100 * a.pozicija - a.sanaudos_eur
        ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
        print(f"{f'~{val}:00':<16} {r['n']:>7} {r['mean']:>+14.3f}% {ci:>22} "
              f"{eur:>+8.2f}€")
    print("\n  Jei velyvesnis matavimas duoda didesni skaiciu — verta pirkti")
    print("  pries pat uzdarymа, o ne bet kada per diena.")

    # ---------- B. AR TARPAS TESIASI PER DIENA ----------
    print("\n" + "=" * 100)
    print("B. AR NAKTIES KILIMAS TESIASI PER DIENA")
    print("Po dipo signalo: jei naktis buvo teigiama, ka daro diena?")
    print("=" * 100)
    print(f"{'NAKTIS BUVO':<22} {'N':>7} {'DALIS':>7} {'DIENOS GRAZA':>14} "
          f"{'95% INTERVALAS':>22}")
    print("-" * 100)
    for lo, hi, lab in [(-99, -1.0, "krito > 1%"), (-1.0, -0.2, "krito 0.2-1%"),
                        (-0.2, 0.2, "beveik nulis"), (0.2, 1.0, "kilo 0.2-1%"),
                        (1.0, 2.0, "kilo 1-2%"), (2.0, 99, "kilo > 2%")]:
        sub = kand[(kand["naktis"] >= lo) & (kand["naktis"] < hi)]
        r, n = ivertink(sub, "kita_diena_dm")
        dalis = len(sub) / max(1, len(kand)) * 100
        if r is None:
            print(f"{lab:<22} {n:>7} {dalis:>6.0f}%   per maza imtis")
            continue
        ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
        print(f"{lab:<22} {r['n']:>7} {dalis:>6.0f}% {r['mean']:>+13.3f}% {ci:>22}")
    print("\n  Jei 'kilo > 2%' eilute teigiama ir intervalas nekerta nulio —")
    print("  tarpas testiasi, ir tokias pozicijas verta laikyti per diena.")

    # ---------- C. AR PIRMA VALANDA PROGNOZUOJA LIKUSIA DIENA ----------
    print("\n" + "=" * 100)
    print("C. AR PIRMA VALANDA PROGNOZUOJA LIKUSIA DIENA")
    print("Jei taip — uztektu palaukti iki 10:00 ir spresti.")
    print("=" * 100)
    if "kt_1val" in kand and kand["kt_1val"].notna().sum() > 1000:
        print(f"{'PIRMA VALANDA':<22} {'N':>7} {'LIKUSI DIENA':>14} "
              f"{'95% INTERVALAS':>22}")
        print("-" * 100)
        for lo, hi, lab in [(-99, -0.5, "krito > 0.5%"), (-0.5, 0, "krito 0-0.5%"),
                            (0, 0.5, "kilo 0-0.5%"), (0.5, 99, "kilo > 0.5%")]:
            sub = kand[(kand["kt_1val"] >= lo) & (kand["kt_1val"] < hi)]
            r, n = ivertink(sub, "kt_likusi_dm")
            if r is None:
                print(f"{lab:<22} {n:>7}   per maza imtis")
                continue
            ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
            print(f"{lab:<22} {r['n']:>7} {r['mean']:>+13.3f}% {ci:>22}")
        kor = kand[["kt_1val", "kt_likusi"]].corr().iloc[0, 1]
        print(f"\n  Koreliacija tarp pirmos valandos ir likusios dienos: {kor:+.3f}")
        print("  Virs +0.15 reikstu, kad kryptis testiasi ir ja verta gaudyti.")
    else:
        print("  duomenu nepakako")

    print("\nAPRIBOJIMAI: valandiniai barai tik 720 d.; atidarymo kainos "
          "\nauckciono metu kartais netikslios; islikimo salismas; "
          "\nsanaudos itrauktos tik EUR stulpelyje.")


if __name__ == "__main__":
    main()
