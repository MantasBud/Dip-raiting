#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Likvidumo "nuslavimo" backtestas — veidrodine SMC / Wyckoff "spring" versija.

KAS TAI
SMC ir LuxAlgo AMD/FVG saltiniai aprasyta BEARISH versija: kaina trumpam
issoka virs ankstesnes virsunes, ismusa stop'us ir apsisuka zemyn. Vartotojas
perka, todel tikrinam VEIDRODINE: kaina nukrenta ZEMIAU ankstesnio dugno,
ismusa pardaveju stop'us, ir ta pacia diena gryzta VIRS to dugno.

Wyckoff teorijoje tai vadinama "spring", SMC — "sell-side liquidity sweep".

SABLONO APIBREZIMAS (uzrasytas pries matavima, nederinamas)
  1. dienos minimumas ZEMIAU N dienu dugno (nuslavimas)
  2. dienos uzdarymas VIRS to paties N dienu dugno (atgavimas)
  3. uzdarymas virsutineje dienos diapazono dalyje (IBS > riba)
Viskas zinoma dienos pabaigoje — zvilgsnio i ateiti nera.
Ijejimas kitos dienos atidarymu.

SVARBU: SMC ir ICT koncepcijos neturi rimto akademinio pagrindo — tai
daugiausia mazmeniniu prekiautoju folkloras. Tikrinam, nes sablonas
formalizuojamas be zvilgsnio i ateiti ir musu dar netestuotas.
Ankstesnis "dvigubas dugnas" buvo KITAS dalykas: du minimumai per 1%,
be pramusimo ir atgavimo.

KONTROLE
  - pramusimas BE atgavimo (uzdare zemiau dugno) — turetu buti blogiau
  - atgavimas be pramusimo (tik zemas IBS) — musu esamas signalas
  - visi dipai — bazine linija

Paleidimas:
    python backtest_sweep.py --rinka eu
"""

import argparse
import sys

import numpy as np
import pandas as pd

import universas as U

MIN_IVYKIU = 100
MIN_DIENU = 40
LAIKYMAS = 10
EXIT_IBS = 0.80


def flatten(df, sym):
    if isinstance(df.columns, pd.MultiIndex):
        if sym in df.columns.get_level_values(0):
            return df[sym].dropna(how="all")
        if sym in df.columns.get_level_values(-1):
            return df.xs(sym, axis=1, level=-1).dropna(how="all")
    return df.dropna(how="all")


def week_bootstrap(vals, dienos, n=2000, seed=47):
    if len(vals) < 15 or float(np.std(vals)) < 1e-12:
        return None
    rng = np.random.default_rng(seed)
    sav = pd.Series([pd.Timestamp(d).to_period("W") for d in dienos])
    gr = [np.asarray(vals)[(sav == w).to_numpy()] for w in sav.unique()]
    gr = [g for g in gr if len(g)]
    if len(gr) < 10:
        return None
    b = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(gr), len(gr))
        b[i] = np.concatenate([gr[j] for j in pick]).mean()
    lo, hi = np.percentile(b, [2.5, 97.5])
    return dict(mean=float(np.mean(vals)), lo=float(lo), hi=float(hi))


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


def rodikliai(d):
    c, h, l, v, o = d["Close"], d["High"], d["Low"], d["Volume"], d["Open"]
    pc = c.shift(1)
    t = pd.DataFrame(index=d.index)
    t["C"], t["L"], t["O1"] = c, l, o.shift(-1)
    t["ret"] = (c / pc - 1) * 100
    rng_ = (h - l).replace(0, np.nan)
    t["ibs"] = (c - l) / rng_
    t["apyv"] = (c * v).rolling(20).median()
    t["apyv_sant"] = v / v.rolling(21).median().shift(1)
    t["sma200"] = c.rolling(200).mean()
    t["sma200_kyla"] = t["sma200"] > t["sma200"].shift(20)
    # Ankstesnis dugnas — BE siandienos (shift 1), kitaip butu zvilgsnis i ateiti
    for n in (10, 20, 50):
        t[f"dugnas{n}"] = l.rolling(n).min().shift(1)
        t[f"sweep{n}"] = (l < t[f"dugnas{n}"]) & (c > t[f"dugnas{n}"])
        t[f"lusis{n}"] = (l < t[f"dugnas{n}"]) & (c <= t[f"dugnas{n}"])
        # kiek giliai nuslave (procentais po dugnu)
        t[f"gylis{n}"] = (t[f"dugnas{n}"] - l) / t[f"dugnas{n}"] * 100
    for k in range(1, LAIKYMAS + 1):
        t[f"C{k}"] = c.shift(-k)
        t[f"ibs{k}"] = ((c.shift(-k) - l.shift(-k))
                        / (h.shift(-k) - l.shift(-k)).replace(0, np.nan))
    t["r1"] = (c.shift(-1) / o.shift(-1) - 1) * 100
    t["r3"] = (c.shift(-3) / o.shift(-1) - 1) * 100
    t["naktis"] = (o.shift(-1) / c - 1) * 100
    return t


def isejimas(g):
    ijej = g["O1"]
    n = len(g)
    rez = np.full(n, np.nan)
    baigta = np.zeros(n, dtype=bool)
    for k in range(1, LAIKYMAS + 1):
        sal = (~baigta) & (g[f"ibs{k}"] >= EXIT_IBS).fillna(False).to_numpy()
        rez = np.where(sal, (g[f"C{k}"] / ijej - 1) * 100, rez)
        baigta = baigta | sal
    return np.where(baigta, rez, (g[f"C{LAIKYMAS}"] / ijej - 1) * 100)


def main():
    ap = argparse.ArgumentParser(description="Likvidumo nuslavimo backtestas")
    ap.add_argument("--metai", type=int, default=10)
    ap.add_argument("--rinka", default="eu", choices=["eu", "us"])
    ap.add_argument("--pozicija", type=float, default=18000.0)
    ap.add_argument("--sanaudos", type=float, default=0.028)
    a = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Paleisk: pip install yfinance pandas numpy")

    uni = U.UNIVERSAS_US if a.rinka == "us" else U.UNIVERSAS
    syms = U.visi_tikeriai(uni)
    print(f"{'JAV' if a.rinka == 'us' else 'EUROPOS'} universas: {len(syms)}\n")
    raw = yf.download(syms, period=f"{a.metai}y", interval="1d", group_by="ticker",
                      progress=False, auto_adjust=False, threads=True)

    dalys, be_d = [], 0
    for s in syms:
        try:
            d = flatten(raw, s).dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            if len(d) < 400:
                be_d += 1
                continue
            t = rodikliai(d)
            t["tag"], t["data"] = s, d.index
            dalys.append(t)
        except Exception:
            be_d += 1
    if not dalys:
        sys.exit("Nepavyko surinkti duomenu.")
    df = pd.concat(dalys, ignore_index=True)
    df = df[df["apyv"] >= 5e6].dropna(subset=["O1", f"C{LAIKYMAS}"]).copy()
    df["rez"] = isejimas(df)
    for c in ("rez", "r1", "r3", "naktis"):
        df[c + "_dm"] = df[c] - df.groupby("data")[c].transform("mean")
    print(f"  akciju: {df['tag'].nunique()}, eiluciu: {len(df):,}, be duomenu: {be_d}")

    riba = pd.to_datetime(df["data"]).quantile(0.5)
    p1 = pd.to_datetime(df["data"]) <= riba
    p2 = ~p1
    print(f"  padalijimas: 1-a puse iki {riba:%Y-%m-%d}\n")

    # ---------- A. PAGRINDINIS PALYGINIMAS ----------
    print("=" * 102)
    print("A. NUSLAVIMAS PRIES KONTROLES (20 dienu dugnas)")
    print("Rezultatas su musu isejimo taisykle: IBS > 0.8 arba iki 10 sesiju")
    print("=" * 102)
    print(f"{'VARIANTAS':<40} {'N':>7} {'DALIS':>7} {'DEMEAN.':>10} "
          f"{'95% INTERVALAS':>22} {'EUR':>9}")
    print("-" * 102)
    dip = (df["ret"] <= -0.8) & (df["ret"] >= -8.0)
    variantai = [
        ("Nuslavimas + atgavimas (SWEEP)", df["sweep20"]),
        ("  + uzdare virsuje (IBS > 0.5)", df["sweep20"] & (df["ibs"] > 0.5)),
        ("  + uzdare virsuje (IBS > 0.7)", df["sweep20"] & (df["ibs"] > 0.7)),
        ("  + didele apyvarta (> 1.5x)", df["sweep20"] & (df["apyv_sant"] > 1.5)),
        ("  + kylantis SMA200", df["sweep20"] & df["sma200_kyla"]),
        ("KONTROLE: pramusimas BE atgavimo", df["lusis20"]),
        ("KONTROLE: tik zemas IBS (musu signalas)",
         dip & (df["ibs"] <= 0.25) & ~df["sweep20"]),
        ("KONTROLE: visi dipai", dip),
    ]
    rez_a = []
    for lab, kauke in variantai:
        sub = df[kauke]
        r, n = ivertink(sub, "rez_dm")
        dalis = len(sub) / len(df) * 100
        if r is None:
            print(f"{lab:<40} {n:>7} {dalis:>6.2f}%   per maza imtis")
            continue
        eur = (sub["rez"].mean() - a.sanaudos) / 100 * a.pozicija
        ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
        print(f"{lab:<40} {r['n']:>7} {dalis:>6.2f}% {r['mean']:>+9.3f}% {ci:>22} "
              f"{eur:>+8.2f}€")
        rez_a.append((lab, kauke, r))

    # ---------- B. DUGNO LANGAS ----------
    print("\n" + "=" * 102)
    print("B. KOKIO DUGNO NUSLAVIMAS SVARBIAUSIAS (10, 20, 50 dienu)")
    print("=" * 102)
    print(f"{'DUGNAS':<24} {'N':>7} {'DEMEAN.':>10} {'95% INTERVALAS':>22} "
          f"{'NAKTIS':>9} {'EUR':>9}")
    print("-" * 102)
    for n_ in (10, 20, 50):
        sub = df[df[f"sweep{n_}"]]
        r, n = ivertink(sub, "rez_dm")
        if r is None:
            print(f"{str(n_) + ' dienu':<24} {n:>7}   per maza imtis")
            continue
        naktis = sub["naktis_dm"].mean()
        eur = (sub["rez"].mean() - a.sanaudos) / 100 * a.pozicija
        ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
        print(f"{str(n_) + ' dienu':<24} {r['n']:>7} {r['mean']:>+9.3f}% {ci:>22} "
              f"{naktis:>+8.3f}% {eur:>+8.2f}€")

    # ---------- C. NUSLAVIMO GYLIS ----------
    print("\n" + "=" * 102)
    print("C. AR SVARBU, KAIP GILIAI NUSLAVE (procentais po 20 d. dugnu)")
    print("=" * 102)
    sw = df[df["sweep20"]]
    for lo, hi, lab in [(0, 0.5, "vos palietE (<0.5%)"), (0.5, 1.5, "0.5-1.5%"),
                        (1.5, 3.0, "1.5-3%"), (3.0, 99, "giliai (>3%)")]:
        sub = sw[(sw["gylis20"] > lo) & (sw["gylis20"] <= hi)]
        r, n = ivertink(sub, "rez_dm")
        if r is None:
            print(f"  {lab:<22} {n:>7}   per maza imtis")
            continue
        print(f"  {lab:<22} {r['n']:>7} {r['mean']:>+8.3f}% "
              f"[{r['lo']:+.3f}..{r['hi']:+.3f}]")

    # ---------- D. LANGAI ----------
    print("\n" + "=" * 102)
    print("D. KUR SUSIDARO GRAZA (nuslavimas, 20 d.)")
    print("=" * 102)
    for c, lab in [("naktis_dm", "pirma naktis"), ("r1_dm", "kita diena"),
                   ("r3_dm", "3 dienos"), ("rez_dm", "iki isejimo salygos")]:
        r, n = ivertink(sw, c)
        if r:
            print(f"  {lab:<22} {r['mean']:>+8.3f}% [{r['lo']:+.3f}..{r['hi']:+.3f}] "
                  f"(n={r['n']})")

    # ---------- PATVIRTINIMAS ----------
    print("\n" + "=" * 102)
    print("PATVIRTINIMAS 2-OJE PUSEJE (nematyta)")
    print("=" * 102)
    print(f"{'VARIANTAS':<40} {'N':>7} {'DEMEAN.':>10} {'95% INTERVALAS':>22} "
          f"{'VERDIKTAS':>14}")
    print("-" * 102)
    for lab, kauke, _ in rez_a:
        sub = df[kauke & p2]
        r, n = ivertink(sub, "rez_dm")
        if r is None:
            print(f"{lab:<40} {n:>7}   per maza imtis")
            continue
        v = ("PATVIRTINTA" if r["lo"] > 0 else
             ("PRIESINGA" if r["hi"] < 0 else "nepatvirtinta"))
        ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
        print(f"{lab:<40} {r['n']:>7} {r['mean']:>+9.3f}% {ci:>22} {v:>14}")

    print("\n" + "=" * 102)
    print("KAIP SKAITYTI")
    print("=" * 102)
    print("  Sablonas veikia, jei SWEEP geresnis uz 'pramusima be atgavimo' IR uz")
    print("  'tik zema IBS'. Jei nuslavimas nesiskiria nuo paprasto dipo — tai tik")
    print("  tas pats signalas kitu vardu, ir nieko nauja neprideda.")
    print("\nAPRIBOJIMAI: dienos barai — tikras SMC nuslavimas vyksta per valandas, "
          "\nnematom, ar kaina pirma nusleido ir tik paskui atgavo; islikimo salismas.")


if __name__ == "__main__":
    main()
