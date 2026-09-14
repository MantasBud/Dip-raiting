#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vertes nuolaidos backtestas — ar dipas pigioje akcijoje elgiasi kitaip.

KLAUSIMAS
Iki siol visi musu signalai buvo KAINOS isvestines. Cia pirma karta i
matavima ieina fundamentalus sluoksnis: ar akcija pigi pagal savo pacios
istorija ir pagal sektoriu.

TRYS NUOLAIDOS APIBREZIMAI (uzrasyti pries matavima)
1. P/E savo istorijos atzvilgiu — dabartinis trailingPE lyginamas su tos
   pacios akcijos mediana per turima laikotarpi. "Pigi pagal savo standarta."
2. P/E sektoriaus atzvilgiu — dabartinis P/E lyginamas su sektoriaus mediana
   ta diena. "Pigi pries bendraamzius."
3. Analitiku tikslines kainos atotrukis — targetMeanPrice / kaina - 1.
   DEMESIO: literatura nuosekliai rodo, kad analitiku tikslines kainos yra
   sistemingai optimistines ir trumpuoju laikotarpiu neprognozuoja. Itraukiam
   tik tam, kad tai patikrintume savo duomenyse, o ne tiketume.

KRITINIS APRIBOJIMAS, KURI REIKIA SKAITYTI PRIES REZULTATUS
yfinance grazina TIK SIANDIENINIUS fundamentalius rodiklius. Istoriniu P/E
pagal datas nera. Todel:
  - P/E laikomas PASTOVIU per visa tikrinimo laikotarpi;
  - tai reiskia, kad matavimas atsako ne "ar pigumas prognozavo", o
    "ar akcijos, kurios PIGIOS SIANDIEN, praeityje elgesi kitaip";
  - tai yra zvilgsnis i ateiti (look-ahead) ir rezultata daro PER GERA.
Jei net su siuo salismu efekto nera — jo tikrai nera. Jei yra, ji reikes
tikrinti is naujo su tikrais istoriniais duomenimis, o ju nemokamai nera.

Paleidimas:
    python backtest_verte.py --rinka eu
"""

import argparse
import sys

import numpy as np
import pandas as pd

import universas as U

MIN_IVYKIU = 150
MIN_DIENU = 40


def flatten(df, sym):
    if isinstance(df.columns, pd.MultiIndex):
        if sym in df.columns.get_level_values(0):
            return df[sym].dropna(how="all")
        if sym in df.columns.get_level_values(-1):
            return df.xs(sym, axis=1, level=-1).dropna(how="all")
    return df.dropna(how="all")


def week_bootstrap(vals, dienos, n=2000, seed=19):
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


def fundamentai(yf, syms, verbose=True):
    """Dabartiniai fundamentalus rodikliai. Po viena tikeri — letas zingsnis."""
    out = {}
    nepavyko = 0
    for i, s in enumerate(syms):
        try:
            inf = yf.Ticker(s).info or {}
            pe = inf.get("trailingPE")
            fpe = inf.get("forwardPE")
            tgt = inf.get("targetMeanPrice")
            px = inf.get("currentPrice") or inf.get("regularMarketPrice")
            pb = inf.get("priceToBook")
            out[s] = dict(
                pe=float(pe) if pe and 0 < pe < 200 else np.nan,
                fpe=float(fpe) if fpe and 0 < fpe < 200 else np.nan,
                pb=float(pb) if pb and 0 < pb < 50 else np.nan,
                tgt_gap=(float(tgt) / float(px) - 1) * 100
                if (tgt and px and px > 0) else np.nan,
            )
        except Exception:
            nepavyko += 1
        if verbose and (i + 1) % 50 == 0:
            print(f"    …{i + 1}/{len(syms)}")
    if verbose:
        turi = sum(1 for v in out.values() if v["pe"] == v["pe"])
        print(f"  fundamentu turi: {turi} is {len(syms)}; nepavyko: {nepavyko}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Vertes nuolaidos backtestas")
    ap.add_argument("--metai", type=int, default=5)
    ap.add_argument("--rinka", default="eu", choices=["eu", "us"])
    ap.add_argument("--pozicija", type=float, default=18000.0)
    ap.add_argument("--sanaudos-eur", type=float, default=5.0)
    a = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Paleisk: pip install yfinance pandas numpy")

    uni = U.UNIVERSAS_US if a.rinka == "us" else U.UNIVERSAS
    sekt = U.sektoriai(uni)
    syms = U.visi_tikeriai(uni)
    print(f"{'JAV' if a.rinka == 'us' else 'EUROPOS'} universas: {len(syms)}\n")

    print("Siunciami fundamentalus rodikliai (po viena tikeri, letai)…")
    fund = fundamentai(yf, syms)

    print("\nSiunciama kainu istorija…")
    raw = yf.download(syms, period=f"{a.metai}y", interval="1d", group_by="ticker",
                      progress=False, auto_adjust=False, threads=True)

    dalys, be_d = [], 0
    for s in syms:
        f = fund.get(s)
        if not f or f["pe"] != f["pe"]:
            be_d += 1
            continue
        try:
            d = flatten(raw, s).dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            if len(d) < 300:
                be_d += 1
                continue
            c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
            pc = c.shift(1)
            t = pd.DataFrame(index=d.index)
            t["ret"] = (c / pc - 1) * 100
            rng_ = (h - l).replace(0, np.nan)
            t["ibs"] = (c - l) / rng_
            t["apyv"] = (c * v).rolling(20).median()
            t["O1"] = d["Open"].shift(-1)
            for n in (1, 3, 5):
                t[f"C{n}"] = c.shift(-n)
            t["tag"], t["sekt"], t["data"] = s, sekt.get(s, "kita"), d.index
            t["pe"], t["fpe"], t["pb"] = f["pe"], f["fpe"], f["pb"]
            t["tgt_gap"] = f["tgt_gap"]
            # P/E savo istorijos atzvilgiu: dabartinis P/E pastovus, bet kaina
            # kinta, todel "numanomas" P/E = pe * (kaina / dabartine kaina)
            t["pe_santykinis"] = f["pe"] * (c / float(c.iloc[-1]))
            dalys.append(t)
        except Exception:
            be_d += 1

    if not dalys:
        sys.exit("Nepavyko surinkti duomenu.")
    df = pd.concat(dalys, ignore_index=True)
    df = df[(df["ret"] <= -0.8) & (df["ret"] >= -6.0) & (df["apyv"] >= 5e6)].copy()
    df = df.dropna(subset=["O1", "C3"])
    for n in (1, 3, 5):
        df[f"r{n}"] = (df[f"C{n}"] / df["O1"] - 1) * 100
        df[f"r{n}_dm"] = df[f"r{n}"] - df.groupby("data")[f"r{n}"].transform("mean")
    print(f"  akciju: {df['tag'].nunique()}, ijejimo tasku: {len(df):,}, "
          f"be fundamentu ar duomenu: {be_d}")

    riba = pd.to_datetime(df["data"]).quantile(0.5)
    p1 = pd.to_datetime(df["data"]) <= riba
    IBS_RIBA = float(df.loc[p1, "ibs"].quantile(0.20))
    kand = df[df["ibs"] <= IBS_RIBA].copy()
    print(f"  kandidatu (IBS <= {IBS_RIBA:.3f}): {len(kand):,}\n")

    # P/E sektoriaus atzvilgiu
    kand["pe_vs_sekt"] = kand["pe"] / kand.groupby(["data", "sekt"])["pe"].transform(
        "median")
    # P/E savo istorijos atzvilgiu
    kand["pe_vs_savo"] = kand["pe_santykinis"] / kand.groupby("tag")[
        "pe_santykinis"].transform("median")

    print("=" * 100)
    print("A. AR NUOLAIDA KEICIA DIPO REZULTATA")
    print("=" * 100)
    matai = [("pe_vs_savo", "P/E vs savo istorija"),
             ("pe_vs_sekt", "P/E vs sektorius"),
             ("pe", "P/E absoliutus"),
             ("pb", "Kaina / balansine verte"),
             ("tgt_gap", "Analitiku tikslo atotrukis")]
    for c, lab in matai:
        if c not in kand or kand[c].notna().sum() < 500:
            print(f"\n  {lab}: duomenu nepakako")
            continue
        q = kand.loc[kand.index.isin(df[p1].index), c].quantile([0.33, 0.67])
        if q.isna().any() or q.iloc[0] == q.iloc[1]:
            continue
        print(f"\n  {lab}:")
        print(f"    {'LANGELIS':<14} {'N':>7} {'DEMEAN. r3':>12} "
              f"{'95% INTERVALAS':>22} {'EUR':>9}")
        for lo, hi, nm in [(-1e9, q.iloc[0], "pigios"),
                           (q.iloc[0], q.iloc[1], "vidutines"),
                           (q.iloc[1], 1e9, "brangios")]:
            sub = kand[(kand[c] > lo) & (kand[c] <= hi)]
            r, n = ivertink(sub, "r3_dm")
            if r is None:
                print(f"    {nm:<14} {n:>7}   per maza imtis")
                continue
            eur = sub["r3"].mean() / 100 * a.pozicija - a.sanaudos_eur
            ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
            print(f"    {nm:<14} {r['n']:>7} {r['mean']:>+11.3f}% {ci:>22} "
                  f"{eur:>+8.2f}€")

    print("\n" + "=" * 100)
    print("B. NUOLAIDA KARTU SU PATVIRTINTU SIGNALU")
    print("Ar prideda prie to, ka jau turim (zemas IBS)?")
    print("=" * 100)
    bazine, _ = ivertink(kand, "r3_dm")
    if bazine:
        print(f"  Vien zemas IBS: {bazine['mean']:+.3f}% (n={bazine['n']})")
        for c, lab in matai:
            if c not in kand or kand[c].notna().sum() < 500:
                continue
            q = kand[c].quantile(0.33)
            sub = kand[kand[c] <= q]
            r, n = ivertink(sub, "r3_dm")
            if r:
                print(f"    + {lab:<28} {r['mean']:>+7.3f}%  "
                      f"skirtumas {r['mean'] - bazine['mean']:>+7.3f}  (n={r['n']})")
            else:
                print(f"    + {lab:<28} per maza imtis ({n})")

    print("\n" + "=" * 100)
    print("SANTRAUKA")
    print("=" * 100)
    print("  Priminimas: P/E cia PASTOVUS per visa laikotarpi, nes istoriniu")
    print("  nemokamai nera. Tai zvilgsnis i ateiti, kuris rezultata daro PER")
    print("  GERA. Jei net taip efekto nera — jo tikrai nera.")
    print("  Jei efektas YRA, ji butina patikrinti su tikrais istoriniais P/E,")
    print("  o tam reiketu mokamo saltinio.")

    print("\nAPRIBOJIMAI: islikimo salismas; fundamentai tik siandieniniai; "
          "\nbankrutavusiu ir nupirktu bendroviu duomenu Yahoo nebeturi, todel "
          "\npigiausiu grupe sistemingai per graži.")


if __name__ == "__main__":
    main()
