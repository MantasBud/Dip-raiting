#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kokybes rodikliu backtestas — ar pelningos, mazai skolingos bendroves saugesnes.

RODIKLIAI IS LITERATUROS

1. BRUTO PELNINGUMAS (Novy-Marx 2013, "The Other Side of Value")
   gross profits / total assets. Prognostine galia panasi i kainos ir balansines
   vertes santyki. Patvirtinta 19 issivysciusiu rinku; Europai Dimensional
   nustate 3.6% metine premija per 1982-2014.

2. PIOTROSKI F-BALAS (2000) — 9 dvejetainiai signalai:
   pelningumas (ROA>0, pinigu srautas>0, ROA auga, srautas>pelnas),
   svertas (skola mazeja, likvidumas auga, nera naujo akciju leidimo),
   efektyvumas (marza auga, turto apyvarta auga).
   Aukstas balas vertes akcijose dave 34.1% pries 7.8% per metus.
   CIA skaiciuojam DALINI F-bala — turim tik dabartinius duomenis, ne
   pokycius, todel vertinam tik tai, ka galima: ROA, marza, likvidumas, skola.

3. MSCI KOKYBES APIBREZIMAS
   aukstas ROE, stabilus pelno augimas, mazas finansinis svertas.

KRITINIS APRIBOJIMAS — SKAITYK PRIES REZULTATUS
yfinance duoda TIK SIANDIENINIUS fundamentus. Todel:
  - rodiklis laikomas pastoviu per visa laikotarpi;
  - tai zvilgsnis i ateiti, kuris VIDURKIUS daro per gerus;
  - bankrutavusiu ir nupirktu bendroviu Yahoo nebeturi, todel "blogos
    kokybes" grupe yra per gerа — blogiausi atvejai tiesiog dingе.

TODEL PAGRINDINIS MATAS CIA NE VIDURKIS, O UODEGA.
Klausimas ne "ar kokybe prognozuoja graza" (to su siais duomenimis
patikimai neismatuosi), o "ar blogos kokybes akcijose dipas baigiasi
blogiau blogiausiais atvejais". Uodegos salismas veikia ta pacia kryptimi
kaip vidurkio, todel jei net taip skirtumas matomas — jis tikras ir
greiciausiai DIDESNIS nei rodo skaiciai.

Paleidimas:
    python backtest_kokybe.py --rinka eu
"""

import argparse
import sys

import numpy as np
import pandas as pd

import universas as U

MIN_IVYKIU = 150
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


def week_bootstrap(vals, dienos, n=2000, seed=31):
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
    """Kokybes rodikliai. Po viena tikeri — letas zingsnis."""
    out, be = {}, 0
    for i, s in enumerate(syms):
        try:
            inf = yf.Ticker(s).info or {}

            def g(k, lo=-1e9, hi=1e9):
                v = inf.get(k)
                try:
                    v = float(v)
                    return v if lo < v < hi else np.nan
                except Exception:
                    return np.nan

            bruto = g("grossProfits")
            turtas = g("totalAssets")
            # Novy-Marx: bruto pelnas / turtas. Jei totalAssets nera, imam
            # apytiksle versija per pajamas ir marza.
            gp = (bruto / turtas) if (bruto == bruto and turtas == turtas
                                      and turtas > 0) else np.nan
            if gp != gp:
                pj, gm = g("totalRevenue"), g("grossMargins", -5, 5)
                mc = g("marketCap")
                if pj == pj and gm == gm and mc == mc and mc > 0:
                    gp = (pj * gm) / mc         # pakaitalas: bruto pelnas / verte

            out[s] = dict(
                gp=gp,
                roe=g("returnOnEquity", -10, 10),
                roa=g("returnOnAssets", -10, 10),
                marza=g("profitMargins", -10, 10),
                bruto_marza=g("grossMargins", -5, 5),
                skola=g("debtToEquity", 0, 2000),
                likvid=g("currentRatio", 0, 50),
                eps=g("trailingEps"),
                pajamu_aug=g("revenueGrowth", -5, 20),
                pelno_aug=g("earningsGrowth", -20, 50),
                fcf=g("freeCashflow"),
            )
        except Exception:
            be += 1
        if verbose and (i + 1) % 60 == 0:
            print(f"    …{i + 1}/{len(syms)}")
    if verbose:
        turi = sum(1 for v in out.values() if v.get("roe") == v.get("roe"))
        print(f"  fundamentu turi: {turi} is {len(syms)}; nepavyko: {be}")
    return out


def f_balas(f):
    """Dalinis Piotroski F-balas is to, ka turim (0-5, ne 0-9).

    Tikri 9 signalai reikalauja POKYCIU pries praeitus metus, o mes turim
    tik dabartines reiksmes. Todel cia tik penki statiniai signalai.
    """
    b = 0
    if f.get("roa", np.nan) > 0:
        b += 1
    if f.get("fcf", np.nan) > 0:
        b += 1
    if f.get("skola", np.nan) < 100:            # skola < 100% nuosavybes
        b += 1
    if f.get("likvid", np.nan) > 1.0:           # trumpalaikis turtas > isipareigojimu
        b += 1
    if f.get("marza", np.nan) > 0:
        b += 1
    return b


def main():
    ap = argparse.ArgumentParser(description="Kokybes rodikliu backtestas")
    ap.add_argument("--metai", type=int, default=8)
    ap.add_argument("--rinka", default="eu", choices=["eu", "us"])
    ap.add_argument("--pozicija", type=float, default=18000.0)
    a = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Paleisk: pip install yfinance pandas numpy")

    uni = U.UNIVERSAS_US if a.rinka == "us" else U.UNIVERSAS
    syms = U.visi_tikeriai(uni)
    print(f"{'JAV' if a.rinka == 'us' else 'EUROPOS'} universas: {len(syms)}\n")
    print("Siunciami fundamentai (letai, po viena)…")
    fund = fundamentai(yf, syms)

    print("\nSiunciama kainu istorija…")
    raw = yf.download(syms, period=f"{a.metai}y", interval="1d", group_by="ticker",
                      progress=False, auto_adjust=False, threads=True)

    dalys, be_d = [], 0
    for s in syms:
        f = fund.get(s)
        if not f:
            be_d += 1
            continue
        try:
            d = flatten(raw, s).dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            if len(d) < 300:
                be_d += 1
                continue
            c, h, l, v, o = (d["Close"], d["High"], d["Low"], d["Volume"], d["Open"])
            pc = c.shift(1)
            t = pd.DataFrame(index=d.index)
            t["ret"] = (c / pc - 1) * 100
            rng_ = (h - l).replace(0, np.nan)
            t["ibs"] = (c - l) / rng_
            t["apyv"] = (c * v).rolling(20).median()
            t["O1"] = o.shift(-1)
            for n in range(1, LAIKYMAS + 1):
                t[f"C{n}"] = c.shift(-n)
                t[f"ibs{n}"] = ((c.shift(-n) - l.shift(-n))
                                / (h.shift(-n) - l.shift(-n)).replace(0, np.nan))
            t["tag"], t["data"] = s, d.index
            for k, val in f.items():
                t[k] = val
            t["fbal"] = f_balas(f)
            dalys.append(t)
        except Exception:
            be_d += 1

    if not dalys:
        sys.exit("Nepavyko surinkti duomenu.")
    df = pd.concat(dalys, ignore_index=True)
    df = df[df["apyv"] >= 5e6].dropna(subset=["O1", f"C{LAIKYMAS}"]).copy()

    # Isejimas: ta pati taisykle kaip modulyje
    ijej = df["O1"]
    rez = np.full(len(df), np.nan)
    baigta = np.zeros(len(df), dtype=bool)
    for i in range(1, LAIKYMAS + 1):
        sal = (~baigta) & (df[f"ibs{i}"] >= EXIT_IBS).fillna(False).to_numpy()
        rez = np.where(sal, (df[f"C{i}"] / ijej - 1) * 100, rez)
        baigta = baigta | sal
    df["rez"] = np.where(baigta, rez, (df[f"C{LAIKYMAS}"] / ijej - 1) * 100)
    df["rez_dm"] = df["rez"] - df.groupby("data")["rez"].transform("mean")

    riba = pd.to_datetime(df["data"]).quantile(0.5)
    p1 = pd.to_datetime(df["data"]) <= riba
    IBS_RIBA = float(df.loc[p1, "ibs"].quantile(0.20))
    kand = df[(df["ret"] <= -0.8) & (df["ret"] >= -8.0)
              & (df["ibs"] <= IBS_RIBA)].copy()
    print(f"  akciju: {df['tag'].nunique()}, kandidatu: {len(kand):,}, "
          f"be duomenu: {be_d}\n")

    matai = [("gp", "Bruto pelningumas (Novy-Marx)"), ("roe", "ROE"),
             ("roa", "ROA"), ("marza", "Pelno marza"),
             ("bruto_marza", "Bruto marza"), ("skola", "Skola / nuosavybe"),
             ("likvid", "Likvidumo rodiklis"), ("fbal", "Dalinis F-balas")]

    # ---------- A. VIDURKIAI ----------
    print("=" * 100)
    print("A. VIDUTINIS REZULTATAS PAGAL KOKYBE")
    print("Priminimas: fundamentai SIANDIENINIAI, todel cia yra zvilgsnis i ateiti.")
    print("=" * 100)
    for c, lab in matai:
        if c not in kand or kand[c].notna().sum() < 800:
            continue
        q = kand.loc[kand.index.isin(df[p1].index), c].quantile([0.33, 0.67])
        if q.isna().any() or q.iloc[0] == q.iloc[1]:
            continue
        atv = c == "skola"          # skolai: maziau = geriau
        print(f"\n  {lab}:")
        langeliai = [(-1e9, q.iloc[0], "geriausias" if atv else "prasciausias"),
                     (q.iloc[0], q.iloc[1], "vidutinis"),
                     (q.iloc[1], 1e9, "prasciausias" if atv else "geriausias")]
        for lo, hi, nm in langeliai:
            sub = kand[(kand[c] > lo) & (kand[c] <= hi)]
            r, n = ivertink(sub, "rez_dm")
            if r is None:
                print(f"    {nm:<14} {n:>7}   per maza imtis")
                continue
            print(f"    {nm:<14} {r['n']:>7} {r['mean']:>+9.3f}% "
                  f"[{r['lo']:+.3f}..{r['hi']:+.3f}]")

    # ---------- B. UODEGA — PAGRINDINIS MATAS ----------
    print("\n" + "=" * 100)
    print("B. UODEGOS RIZIKA — PAGRINDINIS SIO TESTO KLAUSIMAS")
    print("Ar blogos kokybes akcijose dipas baigiasi blogiau BLOGIAUSIAIS atvejais.")
    print("=" * 100)
    print(f"{'RODIKLIS':<30} {'GRUPE':<14} {'N':>7} {'5% blog.':>10} "
          f"{'blogiausias':>12} {'< -10%':>8} {'EUR blog.':>11}")
    print("-" * 100)
    for c, lab in matai:
        if c not in kand or kand[c].notna().sum() < 800:
            continue
        q = kand.loc[kand.index.isin(df[p1].index), c].quantile([0.33, 0.67])
        if q.isna().any() or q.iloc[0] == q.iloc[1]:
            continue
        atv = c == "skola"
        for lo, hi, nm in [(-1e9, q.iloc[0], "geriausias" if atv else "prasciausias"),
                           (q.iloc[1], 1e9, "prasciausias" if atv else "geriausias")]:
            sub = kand[(kand[c] > lo) & (kand[c] <= hi)]
            if len(sub) < 200:
                continue
            p5 = float(np.percentile(sub["rez"], 5))
            blog = float(sub["rez"].min())
            dalis = float((sub["rez"] <= -10).mean() * 100)
            print(f"{lab:<30} {nm:<14} {len(sub):>7} {p5:>9.2f}% {blog:>11.1f}% "
                  f"{dalis:>7.2f}% {blog / 100 * a.pozicija:>10,.0f}€")

    # ---------- C. NUOSTOLINGOS BENDROVES ----------
    print("\n" + "=" * 100)
    print("C. NUOSTOLINGOS PRIES PELNINGAS (paprasciausias pjuvis)")
    print("=" * 100)
    for kauke, lab in [(kand["eps"] > 0, "pelninga (EPS > 0)"),
                       (kand["eps"] <= 0, "nuostolinga (EPS <= 0)"),
                       (kand["marza"] > 0.05, "marza > 5%"),
                       (kand["marza"] <= 0, "neigiama marza")]:
        sub = kand[kauke]
        if len(sub) < 100:
            print(f"  {lab:<26} n={len(sub):>6}   per maza imtis")
            continue
        r, _ = ivertink(sub, "rez_dm")
        vid = f"{r['mean']:+.3f}%" if r else "—"
        print(f"  {lab:<26} n={len(sub):>6}  vid {vid:>9}  "
              f"5% blog. {np.percentile(sub['rez'], 5):>7.2f}%  "
              f"blogiausias {sub['rez'].min():>7.1f}%  "
              f"< -10%: {float((sub['rez'] <= -10).mean() * 100):.2f}%")

    print("\n" + "=" * 100)
    print("KAIP SKAITYTI")
    print("=" * 100)
    print("  Vidurkiai (A) yra nepatikimi — zvilgsnis i ateiti juos issukia.")
    print("  Uodega (B, C) yra tai, del ko sis testas darytas: jei blogos")
    print("  kokybes grupeje blogiausi atvejai gilesni, universo apribojimas")
    print("  pagristas kaip RIZIKOS valdymas, net be irodyto pranasumo.")
    print("  Islikimo salismas veikia ta pacia kryptimi, todel tikras skirtumas")
    print("  greiciausiai DIDESNIS nei cia matomas.")


if __name__ == "__main__":
    main()
