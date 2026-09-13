#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nakties lango ir laukimo efekto backtestas.

DVI HIPOTEZES, ABI UZRASYTOS PRIES MATAVIMA

A. NAKTIES LANGAS (Lou, Polk & Skouras, JFE 2019)
   Literatura: akciju grazа pasiskirsto netolygiai tarp nakties ir dienos, o
   trumpalaikis grizimas prie vidurkio susidaro DAUGIAUSIA NAKTI. Musu IBS ir
   Z-balas yra butent trumpalaikio grizimo signalai, bet visus matavimus
   dareme nuo atidarymo iki uzdarymo arba nuo uzdarymo iki uzdarymo.
   Nakties lango (uzdarymas -> kitas atidarymas) neismatavome ne karto.

   Hipoteze: signalo pranasumas nakties lange stipresnis nei dienos lange.

B. LAUKIMO EFEKTAS (pastebeta musu pacio teste, 2026-09)
   Europos imtyje pirkimas signalo diena dave +0.98 EUR, o palaukus viena
   diena — nuo +9.94 iki +12.71 EUR, NEPRIKLAUSOMAI nuo krypties. Tai rodo,
   kad nauda ateina ne is trigerio salygos, o is paties laukimo.
   JAV imtyje to nebuvo (+44.23 pries +47.22), todel pakartojimo nera.

   Hipoteze: pirmoji diena po signalo sistemingai prastesne uz tolesnes.

Paleidimas:
    python backtest_naktis.py --rinka eu
    python backtest_naktis.py --rinka us
"""

import argparse
import sys

import numpy as np
import pandas as pd

import universas as U

MIN_IVYKIU = 200
MIN_DIENU = 40


def week_bootstrap(vals, dienos, n=2000, seed=11):
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
    p_pos = float((boot > 0).mean())
    return dict(mean=float(np.mean(vals)), lo=float(lo), hi=float(hi),
                p_two=float(2 * min(p_pos, 1 - p_pos)), dienos=len(vals))


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


def flatten(df, sym):
    if isinstance(df.columns, pd.MultiIndex):
        if sym in df.columns.get_level_values(0):
            return df[sym].dropna(how="all")
        if sym in df.columns.get_level_values(-1):
            return df.xs(sym, axis=1, level=-1).dropna(how="all")
    return df.dropna(how="all")


def rsi(c, n=2):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def paruosk(d):
    c, h, l, v, o = (d["Close"], d["High"], d["Low"], d["Volume"], d["Open"])
    pc = c.shift(1)
    t = pd.DataFrame(index=d.index)
    t["C"], t["O"], t["H"], t["L"] = c, o, h, l
    t["ret"] = (c / pc - 1) * 100
    rng = (h - l).replace(0, np.nan)
    t["ibs"] = (c - l) / rng
    t["sma20"] = c.rolling(20).mean()
    t["z20"] = (c - t["sma20"]) / c.rolling(20).std()
    t["rsi2"] = rsi(c, 2)
    t["apyv"] = (c * v).rolling(20).median()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    t["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / c * 100

    # --- A. LANGU SKAIDYMAS ---
    # naktis:  D uzdarymas -> D+1 atidarymas
    # diena:   D+1 atidarymas -> D+1 uzdarymas
    # visa:    D uzdarymas -> D+1 uzdarymas
    t["naktis_1"] = (o.shift(-1) / c - 1) * 100
    t["diena_1"] = (c.shift(-1) / o.shift(-1) - 1) * 100
    t["visa_1"] = (c.shift(-1) / c - 1) * 100

    # Antra ir trecia naktis — ar efektas islieka
    t["naktis_2"] = (o.shift(-2) / c.shift(-1) - 1) * 100
    t["naktis_3"] = (o.shift(-3) / c.shift(-2) - 1) * 100
    t["diena_2"] = (c.shift(-2) / o.shift(-2) - 1) * 100

    # Sukauptos naktys ir dienos per 3 sesijas
    t["naktys_3"] = t["naktis_1"] + t["naktis_2"] + t["naktis_3"]
    t["dienos_3"] = t["diena_1"] + t["diena_2"] + (c.shift(-3) / o.shift(-3) - 1) * 100

    # --- B. LAUKIMO EFEKTAS ---
    # Pirkimas signalo dienos uzdarymu vs kitos dienos uzdarymu, laikymas 3 sesijas
    t["nuo_D"] = (c.shift(-3) / c - 1) * 100          # perki D uzdarymu
    t["nuo_D1"] = (c.shift(-4) / c.shift(-1) - 1) * 100   # perki D+1 uzdarymu
    t["nuo_D2"] = (c.shift(-5) / c.shift(-2) - 1) * 100   # perki D+2 uzdarymu
    # Kiekvienos atskiros dienos grazа po signalo
    for i in range(1, 6):
        t[f"d{i}"] = (c.shift(-i) / c.shift(-(i - 1)) - 1) * 100
    return t


def main():
    ap = argparse.ArgumentParser(description="Nakties lango backtestas")
    ap.add_argument("--metai", type=int, default=10)
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
    print(f"{'JAV' if a.rinka == 'us' else 'EUROPOS'} universas: {len(syms)} akcijos\n")
    raw = yf.download(syms, period=f"{a.metai}y", interval="1d", group_by="ticker",
                      progress=False, auto_adjust=False, threads=True)

    dalys, be_d = [], 0
    for s in syms:
        try:
            d = flatten(raw, s).dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            if len(d) < 400:
                be_d += 1
                continue
            t = paruosk(d)
            t["tag"], t["data"] = s, d.index
            dalys.append(t)
        except Exception:
            be_d += 1
    if not dalys:
        sys.exit("Nepavyko surinkti duomenu.")
    df = pd.concat(dalys, ignore_index=True)
    df = df[(df["ret"] <= -0.8) & (df["ret"] >= -6.0) & (df["apyv"] >= 5e6)].copy()
    print(f"  akciju: {df['tag'].nunique()}, ijejimo tasku: {len(df):,}, be duomenu: {be_d}")

    # Demeanavimas pagal diena kiekvienam langui
    langai = ["naktis_1", "diena_1", "visa_1", "naktis_2", "naktis_3", "diena_2",
              "naktys_3", "dienos_3", "nuo_D", "nuo_D1", "nuo_D2",
              "d1", "d2", "d3", "d4", "d5"]
    for c in langai:
        if c in df:
            df[c + "_dm"] = df[c] - df.groupby("data")[c].transform("mean")

    riba = pd.to_datetime(df["data"]).quantile(0.5)
    p1 = pd.to_datetime(df["data"]) <= riba
    p2 = ~p1
    IBS_RIBA = float(df.loc[p1, "ibs"].quantile(0.20))
    print(f"  kandidato riba (IBS apatinis kvintilis): {IBS_RIBA:.3f}")
    print(f"  padalijimas: 1-a puse iki {riba:%Y-%m-%d}\n")

    kand = df[df["ibs"] <= IBS_RIBA]

    # ---------- A. NAKTIS PRIES DIENA ----------
    print("=" * 100)
    print("A. NAKTIS PRIES DIENA — kur susidaro grizimo grazа")
    print("Literatura (Lou, Polk & Skouras 2019): trumpalaikis grizimas susidaro naktį.")
    print("=" * 100)
    print(f"{'LANGAS':<28} {'N':>7} {'VID.':>9} {'DEMEAN.':>10} "
          f"{'95% INTERVALAS':>22} {'EUR':>9}")
    print("-" * 100)
    for c, lab in [("naktis_1", "1 naktis (uzdar.->atidar.)"),
                   ("diena_1", "1 diena (atidar.->uzdar.)"),
                   ("visa_1", "visa para"),
                   ("naktis_2", "2 naktis"),
                   ("diena_2", "2 diena"),
                   ("naktis_3", "3 naktis"),
                   ("naktys_3", "3 naktys sudejus"),
                   ("dienos_3", "3 dienos sudejus")]:
        if c + "_dm" not in kand:
            continue
        r, n = ivertink(kand, c + "_dm")
        vid = kand[c].mean()
        eur = vid / 100 * a.pozicija - a.sanaudos_eur
        if r is None:
            print(f"{lab:<28} {n:>7}   per maza imtis")
            continue
        ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
        print(f"{lab:<28} {r['n']:>7} {vid:>+8.3f}% {r['mean']:>+9.3f}% {ci:>22} "
              f"{eur:>+8.2f}€")

    # Ar nakties pranasumas priklauso nuo signalo stiprumo
    print("\n  Nakties langas pagal IBS kvintilius (ar signalas veikia butent nakti):")
    q = df.loc[p1, "ibs"].quantile([0.2, 0.4, 0.6, 0.8])
    ribos = [(-1, q.iloc[0], "zemiausias"), (q.iloc[0], q.iloc[1], "2-as"),
             (q.iloc[1], q.iloc[2], "vidurys"), (q.iloc[2], q.iloc[3], "4-as"),
             (q.iloc[3], 2, "aukstiausias")]
    for lo, hi, lab in ribos:
        sub = df[(df["ibs"] > lo) & (df["ibs"] <= hi)]
        rn, _ = ivertink(sub, "naktis_1_dm")
        rd, _ = ivertink(sub, "diena_1_dm")
        if rn and rd:
            print(f"    IBS {lab:<14} naktis {rn['mean']:>+7.3f}%  "
                  f"diena {rd['mean']:>+7.3f}%  skirtumas {rn['mean'] - rd['mean']:>+7.3f}")

    # ---------- B. LAUKIMO EFEKTAS ----------
    print("\n" + "=" * 100)
    print("B. LAUKIMO EFEKTAS — ar pirmoji diena po signalo prastesne")
    print("Pastebeta musu teste: Europoje pirkimas signalo diena dave +0.98 EUR,")
    print("o palaukus viena diena +9.94..+12.71 EUR nepriklausomai nuo krypties.")
    print("=" * 100)
    print(f"{'KADA PERKAMA':<28} {'N':>7} {'VID.':>9} {'DEMEAN.':>10} "
          f"{'95% INTERVALAS':>22} {'EUR':>9}")
    print("-" * 100)
    for c, lab in [("nuo_D", "signalo dienos uzdarymu"),
                   ("nuo_D1", "kitos dienos uzdarymu"),
                   ("nuo_D2", "po dvieju dienu")]:
        r, n = ivertink(kand, c + "_dm")
        vid = kand[c].mean()
        eur = vid / 100 * a.pozicija - a.sanaudos_eur
        if r is None:
            print(f"{lab:<28} {n:>7}   per maza imtis")
            continue
        ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
        print(f"{lab:<28} {r['n']:>7} {vid:>+8.3f}% {r['mean']:>+9.3f}% {ci:>22} "
              f"{eur:>+8.2f}€")

    print("\n  Kiekvienos dienos po signalo grazа atskirai:")
    print(f"    {'DIENA':<10} {'VID.':>9} {'DEMEAN.':>10} {'95% INTERVALAS':>22}")
    for i in range(1, 6):
        r, n = ivertink(kand, f"d{i}_dm")
        if r:
            ci_d = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
            print(f"    {'D+' + str(i):<10} {kand[f'd{i}'].mean():>+8.3f}% "
                  f"{r['mean']:>+9.3f}% "
                  f"{ci_d:>22}")

    # ---------- PATVIRTINIMAS 2-OJE PUSEJE ----------
    print("\n" + "=" * 100)
    print("PATVIRTINIMAS 2-OJE PUSEJE (nematyta)")
    print("=" * 100)
    print(f"{'DYDIS':<28} {'N':>7} {'DEMEAN.':>10} {'95% INTERVALAS':>22} "
          f"{'VERDIKTAS':>14}")
    print("-" * 100)
    kand2 = df[(df["ibs"] <= IBS_RIBA) & p2]
    for c, lab in [("naktis_1", "1 naktis"), ("diena_1", "1 diena"),
                   ("naktys_3", "3 naktys"), ("nuo_D", "pirkti signalo diena"),
                   ("nuo_D1", "pirkti kita diena")]:
        r, n = ivertink(kand2, c + "_dm")
        if r is None:
            print(f"{lab:<28} {n:>7}   per maza imtis")
            continue
        v = ("PATVIRTINTA" if r["lo"] > 0 else
             ("PRIESINGA" if r["hi"] < 0 else "nepatvirtinta"))
        ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
        print(f"{lab:<28} {r['n']:>7} {r['mean']:>+9.3f}% {ci:>22} {v:>14}")

    print("\n" + "=" * 100)
    print(f"GALUTINIS VERTINIMAS (atemus {a.sanaudos_eur} EUR sanaudu)")
    print("Nakties strategija: pirkti uzdarymo metu, parduoti kito ryto atidarymu.")
    print("=" * 100)
    n1 = kand["naktis_1"].mean()
    d1 = kand["diena_1"].mean()
    print(f"  1 naktis:   bruto {n1:+.3f}%  neto "
          f"{n1 / 100 * a.pozicija - a.sanaudos_eur:+.2f}€ vienam sandoriui")
    print(f"  1 diena:    bruto {d1:+.3f}%  neto "
          f"{d1 / 100 * a.pozicija - a.sanaudos_eur:+.2f}€")
    print(f"  Skirtumas:  {n1 - d1:+.3f} p. p. nakties naudai"
          if n1 > d1 else f"  Skirtumas:  {n1 - d1:+.3f} p. p. dienos naudai")
    print("\n  DEMESIO: nakties strategija reikalauja pirkti PRIES pat uzdarymа —")
    print("  17:25-17:30 Berlyno laiku, kai spread'as platejа. Realus vykdymas")
    print("  butu blogesnis nei rodo sie skaiciai.")

    print("\nAPRIBOJIMAI: islikimo salismas; atidarymo kainos Yahoo duomenyse "
          "\nkartais netikslios (auckciono kaina); sanaudos atimtos tik galutineje "
          "\nlenteleje; sprendziam pagal 2-osios puses eilute, ne pagal pjuvius.")


if __name__ == "__main__":
    main()
