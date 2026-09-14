#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tavo sandoriu analize — ka rode signalai tuo metu, kai tu pirkai.

IDEJA
Turim 53 tavo realias pozicijas su tiksliomis datomis ir kainomis. Uz kiekvienos
ju stovi TAVO sprendimas. Jei tavo atranka turi desninguma, jis turetu matytis
signaluose pirkimo diena: pelningu sandoriu metu signalai turetu atrodyti
kitaip nei nuostolingu.

Tai APVERSTAS musu iprastas metodas. Iki siol tikrinom savo sugalvotus signalus
ir ziurejom, ar jie prognozuoja. Cia ziurim i tavo sprendimus ir ieskom, KOKS
signalu derinys juos lydejo.

BUTINAS ISPEJIMAS, KURI REIKIA SKAITYTI PRIES REZULTATUS
53 pozicijos, is ju 42 pelningos. Ieskant desningumo tokioje imtyje su
maždaug 15 rodikliu, kazkas "reiksmingo" atsiras beveik garantuotai vien
is atsitiktinumo. Todel:
  - visi radiniai cia yra HIPOTEZES, ne isvados;
  - bet kuris radinys turi buti patikrintas visame universe per dvieju etapu
    protokola, kaip ir visi ankstesni;
  - jei radinys ten nepasitvirtins, ji reikia atmesti be gailescio.

Papildomas apribojimas: Yahoo intraday duomenu tik 60 d., todel signalai
skaiciuojami is DIENOS baru. IBS cia yra dienos IBS, ne intraday.

Paleidimas:
    python analize_sandoriu.py
"""

import sys
from collections import defaultdict

import numpy as np
import pandas as pd

SANDORIAI = [{"sym": "4GLD", "pirk": "2026-08-21", "pirk_px": 126.4907, "pard": "2026-08-26", "pard_px": 126.65, "pnl": 3.07, "kiekis": 100.0, "yahoo": "4GLD.DE"}, {"sym": "AMD", "pirk": "2026-02-25", "pirk_px": 206.5925, "pard": "2026-03-25", "pard_px": 217.98, "pnl": 397.48, "kiekis": 35.0, "yahoo": "AMD.DE"}, {"sym": "AMD", "pirk": "2026-08-05", "pirk_px": 416.95, "pard": "2026-08-07", "pard_px": 432.35, "pnl": 567.72, "kiekis": 38.0, "yahoo": "AMD.DE"}, {"sym": "AMD", "pirk": "2026-08-10", "pirk_px": 413.3738, "pard": "2026-08-12", "pard_px": 424.0, "pnl": 426.33, "kiekis": 42.0, "yahoo": "AMD.DE"}, {"sym": "ASML", "pirk": "2026-04-17", "pirk_px": 1216.8, "pard": "2026-04-17", "pard_px": 1237.4, "pnl": 290.5, "kiekis": 15.0, "yahoo": "ASML.AS"}, {"sym": "ASML", "pirk": "2026-07-06", "pirk_px": 1626.0, "pard": "2026-07-06", "pard_px": 1629.4, "pnl": 19.32, "kiekis": 11.0, "yahoo": "ASML.AS"}, {"sym": "ASML", "pirk": "2026-07-22", "pirk_px": 1562.4, "pard": "2026-07-22", "pard_px": 1577.4, "pnl": 134.3, "kiekis": 10.0, "yahoo": "ASML.AS"}, {"sym": "BAYN", "pirk": "2026-04-17", "pirk_px": 40.28, "pard": "2026-04-20", "pard_px": 41.07, "pnl": 16.46, "kiekis": 24.0, "yahoo": "BAYN.DE"}, {"sym": "BESI", "pirk": "2026-07-06", "pirk_px": 258.5429, "pard": "2026-07-09", "pard_px": 250.1, "pnl": -608.8, "kiekis": 70.0, "yahoo": "BESI.AS"}, {"sym": "BESI", "pirk": "2026-07-09", "pirk_px": 249.7, "pard": "2026-07-09", "pard_px": 255.7, "pnl": 201.07, "kiekis": 70.0, "yahoo": "BESI.AS"}, {"sym": "BESI", "pirk": "2026-07-24", "pirk_px": 224.8571, "pard": "2026-08-04", "pard_px": 212.4, "pnl": -887.3, "kiekis": 70.0, "yahoo": "BESI.AS"}, {"sym": "BESI", "pirk": "2026-08-04", "pirk_px": 213.5, "pard": "2026-08-05", "pard_px": 220.2, "pnl": 84.28, "kiekis": 73.0, "yahoo": "BESI.AS"}, {"sym": "BESI", "pirk": "2026-08-26", "pirk_px": 192.4, "pard": "2026-08-27", "pard_px": 195.9, "pnl": 15.77, "kiekis": 65.0, "yahoo": "BESI.AS"}, {"sym": "BESI", "pirk": "2026-08-27", "pirk_px": 194.4053, "pard": "2026-08-28", "pard_px": 195.3, "pnl": 65.72, "kiekis": 95.0, "yahoo": "BESI.AS"}, {"sym": "DFTK", "pirk": "2026-02-18", "pirk_px": 2.0766, "pard": "2026-06-18", "pard_px": 1.505, "pnl": -627.4, "kiekis": 1080.0, "yahoo": "DFTK.DE"}, {"sym": "IDR", "pirk": "2026-04-01", "pirk_px": 44.64, "pard": "2026-04-02", "pard_px": 47.32, "pnl": 35.95, "kiekis": 15.0, "yahoo": "IDR.MC"}, {"sym": "IDR", "pirk": "2026-06-29", "pirk_px": 48.12, "pard": "2026-06-30", "pard_px": 48.41, "pnl": 82.19, "kiekis": 340.0, "yahoo": "IDR.MC"}, {"sym": "IFX", "pirk": "2026-04-02", "pirk_px": 37.685, "pard": "2026-04-02", "pard_px": 38.495, "pnl": 12.08, "kiekis": 18.0, "yahoo": "IFX.DE"}, {"sym": "IFX", "pirk": "2026-07-01", "pirk_px": 77.61, "pard": "2026-07-02", "pard_px": 78.84, "pnl": 253.39, "kiekis": 220.0, "yahoo": "IFX.DE"}, {"sym": "IFX", "pirk": "2026-07-02", "pirk_px": 76.55, "pard": "2026-07-03", "pard_px": 78.0, "pnl": 307.74, "kiekis": 225.0, "yahoo": "IFX.DE"}, {"sym": "IFX", "pirk": "2026-07-06", "pirk_px": 76.2605, "pard": "2026-07-06", "pard_px": 77.47, "pnl": 267.28, "kiekis": 238.0, "yahoo": "IFX.DE"}, {"sym": "IFX", "pirk": "2026-07-09", "pirk_px": 73.4583, "pard": "2026-07-21", "pard_px": 65.24, "pnl": -1991.1, "kiekis": 240.0, "yahoo": "IFX.DE"}, {"sym": "IFX", "pirk": "2026-08-07", "pirk_px": 63.3407, "pard": "2026-08-10", "pard_px": 64.24, "pnl": 224.47, "kiekis": 270.0, "yahoo": "IFX.DE"}, {"sym": "IFX", "pirk": "2026-08-13", "pirk_px": 62.74, "pard": "2026-08-13", "pard_px": 62.94, "pnl": 39.64, "kiekis": 289.0, "yahoo": "IFX.DE"}, {"sym": "IFX", "pirk": "2026-08-27", "pirk_px": 56.77, "pard": "2026-08-27", "pard_px": 58.22, "pnl": 306.35, "kiekis": 220.0, "yahoo": "IFX.DE"}, {"sym": "IXUA", "pirk": "2026-01-05", "pirk_px": 5.655, "pard": "2026-01-27", "pard_px": 5.669, "pnl": -1.18, "kiekis": 367.0, "yahoo": "IXUA.DE"}, {"sym": "MC", "pirk": "2026-04-20", "pirk_px": 490.0, "pard": "2026-04-21", "pard_px": 496.8, "pnl": 252.26, "kiekis": 40.0, "yahoo": "MC.PA"}, {"sym": "MC", "pirk": "2026-04-21", "pirk_px": 490.4826, "pard": "2026-06-05", "pard_px": 484.3, "pnl": -287.58, "kiekis": 43.0, "yahoo": "MC.PA"}, {"sym": "MC", "pirk": "2026-07-21", "pirk_px": 478.75, "pard": "2026-07-22", "pard_px": 480.75, "pnl": 4.03, "kiekis": 33.0, "yahoo": "MC.PA"}, {"sym": "MC", "pirk": "2026-08-12", "pirk_px": 461.1, "pard": "2026-08-12", "pard_px": 467.55, "pnl": 47.89, "kiekis": 38.0, "yahoo": "MC.PA"}, {"sym": "PHAG", "pirk": "2026-06-08", "pirk_px": 53.0669, "pard": "2026-06-15", "pard_px": 55.01, "pnl": 752.54, "kiekis": 400.0, "yahoo": "PHAG.AS"}, {"sym": "PHAG", "pirk": "2026-08-21", "pirk_px": 53.64, "pard": "2026-08-27", "pard_px": 53.235, "pnl": -45.96, "kiekis": 170.0, "yahoo": "PHAG.AS"}, {"sym": "PRX", "pirk": "2026-08-13", "pirk_px": 37.9728, "pard": "2026-08-21", "pard_px": 37.83, "pnl": -87.43, "kiekis": 480.0, "yahoo": "PRX.AS"}, {"sym": "RHM", "pirk": "2026-06-15", "pirk_px": 1194.4, "pard": "2026-06-19", "pard_px": 1205.2, "pnl": 124.8, "kiekis": 18.0, "yahoo": "RHM.DE"}, {"sym": "RHM", "pirk": "2026-07-01", "pirk_px": 1007.8, "pard": "2026-07-01", "pard_px": 1016.0, "pnl": 122.2, "kiekis": 17.0, "yahoo": "RHM.DE"}, {"sym": "RHM", "pirk": "2026-07-03", "pirk_px": 1076.4, "pard": "2026-07-03", "pard_px": 1095.6, "pnl": 289.82, "kiekis": 16.0, "yahoo": "RHM.DE"}, {"sym": "RHM", "pirk": "2026-07-21", "pirk_px": 994.7, "pard": "2026-07-21", "pard_px": 1001.2, "pnl": 88.03, "kiekis": 16.0, "yahoo": "RHM.DE"}, {"sym": "SAP", "pirk": "2026-02-26", "pirk_px": 167.06, "pard": "2026-02-26", "pard_px": 172.76, "pnl": 137.95, "kiekis": 25.0, "yahoo": "SAP.DE"}, {"sym": "SAP", "pirk": "2026-04-10", "pirk_px": 140.26, "pard": "2026-04-14", "pard_px": 144.06, "pnl": 256.05, "kiekis": 70.0, "yahoo": "SAP.DE"}, {"sym": "SAP", "pirk": "2026-06-19", "pirk_px": 135.5, "pard": "2026-06-29", "pard_px": 137.34, "pnl": 198.29, "kiekis": 117.0, "yahoo": "SAP.DE"}, {"sym": "SAP", "pirk": "2026-06-30", "pirk_px": 134.32, "pard": "2026-07-01", "pard_px": 136.3, "pnl": 230.59, "kiekis": 125.0, "yahoo": "SAP.DE"}, {"sym": "SAP", "pirk": "2026-07-22", "pirk_px": 132.74, "pard": "2026-07-24", "pard_px": 134.56916666, "pnl": 203.27, "kiekis": 120.0, "yahoo": "SAP.DE"}, {"sym": "STMPA", "pirk": "2026-07-24", "pirk_px": 48.075, "pard": "2026-07-27", "pard_px": 47.685, "pnl": -9.52, "kiekis": 18.0, "yahoo": "STMPA.PA"}, {"sym": "YDX", "pirk": "2026-08-28", "pirk_px": 181.64, "pard": "2026-09-03", "pard_px": 175.432, "pnl": -639.09, "kiekis": 100.0, "yahoo": "YDX.DE"}, {"sym": "AIAI", "pirk": "2026-02-09", "pirk_px": 27.5654, "pard": "2026-02-26", "pard_px": 28.475, "pnl": 87.56, "kiekis": 100.0, "yahoo": "AIAI.L"}, {"sym": "AMZN", "pirk": "2026-02-12", "pirk_px": 203.88, "pard": "2026-02-25", "pard_px": 209.8015, "pnl": 88.09, "kiekis": 15.0, "yahoo": "AMZN"}, {"sym": "IUCM", "pirk": "2026-02-18", "pirk_px": 13.762, "pard": "2026-02-26", "pard_px": 14.024, "pnl": 49.0, "kiekis": 200.0, "yahoo": "IUCM.L"}, {"sym": "IUCM", "pirk": "2026-03-25", "pirk_px": 13.6, "pard": "2026-04-02", "pard_px": 13.572, "pnl": -24.61, "kiekis": 560.0, "yahoo": "IUCM.L"}, {"sym": "MU", "pirk": "2026-04-02", "pirk_px": 362.133, "pard": "2026-04-06", "pard_px": 378.8322, "pnl": 333.12, "kiekis": 20.0, "yahoo": "MU"}, {"sym": "MU", "pirk": "2026-04-07", "pirk_px": 369.075, "pard": "2026-04-07", "pard_px": 375.7542, "pnl": 139.39, "kiekis": 21.0, "yahoo": "MU"}, {"sym": "ODD", "pirk": "2026-02-26", "pirk_px": 14.03, "pard": "2026-03-17", "pard_px": 14.39, "pnl": 8.21, "kiekis": 25.0, "yahoo": "ODD"}, {"sym": "ONDS", "pirk": "2026-04-08", "pirk_px": 9.4573, "pard": "2026-04-15", "pard_px": 9.72454955, "pnl": 287.84, "kiekis": 1110.0, "yahoo": "ONDS"}, {"sym": "TMC", "pirk": "2026-02-25", "pirk_px": 5.2532, "pard": "2026-04-17", "pard_px": 5.4707, "pnl": 79.52, "kiekis": 380.0, "yahoo": "TMC"}]



def flatten(df, sym):
    if isinstance(df.columns, pd.MultiIndex):
        if sym in df.columns.get_level_values(0):
            return df[sym].dropna(how="all")
        if sym in df.columns.get_level_values(-1):
            return df.xs(sym, axis=1, level=-1).dropna(how="all")
    return df.dropna(how="all")


def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def rodikliai(d):
    """Visi signalai, kuriuos testavome per projekta — is dienos baru."""
    c, h, l, v, o = d["Close"], d["High"], d["Low"], d["Volume"], d["Open"]
    pc = c.shift(1)
    t = pd.DataFrame(index=d.index)
    t["C"] = c
    t["dienos_pok"] = (c / pc - 1) * 100
    rng = (h - l).replace(0, np.nan)
    t["ibs"] = (c - l) / rng
    t["sma20"] = c.rolling(20).mean()
    t["sma50"] = c.rolling(50).mean()
    t["sma200"] = c.rolling(200).mean()
    t["z20"] = (c - t["sma20"]) / c.rolling(20).std()
    t["nuo_sma20"] = (c / t["sma20"] - 1) * 100
    t["nuo_sma50"] = (c / t["sma50"] - 1) * 100
    t["nuo_sma200"] = (c / t["sma200"] - 1) * 100
    t["rsi2"] = rsi(c, 2)
    t["rsi14"] = rsi(c, 14)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    t["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / c * 100
    t["apyv_sant"] = v / v.rolling(21).median().shift(1)
    t["mom_5d"] = (c / c.shift(5) - 1) * 100
    t["mom_20d"] = (c / c.shift(20) - 1) * 100
    t["mom_60d"] = (c / c.shift(60) - 1) * 100
    t["nuo_20d_max"] = (c / h.rolling(20).max() - 1) * 100
    t["nuo_52s_max"] = (c / h.rolling(250).max() - 1) * 100
    raud = (c < pc).astype(int)
    t["raud_serija"] = raud.groupby((raud == 0).cumsum()).cumsum()
    zal = (c > pc).astype(int)
    t["zal_serija"] = zal.groupby((zal == 0).cumsum()).cumsum()
    t["naktis"] = (o / pc - 1) * 100
    t["diena"] = (c / o - 1) * 100
    return t


RODIKLIAI = ["dienos_pok", "ibs", "z20", "nuo_sma20", "nuo_sma50", "nuo_sma200",
             "rsi2", "rsi14", "atr", "apyv_sant", "mom_5d", "mom_20d", "mom_60d",
             "nuo_20d_max", "nuo_52s_max", "raud_serija", "zal_serija",
             "naktis", "diena"]


def main():
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Paleisk: pip install yfinance pandas numpy")

    syms = sorted({s["yahoo"] for s in SANDORIAI})
    print(f"Tavo pozicijos: {len(SANDORIAI)} "
          f"({sum(1 for s in SANDORIAI if s['pnl'] > 0)} pelningos, "
          f"{sum(1 for s in SANDORIAI if s['pnl'] <= 0)} nuostolingos)")
    print(f"Skirtingu instrumentu: {len(syms)}\n")
    print("Siunciama istorija…")
    raw = yf.download(syms, start="2025-01-01", end="2026-09-30", interval="1d",
                      group_by="ticker", progress=False, auto_adjust=False,
                      threads=True)

    lentele = {}
    truksta = []
    for s in syms:
        try:
            d = flatten(raw, s).dropna(subset=["Open", "High", "Low", "Close"])
            if len(d) < 100:
                truksta.append(s)
                continue
            lentele[s] = rodikliai(d)
        except Exception:
            truksta.append(s)
    print(f"  duomenu turi: {len(lentele)}; truksta: {', '.join(truksta) or 'nera'}\n")

    # ---------- Signalu reiksmes pirkimo diena ----------
    eil = []
    for s in SANDORIAI:
        t = lentele.get(s["yahoo"])
        if t is None:
            continue
        dt = pd.Timestamp(s["pirk"])
        idx = t.index[t.index <= dt]
        if len(idx) == 0:
            continue
        r = t.loc[idx[-1]]
        rec = dict(sym=s["sym"], pnl=s["pnl"], laimejo=s["pnl"] > 0,
                   pirk=s["pirk"], pard=s["pard"],
                   graza=(s["pard_px"] / s["pirk_px"] - 1) * 100,
                   dienos=(pd.Timestamp(s["pard"]) - dt).days)
        for c in RODIKLIAI:
            rec[c] = float(r[c]) if c in r and r[c] == r[c] else np.nan
        eil.append(rec)
    df = pd.DataFrame(eil)
    print(f"Atkurta signalu: {len(df)} poziciju\n")

    # ---------- A. LAIMEJIMAI PRIES PRALAIMEJIMUS ----------
    print("=" * 100)
    print("A. KAIP ATRODE SIGNALAI PIRKIMO DIENA")
    print("=" * 100)
    print(f"{'RODIKLIS':<16} {'PELNINGOS':>12} {'NUOSTOLINGOS':>14} "
          f"{'SKIRTUMAS':>11} {'t-dydis':>9}")
    print("-" * 100)
    L = df[df["laimejo"]]
    P = df[~df["laimejo"]]
    radiniai = []
    for c in RODIKLIAI:
        a, b = L[c].dropna(), P[c].dropna()
        if len(a) < 10 or len(b) < 5:
            continue
        sk = a.mean() - b.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        tt = sk / se if se > 0 else 0
        zyma = "  <<<" if abs(tt) > 2.0 else ("  <" if abs(tt) > 1.5 else "")
        print(f"{c:<16} {a.mean():>+11.2f} {b.mean():>+13.2f} "
              f"{sk:>+10.2f} {tt:>+8.2f}{zyma}")
        if abs(tt) > 1.5:
            radiniai.append((c, sk, tt))

    print(f"\n  '<' zymi t > 1.5, '<<<' zymi t > 2.0.")
    print(f"  DEMESIO: tikrinant {len(RODIKLIAI)} rodikliu, vidutiniskai 1 is ju")
    print(f"  virsys t=2.0 vien atsitiktinai. Tai hipotezes, ne isvados.")

    # ---------- B. GRAZOS RYSYS ----------
    print("\n" + "=" * 100)
    print("B. RYSYS SU GRAZA (ne tik laimejo/pralaimejo)")
    print("=" * 100)
    print(f"{'RODIKLIS':<16} {'KORELIACIJA':>13} {'|r|':>7}")
    print("-" * 100)
    kor = []
    for c in RODIKLIAI:
        sub = df[[c, "graza"]].dropna()
        if len(sub) < 20:
            continue
        r_ = float(sub[c].corr(sub["graza"]))
        kor.append((c, r_))
    for c, r_ in sorted(kor, key=lambda x: -abs(x[1]))[:10]:
        zyma = "  <<<" if abs(r_) > 0.35 else ("  <" if abs(r_) > 0.25 else "")
        print(f"{c:<16} {r_:>+12.3f} {abs(r_):>6.3f}{zyma}")

    # ---------- C. SET UP'AS ----------
    print("\n" + "=" * 100)
    print("C. KOKS SET UP'AS KARTOJASI PELNINGOSE POZICIJOSE")
    print("=" * 100)
    for c in RODIKLIAI:
        sub = df[[c, "graza", "laimejo"]].dropna()
        if len(sub) < 25:
            continue
        try:
            sub["kv"] = pd.qcut(sub[c], 3, labels=["zemas", "vidutinis", "aukstas"],
                                duplicates="drop")
        except Exception:
            continue
        g = sub.groupby("kv", observed=True).agg(
            n=("graza", "size"), vid=("graza", "mean"), laim=("laimejo", "mean"))
        if len(g) < 3 or g["n"].min() < 6:
            continue
        sk = float(g["vid"].iloc[-1] - g["vid"].iloc[0])
        if abs(sk) < 0.8:
            continue
        print(f"\n  {c}:")
        for lab, row in g.iterrows():
            print(f"    {str(lab):<11} n={int(row['n']):>3}  vid. graza "
                  f"{row['vid']:>+6.2f}%  pelningu {row['laim'] * 100:>3.0f}%")

    # ---------- D. LAIKYMO TRUKME ----------
    print("\n" + "=" * 100)
    print("D. LAIKYMO TRUKME")
    print("=" * 100)
    for lo, hi, lab in [(0, 1, "iki 1 dienos"), (2, 3, "2-3 dienos"),
                        (4, 10, "4-10 dienu"), (11, 999, "virs 10 dienu")]:
        sub = df[(df["dienos"] >= lo) & (df["dienos"] <= hi)]
        if len(sub) < 3:
            continue
        print(f"  {lab:<16} n={len(sub):>3}  vid. graza {sub['graza'].mean():>+6.2f}%  "
              f"pelningu {sub['laimejo'].mean() * 100:>3.0f}%  "
              f"vid. EUR {sub['pnl'].mean():>+8.2f}")

    # ---------- E. SANTRAUKA ----------
    print("\n" + "=" * 100)
    print("SANTRAUKA")
    print("=" * 100)
    if radiniai:
        print("  Rodikliai, kurie skyresi labiausiai (t > 1.5):")
        for c, sk, tt in sorted(radiniai, key=lambda x: -abs(x[2])):
            kryptis = "aukstesnis" if sk > 0 else "zemesnis"
            print(f"    {c:<16} pelningose {kryptis} ({sk:+.2f}, t={tt:+.2f})")
        print("\n  KITAS ZINGSNIS: kiekviena si hipoteze reikia patikrinti visame")
        print("  universe per dvieju etapu protokola. Tik tada ji taps taisykle.")
    else:
        print("  Nei vienas rodiklis reiksmingai neskyre pelningu nuo nuostolingu.")
        print("  Tai reikstu, kad tavo atranka remiasi kazkuo, ko sie rodikliai")
        print("  nemato — arba kad rezultatus leme rinka, ne atranka.")

    print("\nAPRIBOJIMAI: 53 pozicijos yra maza imtis; signalai is DIENOS baru, "
          "\nnes Yahoo intraday duoda tik 60 d.; pirkimo diena imama visa, "
          "\nne tiksli minute; radiniai yra hipotezes generavimas, ne patvirtinimas.")


if __name__ == "__main__":
    main()
