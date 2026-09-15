#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faktorinis priskyrimas — is kur atejo tavo +29%.

METODIKA PERIMTA IS KTD-FIN (arXiv 2605.28359)
Tyrimas patikrino desimt paziangiausiu LLM prekybos agentu su duomenu
nutekejimo kontrole ir nustate, kad ju grazа daugiausia paaiskinama pasyviu
RINKOS ir STILIAUS poveikiu, o nuoseklaus atrankos pranasumo irodymu maza.

Ju metodikos esme: grazа skaidoma i tris dalis, ir tik trecioji yra igudis:
  1. RINKOS BETA — kiek uzdirbai vien todel, kad buvai rinkoje
  2. STILIAUS POVEIKIS — kiek uzdirbai todel, kad prekiavai tais sektoriais,
     kuriems tais metais sekesi (technologijos, gynyba)
  3. LIKUTIS — kiek lieka TAVO sprendimams: kuria akcija ir kada

KODEL TAI SVARBU
Tavo TWR +29.29% per 8 men. ir +13.99% per 12 men. yra tikri skaiciai.
Bet jei didzioji dalis yra rinka ir sektoriai, tai reikstu, kad ta pati
butum gaves pirkdamas indeksa arba sektoriaus ETF ir nieko nedarydamas.
Sis testas atsako, kiek lieka tau.

KAIP MATUOJAMA
Kiekvienai tavo pozicijai skaiciuojam TRIS grazas tuo paciu laikotarpiu:
  - tavo reali graza (pirkimo ir pardavimo kainos)
  - rinkos graza (EXSA.DE tomis paciomis dienomis)
  - sektoriaus graza (to sektoriaus ETF tomis paciomis dienomis)
Likutis = tavo graza minus sektoriaus. Sektoriaus pranasumas = sektorius
minus rinka.

APRIBOJIMAI
53 pozicijos yra maza imtis; dalis poziciju JAV akcijose, kurioms Europos
sektoriaus ETF netinka — jos vertinamos atskirai; islikimo salismo cia nera,
nes tai tavo realus sandoriai.
"""

import sys

import numpy as np
import pandas as pd

SANDORIAI = [{"sym": "4GLD", "pirk": "2026-08-21", "pirk_px": 126.4907, "pard": "2026-08-26", "pard_px": 126.65, "pnl": 3.07, "kiekis": 100.0, "yahoo": "4GLD.DE"}, {"sym": "AMD", "pirk": "2026-02-25", "pirk_px": 206.5925, "pard": "2026-03-25", "pard_px": 217.98, "pnl": 397.48, "kiekis": 35.0, "yahoo": "AMD.DE"}, {"sym": "AMD", "pirk": "2026-08-05", "pirk_px": 416.95, "pard": "2026-08-07", "pard_px": 432.35, "pnl": 567.72, "kiekis": 38.0, "yahoo": "AMD.DE"}, {"sym": "AMD", "pirk": "2026-08-10", "pirk_px": 413.3738, "pard": "2026-08-12", "pard_px": 424.0, "pnl": 426.33, "kiekis": 42.0, "yahoo": "AMD.DE"}, {"sym": "ASML", "pirk": "2026-04-17", "pirk_px": 1216.8, "pard": "2026-04-17", "pard_px": 1237.4, "pnl": 290.5, "kiekis": 15.0, "yahoo": "ASML.AS"}, {"sym": "ASML", "pirk": "2026-07-06", "pirk_px": 1626.0, "pard": "2026-07-06", "pard_px": 1629.4, "pnl": 19.32, "kiekis": 11.0, "yahoo": "ASML.AS"}, {"sym": "ASML", "pirk": "2026-07-22", "pirk_px": 1562.4, "pard": "2026-07-22", "pard_px": 1577.4, "pnl": 134.3, "kiekis": 10.0, "yahoo": "ASML.AS"}, {"sym": "BAYN", "pirk": "2026-04-17", "pirk_px": 40.28, "pard": "2026-04-20", "pard_px": 41.07, "pnl": 16.46, "kiekis": 24.0, "yahoo": "BAYN.DE"}, {"sym": "BESI", "pirk": "2026-07-06", "pirk_px": 258.5429, "pard": "2026-07-09", "pard_px": 250.1, "pnl": -608.8, "kiekis": 70.0, "yahoo": "BESI.AS"}, {"sym": "BESI", "pirk": "2026-07-09", "pirk_px": 249.7, "pard": "2026-07-09", "pard_px": 255.7, "pnl": 201.07, "kiekis": 70.0, "yahoo": "BESI.AS"}, {"sym": "BESI", "pirk": "2026-07-24", "pirk_px": 224.8571, "pard": "2026-08-04", "pard_px": 212.4, "pnl": -887.3, "kiekis": 70.0, "yahoo": "BESI.AS"}, {"sym": "BESI", "pirk": "2026-08-04", "pirk_px": 213.5, "pard": "2026-08-05", "pard_px": 220.2, "pnl": 84.28, "kiekis": 73.0, "yahoo": "BESI.AS"}, {"sym": "BESI", "pirk": "2026-08-26", "pirk_px": 192.4, "pard": "2026-08-27", "pard_px": 195.9, "pnl": 15.77, "kiekis": 65.0, "yahoo": "BESI.AS"}, {"sym": "BESI", "pirk": "2026-08-27", "pirk_px": 194.4053, "pard": "2026-08-28", "pard_px": 195.3, "pnl": 65.72, "kiekis": 95.0, "yahoo": "BESI.AS"}, {"sym": "DFTK", "pirk": "2026-02-18", "pirk_px": 2.0766, "pard": "2026-06-18", "pard_px": 1.505, "pnl": -627.4, "kiekis": 1080.0, "yahoo": "DFTK.DE"}, {"sym": "IDR", "pirk": "2026-04-01", "pirk_px": 44.64, "pard": "2026-04-02", "pard_px": 47.32, "pnl": 35.95, "kiekis": 15.0, "yahoo": "IDR.MC"}, {"sym": "IDR", "pirk": "2026-06-29", "pirk_px": 48.12, "pard": "2026-06-30", "pard_px": 48.41, "pnl": 82.19, "kiekis": 340.0, "yahoo": "IDR.MC"}, {"sym": "IFX", "pirk": "2026-04-02", "pirk_px": 37.685, "pard": "2026-04-02", "pard_px": 38.495, "pnl": 12.08, "kiekis": 18.0, "yahoo": "IFX.DE"}, {"sym": "IFX", "pirk": "2026-07-01", "pirk_px": 77.61, "pard": "2026-07-02", "pard_px": 78.84, "pnl": 253.39, "kiekis": 220.0, "yahoo": "IFX.DE"}, {"sym": "IFX", "pirk": "2026-07-02", "pirk_px": 76.55, "pard": "2026-07-03", "pard_px": 78.0, "pnl": 307.74, "kiekis": 225.0, "yahoo": "IFX.DE"}, {"sym": "IFX", "pirk": "2026-07-06", "pirk_px": 76.2605, "pard": "2026-07-06", "pard_px": 77.47, "pnl": 267.28, "kiekis": 238.0, "yahoo": "IFX.DE"}, {"sym": "IFX", "pirk": "2026-07-09", "pirk_px": 73.4583, "pard": "2026-07-21", "pard_px": 65.24, "pnl": -1991.1, "kiekis": 240.0, "yahoo": "IFX.DE"}, {"sym": "IFX", "pirk": "2026-08-07", "pirk_px": 63.3407, "pard": "2026-08-10", "pard_px": 64.24, "pnl": 224.47, "kiekis": 270.0, "yahoo": "IFX.DE"}, {"sym": "IFX", "pirk": "2026-08-13", "pirk_px": 62.74, "pard": "2026-08-13", "pard_px": 62.94, "pnl": 39.64, "kiekis": 289.0, "yahoo": "IFX.DE"}, {"sym": "IFX", "pirk": "2026-08-27", "pirk_px": 56.77, "pard": "2026-08-27", "pard_px": 58.22, "pnl": 306.35, "kiekis": 220.0, "yahoo": "IFX.DE"}, {"sym": "IXUA", "pirk": "2026-01-05", "pirk_px": 5.655, "pard": "2026-01-27", "pard_px": 5.669, "pnl": -1.18, "kiekis": 367.0, "yahoo": "IXUA.DE"}, {"sym": "MC", "pirk": "2026-04-20", "pirk_px": 490.0, "pard": "2026-04-21", "pard_px": 496.8, "pnl": 252.26, "kiekis": 40.0, "yahoo": "MC.PA"}, {"sym": "MC", "pirk": "2026-04-21", "pirk_px": 490.4826, "pard": "2026-06-05", "pard_px": 484.3, "pnl": -287.58, "kiekis": 43.0, "yahoo": "MC.PA"}, {"sym": "MC", "pirk": "2026-07-21", "pirk_px": 478.75, "pard": "2026-07-22", "pard_px": 480.75, "pnl": 4.03, "kiekis": 33.0, "yahoo": "MC.PA"}, {"sym": "MC", "pirk": "2026-08-12", "pirk_px": 461.1, "pard": "2026-08-12", "pard_px": 467.55, "pnl": 47.89, "kiekis": 38.0, "yahoo": "MC.PA"}, {"sym": "PHAG", "pirk": "2026-06-08", "pirk_px": 53.0669, "pard": "2026-06-15", "pard_px": 55.01, "pnl": 752.54, "kiekis": 400.0, "yahoo": "PHAG.AS"}, {"sym": "PHAG", "pirk": "2026-08-21", "pirk_px": 53.64, "pard": "2026-08-27", "pard_px": 53.235, "pnl": -45.96, "kiekis": 170.0, "yahoo": "PHAG.AS"}, {"sym": "PRX", "pirk": "2026-08-13", "pirk_px": 37.9728, "pard": "2026-08-21", "pard_px": 37.83, "pnl": -87.43, "kiekis": 480.0, "yahoo": "PRX.AS"}, {"sym": "RHM", "pirk": "2026-06-15", "pirk_px": 1194.4, "pard": "2026-06-19", "pard_px": 1205.2, "pnl": 124.8, "kiekis": 18.0, "yahoo": "RHM.DE"}, {"sym": "RHM", "pirk": "2026-07-01", "pirk_px": 1007.8, "pard": "2026-07-01", "pard_px": 1016.0, "pnl": 122.2, "kiekis": 17.0, "yahoo": "RHM.DE"}, {"sym": "RHM", "pirk": "2026-07-03", "pirk_px": 1076.4, "pard": "2026-07-03", "pard_px": 1095.6, "pnl": 289.82, "kiekis": 16.0, "yahoo": "RHM.DE"}, {"sym": "RHM", "pirk": "2026-07-21", "pirk_px": 994.7, "pard": "2026-07-21", "pard_px": 1001.2, "pnl": 88.03, "kiekis": 16.0, "yahoo": "RHM.DE"}, {"sym": "SAP", "pirk": "2026-02-26", "pirk_px": 167.06, "pard": "2026-02-26", "pard_px": 172.76, "pnl": 137.95, "kiekis": 25.0, "yahoo": "SAP.DE"}, {"sym": "SAP", "pirk": "2026-04-10", "pirk_px": 140.26, "pard": "2026-04-14", "pard_px": 144.06, "pnl": 256.05, "kiekis": 70.0, "yahoo": "SAP.DE"}, {"sym": "SAP", "pirk": "2026-06-19", "pirk_px": 135.5, "pard": "2026-06-29", "pard_px": 137.34, "pnl": 198.29, "kiekis": 117.0, "yahoo": "SAP.DE"}, {"sym": "SAP", "pirk": "2026-06-30", "pirk_px": 134.32, "pard": "2026-07-01", "pard_px": 136.3, "pnl": 230.59, "kiekis": 125.0, "yahoo": "SAP.DE"}, {"sym": "SAP", "pirk": "2026-07-22", "pirk_px": 132.74, "pard": "2026-07-24", "pard_px": 134.56916666, "pnl": 203.27, "kiekis": 120.0, "yahoo": "SAP.DE"}, {"sym": "STMPA", "pirk": "2026-07-24", "pirk_px": 48.075, "pard": "2026-07-27", "pard_px": 47.685, "pnl": -9.52, "kiekis": 18.0, "yahoo": "STMPA.PA"}, {"sym": "YDX", "pirk": "2026-08-28", "pirk_px": 181.64, "pard": "2026-09-03", "pard_px": 175.432, "pnl": -639.09, "kiekis": 100.0, "yahoo": "YDX.DE"}, {"sym": "AIAI", "pirk": "2026-02-09", "pirk_px": 27.5654, "pard": "2026-02-26", "pard_px": 28.475, "pnl": 87.56, "kiekis": 100.0, "yahoo": "AIAI.L"}, {"sym": "AMZN", "pirk": "2026-02-12", "pirk_px": 203.88, "pard": "2026-02-25", "pard_px": 209.8015, "pnl": 88.09, "kiekis": 15.0, "yahoo": "AMZN"}, {"sym": "IUCM", "pirk": "2026-02-18", "pirk_px": 13.762, "pard": "2026-02-26", "pard_px": 14.024, "pnl": 49.0, "kiekis": 200.0, "yahoo": "IUCM.L"}, {"sym": "IUCM", "pirk": "2026-03-25", "pirk_px": 13.6, "pard": "2026-04-02", "pard_px": 13.572, "pnl": -24.61, "kiekis": 560.0, "yahoo": "IUCM.L"}, {"sym": "MU", "pirk": "2026-04-02", "pirk_px": 362.133, "pard": "2026-04-06", "pard_px": 378.8322, "pnl": 333.12, "kiekis": 20.0, "yahoo": "MU"}, {"sym": "MU", "pirk": "2026-04-07", "pirk_px": 369.075, "pard": "2026-04-07", "pard_px": 375.7542, "pnl": 139.39, "kiekis": 21.0, "yahoo": "MU"}, {"sym": "ODD", "pirk": "2026-02-26", "pirk_px": 14.03, "pard": "2026-03-17", "pard_px": 14.39, "pnl": 8.21, "kiekis": 25.0, "yahoo": "ODD"}, {"sym": "ONDS", "pirk": "2026-04-08", "pirk_px": 9.4573, "pard": "2026-04-15", "pard_px": 9.72454955, "pnl": 287.84, "kiekis": 1110.0, "yahoo": "ONDS"}, {"sym": "TMC", "pirk": "2026-02-25", "pirk_px": 5.2532, "pard": "2026-04-17", "pard_px": 5.4707, "pnl": 79.52, "kiekis": 380.0, "yahoo": "TMC"}]


# Sektoriaus ETF kiekvienai pozicijai. JAV akcijoms — SPY, nes Europos
# sektoriaus ETF joms netinka; jos vertinamos atskirai.
SEKT_ETF = {
    "AMD": "EXV3.DE", "ASML": "EXV3.DE", "ASM": "EXV3.DE", "BESI": "EXV3.DE",
    "IFX": "EXV3.DE", "SAP": "EXV3.DE", "STMPA": "EXV3.DE", "PRX": "EXV3.DE",
    "IDR": "EXH4.DE", "RHM": "EXH4.DE",
    "MC": "EXH6.DE",
    "BAYN": "EXV4.DE",
    "PHAG": "4GLD.DE", "4GLD": "4GLD.DE",
    "VWCE": "EXSA.DE", "IXUA": "EXSA.DE", "AIAI": "EXSA.DE", "IUCM": "EXSA.DE",
    "YDX": "EXV3.DE", "DFTK": "EXSA.DE",
}
JAV = {"AMZN", "MU", "ONDS", "TMC", "ODD", "FLY", "NAGE", "HNST", "PPCB", "NVNO"}
RINKA = "EXSA.DE"


def flatten(df, sym):
    if isinstance(df.columns, pd.MultiIndex):
        if sym in df.columns.get_level_values(0):
            return df[sym].dropna(how="all")
        if sym in df.columns.get_level_values(-1):
            return df.xs(sym, axis=1, level=-1).dropna(how="all")
    return df.dropna(how="all")


def graza(ser, d1, d2):
    """Graza tarp dvieju datu is uzdarymo kainu."""
    try:
        i1 = ser.index[ser.index <= pd.Timestamp(d1)]
        i2 = ser.index[ser.index <= pd.Timestamp(d2)]
        if len(i1) == 0 or len(i2) == 0:
            return np.nan
        a, b = float(ser.loc[i1[-1]]), float(ser.loc[i2[-1]])
        return (b / a - 1) * 100 if a > 0 else np.nan
    except Exception:
        return np.nan


def boot(v, n=20000, seed=41):
    v = np.asarray([x for x in v if x == x])
    if len(v) < 8:
        return None
    rng = np.random.default_rng(seed)
    b = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(n)])
    return dict(mean=float(v.mean()), lo=float(np.percentile(b, 2.5)),
                hi=float(np.percentile(b, 97.5)), n=len(v))


def main():
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Paleisk: pip install yfinance pandas numpy")

    etfs = sorted(set(SEKT_ETF.values()) | {RINKA, "SPY"})
    print(f"Pozicijos: {len(SANDORIAI)}")
    print(f"Siunciami indeksai ir sektoriu ETF: {', '.join(etfs)}\n")
    raw = yf.download(etfs, start="2025-12-01", end="2026-09-30", interval="1d",
                      group_by="ticker", progress=False, auto_adjust=False,
                      threads=True)
    lent = {}
    for e in etfs:
        try:
            d = flatten(raw, e).dropna(subset=["Close"])
            if len(d) > 50:
                lent[e] = d["Close"]
        except Exception:
            pass
    print(f"  duomenu turi: {len(lent)} is {len(etfs)}")
    truksta = [e for e in etfs if e not in lent]
    if truksta:
        print(f"  truksta: {', '.join(truksta)}")

    eil = []
    for s in SANDORIAI:
        sym = s["sym"]
        jav = sym in JAV
        etf = "SPY" if jav else SEKT_ETF.get(sym, RINKA)
        rinka_e = "SPY" if jav else RINKA
        mano = (s["pard_px"] / s["pirk_px"] - 1) * 100
        sekt_g = graza(lent[etf], s["pirk"], s["pard"]) if etf in lent else np.nan
        rink_g = graza(lent[rinka_e], s["pirk"], s["pard"]) if rinka_e in lent else np.nan
        eil.append(dict(sym=sym, jav=jav, pirk=s["pirk"], pard=s["pard"],
                        pnl=s["pnl"], mano=mano, sekt=sekt_g, rinka=rink_g,
                        etf=etf,
                        likutis=mano - sekt_g if sekt_g == sekt_g else np.nan,
                        sekt_pranasumas=sekt_g - rink_g
                        if (sekt_g == sekt_g and rink_g == rink_g) else np.nan))
    df = pd.DataFrame(eil)
    print(f"  ivertinta poziciju: {df['sekt'].notna().sum()} is {len(df)}\n")

    # ---------- A. PAGRINDINIS SKAIDYMAS ----------
    print("=" * 96)
    print("A. IS KUR ATEJO GRAZA (vidutiniskai vienai pozicijai)")
    print("=" * 96)
    print(f"{'DEDAMOJI':<34} {'VIDURKIS':>11} {'95% INTERVALAS':>24} {'N':>5}")
    print("-" * 96)
    for c, lab in [("mano", "1. Tavo reali graza"),
                   ("rinka", "2. Rinka (EXSA / SPY)"),
                   ("sekt_pranasumas", "3. Sektoriaus pranasumas"),
                   ("likutis", "4. LIKUTIS — tavo sprendimai")]:
        r = boot(df[c].dropna())
        if r is None:
            print(f"{lab:<34} per maza imtis")
            continue
        ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
        zyma = "  <- svarbiausia" if c == "likutis" else ""
        print(f"{lab:<34} {r['mean']:>+10.3f}% {ci:>24} {r['n']:>5}{zyma}")

    r_m = boot(df["mano"].dropna())
    r_r = boot(df["rinka"].dropna())
    r_l = boot(df["likutis"].dropna())
    if r_m and r_r and r_l:
        print(f"\n  Patikra: rinka {r_r['mean']:+.3f} + sektorius "
              f"{boot(df['sekt_pranasumas'].dropna())['mean']:+.3f} + likutis "
              f"{r_l['mean']:+.3f} = {r_r['mean'] + boot(df['sekt_pranasumas'].dropna())['mean'] + r_l['mean']:+.3f}")
        print(f"  Tavo reali graza: {r_m['mean']:+.3f}%")
        dalis = (r_l["mean"] / r_m["mean"] * 100) if r_m["mean"] else 0
        print(f"\n  TAVO SPRENDIMAMS tenka {dalis:.0f}% rezultato.")
        if r_l["lo"] > 0:
            print("  Likutis reiksmingai teigiamas — atrankos pranasumas YRA.")
        elif r_l["hi"] < 0:
            print("  Likutis reiksmingai neigiamas — sektoriaus ETF butu davеs daugiau.")
        else:
            print("  Likutis neatskiriamas nuo nulio — sektoriaus ETF butu davеs")
            print("  maždaug tiek pat, be atskiros imones rizikos.")

    # ---------- B. PAGAL SEKTORIU ----------
    print("\n" + "=" * 96)
    print("B. KURIUOSE SEKTORIUOSE TAVO SPRENDIMAI PRIDEJO")
    print("=" * 96)
    print(f"{'ETF':<12} {'N':>4} {'TAVO':>9} {'SEKTORIUS':>11} {'LIKUTIS':>10}")
    print("-" * 96)
    for e, g in df.dropna(subset=["likutis"]).groupby("etf"):
        if len(g) < 2:
            continue
        print(f"{e:<12} {len(g):>4} {g['mano'].mean():>+8.2f}% "
              f"{g['sekt'].mean():>+10.2f}% {g['likutis'].mean():>+9.2f}%")

    # ---------- C. EUROPA PRIES JAV ----------
    print("\n" + "=" * 96)
    print("C. EUROPA PRIES JAV")
    print("=" * 96)
    for jav, lab in [(False, "Europos akcijos"), (True, "JAV akcijos")]:
        g = df[(df["jav"] == jav)].dropna(subset=["likutis"])
        if len(g) < 3:
            continue
        r = boot(g["likutis"])
        ci = f"[{r['lo']:+.2f}..{r['hi']:+.2f}]" if r else ""
        print(f"  {lab:<20} n={len(g):>3}  tavo {g['mano'].mean():>+6.2f}%  "
              f"likutis {g['likutis'].mean():>+6.2f}% {ci}")

    # ---------- D. KUR LIKUTIS DIDZIAUSIAS ----------
    print("\n" + "=" * 96)
    print("D. DIDZIAUSI IR MAZIAUSI LIKUCIAI (tavo sprendimo indelis)")
    print("=" * 96)
    srt = df.dropna(subset=["likutis"]).sort_values("likutis", ascending=False)
    print("  Geriausi:")
    for _, r in srt.head(6).iterrows():
        print(f"    {r['sym']:<7} {r['pirk']} -> {r['pard']}  tavo {r['mano']:>+7.2f}%  "
              f"sektorius {r['sekt']:>+6.2f}%  likutis {r['likutis']:>+7.2f}%")
    print("  Blogiausi:")
    for _, r in srt.tail(6).iloc[::-1].iterrows():
        print(f"    {r['sym']:<7} {r['pirk']} -> {r['pard']}  tavo {r['mano']:>+7.2f}%  "
              f"sektorius {r['sekt']:>+6.2f}%  likutis {r['likutis']:>+7.2f}%")

    print("\n" + "=" * 96)
    print("KAIP SKAITYTI")
    print("=" * 96)
    print("  Eilute 4 (LIKUTIS) yra vienintele, kuri priklauso TAU.")
    print("  Eilutes 2 ir 3 butum gaves nupirkes indeksa ar sektoriaus ETF")
    print("  ir nieko nedarydamas — be komisiniu ir be laiko sanaudu.")
    print("  KTD-FIN tyrime desimt LLM agentu turejo teigiama bendra grazа,")
    print("  bet likutis buvo apie nuli. Tai standartinis rezultatas.")

    print("\nAPRIBOJIMAI: 53 pozicijos yra maza imtis; sektoriaus ETF priskirti "
          "\nrankomis; JAV akcijoms naudotas SPY, ne ju sektorius; grazos "
          "\nskaiciuotos is dienos uzdarymu, o tu pirkai dienos viduryje.")


if __name__ == "__main__":
    main()
