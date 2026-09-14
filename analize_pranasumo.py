#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kur tavo pranasumas: ijejime, isejime, ar jo nera.

KODEL SIS TESTAS REIKALINGAS
Ankstesne analize parode, kad 24 pozicijos, uzdarytos per diena, dave 100%
pelningumo. Bet tai NE pranasumo irodymas: tu parduodi, kai esi pliuse, ir
lauki, kai esi minuse. Su tokia taisykle trumpi laikymai BUTINAI bus
pelningi — nuostolingi automatiskai patenka i "ilgai laikytu" grupe.
Tai dispozicijos efektas, ir del jo is laikymo trukmes apie atranka
spresti negalima.

TRYS DALYS, KURIOS TAI ATSKIRIA

A. IJEJIMO KOKYBE (svarbiausia)
   Imam 86 tavo pirkimus su TIKSLIU laiku ir ziurim, kas su akcija vyko
   PO TO — nepriklausomai nuo to, kada tu pardavei. Lyginam su atsitiktiniais
   ijejimais tose paciose akcijose tomis paciomis dienomis.
   Jei tavo ijejimai geresni uz atsitiktinius — atranka veikia.
   Sio testo NEVEIKIA tavo pardavimo sprendimai.

B. ISEJIMO KOKYBE
   Is tu paciu ijejimu taikom fiksuotas taisykles (parduoti po 1, 3, 5 d.,
   ties +1%, ties salyga) ir lyginam su tuo, ka realiai gavai.
   Jei tavo rezultatas geresnis uz visas taisykles — pranasumas valdyme.

C. IJEJIMO LAIKAS
   Tavo ataskaitoje yra laikai iki sekundes. Tikrinam, ar tam tikru paros
   metu tavo ijejimai geresni.

APRIBOJIMAI
Yahoo intraday duoda tik 60 d., todel valandiniai matavimai imanomi tik
naujausiems sandoriams. Senesniems naudojam dienos barus: ijejimo kaina
imama TAVO tikroji, o rezultatas skaiciuojamas iki busimu dienu uzdarymu.
Tai tikslu, nes tavo kaina yra faktas, ne prielaida.
"""

import sys

import numpy as np
import pandas as pd

PIRKIMAI = [{"sym": "4GLD", "yahoo": "4GLD.DE", "data": "2026-08-21", "laikas": "03:32:00", "px": 125.4796, "q": 70.0}, {"sym": "4GLD", "yahoo": "4GLD.DE", "data": "2026-08-24", "laikas": "09:50:05", "px": 128.85, "q": 30.0}, {"sym": "AMD", "yahoo": "AMD.DE", "data": "2026-08-05", "laikas": "02:42:34", "px": 416.95, "q": 38.0}, {"sym": "AMD", "yahoo": "AMD.DE", "data": "2026-08-10", "laikas": "09:33:13", "px": 413.45, "q": 40.0}, {"sym": "AMD", "yahoo": "AMD.DE", "data": "2026-08-10", "laikas": "09:33:47", "px": 411.85, "q": 2.0}, {"sym": "ASM", "yahoo": "ASM.AS", "data": "2026-09-03", "laikas": "11:27:21", "px": 774.2, "q": 23.0}, {"sym": "ASML", "yahoo": "ASML.AS", "data": "2026-04-17", "laikas": "04:19:02", "px": 1216.8, "q": 15.0}, {"sym": "ASML", "yahoo": "ASML.AS", "data": "2026-07-06", "laikas": "10:38:41", "px": 1626.0, "q": 11.0}, {"sym": "ASML", "yahoo": "ASML.AS", "data": "2026-07-22", "laikas": "02:07:34", "px": 1562.4, "q": 10.0}, {"sym": "BAYN", "yahoo": "BAYN.DE", "data": "2026-04-17", "laikas": "06:28:06", "px": 40.28, "q": 24.0}, {"sym": "BESI", "yahoo": "BESI.AS", "data": "2026-07-06", "laikas": "12:27:03", "px": 258.4, "q": 20.0}, {"sym": "BESI", "yahoo": "BESI.AS", "data": "2026-07-06", "laikas": "12:27:36", "px": 258.6, "q": 50.0}, {"sym": "BESI", "yahoo": "BESI.AS", "data": "2026-07-09", "laikas": "03:58:55", "px": 249.7, "q": 70.0}, {"sym": "BESI", "yahoo": "BESI.AS", "data": "2026-08-04", "laikas": "06:26:00", "px": 213.5, "q": 73.0}, {"sym": "BESI", "yahoo": "BESI.AS", "data": "2026-08-26", "laikas": "16:56:02", "px": 192.4, "q": 65.0}, {"sym": "BESI", "yahoo": "BESI.AS", "data": "2026-08-27", "laikas": "06:21:43", "px": 195.65, "q": 30.0}, {"sym": "BESI", "yahoo": "BESI.AS", "data": "2026-08-27", "laikas": "08:40:16", "px": 193.95, "q": 60.0}, {"sym": "BESI", "yahoo": "BESI.AS", "data": "2026-08-27", "laikas": "09:46:20", "px": 192.4, "q": 5.0}, {"sym": "DFTK", "yahoo": "DFTK.DE", "data": "2026-02-18", "laikas": "09:45:17", "px": 3.82, "q": 50.0}, {"sym": "DFTK", "yahoo": "DFTK.DE", "data": "2026-02-26", "laikas": "03:02:33", "px": 2.5, "q": 50.0}, {"sym": "DFTK", "yahoo": "DFTK.DE", "data": "2026-02-26", "laikas": "05:39:10", "px": 2.62, "q": 100.0}, {"sym": "DFTK", "yahoo": "DFTK.DE", "data": "2026-02-26", "laikas": "05:41:29", "px": 2.62, "q": 130.0}, {"sym": "DFTK", "yahoo": "DFTK.DE", "data": "2026-04-09", "laikas": "08:19:55", "px": 1.845, "q": 50.0}, {"sym": "DFTK", "yahoo": "DFTK.DE", "data": "2026-04-29", "laikas": "11:05:10", "px": 1.79, "q": 230.0}, {"sym": "DFTK", "yahoo": "DFTK.DE", "data": "2026-05-07", "laikas": "08:54:08", "px": 1.745, "q": 470.0}, {"sym": "IDR", "yahoo": "IDR.MC", "data": "2026-04-01", "laikas": "08:32:36", "px": 44.64, "q": 15.0}, {"sym": "IDR", "yahoo": "IDR.MC", "data": "2026-06-29", "laikas": "03:20:43", "px": 48.12, "q": 340.0}, {"sym": "IFX", "yahoo": "IFX.DE", "data": "2026-04-02", "laikas": "08:06:45", "px": 37.685, "q": 18.0}, {"sym": "IFX", "yahoo": "IFX.DE", "data": "2026-07-01", "laikas": "10:56:50", "px": 77.61, "q": 220.0}, {"sym": "IFX", "yahoo": "IFX.DE", "data": "2026-07-02", "laikas": "11:11:16", "px": 76.55, "q": 225.0}, {"sym": "IFX", "yahoo": "IFX.DE", "data": "2026-07-06", "laikas": "03:01:05", "px": 76.27, "q": 235.0}, {"sym": "IFX", "yahoo": "IFX.DE", "data": "2026-07-06", "laikas": "03:05:46", "px": 75.52, "q": 3.0}, {"sym": "IFX", "yahoo": "IFX.DE", "data": "2026-07-09", "laikas": "09:52:39", "px": 73.47, "q": 100.0}, {"sym": "IFX", "yahoo": "IFX.DE", "data": "2026-07-09", "laikas": "10:03:28", "px": 73.45, "q": 140.0}, {"sym": "IFX", "yahoo": "IFX.DE", "data": "2026-08-07", "laikas": "08:53:00", "px": 63.340740741, "q": 270.0}, {"sym": "IFX", "yahoo": "IFX.DE", "data": "2026-08-13", "laikas": "03:21:29", "px": 62.74, "q": 289.0}, {"sym": "IFX", "yahoo": "IFX.DE", "data": "2026-08-27", "laikas": "02:58:56", "px": 56.77, "q": 220.0}, {"sym": "IXUA", "yahoo": "IXUA.DE", "data": "2026-01-05", "laikas": "09:33:58", "px": 5.554, "q": 54.0}, {"sym": "IXUA", "yahoo": "IXUA.DE", "data": "2026-01-08", "laikas": "03:04:04", "px": 5.592, "q": 88.0}, {"sym": "IXUA", "yahoo": "IXUA.DE", "data": "2026-01-09", "laikas": "03:04:46", "px": 5.6, "q": 20.0}, {"sym": "IXUA", "yahoo": "IXUA.DE", "data": "2026-01-16", "laikas": "06:37:05", "px": 5.714, "q": 205.0}, {"sym": "MC", "yahoo": "MC.PA", "data": "2026-04-20", "laikas": "10:10:23", "px": 490.0, "q": 40.0}, {"sym": "MC", "yahoo": "MC.PA", "data": "2026-04-21", "laikas": "08:22:22", "px": 490.95, "q": 41.0}, {"sym": "MC", "yahoo": "MC.PA", "data": "2026-04-22", "laikas": "03:35:22", "px": 480.9, "q": 2.0}, {"sym": "MC", "yahoo": "MC.PA", "data": "2026-07-21", "laikas": "10:50:29", "px": 478.75, "q": 33.0}, {"sym": "MC", "yahoo": "MC.PA", "data": "2026-08-12", "laikas": "10:32:02", "px": 461.1, "q": 38.0}, {"sym": "PHAG", "yahoo": "PHAG.AS", "data": "2026-06-08", "laikas": "03:18:45", "px": 53.075, "q": 350.0}, {"sym": "PHAG", "yahoo": "PHAG.AS", "data": "2026-06-08", "laikas": "03:54:03", "px": 52.94, "q": 35.0}, {"sym": "PHAG", "yahoo": "PHAG.AS", "data": "2026-06-08", "laikas": "06:40:18", "px": 53.175, "q": 15.0}, {"sym": "PHAG", "yahoo": "PHAG.AS", "data": "2026-08-21", "laikas": "03:32:23", "px": 53.64, "q": 170.0}, {"sym": "PRX", "yahoo": "PRX.AS", "data": "2026-08-13", "laikas": "06:58:32", "px": 38.01, "q": 450.0}, {"sym": "PRX", "yahoo": "PRX.AS", "data": "2026-08-14", "laikas": "11:09:37", "px": 37.415, "q": 30.0}, {"sym": "RHM", "yahoo": "RHM.DE", "data": "2026-06-15", "laikas": "04:34:07", "px": 1194.4, "q": 18.0}, {"sym": "RHM", "yahoo": "RHM.DE", "data": "2026-07-01", "laikas": "03:05:32", "px": 1007.8, "q": 17.0}, {"sym": "RHM", "yahoo": "RHM.DE", "data": "2026-07-03", "laikas": "05:25:42", "px": 1076.4, "q": 16.0}, {"sym": "RHM", "yahoo": "RHM.DE", "data": "2026-07-21", "laikas": "07:41:45", "px": 994.7, "q": 16.0}, {"sym": "SAP", "yahoo": "SAP.DE", "data": "2026-02-26", "laikas": "04:51:42", "px": 167.06, "q": 25.0}, {"sym": "SAP", "yahoo": "SAP.DE", "data": "2026-04-10", "laikas": "03:57:21", "px": 140.26, "q": 70.0}, {"sym": "SAP", "yahoo": "SAP.DE", "data": "2026-06-19", "laikas": "03:10:07", "px": 135.5, "q": 117.0}, {"sym": "SAP", "yahoo": "SAP.DE", "data": "2026-06-30", "laikas": "10:47:21", "px": 134.32, "q": 125.0}, {"sym": "SAP", "yahoo": "SAP.DE", "data": "2026-07-22", "laikas": "10:34:04", "px": 132.74, "q": 120.0}, {"sym": "STMPA", "yahoo": "STMPA.PA", "data": "2026-07-24", "laikas": "03:23:00", "px": 48.075, "q": 18.0}, {"sym": "VWCE", "yahoo": "VWCE.DE", "data": "2026-01-28", "laikas": "04:35:24", "px": 147.34, "q": 35.0}, {"sym": "VWCE", "yahoo": "VWCE.DE", "data": "2026-02-09", "laikas": "08:23:04", "px": 147.74, "q": 2.0}, {"sym": "YDX", "yahoo": "YDX.DE", "data": "2026-08-28", "laikas": "10:59:14", "px": 181.64, "q": 100.0}, {"sym": "AIAI", "yahoo": "AIAI.L", "data": "2026-02-09", "laikas": "08:22:05", "px": 27.5654, "q": 100.0}, {"sym": "AMD", "yahoo": "AMD.DE", "data": "2026-02-25", "laikas": "11:54:20", "px": 212.2488, "q": 11.0}, {"sym": "AMD", "yahoo": "AMD.DE", "data": "2026-02-27", "laikas": "04:00:00", "px": 204.0, "q": 24.0}, {"sym": "AMZN", "yahoo": "AMZN", "data": "2026-02-12", "laikas": "09:30:00", "px": 203.88, "q": 15.0}, {"sym": "FLY", "yahoo": "FLY", "data": "2026-02-10", "laikas": "13:53:42", "px": 23.03, "q": 2.0}, {"sym": "FLY", "yahoo": "FLY", "data": "2026-02-12", "laikas": "09:30:01", "px": 20.84, "q": 15.0}, {"sym": "FLY", "yahoo": "FLY", "data": "2026-02-12", "laikas": "09:35:09", "px": 20.25, "q": 9.0}, {"sym": "IUCM", "yahoo": "IUCM.L", "data": "2026-02-18", "laikas": "09:11:34", "px": 13.762, "q": 200.0}, {"sym": "IUCM", "yahoo": "IUCM.L", "data": "2026-03-25", "laikas": "10:55:44", "px": 13.657168, "q": 500.0}, {"sym": "IUCM", "yahoo": "IUCM.L", "data": "2026-03-30", "laikas": "05:37:50", "px": 13.124, "q": 60.0}, {"sym": "MU", "yahoo": "MU", "data": "2026-04-02", "laikas": "10:19:50", "px": 362.133, "q": 20.0}, {"sym": "MU", "yahoo": "MU", "data": "2026-04-07", "laikas": "10:20:44", "px": 369.075, "q": 21.0}, {"sym": "ODD", "yahoo": "ODD", "data": "2026-02-26", "laikas": "07:49:56", "px": 14.03, "q": 25.0}, {"sym": "ONDS", "yahoo": "ONDS", "data": "2026-04-08", "laikas": "14:27:26", "px": 9.525625, "q": 800.0}, {"sym": "ONDS", "yahoo": "ONDS", "data": "2026-04-08", "laikas": "15:32:01", "px": 9.42, "q": 90.0}, {"sym": "ONDS", "yahoo": "ONDS", "data": "2026-04-10", "laikas": "04:05:38", "px": 9.17, "q": 140.0}, {"sym": "ONDS", "yahoo": "ONDS", "data": "2026-04-14", "laikas": "12:57:34", "px": 9.3193, "q": 80.0}, {"sym": "TMC", "yahoo": "TMC", "data": "2026-02-25", "laikas": "11:42:15", "px": 6.485, "q": 100.0}, {"sym": "TMC", "yahoo": "TMC", "data": "2026-03-19", "laikas": "10:17:46", "px": 5.325, "q": 60.0}, {"sym": "TMC", "yahoo": "TMC", "data": "2026-04-01", "laikas": "08:02:58", "px": 4.81, "q": 140.0}, {"sym": "TMC", "yahoo": "TMC", "data": "2026-04-02", "laikas": "10:25:07", "px": 4.435, "q": 80.0}]



def flatten(df, sym):
    if isinstance(df.columns, pd.MultiIndex):
        if sym in df.columns.get_level_values(0):
            return df[sym].dropna(how="all")
        if sym in df.columns.get_level_values(-1):
            return df.xs(sym, axis=1, level=-1).dropna(how="all")
    return df.dropna(how="all")


def boot(v, n=20000, seed=5):
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

    syms = sorted({p["yahoo"] for p in PIRKIMAI})
    print(f"Tavo pirkimu: {len(PIRKIMAI)}, instrumentu: {len(syms)}")
    print("Siunciama istorija…")
    raw = yf.download(syms, start="2025-06-01", end="2026-09-30", interval="1d",
                      group_by="ticker", progress=False, auto_adjust=False,
                      threads=True)
    lent = {}
    for s in syms:
        try:
            d = flatten(raw, s).dropna(subset=["Open", "High", "Low", "Close"])
            if len(d) > 60:
                lent[s] = d
        except Exception:
            pass
    print(f"  duomenu turi: {len(lent)} is {len(syms)}\n")

    # ---------- A. IJEJIMO KOKYBE ----------
    print("=" * 100)
    print("A. IJEJIMO KOKYBE — kas vyko PO tavo pirkimo, nepriklausomai nuo pardavimo")
    print("Palyginimas: TAVO kaina vs tos pacios dienos ATSITIKTINE kaina toje akcijoje")
    print("=" * 100)
    rng = np.random.default_rng(11)
    eil, neatitinka = [], []
    for p in PIRKIMAI:
        d = lent.get(p["yahoo"])
        if d is None:
            continue
        dt = pd.Timestamp(p["data"])
        idx = d.index[d.index <= dt]
        if len(idx) == 0:
            continue
        i = d.index.get_loc(idx[-1])
        if i + 6 >= len(d):
            continue
        bar = d.iloc[i]
        # APSAUGA: tavo kaina turi patekti i tos dienos diapazona. Jei ne —
        # vadinasi Yahoo tikeris ne tas (kita birza, kita valiuta, skilimas),
        # ir rezultatas butu beprasmis. Tokius praleidziam ir surasom.
        lo0, hi0 = float(bar["Low"]), float(bar["High"])
        if not (lo0 * 0.90 <= p["px"] <= hi0 * 1.10):
            neatitinka.append((p["sym"], p["data"], p["px"], lo0, hi0))
            continue
        rec = dict(sym=p["sym"], data=p["data"], val=int(p["laikas"][:2]),
                   mano_px=p["px"])
        # Atsitiktine tos dienos kaina tarp L ir H — "jei butum pirkes bet kada"
        lo_, hi_ = float(bar["Low"]), float(bar["High"])
        rec["atsit_px"] = float(rng.uniform(lo_, hi_))
        rec["vid_px"] = (lo_ + hi_) / 2
        # Kur tavo kaina dienos diapazone: 0 = pats dugnas, 1 = virsune
        rec["vieta"] = ((p["px"] - lo_) / (hi_ - lo_)) if hi_ > lo_ else 0.5
        for n in (1, 2, 3, 5):
            c = float(d["Close"].iloc[i + n])
            rec[f"mano_{n}d"] = (c / p["px"] - 1) * 100
            rec[f"atsit_{n}d"] = (c / rec["atsit_px"] - 1) * 100
            rec[f"vid_{n}d"] = (c / rec["vid_px"] - 1) * 100
        eil.append(rec)
    df = pd.DataFrame(eil)
    print(f"Ivertinta pirkimu: {len(df)} is {len(PIRKIMAI)}")
    if neatitinka:
        print(f"  PRALEISTA {len(neatitinka)} — kaina nepateko i dienos diapazona "
              f"(greiciausiai ne tas tikeris arba kita valiuta):")
        for s, dt, px, lo0, hi0 in neatitinka[:8]:
            print(f"    {s:<7} {dt}  tavo {px:>9.2f}  "
                  f"Yahoo diapazonas {lo0:.2f}-{hi0:.2f}")
        if len(neatitinka) > 8:
            print(f"    … ir dar {len(neatitinka) - 8}")
    if len(df) < 20:
        sys.exit("\nPer maza imtis po patikros — tikeriu atitikmenys neteisingi.")
    print()

    print(f"{'LANGAS':<10} {'TAVO KAINA':>12} {'ATSITIKTINE':>13} {'DIENOS VID.':>13} "
          f"{'TAVO - VID.':>12} {'95% INTERVALAS':>22}")
    print("-" * 100)
    for n in (1, 2, 3, 5):
        a = df[f"mano_{n}d"].dropna()
        b_ = df[f"atsit_{n}d"].dropna()
        v = df[f"vid_{n}d"].dropna()
        sk = (df[f"mano_{n}d"] - df[f"vid_{n}d"]).dropna()
        r = boot(sk)
        ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}" if r else "—"
        print(f"{f'+{n} d.':<10} {a.mean():>+11.2f}% {b_.mean():>+12.2f}% "
              f"{v.mean():>+12.2f}% {sk.mean():>+11.3f}% {ci:>22}")
    print("\n  'TAVO - VID.' rodo, kiek geriau pirkai uz tos dienos vidurio kaina.")
    print("  Jei intervalas nekerta nulio — ijejimo laikas turi verte.")

    r = boot(df["vieta"].dropna() - 0.5)
    if r:
        print(f"\n  Kur tavo kaina dienos diapazone: vidutiniskai "
              f"{df['vieta'].mean():.3f} (0 = dugnas, 1 = virsune)")
        print(f"    skirtumas nuo vidurio: {r['mean']:+.3f} "
              f"[{r['lo']:+.3f} .. {r['hi']:+.3f}]")
        print("    neigiamas ir intervalas po nuliu = perki pigiau nei vidutiniskai")

    # ---------- B. ISEJIMO KOKYBE ----------
    print("\n" + "=" * 100)
    print("B. ISEJIMO KOKYBE — ar tavo pardavimas geresnis uz fiksuotas taisykles")
    print("=" * 100)
    print(f"{'TAISYKLE':<24} {'VID. GRAZA':>12} {'PELNINGU':>10} "
          f"{'95% INTERVALAS':>22}")
    print("-" * 100)
    taisykles = []
    for n in (1, 2, 3, 5):
        v = df[f"mano_{n}d"].dropna()
        r = boot(v)
        if r:
            ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
            print(f"{f'parduoti po {n} d.':<24} {v.mean():>+11.2f}% "
                  f"{float((v > 0).mean() * 100):>9.0f}% {ci:>22}")
            taisykles.append((f"po {n} d.", v.mean()))
    # Tavo realus rezultatas is ankstesnes analizes
    print(f"\n  TAVO REALUS rezultatas (is IBKR ataskaitos): +2.05% vidutinis")
    print(f"  (53 uzdarytos pozicijos, 42 pelningos)")
    if taisykles:
        ger = max(taisykles, key=lambda x: x[1])
        print(f"  Geriausia fiksuota taisykle: {ger[0]} ({ger[1]:+.2f}%)")
        if 2.05 > ger[1]:
            print(f"  -> tavo pardavimas GERESNIS uz visas fiksuotas taisykles")
            print(f"     skirtumas {2.05 - ger[1]:+.2f} p. p. — cia gali buti pranasumas")
        else:
            print(f"  -> fiksuota taisykle butu davusi tiek pat arba daugiau")

    # ---------- C. IJEJIMO LAIKAS ----------
    print("\n" + "=" * 100)
    print("C. IJEJIMO LAIKAS — ar tam tikru metu perki geriau")
    print("Laikai is IBKR (JAV Rytu laiku). Europos sesija ~03:00-11:30.")
    print("=" * 100)
    print(f"{'VALANDA':<14} {'N':>5} {'+1 d.':>9} {'+3 d.':>9} {'VIETA DIAPAZONE':>18}")
    print("-" * 100)
    for lo, hi, lab in [(0, 4, "rytas 02-04"), (5, 7, "vidudienis 05-07"),
                        (8, 10, "popietė 08-10"), (11, 23, "vėlyvas 11+")]:
        sub = df[(df["val"] >= lo) & (df["val"] <= hi)]
        if len(sub) < 5:
            continue
        print(f"{lab:<14} {len(sub):>5} {sub['mano_1d'].mean():>+8.2f}% "
              f"{sub['mano_3d'].mean():>+8.2f}% {sub['vieta'].mean():>17.3f}")

    # ---------- SANTRAUKA ----------
    print("\n" + "=" * 100)
    print("SANTRAUKA")
    print("=" * 100)
    sk1 = (df["mano_1d"] - df["vid_1d"]).dropna()
    r1 = boot(sk1)
    if r1 and r1["lo"] > 0:
        print("  IJEJIMAS: tavo pirkimo kaina geresne uz dienos vidurki "
              f"({r1['mean']:+.3f} p. p.) — atranka arba laikas turi verte.")
    elif r1 and r1["hi"] < 0:
        print("  IJEJIMAS: tavo pirkimo kaina PRASTESNE uz dienos vidurki.")
    else:
        print("  IJEJIMAS: tavo kaina nesiskiria nuo tos dienos vidurio kainos.")
        print("           Vadinasi pranasumas ne ijejimo momente.")
    print("\n  Jei ijejimas nesiskiria, o realus rezultatas geresnis uz fiksuotas")
    print("  taisykles, tai pranasumas yra ISEJIME — tame, kada tu nusprendi")
    print("  parduoti. Tai butu vienintele vieta, kurios modulis dar nemodeliuoja.")

    print("\nAPRIBOJIMAI: 86 pirkimai yra maza imtis; atsitiktine kaina imituojama "
          "\nis dienos diapazono, ne is realios intraday sekos; tavo kaina yra "
          "\nfaktas, todel A dalis patikimiausia.")


if __name__ == "__main__":
    main()
