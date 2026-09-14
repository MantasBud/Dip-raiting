#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Likvidumo soku ir sektoriaus ETF backtestas.

DVI HIPOTEZES IS LITERATUROS, ABI PRIESTARAUJA DABARTINIAM MODULIUI

A. PRIVERSTINIAI PARDAVIMAI (Da, Liu & Schaumburg)
   Grizimas prie vidurkio isskaidytas i keturias dalis; reiksmingas tik
   LIKUTIS — reakcija i nefundamentalius kainos pokycius. Ilgosios puses
   pelnas kyla is LIKVIDUMO SOKU: priverstiniu pardavimu, kurie reikalauja
   likvidumo.

   Priverstinis pardavimas atrodo taip: didelis kritimas SU DIDELE APYVARTA
   ir BE NAUJIENOS. Musu modulis tokius atvejus ATMETA (apyv_sant <= 2.5),
   laikydamas juos naujiena. Jei literatura teisinga, mes isimetame
   geriausius kandidatus.

   Cia tikrinam tiesiogiai: ar kritimas su apyvartos suoliu, bet be
   ataskaitos, elgiasi geriau ar blogiau nei kritimas be apyvartos.

B. SEKTORIAUS ETF VIETOJ AKCIJOS
   IBS efektas dokumentuotas stipriausias ETF ir indeksuose, ne atskirose
   akcijose (Pagonidis). Kai visas sektorius uzdaro prie dugno, pirkti
   sektoriaus ETF butu tas pats signalas be imones rizikos.

   Tikrinam: dipas sektoriaus ETF pries dipa atskiroje akcijoje, ta pacia
   isejimo taisykle ir ta pati laikotarpi.

C. LIKUTINE GRAZA (Da, Liu & Schaumburg pagrindinis matas)
   Akcijos graza MINUS sektoriaus ETF graza MINUS rinkos graza. Tai izoliuoja
   "nefundamentalu" judesi. Tikrinam, ar rikiavimas pagal ji geresnis uz
   rikiavima pagal paprasta kritima.

Paleidimas:
    python backtest_likvidumas.py --rinka eu
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


def week_bootstrap(vals, dienos, n=2000, seed=29):
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
    t["C"], t["O1"] = c, o.shift(-1)
    t["ret"] = (c / pc - 1) * 100
    rng_ = (h - l).replace(0, np.nan)
    t["ibs"] = (c - l) / rng_
    t["apyv"] = (c * v).rolling(20).median()
    t["apyv_sant"] = v / v.rolling(21).median().shift(1)
    t["sma20"] = c.rolling(20).mean()
    t["z20"] = (c - t["sma20"]) / c.rolling(20).std()
    for n in range(1, LAIKYMAS + 1):
        t[f"C{n}"] = c.shift(-n)
        t[f"ibs{n}"] = ((c.shift(-n) - l.shift(-n))
                        / (h.shift(-n) - l.shift(-n)).replace(0, np.nan))
    return t


def isejimas(g):
    ijej = g["O1"]
    n = len(g)
    rez = np.full(n, np.nan)
    baigta = np.zeros(n, dtype=bool)
    for i in range(1, LAIKYMAS + 1):
        sal = (~baigta) & (g[f"ibs{i}"] >= EXIT_IBS).fillna(False).to_numpy()
        rez = np.where(sal, (g[f"C{i}"] / ijej - 1) * 100, rez)
        baigta = baigta | sal
    return np.where(baigta, rez, (g[f"C{LAIKYMAS}"] / ijej - 1) * 100)


def surink(yf, syms, sekt_map, metai):
    raw = yf.download(syms, period=f"{metai}y", interval="1d", group_by="ticker",
                      progress=False, auto_adjust=False, threads=True)
    dalys, be_d = [], 0
    for s in syms:
        try:
            d = flatten(raw, s).dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            if len(d) < 300:
                be_d += 1
                continue
            t = rodikliai(d)
            t["tag"] = s
            t["sekt"] = sekt_map.get(s, "kita") if sekt_map else "ETF"
            t["data"] = d.index
            dalys.append(t)
        except Exception:
            be_d += 1
    return (pd.concat(dalys, ignore_index=True) if dalys else None), be_d


def main():
    ap = argparse.ArgumentParser(description="Likvidumo soku backtestas")
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
    etf_map = U.US_ETF if a.rinka == "us" else U.SEKTORIU_ETF
    idx_sym = "SPY" if a.rinka == "us" else U.INDEKSAS
    sekt = U.sektoriai(uni)
    syms = U.visi_tikeriai(uni)

    print(f"{'JAV' if a.rinka == 'us' else 'EUROPOS'} universas: {len(syms)}\n")
    print("Siunciamos akcijos…")
    df, be_d = surink(yf, syms, sekt, a.metai)
    if df is None:
        sys.exit("Nepavyko surinkti akciju.")
    df = df[df["apyv"] >= 5e6].dropna(subset=["O1", f"C{LAIKYMAS}"]).copy()
    df["rez"] = isejimas(df)
    df["rez_dm"] = df["rez"] - df.groupby("data")["rez"].transform("mean")
    print(f"  akciju: {df['tag'].nunique()}, eiluciu: {len(df):,}, be duomenu: {be_d}")

    # Sektoriu ETF ir indeksas
    print("\nSiunciami sektoriu ETF…")
    etfs = sorted(set(etf_map.values()) | {idx_sym})
    etf_df, _ = surink(yf, etfs, None, a.metai)
    if etf_df is not None:
        etf_df["rez"] = isejimas(etf_df)
        etf_df["rez_dm"] = etf_df["rez"] - etf_df.groupby("data")["rez"].transform("mean")
        print(f"  ETF: {etf_df['tag'].nunique()}, eiluciu: {len(etf_df):,}")

    # Sektoriaus ir rinkos grazos akcijoms — likutinei grazai
    try:
        lent = {t: g.set_index("data")["ret"] for t, g in etf_df.groupby("tag")}
        df["_d"] = pd.to_datetime(df["data"])
        df["sekt_ret"] = np.nan
        for sek, e in etf_map.items():
            if e in lent:
                m = df["sekt"] == sek
                df.loc[m, "sekt_ret"] = df.loc[m, "_d"].map(lent[e])
        df["rinkos_ret"] = df["_d"].map(lent.get(idx_sym, pd.Series(dtype=float)))
        # LIKUTINE graza: akcija minus sektorius minus rinka
        df["likutis"] = df["ret"] - df["sekt_ret"].fillna(0) - df["rinkos_ret"].fillna(0)
        print(f"  likutine graza turi: {df['likutis'].notna().sum():,} eiluciu")
    except Exception as e:
        df["likutis"] = np.nan
        print(f"  likutines grazos nepavyko: {str(e)[:60]}")

    riba = pd.to_datetime(df["data"]).quantile(0.5)
    p1 = pd.to_datetime(df["data"]) <= riba
    p2 = ~p1
    IBS_RIBA = float(df.loc[p1, "ibs"].quantile(0.20))
    kand = df[(df["ret"] <= -0.8) & (df["ret"] >= -8.0) & (df["ibs"] <= IBS_RIBA)].copy()
    print(f"\n  kandidatu: {len(kand):,} (IBS <= {IBS_RIBA:.3f})")
    print(f"  padalijimas: 1-a puse iki {riba:%Y-%m-%d}\n")

    # ---------- A. PRIVERSTINIAI PARDAVIMAI ----------
    print("=" * 100)
    print("A. APYVARTOS SUOLIS — ar musu filtras isimeta geriausius kandidatus")
    print("Modulis atmeta apyv_sant > 2.5 kaip 'naujiena'. Literatura sako, kad")
    print("priverstiniai pardavimai (didele apyvarta, be naujienos) yra GERIAUSI.")
    print("=" * 100)
    print(f"{'APYVARTOS SANTYKIS':<26} {'N':>7} {'DEMEAN.':>10} "
          f"{'95% INTERVALAS':>22} {'EUR':>9}")
    print("-" * 100)
    rez_a = []
    for lo, hi, lab in [(0, 0.8, "mazesne nei iprasta"), (0.8, 1.2, "iprasta"),
                        (1.2, 1.8, "1.2-1.8x"), (1.8, 2.5, "1.8-2.5x"),
                        (2.5, 4.0, "2.5-4x (dabar atmetam)"),
                        (4.0, 999, "virs 4x (dabar atmetam)")]:
        sub = kand[(kand["apyv_sant"] > lo) & (kand["apyv_sant"] <= hi)]
        r, n = ivertink(sub, "rez_dm")
        if r is None:
            print(f"{lab:<26} {n:>7}   per maza imtis")
            continue
        eur = sub["rez"].mean() / 100 * a.pozicija - a.sanaudos_eur
        ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
        print(f"{lab:<26} {r['n']:>7} {r['mean']:>+9.3f}% {ci:>22} {eur:>+8.2f}€")
        rez_a.append((lab, (kand["apyv_sant"] > lo) & (kand["apyv_sant"] <= hi), r))

    # Gilus kritimas + didele apyvarta = priverstinis pardavimas
    print("\n  PRIVERSTINIO PARDAVIMO POZYMIS (gilus kritimas + didele apyvarta):")
    for rr, vv, lab in [(-3.0, 2.5, "kritimas > 3% ir apyvarta > 2.5x"),
                        (-5.0, 3.0, "kritimas > 5% ir apyvarta > 3x"),
                        (-3.0, 0, "kritimas > 3%, apyvarta bet kokia")]:
        sub = kand[(kand["ret"] <= rr) & (kand["apyv_sant"] > vv)]
        r, n = ivertink(sub, "rez_dm")
        if r:
            eur = sub["rez"].mean() / 100 * a.pozicija - a.sanaudos_eur
            print(f"    {lab:<36} n={r['n']:>5} {r['mean']:>+7.3f}% "
                  f"[{r['lo']:+.3f}..{r['hi']:+.3f}] {eur:>+7.2f}€")
        else:
            print(f"    {lab:<36} per maza imtis ({n})")

    # ---------- B. SEKTORIAUS ETF ----------
    print("\n" + "=" * 100)
    print("B. SEKTORIAUS ETF PRIES ATSKIRA AKCIJA")
    print("IBS efektas dokumentuotas stipriausias ETF, ne atskirose akcijose.")
    print("=" * 100)
    if etf_df is not None and len(etf_df) > 500:
        e_riba = float(etf_df["ibs"].quantile(0.20))
        e_kand = etf_df[(etf_df["ret"] <= -0.5) & (etf_df["ibs"] <= e_riba)]
        print(f"{'RINKINYS':<26} {'N':>7} {'VID. GRAZA':>12} {'>0 dalis':>10} {'EUR':>9}")
        print("-" * 100)
        for lab, sub in [("Atskiros akcijos", kand), ("Sektoriaus ETF", e_kand)]:
            if len(sub) < 50:
                print(f"{lab:<26} {len(sub):>7}   per maza imtis")
                continue
            eur = sub["rez"].mean() / 100 * a.pozicija - a.sanaudos_eur
            print(f"{lab:<26} {len(sub):>7} {sub['rez'].mean():>+11.3f}% "
                  f"{float((sub['rez'] > 0).mean() * 100):>9.0f}% {eur:>+8.2f}€")
        print("\n  DEMESIO: ETF demeanuoti negalima (ju mazai), todel cia lyginam")
        print("  GRYNAS grazas. ETF svyruoja maziau, todel ju grazos mazesnes —")
        print("  klausimas, ar jos geresnes PO SANAUDU ir su mazesne rizika.")
        if len(e_kand) >= 50:
            print(f"  ETF blogiausias sandoris: {e_kand['rez'].min():.1f}%, "
                  f"akciju: {kand['rez'].min():.1f}%")
    else:
        print("  ETF duomenu nepakako")

    # ---------- C. LIKUTINE GRAZA ----------
    print("\n" + "=" * 100)
    print("C. LIKUTINE GRAZA — ar geriau rikiuoti pagal ja, ne pagal kritima")
    print("Likutis = akcijos kritimas minus sektoriaus minus rinkos.")
    print("=" * 100)
    if kand["likutis"].notna().sum() > 1000:
        q = kand.loc[kand.index.isin(df[p1].index), "likutis"].quantile([0.2, 0.4, 0.6, 0.8])
        print(f"{'LIKUTIS':<26} {'N':>7} {'DEMEAN.':>10} {'95% INTERVALAS':>22}")
        print("-" * 100)
        ribos = [(-1e9, q.iloc[0], "zemiausias (labiausiai)"), (q.iloc[0], q.iloc[1], "2-as"),
                 (q.iloc[1], q.iloc[2], "vidurys"), (q.iloc[2], q.iloc[3], "4-as"),
                 (q.iloc[3], 1e9, "aukstiausias")]
        for lo, hi, lab in ribos:
            sub = kand[(kand["likutis"] > lo) & (kand["likutis"] <= hi)]
            r, n = ivertink(sub, "rez_dm")
            if r is None:
                print(f"{lab:<26} {n:>7}   per maza imtis")
                continue
            ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
            print(f"{lab:<26} {r['n']:>7} {r['mean']:>+9.3f}% {ci:>22}")
        # Palyginimas: rikiavimas pagal likuti vs pagal kritima
        print("\n  Rikiavimo palyginimas (apatinis kvintilis):")
        for c, lab in [("likutis", "pagal likutine graza"), ("ret", "pagal kritima"),
                       ("z20", "pagal Z-bala")]:
            if c not in kand:
                continue
            sub = kand[kand[c] <= kand[c].quantile(0.2)]
            r, n = ivertink(sub, "rez_dm")
            if r:
                print(f"    {lab:<26} {r['mean']:>+7.3f}% "
                      f"[{r['lo']:+.3f}..{r['hi']:+.3f}] (n={r['n']})")
    else:
        print("  likutines grazos duomenu nepakako")

    # ---------- 2 ETAPAS ----------
    print("\n" + "=" * 100)
    print("PATVIRTINIMAS 2-OJE PUSEJE (nematyta)")
    print("=" * 100)
    print(f"{'DYDIS':<26} {'N':>7} {'DEMEAN.':>10} {'95% INTERVALAS':>22} "
          f"{'VERDIKTAS':>14}")
    print("-" * 100)
    tikrinti = [("apyvarta > 2.5x", kand["apyv_sant"] > 2.5),
                ("apyvarta <= 2.5x", kand["apyv_sant"] <= 2.5),
                ("kritimas>3% + apyv>2.5x",
                 (kand["ret"] <= -3.0) & (kand["apyv_sant"] > 2.5))]
    if kand["likutis"].notna().sum() > 1000:
        tikrinti.append(("likutis apatinis kvintilis",
                         kand["likutis"] <= kand["likutis"].quantile(0.2)))
    for lab, kauke in tikrinti:
        sub = kand[kauke & kand.index.isin(df[p2].index)]
        r, n = ivertink(sub, "rez_dm")
        if r is None:
            print(f"{lab:<26} {n:>7}   per maza imtis")
            continue
        v = ("PATVIRTINTA" if r["lo"] > 0 else
             ("PRIESINGA" if r["hi"] < 0 else "nepatvirtinta"))
        ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
        print(f"{lab:<26} {r['n']:>7} {r['mean']:>+9.3f}% {ci:>22} {v:>14}")

    print("\nAPRIBOJIMAI: 'be naujienos' cia aproksimuojamas tik apyvarta — "
          "\nistoriniu naujienu nemokamai nera, todel dalis 'priverstiniu pardavimu' "
          "\nis tikruju yra naujienos; ETF imtis maza (14 instrumentu), todel B "
          "\ndalis yra orientacine; islikimo salismas.")


if __name__ == "__main__":
    main()
