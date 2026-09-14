#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minervini pries musu moduli — tiesioginis palyginimas.

KAS LYGINAMA

A. MINERVINI TRENDO SABLONAS (8 salygos, viesai paskelbtos)
   1. kaina virs 50 d. vidurkio
   2. kaina virs 150 d. vidurkio
   3. kaina virs 200 d. vidurkio
   4. 50 d. vidurkis virs 150 d.
   5. 150 d. vidurkis virs 200 d.
   6. 200 d. vidurkis kyla (didesnis nei pries 20 d.)
   7. kaina ne toliau kaip 25% nuo metu maksimumo
   8. kaina bent 30% virs metu minimumo
   Tai ATRANKOS filtras: jis sako, KURIOS akcijos tinkamos, ne KADA pirkti.

B. MUSU MODULIS
   IBS 0.20-0.50 juostoje (ismatuota is vartotojo 84 pirkimu), zemas Z-balas,
   rinkos rezimas teigiamas, be krintancio peilio, be ataskaitos.

C. DERINYS
   Musu dipo salygos TIK tose akcijose, kurios praeina Minervini sablona.
   Tai logiskiausia jungtis: sablonas atrenka akcijas, musu signalai — momenta.

KAIP LYGINAMA
Visi trys vertinami TOMIS PACIOMIS taisyklemis: ta pati imtis, tas pats
laikotarpis, ta pati isejimo taisykle (IBS > 0.8 arba iki 10 sesiju),
tas pats demeanavimas pagal diena, tas pats dvieju etapu protokolas.
Kitaip palyginimas butu apie matavimo buda, ne apie metoda.

KO TIKETIS
Musu trendo filtrai anksciau nepasitvirtino, o Minervini sablonas placiai
zinomas. Bet konkreti 8 salygu specifikacija su metu maksimumo komponentu
yra nauja — tokio derinio netikrinom, o metu maksimumo efektas turi
atskira akademini pagrinda (George ir Hwang, 2004).

Paleidimas:
    python backtest_minervini.py --rinka eu
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


def week_bootstrap(vals, dienos, n=2000, seed=23):
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
    p_pos = float((b > 0).mean())
    return dict(mean=float(np.mean(vals)), lo=float(lo), hi=float(hi),
                p_two=float(2 * min(p_pos, 1 - p_pos)))


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

    t["sma20"] = c.rolling(20).mean()
    t["sma50"] = c.rolling(50).mean()
    t["sma150"] = c.rolling(150).mean()
    t["sma200"] = c.rolling(200).mean()
    t["z20"] = (c - t["sma20"]) / c.rolling(20).std()

    # --- Minervini 8 salygos ---
    m52 = c.rolling(250).max()
    z52 = c.rolling(250).min()
    t["s1"] = c > t["sma50"]
    t["s2"] = c > t["sma150"]
    t["s3"] = c > t["sma200"]
    t["s4"] = t["sma50"] > t["sma150"]
    t["s5"] = t["sma150"] > t["sma200"]
    t["s6"] = t["sma200"] > t["sma200"].shift(20)
    t["s7"] = c >= m52 * 0.75          # ne toliau kaip 25% nuo metu maksimumo
    t["s8"] = c >= z52 * 1.30          # bent 30% virs metu minimumo
    t["minervini_n"] = (t[["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]]
                        .sum(axis=1))
    t["minervini"] = t["minervini_n"] == 8
    t["nuo_52max"] = (c / m52 - 1) * 100

    raud = (c < pc).astype(int)
    t["raud_serija"] = raud.groupby((raud == 0).cumsum()).cumsum()
    t["zem20"] = l.rolling(20).min().shift(1)

    for n in range(1, LAIKYMAS + 1):
        t[f"C{n}"] = c.shift(-n)
        t[f"ibs{n}"] = ((c.shift(-n) - l.shift(-n))
                        / (h.shift(-n) - l.shift(-n)).replace(0, np.nan))
    return t


def isejimas(g):
    """Ta pati isejimo taisykle visiems metodams: IBS>0.8 arba iki 10 sesiju."""
    ijej = g["O1"]
    n = len(g)
    rez = np.full(n, np.nan)
    baigta = np.zeros(n, dtype=bool)
    for i in range(1, LAIKYMAS + 1):
        sal = (~baigta) & (g[f"ibs{i}"] >= EXIT_IBS).fillna(False).to_numpy()
        rez = np.where(sal, (g[f"C{i}"] / ijej - 1) * 100, rez)
        baigta = baigta | sal
    return np.where(baigta, rez, (g[f"C{LAIKYMAS}"] / ijej - 1) * 100)


def main():
    ap = argparse.ArgumentParser(description="Minervini pries musu moduli")
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
    idx_sym = "SPY" if a.rinka == "us" else U.INDEKSAS
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

    # Rinkos rezimas
    df["rezimas"] = True
    try:
        iraw = yf.download(idx_sym, period=f"{a.metai}y", interval="1d",
                           progress=False, auto_adjust=False)
        ic = flatten(iraw, idx_sym)["Close"].dropna()
        reg = ((ic > ic.rolling(50).mean()) & (ic / ic.shift(20) - 1 > 0))
        m = pd.Series(reg.values, index=[i.date() for i in ic.index])
        df["rezimas"] = pd.to_datetime(df["data"]).dt.date.map(m).fillna(True)
        print(f"  rinkos rezimas teigiamas: {df['rezimas'].mean() * 100:.0f}% dienu")
    except Exception as e:
        print(f"  rezimo nepavyko: {str(e)[:50]}")

    df = df[df["apyv"] >= 5e6].dropna(subset=["O1", f"C{LAIKYMAS}"]).copy()
    df["rez"] = isejimas(df)
    df["rez_dm"] = df["rez"] - df.groupby("data")["rez"].transform("mean")
    print(f"  akciju: {df['tag'].nunique()}, eiluciu: {len(df):,}, be duomenu: {be_d}")

    riba = pd.to_datetime(df["data"]).quantile(0.5)
    p1 = pd.to_datetime(df["data"]) <= riba
    p2 = ~p1
    print(f"  padalijimas: 1-a puse iki {riba:%Y-%m-%d}\n")

    # ---------- METODU APIBREZIMAI ----------
    minervini = df["minervini"]
    musu = (
        (df["ret"] <= -0.8) & (df["ret"] >= -6.0)
        & (df["ibs"] >= 0.20) & (df["ibs"] <= 0.50)
        & (df["raud_serija"] <= 2)
        & (df["C"] > df["zem20"])
        & (df["rezimas"])
    )
    derinys = musu & minervini

    metodai = [
        ("A. Minervini sablonas", minervini),
        ("B. Musu modulis", musu),
        ("C. Derinys", derinys),
        ("   kontrole: atsitiktinai", pd.Series(
            np.random.default_rng(7).random(len(df)) < 0.05, index=df.index)),
    ]

    print("=" * 100)
    print("PALYGINIMAS — visi vertinami ta pacia isejimo taisykle")
    print("=" * 100)
    print(f"{'METODAS':<28} {'N':>8} {'DALIS':>7} {'DEMEAN.':>10} "
          f"{'95% INTERVALAS':>22} {'EUR':>9}")
    print("-" * 100)
    rez1 = []
    for lab, kauke in metodai:
        sub = df[kauke]
        r, n = ivertink(sub, "rez_dm")
        dalis = len(sub) / len(df) * 100
        if r is None:
            print(f"{lab:<28} {n:>8} {dalis:>6.1f}%   per maza imtis")
            continue
        eur = sub["rez"].mean() / 100 * a.pozicija - a.sanaudos_eur
        ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
        print(f"{lab:<28} {r['n']:>8} {dalis:>6.1f}% {r['mean']:>+9.3f}% {ci:>22} "
              f"{eur:>+8.2f}€")
        rez1.append((lab, kauke, r))

    # ---------- KIEK SALYGU UZTENKA ----------
    print("\n" + "=" * 100)
    print("KIEK MINERVINI SALYGU UZTENKA (visos 8 gali buti per griezta)")
    print("=" * 100)
    print(f"{'SALYGU':<12} {'N':>8} {'DALIS':>7} {'DEMEAN.':>10} {'95% INTERVALAS':>22}")
    print("-" * 100)
    for k in (5, 6, 7, 8):
        sub = df[df["minervini_n"] >= k]
        r, n = ivertink(sub, "rez_dm")
        dalis = len(sub) / len(df) * 100
        if r is None:
            print(f"{'>=' + str(k):<12} {n:>8} {dalis:>6.1f}%   per maza imtis")
            continue
        ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
        print(f"{'>=' + str(k):<12} {r['n']:>8} {dalis:>6.1f}% {r['mean']:>+9.3f}% {ci:>22}")

    # ---------- KURI SALYGA SVARBIAUSIA ----------
    print("\n" + "=" * 100)
    print("KURI ATSKIRA SALYGA TURI ITAKOS (musu dipo kandidatams)")
    print("=" * 100)
    bazine, _ = ivertink(df[musu], "rez_dm")
    if bazine:
        print(f"  Vien musu modulis: {bazine['mean']:+.3f}% (n={bazine['n']})")
        sal = [("s1", "kaina > SMA50"), ("s2", "kaina > SMA150"),
               ("s3", "kaina > SMA200"), ("s4", "SMA50 > SMA150"),
               ("s5", "SMA150 > SMA200"), ("s6", "SMA200 kyla"),
               ("s7", "25% nuo metu max"), ("s8", "30% virs metu min")]
        for c, lab in sal:
            sub = df[musu & df[c]]
            r, n = ivertink(sub, "rez_dm")
            if r:
                print(f"    + {lab:<22} {r['mean']:>+7.3f}%  "
                      f"skirtumas {r['mean'] - bazine['mean']:>+7.3f}  (n={r['n']})")
            else:
                print(f"    + {lab:<22} per maza imtis ({n})")

    # ---------- 2 ETAPAS ----------
    print("\n" + "=" * 100)
    print("PATVIRTINIMAS 2-OJE PUSEJE (nematyta)")
    print("=" * 100)
    print(f"{'METODAS':<28} {'N':>8} {'DEMEAN.':>10} {'95% INTERVALAS':>22} "
          f"{'VERDIKTAS':>14}")
    print("-" * 100)
    for lab, kauke, _ in rez1:
        sub = df[kauke & p2]
        r, n = ivertink(sub, "rez_dm")
        if r is None:
            print(f"{lab:<28} {n:>8}   per maza imtis")
            continue
        v = ("PATVIRTINTA" if r["lo"] > 0 else
             ("PRIESINGA" if r["hi"] < 0 else "nepatvirtinta"))
        ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
        print(f"{lab:<28} {r['n']:>8} {r['mean']:>+9.3f}% {ci:>22} {v:>14}")

    print("\n" + "=" * 100)
    print("KAIP SKAITYTI")
    print("=" * 100)
    print("  Jei A geresnis uz B — verta svarstyti metodo keitima.")
    print("  Jei C geresnis uz abu — sablonas prideda prie musu, ir ji verta")
    print("     itraukti kaip papildoma filtra.")
    print("  Jei visi apie nuli — sablonas nieko neduoda, ir klausimas uzdarytas.")
    print("  'DALIS' rodo, kiek laiko metodas apskritai duoda kandidatu.")
    print("     Sablonas gali buti geresnis, bet retas — tada sandoriu butu mazai.")

    print("\nAPRIBOJIMAI: islikimo salismas; Minervini naudoja ir fundamentus "
          "\n(20%+ pelno augimas), kuriu istoriskai nemokamai nera — cia tikrinam "
          "\nTIK technine sablono dali; tikras metodas apima ir VCP sablona bei "
          "\npivot ijejima, kuriu formalizuoti neimanoma be zvilgsnio i ateiti.")


if __name__ == "__main__":
    main()
