#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dienos kilimo backtestas — pirkti kylancia, parduoti ta pacia diena ar kita ryta.

HIPOTEZE (vartotojo, uzrasyta 2026-09-15)
Akcija, kuri nuo pat ryto stabiliai kyla, iki uzdarymo ar kito ryto duos
nedideli pelna. Laikymas VALANDOMIS, ne dienomis. Nereikia dipo, atsistatymo
ar trendo — reikia, kad ji kiltu dabar ir kiltu toliau iki isejimo.

KODEL TAI NAUJAS MATAVIMAS
Ankstesni testai apie "kylancias" matavo LAIKYMA DIENOMIS:
  - tesinys po kilimo: Europa -0.029%, JAV -0.101%
  - prasidedantis trendas: -0.539%
  - naktinis tarpas > 2% -> ta diena -0.303%
  - vartotojo 16 pirkimu ties dienos virsune -> -2.72% KITA DIENA
Visi jie pirko ir laike 1-10 dienu. Ne vienas nematavo pirkimo dienos
viduryje su isejimu TA PACIA DIENA arba kito ryto atidarymu.
Tai skirtingas langas ir skirtinga hipoteze.

KA TIKRINAM
A. Ijejimas skirtingu valandu, kai akcija kyla; isejimas ta diena arba kita ryta
B. Ar stipresnis kilimas geriau (rangavimas pagal kilima)
C. Ar VWAP ir apyvarta prideda
D. Ar veikia krentancioje rinkoje — butent tada, kai to reikia
E. Palyginimas su dipo kandidatais tuo paciu laikotarpiu

DUOMENYS
Valandiniai barai, 720 dienu. Europos sesija 9:00-17:30 = 9 barai.
Ijejimas baro UZDARYMU (realiai vykdytum kitos minutes rinkos kaina).

Paleidimas:
    python backtest_kilimas.py --rinka eu
"""

import argparse
import sys

import numpy as np
import pandas as pd

import universas as U

MIN_IVYKIU = 150
MIN_DIENU = 40
SANAUDOS_PCT = 0.028          # 5 EUR nuo 18 000


def flatten(df, sym):
    if isinstance(df.columns, pd.MultiIndex):
        if sym in df.columns.get_level_values(0):
            return df[sym].dropna(how="all")
        if sym in df.columns.get_level_values(-1):
            return df.xs(sym, axis=1, level=-1).dropna(how="all")
    return df.dropna(how="all")


def week_bootstrap(vals, dienos, n=2000, seed=37):
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


def sesiju_eilutes(d, sym):
    """Is valandiniu baru istraukia kiekvienos dienos bukle kas valanda."""
    d = d.copy()
    d["_d"] = [i.date() for i in d.index]
    ses = [(dd, g) for dd, g in d.groupby("_d") if len(g) >= 6]
    eil = []
    for i in range(1, len(ses) - 1):
        dd, g = ses[i]
        _, pr = ses[i - 1]
        _, kt = ses[i + 1]
        try:
            pr_uzd = float(pr["Close"].iloc[-1])
            uzd = float(g["Close"].iloc[-1])
            kt_atid = float(kt["Open"].iloc[0])
            kt_uzd = float(kt["Close"].iloc[-1])
            if min(pr_uzd, uzd, kt_atid) <= 0:
                continue
            apyv_d = float(g["Volume"].sum()) if "Volume" in g else np.nan
            rec = dict(tag=sym, data=pd.Timestamp(dd))
            # Kas valanda: kaina, kilimas nuo vakar, VWAP, vieta diapazone
            for k in range(2, min(len(g), 9) + 1):
                dalis = g.iloc[:k]
                c_ = float(dalis["Close"].iloc[-1])
                hi_, lo_ = float(dalis["High"].max()), float(dalis["Low"].min())
                tp = (dalis["High"] + dalis["Low"] + dalis["Close"]) / 3
                vv = dalis["Volume"].replace(0, np.nan)
                vwap = float((tp * vv).sum() / vv.sum()) if vv.sum() > 0 else np.nan
                rec[f"kilo_{k}"] = (c_ / pr_uzd - 1) * 100
                rec[f"vwap_{k}"] = ((c_ / vwap - 1) * 100) if vwap == vwap else np.nan
                rec[f"ibs_{k}"] = ((c_ - lo_) / (hi_ - lo_)) if hi_ > lo_ else np.nan
                rec[f"px_{k}"] = c_
                # Isejimai nuo sio taško
                rec[f"iki_uzd_{k}"] = (uzd / c_ - 1) * 100
                rec[f"iki_ryt_{k}"] = (kt_atid / c_ - 1) * 100
                rec[f"iki_kt_uzd_{k}"] = (kt_uzd / c_ - 1) * 100
            rec["dienos_pok"] = (uzd / pr_uzd - 1) * 100
            rec["apyv_d"] = apyv_d
            eil.append(rec)
        except Exception:
            continue
    return eil


def main():
    ap = argparse.ArgumentParser(description="Dienos kilimo backtestas")
    ap.add_argument("--rinka", default="eu", choices=["eu", "us"])
    ap.add_argument("--pozicija", type=float, default=18000.0)
    a = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Paleisk: pip install yfinance pandas numpy")

    uni = U.UNIVERSAS_US if a.rinka == "us" else U.UNIVERSAS
    idx_sym = "SPY" if a.rinka == "us" else U.INDEKSAS
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
            eil += sesiju_eilutes(d, s)
        except Exception:
            be_d += 1
    if not eil:
        sys.exit("Nepavyko surinkti duomenu.")
    df = pd.DataFrame(eil)
    print(f"  akciju: {df['tag'].nunique()}, dienu-akciju: {len(df):,}, "
          f"be duomenu: {be_d}")

    # Rinkos rezimas — ar krentanti rinka
    try:
        ir_ = flatten(yf.download(idx_sym, period="720d", interval="60m",
                                  progress=False, auto_adjust=False), idx_sym)
        ir_ = ir_.dropna(subset=["Close"])
        ic = ir_["Close"].resample("1D").last().dropna()
        rez = pd.Series((ic / ic.shift(5) - 1 > 0).values,
                        index=[i.date() for i in ic.index])
        df["rinka_kyla"] = df["data"].dt.date.map(rez)
        print(f"  rinka kyla: {df['rinka_kyla'].mean() * 100:.0f}% dienu")
    except Exception as e:
        df["rinka_kyla"] = True
        print(f"  rezimo nepavyko: {str(e)[:50]}")

    # Demeanavimas kiekvienam isejimui
    for k in range(2, 10):
        for c in (f"iki_uzd_{k}", f"iki_ryt_{k}", f"iki_kt_uzd_{k}"):
            if c in df:
                df[c + "_dm"] = df[c] - df.groupby("data")[c].transform("mean")

    riba = df["data"].quantile(0.5)
    p1, p2 = df["data"] <= riba, df["data"] > riba
    print(f"  padalijimas: 1-a puse iki {riba:%Y-%m-%d}\n")

    # ---------- A. IJEJIMO VALANDA IR ISEJIMAS ----------
    print("=" * 100)
    print("A. PIRKTI KYLANCIA — kada ijeiti ir kada iseiti")
    print("Salyga: akcija kyla bent 0.5% nuo vakar uzdarymo ir yra virs VWAP.")
    print("=" * 100)
    print(f"{'IJEJIMAS':<12} {'ISEJIMAS':<18} {'N':>7} {'VID.':>9} {'DEMEAN.':>10} "
          f"{'95% INTERVALAS':>22} {'EUR':>9}")
    print("-" * 100)
    rez_a = []
    for k, val in [(3, "~11:00"), (4, "~12:00"), (6, "~14:00"), (8, "~16:00")]:
        if f"kilo_{k}" not in df:
            continue
        kauke = (df[f"kilo_{k}"] >= 0.5) & (df[f"vwap_{k}"] > 0)
        sub0 = df[kauke]
        if len(sub0) < MIN_IVYKIU:
            continue
        for c, lab in [(f"iki_uzd_{k}", "ta pati diena"),
                       (f"iki_ryt_{k}", "kito ryto atidarymu"),
                       (f"iki_kt_uzd_{k}", "kitos dienos pabaigoje")]:
            r, n = ivertink(sub0, c + "_dm")
            if r is None:
                print(f"{val:<12} {lab:<18} {n:>7}   per maza imtis")
                continue
            vid = sub0[c].mean()
            eur = (vid - SANAUDOS_PCT) / 100 * a.pozicija
            ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
            print(f"{val:<12} {lab:<18} {r['n']:>7} {vid:>+8.3f}% "
                  f"{r['mean']:>+9.3f}% {ci:>22} {eur:>+8.2f}€")
            rez_a.append((f"{val} -> {lab}", kauke, c, r))
        print()

    # ---------- B. AR STIPRESNIS KILIMAS GERIAU ----------
    print("=" * 100)
    print("B. AR STIPRESNIS KILIMAS GERIAU (ijejimas ~12:00, isejimas ta diena)")
    print("=" * 100)
    k = 4
    if f"kilo_{k}" in df:
        print(f"{'KILIMAS IKI 12:00':<24} {'N':>7} {'DEMEAN.':>10} "
              f"{'95% INTERVALAS':>22} {'EUR':>9}")
        print("-" * 100)
        for lo, hi, lab in [(-99, 0, "krenta"), (0, 0.5, "0-0.5%"),
                            (0.5, 1.0, "0.5-1%"), (1.0, 2.0, "1-2%"),
                            (2.0, 4.0, "2-4%"), (4.0, 99, "virs 4%")]:
            sub = df[(df[f"kilo_{k}"] >= lo) & (df[f"kilo_{k}"] < hi)]
            r, n = ivertink(sub, f"iki_uzd_{k}_dm")
            if r is None:
                print(f"{lab:<24} {n:>7}   per maza imtis")
                continue
            eur = (sub[f"iki_uzd_{k}"].mean() - SANAUDOS_PCT) / 100 * a.pozicija
            ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
            print(f"{lab:<24} {r['n']:>7} {r['mean']:>+9.3f}% {ci:>22} {eur:>+8.2f}€")

    # ---------- C. VWAP IR APYVARTA ----------
    print("\n" + "=" * 100)
    print("C. AR VWAP IR APYVARTA PRIDEDA (kylancioms, ijejimas ~12:00)")
    print("=" * 100)
    bazine, _ = ivertink(df[(df[f"kilo_{k}"] >= 0.5)], f"iki_uzd_{k}_dm")
    if bazine:
        print(f"  Vien kilimas >0.5%: {bazine['mean']:+.3f}% (n={bazine['n']})")
        papildomos = [
            ("+ virs VWAP", df[f"vwap_{k}"] > 0),
            ("+ virs VWAP daugiau 0.3%", df[f"vwap_{k}"] > 0.3),
            ("+ IBS > 0.7 (laikosi virsuje)", df[f"ibs_{k}"] > 0.7),
            ("+ IBS > 0.85", df[f"ibs_{k}"] > 0.85),
        ]
        for lab, kauke in papildomos:
            sub = df[(df[f"kilo_{k}"] >= 0.5) & kauke]
            r, n = ivertink(sub, f"iki_uzd_{k}_dm")
            if r:
                print(f"    {lab:<32} {r['mean']:>+7.3f}%  "
                      f"skirtumas {r['mean'] - bazine['mean']:>+7.3f}  (n={r['n']})")
            else:
                print(f"    {lab:<32} per maza imtis ({n})")

    # ---------- D. KRENTANCIOJE RINKOJE ----------
    print("\n" + "=" * 100)
    print("D. AR VEIKIA KRENTANCIOJE RINKOJE — butent tada, kai to reikia")
    print("=" * 100)
    for kyla, lab in [(True, "rinka kyla"), (False, "rinka krenta")]:
        sub = df[(df[f"kilo_{k}"] >= 0.5) & (df[f"vwap_{k}"] > 0)
                 & (df["rinka_kyla"] == kyla)]
        for c, cl in [(f"iki_uzd_{k}", "ta diena"), (f"iki_ryt_{k}", "kita ryta")]:
            r, n = ivertink(sub, c + "_dm")
            if r is None:
                print(f"  {lab:<16} {cl:<12} {n:>7}   per maza imtis")
                continue
            eur = (sub[c].mean() - SANAUDOS_PCT) / 100 * a.pozicija
            print(f"  {lab:<16} {cl:<12} {r['n']:>7} {r['mean']:>+8.3f}% "
                  f"[{r['lo']:+.3f}..{r['hi']:+.3f}] {eur:>+7.2f}€")

    # ---------- E. PALYGINIMAS SU DIPU ----------
    print("\n" + "=" * 100)
    print("E. KYLANCIOS PRIES DIPO KANDIDATUS (tas pats laikotarpis)")
    print("=" * 100)
    dip = df[(df[f"kilo_{k}"] <= -0.5) & (df[f"ibs_{k}"] <= 0.3)]
    kyl = df[(df[f"kilo_{k}"] >= 0.5) & (df[f"vwap_{k}"] > 0)]
    print(f"{'RINKINYS':<24} {'N':>7} {'TA DIENA':>11} {'KITA RYTA':>12} {'EUR':>9}")
    print("-" * 100)
    for lab, sub in [("Kylancios (12:00)", kyl), ("Dipo kandidatai", dip)]:
        r1, _ = ivertink(sub, f"iki_uzd_{k}_dm")
        r2, _ = ivertink(sub, f"iki_ryt_{k}_dm")
        v1 = f"{r1['mean']:+.3f}%" if r1 else "—"
        v2 = f"{r2['mean']:+.3f}%" if r2 else "—"
        eur = ((sub[f"iki_ryt_{k}"].mean() - SANAUDOS_PCT) / 100 * a.pozicija
               if len(sub) else 0)
        print(f"{lab:<24} {len(sub):>7} {v1:>11} {v2:>12} {eur:>+8.2f}€")

    # ---------- PATVIRTINIMAS ----------
    print("\n" + "=" * 100)
    print("PATVIRTINIMAS 2-OJE PUSEJE (nematyta)")
    print("=" * 100)
    print(f"{'VARIANTAS':<34} {'N':>7} {'DEMEAN.':>10} {'95% INTERVALAS':>22} "
          f"{'VERDIKTAS':>14}")
    print("-" * 100)
    for lab, kauke, c, _ in rez_a:
        sub = df[kauke & p2]
        r, n = ivertink(sub, c + "_dm")
        if r is None:
            print(f"{lab:<34} {n:>7}   per maza imtis")
            continue
        v = ("PATVIRTINTA" if r["lo"] > 0 else
             ("PRIESINGA" if r["hi"] < 0 else "nepatvirtinta"))
        ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
        print(f"{lab:<34} {r['n']:>7} {r['mean']:>+9.3f}% {ci:>22} {v:>14}")

    print(f"\n  Sanaudos {SANAUDOS_PCT}% jau atimtos EUR stulpelyje.")
    print("  Kad butu verta, DEMEAN. turi virsyti sanaudas ir intervalas")
    print("  neturi kirsti nulio 2-oje puseje.")
    print("\nAPRIBOJIMAI: valandiniai barai, ne 5 min. — ijejimas tikslesnis realiai; "
          "\n720 d. imtis; ijejimas baro uzdarymu, realiai vykdytum kiek veliau; "
          "\nislikimo salismas.")


if __name__ == "__main__":
    main()
