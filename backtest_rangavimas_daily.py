#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1 FAZE — rangavimo backtestas dienos barais.

Klausimas: ar kasdien is viso Europos universo isrinkus 5 akcijas pagal A ir B
taisykles gaunamas rezultatas, geresnis uz tos dienos vidurki?

Paleidimas:
    pip install yfinance pandas numpy
    python backtest_rangavimas_daily.py --rinka eu
    python backtest_rangavimas_daily.py --rinka us     # nepriklausomas patikrinimas

METODIKA
- Taisykles importuojamos is rangavimas.py — TO PACIO failo, kuri naudos skeneris.
  Todel skeneris negales tikrinti kitokiu salygu nei tos, kurios cia ismatuotos.
- Ijejimas kitos dienos atidarymu, grazos demeanuojamos pagal diena.
- Iverciai is dienu vidurkiu, intervalai — savaiciu bloku bootstrap'u.
- Dvi laiko puses: atranka 1-oje, patvirtinimas 2-oje. BH pataisa.
- Rangai ir kvantiliai skaiciuojami TIK is 1-os puses.

SEKMES KRITERIJAI (uzrasyti pries paleidziant)
1. Dip'as: TOP5_DIP demeanuota r_1d arba r_3d po sanaudu >= +0.20%,
   intervalo apacia > 0 abiejose pusese, Europa ir JAV.
2. Trendo kontekstas naudingas: A2 > A1 > A4 tvarka abiejose pusese ir rinkose.
3. Prasidedantis trendas: TOP5_TREND r_10d arba r_20d po sanaudu >= +0.50%,
   apacia > 0 abiejose pusese ir rinkose.
"""

import argparse
import sys

import numpy as np
import pandas as pd

import rangavimas as R
import universas as U

SANAUDOS = 0.07
MIN_APYVARTA_EUR = 30e6
MIN_IVYKIU = 100
MIN_DIENU = 40


# ----------------------------- STATISTIKA -----------------------------

def week_block_bootstrap(vals, dienos, n=3000, seed=42):
    if len(vals) < 12:
        return None
    rng = np.random.default_rng(seed)
    sav = pd.Series([pd.Timestamp(d).to_period("W") for d in dienos])
    grupes = [np.asarray(vals)[(sav == w).to_numpy()] for w in sav.unique()]
    grupes = [g for g in grupes if len(g)]
    if len(grupes) < 10:
        return None
    boot = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(grupes), len(grupes))
        boot[i] = np.concatenate([grupes[j] for j in pick]).mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_pos = float((boot > 0).mean())
    return dict(mean=float(np.mean(vals)), lo=float(lo), hi=float(hi),
                p_two=float(2 * min(p_pos, 1 - p_pos)), dienos=len(vals))


def ivertink(sub, col):
    s = sub.dropna(subset=[col])
    if len(s) < MIN_IVYKIU:
        return None, len(s)
    pagal_diena = s.groupby("data")[col].mean()
    if len(pagal_diena) < MIN_DIENU:
        return None, len(s)
    r = week_block_bootstrap(pagal_diena.to_numpy(), pagal_diena.index)
    if r:
        r["n"] = len(s)
    return r, len(s)


def benjamini_hochberg(pv, alpha=0.05):
    pv = np.asarray(pv)
    m = len(pv)
    order = np.argsort(pv)
    ok = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order, start=1):
        if pv[idx] <= alpha * rank / m:
            ok[order[:rank]] = True
    return ok


# ----------------------------- DUOMENYS -----------------------------

def flatten(df, sym):
    if isinstance(df.columns, pd.MultiIndex):
        if sym in df.columns.get_level_values(0):
            return df[sym].dropna(how="all")
        if sym in df.columns.get_level_values(-1):
            return df.xs(sym, axis=1, level=-1).dropna(how="all")
    return df.dropna(how="all")


def surink(yf, metai, uni, sekt_map):
    visi = U.visi_tikeriai(uni)
    print(f"Siunciama {len(visi)} akciju dienos istorija ({metai}m)…")
    raw = yf.download(visi, period=f"{metai}y", interval="1d", group_by="ticker",
                      progress=False, auto_adjust=False, threads=True)

    eil, be_duomenu, nelikvidzios = [], [], []
    for sym in visi:
        try:
            d = flatten(raw, sym).dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            if len(d) < 300:
                be_duomenu.append(sym)
                continue
            apyv = float((d["Close"] * d["Volume"]).tail(60).median())
            if apyv < MIN_APYVARTA_EUR:
                nelikvidzios.append((sym, apyv / 1e6))
                continue
            k = R.konteksto_stulpeliai(d)
            k["tag"] = sym
            k["sekt"] = sekt_map.get(sym, "kita")
            k["data"] = d.index
            eil.append(k.reset_index(drop=True))
        except Exception:
            be_duomenu.append(sym)

    if not eil:
        sys.exit("Nepavyko surinkti duomenu.")
    df = pd.concat(eil, ignore_index=True)
    print(f"  duomenu gauta: {df['tag'].nunique()}; be duomenu: {len(be_duomenu)}; "
          f"atmesta pagal likviduma (<{MIN_APYVARTA_EUR/1e6:.0f}M EUR): {len(nelikvidzios)}")
    if be_duomenu:
        print(f"  be duomenu: {', '.join(be_duomenu[:15])}"
              + (" …" if len(be_duomenu) > 15 else ""))
    return df


def prideti_konteksta(yf, df, etf_map, indeksas, metai):
    """Sektoriaus ETF trendas ir rinkos rezimas."""
    etfs = sorted(set(etf_map.values()) | {indeksas})
    df["_d"] = pd.to_datetime(df["data"]).dt.date
    df["sekt_trend"] = False
    df["rezimas"] = False
    df["sekt_ret"] = np.nan
    try:
        raw = yf.download(etfs, period=f"{metai}y", interval="1d", group_by="ticker",
                          progress=False, auto_adjust=False, threads=True)
        veikia = []
        tab = {}
        for e in etfs:
            try:
                d = flatten(raw, e).dropna(subset=["Close"])
                if len(d) < 200:
                    continue
                c = d["Close"]
                t = pd.DataFrame(index=pd.Index([i.date() for i in d.index]))
                t["ret"] = (c / c.shift(1) - 1).values * 100
                t["trend"] = ((c > c.rolling(50).mean())
                              & (c / c.shift(20) - 1 > 0)).values
                tab[e] = t
                veikia.append(e)
            except Exception:
                pass
        print(f"  ETF veikia: {', '.join(veikia) or 'nei vienas'}")
        for sek, e in etf_map.items():
            if e in tab:
                m = df["sekt"] == sek
                df.loc[m, "sekt_trend"] = df.loc[m, "_d"].map(tab[e]["trend"]).fillna(False)
                df.loc[m, "sekt_ret"] = df.loc[m, "_d"].map(tab[e]["ret"])
        if indeksas in tab:
            df["rezimas"] = df["_d"].map(tab[indeksas]["trend"]).fillna(False)
    except Exception as e:
        print(f"  ETF nepavyko: {str(e)[:60]}")
    return df


def paruosk(df):
    """Rangai, trendo busena, z20 sektoriaus atzvilgiu, grazos."""
    df["mom_rangas"] = df.groupby("data")["mom_6m"].rank(pct=True)
    df["trendas"] = [
        R.trendo_busena(m, v50, v200, st)
        for m, v50, v200, st in zip(df["mom_rangas"], df["virs_sma50"],
                                    df["virs_sma200"], df["sekt_trend"])
    ]
    # z20 sektoriaus atzvilgiu: kiek akcija nukritusi labiau nei jos sektorius
    df["z20_sekt"] = df["z20"] - df.groupby(["data", "sekt"])["z20"].transform("median")

    df = R.zymek_A(df)
    df = R.zymek_B(df)

    for n in (1, 3, 5, 10, 20):
        df[f"r_{n}d"] = (df[f"C{n}"] / df["O1"] - 1) * 100
        df[f"r_{n}d_dm"] = df[f"r_{n}d"] - df.groupby("data")[f"r_{n}d"].transform("mean")
    return df


# ----------------------------- ATASKAITA -----------------------------

def variantu_lentele(df, antraste, kauke=None):
    print("\n" + "=" * 104)
    print(antraste)
    print("=" * 104)
    print(f"{'VARIANTAS':<34} {'LANGAS':<8} {'N':>7} {'DIENU':>6} {'DEMEAN.':>10} "
          f"{'95% INTERVALAS':>22}")
    print("-" * 104)
    rez = []
    variantai = [("A1", "A1 dip'as bet kokiame trende"),
                 ("A2", "A2 dip'as STIPRIAME trende"),
                 ("A3", "A3 dip'as neutraliame"),
                 ("A4", "A4 dip'as SILPNAME (kontrole)"),
                 ("B1", "B1 trendo pradzia + sektorius"),
                 ("B2", "B2 trendo pradzia (kontrole)")]
    for col, lab in variantai:
        if col not in df:
            continue
        sub = df[df[col]] if kauke is None else df[df[col] & kauke]
        langai = ["r_1d_dm", "r_3d_dm"] if col.startswith("A") else ["r_10d_dm", "r_20d_dm"]
        for lang in langai:
            r, n = ivertink(sub, lang)
            if r is None:
                print(f"{lab:<34} {lang:<8} {n:>7}   per maza imtis")
                continue
            ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
            print(f"{lab:<34} {lang:<8} {r['n']:>7} {r['dienos']:>6} "
                  f"{r['mean']:>+9.3f}% {ci:>22}")
            rez.append((col, lab, lang, r))
    return rez


def top5_lentele(df, kauke=None, antraste="TOP-5 RANGAVIMAS"):
    """Pagrindinis matas: tai, ka modulis realiai darys kiekviena diena."""
    print("\n" + "=" * 104)
    print(antraste + " — tai, ka skeneris rodys kasdien")
    print("=" * 104)
    d = df if kauke is None else df[kauke]
    eilutes = []
    for diena, g in d.groupby("data"):
        sel = R.sudek_top5(g)
        if len(sel):
            eilutes.append(sel.assign(data=diena))
    if not eilutes:
        print("  nera nei vienos dienos su kandidatais")
        return []
    top = pd.concat(eilutes, ignore_index=True)

    print(f"{'RINKINYS':<34} {'LANGAS':<8} {'N':>7} {'DIENU':>6} {'DEMEAN.':>10} "
          f"{'95% INTERVALAS':>22}")
    print("-" * 104)
    rez = []
    grupes = [("visi", top, ["r_1d_dm", "r_3d_dm"]),
              ("DIP_TRENDE", top[top["setup"] == "DIP_TRENDE"], ["r_1d_dm", "r_3d_dm"]),
              ("DIP", top[top["setup"] == "DIP"], ["r_1d_dm", "r_3d_dm"]),
              ("TRENDO_PRADZIA", top[top["setup"] == "TRENDO_PRADZIA"],
               ["r_10d_dm", "r_20d_dm"])]
    for lab, sub, langai in grupes:
        for lang in langai:
            r, n = ivertink(sub, lang)
            if r is None:
                print(f"TOP5 {lab:<29} {lang:<8} {n:>7}   per maza imtis")
                continue
            ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
            print(f"TOP5 {lab:<29} {lang:<8} {r['n']:>7} {r['dienos']:>6} "
                  f"{r['mean']:>+9.3f}% {ci:>22}")
            rez.append((f"TOP5_{lab}", f"TOP5 {lab}", lang, r))

    # Kontroles
    print("\n  KONTROLES:")
    a1 = df if kauke is None else df[kauke]
    kontrol = [("atsitiktiniai 5 is A1", a1[a1["A1"]].sample(
        min(len(a1[a1["A1"]]), 20000), random_state=1) if len(a1[a1["A1"]]) else a1.head(0)),
        ("apatiniai 5 pagal z20_sekt (turi buti blogiau)",
         a1[a1["A1"]].sort_values("z20_sekt", ascending=False).head(
             max(1, len(a1[a1["A1"]]) // 20)))]
    for lab, sub in kontrol:
        r, n = ivertink(sub, "r_1d_dm")
        if r:
            print(f"    {lab:<46} {r['mean']:>+8.3f}% "
                  f"[{r['lo']:+.3f}..{r['hi']:+.3f}] n={r['n']}")
        else:
            print(f"    {lab:<46} per maza imtis (n={n})")
    return rez


def metu_lentele(df):
    print("\n" + "=" * 104)
    print("PAGAL METUS (demeanuota r_1d; A variantams)")
    print("=" * 104)
    df["_m"] = pd.to_datetime(df["data"]).dt.year
    print(f"{'METAI':<8}{'A1':>16}{'A2':>16}{'A3':>16}{'A4':>16}")
    print("-" * 72)
    for m, g in df.groupby("_m"):
        eil = f"{m:<8}"
        for col in ("A1", "A2", "A3", "A4"):
            s = g[g[col]]
            eil += (f"{s['r_1d_dm'].mean():>+11.3f}%({len(s):>3})" if len(s) >= 20
                    else f"{'-':>16}")
        print(eil)


def main():
    ap = argparse.ArgumentParser(description="1 faze: rangavimo backtestas")
    ap.add_argument("--metai", type=int, default=10)
    ap.add_argument("--rinka", default="eu", choices=["eu", "us"])
    ap.add_argument("--sanaudos", type=float, default=SANAUDOS)
    args = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Paleisk: pip install yfinance pandas numpy")

    if args.rinka == "us":
        uni, emap, idx = U.UNIVERSAS_US, U.US_ETF, U.US_INDEKSAS
        print("JAV UNIVERSAS — nepriklausomas patikrinimas\n")
    else:
        uni, emap, idx = U.UNIVERSAS, U.SEKTORIU_ETF, U.INDEKSAS
        print("EUROPOS UNIVERSAS (tik EUR birzos)\n")

    sekt_map = U.sektoriai(uni)
    df = surink(yf, args.metai, uni, sekt_map)
    df = prideti_konteksta(yf, df, emap, idx, args.metai)
    df = paruosk(df)
    df = df.dropna(subset=["r_1d_dm"])

    print(f"\nLaikotarpis: {pd.to_datetime(df['data']).min():%Y-%m-%d} .. "
          f"{pd.to_datetime(df['data']).max():%Y-%m-%d}")
    print(f"Eiluciu: {len(df):,} | akciju: {df['tag'].nunique()} | "
          f"sektoriu: {df['sekt'].nunique()}")
    print(f"Rezimas teigiamas: {df['rezimas'].mean()*100:.0f}% dienu")
    print("Ivykiu pagal varianta:")
    for col in ("A1", "A2", "A3", "A4", "B1", "B2"):
        if col in df:
            print(f"  {col}: {int(df[col].sum()):>7}")
    print("\nDEMESIO: universas — DABARTINIAI nariai, todel yra islikimo salismas.")

    riba = pd.to_datetime(df["data"]).quantile(0.5)
    p1 = pd.to_datetime(df["data"]) <= riba
    p2 = ~p1
    print(f"Padalijimas: 1-a puse iki {riba:%Y-%m-%d}")

    variantu_lentele(df, "VISA IMTIS — variantai")
    top5_lentele(df, antraste="TOP-5 RANGAVIMAS (visa imtis)")

    r1 = variantu_lentele(df, "1 ETAPAS — PAIESKA (1-oji puse)", p1)
    r1 += top5_lentele(df, p1, "TOP-5 (1-oji puse)")

    if r1:
        pv = np.array([r["p_two"] for *_, r in r1])
        ok = benjamini_hochberg(pv)
        kand = [(c, lab, lang) for (c, lab, lang, _), o in zip(r1, ok) if o]
        print(f"\n1 etape islaike (su BH pataisa): {len(kand)} is {len(r1)}")

        if kand:
            print("\n" + "=" * 104)
            print("2 ETAPAS — PATVIRTINIMAS (2-oji puse, nematyta)")
            print("=" * 104)
            print(f"{'VARIANTAS':<34} {'LANGAS':<8} {'N':>7} {'DEMEAN.':>10} "
                  f"{'95% INTERVALAS':>22} {'VERDIKTAS':>14}")
            print("-" * 104)
            patv = []
            for col, lab, lang in kand:
                if col.startswith("TOP5"):
                    eil = []
                    for diena, g in df[p2].groupby("data"):
                        sel = R.sudek_top5(g)
                        if len(sel):
                            eil.append(sel.assign(data=diena))
                    if not eil:
                        continue
                    t = pd.concat(eil, ignore_index=True)
                    tipas = col.replace("TOP5_", "")
                    sub = t if tipas == "visi" else t[t["setup"] == tipas]
                else:
                    sub = df[df[col] & p2]
                r, n = ivertink(sub, lang)
                if r is None:
                    print(f"{lab:<34} {lang:<8} {n:>7}   per maza imtis")
                    continue
                v = ("PATVIRTINTA" if r["lo"] > 0 else
                     ("PRIESINGA" if r["hi"] < 0 else "nepatvirtinta"))
                ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
                print(f"{lab:<34} {lang:<8} {r['n']:>7} {r['mean']:>+9.3f}% "
                      f"{ci:>22} {v:>14}")
                if r["lo"] > 0:
                    patv.append((lab, lang, r))

            print("\n" + "=" * 104)
            print(f"GALUTINIS VERTINIMAS (atemus {args.sanaudos}% sanaudu)")
            print("Kriterijai: dip'ui >= +0.20%, trendo pradziai >= +0.50%")
            print("=" * 104)
            for lab, lang, r in patv:
                neto = r["mean"] - args.sanaudos
                riba_k = 0.50 if lang in ("r_10d_dm", "r_20d_dm") else 0.20
                print(f"{lab:<34} {lang:<8} bruto {r['mean']:>+7.3f}%  "
                      f"neto {neto:>+7.3f}%  "
                      f"{'TENKINA' if neto >= riba_k else 'netenkina'}")
            if not patv:
                print("  Ne vienas variantas nepasitvirtino nematytoje puseje.")

    # Trendo konteksto tvarka
    print("\n" + "=" * 104)
    print("AR TRENDO KONTEKSTAS NAUDINGAS? (laukiama A2 > A1 > A4)")
    print("=" * 104)
    for pav, kauke in [("1-oji puse", p1), ("2-oji puse", p2)]:
        eil = f"  {pav:<12}"
        for col in ("A2", "A1", "A4"):
            s = df[df[col] & kauke]
            eil += f"  {col}={s['r_1d_dm'].mean():+.3f}%" if len(s) >= 100 else f"  {col}=n/a"
        tvarka = all(
            df[df[a] & kauke]["r_1d_dm"].mean() >= df[df[b] & kauke]["r_1d_dm"].mean()
            for a, b in (("A2", "A1"), ("A1", "A4"))
            if len(df[df[a] & kauke]) >= 100 and len(df[df[b] & kauke]) >= 100)
        print(eil + f"   tvarka islaikyta: {'TAIP' if tvarka else 'ne'}")

    metu_lentele(df)
    print("\nAPRIBOJIMAI: islikimo salismas; grazos be stop'u ir tikslu; "
          "\nsanaudos atimtos tik galutineje lenteleje; "
          "\ntaisykles importuotos is rangavimas.py — skeneris naudos tas pacias.")


if __name__ == "__main__":
    main()
