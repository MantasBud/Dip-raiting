#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tesinio hipotezes backtestas — dienos barai, platus universas.

Klausimas: ar stiprus vienos dienos kilimas testiasi kita diena ir per 3-5 dienas?

Paleidimas:
    pip install yfinance pandas numpy
    python backtest_tesinys_daily.py
    python backtest_tesinys_daily.py --metai 10 --sanaudos 0.07

KUO SKIRIASI NUO backtest_intraday.py
Tai savarankiskas failas. Jis NEIMPORTUOJA dip_reitingas.py, nenaudoja score_stock,
jo stop'u, tikslu ar kriteriju — tai kita strategija ir jai reikia savo iseijimu.
Set up'as apibreztas vien dienos barais, todel imtis gali buti 10 metu ir 200+
akciju vietoj 2 metu ir 25 akciju. Ankstesnis matavimas dave n=410, d=70 —
per mazai; sis turetu duoti tukstancius ivykiu.

METODIKA
- Ivykis fiksuojamas D dienos uzdarymo metu is UZBAIGTU baru.
- Ijejimas kitos dienos atidarymo kaina.
- Pagrindinis matas — grynos grazos be stop'u ir tikslu.
- Kiekviena graza DEMEANUOJAMA pagal ta pacia diena (viso universo vidurkis),
  todel "geros dienos" efektas atmetamas.
- Iverciai skaiciuojami is dienu vidurkiu, intervalai — savaiciu bloku bootstrap'u.
- Dvi kalendorines puses: variantai atrenkami 1-oje, patvirtinami 2-oje.
- BH pataisa per visas eilutes; sanaudos atimamos galutineje lenteleje.

APRIBOJIMAI
- Universas sudarytas is DABARTINIU indekso nariu — yra islikimo salismas
  (survivorship bias), kuris paprastai DIDINA teigiamus rezultatus.
- Slenksciai uzrasyti pries paleidziant ir nederinami pagal rezultatus.
"""

import argparse
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ----------------------------- SLENKSCIAI -----------------------------
# Uzrasyti pries matuojant. Nederinami.

KILIMAS_STIPRUS = 3.0     # % dienos pokytis
KILIMAS_SVELNUS = 2.0
APYVARTA_KARTU = 2.0      # kartai pries 20 d. mediana
IBS_VIRSUJE = 0.80
VIRSUNES_LANGAS = 20      # dienu

MIN_IVYKIU = 100          # maziau — spausdinam "per maza imtis"
MIN_DIENU = 40
MIN_APYVARTA_EUR = 5e6    # universo filtras

# ----------------------------- UNIVERSAS -----------------------------
# STOXX Europe 600 nariai, sugrupuoti pagal sektoriu. Valiutos konvertuojamos
# tik APYVARTOS filtrui, ne grazoms (grazos yra procentai, valiuta nesvarbi).

FX = {"EUR": 1.0, "GBp": 0.0117, "GBP": 1.17, "CHF": 1.06, "SEK": 0.088,
      "DKK": 0.134, "NOK": 0.086}
GALUNIU_VALIUTA = {"DE": "EUR", "PA": "EUR", "AS": "EUR", "MI": "EUR", "MC": "EUR",
                   "BR": "EUR", "HE": "EUR", "LS": "EUR", "VI": "EUR", "IR": "EUR",
                   "SW": "CHF", "L": "GBp", "ST": "SEK", "CO": "DKK", "OL": "NOK"}

UNIVERSAS = {
    "Technologijos": [
        "ASML.AS", "ASM.AS", "BESI.AS", "IFX.DE", "STM.PA", "AIXA.DE", "SOI.PA",
        "SAP.DE", "DSY.PA", "CAP.PA", "TEP.PA", "ADYEN.AS", "PRX.AS", "NOKIA.HE",
        "ERIC-B.ST", "LOGN.SW", "TEMN.SW", "AMS.SW", "SW.PA", "ATO.PA", "ALTR.LS",
    ],
    "Pramone": [
        "SIE.DE", "SU.PA", "ABBN.SW", "KGX.DE", "LR.PA", "PRY.MI", "NEX.PA",
        "ENR.DE", "VIE.PA", "HOLN.SW", "BNR.DE", "GEBN.SW", "ATCO-A.ST",
        "SAND.ST", "VOLV-B.ST", "ALFA.ST", "SKF-B.ST", "EPI-A.ST", "WEIR.L", "RR.L",
    ],
    "Gynyba ir aviacija": [
        "RHM.DE", "LDO.MI", "HO.PA", "AIR.PA", "SAF.PA", "BA.L", "MTX.DE", "AM.PA",
    ],
    "Automobiliai": [
        "MBG.DE", "BMW.DE", "VOW3.DE", "P911.DE", "STLAM.MI", "CON.DE", "RNO.PA",
        "PAH3.DE", "FORVIA.PA", "MICP.PA", "PIRC.MI",
    ],
    "Bankai ir finansai": [
        "BNP.PA", "ACA.PA", "GLE.PA", "DBK.DE", "CBK.DE", "UCG.MI", "ISP.MI",
        "SAN.MC", "BBVA.MC", "CABK.MC", "INGA.AS", "ABN.AS", "KBC.BR", "UBSG.SW",
        "NDA-SE.ST", "SEB-A.ST", "SWED-A.ST", "DANSKE.CO", "HSBA.L", "BARC.L", "LLOY.L",
    ],
    "Draudimas": [
        "ALV.DE", "CS.PA", "MUV2.DE", "HNR1.DE", "ZURN.SW", "G.MI", "NN.AS",
        "AGN.AS", "SREN.SW", "LGEN.L", "AV.L",
    ],
    "Energetika": [
        "TTE.PA", "ENI.MI", "SHEL.L", "BP.L", "EQNR.OL", "REP.MC", "OMV.VI",
        "RWE.DE", "EOAN.DE", "ENEL.MI", "IBE.MC", "ENGI.PA", "SSE.L", "ORSTED.CO",
    ],
    "Medziagos": [
        "BAS.DE", "1COV.DE", "AI.PA", "LIN.DE", "MT.AS", "GLEN.L", "AAL.L",
        "RIO.L", "ANTO.L", "BOL.ST", "UPM.HE", "STERV.HE", "SIKA.SW", "GIVN.SW",
    ],
    "Sveikata": [
        "SAN.PA", "BAYN.DE", "NOVN.SW", "ROG.SW", "NOVO-B.CO", "AZN.L", "GSK.L",
        "PHIA.AS", "FRE.DE", "FME.DE", "SRT3.DE", "EL.PA", "UCB.BR", "MRK.DE",
        "GN.CO", "COLO-B.CO", "DIM.PA",
    ],
    "Vartojimas ir prabanga": [
        "MC.PA", "KER.PA", "RMS.PA", "MONC.MI", "CFR.SW", "UHR.SW", "ADS.DE",
        "PUM.DE", "ITX.MC", "HM-B.ST", "BRBY.L", "NXT.L", "PNDORA.CO", "SW.L",
    ],
    "Maistas ir kasdienes prekes": [
        "NESN.SW", "ABI.BR", "HEIA.AS", "DGE.L", "ULVR.L", "BN.PA", "OR.PA",
        "AD.AS", "CA.PA", "TATE.L", "CARL-B.CO", "ORK.OL",
    ],
    "Telekomai ir ziniasklaida": [
        "DTE.DE", "ORA.PA", "TEF.MC", "TIT.MI", "KPN.AS", "VOD.L", "BT-A.L",
        "PUB.PA", "WPP.L", "RELX.L", "TEL2-B.ST",
    ],
    "Keliones ir laisvalaikis": [
        "LHA.DE", "AF.PA", "IAG.L", "RYA.IR", "ACS.MC", "AC.PA", "FLTR.L",
        "EVD.DE", "TUI1.DE", "CPG.L",
    ],
    "Nekilnojamas turtas ir statyba": [
        "VNA.DE", "URW.AS", "SGO.PA", "HEI.DE", "CRH.L", "SKA-B.ST", "BZU.MI",
    ],
    "Logistika ir mazmena": [
        "DHL.DE", "DSV.CO", "KNIN.SW", "MAERSK-B.CO", "AHT.L", "BOO.L",
    ],
}

SEKTORIU_ETF = {
    "Technologijos": "EXV3.DE", "Pramone": "EXH4.DE", "Gynyba ir aviacija": "EXH4.DE",
    "Automobiliai": "EXV5.DE", "Bankai ir finansai": "EXV1.DE", "Draudimas": "EXH5.DE",
    "Energetika": "EXH1.DE", "Medziagos": "EXV6.DE", "Sveikata": "EXV4.DE",
    "Vartojimas ir prabanga": "EXH6.DE", "Maistas ir kasdienes prekes": "EXH7.DE",
    "Telekomai ir ziniasklaida": "EXV2.DE", "Keliones ir laisvalaikis": "EXV9.DE",
    "Nekilnojamas turtas ir statyba": "EXV8.DE", "Logistika ir mazmena": "EXH4.DE",
}
INDEKSAS = "EXSA.DE"


# ----------------------------- STATISTIKA -----------------------------

def week_block_bootstrap(dienu_vid, dienos, n=3000, seed=42):
    """Intervalas perrenkant SAVAITES: tos pacios savaites dienos yra susijusios."""
    if len(dienu_vid) < 12:
        return None
    rng = np.random.default_rng(seed)
    sav = pd.Series([pd.Timestamp(d).to_period("W") for d in dienos])
    grupes = [np.asarray(dienu_vid)[(sav == w).to_numpy()] for w in sav.unique()]
    grupes = [g for g in grupes if len(g)]
    if len(grupes) < 10:
        return None
    boot = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(grupes), len(grupes))
        boot[i] = np.concatenate([grupes[j] for j in pick]).mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_pos = float((boot > 0).mean())
    return dict(mean=float(np.mean(dienu_vid)), lo=float(lo), hi=float(hi),
                p_two=float(2 * min(p_pos, 1 - p_pos)), dienos=len(dienu_vid))


def ivertink(sub, stulpelis):
    """Demeanuotos grazos ivertis: dienu vidurkiai + savaiciu bootstrap."""
    s = sub.dropna(subset=[stulpelis])
    if len(s) < MIN_IVYKIU:
        return None, len(s)
    pagal_diena = s.groupby("data")[stulpelis].mean()
    if len(pagal_diena) < MIN_DIENU:
        return None, len(s)
    r = week_block_bootstrap(pagal_diena.to_numpy(), pagal_diena.index)
    if r:
        r["n"] = len(s)
    return r, len(s)


def benjamini_hochberg(pvals, alpha=0.05):
    pv = np.asarray(pvals)
    m = len(pv)
    order = np.argsort(pv)
    passed = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order, start=1):
        if pv[idx] <= alpha * rank / m:
            passed[order[:rank]] = True
    return passed


# ----------------------------- DUOMENYS -----------------------------

def flatten(df, sym):
    if isinstance(df.columns, pd.MultiIndex):
        if sym in df.columns.get_level_values(0):
            return df[sym].dropna(how="all")
        if sym in df.columns.get_level_values(-1):
            return df.xs(sym, axis=1, level=-1).dropna(how="all")
    return df.dropna(how="all")


def valiutos_kursas(sym):
    g = sym.rsplit(".", 1)[-1].upper() if "." in sym else ""
    return FX.get(GALUNIU_VALIUTA.get(g, "EUR"), 1.0)


def surink(yf, metai):
    visi = [s for lst in UNIVERSAS.values() for s in lst]
    sekt = {s: k for k, lst in UNIVERSAS.items() for s in lst}
    print(f"Siunciama {len(visi)} akciju dienos istorija ({metai}m)…")
    raw = yf.download(visi, period=f"{metai}y", interval="1d", group_by="ticker",
                      progress=False, auto_adjust=False, threads=True)

    eil, praleista = [], []
    for sym in visi:
        try:
            d = flatten(raw, sym).dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            if len(d) < 300:
                praleista.append((sym, "per trumpa istorija"))
                continue
            apyv = float((d["Close"] * d["Volume"]).tail(60).median()) * valiutos_kursas(sym)
            if apyv < MIN_APYVARTA_EUR:
                praleista.append((sym, f"apyvarta {apyv/1e6:.1f}M"))
                continue

            o, h, l, c, v = (d["Open"], d["High"], d["Low"], d["Close"], d["Volume"])
            pc = c.shift(1)
            rng_d = (h - l).replace(0, np.nan)

            t = pd.DataFrame({
                "data": d.index, "tag": sym, "sekt": sekt[sym],
                "C": c.values, "O": o.values, "H": h.values, "L": l.values,
                "chg": ((c / pc - 1) * 100).values,
                "ibs": ((c - l) / rng_d).values,
                "vol_x": (v / v.rolling(21).median().shift(1)).values,
                "hi20": h.rolling(VIRSUNES_LANGAS).max().shift(1).values,
                "zalios": (c > pc).astype(int).values,
                # ijejimas kitos dienos atidarymu, grazos is jo
                "O1": o.shift(-1).values,
                "C1": c.shift(-1).values,
                "C3": c.shift(-3).values,
                "C5": c.shift(-5).values,
                "sma20": c.rolling(20).mean().values,
                "sma50": c.rolling(50).mean().values,
            })
            tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
            t["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean().values
            # kiek dienu is eiles zalia
            t["zalios_serija"] = (t["zalios"].groupby((t["zalios"] == 0).cumsum()).cumsum())
            # dienu nuo ankstesnes 20 d. virsunes
            t["nuo_virsunes"] = (h.groupby((h >= h.rolling(VIRSUNES_LANGAS).max()).cumsum())
                                 .cumcount().values)
            eil.append(t)
        except Exception as e:
            praleista.append((sym, str(e)[:30]))

    if not eil:
        sys.exit("Nepavyko surinkti duomenu.")
    df = pd.concat(eil, ignore_index=True)
    print(f"  duomenu gauta: {df['tag'].nunique()} akcijos; praleista: {len(praleista)}")
    if praleista:
        print(f"  praleistos: {', '.join(s for s, _ in praleista[:12])}"
              + (" …" if len(praleista) > 12 else ""))
    return df


def prideti_kontekstą(yf, df):
    """Sektoriaus ETF ir indekso dienos grazos."""
    etfs = sorted(set(SEKTORIU_ETF.values()) | {INDEKSAS})
    try:
        raw = yf.download(etfs, period="10y", interval="1d", group_by="ticker",
                          progress=False, auto_adjust=False, threads=True)
        veikia = []
        lentelės = {}
        for e in etfs:
            try:
                d = flatten(raw, e).dropna(subset=["Close"])
                if len(d) < 200:
                    continue
                s = pd.Series((d["Close"] / d["Close"].shift(1) - 1).values * 100,
                              index=pd.Index([i.date() for i in d.index]))
                lentelės[e] = s
                veikia.append(e)
            except Exception:
                pass
        print(f"  sektoriu ETF veikia: {', '.join(veikia) or 'nei vienas'}")
        df["_d"] = df["data"].dt.date
        df["sekt_ret"] = np.nan
        for sek, e in SEKTORIU_ETF.items():
            if e in lentelės:
                m = df["sekt"] == sek
                df.loc[m, "sekt_ret"] = df.loc[m, "_d"].map(lentelės[e])
        if INDEKSAS in lentelės:
            df["idx_ret"] = df["_d"].map(lentelės[INDEKSAS])
    except Exception as e:
        print(f"  ETF nepavyko: {str(e)[:50]}")
    return df


# ----------------------------- IVYKIAI -----------------------------

def pazymėk_ivykius(df):
    df["p_kilimas"] = df["chg"] >= KILIMAS_STIPRUS
    df["p_kilimas_sv"] = df["chg"] >= KILIMAS_SVELNUS
    df["p_apyvarta"] = df["vol_x"] >= APYVARTA_KARTU
    df["p_virsuje"] = df["ibs"] >= IBS_VIRSUJE
    df["p_virsune"] = df["C"] > df["hi20"]

    df["v1_pilnas"] = df["p_kilimas"] & df["p_apyvarta"] & df["p_virsuje"] & df["p_virsune"]
    df["v2_svelnus"] = df["p_kilimas_sv"] & df["p_apyvarta"] & df["p_virsuje"]
    df["v3_kilimas_virsune"] = df["p_kilimas"] & df["p_virsune"]
    df["v4_kilimas_apyvarta"] = df["p_kilimas"] & df["p_apyvarta"]
    return df


def grazos(df):
    """Grynos grazos nuo kitos dienos atidarymo + demeanavimas pagal diena."""
    df["r_gap"] = (df["O1"] / df["C"] - 1) * 100
    df["r_1d"] = (df["C1"] / df["O1"] - 1) * 100
    df["r_3d"] = (df["C3"] / df["O1"] - 1) * 100
    df["r_5d"] = (df["C5"] / df["O1"] - 1) * 100

    for k in ("r_1d", "r_3d", "r_5d"):
        df[f"{k}_dm"] = df[k] - df.groupby("data")[k].transform("mean")
        df[f"{k}_sekt"] = df[k] - df.groupby(["data", "sekt"])[k].transform("mean")
    return df


# ----------------------------- ISEJIMAI -----------------------------

def realistiski_isejimai(df):
    """Du realus isejimai variantams 1-2. Kai bare paliesti abu — stop."""
    stop = df["L"] * 0.995
    ijejimas = df["O1"]
    # A: stop po D dienos dugno, laikymas iki C_{D+3}
    stop_hit = df[["C1", "C3"]].min(axis=1) <= stop      # apytiksliai, is uzdarymu
    df["ex_stop3d"] = np.where(stop_hit,
                               (stop / ijejimas - 1) * 100,
                               (df["C3"] / ijejimas - 1) * 100)
    # B: tas pats, bet iki C_{D+5}
    stop_hit5 = df[["C1", "C3", "C5"]].min(axis=1) <= stop
    df["ex_stop5d"] = np.where(stop_hit5,
                               (stop / ijejimas - 1) * 100,
                               (df["C5"] / ijejimas - 1) * 100)
    return df


# ----------------------------- ATASKAITA -----------------------------

VARIANTAI = [
    ("v1_pilnas", "Pilnas: >3% + 2x + IBS>0.8 + virsune"),
    ("v2_svelnus", "Svelnus: >2% + 2x + IBS>0.8"),
    ("v3_kilimas_virsune", "Kilimas + virsune (be apyvartos)"),
    ("v4_kilimas_apyvarta", "Kilimas + apyvarta (be virsunes)"),
    ("p_kilimas", "Kontrole: tik kilimas >3%"),
    ("p_apyvarta", "Kontrole: tik apyvarta 2x"),
    ("p_virsuje", "Kontrole: tik IBS>0.8"),
    ("p_virsune", "Kontrole: tik nauja virsune"),
]


def pagrindine_lentele(df, pusė=None, antraste=""):
    if antraste:
        print("\n" + "=" * 104)
        print(antraste)
        print("=" * 104)
    print(f"{'VARIANTAS':<40} {'LANGAS':<8} {'N':>7} {'DIENU':>7} "
          f"{'DEMEAN.':>10} {'95% INTERVALAS':>22}")
    print("-" * 104)
    rez = []
    for col, lab in VARIANTAI:
        sub = df[df[col]] if pusė is None else df[df[col] & pusė]
        for lang, lab2 in [("r_1d_dm", "1 diena"), ("r_3d_dm", "3 dienos"),
                           ("r_5d_dm", "5 dienos")]:
            r, n = ivertink(sub, lang)
            if r is None:
                if lang == "r_1d_dm":
                    print(f"{lab:<40} {lab2:<8} {n:>7}   per maza imtis")
                continue
            ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
            print(f"{lab:<40} {lab2:<8} {r['n']:>7} {r['dienos']:>7} "
                  f"{r['mean']:>+9.3f}% {ci:>22}")
            rez.append((col, lab, lang, r))
    return rez


PJŪVIAI = [
    ("Kilimo dydis", lambda d: pd.cut(d["chg"], [-99, 5, 8, 99],
                                      labels=["3-5%", "5-8%", ">8%"])),
    ("Istemptumas (nuo SMA20 / ATR)",
     lambda d: pd.cut((d["C"] - d["sma20"]) / d["atr14"], [-99, 1.5, 3, 99],
                      labels=["<1.5", "1.5-3", ">3"])),
    ("Sektoriaus judesys", lambda d: pd.cut(d["sekt_ret"], [-99, 1.0, 99],
                                            labels=["sekt <1%", "sekt >=1%"])),
    ("Baze pries pramusima", lambda d: pd.cut(d["nuo_virsunes"], [-1, 15, 999],
                                              labels=["trumpa <15 d.", "ilga >=15 d."])),
    ("Rinkos rezimas", lambda d: pd.cut(d["C"] - d["sma50"], [-1e9, 0, 1e9],
                                        labels=["po SMA50", "virs SMA50"])),
    ("Zalios dienos is eiles", lambda d: pd.cut(d["zalios_serija"], [0, 1, 2, 99],
                                                labels=["1-a", "2-a", "3+"])),
    ("Savaites diena", lambda d: np.where(d["data"].dt.dayofweek == 4,
                                          "penktadienis", "kitos")),
]


def pjuviu_lentele(df):
    print("\n" + "=" * 104)
    print("PJUVIAI — kada tesinys veikia (variantai 1-2). SIOS LENTELES NELAIKYK "
          "PATVIRTINIMU:")
    print("ji skirta hipotezems generuoti, ne irodyti.")
    print("=" * 104)
    for col, lab in VARIANTAI[:2]:
        sub = df[df[col]].copy()
        if len(sub) < MIN_IVYKIU:
            print(f"\n{lab}: per maza imtis ({len(sub)})")
            continue
        print(f"\n{lab}  (n={len(sub)})")
        print(f"  {'PJUVIS':<32} {'LANGELIS':<16} {'N':>6} {'r_1d':>9} {'r_3d':>9}")
        print("  " + "-" * 78)
        for pav, f in PJŪVIAI:
            try:
                sub["_g"] = f(sub)
            except Exception:
                continue
            for g, gg in sub.groupby("_g", observed=True):
                if len(gg) < 50:
                    continue
                print(f"  {pav:<32} {str(g):<16} {len(gg):>6} "
                      f"{gg['r_1d_dm'].mean():>+8.3f}% {gg['r_3d_dm'].mean():>+8.3f}%")


def metu_lentele(df):
    print("\n" + "=" * 104)
    print("PAGAL METUS — ar efektas stabilus, ar is vieno laikotarpio")
    print("=" * 104)
    print(f"{'METAI':<8}" + "".join(f"{lab[:16]:>20}" for _, lab in VARIANTAI[:3]))
    print("-" * 72)
    df["_m"] = df["data"].dt.year
    for m, g in df.groupby("_m"):
        eil = f"{m:<8}"
        for col, _ in VARIANTAI[:3]:
            s = g[g[col]]
            eil += (f"{s['r_1d_dm'].mean():>+13.3f}% ({len(s):>3})"
                    if len(s) >= 20 else f"{'-':>20}")
        print(eil)


def main():
    ap = argparse.ArgumentParser(description="Tesinio hipotezes backtestas")
    ap.add_argument("--metai", type=int, default=10)
    ap.add_argument("--sanaudos", type=float, default=0.07)
    args = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Paleisk: pip install yfinance pandas numpy")

    df = surink(yf, args.metai)
    df = prideti_kontekstą(yf, df)
    df = pazymėk_ivykius(df)
    df = grazos(df)
    df = realistiski_isejimai(df)

    df = df.dropna(subset=["r_1d_dm"])
    print(f"\nLaikotarpis: {df['data'].min():%Y-%m-%d} .. {df['data'].max():%Y-%m-%d}")
    print(f"Eiluciu: {len(df):,} | akciju: {df['tag'].nunique()} | "
          f"sektoriu: {df['sekt'].nunique()}")
    print("Ivykiu pagal varianta:")
    for col, lab in VARIANTAI[:4]:
        print(f"  {lab:<42} {int(df[col].sum()):>7}")
    print("\nDEMESIO: universas — DABARTINIAI indekso nariai, todel yra islikimo "
          "salismas,\nkuris paprastai DIDINA teigiamus rezultatus.")

    pagrindine_lentele(df, antraste="PAGRINDINE LENTELE (visa imtis, demeanuota pagal diena)")

    # Dvi kalendorines puses
    riba = df["data"].quantile(0.5)
    p1, p2 = df["data"] <= riba, df["data"] > riba
    print(f"\nPadalijimas: 1-a puse iki {riba:%Y-%m-%d}, 2-a po jos")
    r1 = pagrindine_lentele(df, p1, "1 ETAPAS — PAIESKA (1-oji puse)")

    if r1:
        pv = np.array([r["p_two"] for *_, r in r1])
        islaike = benjamini_hochberg(pv)
        kand = [(c, lab, lang) for (c, lab, lang, _), ok in zip(r1, islaike) if ok]
        print(f"\n1 etape islaike (su BH pataisa): {len(kand)} is {len(r1)}")

        if kand:
            print("\n" + "=" * 104)
            print("2 ETAPAS — PATVIRTINIMAS (2-oji puse, nematyta)")
            print("=" * 104)
            print(f"{'VARIANTAS':<40} {'LANGAS':<8} {'N':>7} {'DEMEAN.':>10} "
                  f"{'95% INTERVALAS':>22} {'VERDIKTAS':>14}")
            print("-" * 104)
            patvirtinti = []
            for col, lab, lang in kand:
                r, n = ivertink(df[df[col] & p2], lang)
                if r is None:
                    print(f"{lab:<40} {lang:<8} {n:>7}   per maza imtis")
                    continue
                v = ("PATVIRTINTA" if r["lo"] > 0 else
                     ("PRIESINGA" if r["hi"] < 0 else "nepatvirtinta"))
                ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
                print(f"{lab:<40} {lang:<8} {r['n']:>7} {r['mean']:>+9.3f}% "
                      f"{ci:>22} {v:>14}")
                if r["lo"] > 0:
                    patvirtinti.append((col, lab, lang, r))

            # Atsparumas: isbraukiam po sektoriu
            if patvirtinti:
                print("\n" + "=" * 104)
                print("ATSPARUMAS — ar islieka isbraukus bet kuri sektoriu")
                print("=" * 104)
                for col, lab, lang, r in patvirtinti[:3]:
                    mn, mx = 99.0, -99.0
                    for sek in df["sekt"].unique():
                        rr, _ = ivertink(df[df[col] & (df["sekt"] != sek)], lang)
                        if rr:
                            mn, mx = min(mn, rr["mean"]), max(mx, rr["mean"])
                    print(f"{lab} ({lang}): nuo {mn:+.3f}% iki {mx:+.3f}%"
                          + ("  -> atsparus" if mn > 0 else "  -> DEMESIO: priklauso nuo sektoriaus"))

                print("\n" + "=" * 104)
                print(f"GALUTINIS VERTINIMAS (atemus {args.sanaudos}% sanaudu)")
                print("Sekmes kriterijus, uzrasytas pries paleidziant: neto >= +0.20%, "
                      "intervalo apacia > 0")
                print("=" * 104)
                for col, lab, lang, r in patvirtinti:
                    neto = r["mean"] - args.sanaudos
                    verd = "TENKINA" if (neto >= 0.20 and r["lo"] > 0) else "netenkina"
                    print(f"{lab:<40} {lang:<8} bruto {r['mean']:>+7.3f}%  "
                          f"neto {neto:>+7.3f}%  {verd}")
            else:
                print("\nNe vienas variantas nepasitvirtino nematytoje puseje.")
    else:
        print("\n1 etape nepakako duomenu.")

    # Realistiski isejimai
    print("\n" + "=" * 104)
    print("REALISTISKI ISEJIMAI (variantai 1-2, be demeanavimo — grynas rezultatas)")
    print("=" * 104)
    print(f"{'VARIANTAS':<40} {'ISEJIMAS':<22} {'N':>7} {'VID.':>10}")
    print("-" * 84)
    for col, lab in VARIANTAI[:2]:
        s = df[df[col]]
        if len(s) < MIN_IVYKIU:
            continue
        for c, lab2 in [("ex_stop3d", "stop po dugno, 3 d."),
                        ("ex_stop5d", "stop po dugno, 5 d.")]:
            v = s[c].dropna()
            if len(v) >= MIN_IVYKIU:
                print(f"{lab:<40} {lab2:<22} {len(v):>7} {v.mean():>+9.3f}%")

    pjuviu_lentele(df)
    metu_lentele(df)

    print("\nAPRIBOJIMAI: islikimo salismas (dabartiniai indekso nariai); "
          "\nstop tikrinamas is uzdarymu, ne intraday — realiai stop suveiktu daznlau; "
          "\nsanaudos ir spread'as atimti tik galutineje lenteleje; "
          "\npjuviu lentele skirta hipotezems, ne patvirtinimui.")


if __name__ == "__main__":
    main()
