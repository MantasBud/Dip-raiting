#!/usr/bin/env python3
"""
backtest_universas_60m.py — ar patvirtintas INTRADAY dip'o signalas (Z-balas žemas,
IBS žemas, kaina žemiau VWAP) duoda daugiau, kai rikiuojama tarp 100–200 EUR akcijų,
o ne tarp 25?

Metodika (ta pati, kuri buvo naudota 60m testuose):
  - 60m barai, ~730 prekybos dienų (Yahoo riba), tik EUR biržos
  - kontroliniai taškai kiekvienoje sesijoje: barų 1..6 uždarymai (10:00–16:00 Berlyno laiku)
  - signalas skaičiuojamas TUO MOMENTU: Z-balas (40 valandinių barų), IBS nuo dienos
    pradžios, atstumas nuo sesijos VWAP; visi trys sujungiami į vieną rangą
  - laikymas: N sesijų (pagal nutylėjimą 3 = 25,5 prekybos val.), išėjimas tame pačiame
    kontroliniame taške; papildomai — su stop/tikslu
  - rezultatas demeanuojamas pagal (diena, taškas) per visą universą → matuojama ATRANKA,
    ne rinkos diena
  - kiekviename taške: TOP-5 pagal rangą (≤2 iš sektoriaus), kontrolės: atsitiktiniai 5,
    BLOGIAUSI 5 (turi būti blogiau), siauras sąrašas (tavo dabartinės 25) — top-1 ir top-5
  - statistika: dienų vidurkiai, savaičių blokų bootstrap, dvi laiko pusės, metų lentelė

Slenksčiai fiksuoti. Nederinti pagal rezultatą.
Priklausomybės: yfinance, pandas, numpy.
"""

import argparse
import sys
import time
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

TZ = "Europe/Berlin"
INDEKSAS = "EXSA.DE"          # STOXX Europe 600 ETF — režimui
BARU_SESIJOJE = 9             # 09:00 … 17:00 (paskutinis baras 17:00–17:30)
TASKAI = [1, 2, 3, 4, 5, 6]   # kontroliniai barai (uždarymai 11:00 … 16:00)
Z_LANGAS = 20                 # valandiniu baru (~2 sesijos) — TAI, KAS PATVIRTINTA
                              # backtest_intraday.py teste. Buvo 40 (~4.5 sesijos),
                              # t. y. kitas dydis nei patvirtintas signalas.
MAX_SEKT = 2                  # ne daugiau iš vieno sektoriaus top-5
BOOT_N = 1000
SEED = 7

# ---------------------------------------------------------------------------
# UNIVERSAS — EUR biržos. Sektoriaus žymos grubios; svarbu tik, kad tos pačios
# pramonės akcijos turėtų tą pačią žymą.
# ---------------------------------------------------------------------------
UNIVERSAS = {
    # --- XETRA ---
    "SAP.DE": "tech", "SIE.DE": "pramone", "ALV.DE": "finansai", "DTE.DE": "telekom",
    "MUV2.DE": "finansai", "AIR.DE": "gynyba", "BAS.DE": "chemija", "BAYN.DE": "farma",
    "BMW.DE": "auto", "MBG.DE": "auto", "VOW3.DE": "auto", "P911.DE": "auto",
    "DHL.DE": "logistika", "DB1.DE": "finansai", "DBK.DE": "bankai", "CBK.DE": "bankai",
    "IFX.DE": "puslaidininkiai", "ADS.DE": "vartojimas", "HEN3.DE": "vartojimas",
    "BEI.DE": "vartojimas", "MRK.DE": "farma", "FRE.DE": "sveikata", "RHM.DE": "gynyba",
    "MTX.DE": "aviacija", "HEI.DE": "statyba", "SY1.DE": "chemija", "ZAL.DE": "prekyba",
    "EOAN.DE": "energetika", "RWE.DE": "energetika", "VNA.DE": "nt", "1COV.DE": "chemija",
    "BNR.DE": "chemija", "CON.DE": "auto", "SRT3.DE": "sveikata", "QIA.DE": "sveikata",
    "HNR1.DE": "finansai", "ENR.DE": "energetika", "PAH3.DE": "auto", "AIXA.DE": "puslaidininkiai",
    "SHL.DE": "sveikata", "DTG.DE": "auto", "HLAG.DE": "logistika", "NEM.DE": "tech",
    "EVK.DE": "chemija", "LHA.DE": "aviacija", "TKA.DE": "pramone", "PUM.DE": "vartojimas",
    "SDF.DE": "chemija", "G1A.DE": "pramone", "LEG.DE": "nt", "FNTN.DE": "telekom",
    "TEG.DE": "nt", "NDA.DE": "energetika", "KBX.DE": "pramone", "BC8.DE": "tech",
    "SIX2.DE": "sveikata", "WAF.DE": "puslaidininkiai", "KGX.DE": "pramone", "RAA.DE": "sveikata",
    "HFG.DE": "prekyba", "SAX.DE": "pramone", "JEN.DE": "tech", "GXI.DE": "sveikata",
    "AOF.DE": "tech", "S92.DE": "energetika", "BOSS.DE": "vartojimas", "NDX1.DE": "tech",
    # --- Euronext Paris ---
    "MC.PA": "prabanga", "OR.PA": "vartojimas", "TTE.PA": "nafta", "SAN.PA": "farma",
    "AI.PA": "chemija", "SU.PA": "pramone", "BNP.PA": "bankai", "CS.PA": "finansai",
    "RMS.PA": "prabanga", "KER.PA": "prabanga", "DG.PA": "statyba", "EL.PA": "sveikata",
    "SAF.PA": "aviacija", "HO.PA": "gynyba", "DSY.PA": "tech", "CAP.PA": "tech",
    "BN.PA": "vartojimas", "RI.PA": "vartojimas", "ENGI.PA": "energetika", "ORA.PA": "telekom",
    "VIE.PA": "energetika", "SGO.PA": "statyba", "ML.PA": "auto", "STLAP.PA": "auto",
    "RNO.PA": "auto", "ACA.PA": "bankai", "GLE.PA": "bankai", "PUB.PA": "media",
    "LR.PA": "pramone", "STMPA.PA": "puslaidininkiai", "WLN.PA": "tech", "EDEN.PA": "tech",
    "ERF.PA": "sveikata", "TEP.PA": "tech", "ALO.PA": "pramone", "BVI.PA": "pramone",
    "AC.PA": "kelioniu", "ENX.PA": "finansai", "SW.PA": "vartojimas", "CA.PA": "prekyba",
    "FR.PA": "auto", "VIV.PA": "media", "EN.PA": "statyba", "URW.PA": "nt", "RXL.PA": "pramone",
    "DIM.PA": "sveikata", "GTT.PA": "pramone", "NK.PA": "pramone", "COV.PA": "nt",
    "AMUN.PA": "finansai", "SOP.PA": "tech", "IPN.PA": "farma", "MF.PA": "pramone",
    # --- Euronext Amsterdam ---
    "ASML.AS": "puslaidininkiai", "ASM.AS": "puslaidininkiai", "BESI.AS": "puslaidininkiai",
    "ADYEN.AS": "tech", "INGA.AS": "bankai", "PHIA.AS": "sveikata", "PRX.AS": "tech",
    "HEIA.AS": "vartojimas", "AD.AS": "prekyba", "DSFIR.AS": "chemija", "WKL.AS": "media",
    "KPN.AS": "telekom", "RAND.AS": "paslaugos", "NN.AS": "finansai", "ABN.AS": "bankai",
    "AKZA.AS": "chemija", "IMCD.AS": "chemija", "MT.AS": "metalai", "UMG.AS": "media",
    "SHELL.AS": "nafta", "UNA.AS": "vartojimas", "AGN.AS": "finansai", "ASRNL.AS": "finansai",
    "JDEP.AS": "vartojimas", "TKWY.AS": "tech", "ARCAD.AS": "paslaugos", "FUR.AS": "pramone",
    "OCI.AS": "chemija", "SBMO.AS": "nafta", "LIGHT.AS": "pramone", "AALB.AS": "pramone",
    # --- Borsa Italiana ---
    "ENI.MI": "nafta", "ENEL.MI": "energetika", "ISP.MI": "bankai", "UCG.MI": "bankai",
    "STLAM.MI": "auto", "RACE.MI": "auto", "G.MI": "finansai", "LDO.MI": "gynyba",
    "PRY.MI": "pramone", "TEN.MI": "nafta", "MONC.MI": "prabanga", "BAMI.MI": "bankai",
    "MB.MI": "bankai", "TIT.MI": "telekom", "SRG.MI": "energetika", "TRN.MI": "energetika",
    "A2A.MI": "energetika", "REC.MI": "farma", "CPR.MI": "vartojimas", "AMP.MI": "farma",
    "BMED.MI": "finansai", "PST.MI": "finansai", "FBK.MI": "bankai", "DIA.MI": "sveikata",
    "IP.MI": "pramone", "BPE.MI": "bankai", "UNI.MI": "finansai", "NEXI.MI": "tech",
    "STM.MI": "puslaidininkiai", "BZU.MI": "statyba", "BC.MI": "prabanga", "IVG.MI": "pramone",
    "MONC.MI": "prabanga", "INW.MI": "telekom", "ERG.MI": "energetika", "AZM.MI": "finansai",
    "BGN.MI": "finansai", "IG.MI": "energetika", "SFER.MI": "prabanga", "HER.MI": "energetika",
    # --- Bolsa de Madrid ---
    "SAN.MC": "bankai", "BBVA.MC": "bankai", "IBE.MC": "energetika", "ITX.MC": "prekyba",
    "TEF.MC": "telekom", "REP.MC": "nafta", "AMS.MC": "tech", "FER.MC": "statyba",
    "CABK.MC": "bankai", "AENA.MC": "kelioniu", "CLNX.MC": "telekom", "ACS.MC": "statyba",
    "IAG.MC": "aviacija", "ENG.MC": "energetika", "RED.MC": "energetika", "GRF.MC": "farma",
    "SAB.MC": "bankai", "BKT.MC": "bankai", "ELE.MC": "energetika", "NTGY.MC": "energetika",
    "ANA.MC": "statyba", "MAP.MC": "finansai", "ACX.MC": "metalai", "SCYR.MC": "statyba",
    "LOG.MC": "logistika", "ROVI.MC": "farma", "PUIG.MC": "vartojimas", "UNI.MC": "bankai",
    "MTS.MC": "metalai", "IDR.MC": "tech", "SLR.MC": "energetika", "FDR.MC": "auto",
    "COL.MC": "nt", "MRL.MC": "nt", "VIS.MC": "sveikata", "CIE.MC": "auto",
    # --- Euronext Brussels ---
    "ABI.BR": "vartojimas", "KBC.BR": "bankai", "UCB.BR": "farma", "SOLB.BR": "chemija",
    "AGS.BR": "finansai", "GBLB.BR": "finansai", "ARGX.BR": "farma", "ELI.BR": "energetika",
    "PROX.BR": "telekom", "COLR.BR": "prekyba", "DIE.BR": "prekyba", "MELE.BR": "tech",
    "WDP.BR": "nt", "ACKB.BR": "finansai", "LOTB.BR": "vartojimas", "UMI.BR": "chemija",
    "AED.BR": "nt", "SYENS.BR": "chemija", "XIOR.BR": "nt", "BEKB.BR": "pramone",
    # --- Nasdaq Helsinki ---
    "NOKIA.HE": "telekom", "NESTE.HE": "nafta", "SAMPO.HE": "finansai", "UPM.HE": "miskas",
    "KNEBV.HE": "pramone", "FORTUM.HE": "energetika", "STERV.HE": "miskas", "WRT1V.HE": "pramone",
    "ELISA.HE": "telekom", "ORNBV.HE": "farma", "METSO.HE": "pramone", "NDA-FI.HE": "bankai",
    "KESKOB.HE": "prekyba", "HUH1V.HE": "miskas", "VALMT.HE": "pramone", "TYRES.HE": "auto",
    "OUT1V.HE": "metalai", "KCR.HE": "pramone", "CGCBV.HE": "pramone", "QTCOM.HE": "tech",
    "TIETO.HE": "tech", "KEMIRA.HE": "chemija", "MANTA.HE": "finansai", "TELIA1.HE": "telekom",
    # --- Euronext Lisbon ---
    "EDP.LS": "energetika", "GALP.LS": "nafta", "JMT.LS": "prekyba", "EDPR.LS": "energetika",
    "BCP.LS": "bankai", "NOS.LS": "telekom", "SON.LS": "prekyba", "CTT.LS": "logistika",
    "NVG.LS": "miskas", "ALTR.LS": "miskas", "SEM.LS": "miskas", "REN.LS": "energetika",
    # --- Wiener Börse ---
    "OMV.VI": "nafta", "EBS.VI": "bankai", "VER.VI": "energetika", "VOE.VI": "metalai",
    "ANDR.VI": "pramone", "RBI.VI": "bankai", "BG.VI": "bankai", "WIE.VI": "statyba",
    "CAI.VI": "nt", "POST.VI": "logistika", "LNZ.VI": "chemija", "SBO.VI": "nafta",
    "UQA.VI": "finansai", "ATS.VI": "tech", "DOC.VI": "pramone", "IIA.VI": "nt",
    # --- Euronext Dublin ---
    "RYA.IR": "aviacija", "KRZ.IR": "vartojimas", "BIRG.IR": "bankai", "A5G.IR": "bankai",
    "GL9.IR": "vartojimas", "KSP.IR": "statyba", "DHG.IR": "kelioniu", "PTSB.IR": "bankai",
}

# Siauras sąrašas — dabartinės skenerio 25 akcijos. Jei dip_reitingas.py yra šalia,
# bandoma paimti iš jo; kitaip užpildyk ranka.
SIAURAS = []


def suvienodinti_universa():
    """Jei repozitorijoje yra universas.py, naudojam JI kaip vieninteli saltini.

    Kitaip du failai turetu du skirtingus sarasus ir skeneris matuotu ne ta,
    kas patikrinta. Sio failo vidinis sarasas lieka kaip atsarginis.
    """
    global UNIVERSAS
    try:
        import universas as U
        sekt = U.sektoriai()
        if len(sekt) >= 50:
            papildyta = dict(UNIVERSAS)
            papildyta.update(sekt)      # universas.py laimi, likusius paliekam
            UNIVERSAS = papildyta
            print(f"Universas suvienodintas su universas.py: {len(UNIVERSAS)} tikeriu")
    except Exception:
        pass


def paimti_siaura_sarasa():
    global SIAURAS
    if SIAURAS:
        return
    try:
        import dip_reitingas as dr
        wl = getattr(dr, "WATCHLIST", None) or getattr(dr, "TICKERS", None)
        if isinstance(wl, dict):
            SIAURAS = list(wl.keys())
        elif isinstance(wl, (list, tuple)):
            out = []
            for x in wl:
                if isinstance(x, str):
                    out.append(x)
                elif isinstance(x, dict):
                    out.append(x.get("ticker") or x.get("tag") or x.get("symbol"))
                elif isinstance(x, (list, tuple)):
                    # WATCHLIST formatas: (tag, yahoo_simbolis, pavadinimas).
                    # Imam Yahoo simboli — x[0] yra tik trumpinys ("AIXA"), jis
                    # nesutaptu su atsisiustais duomenimis ("AIXA.DE").
                    imk = next((v for v in x if isinstance(v, str) and "." in v), None)
                    out.append(imk or (x[0] if x else None))
            SIAURAS = [t for t in out if t]
    except Exception:
        SIAURAS = []


# ---------------------------------------------------------------------------
# Atsisiuntimas
# ---------------------------------------------------------------------------
def siusti_60m(tickers, chunk=25):
    out = {}
    for i in range(0, len(tickers), chunk):
        dalis = tickers[i:i + chunk]
        for bandymas in range(3):
            try:
                df = yf.download(dalis, interval="60m", period="730d", group_by="ticker",
                                 auto_adjust=False, progress=False, threads=True)
                break
            except Exception as e:
                print(f"  klaida ({e}); kartojama…", file=sys.stderr)
                time.sleep(5)
        else:
            continue
        if df is None or df.empty:
            continue
        if len(dalis) == 1:
            out[dalis[0]] = df
            continue
        for t in dalis:
            if t in df.columns.get_level_values(0):
                sub = df[t].dropna(subset=["Close"])
                if len(sub) > 500:
                    out[t] = sub
        print(f"  {min(i + chunk, len(tickers))}/{len(tickers)}", file=sys.stderr)
    return out


def siusti_dieninius(ticker):
    df = yf.download(ticker, period="4y", interval="1d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close"])


# ---------------------------------------------------------------------------
# Vienos akcijos eilutės
# ---------------------------------------------------------------------------
def eilutes(tag, sekt, df, laikymas, stop_pct, tikslas_pct):
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(TZ)
    df = df[(df.index.hour >= 9) & (df.index.hour <= 17)]
    df["data"] = df.index.date
    df["k"] = df.groupby("data").cumcount()
    pilnos = df.groupby("data")["k"].max()
    geros = pilnos[pilnos >= BARU_SESIJOJE - 2].index          # bent 8 barai
    df = df[df["data"].isin(geros)].copy()
    if len(df) < 300:
        return None

    # sesijos dydžiai
    df["c_high"] = df.groupby("data")["High"].cummax()
    df["c_low"] = df.groupby("data")["Low"].cummin()
    df["c_vol"] = df.groupby("data")["Volume"].cumsum()
    df["c_pv"] = (df["Close"] * df["Volume"]).groupby(df["data"]).cumsum()
    df["vwap"] = df["c_pv"] / df["c_vol"].replace(0, np.nan)
    df["ibs"] = (df["Close"] - df["c_low"]) / (df["c_high"] - df["c_low"]).replace(0, np.nan)
    df["vwap_d"] = df["Close"] / df["vwap"] - 1

    # Z-balas per valandinius barus
    r = df["Close"].rolling(Z_LANGAS)
    df["z"] = (df["Close"] - r.mean()) / r.std().replace(0, np.nan)

    # dienos kontekstas iš užbaigtų sesijų
    dien = df.groupby("data").agg(close=("Close", "last"), vol=("Volume", "sum"))
    dien["turnover"] = dien["close"] * dien["vol"]
    dien["prev_close"] = dien["close"].shift(1)
    dien["raud"] = (dien["close"] < dien["prev_close"]).astype(int)
    # raudonų sesijų iš eilės IKI vakar imtinai
    seka = []
    n = 0
    for v in dien["raud"].values:
        seka.append(n)
        n = n + 1 if v == 1 else 0
    dien["raud_pries"] = seka
    dien["liq20"] = dien["turnover"].rolling(20).median().shift(1)
    dien["idx"] = np.arange(len(dien))
    df = df.join(dien[["prev_close", "raud_pries", "liq20", "idx"]], on="data")
    df["day_chg"] = df["Close"] / df["prev_close"] - 1

    df = df[df["k"].isin(TASKAI)].copy()
    df = df.dropna(subset=["z", "ibs", "vwap_d", "prev_close", "liq20"])
    if df.empty:
        return None

    # išėjimai: (sesija + laikymas, tas pats k)
    raktas = {(int(i), int(k)): c for i, k, c in zip(df["idx"], df["k"], df["Close"])}
    df["exit_raw"] = [raktas.get((int(i) + laikymas, int(k)), np.nan)
                      for i, k in zip(df["idx"], df["k"])]
    df["r_raw"] = df["exit_raw"] / df["Close"] - 1

    return df


def keliai_stop(tag, df_pilnas, df_taskai, laikymas, stop_pct, tikslas_pct):
    """Stop/tikslo rezultatas kiekvienam kontroliniam taškui, einant per visus 60m barus."""
    full = df_pilnas.copy()
    full.index = pd.to_datetime(full.index, utc=True).tz_convert(TZ)
    full = full[(full.index.hour >= 9) & (full.index.hour <= 17)].dropna(subset=["Close"])
    full["data"] = full.index.date
    full["k"] = full.groupby("data").cumcount()
    dienos = {d: i for i, d in enumerate(sorted(full["data"].unique()))}
    full["idx"] = full["data"].map(dienos)
    hi = full["High"].values
    lo = full["Low"].values
    cl = full["Close"].values
    key = {(int(i), int(k)): p for p, (i, k) in enumerate(zip(full["idx"], full["k"]))}

    out = []
    for data, k, ent in zip(df_taskai["data"], df_taskai["k"], df_taskai["Close"]):
        i0 = dienos.get(data)
        if i0 is None:
            out.append(np.nan)
            continue
        p0 = key.get((i0, int(k)))
        p1 = key.get((i0 + laikymas, int(k)))
        if p0 is None or p1 is None:
            out.append(np.nan)
            continue
        stop = ent * (1 - stop_pct / 100)
        tp = ent * (1 + tikslas_pct / 100)
        h = hi[p0 + 1:p1 + 1]
        l = lo[p0 + 1:p1 + 1]
        s_hit = np.where(l <= stop)[0]
        t_hit = np.where(h >= tp)[0]
        s_i = s_hit[0] if len(s_hit) else 10 ** 9
        t_i = t_hit[0] if len(t_hit) else 10 ** 9
        if s_i <= t_i and s_i < 10 ** 9:
            out.append(stop / ent - 1)
        elif t_i < 10 ** 9:
            out.append(tp / ent - 1)
        else:
            out.append(cl[p1] / ent - 1)
    return np.array(out)


# ---------------------------------------------------------------------------
# Statistika
# ---------------------------------------------------------------------------
def boot_ci(dienu_vid: pd.Series, n=BOOT_N, seed=SEED):
    s = dienu_vid.dropna()
    if len(s) < 10:
        return np.nan, np.nan
    s.index = pd.to_datetime(s.index)
    sav = s.index.to_period("W")
    grupes = [g.values for _, g in s.groupby(sav)]
    rng = np.random.default_rng(seed)
    vid = []
    for _ in range(n):
        pick = rng.integers(0, len(grupes), len(grupes))
        vid.append(np.concatenate([grupes[p] for p in pick]).mean())
    return np.percentile(vid, 2.5), np.percentile(vid, 97.5)


def apibendrinti(sel: pd.DataFrame, stulpelis: str):
    """sel — atrinktos eilutės su 'data', 'k', stulpelis. Grąžina n, dienų, vid, lo, hi."""
    if sel is None or sel.empty:
        return 0, 0, np.nan, np.nan, np.nan
    per_taska = sel.groupby(["data", "k"])[stulpelis].mean()
    per_diena = per_taska.groupby(level=0).mean()
    lo, hi = boot_ci(per_diena)
    return len(sel), len(per_diena), per_diena.mean(), lo, hi


def fmt(n, d, m, lo, hi):
    if n == 0 or np.isnan(m):
        return f"{'—':>8} {'—':>6}   per maža imtis"
    return f"{n:>8} {d:>6}   {m*100:+.3f}%   {lo*100:+.3f} .. {hi*100:+.3f}"


# ---------------------------------------------------------------------------
# Atranka kiekviename (diena, taškas)
# ---------------------------------------------------------------------------
def atrinkti(df, rinkinys, top, rng):
    """Grąžina žodyną {pavadinimas: DataFrame atrinktų eilučių}."""
    rez = {p: [] for p in ["TOP pagal ranga", "TOP tik Z", "BLOGIAUSI (kontrole)",
                            "ATSITIKTINIAI (kontrole)", "SIAURAS top-1", "SIAURAS top-5"]}
    for (data, k), g in df.groupby(["data", "k"], sort=False):
        kand = g[g["tinka"]]
        if len(kand) < 3:
            continue
        # top pagal jungtinį rangą su sektoriaus riba
        ks = kand.sort_values("rangas")
        imti, sekt_cnt = [], {}
        for i, r in ks.iterrows():
            if sekt_cnt.get(r["sekt"], 0) >= MAX_SEKT:
                continue
            imti.append(i)
            sekt_cnt[r["sekt"]] = sekt_cnt.get(r["sekt"], 0) + 1
            if len(imti) >= top:
                break
        rez["TOP pagal ranga"].append(kand.loc[imti])
        rez["TOP tik Z"].append(kand.nsmallest(top, "z"))
        rez["BLOGIAUSI (kontrole)"].append(kand.nlargest(top, "rangas"))
        rez["ATSITIKTINIAI (kontrole)"].append(kand.sample(min(top, len(kand)), random_state=int(rng.integers(1e9))))
        s = kand[kand["siauras"]]
        if len(s) >= 1:
            rez["SIAURAS top-1"].append(s.nsmallest(1, "rangas"))
            rez["SIAURAS top-5"].append(s.nsmallest(min(top, len(s)), "rangas"))
    return {p: (pd.concat(v) if v else pd.DataFrame()) for p, v in rez.items()}


def lentele(pav, rinkiniai, stulp, sanaudos):
    print(f"\n{'RINKINYS':<28} {'LANGAS':<10} {'N':>8} {'DIENU':>6}   {'DEMEAN.':>9}   95% INTERVALAS")
    print("-" * 96)
    for p, sel in rinkiniai.items():
        n, d, m, lo, hi = apibendrinti(sel, stulp)
        print(f"{p:<28} {pav:<10} {fmt(n, d, m, lo, hi)}")
    print(f"  (sąnaudos {sanaudos:.2f} % dar neatimtos; jos taikomos TOP rinkiniams, ne universo vidurkiui)")


# ---------------------------------------------------------------------------
def main():
    suvienodinti_universa()
    global Z_LANGAS
    ap = argparse.ArgumentParser()
    ap.add_argument("--laikymas", type=int, default=3, help="sesijų (3 = 25,5 prekybos val.)")
    ap.add_argument("--z-langas", type=int, default=Z_LANGAS,
                    help="Z-balo langas valandiniais barais (20 = ~2 sesijos, patvirtinta)")
    ap.add_argument("--sanaudos", type=float, default=0.07)
    ap.add_argument("--min-apyvarta", type=float, default=20.0, help="mln EUR, 20 d. mediana")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--stop", type=float, default=1.5)
    ap.add_argument("--tikslas", type=float, default=2.0)
    ap.add_argument("--be-rezimo", action="store_true", help="netaikyti rinkos režimo filtro")
    a = ap.parse_args()
    Z_LANGAS = a.z_langas

    paimti_siaura_sarasa()
    tickers = sorted(UNIVERSAS)
    for t in SIAURAS:
        if t not in UNIVERSAS:
            UNIVERSAS[t] = "siauras_kita"
            tickers.append(t)

    print(f"Universas: {len(tickers)} EUR akcijų; siauras sąrašas: {len(SIAURAS)}")
    print("Siunčiama 60m istorija (~730 d.)…", file=sys.stderr)
    duom = siusti_60m(tickers)
    print(f"  gauta: {len(duom)}; be duomenų: {len(tickers) - len(duom)}")

    # režimas iš indekso dienos barų
    idx = siusti_dieninius(INDEKSAS)
    idx["sma50"] = idx["Close"].rolling(50).mean()
    idx["ret20"] = idx["Close"].pct_change(20)
    rez = ((idx["Close"] > idx["sma50"]) & (idx["ret20"] > 0)).shift(1)   # vakar dienos būsena
    rezimas = {d.date(): bool(v) for d, v in rez.dropna().items()}

    dalys = []
    atmesta_liq = 0
    for tag, df in duom.items():
        e = eilutes(tag, UNIVERSAS.get(tag, "kita"), df, a.laikymas, a.stop, a.tikslas)
        if e is None:
            continue
        if e["liq20"].median() < a.min_apyvarta * 1e6:
            atmesta_liq += 1
            continue
        e["r_stop"] = keliai_stop(tag, df, e, a.laikymas, a.stop, a.tikslas)
        e["tag"] = tag
        e["sekt"] = UNIVERSAS.get(tag, "kita")
        e["siauras"] = tag in SIAURAS
        dalys.append(e[["tag", "sekt", "siauras", "data", "k", "Close", "z", "ibs", "vwap_d",
                        "day_chg", "raud_pries", "liq20", "r_raw", "r_stop"]])
    df = pd.concat(dalys, ignore_index=True)
    df = df.dropna(subset=["r_raw"])
    df["rezimas"] = df["data"].map(rezimas).fillna(False).astype(bool)
    print(f"  po likvidumo filtro: {df['tag'].nunique()} akcijų (atmesta {atmesta_liq}); "
          f"eilučių {len(df):,}; dienų {df['data'].nunique()}")
    print(f"  režimas teigiamas: {df.groupby('data')['rezimas'].first().mean()*100:.0f}% dienų")

    # jungtinis grįžimo-prie-vidurkio rangas (mažesnis = geriau)
    for c in ["z", "ibs", "vwap_d"]:
        df[c + "_pr"] = df.groupby(["data", "k"])[c].rank(pct=True)
    df["rangas"] = df[["z_pr", "ibs_pr", "vwap_d_pr"]].mean(axis=1)

    # kieti filtrai — tinkamumas kandidatu
    df["tinka"] = (
        (df["day_chg"] >= -0.06) & (df["day_chg"] <= 0.02)
        & (df["raud_pries"] <= 2)
        & (df["rezimas"] | a.be_rezimo)
    )

    # demeaninimas pagal (diena, taškas) per VISĄ likvidų universą
    for c in ["r_raw", "r_stop"]:
        df[c + "_dm"] = df[c] - df.groupby(["data", "k"])[c].transform("mean")

    kand = df[df["tinka"]].groupby(["data", "k"]).size()
    print(f"  kandidatų per tašką: mediana {kand.median():.0f}; ≥5 kandidatų: {(kand >= 5).mean()*100:.0f}% taškų")

    rng = np.random.default_rng(SEED)
    vid = df["data"].sort_values().iloc[len(df) // 2]
    puses = {"VISA IMTIS": df, "1-oji PUSĖ": df[df["data"] <= vid], "2-oji PUSĖ (nematyta)": df[df["data"] > vid]}
    print(f"\nPadalijimas: 1-a pusė iki {vid}")

    for pav, d in puses.items():
        print("\n" + "=" * 96 + f"\n{pav}  — laikymas {a.laikymas} sesijos, top-{a.top}\n" + "=" * 96)
        rink = atrinkti(d, None, a.top, rng)
        lentele("raw", rink, "r_raw_dm", a.sanaudos)
        lentele("stop/tikslas", rink, "r_stop_dm", a.sanaudos)

    # metų lentelė TOP pagal rangą ir SIAURAS top-1
    print("\n" + "=" * 96 + "\nPAGAL METUS (demeanuota r_raw)\n" + "=" * 96)
    rink = atrinkti(df, None, a.top, rng)
    met = {}
    for p in ["TOP pagal ranga", "SIAURAS top-1", "BLOGIAUSI (kontrole)"]:
        s = rink[p]
        if s.empty:
            continue
        s = s.copy()
        s["metai"] = pd.to_datetime(s["data"]).dt.year
        met[p] = s.groupby("metai")["r_raw_dm"].agg(["mean", "size"])
    metai = sorted({m for v in met.values() for m in v.index})
    print(f"{'METAI':<8}" + "".join(f"{p:>28}" for p in met))
    for m in metai:
        eil = f"{m:<8}"
        for p, v in met.items():
            if m in v.index:
                eil += f"{v.loc[m,'mean']*100:+.3f}% ({int(v.loc[m,'size'])})".rjust(28)
            else:
                eil += f"{'—':>28}"
        print(eil)

    # galutinis vertinimas
    print("\n" + "=" * 96 + "\nGALUTINIS VERTINIMAS (2-oji pusė, po sąnaudų)\n" + "=" * 96)
    d2 = puses["2-oji PUSĖ (nematyta)"]
    rink2 = atrinkti(d2, None, a.top, rng)
    for p in ["TOP pagal ranga", "TOP tik Z", "SIAURAS top-1", "SIAURAS top-5"]:
        n, d, m, lo, hi = apibendrinti(rink2[p], "r_raw_dm")
        if n == 0 or np.isnan(m):
            print(f"{p:<28} per maža imtis")
            continue
        neto = m - a.sanaudos / 100
        verd = "PATVIRTINTA" if lo > 0 and neto > 0 else "nepatvirtinta"
        print(f"{p:<28} bruto {m*100:+.3f}%  neto {neto*100:+.3f}%  apačia {lo*100:+.3f}%  → {verd}")

    print("\nKriterijus (užrašytas prieš paleidžiant): TOP pagal rangą neto > 0, intervalo apačia > 0 "
          "abiejose pusėse, ir geriau už SIAURAS top-1 — tik tada plėsti universą.")
    print("BLOGIAUSI turi būti blogiau už TOP; jei ne — rangas neneša informacijos.")
    print("APRIBOJIMAI: išlikimo šališkumas (dabartiniai nariai); 60m barai (~730 d.); "
          "stop iš 60m barų H/L; režimas — vakar dienos būsena.")


if __name__ == "__main__":
    main()
