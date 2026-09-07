#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rangavimas — A ir B set up'u taisykles.

SI FAILA IMPORTUOJA IR BACKTESTAS, IR SKENERIS. Taisykles aprasytos vienoje
vietoje, todel skeneris negali tikrinti kitokiu salygu nei tos, kurios buvo
ismatuotos. Slenksciai uzrasyti pries matavima ir nederinami pagal rezultatus.
"""

import numpy as np
import pandas as pd

# ----------------------------- SLENKSCIAI -----------------------------
# Uzrasyti 2026-09-08, pries paleidziant 1 faze. Nederinami.

# A: dip'as (triuksmo atsitraukimas)
DIP_MIN = -4.0          # dienos pokytis nuo -4%
DIP_MAX = -1.0          # iki -1%
DIP_APYV_MAX = 2.5      # apyvarta ne didesne (kitaip tai naujiena)
DIP_RAUD_MAX = 2        # raudonu dienu is eiles ne daugiau
ATASKAITOS_LANGAS = (-1, 5)   # dienos iki/po ataskaitos, kurias praleidziam

# B: prasidedantis trendas
B_VIRS_SMA50_DIENOS = 3       # tiek sesiju is eiles virs SMA50
B_PRIES_TAI_ZEMIAU = 40       # tiek is 60 ankstesniu sesiju buvo zemiau
B_APYV_MIN = 1.5              # kirtimo dienos apyvarta
B_MAX_ISTEMPIMAS = 1.08       # ne auksciau nei SMA50 x tiek

# Trendo busena
TRENDAS_STIPRUS_MOM = 0.67    # mom_6m rangas universe
TRENDAS_SILPNAS_MOM = 0.33

TOP_N = 5                     # kiek pozicijų rodoma
MAX_IS_SEKTORIAUS = 2


def konteksto_stulpeliai(d):
    """Dienos barų sluoksnis vienai akcijai. d — DataFrame su O/H/L/C/Volume.

    Visi rodikliai skaiciuojami is UZBAIGTU baru: paskutine eilute yra vakar.
    """
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
    pc = c.shift(1)
    out = pd.DataFrame(index=d.index)

    out["C"] = c
    out["O"] = d["Open"]
    out["H"] = h
    out["L"] = l
    out["ret"] = (c / pc - 1) * 100
    rng = (h - l).replace(0, np.nan)
    out["ibs"] = (c - l) / rng

    out["sma20"] = c.rolling(20).mean()
    out["sma50"] = c.rolling(50).mean()
    out["sma200"] = c.rolling(200).mean()
    out["std20"] = c.rolling(20).std()
    out["z20"] = (c - out["sma20"]) / out["std20"]

    out["mom_3m"] = c / c.shift(63) - 1
    out["mom_6m"] = c / c.shift(126) - 1

    out["virs_sma50"] = (c > out["sma50"]) & (out["sma50"] > out["sma50"].shift(10))
    out["virs_sma200"] = c > out["sma200"]

    out["apyv_sant"] = v / v.rolling(21).median().shift(1)
    out["zem20"] = l.rolling(20).min().shift(1)
    out["hi20"] = h.rolling(20).max().shift(1)

    raud = (c < pc).astype(int)
    out["raud_is_eiles"] = raud.groupby((raud == 0).cumsum()).cumsum()

    # B salyga: kiek is paskutiniu 3 sesiju virs SMA50, ir kiek is 60 pries tai zemiau
    virs = (c > out["sma50"]).astype(int)
    out["virs3"] = virs.rolling(B_VIRS_SMA50_DIENOS).sum()
    out["zemiau60"] = (1 - virs).shift(B_VIRS_SMA50_DIENOS).rolling(60).sum()
    out["istempimas"] = c / out["sma50"]

    # Isejimui
    out["O1"] = out["O"].shift(-1)
    for n in (1, 3, 5, 10, 20):
        out[f"C{n}"] = c.shift(-n)
    return out


def trendo_busena(mom_rangas, virs_sma50, virs_sma200, sekt_trend):
    """Viena reiksme: STIPRUS / NEUTRALUS / SILPNAS."""
    if mom_rangas is None or mom_rangas != mom_rangas:
        return "NEUTRALUS"
    if mom_rangas >= TRENDAS_STIPRUS_MOM and virs_sma50 and sekt_trend:
        return "STIPRUS"
    if mom_rangas <= TRENDAS_SILPNAS_MOM or not virs_sma200:
        return "SILPNAS"
    return "NEUTRALUS"


def zymek_A(df):
    """A: dip'as. Visos salygos kartu."""
    salygos = (
        (df["ret"] >= DIP_MIN) & (df["ret"] <= DIP_MAX)
        & (df["apyv_sant"] <= DIP_APYV_MAX)
        & (df["raud_is_eiles"] <= DIP_RAUD_MAX)
        & (df["C"] > df["zem20"])
        & (df["rezimas"])
    )
    if "dienos_iki_ataskaitos" in df:
        di = df["dienos_iki_ataskaitos"]
        blogas = di.notna() & (di >= ATASKAITOS_LANGAS[0]) & (di <= ATASKAITOS_LANGAS[1])
        salygos &= ~blogas

    df["A_dip"] = salygos
    df["A1"] = df["A_dip"]
    df["A2"] = df["A_dip"] & (df["trendas"] == "STIPRUS")
    df["A3"] = df["A_dip"] & (df["trendas"] == "NEUTRALUS")
    df["A4"] = df["A_dip"] & (df["trendas"] == "SILPNAS")
    return df


def zymek_B(df):
    """B: prasidedantis trendas."""
    bendra = (
        (df["virs3"] >= B_VIRS_SMA50_DIENOS)
        & (df["zemiau60"] >= B_PRIES_TAI_ZEMIAU)
        & (df["apyv_sant"] >= B_APYV_MIN)
        & (df["istempimas"] <= B_MAX_ISTEMPIMAS)
        & (df["rezimas"])
    )
    df["B1"] = bendra & df["sekt_trend"]
    df["B2"] = bendra
    return df


def top5(kand, rikiavimo_stulpelis, didejant=True, n=TOP_N,
         max_sektoriuje=MAX_IS_SEKTORIAUS):
    """Isrenka iki n kandidatu, ne daugiau max_sektoriuje is vieno sektoriaus."""
    if kand.empty:
        return kand
    k = kand.sort_values(rikiavimo_stulpelis, ascending=didejant)
    imti, per_sekt = [], {}
    for idx, eil in k.iterrows():
        s = eil.get("sekt", "kita")
        if per_sekt.get(s, 0) >= max_sektoriuje:
            continue
        imti.append(idx)
        per_sekt[s] = per_sekt.get(s, 0) + 1
        if len(imti) >= n:
            break
    return k.loc[imti]


def sudek_top5(diena_df):
    """Pagrindine rangavimo taisykle — ta pati backteste ir skeneryje.

    Eile: A2 (dip'as stipriame trende) -> B1 (prasidedantis trendas, iki 2)
          -> A3 (dip'as neutraliame) papildymui. A4 niekada.
    """
    dalys = []
    a2 = top5(diena_df[diena_df["A2"]], "z20_sekt", didejant=True)
    dalys.append(a2.assign(setup="DIP_TRENDE"))

    liko = TOP_N - len(a2)
    if liko > 0 and "B1" in diena_df:
        b1 = top5(diena_df[diena_df["B1"] & ~diena_df.index.isin(a2.index)],
                  "mom_3m", didejant=False, n=min(2, liko))
        if len(b1):
            dalys.append(b1.assign(setup="TRENDO_PRADZIA"))
            liko -= len(b1)

    if liko > 0:
        jau = pd.concat(dalys).index if dalys else []
        a3 = top5(diena_df[diena_df["A3"] & ~diena_df.index.isin(jau)],
                  "z20_sekt", didejant=True, n=liko)
        if len(a3):
            dalys.append(a3.assign(setup="DIP"))

    return pd.concat(dalys) if dalys else diena_df.head(0).assign(setup="")
