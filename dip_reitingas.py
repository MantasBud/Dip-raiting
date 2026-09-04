#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dip reitingas — intraday kritimo pirkimo skeneris.

Vienkartinis paleidimas:
    pip install yfinance pandas numpy tzdata
    python dip_reitingas.py

Nuolatinis atnaujinimas (paleidi kartą, veikia savaime kas 5 min.):
    python dip_reitingas.py --loop

Paprasčiausia — dukart spausk Atnaujinti_MacOS.command arba Atnaujinti_Windows.bat,
jie patys įdiegia bibliotekas ir paleidžia --loop režimu.

Rezultatas: surikiuotas sąrašas terminale + HTML ataskaita, kuri atsidaro naršyklėje
ir --loop režime pati atsinaujina. Viršuje aukščiausias reitingas, apačioje žemiausias.
"""

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
import webbrowser
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo("Europe/Berlin")       # birzos laikas
    _DTZ = ZoneInfo("Europe/Vilnius")     # tavo laikas ekrane
except Exception:
    _TZ = None
    _DTZ = None

# ----------------------------- NUSTATYMAI -----------------------------

TARGET_PCT = 3.0        # tikslinis pelnas procentais
ACCOUNT = 18000.0       # sąskaitos dydis, EUR
RISK_PCT = 1.0          # rizika vienam sandoriui, % nuo sąskaitos
MAX_POSITION_PCT = 100.0  # daugiausia % portfelio i viena pozicija (100 = visas)
# "full" = perki uz visa MAX_POSITION_PCT dali, rizika tokia, kokia iseina pagal stop
# "risk" = kiekis skaiciuojamas taip, kad stop kainuotu lygiai RISK_PCT portfelio
SIZING_MODE = "full"
FEE_PER_TRADE = 2.0     # brokerio mokestis vienam sandoriui (pirkimas ARBA pardavimas), EUR

# Stop turi buti UZ triuksmo ribu, kitaip ji ismus atsitiktinis svyravimas.
# Backtestas parode: ankstus stop'ai (0.35 x ATR) buvo pagrindine nuostoliu priezastis.
STOP_ATR_MULT = 0.55    # stop atstumas = tiek kartu dienos ATR
STOP_MIN_PCT = 0.8      # bet ne arciau nei tiek procentu

# Prekybos laikas (Europe/Berlin). Sesija 9:00-17:30, po jos - Tradegate/LS iki 22:00.
SESSION_OPEN_MIN = 9 * 60
SESSION_CLOSE_MIN = 17 * 60 + 30
EXTENDED_TRADING = True      # ar prekiauji ir po pagrindinės sesijos
EXTENDED_CLOSE_MIN = 22 * 60
EXTENDED_WEIGHT = 0.45       # po sesijos judesiai silpnesni, todėl laikas sveria mažiau

# Likvidumas - liberalios ribos, taikomos tik realiai problemiškiems atvejams
MAX_POS_OF_TURNOVER_PCT = 2.0    # pozicija kaip % dienos apyvartos
MIN_DAILY_TURNOVER_EUR = 2_000_000
OPEN_BROWSER = True     # ar automatiškai atidaryti HTML ataskaitą
LOOP_INTERVAL_SEC = 300     # kas kiek atsinaujina --loop režime (biržos valandomis)
LOOP_INTERVAL_OFF_SEC = 1800  # kas kiek tikrina ne prekybos metu (kad netrukdytų Yahoo)

WATCHLIST = [
    ("ADYEN", "ADYEN.AS", "Adyen NV"),
    ("AMD",   "AMD.DE",   "Advanced Micro Devices"),
    ("ASM",   "ASM.AS",   "ASM International"),
    ("ASML",  "ASML.AS",  "ASML Holding NV"),
    ("BESI",  "BESI.AS",  "BE Semiconductor"),
    ("CAP",   "CAP.PA",   "Capgemini SE"),
    ("ENR",   "ENR.DE",   "Siemens Energy AG"),
    ("IFX",   "IFX.DE",   "Infineon Technologies"),
    ("KER",   "KER.PA",   "Kering SA"),
    ("LR",    "LR.PA",    "Legrand SA"),
    ("MC",    "MC.PA",    "LVMH Moet Hennessy"),
    ("NVD",   "NVD.DE",   "Nvidia Corp"),
    ("PRX",   "PRX.AS",   "Prosus NV"),
    ("PTX",   "PTX.DE",   "Palantir Technologies"),
    ("RHM",   "RHM.DE",   "Rheinmetall AG"),
    ("RMS",   "RMS.PA",   "Hermes International"),
    ("SAP",   "SAP.DE",   "SAP SE"),
    ("SIE",   "SIE.DE",   "Siemens AG"),
    ("YDX",   "YDX.DE",   "Nebius Group NV"),
]

MARKET_INDEX = "^STOXX50E"   # rinkos kryptis

# Valiuta pagal biržos galūnę. Portfelis laikomas ACCOUNT_CURRENCY valiuta.
ACCOUNT_CURRENCY = "EUR"

# Laikymo horizontas valandomis. Tikslas turi buti pasiektas per si laika.
#   1-2  = "triuksmo" gaudymas per kelias valandas
#   8    = visa prekybos diena
#   16   = laikymas iki kitos dienos uzdarymo (numatyta)
HOLD_HOURS = 16.0

# Kiek akcija gali buti pakilusi siandien, kad dar laikytume tai atsigavimu, o ne
# jau ivykusiu suoliu. Virs sios ribos nuolaidos nebera.
MAX_RECOVERY_GAIN = 3.0
CURRENCY_BY_SUFFIX = {
    "DE": ("EUR", "\u20ac"), "AS": ("EUR", "\u20ac"), "PA": ("EUR", "\u20ac"),
    "MI": ("EUR", "\u20ac"), "MC": ("EUR", "\u20ac"), "BR": ("EUR", "\u20ac"),
    "LS": ("EUR", "\u20ac"), "VI": ("EUR", "\u20ac"), "HE": ("EUR", "\u20ac"),
    "IR": ("EUR", "\u20ac"), "F":  ("EUR", "\u20ac"),
    "L":  ("GBP", "\u00a3"), "SW": ("CHF", "CHF "), "ST": ("SEK", "kr "),
    "CO": ("DKK", "kr "), "OL": ("NOK", "kr "), "TO": ("CAD", "C$"),
}


def currency_of(sym):
    """Valiuta pagal Yahoo simbolio galūnę. Be galūnės - JAV birža, USD."""
    if "." in sym:
        suffix = sym.rsplit(".", 1)[1].upper()
        return CURRENCY_BY_SUFFIX.get(suffix, ("?", ""))
    return ("USD", "$")

CRITERIA = [
    ("dip",      "Kritimo gylis",            16),
    ("stab",     "Ar kritimas sustojo",      15),
    ("multiday", "Vienadienis ar tęstinis",  12),
    ("room",     "Vieta iki pasipriešinimo", 12),
    ("atr",      "Judrumas (ATR)",           12),
    ("rsi",      "RSI (5 min)",               9),
    ("vwap",     "Padėtis prieš VWAP",        8),
    ("rvol",     "Apyvarta (RVOL)",           7),
    ("trend",    "Trendas (20/50 SMA)",       7),
    ("support",  "Atstumas iki atramos",      2),
]

# Sektoriai — skaičiuojami iš paties sąrašo, be papildomų atsisiuntimų
SECTORS = {
    "ASML.AS": "puslaidininkiai", "ASM.AS": "puslaidininkiai", "BESI.AS": "puslaidininkiai",
    "IFX.DE": "puslaidininkiai", "AMD.DE": "puslaidininkiai", "NVD.DE": "puslaidininkiai",
    "MC.PA": "prabangos prekės", "RMS.PA": "prabangos prekės", "KER.PA": "prabangos prekės",
    "RHM.DE": "pramonė ir gynyba", "SIE.DE": "pramonė ir gynyba",
    "ENR.DE": "pramonė ir gynyba", "LR.PA": "pramonė ir gynyba",
    "SAP.DE": "programinė įranga", "ADYEN.AS": "programinė įranga", "PRX.AS": "programinė įranga",
    "PTX.DE": "programinė įranga", "YDX.DE": "programinė įranga", "CAP.PA": "programinė įranga",
}

# ----------------------------- SKAIČIAVIMAI -----------------------------


def num(v):
    """Skaicius arba None. Apsaugo nuo tusciu reiksmiu skaiciavimuose."""
    try:
        if v is None or v != v:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def curve(x, pts):
    """Tiesinė interpoliacija tarp kontrolinių taškų. None -> neutralus 50."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return 50.0
    if x <= pts[0][0]:
        return float(pts[0][1])
    if x >= pts[-1][0]:
        return float(pts[-1][1])
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        if x1 <= x <= x2:
            return float(y1 + (x - x1) / (x2 - x1) * (y2 - y1))
    return 50.0


def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50)


def atr_pct(daily, n=14):
    daily = daily.dropna(subset=["High", "Low", "Close"])
    if len(daily) < 3:
        return None
    h, l, c = daily["High"], daily["Low"], daily["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    last = c.iloc[-1]
    if not (isinstance(a, float) or hasattr(a, "__float__")) or not last:
        return None
    val = float(a) / float(last) * 100
    return val if math.isfinite(val) else None


def levels(daily, price):
    """Atramos ir pasipriešinimo kandidatai: pivotai + 20 d. swing lygiai."""
    prev = daily.iloc[-2] if len(daily) > 1 else daily.iloc[-1]
    p = (float(prev["High"]) + float(prev["Low"]) + float(prev["Close"])) / 3
    s1, r1 = 2 * p - float(prev["High"]), 2 * p - float(prev["Low"])
    lo20 = float(daily["Low"].tail(20).min())
    hi20 = float(daily["High"].tail(20).max())

    sup = [x for x in (s1, lo20, float(prev["Low"]), p) if x and x < price * 0.999]
    res = [x for x in (r1, hi20, float(prev["High"]), p) if x and x > price * 1.001]
    return (max(sup) if sup else None), (min(res) if res else None)


def relative_volume(intraday, today_mask):
    """Šiandienos apyvarta prieš tų pačių valandų apyvartą ankstesnėmis dienomis."""
    today = intraday[today_mask]
    if today.empty:
        return None
    bars = len(today)
    today_vol = float(today["Volume"].sum())
    prev = intraday[~today_mask]
    if prev.empty:
        return None
    per_day = []
    for _, grp in prev.groupby(prev.index.date):
        if len(grp) >= bars:
            per_day.append(float(grp["Volume"].iloc[:bars].sum()))
    if not per_day:
        return None
    base = float(np.median(per_day))
    return today_vol / base if base > 0 else None


def time_budget(now_min=None):
    """Kiek efektyvaus prekybos laiko liko iki uždarymo (minutėmis ir dalimi sesijos)."""
    if now_min is None:
        p = new_intl_now()
        if p is None:
            return None
        now_min = p

    session_len = SESSION_CLOSE_MIN - SESSION_OPEN_MIN
    regular_left = max(0, SESSION_CLOSE_MIN - now_min)
    ext_left = 0
    if EXTENDED_TRADING:
        start = max(now_min, SESSION_CLOSE_MIN)
        ext_left = max(0, EXTENDED_CLOSE_MIN - start) * EXTENDED_WEIGHT

    effective = regular_left + ext_left
    return dict(now_min=now_min, regular_left=regular_left,
                ext_left=ext_left / EXTENDED_WEIGHT if EXTENDED_WEIGHT else 0,
                effective=effective, frac=effective / session_len if session_len else 0,
                after_hours=now_min >= SESSION_CLOSE_MIN)


def new_intl_now():
    try:
        now = datetime.now(_TZ) if _TZ else datetime.now()
        return now.hour * 60 + now.minute
    except Exception:
        return None


def short_momentum(today_bars, bph=12):
    """Kryptis per 1 ir 3 valandas: ar kaina dar krinta, ar jau atsispyre.

    Valandos, o ne minutes, nes 15 min. atkarpoje matosi tik triuksmas.
    Anksti sesijoje, kai bary dar mazai, skaiciuojama nuo atidarymo ir tai pazymima.
    """
    if today_bars is None or len(today_bars) < 4:
        return dict(m1h=None, m3h=None, pos1h=None, span_h=None, partial=True)

    c = today_bars["Close"]
    last = float(c.iloc[-1])

    def trend(bars_back):
        """Krypties nuolydis per atkarpa, ivertinant VISUS barus, ne tik du galus.
        Taip vienas atsitiktinis suolis nebeiskraipo rodiklio."""
        i = max(0, len(c) - 1 - bars_back)
        seg = c.iloc[i:].astype(float).to_numpy()
        n = len(seg)
        if n < 3:
            return (0.0, 0.0)
        x = np.arange(n)
        slope = float(np.polyfit(x, seg, 1)[0])       # kainos pokytis per bara
        base = float(seg.mean())
        total = slope * (n - 1) / base * 100 if base else 0.0
        return (total, (n - 1) / bph)

    m1h, span1 = trend(bph)          # 1 val.
    m3h, span3 = trend(bph * 3)      # 3 val.

    win = today_bars.tail(bph + 1)
    lo, hi = float(win["Low"].min()), float(win["High"].max())
    pos1h = (last - lo) / (hi - lo) * 100 if hi > lo else None

    return dict(m1h=m1h, m3h=m3h, pos1h=pos1h, span_h=span3, partial=span1 < 0.9)


def intraday_vol(today_bars):
    """Tipinis 5 min. baro diapazonas procentais — realus intraday judrumo matas."""
    if today_bars is None or len(today_bars) < 6:
        return None
    rng = (today_bars["High"] - today_bars["Low"]) / today_bars["Close"] * 100
    v = float(rng.median())
    return v if math.isfinite(v) and v > 0 else None


def expected_move(vol_bar, hours, bph=12):
    """Tikėtinas kainos judesys per N valandų. Svyravimas auga ~sqrt(laiko).

    bph = kiek baru telpa i valanda (12 penkiaminuciu, 1 valandinis)."""
    if not vol_bar:
        return None
    bars = max(1.0, hours * bph)
    return vol_bar * math.sqrt(bars)


def overnight_gap(daily, n=60):
    """Tipinis nakties suolis: |atidarymas - vakarykstis uzdarymas| procentais.
    Svarbu, nes laikant per naktį stop nesuveikia — parduosi ten, kur atsidarys."""
    d = daily.dropna(subset=["Open", "Close"]).tail(n + 1)
    if len(d) < 10:
        return None
    gaps = (d["Open"] - d["Close"].shift(1)).abs() / d["Close"].shift(1) * 100
    gaps = gaps.dropna()
    if gaps.empty:
        return None
    v = float(gaps.median())
    return v if math.isfinite(v) else None


def multiday_context(daily, price):
    """Ar tai vienos dienos kritimas, ar tęstinis kelių dienų slydimas."""
    closes = daily["Close"].tail(6).tolist()
    highs = daily["High"].tail(5).tolist()
    if len(closes) < 4:
        return dict(down_days=0, dd5=None, chg3d=None)

    # kiek dienų iš eilės uždaryta žemyn (neįskaitant šiandienos, nes ji dar nebaigta)
    down_days = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes[i] < closes[i - 1]:
            down_days += 1
        else:
            break

    hi5 = max(highs) if highs else None
    dd5 = (hi5 - price) / hi5 * 100 if hi5 else None          # kritimas nuo 5 d. maksimumo
    chg3d = (price - closes[-4]) / closes[-4] * 100 if len(closes) >= 4 else None
    return dict(down_days=down_days, dd5=dd5, chg3d=chg3d)


def score_stock(d, target=TARGET_PCT, market="neutral", sector_chg=None, tb=None):
    """d — surinktų rodiklių žodynas. Grąžina balą, dedamąsias, planą, įspėjimus."""
    price = num(d.get("price"))
    if not price or price <= 0:
        return dict(empty=True, score=0.0, grade="?", tradeable=False, blocking=[],
                    parts={k: 0.0 for k, _, _ in CRITERIA}, flags=[],
                    stop=0.0, tp=0.0, rr=0.0, shares=0, pos_value=0.0,
                    gross=0.0, net=0.0, real_risk=0.0, setup="nėra duomenų")

    high, low = d.get("dayHigh"), d.get("dayLow")
    # Atrama: dienos sandoriui stop dedamas po šios dienos dugnu
    support = d.get("sup_intra") or d.get("support") or low

    # Pasipriešinimas dviem sluoksniais: artimiausios lubos (dažnai dienos maksimumas,
    # kuris per dieną pramušamas) ir tolimesnės (pivotas, 20 d. viršūnė). Sandoris
    # beprasmis tik tada, kai net tolimesnės lubos arčiau nei tikslas.
    res_list = d.get("res_list") or []
    res_near = res_list[0] if res_list else (d.get("resistance") or high)
    res_far = res_list[-1] if len(res_list) > 1 else res_near
    # Balui svarbi artimiausia kliūtis (ji realiai stabdo judesį), o sandoris
    # blokuojamas tik kai net tolimiausios lubos arčiau nei tikslas.
    resistance = res_near
    vwap = d.get("vwap")
    a_pct = d.get("atrPct")

    dip = (high - price) / high * 100 if high else None
    rng = (price - low) / (high - low) * 100 if high and low and high > low else None
    sup_d = (price - support) / price * 100 if support else None
    room = (resistance - price) / price * 100 if resistance else None
    room_far = (res_far - price) / price * 100 if res_far else None
    vw_d = (price - vwap) / vwap * 100 if vwap else None
    rv = d.get("rvol")

    sma20, sma50 = d.get("sma20"), d.get("sma50")
    trend = 60.0
    if sma20 and sma50:
        a20 = (price - sma20) / sma20 * 100
        a50 = (price - sma50) / sma50 * 100
        if a50 > 0 and a20 > 0:
            trend = 100.0
        elif a50 > 0 and a20 > -3:
            trend = 85.0
        elif a50 > 0:
            trend = 65.0
        elif a50 > -4:
            trend = 40.0
        else:
            trend = 18.0

    # Vienadienis kritimas ar tęstinis slydimas
    down_days = d.get("down_days", 0) or 0
    dd5 = d.get("dd5")
    chg3d = d.get("chg3d")
    # --- Du skirtingi scenarijai ---
    # A) Kritimas: akcija šiandien nukrito nuo dienos maksimumo, perkam nuolaidą
    # B) Atsigavimas: akcija buvo nukritusi kelias dienas, šiandien kyla nuo dugno
    #    ir dar nepasiekė ankstesnės viršūnės — dar yra kur augti
    day_chg = d.get("day_chg")
    # Atsigavimas galioja tik kol nuolaida dar yra. Jei akcija jau pašoko
    # (MAX_RECOVERY_GAIN ar daugiau) arba iki 5 d. viršūnės liko mažiau nei tikslas,
    # pirkimas vyktų jau po įvykusio judesio.
    recovering = (day_chg is not None and 0.2 < day_chg <= MAX_RECOVERY_GAIN
                  and dd5 is not None and max(1.0, target) <= dd5 <= 9.0
                  and rng is not None and 55 <= rng <= 92)
    overextended = (day_chg is not None and day_chg > MAX_RECOVERY_GAIN
                    and dip is not None and dip < 1.0)

    dip_part = curve(dip, [(0, 5), (0.5, 35), (1.2, 85), (1.8, 100), (4, 100),
                           (6, 55), (9, 18), (15, 5)])
    if recovering:
        # Čia "nuolaida" matuojama ne nuo šios dienos maksimumo, o nuo 5 d. viršūnės
        rec_part = curve(dd5, [(0.5, 30), (1.5, 80), (3, 100), (5, 92), (8, 55), (12, 20)])
        dip_part = max(dip_part, rec_part)
        setup = "atsigavimas"
    elif overextended:
        dip_part = min(dip_part, 12.0)
        setup = "jau pakilusi"
    else:
        setup = "kritimas"

    # dd5 = kritimas nuo 5 d. maksimumo. Sveikas dip: 1.5-4%. Tęstinis slydimas: 7%+
    # Tikėtinas judesys per laikymo horizontą prieš tikslą
    exp_mv = d.get("exp_move")
    if exp_mv is None and a_pct:
        exp_mv = a_pct * math.sqrt(min(HOLD_HOURS, 8.5) / 8.5)   # atsarginis variantas
    move_ratio = exp_mv / target if (exp_mv and target) else None

    multiday_part = curve(dd5, [(0, 25), (1, 60), (2, 95), (4, 100), (6, 65), (9, 30), (14, 8)])
    if not recovering:
        if down_days >= 3:
            multiday_part *= 0.45
        elif down_days == 2:
            multiday_part *= 0.75
    else:
        # Kritimas jau baigėsi ir kaina kyla — gylis tampa privalumu, ne rizika
        multiday_part = max(multiday_part, 75.0)
    if chg3d is not None and chg3d < -6:
        multiday_part *= 0.7

    # --- Ar kritimas jau sustojo? Krentantis peilis atrodo taip pat kaip dip,
    # skiriasi tik tuo, kad jis vis dar krinta. ---
    m1h, m3h, pos1h = d.get("m1h"), d.get("m3h"), d.get("pos1h")
    stab = curve(m1h, [(-2.0, 8), (-0.8, 25), (-0.2, 55), (0.1, 85),
                       (0.6, 100), (1.5, 90), (3.0, 65)])
    knife = False
    if m3h is not None and m1h is not None:
        if m3h < -0.8 and m1h > 0.1:
            stab = min(100.0, stab * 1.12)      # krito 3 val. ir per pastarąją atsispyrė
        elif m3h < -0.8 and m1h < -0.2:
            stab *= 0.55                        # kryptis žemyn nesikeičia
            knife = True
    if pos1h is not None and pos1h < 20:
        stab *= 0.85                            # laikosi prie valandos dugno

    parts = {
        "dip":  dip_part,
        "stab": stab,
        "multiday": multiday_part,
        "room": curve(None if room is None else room / target,
                      [(0.3, 5), (1, 45), (1.5, 70), (2, 90), (3, 100), (6, 95)]),
        # Judrumas matuojamas tavo laikymo horizonte: ar per HOLD_HOURS realiai
        # tikėtinas judesys pasiekia tikslą
        "atr":  curve(None if move_ratio is None else move_ratio,
                      [(0.3, 5), (0.6, 30), (0.9, 65), (1.2, 92), (1.8, 100), (3.5, 90)]),
        # Kritimo scenarijuje ieškom išpardavimo zonos, atsigavimo - jau pakilusio,
        # bet dar neperpirkto RSI
        "rsi":  curve(d.get("rsi"),
                      [(25, 20), (40, 60), (50, 90), (60, 100), (70, 70), (80, 25)]
                      if recovering else
                      [(10, 25), (20, 55), (28, 85), (35, 100), (45, 85), (55, 55), (65, 30), (80, 10)]),
        "vwap": curve(vw_d, [(-4, 20), (-2, 45), (-0.8, 85), (-0.2, 100), (0.3, 90), (1.5, 60), (3, 35), (6, 15)]),
        "rvol": curve(rv,   [(0.3, 15), (0.7, 45), (1, 70), (1.4, 95), (2.5, 100), (4, 80), (7, 55), (12, 35)]),
        "trend": trend,
        "support": 15.0 if (sup_d is not None and sup_d < 0) else
                   curve(sup_d, [(0, 95), (0.3, 100), (1, 85), (2, 55), (3.5, 30), (6, 10)]),
    }

    base = sum(parts[k] * w / 100 for k, _, w in CRITERIA)

    # Stop dedamas pagal akcijos svyravimą, o ne pagal atramos artumą. Arti atramos
    # esanti kaina yra geras ĮĖJIMAS, bet tai nereiškia, kad stop gali būti ankštas:
    # judrioje akcijoje 1% stop yra triukšmo lygyje ir bus išmuštas.
    min_stop = max(STOP_MIN_PCT, STOP_ATR_MULT * (a_pct or 2.0))
    stop = price * (1 - min_stop / 100)
    if support and support < price:
        sup_stop = support * 0.997
        stop = min(stop, sup_stop)      # jei atrama dar žemiau, stop dedam po ja
    if (price - stop) / price * 100 > target * 1.8:
        stop = price * (1 - target * 1.8 / 100)
    tp = price * (1 + target / 100)
    rr = (tp - price) / (price - stop) if price > stop else 0.0
    risk_cash = ACCOUNT * RISK_PCT / 100
    max_shares = int((ACCOUNT * MAX_POSITION_PCT / 100) / price) if price > 0 else 0
    risk_shares = int(risk_cash / (price - stop)) if price > stop else 0

    if SIZING_MODE == "full":
        shares = max_shares          # perkam visa numatyta dali, rizika = kiek iseina
        capped = False
    else:
        shares = min(risk_shares, max_shares)
        capped = risk_shares > max_shares

    pos_value = shares * price
    gross = pos_value * target / 100
    net = gross - 2 * FEE_PER_TRADE      # mokestis perkant ir parduodant
    real_risk = shares * (price - stop)

    flags, mult = [], 1.0
    if d.get("earnings"):
        mult *= 0.35
        flags.append(("stop", "Ataskaita per 2 dienas — kaina šoks bet kuria kryptimi"))
    if market == "bear":
        mult *= 0.75
        flags.append(("stop", "Rinka krenta. Per 2 metus tokiomis dienomis net geriausiai "
                              "įvertinti įėjimai vidutiniškai prarado 0,35%, o vidutinis "
                              "sandoris 0,45% — atrankos pranašumo tam nepakanka"))
    elif market == "bull":
        mult *= 1.05
    if room_far is not None and room_far < target:
        flags.append(("stop", f"Net iki tolimesnių lubų tik {room_far:.1f}% — "
                              f"{target}% tikslas netelpa niekur"))
    elif room is not None and room < target:
        flags.append(("warn", f"Kelyje kliūtis ({room:.1f}% aukščiau) — "
                              f"jį reikės pramušti, kad tikslas būtų pasiektas"))
    if a_pct is None:
        flags.append(("warn", "ATR nepavyko suskaičiuoti — judrumo kriterijus neįvertintas, "
                              "balas mažiau patikimas"))
    if exp_mv and exp_mv < target * 0.9:
        flags.append(("warn", f"Per {HOLD_HOURS:.1f} val. tikėtinas judesys ~{exp_mv:.1f}%, "
                              f"o tikslas {target}% — realistiškesnis tikslas šiai akcijai "
                              f"būtų ~{exp_mv:.1f}% arba ilgesnis laikymas"))
    # --- Laiko biudžetas: ar likusio laiko realiai užtenka tikslui pasiekti? ---
    # Kainos svyravimas auga proporcingai laiko šaknims, todėl tikėtinas likęs
    # judesys = dienos ATR * sqrt(likusi sesijos dalis).
    if tb and a_pct:
        exp_range = a_pct * math.sqrt(max(tb["frac"], 0.01))
        ratio = exp_range / target if target else 0
        hrs = tb["effective"] / 60
        if ratio < 0.7:
            if HOLD_HOURS > 8:
                flags.append(("info", f"Iki uždarymo liko ~{hrs:.1f} val. — tikslas greičiausiai "
                                      f"bus pasiektas rytoj, pozicija liks per naktį"))
            else:
                mult *= 0.85
                flags.append(("warn", f"Liko ~{hrs:.1f} val. efektyvios prekybos — tikėtinas "
                                      f"judesys ~{exp_range:.1f}% nesiekia {target}% tikslo"))
        elif ratio < 1.1:
            flags.append(("info", f"Liko ~{hrs:.1f} val. — tikslas pasiekiamas, bet be atsargos "
                                  f"(tikėtinas judesys ~{exp_range:.1f}%)"))
        if tb["after_hours"]:
            flags.append(("info", "Pagrindinė sesija baigta — po sesijos prekyboje spread'as "
                                  "platesnis, naudok limit pavedimus"))

    # --- Nakties šuolio rizika: laikant per naktį stop neveikia ---
    gap = d.get("gap")
    if HOLD_HOURS > 8 and gap and shares > 0:
        stop_dist = (price - stop) / price * 100
        gap_loss = pos_value * gap / 100
        if gap > stop_dist:
            flags.append(("warn", f"Laikant per naktį stop neapsaugo: tipinis šuolis šioje "
                                  f"akcijoje {gap:.1f}%, o stop tik {stop_dist:.1f}% žemiau — "
                                  f"nepalankus atidarymas kainuotų ~{gap_loss:.0f} EUR"))
        else:
            flags.append(("info", f"Tipinis nakties šuolis {gap:.1f}% — telpa į stop atstumą "
                                  f"({stop_dist:.1f}%)"))

    # --- Likvidumas: ar pozicija realiai išpildoma ---
    avg_vol = d.get("avgVolume")
    if avg_vol and shares > 0:
        turnover = avg_vol * price
        share_pct = pos_value / turnover * 100 if turnover else 0
        if turnover < MIN_DAILY_TURNOVER_EUR:
            flags.append(("warn", f"Plona akcija: dienos apyvarta ~{turnover/1e6:.1f} mln. EUR — "
                                  f"įėjimas ir išėjimas gali kainuoti brangiau nei mokesčiai"))
        elif share_pct > MAX_POS_OF_TURNOVER_PCT:
            flags.append(("warn", f"Pozicija sudaro {share_pct:.1f}% dienos apyvartos — "
                                  f"gali tekti pildyti dalimis"))

    if d.get("cur") and d["cur"] != ACCOUNT_CURRENCY:
        flags.append(("warn", f"Ši akcija kotiruojama {d['cur']}, o portfelis "
                              f"{ACCOUNT_CURRENCY} — pozicijos dydis ir pelnas rodomi "
                              f"{d['cur']}, neperskaičiuoti į {ACCOUNT_CURRENCY}"))
    if shares > 0:
        risk_pct_real = real_risk / ACCOUNT * 100 if ACCOUNT else 0
        if risk_pct_real > 2.5:
            flags.append(("warn", f"Vienas nesėkmingas sandoris kainuotų {risk_pct_real:.1f}% portfelio "
                                  f"— tiek pat, kiek duotų {risk_pct_real/2:.0f} sėkmingi"))
    if shares > 0 and gross > 0 and net < gross * 0.75:
        flags.append(("warn", f"Mokesčiai suvalgo dalį pelno: bruto {gross:.0f} EUR, "
                              f"neto ~{net:.0f} EUR"))
    # R:R nebeduoda premijos: aukštas R:R pasiekiamas ankštu stop'u, o backtestas
    # parodė, kad būtent ankšti stop'ai ir generuoja nuostolius. Lieka tik bauda,
    # kai santykis tikrai blogas.
    if 0 < rr < 1.0:
        mult *= 0.8
        flags.append(("warn", f"Rizika/nauda {rr:.2f} — rizikuoji daugiau nei sieki"))
    if dip is not None and dip > 8:
        flags.append(("warn", "Kritimas gilus — gali būti krentantis peilis"))
    if overextended:
        flags.append(("stop", f"Šiandien jau pakilusi {day_chg:+.1f}% ir prekiauja prie pat "
                              f"dienos viršūnės — nuolaidos nebėra, tai pirkimas po judesio"))
    elif recovering:
        flags.append(("info", f"Atsigavimo faze: nukritusi {dd5:.1f}% nuo 5 d. viršūnės, "
                              f"šiandien {day_chg:+.1f}% ir laikosi dienos viršuje"))
    elif down_days >= 3:
        mult *= 0.8
        flags.append(("stop", f"{down_days} kritimo dienos iš eilės — tai ne vienadienis dip, o kryptis"))
    elif down_days == 2:
        flags.append(("warn", "Antra kritimo diena iš eilės — palauk stabilizacijos ženklo"))
    if dd5 is not None and dd5 > 7:
        flags.append(("warn", f"Nuo 5 d. maksimumo nukritusi {dd5:.1f}% — kritimas prasidėjo ne šiandien"))
    # --- Triukšmas ar trendinis kritimas? Lyginam akciją su jos sektoriumi ---
    day_chg_v = d.get("day_chg")
    if sector_chg is not None and day_chg_v is not None:
        rel = day_chg_v - sector_chg          # kiek akcija atsilieka nuo saviškių
        if sector_chg < -1.2 and rel > -0.6:
            mult *= 0.85
            flags.append(("warn", f"Krenta visas sektorius ({sector_chg:+.1f}%), o akcija "
                                  f"juda kartu ({day_chg_v:+.1f}%) — tai trendinis judesys, "
                                  f"ne šios akcijos triukšmas"))
        elif rel < -2.5:
            mult *= 0.8
            flags.append(("warn", f"Akcija krinta {abs(rel):.1f} p. p. labiau nei sektorius — "
                                  f"toks atsilikimas dažniau reiškia naujieną, ne triukšmą"))
        elif -1.8 <= rel <= -0.3 and sector_chg > -1.0:
            flags.append(("info", f"Izoliuotas atsitraukimas: akcija {day_chg_v:+.1f}%, "
                                  f"sektorius {sector_chg:+.1f}% — būtent toks triukšmas, "
                                  f"kurio ieškai"))
    elif sector_chg is not None and sector_chg < -1.5:
        mult *= 0.85
        flags.append(("warn", f"Visas sektorius krenta ({sector_chg:+.1f}%)"))
    if rv is not None and rv < 0.7:
        flags.append(("warn", "Apyvarta mažesnė nei įprasta — atšokimas gali neįvykti"))
    if knife:
        mult *= 0.9
        flags.append(("warn", f"Kryptis vis dar žemyn: per 3 val. {m3h:+.1f}%, per pastarąją "
                              f"valandą {m1h:+.1f}% — dugno ženklo dar nėra"))
    elif m3h is not None and m1h is not None and m3h < -0.8 and m1h > 0.1:
        flags.append(("info", f"Kritimas sustojo: po {m3h:+.1f}% per 3 val. pastarąją valandą "
                              f"jau {m1h:+.1f}%"))
    if d.get("mom_partial") and m1h is not None:
        flags.append(("info", "Sesija dar trumpa — krypties rodikliai skaičiuoti nuo atidarymo"))
    if rng is not None and rng < 6:
        flags.append(("warn", "Kaina prie pat dienos dugno — atsigavimo ženklo dar nėra"))
    if sup_d is not None and sup_d < 0:
        flags.append(("stop", "Kaina žemiau atramos — atrama pralaužta"))

    score = max(0.0, min(100.0, base * mult))

    # Jei yra bent viena fatališka yda (tikslas netelpa, ataskaita, pralaužta atrama,
    # neigiama naujiena) — sandoris netinkamas, kad ir kokie geri kiti rodikliai.
    # Nepilni duomenys neturi atrodyti kaip vidutinis kandidatas
    key_fields = ["dayHigh", "dayLow", "atrPct", "vwap", "rsi", "rvol", "sma20", "vol5m"]
    missing = [k for k in key_fields if d.get(k) is None]
    if len(missing) >= 3:
        flags.append(("stop", f"Trūksta {len(missing)} iš {len(key_fields)} rodiklių "
                              f"({', '.join(missing[:4])}) — balas nepatikimas"))

    blocking = [t for lvl, t in flags if lvl == "stop"]
    if blocking:
        score = min(score, 45.0)

    tradeable = (not blocking) and rr >= 1.0 and (room_far is None or room_far >= target * 1.2)
    grade = "A" if score >= 78 else "B" if score >= 64 else "C" if score >= 50 else "D"

    return dict(score=score, grade=grade, tradeable=tradeable, blocking=blocking,
                setup=setup, recovering=recovering,
                parts=parts, flags=flags, dip=dip, rng=rng,
                room=room, sup_d=sup_d, vw_d=vw_d, stop=stop, tp=tp, rr=rr, shares=shares,
                down_days=down_days, dd5=dd5, chg3d=chg3d, sector_chg=sector_chg,
                pos_value=pos_value, gross=gross, net=net, real_risk=real_risk,
                exp_move=exp_mv, move_ratio=move_ratio)


# ----------------------------- DUOMENŲ SURINKIMAS -----------------------------


def flatten(df, symbol):
    """yfinance grąžina MultiIndex stulpelius, kai simbolių daugiau nei vienas."""
    if isinstance(df.columns, pd.MultiIndex):
        if symbol in df.columns.get_level_values(0):
            return df[symbol].dropna(how="all")
        if symbol in df.columns.get_level_values(-1):
            return df.xs(symbol, axis=1, level=-1).dropna(how="all")
    return df.dropna(how="all")


def market_bias(yf):
    """Rinkos rezimas is 5 dienu krypties ir padeties pries 20 d. vidurki.

    Backtestas (2 metai, 38 tukst. ijejimo tasku) parode, kad rezimas lemia
    rezultata ~10 kartu labiau nei akcijos atranka: krentancioje rinkoje
    vidutinis sandoris -0.45%, kylancioje +0.61%. Todel matuojame ji rimtai,
    o ne pagal vienos dienos pokyti.
    """
    try:
        idx = yf.download(MARKET_INDEX, period="3mo", interval="1d",
                          progress=False, auto_adjust=False)
        idx = flatten(idx, MARKET_INDEX).dropna(subset=["Close"])
        c = idx["Close"]
        if len(c) < 25:
            return "neutral"
        last = float(c.iloc[-1])
        chg5 = (last / float(c.iloc[-6]) - 1) * 100
        sma20 = float(c.tail(20).mean())
        above = last > sma20

        if chg5 < -1.0 or (not above and chg5 < 0):
            return "bear"
        if chg5 > 1.0 and above:
            return "bull"
        return "neutral"
    except Exception:
        return "neutral"


def earnings_soon(yf, symbol):
    try:
        t = yf.Ticker(symbol)
        cal = t.get_earnings_dates(limit=8)
        if cal is None or cal.empty:
            return False
        now = pd.Timestamp.now(tz=cal.index.tz)
        upcoming = cal[cal.index >= now]
        if upcoming.empty:
            return False
        return (upcoming.index[0] - now).days <= 2
    except Exception:
        return False


def collect(yf, symbols):
    """Vienu kreipimusi paimam visų akcijų 5 min ir dienos duomenis."""
    intraday = yf.download(symbols, period="10d", interval="5m", group_by="ticker",
                           progress=False, auto_adjust=False, threads=True)
    daily = yf.download(symbols, period="6mo", interval="1d", group_by="ticker",
                        progress=False, auto_adjust=False, threads=True)
    return intraday, daily


def build_row(yf, tag, sym, name, intraday_all, daily_all):
    intra = flatten(intraday_all, sym)
    daily = flatten(daily_all, sym)
    if intra.empty or daily.empty or "Close" not in intra:
        raise ValueError("nėra duomenų")

    intra = intra.dropna(subset=["Close"])
    daily = daily.dropna(subset=["Close", "High", "Low"])
    last_day = intra.index[-1].date()
    mask = pd.Series(intra.index.date == last_day, index=intra.index)
    today = intra[mask]
    if today.empty:
        raise ValueError("nėra šios dienos barų")

    price = float(today["Close"].iloc[-1])
    high = float(today["High"].max())
    low = float(today["Low"].min())

    tp_series = (today["High"] + today["Low"] + today["Close"]) / 3
    vol = today["Volume"].replace(0, np.nan)
    vwap = float((tp_series * vol).sum() / vol.sum()) if vol.sum() > 0 else None

    r = float(rsi(intra["Close"]).iloc[-1])
    a = atr_pct(daily)
    sup, res = levels(daily, price)
    rv = relative_volume(intra, mask)
    c = daily["Close"]
    sma20 = float(c.tail(20).mean())
    sma50 = float(c.tail(50).mean())
    ctx = multiday_context(daily, price)
    avg_vol = float(daily["Volume"].tail(20).mean())
    gap = overnight_gap(daily)
    v5 = intraday_vol(today)
    mom = short_momentum(today)
    # Valandos sandoriui svarbios šios dienos lubos, ne 20 d. swing lygiai
    res_intra = high if high > price * 1.001 else None
    sup_intra = low if low < price * 0.999 else None
    # Visos lubos virs kainos, nuo artimiausios: dienos max, pivotas, 20 d. virsune.
    # Dienos maksimumas dazniausiai pramusamas, todel jis - ispejimas, ne kliutis.
    cands = sorted(x for x in (res_intra, res, float(daily["High"].tail(20).max()))
                   if x and x > price * 1.001)
    prev_close = float(c.iloc[-2]) if len(c) > 1 else None
    day_chg = (price - prev_close) / prev_close * 100 if prev_close else None

    return dict(tag=tag, sym=sym, name=name, price=price, dayHigh=high, dayLow=low,
                vwap=vwap, rsi=r, atrPct=a, support=sup, resistance=res, rvol=rv,
                sma20=sma20, sma50=sma50, earnings=earnings_soon(yf, sym),
                sector=SECTORS.get(sym, "kita"), day_chg=day_chg, avgVolume=avg_vol,
                cur=currency_of(sym)[0], cur_sym=currency_of(sym)[1], gap=gap,
                vol5m=v5, res_intra=res_intra, sup_intra=sup_intra, res_list=cands,
                m1h=mom["m1h"], m3h=mom["m3h"], pos1h=mom["pos1h"],
                span_h=mom["span_h"], mom_partial=mom["partial"],
                exp_move=expected_move(v5, HOLD_HOURS),
                down_days=ctx["down_days"], dd5=ctx["dd5"], chg3d=ctx["chg3d"],
                asOf=str(intra.index[-1]))


# ----------------------------- ATASKAITA -----------------------------


def print_table(rows):
    print()
    print(f"{'#':>2}  {'AKCIJA':<7} {'BALAS':>6} {'':2} {'KAINA':>9} {'KRIT.':>7} "
          f"{'ATR':>6} {'RSI':>5} {'RVOL':>5} {'R:R':>5}")
    print("-" * 68)
    for i, (d, s) in enumerate(rows, 1):
        atr_txt = f"{d['atrPct']:>5.1f}%" if d.get("atrPct") else "   n/a"
        print(f"{i:>2}. {d['tag']:<7} {s['score']:>6.1f} {s['grade']:>2} "
              f"{d.get('cur_sym','')}{d['price']:>8.2f} {(s['dip'] or 0):>6.1f}% "
              f"{atr_txt} {d['rsi']:>5.0f} {(d['rvol'] or 0):>5.2f} {s['rr']:>5.2f}")
    best, bs = rows[0]
    print("-" * 68)
    if bs["score"] >= 50:
        print(f"\nGERIAUSIAS: {best['tag']} ({best['name']})")
        c = best.get('cur_sym', '')
        print(f"  Įėjimas {c}{best['price']:.2f} | Stop {c}{bs['stop']:.2f} | "
              f"Tikslas {c}{bs['tp']:.2f} | R:R {bs['rr']:.2f} | {bs['shares']} vnt. "
              f"| pozicija {c}{bs['pos_value']:,.0f}")
        for lvl, txt in bs["flags"]:
            print(f"  {'!!' if lvl == 'stop' else ' !'} {txt}")
    else:
        print("\nNė viena akcija nesurenka 50 balų — šiandien geriau praleisti.")
    print()


def market_overview(rows, market, sector_state, target):
    """Bendra dienos apžvalga tekstu — kokia diena, kur dėmesys, ko saugotis."""
    scores = [s["score"] for _, s in rows]
    strong = [r for r in rows if r[1]["score"] >= 64 and r[1].get("tradeable")]
    blocked = [r for r in rows if r[1].get("blocking")]
    weak = [r for r in rows if r[1]["score"] < 50]
    mkt = {"bull": "kylanti", "bear": "krentanti", "neutral": "šoninė"}[market]

    p = []
    if not strong:
        p.append(f"Rinka {mkt}. Nė viena iš {len(rows)} akcijų šiuo metu neatitinka visų sąlygų: "
                 f"reikia, kad tikslas tilptų iki pasipriešinimo, o rizika/nauda būtų bent 1,3. "
                 f"Tokia diena tinka praleisti — tai irgi sprendimas.")
        if blocked:
            p.append(f"{len(blocked)} akcijos turi lemiamą kliūtį (tikslas netelpa, artėja "
                     f"ataskaita ar pralaužta atrama), todėl jų balas apribotas.")
    else:
        best_sec = strong[0][0]["sector"]
        same = sum(1 for r in strong if r[0]["sector"] == best_sec)
        p.append(f"Rinka {mkt}. Iš {len(rows)} akcijų {len(strong)} atitinka kritimo pirkimo "
                 f"kriterijus bent gerai, {len(weak)} šiandien geriau nevertos dėmesio.")
        if same >= 3:
            p.append(f"Svarbu: {same} iš stipriausių pozicijų yra tas pats sektorius "
                     f"({best_sec}). Perkant kelias iš jų, rizika nepasiskirsto — "
                     f"tai iš esmės viena pozicija keliais tikeriais.")

    moves = [s.get("exp_move") for _, s in rows if s.get("exp_move")]
    if moves:
        med_move = float(np.median(moves))
        if med_move < target * 0.85:
            p.append(f"Dėmesio: per tavo {HOLD_HOURS:.1f} val. laikymo laiką tipinis judesys "
                     f"šiose akcijose yra ~{med_move:.1f}%, o tikslas nustatytas {target}%. "
                     f"Arba laikyk ilgiau, arba sumažink tikslą iki ~{med_move:.1f}% — "
                     f"kitaip dauguma sandorių nespės pasiekti tikslo.")

    falling = [sec for sec, chg in sector_state.items() if chg < -1.5]
    rising = [sec for sec, chg in sector_state.items() if chg > 1.0]
    if falling:
        p.append(f"Ištisai krenta: {', '.join(falling)} — čia kritimas dažniau tęsiasi nei atšoka.")
    if rising:
        p.append(f"Laikosi tvirtai: {', '.join(rising)}.")

    slides = [r[0]["tag"] for r in rows if (r[1].get("down_days") or 0) >= 3]
    if slides:
        p.append(f"Kelias dienas iš eilės krenta: {', '.join(slides[:6])} — šioms balas "
                 f"sąmoningai sumažintas, nes tai nebe vienadienis kritimas.")

    return " ".join(p)


def explain(d, s, target, rows):
    """Kodėl būtent ši, o ne kitos — palyginimas su likusiu sąrašu."""
    others = [x for x in rows if x[0]["sym"] != d["sym"]]
    med = float(np.median([x[1]["score"] for x in others])) if others else 0

    reasons = []
    setup = s.get("setup", "kritimas")
    if setup == "atsigavimas":
        reasons.append(f"akcija buvo nukritusi ir dabar kyla nuo dugno, dar nepasiekusi "
                       f"ankstesnės viršūnės — kelias aukštyn dar neišnaudotas")
    elif setup == "jau pakilusi":
        reasons.append("akcija šiandien jau smarkiai pakilo, todėl tai nebėra nuolaidos "
                       "pirkimas — balas surinktas iš kitų rodiklių")
    elif s["parts"]["multiday"] >= 70 and (s.get("down_days") or 0) <= 1:
        reasons.append("kritimas prasidėjo šiandien, o ne tęsiasi kelias dienas — "
                       "būtent toks vienadienis nuosmukis dažniausiai ir atšoka")
    if s["parts"]["trend"] >= 85:
        reasons.append("bendra akcijos kryptis vis dar kylanti, tad perki nuolaidą "
                       "augančioje akcijoje, o ne bandai gaudyti krentančią")
    if s["parts"]["room"] >= 80:
        reasons.append(f"virš kainos yra pakankamai laisvos erdvės — {target}% tikslas "
                       f"telpa neatsimušant į pasipriešinimą")
    if s["parts"]["atr"] >= 80:
        reasons.append("akcija juda pakankamai gyvai, kad toks judesys realiai įvyktų per dieną")
    if s["parts"]["rvol"] >= 85:
        reasons.append("apyvarta didesnė nei įprastai — kritimą pastebėjo ir kiti")
    if s["parts"]["vwap"] >= 85:
        reasons.append("kaina nusileidusi kiek žemiau dienos vidurkio, iš kur dažnai grįžtama")

    if not reasons:
        reasons.append("nė vienas rodiklis nėra išskirtinis, bet ir silpnų vietų nedaug — "
                       "balą surinko tolygumu")

    lead = ("Stipriausia šiandienos pozicija" if s["score"] == max(x[1]["score"] for x in rows)
            else "Viena iš stipresnių pozicijų")
    diff = s["score"] - med
    comp = (f"{lead}: balas {s['score']:.0f} prieš sąrašo medianą {med:.0f} "
            f"({diff:+.0f}). ")

    body = comp + "Ją į priekį kelia tai, kad " + "; ".join(reasons[:3]) + "."

    risks = [t for lvl, t in s["flags"] if lvl in ("warn", "stop")]
    if risks:
        body += " Prieš perkant verta žinoti: " + risks[0].lower() + "."
    return body


def write_html(rows, market, path, refresh_seconds=None, sector_state=None, stats=None):
    market_lt = {"bull": "kyla", "bear": "krenta", "neutral": "šoninė"}[market]
    overview = market_overview(rows, market, sector_state or {}, TARGET_PCT)
    now_lt = datetime.now(_DTZ) if _DTZ else datetime.now()

    stats_html = ""
    if stats:
        done = sum(b["n"] for b in stats.values())
        if done >= 5:
            lines = []
            for g in ["A", "B", "C", "D"]:
                b = stats.get(g)
                if b and b["n"]:
                    pct = b["tikslas"] / b["n"] * 100
                    lines.append(f"<div class='srow'><span>{g}</span>"
                                 f"<i><b style='width:{pct:.0f}%'></b></i>"
                                 f"<u>{pct:.0f}% ({b['n']})</u></div>")
            stats_html = ("<div class='stats'><div class='sh'>Live backtest — "
                          "Dalis sandorių, pasiekusių tikslą, pagal pakopą</div>"
                          + "".join(lines) + "</div>")

    # Naujausio 5 min. baro laikas — parodo tikrą duomenų šviežumą
    data_lt = "?"
    try:
        stamps = [pd.Timestamp(d["asOf"]) for d, _ in rows if d.get("asOf")]
        if stamps:
            ts = max(stamps)
            if ts.tzinfo is not None and _DTZ:
                ts = ts.tz_convert(_DTZ)
            data_lt = f"{ts:%H:%M}"
    except Exception:
        pass
    refresh_tag = (f'<meta http-equiv="refresh" content="{refresh_seconds}">'
                   if refresh_seconds else "")

    def bars(s):
        return "".join(
            f"<div class='br'><span>{lbl}</span>"
            f"<i><b style='width:{s['parts'][k]:.0f}%'></b></i>"
            f"<u>{s['parts'][k]:.0f}</u></div>"
            for k, lbl, _ in CRITERIA)

    cards = []
    for i, (d, s) in enumerate(rows, 1):
        fl = "".join(f"<li class='{lvl}'>{txt}</li>" for lvl, txt in s["flags"])
        cs = d.get("cur_sym", "")
        cur = d.get("cur", "")
        cards.append(f"""
        <details class="card" {'open' if i == 1 else ''}>
          <summary><span class="rk">{i}</span><span class="tk">{d['tag']}</span>
            <span class="bar"><i style="width:{s['score']:.0f}%"></i></span>
            <span class="sc">{s['score']:.0f}</span>
            <span class="gr g{s['grade']}">{s['grade']}</span></summary>
          <div class="in">
            <div class="nm">{d['name']} · {d['sym']} · {s.get('setup','')}</div>
            <div class="verdict {'ok' if s.get('tradeable') else 'no'}">{
              'Atitinka visas sąlygas' if s.get('tradeable')
              else 'NETINKAMA: ' + (s['blocking'][0] if s.get('blocking')
                   else f"rizika/nauda {s['rr']:.2f} per maža")}</div>
            {f'<p class="why">{explain(d, s, TARGET_PCT, rows)}</p>' if i <= 3 else ''}
            <div class="tags">
              <span>kaina {cs}{d['price']:.2f}</span><span>kritimas {(s['dip'] or 0):.1f}%</span>
              <span>diapazone {(s['rng'] or 0):.0f}%</span><span>iki pasipr. {(s['room'] or 0):.1f}%</span>
              <span>ATR {d['atrPct']:.1f}%</span><span>RSI {d['rsi']:.0f}</span>
              <span>RVOL {(d['rvol'] or 0):.2f}</span><span>VWAP {(s['vw_d'] or 0):+.1f}%</span><span>realus tikslas per {HOLD_HOURS:.1f}h ~{(s.get('exp_move') or 0):.1f}%</span><span>1 val. {(d.get('m1h') or 0):+.1f}%</span><span>3 val. {(d.get('m3h') or 0):+.1f}%</span><span>nakties šuolis ~{(d.get('gap') or 0):.1f}%</span>
            </div>
            <div class="plan"><div><span>Įėjimas</span><b>{cs}{d['price']:.2f}</b></div>
              <div><span>Stop</span><b>{cs}{s['stop']:.2f}</b></div>
              <div><span>Tikslas</span><b>{cs}{s['tp']:.2f}</b></div>
              <div><span>R:R</span><b>{s['rr']:.2f}</b></div>
              <div><span>Kiekis</span><b>{s['shares']} vnt.</b></div>
              <div><span>Pozicija</span><b>{cs}{s['pos_value']:,.0f}</b></div>
              <div><span>Pelnas neto</span><b>{cs}{s['net']:.0f}</b></div>
              <div><span>Rizikuoji</span><b>{cs}{s['real_risk']:.0f}</b></div></div>
            {bars(s)}
            <ul class="fl">{fl}</ul>
          </div>
        </details>""")

    html = f"""<!doctype html><html lang="lt"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
{refresh_tag}
<title>Dip reitingas</title><style>
:root{{--ink:#16233A;--ink2:#54637E;--line:#C9D2E0;--bg:#E9EDF3;--card:#FDFDFB;--up:#1F7A5C;--warn:#B26B00;--stop:#A8322D}}
*{{box-sizing:border-box}}body{{margin:0;padding:22px 16px 50px;background:var(--bg);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;max-width:760px;margin:0 auto}}
h1{{font-size:26px;margin:0 0 6px;font-weight:600;letter-spacing:-0.01em}}
.meta{{font-size:12px;color:var(--ink2);margin-bottom:12px}}
.overview{{font-size:13.5px;line-height:1.6;color:var(--ink);background:var(--card);
border:1px solid var(--line);border-radius:8px;padding:14px;margin-bottom:20px}}
.stats{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:13px;margin-bottom:20px}}
.sh{{font-size:11.5px;color:var(--ink2);margin-bottom:9px}}
.srow{{display:flex;align-items:center;gap:9px;margin-bottom:5px}}
.srow span{{font-size:12px;font-weight:600;width:14px}}
.srow i{{flex:1;height:5px;background:#E4E9F1;border-radius:3px;overflow:hidden}}
.srow b{{display:block;height:100%;background:var(--up)}}
.srow u{{font-size:11px;color:var(--ink2);width:64px;text-align:right;text-decoration:none;
font-variant-numeric:tabular-nums}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;margin-bottom:7px;overflow:hidden}}
summary{{display:flex;align-items:center;gap:10px;padding:12px;cursor:pointer;list-style:none}}
summary::-webkit-details-marker{{display:none}}
.rk{{font-size:11px;color:var(--ink2);width:16px}}
.tk{{font-weight:600;font-size:15px;min-width:56px;font-variant-numeric:tabular-nums}}
.bar{{flex:1;height:5px;background:#DFE5EE;border-radius:3px;overflow:hidden}}
.bar i{{display:block;height:100%;background:var(--ink)}}
.sc{{font-size:13px;width:26px;text-align:right;font-variant-numeric:tabular-nums}}
.gr{{width:24px;height:24px;border-radius:5px;color:#fff;font-weight:700;font-size:12px;
display:flex;align-items:center;justify-content:center}}
.gA{{background:var(--up)}}.gB{{background:#3F7FA8}}.gC{{background:var(--warn)}}.gD{{background:var(--stop)}}
.in{{padding:0 12px 14px;border-top:1px solid var(--line)}}
.nm{{font-size:12px;color:var(--ink2);margin:10px 0 8px}}
.verdict{{font-size:12px;font-weight:600;padding:7px 10px;border-radius:5px;margin-bottom:12px}}
.verdict.ok{{background:#E6F2EC;color:#14543E}}
.verdict.no{{background:#F3F0EC;color:#6B5E4E}}
.why{{font-size:12.5px;line-height:1.55;color:var(--ink);background:var(--bg);border-left:3px solid var(--up);
padding:9px 11px;border-radius:5px;margin:0 0 12px}}
.tags{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}}
.tags span{{font-size:11px;background:var(--bg);padding:4px 7px;border-radius:4px;color:var(--ink2);
font-variant-numeric:tabular-nums}}
.plan{{display:grid;grid-template-columns:repeat(3,1fr);gap:9px;background:var(--ink);color:#F4F7FB;
padding:12px;border-radius:7px;margin-bottom:14px}}
.plan div{{display:flex;flex-direction:column;gap:2px}}
.plan span{{font-size:10px;opacity:.65}}
.plan b{{font-size:14px;font-variant-numeric:tabular-nums}}
.br{{display:flex;align-items:center;gap:8px;margin-bottom:5px}}
.br span{{font-size:12px;color:var(--ink2);width:150px}}
.br i{{flex:1;height:4px;background:#E4E9F1;border-radius:2px;overflow:hidden}}
.br b{{display:block;height:100%;background:var(--up)}}
.br u{{font-size:11px;color:var(--ink2);width:22px;text-align:right;text-decoration:none;
font-variant-numeric:tabular-nums}}
.fl{{list-style:none;padding:0;margin:12px 0 0;display:flex;flex-direction:column;gap:6px}}
.fl li{{font-size:12.5px;padding:7px 9px;border-radius:5px;border-left:3px solid}}
.fl .warn{{background:#FBF3E4;border-color:var(--warn);color:#6E4400}}
.fl .stop{{background:#FBEBEA;border-color:var(--stop);color:#7A2320}}
</style>
<h1>Intraday modelis</h1>
<div class="meta">Atnaujinta {now_lt:%H:%M} (Vilnius) · duomenys iš {data_lt} ·
tikslas {TARGET_PCT}% · rinka: {market_lt} · {len(rows)} akcijos</div>
<div class="overview">{overview}</div>
{stats_html}
{''.join(cards)}
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)



# ----------------------------- REZULTATU ZURNALAS -----------------------------

JOURNAL_FIELDS = ["data", "laikas", "sym", "tag", "balas", "pakopa", "scenarijus",
                  "tinkamas", "ijejimas", "stop", "tikslas", "busena", "rezultatas",
                  "baigties_laikas", "baigties_kaina"]
JOURNAL_TOP_N = 3          # kiek geriausiu irasyti kiekviena diena


def load_journal(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def save_journal(path, entries):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
        w.writeheader()
        for e in entries:
            w.writerow({k: e.get(k, "") for k in JOURNAL_FIELDS})


def resolve_entry(entry, intraday_all):
    """Ka kaina padare PO irasymo: pasieke tiksla, stop, ar nei viena."""
    try:
        intra = flatten(intraday_all, entry["sym"]).dropna(subset=["Close"])
        if intra.empty:
            return entry
        start = pd.Timestamp(f"{entry['data']} {entry['laikas']}")
        if intra.index.tz is not None:
            start = start.tz_localize(intra.index.tz) if start.tzinfo is None else start
        after = intra[intra.index > start]
        if after.empty:
            return entry

        stop, target = float(entry["stop"]), float(entry["tikslas"])
        for ts, bar in after.iterrows():
            hit_t = float(bar["High"]) >= target
            hit_s = float(bar["Low"]) <= stop
            if hit_t and hit_s:
                # tame paciame bare abu - nezinom eiliskumo, laikom nuostoliu
                entry.update(busena="baigta", rezultatas="neaisku (abu)",
                             baigties_laikas=str(ts), baigties_kaina=f"{float(bar['Close']):.2f}")
                return entry
            if hit_t:
                entry.update(busena="baigta", rezultatas="tikslas",
                             baigties_laikas=str(ts), baigties_kaina=f"{target:.2f}")
                return entry
            if hit_s:
                entry.update(busena="baigta", rezultatas="stop",
                             baigties_laikas=str(ts), baigties_kaina=f"{stop:.2f}")
                return entry

        # Nei tikslas, nei stop. Jei nuo irasymo praejo daugiau nei diena - uzdarom.
        last_ts = after.index[-1]
        if (last_ts.date() - start.date()).days >= 1:
            last_close = float(after["Close"].iloc[-1])
            entry.update(busena="baigta",
                         rezultatas="be rezultato" if last_close < target else "tikslas",
                         baigties_laikas=str(last_ts), baigties_kaina=f"{last_close:.2f}")
        return entry
    except Exception:
        return entry


def update_journal(path, rows, intraday_all, now):
    """Uzbaigia senus irasus ir prideda siandienos geriausius. Klaidos neblokuoja skenerio."""
    try:
        entries = load_journal(path)
        for e in entries:
            if e.get("busena") == "atviras":
                resolve_entry(e, intraday_all)

        today = now.strftime("%Y-%m-%d")
        have = {(e["sym"], e["data"]) for e in entries}
        added = 0
        for d, s in rows:
            if added >= JOURNAL_TOP_N:
                break
            if (d["sym"], today) in have or not d.get("price"):
                continue
            entries.append(dict(
                data=today, laikas=now.strftime("%H:%M"), sym=d["sym"], tag=d["tag"],
                balas=f"{s['score']:.1f}", pakopa=s["grade"], scenarijus=s.get("setup", ""),
                tinkamas="taip" if s.get("tradeable") else "ne",
                ijejimas=f"{d['price']:.2f}", stop=f"{s['stop']:.2f}",
                tikslas=f"{s['tp']:.2f}", busena="atviras", rezultatas="",
                baigties_laikas="", baigties_kaina=""))
            added += 1

        entries = entries[-500:]          # neauginam failo be galo
        save_journal(path, entries)
        return entries
    except Exception:
        return []


def journal_stats(entries):
    """Ar auksciau ivertinti sandoriai realiai baigesi geriau?"""
    out = {}
    for e in entries:
        if e.get("busena") != "baigta":
            continue
        g = e.get("pakopa", "?")
        b = out.setdefault(g, {"n": 0, "tikslas": 0, "stop": 0, "kita": 0})
        b["n"] += 1
        r = e.get("rezultatas", "")
        if r == "tikslas":
            b["tikslas"] += 1
        elif r in ("stop", "neaisku (abu)"):
            b["stop"] += 1
        else:
            b["kita"] += 1
    return out

# ----------------------------- MARKET LAIKAS -----------------------------


def market_open_now():
    """Ar dabar (Europe/Berlin laiku) tikėtina prekybos sesija Xetra/Euronext."""
    now = datetime.now(_TZ) if _TZ else datetime.now()
    if now.weekday() >= 5:
        return False
    mins = now.hour * 60 + now.minute
    return 9 * 60 - 15 <= mins <= 17 * 60 + 40


# ----------------------------- MAIN -----------------------------


def run_once(yf, out_dir, refresh_seconds=None, quiet=False):
    """Vienas visų akcijų surinkimo + reitingavimo ciklas. Grąžina (rows, failed)."""
    os.makedirs(out_dir, exist_ok=True)
    symbols = [s for _, s, _ in WATCHLIST]
    if not quiet:
        print(f"[{datetime.now():%H:%M:%S}] Renkami duomenys ({len(symbols)} akcijos)…")

    intraday_all, daily_all = collect(yf, symbols)
    market = market_bias(yf)

    rows, failed = [], []
    collected = []
    for tag, sym, name in WATCHLIST:
        try:
            collected.append(build_row(yf, tag, sym, name, intraday_all, daily_all))
        except Exception as e:
            failed.append((tag, sym, str(e)[:60]))

    # Sektoriaus būsena — mediana iš to paties sąrašo bendraamžių
    sector_state = {}
    for sec in set(x["sector"] for x in collected):
        chgs = [x["day_chg"] for x in collected
                if x["sector"] == sec and x["day_chg"] is not None]
        if len(chgs) >= 2:
            sector_state[sec] = float(np.median(chgs))

    tb = time_budget()
    for d in collected:
        rows.append((d, score_stock(d, TARGET_PCT, market,
                                    sector_chg=sector_state.get(d["sector"]), tb=tb)))

    if not rows:
        print("Nepavyko gauti nė vienos akcijos duomenų šį kartą. Bandysiu vėl.")
        return [], failed

    # Pirma tinkami sandoriai pagal balą, tada visi kiti — kad viršuje būtų tai,
    # ką realiai galima pirkti, o ne tik aukščiausias balas
    rows.sort(key=lambda x: (bool(x[1].get("tradeable")), x[1]["score"]), reverse=True)
    print_table(rows)
    if failed:
        print("Nepavyko:", ", ".join(f"{t} ({m})" for t, _, m in failed), "\n")

    # Rezultatų žurnalas: įrašom šiandienos geriausius, užbaigiam senus įrašus
    now_local = datetime.now(_DTZ) if _DTZ else datetime.now()
    entries = update_journal(os.path.join(out_dir, "zurnalas.csv"),
                             rows, intraday_all, now_local)
    stats = journal_stats(entries)
    if stats:
        print("\nŽurnalas (užbaigti sandoriai):")
        for g in ["A", "B", "C", "D"]:
            b = stats.get(g)
            if b and b["n"]:
                print(f"  {g}: {b['n']:>3} sandorių, tikslą pasiekė "
                      f"{b['tikslas']/b['n']*100:>5.1f}%")

    html_path = os.path.join(out_dir, "index.html")
    write_html(rows, market, html_path, refresh_seconds=refresh_seconds,
               sector_state=sector_state, stats=stats)

    with open(os.path.join(out_dir, "dip_reitingas.json"), "w", encoding="utf-8") as f:
        json.dump([{**d, **{k: v for k, v in s.items() if k != "parts"}}
                   for d, s in rows], f, ensure_ascii=False, indent=1, default=str)

    return rows, failed


def main():
    parser = argparse.ArgumentParser(description="Dip reitingas — intraday skeneris")
    parser.add_argument("--loop", action="store_true",
                        help="Veikti nuolat, atsinaujinant automatiškai")
    parser.add_argument("--interval", type=int, default=LOOP_INTERVAL_SEC,
                        help="Atsinaujinimo intervalas sekundėmis --loop režime")
    parser.add_argument("--out", type=str, default=None,
                        help="Aplankas, kur rašyti index.html/json (numatyta: laikinas aplankas)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Nebandyti atidaryti naršyklės (naudoti serveryje / CI)")
    args = parser.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Trūksta bibliotekos. Paleisk: pip install yfinance pandas numpy tzdata")

    out_dir = args.out or tempfile.gettempdir()
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, "index.html")
    open_browser = OPEN_BROWSER and not args.no_browser

    if not args.loop:
        rows, _ = run_once(yf, out_dir)
        if rows:
            print(f"HTML ataskaita: {html_path}")
            if open_browser:
                webbrowser.open("file://" + html_path)
        return

    print("Veikimas kartojamas automatiškai. Sustabdyti: Ctrl+C arba uždaryk šį langą.\n")
    opened = False
    try:
        while True:
            open_now = market_open_now()
            refresh_s = args.interval if open_now else LOOP_INTERVAL_OFF_SEC
            rows, _ = run_once(yf, out_dir, refresh_seconds=refresh_s)

            if rows and not opened:
                if open_browser:
                    webbrowser.open("file://" + html_path)
                opened = True

            if not open_now:
                print(f"Rinka šiuo metu uždaryta — kitas patikrinimas po "
                      f"{LOOP_INTERVAL_OFF_SEC // 60} min.\n")

            time.sleep(refresh_s)
    except KeyboardInterrupt:
        print("\nSustabdyta.")


if __name__ == "__main__":
    main()
