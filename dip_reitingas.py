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
from datetime import datetime, timezone

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

# ISEJIMO TAISYKLE (patikrinta 2026-09: vid. rezultatas +0.223% vs +0.082% su
# fiksuotu 3% tikslu; IBS signalo pranasumas +0.199% vs +0.154%).
# Pelnas nefiksuojamas ties riba — pasiekus MIN_TARGET_PCT ijungiamas slenkantis
# stop, ir pozicija laikoma tol, kol kaina atsitraukia TRAIL_PCT nuo virsunes.
# ISEJIMO TAISYKLE. Patikrinta 2026-09 dviejose nepriklausomose imtyse
# (Europa 194 akcijos, JAV 143 akcijos, 10 metu):
#   RSI(2) > 70          Europa +27.31 EUR   JAV +101.19 EUR   67-70% pelningu
#   IBS > 0.8            Europa +16.74 EUR   JAV  +85.48 EUR   62-66%
#   uzdarymas virs SMA5  Europa +13.37 EUR   JAV  +81.57 EUR   66-68%
#   fiksuotas 2%/1.5%    Europa -19.13 EUR   JAV   -6.32 EUR   42-44%
# Fiksuotas stop 1.5% yra triuksmo lygyje ir uzbaigia 58% sandoriu nuostoliu.
# Salyginis isejimas laukia, kol grizimas prie vidurkio realiai ivyks.
EXIT_MODE = "salyginis"   # "salyginis" arba "fiksuotas"
EXIT_RSI = 70.0           # parduoti, kai RSI(2) pakyla virs sios ribos
EXIT_IBS = 0.80           # arba kai IBS pakyla virs sios ribos
EXIT_MAX_DIENU = 10       # ilgiausiai laikoma
EXIT_STOP_PCT = 3.0       # apsauginis stop (be jo Europoje +27 EUR, su juo +11 EUR,
                          # bet be jo YDX tipo epizodas neturi pabaigos)

TARGET_PCT = 2.0        # orientacinis tikslas rodmenims; isejima lemia EXIT_MODE
TRAIL_PCT = 1.5
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
MIN_RR = 1.3            # minimalus rizikos/naudos santykis, kad sandoris butu tinkamas

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

# Universas imamas is universas.py — TO PACIO failo, kuri naudoja backtestai.
# Anksciau cia buvo atskiras 25 akciju sarasas, todel skeneris ir matavimas
# dirbo su skirtingomis imtimis.
try:
    import universas as _U
    _SEKT = _U.sektoriai()
    WATCHLIST = [(s.split(".")[0], s, s) for s in _U.visi_tikeriai()]
    SECTORS = dict(_SEKT)
except Exception:                     # atsarginis variantas, jei failo nera
    WATCHLIST = [("SAP", "SAP.DE", "SAP SE"), ("IFX", "IFX.DE", "Infineon"),
                 ("BESI", "BESI.AS", "BE Semiconductor")]
    SECTORS = {"SAP.DE": "Technologijos", "IFX.DE": "Technologijos",
               "BESI.AS": "Technologijos"}

# Kiek akciju rodoma puslapyje ir kiek ju tikrinama 5 min. duomenimis
RODOMA = 5
INTRADAY_KANDIDATU = 60      # tiek geriausiai atitinkanciu tikrinama detaliai
                             # (universas isaugo iki 266, tad ir kandidatu daugiau)
MIN_APYVARTA_EUR = 5e6

MARKET_INDEX = "^STOXX50E"   # rinkos kryptis

# Valiuta pagal biržos galūnę. Portfelis laikomas ACCOUNT_CURRENCY valiuta.
ACCOUNT_CURRENCY = "EUR"

# Laikymo horizontas valandomis. Tikslas turi buti pasiektas per si laika.
# HOLD_DAYS matuojamas PREKYBOS dienomis, ne kalendorinemis. Anksciau cia buvo
# 72 "valandos", ir backtestas jas verte 8 sesijomis, modulis — 3 kalendorinemis
# dienomis, o zurnalas — 3 dienomis. Trys skirtingi modeliai viename projekte.
HOLD_DAYS = 3            # kiek PREKYBOS dienu laikoma pozicija (2-5)
HOLD_HOURS = HOLD_DAYS * 8.5    # tas pats dydis prekybos valandomis

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

# BUKLE 2026-09-07 po pakartotinio patikrinimo su 25 akciju sarasu:
# NE VIENAS kriterijus nepatvirtino pranasumo nematytoje imties dalyje.
# IBS su ankstesniu 19 akciju sarasu rode +0.157%, su siuo — +0.060% (intervalas
# kerta nuli). Todel balas laikomas ATITIKIMO KRITERIJAMS matu, ne prognoze.
# Svoriu nekeiciam: perdelioti juos pagal tuos pacius duomenis, ant kuriu jau
# derinta, reikstu persimokyma. Patvirtinta modulio verte yra kitur — rinkos
# rezimo filtre, isejimo taisykleje ir rizikos skaiciavime.
CRITERIA = [
    # Du virsutiniai patvirtinti nematytoje imties dalyje (2026-09, 732 dienos,
    # savaitiniai blokai, atsparumo patikra isbraukiant akcijas):
    #   Z-balas zemas  +0.145% (+0.032..+0.256), neto +0.075%
    #   IBS zemas      +0.103% (+0.019..+0.188), neto +0.033%
    # Likusieji patvirtinimo NEISLAIKE — ju svoris mazas, nes kompozitas su
    # 60% nepatvirtintu kriteriju praskiede signala iki triuksmo (+0.061%).
    ("zbal",     "Nutolimas nuo vidurkio",   35),   # patvirtinta, neto +0.076%
    ("ibs",      "Padėtis dienos diapazone", 25),   # patvirtinta, neto +0.013%
    ("vwap",     "Padėtis prieš VWAP",       15),   # patvirtinta, neto +0.051%
    ("stab",     "Ar kritimas sustojo",       8),
    ("multiday", "Vienadienis ar tęstinis",   6),
    ("room",     "Vieta iki pasipriešinimo",  5),
    ("dip",      "Kritimo gylis",             3),
    ("atr",      "Judrumas (ATR)",            2),
    ("trend",    "Trendas (20/50 SMA)",       1),
]

# Sektoriai imami is universas.py kartu su WATCHLIST (zr. virsuje).
# Anksciau cia buvo senas 25 akciju zodynas, kuris PERRASYDAVO importuota:
# modulis dirbo su 270 instrumentu, bet sektoriu zinojo tik 25, o likusios
# gaudavo "kita" — todel sektoriaus filtras ir zaliavu atpazinimas neveike.
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
    # daily cia jau be siandienos, tad paskutine eilute yra vakar diena
    prev = daily.iloc[-1]
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


SESSION_HOURS = 8.5      # kiek valandu per para birza realiai prekiauja


def expected_move(vol_bar, hours, bph=12):
    """Tiketinas kainos judesys per N PREKYBOS valandu.

    Anksciau 16 val. horizontas buvo verciamas i 16*12 baru, lyg nakti rinka
    prekiautu — judesys pervertinamas ~1.4 karto. Dabar naktys neskaiciuojamos:
    16 val. horizontas = 1.9 prekybos dienos = ~16 realiu prekybos valandu.
    """
    if not vol_bar:
        return None
    bars = max(1.0, hours * bph)      # hours jau yra PREKYBOS valandos
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


def completed_daily(daily):
    """Tik UZBAIGTOS dienos — be siandienos nebaigto baro.

    Yahoo dienos duomenyse siandienos eilute pildoma realiu laiku, todel ATR,
    SMA ir kritimo dienu skaicius keistusi kas 5 minutes kartu su kaina.
    Tai reiskia, kad rodikliai reikstu ne ta, ka sako ju pavadinimas.
    """
    try:
        today = datetime.now(_TZ).date() if _TZ else datetime.now().date()
        idx_dates = pd.Index([i.date() if hasattr(i, "date") else i for i in daily.index])
        closed = daily[idx_dates < today]
        return closed if len(closed) >= 30 else daily
    except Exception:
        return daily


Z_SESIJOS = 2.0      # Z-balo langas SESIJOMIS (patvirtinta: 20 valandiniu baru)


def rsi2_daily(daily_close, n=2):
    """RSI(2) is dienos uzdarymu — isejimo salygai.

    Butent sis periodas naudojamas isejime: RSI(14) beveik niekada nepasiekia 70,
    o RSI(2) reaguoja per viena dvi dienas, todel tinka trumpam laikymui.
    """
    try:
        d = daily_close.astype(float).diff()
        up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
        val = 100 - 100 / (1 + up / dn.replace(0, np.nan))
        return float(val.iloc[-1]) if len(val) else None
    except Exception:
        return None


def price_zscore(cont_close, price, n=None):
    """Kiek standartiniu nuokrypiu kaina nutolusi nuo pastaruju sesiju vidurkio.

    SVARBU: langas matuojamas SESIJOMIS, ne barais. Patvirtinimas gautas su
    20 valandiniu baru (~2 sesijos). Anksciau cia buvo 20 PENKIAMINUCIU baru
    (~100 minuciu) — modulis skaiciavo visai kita dydi nei tas, kuris patvirtintas.
    5 min. duomenims 2 sesijos = ~204 barai.
    """
    if n is None:
        n = int(Z_SESIJOS * 8.5 * 12)     # sesijos x valandos x barai valandoje
    try:
        seg = cont_close.tail(n).astype(float)
        if len(seg) < n:
            return None
        m, s = float(seg.mean()), float(seg.std())
        return (price - m) / s if s > 0 else None
    except Exception:
        return None


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
        setup = "Atsigavimas"
    elif overextended:
        dip_part = min(dip_part, 12.0)
        setup = "Jau pakilusi"
    elif (day_chg is not None and day_chg > 0.5
          and d.get("ibs") is not None and d["ibs"] > 0.6):
        setup = "Kyla, prie viršūnės"
    else:
        setup = "Kritimas"

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

    # Krintantis peilis — atskiras scenarijus, o ne tik bauda. Skirtumas svarbus:
    # tokia akcija nera "neidomi", ji yra pavojinga, ir tai turi matytis atskirai.
    # Rytoj ji dazniausiai tampa "atsigavimo" kandidate, todel ja verta stebeti.
    if knife:
        setup = "Krintantis peilis"

    # IBS: 0 = uzdaro prie dienos dugno (geriausia), 1 = prie virsunes.
    # Kreive pagal ismatuotas reiksmes: <0.2 stipriai geriau, >0.8 stipriai blogiau.
    ibs_v = num(d.get("ibs"))
    z_v = num(d.get("zscore"))
    parts = {
        # Zemas z = kaina toli zemiau pastaruju baru vidurkio
        "zbal": curve(z_v, [(-3.0, 100), (-1.5, 95), (-0.7, 80), (0.0, 55),
                            (0.7, 35), (1.5, 18), (3.0, 8)]),
        "ibs":  curve(ibs_v, [(0.0, 100), (0.15, 96), (0.3, 78), (0.45, 58),
                              (0.6, 40), (0.8, 20), (1.0, 8)]),
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
        "_nenaudojamas_rsi":  curve(d.get("rsi"),
                      [(25, 20), (40, 60), (50, 90), (60, 100), (70, 70), (80, 25)]
                      if recovering else
                      [(10, 25), (20, 55), (28, 85), (35, 100), (45, 85), (55, 55), (65, 30), (80, 10)]),
        # Patvirtinta nematytoje imties dalyje: kaina ZEMIAU VWAP +0.121%
        # (+0.025..+0.211), neto +0.051%. Anksciau kreive maksimuma dave ties
        # -0.2%, t. y. matavo ne ta, kas pasitvirtino.
        "vwap": curve(vw_d, [(-5, 100), (-2, 92), (-1, 82), (-0.3, 62),
                             (0.3, 42), (1.0, 28), (3, 12)]),
        "_nenaudojamas_rvol": curve(rv,   [(0.3, 15), (0.7, 45), (1, 70), (1.4, 95), (2.5, 100), (4, 80), (7, 55), (12, 35)]),
        "trend": trend,
        "_nenaudojamas_support": 15.0 if (sup_d is not None and sup_d < 0) else
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
    tp = price * (1 + target / 100)          # orientacinis, rodmenims
    trail_from = tp * (1 - TRAIL_PCT / 100)
    rr = (tp - price) / (price - stop) if price > stop else 0.0

    # Salyginio isejimo stop platesnis: 1.5% yra triuksmo lygyje ir uzbaigia
    # 58% sandoriu nuostoliu. 3% leidzia grizimui prie vidurkio ivykti.
    if EXIT_MODE == "salyginis":
        # Tik stop'o korekcija — kiekis, pozicija ir rizika skaiciuojami zemiau,
        # bendrame sizingo bloke, is jau pakoreguoto stop'o.
        stop = min(stop, price * (1 - EXIT_STOP_PCT / 100))
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
    gross = pos_value * target / 100      # konservatyvu: realiai gali buti daugiau
    net = gross - 2 * FEE_PER_TRADE      # mokestis perkant ir parduodant
    real_risk = shares * (price - stop)

    flags, mult = [], 1.0
    if d.get("earnings"):
        mult *= 0.35
        flags.append(("stop", "Ataskaita per 2 dienas — kaina šoks bet kuria kryptimi"))
    if market == "bear":
        mult *= 0.75
        flags.append(("stop", "Rinka krenta ir šiandien. Per 2 metus tokiomis dienomis net "
                              "geriausiai įvertinti įėjimai vidutiniškai prarado 0,35%"))
    elif market == "bear_soft":
        mult *= 0.92
        flags.append(("warn", "Bendra rinkos kryptis vis dar žemyn, nors šiandien kyla — "
                              "atšokimas gali būti trumpalaikis"))
    elif market == "bull":
        mult *= 1.05
    if room_far is not None and room_far < target:
        flags.append(("stop", f"Net iki tolimesnių lubų tik {room_far:.1f}% — "
                              f"net minimalus {target}% tikslas netelpa"))
    elif room is not None and room < target:
        flags.append(("warn", f"Kelyje kliūtis ({room:.1f}% aukščiau) — "
                              f"jį reikės pramušti, kad tikslas būtų pasiektas"))
    if a_pct is None:
        flags.append(("warn", "ATR nepavyko suskaičiuoti — judrumo kriterijus neįvertintas, "
                              "balas mažiau patikimas"))
    if exp_mv and exp_mv < target * 0.9:
        flags.append(("warn", f"Per {HOLD_HOURS:.1f} val. tikėtinas judesys ~{exp_mv:.1f}%, "
                              f"o minimalus tikslas {target}% — šiai akcijai jis ant ribos"))
    # --- Laiko biudžetas: ar likusio laiko realiai užtenka tikslui pasiekti? ---
    # Kainos svyravimas auga proporcingai laiko šaknims, todėl tikėtinas likęs
    # judesys = dienos ATR * sqrt(likusi sesijos dalis).
    if tb and a_pct:
        exp_range = a_pct * math.sqrt(max(tb["frac"], 0.01))
        ratio = exp_range / target if target else 0
        hrs = tb["effective"] / 60
        # Laiko biudzetas toliau skaiciuojamas ir veikia bala (kai HOLD_HOURS <= 8),
        # bet informaciniu zymu puslapyje nerodom — jos tik apkraudavo vaizda.
        if ratio < 0.7 and HOLD_HOURS <= 8:
            mult *= 0.85
            flags.append(("warn", f"Liko ~{hrs:.1f} val. efektyvios prekybos — tikėtinas "
                                  f"judesys ~{exp_range:.1f}% nesiekia {target}% tikslo"))

    # --- Nakties šuolio rizika: laikant per naktį stop neveikia ---
    gap = d.get("gap")
    if HOLD_HOURS > 8 and gap and shares > 0:
        stop_dist = (price - stop) / price * 100
        gap_loss = pos_value * gap / 100
        if gap > stop_dist:
            flags.append(("warn", f"Laikant per naktį stop neapsaugo: tipinis šuolis šioje "
                                  f"akcijoje {gap:.1f}%, o stop tik {stop_dist:.1f}% žemiau — "
                                  f"nepalankus atidarymas kainuotų ~{gap_loss:.0f} EUR"))

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

    if ibs_v is not None:
        if ibs_v <= 0.2:
            flags.append(("info", f"IBS {ibs_v:.2f} — kaina prie dienos dugno. Tai stipriausias "
                                  f"modulio kriterijus, bet paskutiniame patikrinime su šiuo "
                                  f"sąrašu jis pranašumo nepatvirtino"))
        elif ibs_v >= 0.8:
            flags.append(("warn", f"IBS {ibs_v:.2f} — kaina prie dienos viršūnės. Istoriškai "
                                  f"tokie įėjimai pasirodo prasčiau už dienos vidurkį"))

    if d.get("cur") and d["cur"] != ACCOUNT_CURRENCY:
        flags.append(("stop", f"Ši akcija kotiruojama {d['cur']}, o portfelis "
                              f"{ACCOUNT_CURRENCY}. Pozicijos dydis skaičiuojamas be valiutos "
                              f"perskaičiavimo, todėl būtų neteisingas"))
    if False:
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
    if 0 < rr < MIN_RR:
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
        laukti = []
        if m1h is not None and m1h <= 0.1:
            laukti.append("kad pastaroji valanda taptų teigiama")
        if down_days and down_days >= 2:
            laukti.append(f"kad nutrūktų kritimo dienų serija (dabar {down_days})")
        if sector_chg is not None and sector_chg < -1.0:
            laukti.append(f"kad nustotų kristi sektorius ({sector_chg:+.1f}%)")
        flags.append(("stop", f"Krintantis peilis: per 3 val. {m3h:+.1f}%, per pastarąją "
                              f"valandą {m1h:+.1f}% — dugno ženklo dar nėra. Nepirkti, stebėti."))
        if laukti:
            flags.append(("info", "Ko laukti, kad taptų pirkimo kandidatu: " +
                                  ", ".join(laukti[:3])))
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

    # Prie salyginio isejimo nera fiksuoto tikslo, todel R:R pries ji nieko nemat
    # uoja: stop 3%, "tikslas" 2% -> santykis visada zemiau 1.3, ir joks sandoris
    # nepraeitu. Vietoj to reikalaujam, kad (a) stop butu protingo dydzio ir
    # (b) akcija per laikymo laika realiai galetu nueiti bent stop atstuma.
    stop_dist = (price - stop) / price * 100 if price > 0 else 99
    if EXIT_MODE == "salyginis":
        exp_mv = num(d.get("exp_move")) or 0.0
        tradeable = (
            (not blocking)
            and stop_dist <= EXIT_STOP_PCT * 1.25
            and (exp_mv <= 0 or exp_mv >= stop_dist * 0.8)
            and (room_far is None or room_far >= stop_dist)
        )
    else:
        tradeable = ((not blocking) and rr >= MIN_RR
                     and (room_far is None or room_far >= target * 1.2))
    grade = "A" if score >= 78 else "B" if score >= 64 else "C" if score >= 50 else "D"

    # --- Frazė vietoj raidės: ka konkreciai rodo duomenys ---
    if blocking:
        if setup == "Krintantis peilis":
            fraze, spalva = "Krintantis peilis", "raus"
        elif any("Rinka krenta" in t for t in blocking):
            fraze, spalva = "Rinka krenta — praleisti", "raus"
        elif any("Ataskaita" in t for t in blocking):
            fraze, spalva = "Ataskaita per 2 dienas", "raus"
        elif any("netelpa" in t for t in blocking):
            fraze, spalva = "Tikslui nėra vietos", "raus"
        else:
            fraze, spalva = "Netinkama", "raus"
    elif setup == "Krintantis peilis":
        fraze, spalva = "Krintantis peilis", "raus"
    elif setup == "Jau pakilusi":
        fraze, spalva = "Jau pakilusi — nuolaidos nėra", "gelt"
    elif setup == "Kyla, prie viršūnės":
        fraze = ("Kylančio trendo tęsinys" if trend >= 85 else "Kyla, prie viršūnės")
        spalva = "gelt"
    elif setup == "Atsigavimas":
        fraze = ("Atsigavimas su sektoriumi"
                 if (sector_chg is not None and sector_chg > 0.3) else "Atsigavimas po kritimo")
        spalva = "zal" if (tradeable and score >= 64) else "gelt"
    else:
        # Kritimo šeima — skiriam pagal tai, ar kritimas jau sustojo
        # Frazes skiriamos pagal FAKTUS, ne pagal kriterijaus bala: anksciau
        # "stab >= 70" buvo tenkinamas beveik visada, todel gilus kritimas ir
        # ramus atsitraukimas gaudavo ta pacia fraze "Kritimas sustojo".
        val_teig = m1h is not None and m1h > 0.05        # pastaroji valanda kyla
        val_neig = m1h is not None and m1h < -0.10       # pastaroji valanda krenta
        gilus = (dd5 or 0) >= 3.0 or (day_chg is not None and day_chg <= -2.0)
        zemai = ibs_v is not None and ibs_v <= 0.25

        if gilus and not val_teig:
            fraze = "Neapibrėžtas kritimas"
        elif val_teig and zemai:
            fraze = "Atsitraukimas, kryptis stabilizavosi"
        elif val_teig:
            fraze = "Kritimas sustojo"
        elif val_neig:
            fraze = "Kritimas dar tęsiasi"
        else:
            fraze = "Ramus atsitraukimas"
        if sector_chg is not None and sector_chg > 0.5 and fraze.startswith("Atsitraukimas"):
            fraze = "Atsitraukimas kylančiame sektoriuje"
        spalva = "zal" if (tradeable and score >= 64) else "gelt"

    return dict(score=score, grade=grade, fraze=fraze, spalva=spalva,
                tradeable=tradeable, blocking=blocking,
                setup=setup, recovering=recovering,
                parts=parts, flags=flags, dip=dip, rng=rng,
                room=room, sup_d=sup_d, vw_d=vw_d, stop=stop, tp=tp, rr=rr, shares=shares,
                down_days=down_days, dd5=dd5, chg3d=chg3d, sector_chg=sector_chg,
                pos_value=pos_value, gross=gross, net=net, real_risk=real_risk,
                trail_from=trail_from, trail_pct=TRAIL_PCT,
                exit_mode=EXIT_MODE, exit_rsi=EXIT_RSI, exit_ibs=EXIT_IBS,
                exit_dienu=EXIT_MAX_DIENU, rsi2=num(d.get("rsi2")),
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
        chg1 = (last / float(c.iloc[-2]) - 1) * 100
        chg5 = (last / float(c.iloc[-6]) - 1) * 100
        sma20 = float(c.tail(20).mean())
        above = last > sma20

        # Svarbu: ilgesne kryptis zemyn NEREISKIA, kad siandien prasta diena.
        # Kieta salyga ijungiama tik kai ir kryptis, ir siandiena zemyn.
        if chg5 < -1.0 and chg1 < 0:
            return "bear"
        if chg5 < -1.0 or (not above and chg5 < 0):
            return "bear_soft"          # kryptis zemyn, bet siandien kyla
        if chg5 > 1.0 and above:
            return "bull"
        return "neutral"
    except Exception:
        return "neutral"


_EARNINGS_CACHE = {}
_NAUJIENU_CACHE = {}

# Raktazodziai antrasciu vertinimui. APYTIKSLU: tai zodziu paieska, ne analize.
# Rodoma tik kaip uzuomina — tikra prasme pamatysi tik perskaites antraste.
NEIG_ZODZIAI = (
    "profit warning", "guidance cut", "cuts guidance", "lowers guidance", "downgrade",
    "downgraded", "investigation", "probe", "lawsuit", "recall", "fraud", "loss",
    "miss", "misses", "slump", "plunge", "falls", "cut to", "warns", "warning",
    "resigns", "layoff", "job cuts", "delay", "halted", "suspend", "weak", "fine",
    "penalty", "strike", "bankruptcy", "restructuring", "impairment", "writedown",
)
TEIG_ZODZIAI = (
    "beats", "beat", "raises guidance", "raised", "upgrade", "upgraded", "record",
    "surge", "jumps", "soars", "rally", "wins", "win", "contract", "deal",
    "approval", "approved", "launch", "buyback", "dividend increase", "expands",
    "partnership", "breakthrough", "strong", "outperform", "profit rises", "growth",
)


def naujienos(yf, symbol, kiek=3, cache_min=30):
    """Paskutines antrastes su apytiksliu atspalviu.

    Grazina saraso elementus: (antraste, saltinis, valandu_senumas, atspalvis),
    kur atspalvis yra "teig" / "neig" / "neutr". Vertinimas paremtas raktazodziu
    paieska antrasteje — tai NERA turinio analize ir gali klysti; todel puslapyje
    antraste rodoma visa, kad galetum perskaityti pats.
    """
    now = time.time()
    hit = _NAUJIENU_CACHE.get(symbol)
    if hit and now - hit[1] < cache_min * 60:
        return hit[0]
    out = []
    try:
        raw = yf.Ticker(symbol).news or []
        for n in raw[:kiek]:
            turinys = n.get("content", n) if isinstance(n, dict) else {}
            antr = (turinys.get("title") or n.get("title") or "").strip()
            if not antr:
                continue
            saltinis = ""
            try:
                saltinis = (turinys.get("provider", {}).get("displayName")
                            or n.get("publisher") or "")
            except Exception:
                pass
            val = None
            try:
                ts = n.get("providerPublishTime")
                if ts:
                    val = (now - float(ts)) / 3600
                else:
                    pub = turinys.get("pubDate") or ""
                    if pub:
                        val = (datetime.now(timezone.utc)
                               - datetime.fromisoformat(pub.replace("Z", "+00:00"))
                               ).total_seconds() / 3600
            except Exception:
                pass
            z = antr.lower()
            neig = sum(1 for w in NEIG_ZODZIAI if w in z)
            teig = sum(1 for w in TEIG_ZODZIAI if w in z)
            atsp = "neig" if neig > teig else ("teig" if teig > neig else "neutr")
            out.append((antr, saltinis, val, atsp))
    except Exception:
        pass
    _NAUJIENU_CACHE[symbol] = (out, now)
    return out


def earnings_soon(yf, symbol, cache_hours=6):
    """Su kesavimu: be jo butu 19 atskiru uzklausu kas 5 min. (~1900 per diena),
    o ataskaitu datos keiciasi kartus per ketvirti."""
    now = time.time()
    hit = _EARNINGS_CACHE.get(symbol)
    if hit and now - hit[1] < cache_hours * 3600:
        return hit[0]
    val = _earnings_soon_uncached(yf, symbol)
    _EARNINGS_CACHE[symbol] = (val, now)
    return val


def _earnings_soon_uncached(yf, symbol):
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


def collect_daily(yf, symbols):
    """1 pakopa: dienos duomenys VISOMS akcijoms — viena bendra uzklausa."""
    return yf.download(symbols, period="6mo", interval="1d", group_by="ticker",
                       progress=False, auto_adjust=False, threads=True)


def preatranka(daily_all, symbols, n=None):
    """Kurias akcijas verta tikrinti 5 min. duomenimis.

    Su 200+ akciju kas 5 min. siusti intraday duomenis visoms reikstu ~30 tuks.
    uzklausu per diena — Yahoo blokuotu. Todel pirma dienos barais atmetam tas,
    kurios akivaizdziai netinka, ir detaliai tikrinam tik likusias.

    Atmetama: per maza apyvarta; kaina virs 20 d. vidurkio (ne atsitraukimas);
    vakar uzdare dienos virsuje (IBS aukstas); kelios kritimo dienos is eiles.
    """
    n = n or INTRADAY_KANDIDATU
    kand = []
    for sym in symbols:
        try:
            d = flatten(daily_all, sym).dropna(subset=["Close", "High", "Low", "Volume"])
            if len(d) < 60:
                continue
            c = d["Close"]
            apyv = float((c * d["Volume"]).tail(20).median())
            if apyv < MIN_APYVARTA_EUR:
                continue
            px = float(c.iloc[-1])
            sma20 = float(c.tail(20).mean())
            rng = float(d["High"].iloc[-1] - d["Low"].iloc[-1])
            ibs_v = ((px - float(d["Low"].iloc[-1])) / rng) if rng > 0 else 0.5
            # kritimo dienos is eiles
            zem = (c.diff() < 0).astype(int).tail(5).tolist()
            serija = 0
            for v in reversed(zem):
                if v:
                    serija += 1
                else:
                    break
            if px > sma20 * 1.01 or ibs_v > 0.75 or serija >= 3:
                continue
            # atstumas nuo 20 d. vidurkio — kuo zemiau, tuo idomiau
            kand.append((sym, (px - sma20) / sma20, apyv))
        except Exception:
            continue
    kand.sort(key=lambda x: x[1])          # labiausiai atsitrauke pirmi
    return [s for s, _, _ in kand[:n]]


def collect_intraday(yf, symbols):
    """2 pakopa: 5 min. duomenys TIK preatranka praejusioms akcijoms."""
    if not symbols:
        return None
    return yf.download(symbols, period="10d", interval="5m", group_by="ticker",
                       progress=False, auto_adjust=False, threads=True)


def collect(yf, symbols):
    """Suderinamumui: viskas vienu kartu (naudojama testuose)."""
    daily = collect_daily(yf, symbols)
    intraday = collect_intraday(yf, symbols)
    return intraday, daily


def build_row(yf, tag, sym, name, intraday_all, daily_all):
    intra = flatten(intraday_all, sym)
    daily = flatten(daily_all, sym)
    if intra.empty or daily.empty or "Close" not in intra:
        raise ValueError("nėra duomenų")

    intra = intra.dropna(subset=["Close"])
    daily = daily.dropna(subset=["Close", "High", "Low"])
    # Dienos rodikliai (ATR, SMA, kritimo dienos, lygiai, nakties suolis) skaiciuojami
    # TIK is uzbaigtu dienu. Intraday rodikliai toliau naudoja siandienos barus.
    daily = completed_daily(daily)
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
    # SVARBU: daily jau praeitas per completed_daily(), tad c.iloc[-1] yra VAKAR.
    # Anksciau cia buvo iloc[-2] — tai grazindavo uzvakar ir iskreipdavo dienos
    # pokyti, atsigavimo scenariju, sektoriaus mediana ir rinkos ploti.
    prev_close = float(c.iloc[-1]) if len(c) > 0 else None
    sma20 = float(c.tail(20).mean())
    sma50 = float(c.tail(50).mean())
    ctx = multiday_context(daily, price)
    avg_vol = float(daily["Volume"].tail(20).mean())
    gap = overnight_gap(daily)
    v5 = intraday_vol(today)
    mom = short_momentum(today)
    zbal = price_zscore(intra["Close"], price)

    # Grafikui: 5 MIN. barai (smulki kreive), paskutines 5 sesijos.
    # Horizontale zymima valandomis, ne dienomis.
    grafikas = []
    try:
        ser = intra["Close"].dropna()
        dienos = sorted({i.date() for i in ser.index})[-5:]
        ser = ser[[i.date() in set(dienos) for i in ser.index]]
        grafikas = [(i, float(v)) for i, v in ser.items()]
    except Exception:
        pass
    rsi2_v = rsi2_daily(daily["Close"])

    # IBS ir nakties tarpas — rodomi kaip informacija. I bala neijungti, kol
    # nepatvirtinta tavo akcijose (backtest_intraday.py juos matuoja atskirai).
    ibs = (price - low) / (high - low) if high > low else None
    day_open = float(today["Open"].iloc[0]) if len(today) else None
    gap_ret = ((day_open - prev_close) / prev_close * 100
               if day_open and prev_close else None)
    # Valandos sandoriui svarbios šios dienos lubos, ne 20 d. swing lygiai
    res_intra = high if high > price * 1.001 else None
    sup_intra = low if low < price * 0.999 else None
    # Visos lubos virs kainos, nuo artimiausios: dienos max, pivotas, 20 d. virsune.
    # Dienos maksimumas dazniausiai pramusamas, todel jis - ispejimas, ne kliutis.
    cands = sorted(x for x in (res_intra, res, float(daily["High"].tail(20).max()))
                   if x and x > price * 1.001)
    day_chg = (price - prev_close) / prev_close * 100 if prev_close else None

    return dict(tag=tag, sym=sym, name=name, price=price, dayHigh=high, dayLow=low,
                vwap=vwap, rsi=r, atrPct=a, support=sup, resistance=res, rvol=rv,
                # ETC (zaliavos) ataskaitu neturi — uzklausos joms nesiunciam
                sma20=sma20, sma50=sma50,
                earnings=(False if SECTORS.get(sym) == "Zaliavos"
                          else earnings_soon(yf, sym)),
                sector=SECTORS.get(sym, "kita"), day_chg=day_chg, avgVolume=avg_vol,
                cur=currency_of(sym)[0], cur_sym=currency_of(sym)[1], gap=gap,
                ibs=ibs, gap_ret=gap_ret, zscore=zbal, rsi2=rsi2_v, grafikas=grafikas,
                vol5m=v5, res_intra=res_intra, sup_intra=sup_intra, res_list=cands,
                m1h=mom["m1h"], m3h=mom["m3h"], pos1h=mom["pos1h"],
                span_h=mom["span_h"], mom_partial=mom["partial"],
                exp_move=expected_move(v5, HOLD_HOURS),
                down_days=ctx["down_days"], dd5=ctx["dd5"], chg3d=ctx["chg3d"],
                asOf=str(intra.index[-1]))



# ----------------------------- SAVIKONTROLE -----------------------------

def sanity_check(rows, market):
    """Ieskо poziymiu, kad modulis pats veikia netinkamai.

    Sitos patikros atsirado is realiu klaidu: ATR virsdavo 'nan' ir kriterijus
    tyliai isjungdavo; rinkos filtras uzblokuodavo visas 19 akciju; vienas
    kriterijus visiems duodavo 95 balus. Kiekviena tokia klaida atrodo kaip
    normalus rezultatas, jei nezinai, ko ieskoti.
    """
    warn = []
    n = len(rows)
    if not n:
        return warn

    scored = [(d, s) for d, s in rows if not s.get("empty")]
    if not scored:
        return ["Nė viena akcija neturi duomenų — tikėtina duomenų šaltinio problema"]

    grades = [s["grade"] for _, s in scored]
    if len(set(grades)) == 1 and len(scored) > 5:
        warn.append(f"Visos {len(scored)} akcijos gavo tą pačią pakopą ({grades[0]}) — "
                    f"greičiausiai suveikė bendras filtras, o ne akcijų savybės")

    blocked = [s for _, s in scored if s.get("blocking")]
    if len(blocked) == len(scored) and len(scored) > 5:
        reasons = {}
        for s in blocked:
            key = s["blocking"][0][:40]
            reasons[key] = reasons.get(key, 0) + 1
        top = max(reasons.items(), key=lambda x: x[1])
        warn.append(f"Visos akcijos užblokuotos, {top[1]} iš jų ta pačia priežastimi: "
                    f"„{top[0]}…“ — patikrink, ar filtras nėra per griežtas")

    # Kriterijus laikomas sugedusiu tik tada, kai reiksmes yra TIKSLIAI vienodos.
    # Ankstesne, svelnesne riba duodavo klaidingu ispejimu, o tokia savikontrole
    # yra blogesne uz jokia — ja imama ignoruoti.
    broken = []
    for key, label, _ in CRITERIA:
        vals = [s["parts"].get(key) for _, s in scored if s.get("parts")]
        vals = [v for v in vals if v is not None]
        if len(vals) > 5 and max(vals) - min(vals) < 0.01:
            if abs(vals[0] - 50.0) < 0.01:
                broken.append(f"„{label}“ = 50 visoms (trūksta duomenų)")
            else:
                broken.append(f"„{label}“ = {vals[0]:.0f} visoms")
    if broken:
        warn.append("Šie kriterijai visoms akcijoms grąžina tą pačią reikšmę, todėl "
                    "nieko neskiria: " + "; ".join(broken[:4]))

    # Duomenu sviezumas: su 15 min. vėlavimu ir GitHub delsa senesni nei 45 min.
    # duomenys reiskia, kad kazkas neveikia — o balas atrodo normaliai.
    try:
        stamps = [pd.Timestamp(d["asOf"]) for d, _ in scored if d.get("asOf")]
        if stamps:
            newest = max(stamps)
            now = (pd.Timestamp.now(tz=newest.tz) if newest.tzinfo
                   else pd.Timestamp.now())
            amz = (now - newest).total_seconds() / 60
            # Sesijos pradzioje siandienos baru dar NERA: 9:01 Berlyno laiku
            # naujausias egzistuojantis baras yra vakarykstis uzdarymas, o su
            # 15 min. velavimu pirmas siandienos baras pasirodo apie 9:20.
            # Anksciau si patikra tuo metu duodavo klaidinga ispejima.
            nuo_atidarymo = None
            try:
                dabar = new_intl_now()
                if dabar is not None:
                    nuo_atidarymo = dabar - SESSION_OPEN_MIN
            except Exception:
                pass
            per_anksti = nuo_atidarymo is not None and nuo_atidarymo < 35

            if amz > 45 and market_open_now() and not per_anksti:
                warn.append(f"Naujausias baras {amz:.0f} min. senumo, nors birža dirba — "
                            f"duomenys nebeatnaujinami")
    except Exception:
        pass

    prices = [d["price"] for d, _ in scored if d.get("price")]
    if prices and (min(prices) <= 0 or max(prices) / max(min(prices), 0.01) > 5000):
        warn.append("Kainų reikšmės neįtikėtinos — galimai sumaišyti duomenys")

    return warn


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
        print(f"\nAUKŠČIAUSIAS BALAS: {best['tag']} ({best['name']}) — "
              f"atitikimas kriterijams, ne prognozė")
        c = best.get('cur_sym', '')
        print(f"  Įėjimas {c}{best['price']:.2f} | Stop {c}{bs['stop']:.2f} | "
              f"Min. tikslas {c}{bs['tp']:.2f} (toliau slenkantis "
              f"{bs.get('trail_pct', 1.5):.1f}%) | R:R {bs['rr']:.2f} | "
              f"{bs['shares']} vnt. | pozicija {c}{bs['pos_value']:,.0f}")
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
    mkt = {"bull": "kylanti", "bear": "krentanti", "bear_soft": "atsigaunanti po kritimo",
           "neutral": "šoninė"}.get(market, "šoninė")

    p = []
    if not strong:
        p.append(f"Rinka {mkt}. Nė viena iš {len(rows)} akcijų šiuo metu neatitinka visų sąlygų: "
                 f"reikia, kad tikslas tilptų iki pasipriešinimo, o rizika/nauda būtų bent {MIN_RR}. "
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


def write_html(rows, market, path, refresh_seconds=None, sector_state=None,
               stats=None, problems=None, yf_mod=None):
    market_lt = {"bull": "kyla", "bear": "krenta", "bear_soft": "kryptis žemyn, šiandien kyla",
                 "neutral": "šoninė"}.get(market, "šoninė")
    overview = market_overview(rows, market, sector_state or {}, TARGET_PCT)
    now_lt = datetime.now(_DTZ) if _DTZ else datetime.now()

    # Dienos kilimai — rodomi visada, nesvarbu koks balas. Matavimas sako, kad
    # pirkti prie dienos virsunes vidutiniskai blogiau, bet tai informacija, kuria
    # vartotojas turi matyti ir spresti pats.
    # Platus judesys — i abi puses. Sektoriai isvardijami sarasu, nes su didesne
    # imtimi bus svarbu matyti, kuris sektorius juda, o kuris ne.
    rally_html = ""
    try:
        chgs = [d["day_chg"] for d, _ in rows if d.get("day_chg") is not None]
        if len(chgs) >= 8:
            med = float(np.median(chgs))
            up = sum(1 for c in chgs if c > 0) / len(chgs) * 100
            secs = {}
            for d, _ in rows:
                if d.get("day_chg") is not None:
                    secs.setdefault(d.get("sector", "kita"), []).append(d["day_chg"])
            sec_med = sorted([(s, float(np.median(v))) for s, v in secs.items() if len(v) >= 2],
                             key=lambda x: -x[1])
            sec_items = "".join(
                f"<div class='si'><span class='sn'>{s}</span>"
                f"<span class='sv {'up' if v > 0 else 'dn'}'>{v:+.1f}%</span></div>"
                for s, v in sec_med)

            wide_up = med > 0.5 and up >= 65
            wide_dn = med < -0.5 and up <= 35
            if wide_up or wide_dn:
                strong = sum(1 for c in chgs if (c > 1.0 if wide_up else c < -1.0))
                title = "Platus kilimas" if wide_up else "Platus kritimas"
                verb = "Kyla" if wide_up else "Krenta"
                cnt = up if wide_up else 100 - up
                cls = "rally" if wide_up else "rally dn"
                rally_html = (
                    f"<div class='{cls}'><div class='rh'>{title}</div>"
                    f"<div class='rb'>{verb} {cnt:.0f}% sąrašo · mediana {med:+.1f}% · "
                    f"{strong} akcijos virš {'+' if wide_up else '−'}1%</div>"
                    f"<div class='secs'>{sec_items}</div></div>")
            elif sec_items:
                rally_html = (f"<div class='rally flat'><div class='rh'>Sektoriai šiandien</div>"
                              f"<div class='secs'>{sec_items}</div></div>")
    except Exception:
        pass

    movers_html = ""
    try:
        def mv_items(sel):
            return "".join(
                f"<div class='mv'><b>{d['tag']}</b>"
                f"<span class='mvc {'up' if d['day_chg'] > 0 else 'dn'}'>{d['day_chg']:+.1f}%</span>"
                f"<span class='mvi'>IBS {(d.get('ibs') or 0):.2f}</span>"
                f"<span class='mvs'>{s.get('setup','')}</span>"
                f"</div>"
                for d, s in sel)

        risers = sorted([(d, s) for d, s in rows
                         if d.get("day_chg") is not None and d["day_chg"] > 0.5],
                        key=lambda x: -x[0]["day_chg"])[:5]
        # Krentancios rodomos tik tos, kurias modulis laiko tesiancioms kritima
        fallers = sorted([(d, s) for d, s in rows
                          if d.get("day_chg") is not None and d["day_chg"] < -0.5
                          and (s.get("setup") == "Krintantis peilis"
                               or (s.get("down_days") or 0) >= 2)],
                         key=lambda x: x[0]["day_chg"])[:5]
        cols = ""
        if risers:
            cols += f"<div class='mcol'><div class='mh'>Šiandien kyla</div>{mv_items(risers)}</div>"
        if fallers:
            cols += (f"<div class='mcol'><div class='mh'>Krenta ir tęsia kritimą</div>"
                     f"{mv_items(fallers)}</div>")
        if cols:
            movers_html = f"<div class='movers'>{cols}</div>"
    except Exception:
        pass

    # Pagrindinis atsakymas: ar siandien apskritai verta prekiauti. Backteste tai
    # buvo vienintelis efektas, ~10 kartu didesnis uz akciju atranka
    # (-0.45% krentanciomis dienomis pries +0.61% kylanciomis).
    tinkami = [1 for _, s in rows if s.get("tradeable")]
    if market == "bear":
        v_cls, v_txt = "no", "Šiandien geriau neprekiauti"
        v_sub = ("Rinka krenta ir šiandien. Per 2 metus tokiomis dienomis vidutinis "
                 "sandoris prarado 0,45%, o geriausiai įvertinti — 0,35%.")
    elif not tinkami:
        v_cls, v_txt = "no", "Šiandien nėra tinkamų kandidatų"
        v_sub = "Nė viena akcija nepraeina kietųjų filtrų. Praleisti dieną irgi yra sprendimas."
    elif market == "bear_soft":
        v_cls, v_txt = "care", f"Atsargiai · {len(tinkami)} kandidatai"
        v_sub = "Bendra kryptis vis dar žemyn, nors šiandien kyla."
    else:
        v_cls, v_txt = "ok", f"Galima prekiauti · {len(tinkami)} kandidatai"
        v_sub = "Rinkos režimas netrukdo. Sąrašas žemiau — peržiūrai, ne pirkimo eilei."
    verdict_html = (f"<div class='verdict {v_cls}'><div class='vt'>{v_txt}</div>"
                    f"<div class='vs'>{v_sub}</div></div>")

    problems_html = ""
    if problems:
        items = "".join(f"<li>{w}</li>" for w in problems)
        problems_html = (f"<div class='selfcheck'><b>Modulio savikontrolė įspėja</b>"
                         f"<ul>{items}</ul>"
                         f"<i>Tai gali reikšti modulio, o ne rinkos problemą — "
                         f"prieš remiantis šiuo sąrašu, pasitikrink grafike.</i></div>")

    stats_html = ""
    if stats:
        done = sum(b["n"] for b in stats.values())
        if done >= 5:
            lines = []
            for g, b in sorted(stats.items(), key=lambda x: -x[1]["n"]):
                if not b["n"]:
                    continue
                pct = b["tikslas"] / b["n"] * 100
                pel = b.get("pelnai") or []
                vid = (f"{sum(pel) / len(pel):+.2f}%" if pel else "—")
                lines.append(f"<div class='srow'><span class='sn2'>{g}</span>"
                             f"<i><b style='width:{min(pct, 100):.0f}%'></b></i>"
                             f"<u>{pct:.0f}% · {vid} · n={b['n']}</u></div>")
            stats_html = ("<div class='stats'><div class='sh'>Live backtest — tavo "
                          "realūs sandoriai pagal scenarijų<br>"
                          "(dalis pasiekusių tikslą · vidutinis rezultatas · kiek sandorių)"
                          "</div>" + "".join(lines) +
                          f"<div class='sn3'>Iš viso užbaigtų: {done}. "
                          f"Patikimai vertinti galima nuo ~40.</div></div>")

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

    def nws(d):
        """Antrastes su apytiksliu atspalviu. Zalsvas / rausvas fonas — tik
        raktazodziu paieskos rezultatas, todel antraste rodoma visa."""
        n = d.get("naujienos") or []
        if not n:
            return ""
        eil = []
        for antr, salt, val, atsp in n[:3]:
            laikas = (f"prieš {val:.0f} val." if val is not None and val < 48
                      else ("prieš {:.0f} d.".format(val / 24) if val else "—"))
            zenklas = {"teig": "+", "neig": "−", "neutr": "·"}[atsp]
            eil.append(f"<tr class='n{atsp}'><td class='nz'>{zenklas}</td>"
                       f"<td class='nt'>{antr}</td>"
                       f"<td class='ns'>{salt or '—'}</td>"
                       f"<td class='nl'>{laikas}</td></tr>")
        return f"<table class='news'>{''.join(eil)}</table>"

    def bars(d, s):
        """Kainos grafikas. Visi stiliai — TIESIOGIAI elementuose.

        Anksciau stiliai buvo CSS klasese, o joms dingus polyline buvo uzpildomas
        juodai (SVG numatytoji fill reiksme yra black). Su tiesioginiais atributais
        grafikas atrodo taip pat, net jei stiliu lentele pasikeistu.
        """
        taskai = d.get("grafikas") or []
        if len(taskai) < 10:
            return ""

        W, H = 700, 170
        PAD_L, PAD_R, PAD_T, PAD_B = 54, 8, 14, 18
        LINIJA, PILKA, TEKST = "#2F7A57", "#E3E1DC", "#8A857E"
        FS = 9                                   # toks pat kaip smulkus puslapio tekstas

        kainos = [v for _, v in taskai]
        lo, hi = min(kainos), max(kainos)
        if hi <= lo:
            return ""
        marza = (hi - lo) * 0.10
        lo, hi = lo - marza, hi + marza
        n = len(taskai)

        def X(i):
            return PAD_L + i * (W - PAD_L - PAD_R) / max(1, n - 1)

        def Y(v):
            return PAD_T + (hi - v) * (H - PAD_T - PAD_B) / (hi - lo)

        # --- Kainos asis: smulkus zingsnis ---
        diap = hi - lo
        z = 10 ** math.floor(math.log10(diap / 4)) if diap > 0 else 1
        for m in (1, 2, 2.5, 5, 10):
            if diap / (z * m) <= 5:
                z *= m
                break
        cs_ = d.get("cur_sym", "")
        tiksl = 2 if z >= 0.05 else 3
        dalys, v = [], math.ceil(lo / z) * z
        while v <= hi:
            y = Y(v)
            dalys.append(
                f"<line x1='{PAD_L}' y1='{y:.1f}' x2='{W - PAD_R}' y2='{y:.1f}' "
                f"stroke='{PILKA}' stroke-width='.8'/>"
                f"<text x='{PAD_L - 5}' y='{y + 3:.1f}' font-size='{FS}' fill='{TEKST}' "
                f"text-anchor='end'>{cs_}{v:.{tiksl}f}</text>")
            v += z

        # --- Horizontale: valandos. Kai dienu daugiau, zymos retesnes,
        # kitaip skaiciai susigrudzia ir tampa neiskaitomi.
        dienu = len({ts.date() for ts, _ in taskai})
        zymos_kas = 2 if dienu <= 3 else 3
        pr_h, pr_d = None, None
        for i, (ts, _) in enumerate(taskai):
            if ts.hour != pr_h:
                nauja = pr_d is not None and ts.date() != pr_d
                if nauja:
                    dalys.append(
                        f"<line x1='{X(i):.1f}' y1='{PAD_T}' x2='{X(i):.1f}' "
                        f"y2='{H - PAD_B}' stroke='{TEKST}' stroke-width='.8' "
                        f"stroke-dasharray='3 3' opacity='.4'/>")
                if ts.hour % zymos_kas == 0 or nauja:
                    dalys.append(
                        f"<text x='{X(i):.1f}' y='{H - 5:.1f}' font-size='{FS}' "
                        f"fill='{TEKST}' text-anchor='middle'>{ts.hour:02d}</text>")
                pr_h, pr_d = ts.hour, ts.date()

        # --- Kreive ---
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, (_, v) in enumerate(taskai))
        dalys.append(f"<polygon points='{PAD_L},{H - PAD_B} {pts} "
                     f"{X(n - 1):.1f},{H - PAD_B}' fill='{LINIJA}' opacity='.06' "
                     f"stroke='none'/>")
        dalys.append(f"<polyline points='{pts}' fill='none' stroke='{LINIJA}' "
                     f"stroke-width='1.3' stroke-linejoin='round'/>")

        # --- Trendo linija ---
        xs = list(range(n))
        mx, my = sum(xs) / n, sum(kainos) / n
        var = sum((x - mx) ** 2 for x in xs)
        k = (sum((x - mx) * (y - my) for x, y in zip(xs, kainos)) / var) if var else 0.0
        y0, y1 = my + k * (0 - mx), my + k * (n - 1 - mx)
        sp = "#2F7A57" if k > 0 else "#C25C55"
        pok = k * (n - 1) / kainos[0] * 100 if kainos[0] else 0
        dalys.append(
            f"<line x1='{X(0):.1f}' y1='{Y(max(lo, min(hi, y0))):.1f}' "
            f"x2='{X(n - 1):.1f}' y2='{Y(max(lo, min(hi, y1))):.1f}' stroke='{sp}' "
            f"stroke-width='1.2' stroke-dasharray='6 4' opacity='.75'/>"
            f"<text x='{PAD_L + 3}' y='{PAD_T + 8}' font-size='{FS}' fill='{sp}' "
            f"font-weight='600'>trendas {pok:+.1f}%</text>")

        # --- Dabartine kaina ir stop ---
        dab = taskai[-1][1]
        dalys.append(f"<circle cx='{X(n - 1):.1f}' cy='{Y(dab):.1f}' r='3' "
                     f"fill='{LINIJA}'/>")
        st = s.get("stop")
        if st and lo <= st <= hi:
            dalys.append(
                f"<line x1='{PAD_L}' y1='{Y(st):.1f}' x2='{W - PAD_R}' y2='{Y(st):.1f}' "
                f"stroke='#C25C55' stroke-width='1' stroke-dasharray='4 3' opacity='.8'/>"
                f"<text x='{W - PAD_R - 2}' y='{Y(st) - 4:.1f}' font-size='{FS}' "
                f"fill='#C25C55' text-anchor='end'>stop</text>")

        return (f"<svg viewBox='0 0 {W} {H}' style='width:100%;height:auto;"
                f"display:block;margin:6px 0 2px'>" + "".join(dalys) + "</svg>")

    # Rodom praejusias filtrus. Jei tokiu nera — rodom artimiausias su aiskia
    # zyma, kodel netinka. Anksciau puslapis likdavo tuscias be paaiskinimo,
    # ir nesimate, ar tai teisingas verdiktas, ar modulio gedimas.
    tinkami = [(d, s) for d, s in rows if s.get("tradeable")]
    rodomi = tinkami[:RODOMA]
    tik_informacijai = False
    if not rodomi:
        tik_informacijai = True
        # Artimiausi: be blokuojanciu zymu, tada pagal apyvarta
        be_kliuciu = [(d, s) for d, s in rows if not s.get("blocking")]
        rodomi = (be_kliuciu or rows)[:RODOMA]

    info_html = ""
    if tik_informacijai and rodomi:
        info_html = ("<div class='rally flat'><div class='rh'>Šiandien nė viena "
                     "nepraėjo filtrų</div><div class='rb'>Žemiau — artimiausios "
                     "pagal apyvartą, tik informacijai. Kiekvienos kortelėje "
                     "nurodyta, kas netinka.</div></div>")
    # Naujienos siunciamos TIK rodomoms pozicijoms — penkios uzklausos, ne 270
    for d, _ in rodomi:
        if "naujienos" not in d:
            d["naujienos"] = naujienos(yf_mod, d["sym"]) if yf_mod else []
    cards = []
    for i, (d, s) in enumerate(rodomi, 1):
        fl = "".join(f"<li class='{lvl}'>{txt}</li>" for lvl, txt in s["flags"])
        cs = d.get("cur_sym", "")
        cur = d.get("cur", "")
        cards.append(f"""
        <details class="card c{s.get('spalva','gelt')}" {'open' if i == 1 else ''}>
          <summary><span class="rk">{i}</span><span class="tk">{d['tag']}</span>
            <span class="px">{cs}{d['price']:.2f}</span>
            <span class="fr f{s.get('spalva','gelt')}">{s.get('fraze','')}</span></summary>
          <div class="in">
            <div class="nm">{d['name']} · {d['sym']} · {s.get('setup','')}</div>
            <div class="verdict {'ok' if s.get('tradeable') else 'no'}">{
              'Atitinka visas sąlygas' if s.get('tradeable')
              else 'NETINKAMA: ' + (s['blocking'][0] if s.get('blocking')
                   else f"rizika/nauda {s['rr']:.2f} per maža")}</div>
            <div class="plan"><div><span>Įėjimas</span><b>{cs}{d['price']:.2f}</b></div>
              <div><span>Stop</span><b>{cs}{s['stop']:.2f}</b></div>
              <div><span>Parduoti kai</span><b>RSI(2) virš {s.get('exit_rsi', 70):.0f}</b></div>
              <div><span>arba</span><b>IBS virš {s.get('exit_ibs', 0.8):.2f}</b></div>
              <div><span>IBS dabar</span><b>{(d.get('ibs') or 0):.2f}</b></div>
              <div><span>RSI(2) dabar</span><b>{(s.get('rsi2') or 0):.0f}</b></div>
              <div><span>Judrumas (ATR)</span><b>{(d.get('atrPct') or 0):.1f}%</b></div>
              <div><span>Stop atstumas</span><b>{((d['price'] - s['stop']) / d['price'] * 100):.1f}%</b></div>
              <div><span>Kiekis</span><b>{s['shares']} vnt.</b></div>
              <div><span>Pozicija</span><b>{cs}{s['pos_value']:,.0f}</b></div>
              <div><span>Pelnas ties {TARGET_PCT}%</span><b>{cs}{s['net']:.0f}</b></div>
              <div><span>Rizikuoji</span><b>{cs}{s['real_risk']:.0f}</b></div></div>
            {bars(d, s)}
            {nws(d)}
            <ul class="fl">{fl}</ul>
          </div>
        </details>""")

    html = f"""<!doctype html><html lang="lt"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
{refresh_tag}
<title>Dip reitingas</title><style>
.news{{width:100%;border-collapse:collapse;background:#fff;margin:8px 0 4px;
border:1px solid var(--line);border-radius:6px;overflow:hidden}}
.news td{{padding:7px 9px;border-top:1px solid var(--line);vertical-align:top;
background:#fff}}
.news tr:first-child td{{border-top:none}}
.news .nz{{width:14px;font-weight:700;font-size:13px;text-align:center;
color:var(--ink2)}}
.news tr.nteig .nz{{color:#2F7A57}}
.news tr.nneig .nz{{color:#C25C55}}
.news .nt{{font-size:12px;line-height:1.4}}
.news .ns{{font-size:10.5px;color:var(--ink2);white-space:nowrap;width:70px}}
.news .nl{{font-size:10.5px;color:var(--ink2);white-space:nowrap;width:80px;
text-align:right}}
:root{{--ink:#16233A;--ink2:#54637E;--line:#C9D2E0;--bg:#E9EDF3;--card:#FDFDFB;--up:#1F7A5C;--warn:#B26B00;--stop:#A8322D}}
*{{box-sizing:border-box}}body{{margin:0;padding:22px 16px 50px;background:var(--bg);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;max-width:760px;margin:0 auto}}
h1{{font-size:26px;margin:0 0 6px;font-weight:600;letter-spacing:-0.01em}}
.meta{{font-size:12px;color:var(--ink2);margin-bottom:12px}}
.overview{{font-size:13.5px;line-height:1.6;color:var(--ink);background:var(--card);
border:1px solid var(--line);border-radius:8px;padding:14px;margin-bottom:20px}}
.rally{{background:#E6F2EC;border-left:4px solid var(--up);border-radius:6px;
padding:12px 14px;margin-bottom:18px}}
.rally.dn{{background:#FBEBEA;border-color:var(--stop)}}
.rally.flat{{background:var(--card);border:1px solid var(--line);border-left:1px solid var(--line)}}
.rh{{font-size:12px;font-weight:700;margin-bottom:6px}}
.rb{{font-size:13px;line-height:1.5;margin-bottom:9px}}
.secs{{display:flex;flex-wrap:wrap;gap:6px}}
.si{{display:flex;align-items:center;gap:6px;background:rgba(255,255,255,.65);
padding:4px 9px;border-radius:5px}}
.sn{{font-size:11.5px}}
.sv{{font-size:11.5px;font-weight:700;font-variant-numeric:tabular-nums}}
.sv.up{{color:var(--up)}}
.sv.dn{{color:var(--stop)}}
.movers{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:20px}}
.mcol{{flex:1;min-width:260px;background:var(--card);border:1px solid var(--line);
border-radius:8px;padding:13px}}
.mh{{font-size:11.5px;color:var(--ink2);margin-bottom:9px;font-weight:600}}
.mv{{display:flex;align-items:center;gap:10px;padding:5px 0;font-size:12.5px;
border-top:1px solid var(--bg)}}
.mv b{{min-width:52px}}
.mvc{{font-weight:600;min-width:50px;font-variant-numeric:tabular-nums}}
.mvc.up{{color:var(--up)}}
.mvc.dn{{color:var(--stop)}}
.mvi,.mvb{{color:var(--ink2);font-size:11.5px;font-variant-numeric:tabular-nums}}
.mvs{{color:var(--ink2);font-size:11.5px;flex:1}}
.mn{{font-size:11px;color:var(--ink2);line-height:1.5;margin-top:9px;
padding-top:9px;border-top:1px solid var(--line)}}
.selfcheck{{background:#FBEBEA;border-left:4px solid var(--stop);color:#7A2320;
padding:12px 14px;border-radius:6px;margin-bottom:18px;font-size:13px;line-height:1.5}}
.selfcheck ul{{margin:8px 0;padding-left:18px}}
.selfcheck i{{font-size:11.5px;opacity:.85}}
.stats{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:13px;margin-bottom:20px}}
.sh{{font-size:11.5px;color:var(--ink2);margin-bottom:9px}}
.srow{{display:flex;align-items:center;gap:9px;margin-bottom:5px}}
.srow span.sn2{{font-size:11.5px;font-weight:600;width:200px}}
.sn3{{font-size:11px;color:var(--ink2);margin-top:9px;padding-top:8px;
border-top:1px solid var(--line)}}
.srow i{{flex:1;height:5px;background:#E4E9F1;border-radius:3px;overflow:hidden}}
.srow b{{display:block;height:100%;background:var(--up)}}
.srow u{{font-size:11px;color:var(--ink2);width:135px;text-align:right;text-decoration:none;
font-variant-numeric:tabular-nums}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;margin-bottom:7px;overflow:hidden}}
summary{{display:flex;align-items:center;gap:10px;padding:12px;cursor:pointer;list-style:none}}
summary::-webkit-details-marker{{display:none}}
.rk{{font-size:11px;color:var(--ink2);width:16px}}
.tk{{font-weight:600;font-size:15px;width:74px;flex:none;
overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.bar{{flex:1;height:5px;background:#DFE5EE;border-radius:3px;overflow:hidden}}
.bar i{{display:block;height:100%;background:var(--ink)}}
.px{{font-size:12.5px;color:var(--ink2);font-variant-numeric:tabular-nums;
width:82px;flex:none;text-align:right;padding-right:14px}}
.fr{{font-size:12px;flex:1;text-align:left;white-space:nowrap;color:var(--ink)}}
.fzal,.fgelt{{color:var(--ink)}}
.fraus{{color:var(--stop)}}
.card.craus{{border-left:3px solid var(--stop)}}
.sc_nenaudojamas{{font-size:13px;width:26px;text-align:right;font-variant-numeric:tabular-nums}}
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
<h1>Kandidatų filtras</h1>
<div class="meta">Atnaujinta {now_lt:%H:%M} (Vilnius) · duomenys iš {data_lt} ·
tikslas {TARGET_PCT}% · rinka: {market_lt} · {len(rows)} akcijos</div>
{problems_html}
{verdict_html}
{info_html}
{rally_html}
{movers_html}
{stats_html}
{''.join(cards)}
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)



# ----------------------------- REZULTATU ZURNALAS -----------------------------

MODEL_VERSION = "2026-09-08 z35-ibs25-vwap15-salyginis-isejimas"   # keiciant svorius ar isejima — atnaujink

JOURNAL_FIELDS = ["versija", "data", "laikas", "sym", "tag", "balas", "pakopa", "scenarijus",
                  "tinkamas", "ibs", "rinka", "sektorius", "atr", "ijejimas", "stop",
                  "min_tikslas", "busena", "rezultatas", "baigties_laikas",
                  "baigties_kaina", "pelnas_pct", "virsune_pct"]
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

        entry_px = float(entry["ijejimas"])
        stop = float(entry["stop"])
        trigger = float(entry["min_tikslas"])

        # Rezultatas skaiciuojamas TA PACIA taisykle, kuria rekomenduoja modulis.
        # Prie salyginio isejimo tai reiskia: parduodam, kai dienos IBS pakyla virs
        # EXIT_IBS, arba suveikia apsauginis stop, arba praeina EXIT_MAX_DIENU.
        if EXIT_MODE == "salyginis":
            entry_px = float(entry["ijejimas"])
            stop_k = entry_px * (1 - EXIT_STOP_PCT / 100)
            peak = entry_px
            for ts, bar in after.iterrows():
                hi, lo, cl = float(bar["High"]), float(bar["Low"]), float(bar["Close"])
                peak = max(peak, hi)
                if lo <= stop_k:
                    pnl = (stop_k - entry_px) / entry_px * 100
                    entry.update(busena="baigta", rezultatas="stop",
                                 baigties_laikas=str(ts), baigties_kaina=f"{stop_k:.2f}",
                                 pelnas_pct=f"{pnl:+.2f}",
                                 virsune_pct=f"{(peak - entry_px) / entry_px * 100:+.2f}")
                    return entry
                d_rng = hi - lo
                ibs_now = (cl - lo) / d_rng if d_rng > 0 else 0.5
                # RSI(2) is dienos uzdarymu iki sio baro imtinai. Anksciau zurnalas
                # tikrino TIK IBS, nors kortele zada "RSI(2) virs 70 ARBA IBS virs
                # 0.8" — live testas matavo kita taisykle, nei rodo puslapis.
                rsi_now = None
                try:
                    dien = intra["Close"][intra.index <= ts].resample("1D").last().dropna()
                    if len(dien) >= 4:
                        rsi_now = rsi2_daily(dien)
                except Exception:
                    pass
                salyga = (ibs_now >= EXIT_IBS
                          or (rsi_now is not None and rsi_now >= EXIT_RSI))
                if salyga and cl > entry_px:
                    pnl = (cl - entry_px) / entry_px * 100
                    entry.update(busena="baigta",
                                 rezultatas=("salyga (RSI)" if (rsi_now is not None
                                             and rsi_now >= EXIT_RSI) else "salyga (IBS)"),
                                 baigties_laikas=str(ts), baigties_kaina=f"{cl:.2f}",
                                 pelnas_pct=f"{pnl:+.2f}",
                                 virsune_pct=f"{(peak - entry_px) / entry_px * 100:+.2f}")
                    return entry
            last_ts = after.index[-1]
            if (last_ts.date() - start.date()).days >= EXIT_MAX_DIENU:
                cl = float(after["Close"].iloc[-1])
                pnl = (cl - entry_px) / entry_px * 100
                entry.update(busena="baigta", rezultatas="laikas",
                             baigties_laikas=str(last_ts), baigties_kaina=f"{cl:.2f}",
                             pelnas_pct=f"{pnl:+.2f}",
                             virsune_pct=f"{(peak - entry_px) / entry_px * 100:+.2f}")
            return entry

        # Fiksuoto isejimo variantas (EXIT_MODE = "fiksuotas"):
        # pasiekus minimalu tiksla ijungiamas slenkantis stop TRAIL_PCT nuo virsunes.
        # Anksciau cia buvo fiksuotas tikslas — zurnalas rodydavo mazesni pelna,
        # nei realiai duotu modulio rekomenduojamas isejimas.
        armed = False
        peak = entry_px
        cur_stop = stop

        for ts, bar in after.iterrows():
            hi, lo = float(bar["High"]), float(bar["Low"])
            if lo <= cur_stop:
                pnl = (cur_stop - entry_px) / entry_px * 100
                entry.update(busena="baigta",
                             rezultatas="slenkantis stop" if armed else "stop",
                             baigties_laikas=str(ts), baigties_kaina=f"{cur_stop:.2f}",
                             pelnas_pct=f"{pnl:+.2f}",
                             virsune_pct=f"{(peak - entry_px) / entry_px * 100:+.2f}")
                return entry
            if hi > peak:
                peak = hi
            if not armed and hi >= trigger:
                armed = True
            if armed:
                cur_stop = max(cur_stop, peak * (1 - TRAIL_PCT / 100))

        # Nei tikslas, nei stop. Jei nuo irasymo praejo daugiau nei diena - uzdarom.
        last_ts = after.index[-1]
        # Laikymas 3 sesijos (HOLD_HOURS=72), todel uzdarom tik po 3 dienu,
        # o ne po vienos — anksciau zurnalas fiksuodavo per anksti.
        if (last_ts.date() - start.date()).days >= HOLD_DAYS:
            last_close = float(after["Close"].iloc[-1])
            pnl = (last_close - entry_px) / entry_px * 100
            entry.update(busena="baigta",
                         rezultatas="uzdaryta pabaigoje" if armed else "be rezultato",
                         baigties_laikas=str(last_ts), baigties_kaina=f"{last_close:.2f}",
                         pelnas_pct=f"{pnl:+.2f}",
                         virsune_pct=f"{(peak - entry_px) / entry_px * 100:+.2f}")
        return entry
    except Exception:
        return entry


def update_journal(path, rows, intraday_all, now, market="neutral"):
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
                versija=MODEL_VERSION, data=today, laikas=now.strftime("%H:%M"), sym=d["sym"], tag=d["tag"],
                balas=f"{s['score']:.1f}", pakopa=s["grade"],
                scenarijus=s.get("fraze", s.get("setup", "")),
                tinkamas="taip" if s.get("tradeable") else "ne",
                ibs=f"{d['ibs']:.3f}" if d.get("ibs") is not None else "",
                rinka=market, sektorius=d.get("sector", ""),
                atr=f"{d['atrPct']:.2f}" if d.get("atrPct") else "",
                ijejimas=f"{d['price']:.2f}", stop=f"{s['stop']:.2f}",
                min_tikslas=f"{s['tp']:.2f}", busena="atviras", rezultatas="",
                baigties_laikas="", baigties_kaina="", pelnas_pct="", virsune_pct=""))
            added += 1

        entries = entries[-500:]          # neauginam failo be galo
        save_journal(path, entries)
        return entries
    except Exception:
        return []


def journal_stats(entries):
    """Statistika pagal SCENARIJU, ne pagal raide.

    Raides (A/B/C/D) nieko nesako apie tai, kas realiai vyko — "atsitraukimas,
    kryptis stabilizavosi" ir "kylancio trendo tesinys" gali turėti ta pati bala,
    bet tai skirtingi sandoriai. Seni irasai su raidemis sugrupuojami atskirai.
    """
    out = {}
    for e in entries:
        if e.get("busena") != "baigta":
            continue
        # Naujuose irasuose scenarijus yra fraze; senuose — "kritimas"/"atsigavimas"
        # arba tuscia, tada griztam prie raides
        # Senuose irasuose scenarijai rasyti mazaja raide ("kritimas"), naujuose —
        # didziaja. Suvienodinam, kad statistikoje nesidubliuotu dvi eilutes.
        g = (e.get("scenarijus") or "").strip()
        g = (g[0].upper() + g[1:]) if g else f"(sena pakopa {e.get('pakopa', '?')})"
        b = out.setdefault(g, {"n": 0, "tikslas": 0, "stop": 0, "kita": 0, "pelnai": []})
        b["n"] += 1
        # Salyginio isejimo rezultatai: "salyga (IBS)" = isejimas ivykus salygai,
        # "laikas" = pasibaige laikymo langas, "stop" = apsauginis stop.
        # Be sio atnaujinimo nauji irasai butu skaiciuojami kaip "kita" ir
        # statistika rodytu nuli.
        r = e.get("rezultatas", "")
        if r in ("salyga (IBS)", "salyga (RSI)", "slenkantis stop", "uzdaryta pabaigoje"):
            b["tikslas"] += 1
        elif r == "stop":
            b["stop"] += 1
        else:                       # "laikas", "be rezultato" — nei viena, nei kita
            b["kita"] += 1
        try:
            b["pelnai"].append(float(e.get("pelnas_pct", "")))
        except (TypeError, ValueError):
            pass
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

    # 1 pakopa: dienos duomenys visoms
    daily_all = collect_daily(yf, symbols)
    market = market_bias(yf)

    # 2 pakopa: 5 min. duomenys tik preatranka praejusioms
    atrinkti = preatranka(daily_all, symbols)
    if not quiet:
        print(f"Preatranka: {len(atrinkti)} is {len(symbols)} akciju tikrinamos "
              f"5 min. duomenimis")
    intraday_all = collect_intraday(yf, atrinkti)
    symbols_intraday = set(atrinkti)

    rows, failed = [], []
    collected = []
    for tag, sym, name in WATCHLIST:
        if sym not in symbols_intraday:      # 5 min. duomenu siai akcijai nesiuntem
            continue
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

    # Rinkos plotis is paties saraso — tikslesnis nei indeksas, nes tai TAVO akcijos.
    # Jei dauguma ju siandien kyla, tai nera krentanti diena, kad ir ka rodo 5 d. kryptis.
    chgs_all = [x["day_chg"] for x in collected if x.get("day_chg") is not None]
    if chgs_all:
        breadth = float(np.median(chgs_all))
        rising = sum(1 for x in chgs_all if x > 0) / len(chgs_all) * 100
        if market == "bear" and (breadth > 0.4 or rising > 60):
            market = "bear_soft"
        elif market == "bear_soft" and breadth > 1.0 and rising > 70:
            market = "neutral"
        print(f"Rinkos plotis: mediana {breadth:+.2f}%, kyla {rising:.0f}% akciju "
              f"-> rezimas: {market}")

    tb = time_budget()
    for d in collected:
        rows.append((d, score_stock(d, TARGET_PCT, market,
                                    sector_chg=sector_state.get(d["sector"]), tb=tb)))

    if not rows:
        print("Nepavyko gauti nė vienos akcijos duomenų šį kartą. Bandysiu vėl.")
        return [], failed

    # RIKIAVIMAS NE PAGAL BALĄ.
    # Matavimas (2026-09, 93 akcijos, 716 dienu, demeanuota pagal diena ir taska)
    # parode, kad TOP pagal ranga, BLOGIAUSI ir ATSITIKTINIAI duoda vienoda
    # rezultata — atrankos pranasumo nera. Todel balas nebeleidziamas nustatyti
    # eiles: jis lieka kortelėje kaip informacija, kiek kriteriju sutampa.
    # Eile nustatoma pagal LIKVIDUMA — vienintelis dydis, kuris tau realiai
    # svarbus (mazesne ivykdymo kaina) ir kuris nieko neprognozuoja.
    def _apyvarta(x):
        d = x[0]
        av, px = d.get("avgVolume"), d.get("price")
        return (av * px) if (av and px) else 0.0

    rows.sort(key=lambda x: (bool(x[1].get("tradeable")), _apyvarta(x)), reverse=True)
    print_table(rows)
    if failed:
        print("Nepavyko:", ", ".join(f"{t} ({m})" for t, _, m in failed), "\n")

    # Rezultatų žurnalas: įrašom šiandienos geriausius, užbaigiam senus įrašus
    now_local = datetime.now(_DTZ) if _DTZ else datetime.now()
    entries = update_journal(os.path.join(out_dir, "zurnalas.csv"),
                             rows, intraday_all, now_local, market=market)
    stats = journal_stats(entries)
    if stats:
        print("\nŽurnalas pagal scenarijų (užbaigti sandoriai):")
        for g, b in sorted(stats.items(), key=lambda x: -x[1]["n"]):
            if b["n"]:
                pel = b.get("pelnai") or []
                vid = f"{sum(pel)/len(pel):+.2f}%" if pel else "—"
                print(f"  {g[:38]:<40} n={b['n']:>3}  tikslą {b['tikslas']/b['n']*100:>5.1f}%"
                      f"  vid. {vid}")

    problems = sanity_check(rows, market)
    if problems:
        print("\n!!! SAVIKONTROLE RADO GALIMU PROBLEMU:")
        for w in problems:
            print(f"  - {w}")
        print()

    html_path = os.path.join(out_dir, "index.html")
    write_html(rows, market, html_path, refresh_seconds=refresh_seconds,
               sector_state=sector_state, stats=stats, problems=problems, yf_mod=yf)

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
