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
    _TZ = ZoneInfo("Europe/Berlin")
except Exception:
    _TZ = None

# ----------------------------- NUSTATYMAI -----------------------------

TARGET_PCT = 2.0        # tikslinis pelnas procentais
ACCOUNT = 18000.0       # sąskaitos dydis, EUR
RISK_PCT = 1.0          # rizika vienam sandoriui, % nuo sąskaitos
MAX_POSITION_PCT = 30.0 # daugiausia % portfelio į vieną poziciją
FEE_PER_TRADE = 2.0     # brokerio mokestis vienam sandoriui (pirkimas ARBA pardavimas), EUR

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
    ("dip",      "Kritimo gylis",            18),
    ("multiday", "Vienadienis ar tęstinis",  14),
    ("room",     "Vieta iki pasipriešinimo", 14),
    ("atr",      "Judrumas (ATR)",           14),
    ("rsi",      "RSI (5 min)",              10),
    ("vwap",     "Padėtis prieš VWAP",       10),
    ("rvol",     "Apyvarta (RVOL)",           8),
    ("trend",    "Trendas (20/50 SMA)",       8),
    ("support",  "Atstumas iki atramos",      4),
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
    h, l, c = daily["High"], daily["Low"], daily["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    return float(a / c.iloc[-1] * 100)


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
    price = d["price"]
    high, low = d["dayHigh"], d["dayLow"]
    support = d.get("support") or low
    resistance = d.get("resistance") or high
    vwap = d.get("vwap")
    a_pct = d.get("atrPct")

    dip = (high - price) / high * 100 if high else None
    rng = (price - low) / (high - low) * 100 if high and low and high > low else None
    sup_d = (price - support) / price * 100 if support else None
    room = (resistance - price) / price * 100 if resistance else None
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
    # dd5 = kritimas nuo 5 d. maksimumo. Sveikas dip: 1.5-4%. Tęstinis slydimas: 7%+
    multiday_part = curve(dd5, [(0, 25), (1, 60), (2, 95), (4, 100), (6, 65), (9, 30), (14, 8)])
    if down_days >= 3:
        multiday_part *= 0.45
    elif down_days == 2:
        multiday_part *= 0.75
    if chg3d is not None and chg3d < -6:
        multiday_part *= 0.7

    parts = {
        "dip":  curve(dip,  [(0, 5), (0.5, 35), (1.2, 85), (1.8, 100), (4, 100), (6, 55), (9, 18), (15, 5)]),
        "multiday": multiday_part,
        "room": curve(None if room is None else room / target,
                      [(0.3, 5), (1, 45), (1.5, 70), (2, 90), (3, 100), (6, 95)]),
        "atr":  curve(None if a_pct is None else a_pct / target,
                      [(0.5, 5), (1, 30), (1.5, 60), (2, 90), (3, 100), (5, 85), (8, 55)]),
        "rsi":  curve(d.get("rsi"), [(10, 25), (20, 55), (28, 85), (35, 100), (45, 85), (55, 55), (65, 30), (80, 10)]),
        "vwap": curve(vw_d, [(-4, 20), (-2, 45), (-0.8, 85), (-0.2, 100), (0.3, 90), (1.5, 60), (3, 35), (6, 15)]),
        "rvol": curve(rv,   [(0.3, 15), (0.7, 45), (1, 70), (1.4, 95), (2.5, 100), (4, 80), (7, 55), (12, 35)]),
        "trend": trend,
        "support": 15.0 if (sup_d is not None and sup_d < 0) else
                   curve(sup_d, [(0, 95), (0.3, 100), (1, 85), (2, 55), (3.5, 30), (6, 10)]),
    }

    base = sum(parts[k] * w / 100 for k, _, w in CRITERIA)

    stop = support * 0.997 if support and support < price else price * 0.99
    # stop negali būti nei per platus, nei per ankštas: per ankštą išmuš eilinis triukšmas
    min_stop = max(0.4, target * 0.35, 0.35 * (a_pct or 0))
    if (price - stop) / price * 100 > target * 1.5:
        stop = price * (1 - target * 1.5 / 100)
    if (price - stop) / price * 100 < min_stop:
        stop = price * (1 - min_stop / 100)
    tp = price * (1 + target / 100)
    rr = (tp - price) / (price - stop) if price > stop else 0.0
    risk_cash = ACCOUNT * RISK_PCT / 100
    shares = int(risk_cash / (price - stop)) if price > stop else 0

    # Lubos: viena pozicija negali viršyti MAX_POSITION_PCT portfelio dalies
    max_shares = int((ACCOUNT * MAX_POSITION_PCT / 100) / price) if price > 0 else 0
    capped = shares > max_shares
    if capped:
        shares = max_shares

    pos_value = shares * price
    gross = pos_value * target / 100
    net = gross - 2 * FEE_PER_TRADE      # mokestis perkant ir parduodant
    real_risk = shares * (price - stop)

    flags, mult = [], 1.0
    if d.get("earnings"):
        mult *= 0.35
        flags.append(("stop", "Ataskaita per 2 dienas — kaina šoks bet kuria kryptimi"))
    if market == "bear":
        mult *= 0.85
        flags.append(("warn", "Rinka krenta — pirkimas prieš srovę"))
    elif market == "bull":
        mult *= 1.05
    if room is not None and room < target:
        flags.append(("stop", f"Iki pasipriešinimo tik {room:.1f}% — {target}% tikslas netelpa"))
    if a_pct is not None and a_pct < target * 1.2:
        flags.append(("warn", "Akcija per rami tokiam tikslui per vieną dieną"))
    # --- Laiko biudžetas: ar likusio laiko realiai užtenka tikslui pasiekti? ---
    # Kainos svyravimas auga proporcingai laiko šaknims, todėl tikėtinas likęs
    # judesys = dienos ATR * sqrt(likusi sesijos dalis).
    if tb and a_pct:
        exp_range = a_pct * math.sqrt(max(tb["frac"], 0.01))
        ratio = exp_range / target if target else 0
        hrs = tb["effective"] / 60
        if ratio < 0.7:
            mult *= 0.85
            flags.append(("warn", f"Liko ~{hrs:.1f} val. efektyvios prekybos — tikėtinas "
                                  f"judesys ~{exp_range:.1f}% nesiekia {target}% tikslo; "
                                  f"realiau uždaryti rytoj"))
        elif ratio < 1.1:
            flags.append(("info", f"Liko ~{hrs:.1f} val. — tikslas pasiekiamas, bet be atsargos "
                                  f"(tikėtinas judesys ~{exp_range:.1f}%)"))
        if tb["after_hours"]:
            flags.append(("info", "Pagrindinė sesija baigta — po sesijos prekyboje spread'as "
                                  "platesnis, naudok limit pavedimus"))

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
    if capped and shares > 0:
        flags.append(("info", f"Kiekis apribotas iki {MAX_POSITION_PCT:.0f}% portfelio "
                              f"({pos_value:.0f} EUR) — reali rizika {real_risk:.0f} EUR"))
    if shares > 0 and gross > 0 and net < gross * 0.75:
        flags.append(("warn", f"Mokesčiai suvalgo dalį pelno: bruto {gross:.0f} EUR, "
                              f"neto ~{net:.0f} EUR"))
    if 0 < rr < 1:
        mult *= 0.85
        flags.append(("warn", f"Rizika/nauda {rr:.2f} — rizikuoji daugiau nei sieki"))
    if dip is not None and dip > 8:
        flags.append(("warn", "Kritimas gilus — gali būti krentantis peilis"))
    if down_days >= 3:
        mult *= 0.8
        flags.append(("stop", f"{down_days} kritimo dienos iš eilės — tai ne vienadienis dip, o kryptis"))
    elif down_days == 2:
        flags.append(("warn", "Antra kritimo diena iš eilės — palauk stabilizacijos ženklo"))
    if dd5 is not None and dd5 > 7:
        flags.append(("warn", f"Nuo 5 d. maksimumo nukritusi {dd5:.1f}% — kritimas prasidėjo ne šiandien"))
    if sector_chg is not None and sector_chg < -1.5:
        mult *= 0.9
        flags.append(("warn", f"Visas sektorius krenta ({sector_chg:+.1f}%) — tai ne šios akcijos problema"))
    if rv is not None and rv < 0.7:
        flags.append(("warn", "Apyvarta mažesnė nei įprasta — atšokimas gali neįvykti"))
    if rng is not None and rng < 6:
        flags.append(("warn", "Kaina prie pat dienos dugno — atsigavimo ženklo dar nėra"))
    if sup_d is not None and sup_d < 0:
        flags.append(("stop", "Kaina žemiau atramos — atrama pralaužta"))

    score = max(0.0, min(100.0, base * mult))
    grade = "A" if score >= 78 else "B" if score >= 64 else "C" if score >= 50 else "D"

    return dict(score=score, grade=grade, parts=parts, flags=flags, dip=dip, rng=rng,
                room=room, sup_d=sup_d, vw_d=vw_d, stop=stop, tp=tp, rr=rr, shares=shares,
                down_days=down_days, dd5=dd5, chg3d=chg3d, sector_chg=sector_chg,
                pos_value=pos_value, gross=gross, net=net, real_risk=real_risk)


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
    try:
        idx = yf.download(MARKET_INDEX, period="3mo", interval="1d",
                          progress=False, auto_adjust=False)
        idx = flatten(idx, MARKET_INDEX)
        c = idx["Close"]
        chg = (c.iloc[-1] / c.iloc[-2] - 1) * 100
        sma20 = c.tail(20).mean()
        if chg > 0.3 and c.iloc[-1] > sma20:
            return "bull"
        if chg < -0.3 and c.iloc[-1] < sma20:
            return "bear"
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
    prev_close = float(c.iloc[-2]) if len(c) > 1 else None
    day_chg = (price - prev_close) / prev_close * 100 if prev_close else None

    return dict(tag=tag, sym=sym, name=name, price=price, dayHigh=high, dayLow=low,
                vwap=vwap, rsi=r, atrPct=a, support=sup, resistance=res, rvol=rv,
                sma20=sma20, sma50=sma50, earnings=earnings_soon(yf, sym),
                sector=SECTORS.get(sym, "kita"), day_chg=day_chg, avgVolume=avg_vol,
                cur=currency_of(sym)[0], cur_sym=currency_of(sym)[1],
                down_days=ctx["down_days"], dd5=ctx["dd5"], chg3d=ctx["chg3d"],
                asOf=str(intra.index[-1]))


# ----------------------------- ATASKAITA -----------------------------


def print_table(rows):
    print()
    print(f"{'#':>2}  {'AKCIJA':<7} {'BALAS':>6} {'':2} {'KAINA':>9} {'KRIT.':>7} "
          f"{'ATR':>6} {'RSI':>5} {'RVOL':>5} {'R:R':>5}")
    print("-" * 68)
    for i, (d, s) in enumerate(rows, 1):
        print(f"{i:>2}. {d['tag']:<7} {s['score']:>6.1f} {s['grade']:>2} "
              f"{d.get('cur_sym','')}{d['price']:>8.2f} {(s['dip'] or 0):>6.1f}% "
              f"{d['atrPct']:>5.1f}% {d['rsi']:>5.0f} {(d['rvol'] or 0):>5.2f} {s['rr']:>5.2f}")
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
    strong = [r for r in rows if r[1]["score"] >= 64]
    weak = [r for r in rows if r[1]["score"] < 50]
    mkt = {"bull": "kylanti", "bear": "krentanti", "neutral": "šoninė"}[market]

    p = []
    if not strong:
        p.append(f"Rinka {mkt}, bet nė viena iš {len(rows)} stebimų akcijų šiuo metu nesudaro "
                 f"aiškaus kritimo pirkimo vaizdo. Tokia diena dažniausiai tinka praleisti — "
                 f"geriausias balas tik {max(scores):.0f} iš 100.")
    else:
        best_sec = strong[0][0]["sector"]
        same = sum(1 for r in strong if r[0]["sector"] == best_sec)
        p.append(f"Rinka {mkt}. Iš {len(rows)} akcijų {len(strong)} atitinka kritimo pirkimo "
                 f"kriterijus bent gerai, {len(weak)} šiandien geriau nevertos dėmesio.")
        if same >= 3:
            p.append(f"Svarbu: {same} iš stipriausių pozicijų yra tas pats sektorius "
                     f"({best_sec}). Perkant kelias iš jų, rizika nepasiskirsto — "
                     f"tai iš esmės viena pozicija keliais tikeriais.")

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
    if s["parts"]["multiday"] >= 70 and (s.get("down_days") or 0) <= 1:
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


def write_html(rows, market, path, refresh_seconds=None, sector_state=None):
    market_lt = {"bull": "kyla", "bear": "krenta", "neutral": "šoninė"}[market]
    overview = market_overview(rows, market, sector_state or {}, TARGET_PCT)
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
            <div class="nm">{d['name']} · {d['sym']} · kainos {cur}</div>
            {f'<p class="why">{explain(d, s, TARGET_PCT, rows)}</p>' if i <= 3 else ''}
            <div class="tags">
              <span>kaina {cs}{d['price']:.2f} {cur}</span><span>kritimas {(s['dip'] or 0):.1f}%</span>
              <span>diapazone {(s['rng'] or 0):.0f}%</span><span>iki pasipr. {(s['room'] or 0):.1f}%</span>
              <span>ATR {d['atrPct']:.1f}%</span><span>RSI {d['rsi']:.0f}</span>
              <span>RVOL {(d['rvol'] or 0):.2f}</span><span>VWAP {(s['vw_d'] or 0):+.1f}%</span><span>valiuta {cur}</span>
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
h1{{font-family:Georgia,serif;font-size:27px;margin:0 0 6px;font-weight:600}}
.meta{{font-size:12px;color:var(--ink2);margin-bottom:12px}}
.overview{{font-size:13.5px;line-height:1.6;color:var(--ink);background:var(--card);
border:1px solid var(--line);border-radius:8px;padding:14px;margin-bottom:20px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;margin-bottom:7px;overflow:hidden}}
summary{{display:flex;align-items:center;gap:10px;padding:12px;cursor:pointer;list-style:none}}
summary::-webkit-details-marker{{display:none}}
.rk{{font-size:11px;color:var(--ink2);width:16px}}
.tk{{font-weight:600;font-size:15px;min-width:56px;font-family:ui-monospace,Menlo,monospace}}
.bar{{flex:1;height:5px;background:#DFE5EE;border-radius:3px;overflow:hidden}}
.bar i{{display:block;height:100%;background:var(--ink)}}
.sc{{font-family:ui-monospace,Menlo,monospace;font-size:13px;width:26px;text-align:right}}
.gr{{width:24px;height:24px;border-radius:5px;color:#fff;font-weight:700;font-size:12px;
display:flex;align-items:center;justify-content:center}}
.gA{{background:var(--up)}}.gB{{background:#3F7FA8}}.gC{{background:var(--warn)}}.gD{{background:var(--stop)}}
.in{{padding:0 12px 14px;border-top:1px solid var(--line)}}
.nm{{font-size:12px;color:var(--ink2);margin:10px 0}}
.why{{font-size:12.5px;line-height:1.55;color:var(--ink);background:var(--bg);border-left:3px solid var(--up);
padding:9px 11px;border-radius:5px;margin:0 0 12px}}
.tags{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}}
.tags span{{font-size:11px;background:var(--bg);padding:4px 7px;border-radius:4px;color:var(--ink2);
font-family:ui-monospace,Menlo,monospace}}
.plan{{display:grid;grid-template-columns:repeat(3,1fr);gap:9px;background:var(--ink);color:#F4F7FB;
padding:12px;border-radius:7px;margin-bottom:14px}}
.plan div{{display:flex;flex-direction:column;gap:2px}}
.plan span{{font-size:10px;opacity:.65}}
.plan b{{font-size:14px;font-family:ui-monospace,Menlo,monospace}}
.br{{display:flex;align-items:center;gap:8px;margin-bottom:5px}}
.br span{{font-size:12px;color:var(--ink2);width:150px}}
.br i{{flex:1;height:4px;background:#E4E9F1;border-radius:2px;overflow:hidden}}
.br b{{display:block;height:100%;background:var(--up)}}
.br u{{font-size:11px;color:var(--ink2);width:22px;text-align:right;text-decoration:none;
font-family:ui-monospace,Menlo,monospace}}
.fl{{list-style:none;padding:0;margin:12px 0 0;display:flex;flex-direction:column;gap:6px}}
.fl li{{font-size:12.5px;padding:7px 9px;border-radius:5px;border-left:3px solid}}
.fl .warn{{background:#FBF3E4;border-color:var(--warn);color:#6E4400}}
.fl .stop{{background:#FBEBEA;border-color:var(--stop);color:#7A2320}}
</style>
<h1>Kurią pirkti dabar</h1>
<div class="meta">{datetime.now():%Y-%m-%d %H:%M} · tikslas {TARGET_PCT}% ·
rinka: {market_lt} · {len(rows)} akcijos</div>
<div class="overview">{overview}</div>
{''.join(cards)}
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


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

    rows.sort(key=lambda x: x[1]["score"], reverse=True)
    print_table(rows)
    if failed:
        print("Nepavyko:", ", ".join(f"{t} ({m})" for t, _, m in failed), "\n")

    html_path = os.path.join(out_dir, "index.html")
    write_html(rows, market, html_path, refresh_seconds=refresh_seconds,
               sector_state=sector_state)

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
