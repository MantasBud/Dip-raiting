#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intraday backtestas — testuoja TIKRA sandori: ijejimas dienos viduryje, isejimas ta pacia diena.

Paleidimas:
    pip install yfinance pandas numpy tzdata
    python backtest_intraday.py                # 60 d., visos sarasos akcijos
    python backtest_intraday.py --target 3

KUO SKIRIASI NUO backtest.py
backtest.py naudoja dienos barus: ijejimas uzdarymo kaina, rezultatas kita diena.
Sis naudoja 5 min. barus ir vertina tuos pacius rodiklius, kuriuos mato realus
modulis: VWAP, RSI(5min), RVOL pagal valanda, 1 ir 3 val. krypti, dienos maksimuma.
Ijejimo taskai tikrinami kelis kartus per diena, kaip ir realiai ziurint i moduli.

APRIBOJIMAI
1. Yahoo duoda tik 60 dienu 5 min. istorijos — tai riboja imti.
2. Kai tame paciame bare paliecti ir tikslas, ir stop, laikoma pralaimejimu.
3. Mokesciai ir spread'as neiskaiciuoti.
4. Tie patys keli ijejimo taskai per diena yra susije tarpusavyje, todel
   efektyvus nepriklausomu atveju skaicius mazesnis nei eiluciu skaicius.
"""

import argparse
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

import dip_reitingas as dr


def curve_local(x, pts):
    """Ta pati interpoliacija kaip modulyje — kad V3 butu skaidrus ir patikrinamas."""
    if x is None or not np.isfinite(x):
        return 50.0
    if x <= pts[0][0]:
        return float(pts[0][1])
    if x >= pts[-1][0]:
        return float(pts[-1][1])
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        if x1 <= x <= x2:
            return float(y1 + (x - x1) / (x2 - x1) * (y2 - y1))
    return 50.0

CHECKPOINTS_5M = [18, 42, 66, 84, 96]   # ~10:30, 12:30, 14:30, 16:00, 17:00
CHECKPOINTS_60M = [1, 3, 5, 7, 8]       # ~10:00, 12:00, 14:00, 16:00, 17:00
                                        # (paskutiniai du — po JAV atidarymo 15:30)


def session_frames(intra, min_bars):
    """Suskaido duomenis i sesijas pagal data."""
    return [(d, g) for d, g in intra.groupby(intra.index.date) if len(g) >= min_bars]


def rvol_at(sessions, day_idx, k):
    """Apyvarta iki k-tojo baro, palyginti su tomis paciomis valandomis anksciau."""
    today = sessions[day_idx][1]
    cur = float(today["Volume"].iloc[:k + 1].sum())
    prev = []
    for i in range(max(0, day_idx - 10), day_idx):
        g = sessions[i][1]
        if len(g) > k:
            prev.append(float(g["Volume"].iloc[:k + 1].sum()))
    if not prev or cur <= 0:
        return None
    base = float(np.median(prev))
    return cur / base if base > 0 else None


def build_snapshot(sessions, day_idx, k, daily_hist, rsi_series, target, bph=12,
                   full_series=None):
    """Atkuria tiksliai ta vaizda, kuri modulis matytu ta minute."""
    day, today = sessions[day_idx]
    bars = today.iloc[:k + 1]
    price = float(bars["Close"].iloc[-1])
    high = float(bars["High"].max())
    low = float(bars["Low"].min())
    if not price or high <= low:
        return None

    tp = (bars["High"] + bars["Low"] + bars["Close"]) / 3
    vol = bars["Volume"].replace(0, np.nan)
    vwap = float((tp * vol).sum() / vol.sum()) if vol.sum() > 0 else None

    try:
        rsi = float(rsi_series.loc[bars.index[-1]])
    except Exception:
        rsi = None

    a = dr.atr_pct(daily_hist)
    sup, res = dr.levels(daily_hist, price)
    ctx = dr.multiday_context(daily_hist, price)
    mom = dr.short_momentum(bars, bph=bph)
    v5 = dr.intraday_vol(bars)

    prev_close = float(daily_hist["Close"].iloc[-1])
    day_chg = (price - prev_close) / prev_close * 100 if prev_close else None

    # Kandidatiniai signalai is literaturos (dar neidiegti i bala - pirma matuojam):
    #  IBS = kur kaina dienos diapazone (Pagonidis 2013: IBS<0.2 -> +0.35% kita diena)
    #  gap = nakties tarpas (uzdarymas -> atidarymas), dokumentuotas atsokimo signalas
    ibs = (price - low) / (high - low) if high > low else None

    # IBS VARIANTAS 1: dvieju dienu diapazonas. Gaudo gilesni atsitraukima —
    # akcija gali buti dienos viduryje, bet zemai dvieju dienu masteliu.
    ibs2d = None
    try:
        prev_sess = sessions[day_idx - 1][1] if day_idx >= 1 else None
        if prev_sess is not None and len(prev_sess):
            h2 = max(high, float(prev_sess["High"].max()))
            l2 = min(low, float(prev_sess["Low"].min()))
            ibs2d = (price - l2) / (h2 - l2) if h2 > l2 else None
    except Exception:
        pass
    day_open = float(bars["Open"].iloc[0])
    gap_ret = (day_open - prev_close) / prev_close * 100 if prev_close else None

    # Atidarymo diapazonas: pirma valanda. Kaina virs jo = pramusimas.
    or_bars = bars.iloc[:max(1, bph)]
    or_high = float(or_bars["High"].max())
    or_break = (price - or_high) / price * 100 if or_high else None

    # Atsitraukimas nuo dienos maksimumo, isreikstas ATR dalimis
    a_for_pb = a if a else None
    pullback_atr = ((high - price) / price * 100) / a_for_pb if a_for_pb else None

    # --- PROGNOSTINIAI KANDIDATAI (klasikine technine analize) ---
    # MACD, Z-balas ir svyravimo pletra skaiciuojami is ISTISINES sekos iki sio baro
    # (kaip realiame grafike), o ne is vienos dienos baru — pastarųjų valandiniuose
    # duomenyse yra tik ~9 per sesija, todel indikatoriai negalejo susiskaiciuoti.
    if full_series is not None:
        cutoff = bars.index[-1]
        cont = full_series[full_series.index <= cutoff].tail(120)
    else:
        cont = bars
    cl = cont["Close"].astype(float)
    hi_s, lo_s = cont["High"].astype(float), cont["Low"].astype(float)

    # 1. MACD histogramos zenklas ir kryptis (12/26/9 intraday barais)
    macd_h = None
    if len(cl) >= 30:
        ema12 = cl.ewm(span=12, adjust=False).mean()
        ema26 = cl.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        sig = macd.ewm(span=9, adjust=False).mean()
        hist = macd - sig
        macd_h = float(hist.iloc[-1] / price * 100)

    # 2. Z-balas: kiek standartiniu nuokrypiu kaina nuo 20 baru vidurkio (Bollinger logika)
    zscore = None
    if len(cl) >= 20:
        m20, s20 = float(cl.tail(20).mean()), float(cl.tail(20).std())
        zscore = (price - m20) / s20 if s20 > 0 else None

    # 3. Svyravimo pletra: ar dabartinis judrumas didesnis uz iprasta (Bollinger squeeze)
    vol_exp = None
    if len(cl) >= 30:
        r_now = float((hi_s.tail(6) - lo_s.tail(6)).mean())
        r_base = float((hi_s.tail(30) - lo_s.tail(30)).mean())
        vol_exp = r_now / r_base if r_base > 0 else None

    # 4. Aukstesniu dugnu struktura: kiek is paskutiniu 4 atsitraukimu buvo aukstesni
    hl_struct = None
    day_lo = bars["Low"].astype(float)
    if len(day_lo) >= 12:
        seg = [float(day_lo.iloc[i:i+3].min()) for i in range(len(day_lo)-12, len(day_lo), 3)]
        if len(seg) >= 3:
            hl_struct = sum(1 for a, b in zip(seg, seg[1:]) if b > a) / (len(seg) - 1) * 100

    # 5. Apyvartos ir kainos sutapimas: didele apyvarta + kylanti kaina = patvirtinimas
    vol_price = None
    rv_now = rvol_at(sessions, day_idx, k)
    if rv_now is not None and len(cl) >= 12:
        recent_dir = 1 if float(cl.iloc[-1]) > float(cl.iloc[-min(12, len(cl)-1)]) else -1
        vol_price = rv_now * recent_dir

    # 6. Vakarykscio maksimumo/minimumo pramusimas
    pd_break = None
    try:
        pd_high = float(daily_hist["High"].iloc[-1])
        pd_low = float(daily_hist["Low"].iloc[-1])
        if price > pd_high:
            pd_break = (price - pd_high) / price * 100
        elif price < pd_low:
            pd_break = (price - pd_low) / price * 100
        else:
            pd_break = 0.0
    except Exception:
        pass

    # 7. Paros laikas: kelintas ijejimo taskas (ar yra geresniu valandu)
    tod = k

    # Slankiuju vidurkiu issidestymas: 2 = kaina > SMA20 > SMA50, 0 = zemiau abieju
    sma20_v = float(daily_hist["Close"].tail(20).mean())
    sma50_v = float(daily_hist["Close"].tail(50).mean())
    sma_align = (1 if price > sma20_v else 0) + (1 if sma20_v > sma50_v else 0)

    # SET-UP: atsitraukimas prie kylancio 20 d. vidurkio. Klasikinis trendo pirkimas:
    # akcija virs 50 d., SMA20 kyla, kaina prie pat jo, ir siandien jau zalia.
    setup_sma_pb = 0.0
    try:
        sma20_prev = float(daily_hist["Close"].tail(25).head(20).mean())
        atstumas = abs(price - sma20_v) / sma20_v * 100
        if (price > sma50_v and sma20_v > sma20_prev and atstumas <= 1.5
                and day_chg is not None and day_chg > 0):
            setup_sma_pb = 1.0
    except Exception:
        pass

    # SET-UP: svyravimo suspaudimas. Pats savaime prognozuoja judesio DYDI, ne
    # krypti — todel jis naudingas tik kartu su sektoriaus kryptimi (sujungiama
    # veliau, main() funkcijoje). Cia tik uzfiksuojam, ar suspaudimas yra.
    squeeze = 0.0
    try:
        dh = daily_hist.tail(20)
        rng20 = (dh["High"] - dh["Low"]) / dh["Close"] * 100
        if len(rng20) >= 20:
            # NR7: paskutines 5 dienos ramesnes nei 80% pastaruju 20 dienu
            if float(rng20.tail(5).mean()) <= float(rng20.quantile(0.2)):
                squeeze = 1.0
    except Exception:
        pass

    # SET-UP: VWAP atgavimas. Diena praleido zemiau VWAP, dabar virs jo.
    setup_vwap_rec = 0.0
    try:
        if vwap and price > vwap:
            tp_run = (bars["High"] + bars["Low"] + bars["Close"]) / 3
            v_run = (tp_run * bars["Volume"]).cumsum() / bars["Volume"].cumsum().replace(0, np.nan)
            zemiau = float((bars["Close"] < v_run).mean())
            if zemiau > 0.5:
                setup_vwap_rec = 1.0
    except Exception:
        pass

    res_intra = high if high > price * 1.001 else None
    sup_intra = low if low < price * 0.999 else None
    hi20 = float(daily_hist["High"].tail(20).max())
    cands = sorted(x for x in (res_intra, res, hi20) if x and x > price * 1.001)

    return dict(
        price=price, dayHigh=high, dayLow=low, vwap=vwap, rsi=rsi, atrPct=a,
        support=sup, resistance=res, sup_intra=sup_intra, res_intra=res_intra,
        res_list=cands, rvol=rvol_at(sessions, day_idx, k),
        sma20=float(daily_hist["Close"].tail(20).mean()),
        sma50=float(daily_hist["Close"].tail(50).mean()),
        earnings=False, day_chg=day_chg,
        down_days=ctx["down_days"], dd5=ctx["dd5"], chg3d=ctx["chg3d"],
        avgVolume=float(daily_hist["Volume"].tail(20).mean()),
        cur="EUR", cur_sym="\u20ac", vol5m=v5,
        exp_move=dr.expected_move(v5, dr.HOLD_HOURS, bph=bph),
        m1h=mom["m1h"], m3h=mom["m3h"], pos1h=mom["pos1h"],
        span_h=mom["span_h"], mom_partial=mom["partial"],
        ibs=ibs, ibs2d=ibs2d, gap_ret=gap_ret, or_break=or_break, pullback_atr=pullback_atr,
        sma_align=sma_align, setup_sma_pb=setup_sma_pb, setup_vwap_rec=setup_vwap_rec,
        squeeze=squeeze,
        macd_h=macd_h, zscore=zscore, vol_exp=vol_exp,
        hl_struct=hl_struct, vol_price=vol_price, pd_break=pd_break, tod=tod)


def outcome(sessions, day_idx, k, entry, stop, target_price, hold_hours):
    """Ka kaina padare po ijejimo: tikslas, stop ar nei viena."""
    bars = [sessions[day_idx][1].iloc[k + 1:]]
    # Kiek sesiju apima laikymo horizontas (8.5 val. = viena prekybos diena)
    extra = max(0, int(round(hold_hours / 8.5)) - 1) if hold_hours > 8.5 else 0
    for j in range(1, extra + 1):
        if day_idx + j < len(sessions):
            bars.append(sessions[day_idx + j][1])
    future = pd.concat(bars) if bars else None
    if future is None or future.empty:
        return None, 0.0

    for _, b in future.iterrows():
        hit_t = float(b["High"]) >= target_price
        hit_s = float(b["Low"]) <= stop
        if hit_t and hit_s:
            return "stop", (stop - entry) / entry * 100
        if hit_t:
            return "tikslas", (target_price - entry) / entry * 100
        if hit_s:
            return "stop", (stop - entry) / entry * 100

    last = float(future["Close"].iloc[-1])
    return "be rezultato", (last - entry) / entry * 100


def day_block_bootstrap(vals, n=3000, seed=42):
    """Pasikliautinasis intervalas perrenkant DIENAS, ne eilutes."""
    if len(vals) < 25:
        return None
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_pos = float((boot > 0).mean())
    return dict(mean=float(vals.mean()), lo=float(lo), hi=float(hi),
                p_two=float(2 * min(p_pos, 1 - p_pos)), n_days=len(vals))


def week_block_bootstrap(vals, per_day_index=None, n=3000, seed=42):
    """Perrenka SAVAITES, ne atskiras dienas.

    Tos pacios savaites dienos yra susijusios (ta pati rinkos busena), todel
    dienu perrinkimas duoda per siaurus intervalus.
    """
    if len(vals) < 25:
        return None
    rng = np.random.default_rng(seed)
    if per_day_index is not None and len(per_day_index) == len(vals):
        weeks = pd.Series([pd.Timestamp(d).to_period("W") for d in per_day_index])
        grupes = [vals[(weeks == w).to_numpy()] for w in weeks.unique()]
    else:
        grupes = [vals[i:i + 5] for i in range(0, len(vals), 5)]
    grupes = [g for g in grupes if len(g)]
    if len(grupes) < 12:
        return None
    boot = []
    for _ in range(n):
        pick = rng.integers(0, len(grupes), len(grupes))
        boot.append(np.concatenate([grupes[i] for i in pick]).mean())
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_pos = float((boot > 0).mean())
    return dict(mean=float(vals.mean()), lo=float(lo), hi=float(hi),
                p_two=float(2 * min(p_pos, 1 - p_pos)), n_days=len(vals))


def within_day_edge(df, col, higher_better=True, top_pct=0.9, min_dienu=30):
    """Ar signalas isrenka geresne akcija TARP TOS PACIOS DIENOS akciju.

    Tai vienintelis matas, atmetantis "geros dienos" efekta. Kiekvienai dienai
    lyginam signalo isrinktu akciju rezultata su tos dienos vidurkiu, tada
    perrenkam dienas.
    """
    sub = df.dropna(subset=[col, "pnl"])
    if len(sub) < 1500:
        return None
    q = sub[col].quantile(top_pct if higher_better else 1 - top_pct)
    per_day, n_sel = [], 0
    for _, g in sub.groupby("_day"):
        sel = g[g[col] >= q]["pnl"] if higher_better else g[g[col] <= q]["pnl"]
        # Jei atrenkama VISA diena, signalas nieko neskiria — praleidziam
        if 1 <= len(sel) < len(g):
            per_day.append(sel.mean() - g["pnl"].mean())
            n_sel += len(sel)
    vals = np.array(per_day)
    # Nulines sklaidos apsauga: anksciau signalas, kuris nesuveike ne karto,
    # duodavo p=0 ir "ISLAIKO". Dabar toks atmetamas.
    if len(vals) < min_dienu or n_sel < 50 or float(np.std(vals)) < 1e-9:
        return None
    r = week_block_bootstrap(vals, per_day_index=[d for d, _ in sub.groupby("_day")
                                                  if True][:len(vals)])
    if r:
        r["n"] = n_sel
        r["dienos"] = len(vals)
    return r


def benjamini_hochberg(pvals, alpha=0.05):
    """Kuriuos rezultatus laikyti reiksmingais, kai tikrinam daug hipoteziu.

    Be sios pataisos, tikrinant 25 signalus, 1-2 "reiksmingi" atsiranda
    vien is atsitiktinumo.
    """
    order = np.argsort(pvals)
    m = len(pvals)
    passed = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= alpha * rank / m:
            passed[order[:rank]] = True
    return passed


# Sektoriu ETF (iShares STOXX Europe 600, Xetra, EUR). Kiekvienas apima ~30-90
# bendroviu, todel rodo TIKRA sektoriaus judejima, o ne 1-2 saraso akciju vidurki.
# Dalis tikeriu patvirtinti (EXV3/4/5/6, EXH4), dalis speliami — kodas pats
# patikrina, kurie grazina duomenis, ir netinkamus praleidzia.
SEKTORIU_ETF = {
    "AI infrastruktūra": "EXV3.DE",     # technologijos
    "puslaidininkiai":   "EXV3.DE",
    "programinė įranga": "EXV3.DE",
    "IT paslaugos":      "EXV3.DE",
    "telekomų įranga":   "EXV3.DE",
    "automobiliai":      "EXV5.DE",     # automobiliai ir dalys
    "auto komponentai":  "EXV5.DE",
    "pramonė":           "EXH4.DE",     # pramones prekes ir paslaugos
    "gynyba":            "EXH4.DE",
    "metalai":           "EXV6.DE",     # baziniai istekliai
    "farmacija":         "EXV4.DE",     # sveikatos apsauga
    "aviakompanijos":    "EXV9.DE",     # keliones ir laisvalaikis
    "prabanga":          "EXH6.DE",     # asmenines ir namu prekes (tikrinama)
    "vartojimo prekės":  "EXH6.DE",
}
PLATUS_INDEKSAS = "EXSA.DE"             # visas STOXX Europe 600

# JAV sektoriu ETF. Jie prekiauja 15:30-22:00 Berlyno laiku ir yra likvidus, todel
# ju 5 min. barai patikimi — skirtingai nei europietisku sektoriu ETF. Mechanizmas
# priezastinis: JAV puslaidininkiai juda pirmi, Europos atitikmenys Tradegate
# reaguoja veliau ir nepilnai. Tai vienintelis TIKRAI reaktyvus kanalas.
JAV_ETF = {
    "AI infrastruktūra": "SMH",      # puslaidininkiu iranga ir gamintojai
    "puslaidininkiai":   "SMH",
    "programinė įranga": "XLK",      # technologijos
    "IT paslaugos":      "XLK",
    "telekomų įranga":   "XLK",
    "automobiliai":      "XLY",      # vartojimo cikliskos prekes
    "auto komponentai":  "XLY",
    "vartojimo prekės":  "XLY",
    "prabanga":          "XLY",
    "gynyba":            "ITA",      # aviacija ir gynyba
    "pramonė":           "XLI",      # pramone
    "metalai":           "XLB",      # medziagos
    "farmacija":         "XLV",      # sveikata
    "aviakompanijos":    "JETS",     # aviakompanijos
}
JAV_ATIDARYMAS = 15 * 60 + 30       # 15:30 Berlyno laiku


SIGNALS = [
    ("score", True, "Dabartinis balas"),
    ("ibs", False, "IBS zemas (dabartinis)"),
    ("ibs2d", False, "IBS 2 dienu (variantas 1)"),
    ("ibs_pct", False, "IBS procentilis (variantas 2)"),
    ("gap_ret", False, "Nakties tarpas zemyn"),
    ("or_break", True, "Atid. diapazono pramusimas"),
    ("vwap_d", False, "Kaina zemiau VWAP"),
    ("pullback_atr", True, "Gilesnis atsitraukimas"),
    ("sma_align", True, "SMA issidestymas"),
    ("macd_h", True, "MACD histograma +"),
    ("zscore", False, "Z-balas zemas"),
    ("zscore", True, "Z-balas aukstas"),
    ("vol_exp", True, "Svyravimo pletra"),
    ("hl_struct", True, "Aukstesniu dugnu struktura"),
    ("vol_price", True, "Apyvarta + kilimas"),
    ("pd_break", True, "Vakar max pramusimas"),
    # Sektoriaus momentumas — pirmas kartas matuojamas
    ("sekt_mom5", True, "Sektorius kyla (5 d.)"),
    ("sekt_mom10", True, "Sektorius kyla (10 d.)"),
    ("likutis3", False, "Akcija atsilieka nuo sektoriaus (3 d.)"),
    ("likutis5", False, "Akcija atsilieka nuo sektoriaus (5 d.)"),
    ("r5", True, "Akcijos 5 d. momentumas"),
    ("setup_sektorius", True, "SET-UP: kylantis sekt. + atsilikimas"),
    ("setup_stiprus", True, "SET-UP: stiprus sekt. + gilesnis atsilikimas"),
    ("sekt_vs_rinka", True, "Sektorius stipresnis uz visa rinka"),
    ("sekt_z", True, "Sektoriaus z-balas pries rinka"),
    # JAV kanalas — reaktyvumas dienos viduje (15:30-22:00)
    ("us_move", True, "JAV sektorius kyla nuo 15:30"),
    ("us_move", False, "JAV sektorius krenta nuo 15:30"),
    ("us_lag", True, "Akcija atsilieka nuo JAV judesio"),
    # Klasikiniai set-up'ai kaip atskiri ivykiai
    ("setup_sma_pb", True, "SET-UP: atsitraukimas prie kylancio SMA20"),
    ("setup_vwap_rec", True, "SET-UP: VWAP atgavimas"),
    ("setup_squeeze_sekt", True, "SET-UP: suspaudimas + kylantis sektorius"),
]


def report_signals(df, label, correct=True):
    """Visu signalu patikra su daugybinio tikrinimo pataisa."""
    rows = []
    for col, hb, name in SIGNALS:
        if col not in df:
            continue
        r = within_day_edge(df, col, hb)
        if r:
            rows.append((name, col, hb, r))
    for key, lab, _w in dr.CRITERIA:
        c = f"c_{key}"
        if c in df:
            r = within_day_edge(df, c, True)
            if r:
                rows.append((f"  kriterijus: {lab[:24]}", c, True, r))
    if not rows:
        print(f"\n{label}: nepakako duomenu.")
        return []

    pvals = np.array([r[3]["p_two"] for r in rows])
    passed = benjamini_hochberg(pvals) if correct else pvals < 0.05

    print(f"\n{label}  (tikrinta {len(rows)} signalu, "
          f"{'su daugybinio tikrinimo pataisa' if correct else 'be pataisos'})")
    print(f"{'SIGNALAS':<30} {'PRANASUMAS':>11} {'95% INTERVALAS':>22} {'p':>7} "
          f"{'ATVEJU':>8} {'DIENU':>7} {'ISLAIKO':>8}")
    print("-" * 100)
    order = np.argsort([r[3]["mean"] for r in rows])[::-1]
    out = []
    for i in order:
        name, col, hb, r = rows[i]
        mark = "TAIP" if passed[i] else ""
        ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
        print(f"{name:<30} {r['mean']:>+10.3f}% {ci:>22} {r['p_two']:>7.3f} "
              f"{r.get('n', 0):>8} {r.get('dienos', 0):>7} {mark:>8}")
        if passed[i]:
            out.append((name, col, hb, r))
    return out


def outcome_vwap_exit(sessions, day_idx, k, entry, stop, min_target, hold_hours):
    """Isejimas, kai kaina uzdaro barа zemiau slenkancio VWAP po to, kai pasieke tiksla.

    Skirtumas nuo slenkancio stopo: slenkantis stop iseina mechaniskai, praradus
    fiksuota % nuo virsunes. Sis iseina, kai kaina praranda dienos vidutine kaina —
    t. y. kai pirkejai nustoja kontroliuoti diena.
    """
    bars = [sessions[day_idx][1].iloc[k + 1:]]
    extra = max(0, int(round(hold_hours / 8.5)) - 1) if hold_hours > 8.5 else 0
    for j in range(1, extra + 1):
        if day_idx + j < len(sessions):
            bars.append(sessions[day_idx + j][1])
    future = pd.concat(bars) if bars else None
    if future is None or future.empty:
        return None, 0.0

    trigger = entry * (1 + min_target / 100)
    armed = False
    tp_run = (future["High"] + future["Low"] + future["Close"]) / 3
    vol = future["Volume"].replace(0, np.nan)
    vwap_run = (tp_run * vol).cumsum() / vol.cumsum()

    for i, (_, b) in enumerate(future.iterrows()):
        lo, hi, cl = float(b["Low"]), float(b["High"]), float(b["Close"])
        if lo <= stop:
            return "stop", (stop - entry) / entry * 100
        if not armed and hi >= trigger:
            armed = True
            continue
        if armed:
            v = vwap_run.iloc[i]
            if not pd.isna(v) and cl < v:
                return "vwap prarastas", (cl - entry) / entry * 100

    last = float(future["Close"].iloc[-1])
    return ("uzdaryta pabaigoje" if armed else "be rezultato"), (last - entry) / entry * 100


def outcome_trailing(sessions, day_idx, k, entry, stop, min_target, trail_pct, hold_hours):
    """Isejimas be virsutines ribos: pasiekus min_target, ijungiamas slenkantis stop.

    Skirtumas nuo fiksuoto tikslo: kai judesys stiprus, pozicija laikoma toliau,
    o pelnas fiksuojamas tik kai kaina atsitraukia trail_pct nuo pasiektos virsunes.
    Kai bare paliesti abu lygiai, laikoma nepalankiu variantu (konservatyvu).
    """
    bars = [sessions[day_idx][1].iloc[k + 1:]]
    extra = max(0, int(round(hold_hours / 8.5)) - 1) if hold_hours > 8.5 else 0
    for j in range(1, extra + 1):
        if day_idx + j < len(sessions):
            bars.append(sessions[day_idx + j][1])
    future = pd.concat(bars) if bars else None
    if future is None or future.empty:
        return None, 0.0

    trigger = entry * (1 + min_target / 100)
    armed = False
    peak = entry
    cur_stop = stop

    for _, b in future.iterrows():
        hi, lo = float(b["High"]), float(b["Low"])
        if lo <= cur_stop:
            pnl = (cur_stop - entry) / entry * 100
            return ("stop" if not armed else "slenkantis stop"), pnl
        if hi > peak:
            peak = hi
        if not armed and hi >= trigger:
            armed = True
        if armed:
            new_stop = peak * (1 - trail_pct / 100)
            cur_stop = max(cur_stop, new_stop)

    last = float(future["Close"].iloc[-1])
    return ("uzdaryta pabaigoje" if armed else "be rezultato"), (last - entry) / entry * 100


def main():
    ap = argparse.ArgumentParser(description="Intraday backtestas su griezta patikra")
    ap.add_argument("--target", type=float, default=dr.TARGET_PCT)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--interval", default="60m", choices=["5m", "60m"])
    ap.add_argument("--leisti-5m", action="store_true",
                    help="Leisti 5m testa (su 3+ dienu laikymu jis neinformatyvus)")
    ap.add_argument("--costs", type=float, default=0.07,
                    help="Mokesciai + spread'as procentais sandoriui")
    args = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Paleisk: pip install yfinance pandas numpy tzdata")

    if args.interval == "5m" and dr.HOLD_HOURS > 16 and not args.leisti_5m:
        sys.exit("5m testas isjungtas: 60 dienu imtis su "
                 f"{dr.HOLD_HOURS:.0f} val. laikymu duoda ~10 nepriklausomu langu — "
                 "rezultatai butu triuksmas. Naudok --interval 60m arba --leisti-5m.")

    symbols = [s for _, s, _ in dr.WATCHLIST]
    if args.interval == "60m":
        bph, checkpoints, min_bars = 1, CHECKPOINTS_60M, 5
        days = 720
    else:
        bph, checkpoints, min_bars = 12, CHECKPOINTS_5M, 40
        days = min(args.days, 60)

    print(f"Siunciama {len(symbols)} akciju {args.interval} istorija ({days} d.)…")
    intraday = yf.download(symbols, period=f"{days}d", interval=args.interval,
                           group_by="ticker", progress=False, auto_adjust=False, threads=True)
    daily = yf.download(symbols, period="max" if args.interval == "60m" else "12mo",
                        interval="1d", group_by="ticker", progress=False,
                        auto_adjust=False, threads=True)

    # --- Sektoriu ETF: tikras sektoriaus judejimas ---
    etf_syms = sorted(set(SEKTORIU_ETF.values()) | {PLATUS_INDEKSAS})
    print(f"Siunciama {len(etf_syms)} sektoriu ETF…")
    etf_ret = {}
    try:
        etf_raw = yf.download(etf_syms, period="2y", interval="1d", group_by="ticker",
                              progress=False, auto_adjust=False, threads=True)
        veikia, neveikia = [], []
        for e in etf_syms:
            try:
                ed = dr.flatten(etf_raw, e).dropna(subset=["Close"])
                if len(ed) < 100:
                    neveikia.append(e)
                    continue
                c = ed["Close"]
                d = pd.DataFrame(index=pd.Index([i.date() for i in ed.index], name="_d"))
                for n in (3, 5, 10):
                    d[f"e{n}"] = (c / c.shift(n) - 1).values * 100
                etf_ret[e] = d
                veikia.append(e)
            except Exception:
                neveikia.append(e)
        print(f"  veikia: {', '.join(veikia)}")
        if neveikia:
            print(f"  NEVEIKIA (sektoriai kris atgal i saraso mediana): {', '.join(neveikia)}")
    except Exception as e:
        print(f"  ETF atsisiuntimas nepavyko: {str(e)[:60]}")

    # --- JAV sektoriu ETF intraday ---
    jav_intra = {}
    try:
        jav_syms = sorted(set(JAV_ETF.values()))
        print(f"Siunciama {len(jav_syms)} JAV sektoriu ETF ({args.interval})…")
        jav_raw = yf.download(jav_syms, period=f"{days}d", interval=args.interval,
                              group_by="ticker", progress=False, auto_adjust=False,
                              threads=True)
        ok_j, bad_j = [], []
        for e in jav_syms:
            try:
                ed = dr.flatten(jav_raw, e).dropna(subset=["Close"])
                if len(ed) < 200:
                    bad_j.append(e)
                    continue
                if ed.index.tz is None:
                    ed.index = ed.index.tz_localize("UTC")
                jav_intra[e] = ed["Close"].tz_convert("Europe/Berlin").sort_index()
                ok_j.append(e)
            except Exception:
                bad_j.append(e)
        print(f"  veikia: {', '.join(ok_j)}" + (f" | neveikia: {', '.join(bad_j)}" if bad_j else ""))
    except Exception as e:
        print(f"  JAV ETF atsisiuntimas nepavyko: {str(e)[:60]}")

    rows = []
    for tag, sym, _ in dr.WATCHLIST:
        try:
            intra = dr.flatten(intraday, sym).dropna(subset=["Close"])
            dhist = dr.flatten(daily, sym).dropna(subset=["Close", "High", "Low"])
            if intra.empty or len(dhist) < 60:
                continue
            sessions = session_frames(intra, min_bars)
            rsi_series = dr.rsi(intra["Close"])
            for di, (day, _) in enumerate(sessions):
                if di < 1 or di + 1 >= len(sessions):
                    continue
                hist = dhist[dhist.index.date < day]
                if len(hist) < 55:
                    continue
                for k in checkpoints:
                    if k + max(2, bph // 2) >= len(sessions[di][1]):
                        continue
                    d = build_snapshot(sessions, di, k, hist, rsi_series, args.target,
                                       bph, full_series=intra)
                    if not d:
                        continue
                    s = dr.score_stock(d, args.target, "neutral")
                    res, pnl = outcome(sessions, di, k, d["price"], s["stop"],
                                       s["tp"], dr.HOLD_HOURS)
                    if res is None:
                        continue
                    # Tas pats ijejimas, kitokia isejimo taisykle
                    tr_res, tr_pnl = outcome_trailing(sessions, di, k, d["price"], s["stop"],
                                                      2.0, 1.0, dr.HOLD_HOURS)
                    tr_res2, tr_pnl2 = outcome_trailing(sessions, di, k, d["price"], s["stop"],
                                                        2.0, 1.5, dr.HOLD_HOURS)
                    vw_res, vw_pnl = outcome_vwap_exit(sessions, di, k, d["price"], s["stop"],
                                                       2.0, dr.HOLD_HOURS)
                    # --- JAV atsilikimas: kiek akcija dar neatkartojo JAV judesio ---
                    us_move = us_lag = None
                    try:
                        etf_sym = JAV_ETF.get(dr.SECTORS.get(sym, ""))
                        ser = jav_intra.get(etf_sym) if etf_sym else None
                        if ser is not None:
                            ts = sessions[di][1].index[k]
                            atid = ts.replace(hour=JAV_ATIDARYMAS // 60,
                                              minute=JAV_ATIDARYMAS % 60,
                                              second=0, microsecond=0)
                            if ts > atid:      # tik po JAV atidarymo
                                e_now, e_open = ser.asof(ts), ser.asof(atid)
                                bars_day = sessions[di][1]
                                s_open = bars_day["Close"].asof(atid)
                                if (e_now and e_open and s_open
                                        and not pd.isna(e_now) and not pd.isna(e_open)):
                                    us_move = (e_now / e_open - 1) * 100
                                    stock_move = (d["price"] / s_open - 1) * 100
                                    us_lag = us_move - stock_move
                    except Exception:
                        pass

                    rec = dict(tag=tag, _day=day, _k=k, pnl=pnl, result=res,
                               us_move=us_move, us_lag=us_lag,
                               pnl_trail=tr_pnl, res_trail=tr_res,
                               pnl_trail15=tr_pnl2, pnl_vwap=vw_pnl, res_vwap=vw_res,
                               score=s["score"], setup=s.get("setup"))
                    for key, _l, _w in dr.CRITERIA:
                        rec[f"c_{key}"] = s["parts"].get(key)
                    for f in ["ibs", "gap_ret", "or_break", "pullback_atr", "sma_align",
                              "macd_h", "zscore", "vol_exp", "hl_struct", "vol_price",
                              "pd_break", "day_chg", "ibs2d", "atrPct",
                              "setup_sma_pb", "setup_vwap_rec", "squeeze"]:
                        rec[f] = d.get(f)
                    rec["vwap_d"] = ((d["price"] - d["vwap"]) / d["vwap"] * 100
                                     if d.get("vwap") else None)
                    rows.append(rec)
        except Exception as e:
            print(f"  {tag}: praleista ({str(e)[:50]})")

    if not rows:
        sys.exit("Nepavyko surinkti duomenu.")

    df = pd.DataFrame(rows)
    df["_day"] = pd.to_datetime(df["_day"])
    # IBS VARIANTAS 2: kur siandienos IBS yra tos akcijos ISTORINIU IBS fone.
    # YDX su 8.6% ATR ir SAP su 2.5% nera palyginami absoliuciu IBS 0.15.
    try:
        df["ibs_pct"] = df.groupby("tag")["ibs"].rank(pct=True)
    except Exception:
        df["ibs_pct"] = None

    # --- SEKTORIAUS MOMENTUMAS (is ETF, ne is saraso) ---
    # Moskowitz & Grinblatt (1999): sektoriaus momentumas paaiskina 60-73% viso
    # akciju momentumo, o akcijos graza PRIES savo sektoriu prognozuoja geriau nei
    # jos pacios graza. Sektorius imamas is ETF (30-90 bendroviu), nes sarase
    # 8 sektoriai turi po viena akcija — ju "mediana" butu ta pati akcija.
    try:
        df["sekt"] = df["tag"].map(
            {t: dr.SECTORS.get(s, "kita") for t, s, _ in dr.WATCHLIST})
        df["_dt"] = df["_day"].dt.date
        df["_etf"] = df["sekt"].map(SEKTORIU_ETF)

        daily_ret = (df.groupby(["tag", "_day"])["day_chg"].mean()
                       .reset_index().sort_values(["tag", "_day"]))
        for n in (3, 5, 10):
            daily_ret[f"r{n}"] = (daily_ret.groupby("tag")["day_chg"]
                                  .transform(lambda x: x.rolling(n, min_periods=2).sum()))
        df = df.merge(daily_ret[["tag", "_day", "r3", "r5", "r10"]],
                      on=["tag", "_day"], how="left")

        is_etf = 0.0
        for n in (3, 5, 10):
            saraso_med = df.groupby(["_day", "sekt"])[f"r{n}"].transform("median")
            etf_val = pd.Series(np.nan, index=df.index)
            for e, tab in etf_ret.items():
                mask = df["_etf"] == e
                if mask.any():
                    etf_val.loc[mask] = df.loc[mask, "_dt"].map(tab[f"e{n}"])
            df[f"sekt_mom{n}"] = etf_val.fillna(saraso_med)
            df[f"likutis{n}"] = df[f"r{n}"] - df[f"sekt_mom{n}"]
            if n == 10:
                is_etf = float(etf_val.notna().mean() * 100)

        if PLATUS_INDEKSAS in etf_ret:
            rinka = df["_dt"].map(etf_ret[PLATUS_INDEKSAS]["e10"])
            df["sekt_vs_rinka"] = df["sekt_mom10"] - rinka

        # SET-UP: suspaudimas + kylantis sektorius. Suspaudimas duoda judesio
        # dydi, sektorius — krypti.
        df["setup_squeeze_sekt"] = np.where(
            (df["squeeze"] > 0) & (df["sekt_mom10"] > 0.5), 1.0, 0.0)

        # SET-UP: kylantis sektorius + akcija jame trumpam atsilikusi.
        # Slenksciai uzrasyti PRIES matant rezultatus ir nebus derinami.
        # Set-up turi buti RETAS ivykis. Anksciau suveikdavo 21% laiko — tai ne
        # ivykis, o busena, ir tokia imtis beveik atsitiktine. Dabar matuojam
        # sektoriu SANTYKINAI pries rinka ir normalizuojam jo svyravimu (z-balas).
        if "sekt_vs_rinka" in df:
            sv = df["sekt_vs_rinka"]
            z = (sv - sv.rolling(200, min_periods=50).mean()) / \
                sv.rolling(200, min_periods=50).std()
            df["sekt_z"] = z
            lik_z = ((df["likutis3"] - df["likutis3"].rolling(200, min_periods=50).mean())
                     / df["likutis3"].rolling(200, min_periods=50).std())
            # Retumas apibreziamas procentiliais, ne fiksuotomis z ribomis: taip
            # ivykis lieka retas, bet imtis niekada nebuna nuline. Ribos nustatytos
            # pries matant rezultatus ir nederinamos.
            z_riba = z.quantile(0.90)
            z_riba_st = z.quantile(0.95)
            lik_riba = lik_z.quantile(0.25)
            lik_riba_st = lik_z.quantile(0.15)
            df["setup_sektorius"] = np.where((z >= z_riba) & (lik_z <= lik_riba), 1.0, 0.0)
            df["setup_stiprus"] = np.where((z >= z_riba_st) & (lik_z <= lik_riba_st), 1.0, 0.0)
        else:
            df["setup_sektorius"] = 0.0
            df["setup_stiprus"] = 0.0

        d1 = df["setup_sektorius"].mean() * 100
        d2 = df["setup_stiprus"].mean() * 100
        print(f"Sektoriaus judejimas is ETF: {is_etf:.0f}% eiluciu; "
              f"set-up'ai suveikia {d1:.1f}% / {d2:.1f}% atveju")
        if d1 < 0.3 or d2 < 0.3:
            print("  DEMESIO: set-up per retas — imtis bus per maza isvadai")
        elif d1 > 10:
            print("  DEMESIO: set-up suveikia per daznai — tai busena, ne ivykis")
    except Exception as e:
        import traceback
        print(f"!!! SEKTORIAUS MOMENTUMAS NEVEIKIA: {type(e).__name__}: {e}")
        traceback.print_exc()

    base = df["pnl"].mean()
    n_days = df["_day"].nunique()

    print(f"\nIjejimo tasku: {len(df)}  |  nepriklausomu dienu: {n_days}  "
          f"|  tikslas {args.target}%  |  laikymas {dr.HOLD_HOURS} val.")
    print(f"BAZINE LINIJA (atsitiktinis ijejimas): {base:+.3f}% sandoriui")
    print(f"Mokesciai ir spread'as: {args.costs:.2f}% sandoriui — "
          f"signalas verta demesio tik virs sios ribos.")
    print(f"\nDEMESIO: {len(SIGNALS) + len(dr.CRITERIA)} signalu ant TOS PACIOS imties. "
          f"Be pataisos 1-2 'reiksmingi' atsiranda vien is atsitiktinumo.")

    # ---------- ISEJIMO TAISYKLIU PALYGINIMAS ----------
    # Tas pats ijejimas, trys skirtingos isejimo taisykles. Klausimas: ar atsisakius
    # virsutines ribos rezultatas pagereja, ir ar SIGNALAS tampa informatyvesnis.
    try:
        if "pnl_trail" in df:
            print("\n" + "=" * 84)
            print("ISEJIMO TAISYKLES — tas pats ijejimas, skirtingi isejimai")
            print("=" * 84)
            rules = [("pnl", f"Fiksuotas tikslas {args.target}%"),
                     ("pnl_trail", "Min 2%, tada slenkantis stop 1.0%"),
                     ("pnl_trail15", "Min 2%, tada slenkantis stop 1.5%"),
                     ("pnl_vwap", "Min 2%, tada isejimas praradus VWAP")]
            print(f"{'TAISYKLE':<34} {'VID. REZ.':>11} {'>0 dalis':>10} "
                  f"{'VID. PELNAS':>12} {'VID. NUOSTOLIS':>15}")
            print("-" * 84)
            for col, lab in rules:
                v = df[col].dropna()
                if len(v) < 500:
                    continue
                wins, losses = v[v > 0], v[v <= 0]
                print(f"{lab:<34} {v.mean():>+10.3f}% {len(wins)/len(v)*100:>9.1f}% "
                      f"{wins.mean():>+11.2f}% {losses.mean():>+14.2f}%")

            print(f"\n{'TAISYKLE':<34} {'SIGNALO PRANASUMAS':>20} {'95% INTERVALAS':>24}")
            print("-" * 84)
            for col, lab in rules:
                tmp = df.copy()
                tmp["pnl"] = tmp[col]
                r = within_day_edge(tmp, "ibs", higher_better=False)
                if r:
                    ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
                    print(f"{lab:<34} {r['mean']:>+19.3f}% {ci:>24}")
            print("\n(Signalo pranasumas = IBS, vienintelis patvirtintas signalas. "
                  "\nJei jis didesnis su slenkanciu isejimu, verta keisti taisykle.)")
    except Exception as e:
        print(f"(isejimo palyginimas praleistas: {str(e)[:60]})")

    # ---------- RYTINIS RALIS: ar rytinis stiprumas testiasi? ----------
    # Klausimas siauresnis nei ankstesni: imam TIK anksciausia dienos taska
    # (~10:30) ir ziurim, ka kaina padare iki galo. Tai tiesiogiai atsako,
    # ar rytinis kilimas turi tesinio potencialo.
    try:
        first_k = df["_k"].min()
        am = df[df["_k"] == first_k].copy()
        if len(am) > 800:
            am_base = am["pnl"].mean()
            mid_am = am["_day"].median()
            h1, h2 = am[am["_day"] <= mid_am], am[am["_day"] > mid_am]
            print("\n" + "=" * 84)
            print(f"RYTINIS RALIS — tik anksciausias dienos taskas ({len(am)} atveju). "
                  f"Ar rytinis kilimas testiasi?")
            print(f"Bazine linija sioje imtyje: {am_base:+.3f}%")
            print("=" * 84)
            print(f"{'RYTINIS POKYTIS':<18} {'ATVEJU':>7} {'VISA IMTIS':>12} "
                  f"{'1-OJI PUSE':>12} {'2-OJI PUSE':>12} {'STABILUS':>9}")
            print("-" * 84)
            for lo, hi, lab in [(-99, -1.0, "krenta > 1%"), (-1.0, -0.3, "krenta 0.3-1%"),
                                (-0.3, 0.3, "stovi vietoje"), (0.3, 1.0, "kyla 0.3-1%"),
                                (1.0, 2.0, "kyla 1-2%"), (2.0, 99, "ralis > 2%")]:
                g = am[(am["day_chg"] >= lo) & (am["day_chg"] < hi)]
                g1 = h1[(h1["day_chg"] >= lo) & (h1["day_chg"] < hi)]
                g2 = h2[(h2["day_chg"] >= lo) & (h2["day_chg"] < hi)]
                if len(g) < 60 or len(g1) < 25 or len(g2) < 25:
                    continue
                d_all = g["pnl"].mean() - am_base
                d1 = g1["pnl"].mean() - h1["pnl"].mean()
                d2 = g2["pnl"].mean() - h2["pnl"].mean()
                stab = "TAIP" if (d1 > 0.02 and d2 > 0.02) or (d1 < -0.02 and d2 < -0.02) else "ne"
                print(f"{lab:<18} {len(g):>7} {d_all:>+11.3f}% {d1:>+11.3f}% "
                      f"{d2:>+11.3f}% {stab:>9}")

            # Ar svarbu, kad kyla visa rinka, ar tik viena akcija?
            am["_breadth"] = am.groupby("_day")["day_chg"].transform("median")
            print(f"\n{'RALIS + RINKOS PLOTIS':<34} {'ATVEJU':>7} {'PRIES BAZE':>12}")
            print("-" * 58)
            for cond, lab in [
                ((am["day_chg"] > 1.0) & (am["_breadth"] > 0.5), "akcija kyla + kyla visas sarasas"),
                ((am["day_chg"] > 1.0) & (am["_breadth"] <= 0.5), "akcija kyla viena"),
                ((am["day_chg"] > 1.0) & (am["ibs"] > 0.7), "kyla ir laikosi virsuje"),
                ((am["day_chg"] > 1.0) & (am["ibs"] <= 0.4), "kyla, bet atsitrauke")]:
                g = am[cond]
                if len(g) >= 60:
                    print(f"{lab:<34} {len(g):>7} {g['pnl'].mean()-am_base:>+11.3f}%")
    except Exception as e:
        print(f"(rytinio ralio analize praleista: {str(e)[:60]})")

    # ---------- 1 etapas: paieska pirmoje laiko puseje ----------
    mid = df["_day"].median()
    train = df[df["_day"] <= mid]
    test = df[df["_day"] > mid]
    print("\n" + "=" * 84)
    print(f"1 ETAPAS — PAIESKA (1-oji puse: {train['_day'].nunique()} dienu). "
          f"Cia ieskom kandidatu.")
    print("=" * 84)
    found = report_signals(train, "Rezultatai pirmoje puseje")

    # ---------- 2 etapas: patvirtinimas antroje, NEMATYTOJE puseje ----------
    print("\n" + "=" * 84)
    print(f"2 ETAPAS — PATVIRTINIMAS (2-oji puse: {test['_day'].nunique()} dienu). "
          f"Tikrinami TIK 1 etape islaike kandidatai.")
    print("Cia pataisos nereikia — hipotezes buvo pasirinktos pries pamatant siuos duomenis.")
    print("=" * 84)
    if not found:
        print("\n1 etape nei vienas signalas neislaike — patvirtinti nera ko.")
        print("Tai reiskia, kad ankstesni 'reiksmingi' rezultatai buvo daugybinio")
        print("tikrinimo pasekme, o ne tikras pranasumas.")
    else:
        base_t = test["pnl"].mean()
        print(f"\n{'SIGNALAS':<30} {'PRANASUMAS':>11} {'95% INTERVALAS':>22} {'VERDIKTAS':>16}")
        print("-" * 84)
        confirmed = []
        for name, col, hb, _ in found:
            r = within_day_edge(test, col, hb)
            if not r:
                print(f"{name:<30} {'per maza imtis':>11}")
                continue
            if r["lo"] > 0:
                v = "PATVIRTINTA"
                confirmed.append((name, col, hb, r))
            elif r["hi"] < 0:
                v = "PRIESINGA KRYPTIS"
            else:
                v = "nepatvirtinta"
            ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
            print(f"{name:<30} {r['mean']:>+10.3f}% {ci:>22} "
                  f"n={r.get('n', 0)} d={r.get('dienos', 0)} {v:>16}")

        # ---------- 3 etapas: ar pranasumas is visu akciju, ar is vienos ----------
        if confirmed:
            print("\n" + "=" * 84)
            print("3 ETAPAS — ATSPARUMAS: ar pranasumas islieka isbraukus bet kuria akcija?")
            print("=" * 84)
            for name, col, hb, _ in confirmed[:3]:
                mins, maxs = 99.0, -99.0
                worst = best = ""
                for tag in df["tag"].unique():
                    r = within_day_edge(df[df["tag"] != tag], col, hb)
                    if r:
                        if r["mean"] < mins:
                            mins, worst = r["mean"], tag
                        if r["mean"] > maxs:
                            maxs, best = r["mean"], tag
                full = within_day_edge(df, col, hb)
                print(f"\n{name}: visos akcijos {full['mean']:+.3f}%")
                print(f"  isbraukus po viena: nuo {mins:+.3f}% (be {worst}) "
                      f"iki {maxs:+.3f}% (be {best})")
                if mins > 0:
                    print("  -> Pranasumas nepriklauso nuo vienos akcijos.")
                else:
                    print("  -> DEMESIO: isbraukus viena akcija pranasumas dingsta. "
                          "Greiciausiai atsitiktinumas.")

            print("\n" + "=" * 84)
            print("GALUTINIS VERTINIMAS (atemus mokescius)")
            print("=" * 84)
            for name, col, hb, r in confirmed:
                net = r["mean"] - args.costs
                print(f"{name:<34} bruto {r['mean']:+.3f}%  neto {net:+.3f}%  "
                      f"{'verta' if net > 0 else 'po mokesciu nelieka'}")
        else:
            print("\nNe vienas kandidatas nepasitvirtino nematytoje imties dalyje.")
            print("Tai stipriausias imanomas signalas, kad pranasumo nera.")

    print("\nAPRIBOJIMAI: be mokesciu ir spread'o skaiciuose auksciau; kai bare paliesti "
          "\nabu lygiai — laikoma pralaimejimu; duomenys is vieno saltinio; "
          "\npraeities rezultatai negarantuoja ateities.")


if __name__ == "__main__":
    main()
