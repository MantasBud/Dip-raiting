#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PIRMO PRISILIETIMO BACKTESTAS  (v2 - pataisyta pagal gyva zurnala 2026-09-21)
=============================================================================

KODEL SIS TESTAS KITOKS NEI VISI ANKSTESNI
------------------------------------------
Visi ankstesni matavimai vertino DEMEANUOTA VIDUTINE GRAZA per N dienu.
Tavo sandoris tokio mato neturi. Tavo taisykle:

    perku visu portfeliu -> parduodu, kai kaina PALIECIA +1..3%
    -> jei nepaliecia, laikau, kol atsistato (~-500 EUR riba)

Tai PIRMO PRISILIETIMO uzdavinys. Vidutine graza ir prisilietimo tikimybe
beveik nesusije: akcija su nuline vidutine graza gali tureti 80% tikimybe
paliesti +1% pirmiau nei -2.8%.

KA PARODE GYVAS ZURNALAS (250 baigtu, 2026-09-11..18) - LEMIAMAS FAKTAS
-----------------------------------------------------------------------
    virsunes mediana:  +0.105%
    tikslas:           +1.0%
    P(virsune >= 1%):   5.2%
    vid. rezultatas:  -26.30 EUR sandoriui (-6575 EUR is viso)

Modulis atrenka akcijas, kurios per laikymo langa NET NEPRIARTEJA prie tavo
tikslo. Tai ne isejimo derinimo klausimas - tai atrankos kriteriju klausimas.
Todel pagrindinis sio testo matas yra ne graza, o:

    P(virsune >= tikslas per H sesiju)

Sis dydis daug maziau triuksmingas uz EUR ir tiesiogiai atsako, ar kandidatu
aibe apskritai tinka tavo taisyklei.

KODEL DVI ISEJIMO TAISYKLES
---------------------------
Zurnalas rodo auksto ATR (>3.5%) grupe kaip vienintele pelninga (+0.50%),
o mano pradine simuliacija prognozavo priesingai - kad geriausia 1.5-2.5%.

Abu gali buti teisingi, nes matuoja SKIRTINGAS taisykles:
  - laikant per nakti ir parduodant atidarymu svarbus tik nakties dreifas,
    ir ji STIPRINA didelis ATR
  - lenktyniaujant tikslo su skausmo riba didelis ATR PIRMIAU atveda i
    skausmo riba

Todel ATR kryptis NERA uzduota is anksto. Testas matuoja abi taisykles
greta ir leidzia duomenims atsakyti.

PERSPEJIMAS DEL ZURNALO: 250 sandoriu, bet tik 7 nepriklausomos dienos ir
visos bear_soft rezime. Bootstrap pagal dienas duoda auksto ATR grupei
intervala [-0.94, +1.50] - t.y. nereiksminga. Zurnalas kelia hipoteze,
neatsako i ja.

KA SIS TESTAS TIKRINA
---------------------
H1  GRIZIMO PUSEJIMAS kaip stoties savybe. Visi ~60 ankstesniu signalu buvo
    BUSENOS matai (kur kaina dabar). Nei vienas nebuvo STOTIES matas (kaip
    greitai si akcija istoriskai atsoka). Klausimas: ar pirmoje puseje
    ismatuotas pusejimas prognozuoja prisilietima antroje, nematytoje.

H2  ATR - ar egzistuoja optimalus ruozas, ir ar jis SKIRTINGAS dviem
    isejimo taisyklems. Kryptis neuzduota.

H3  KYLANCIOS akcijos tuo paciu matu per 10 metu. Sesi ankstesni atmetimai
    visi mataveee vidutine graza. 2026-09-17 testas, vienintelis maaves
    prisilietima, rode 48.1% vs 38.6% - kryptis palanki, bet imtis 59 dienos.

METODIKA: PRISILIETIMAS BE INTRADAY DUOMENU
--------------------------------------------
Dienos barai nerodo tvarkos dienos viduje, bet duoda GRIEZTAS RIBAS:
  High >= tikslas IR Low > skausmo riba -> tikslas pirmiau (TIKRAI)
  Low <= skausmo riba IR High < tikslas -> skausmas pirmiau (TIKRAI)
  abu ta pacia diena                    -> NEAISKU
Skaiciuojame P_min (neaisku=skausmas) ir P_max (neaisku=tikslas).
Sausame paleidime neaisku buvo 1.6% - ribos siauros, tad uztenka 10 metu
nemokamos istorijos vietoj 60 dienu.

SEKMES KRITERIJAI - UZRASYTI PRIES PALEIDIMA
---------------------------------------------
H1: antroje, NEMATYTOJE puseje greiciausio grizimo grupe duoda
    P_min(tikslas pirmiau) bent 8 procentiniais punktais didesne uz leciausia,
    intervalas nekerta nulio, tvarka monotoniska ABIEJOSE rinkose.
H2: geriausias ATR ruozas skiriasi nuo blogiausio bent 8 p.p. abiejose
    rinkose ir abiejose pusese. Jei kryptis skiriasi tarp isejimo taisykliu -
    tai irgi rezultatas, ir ji reikia fiksuoti.
H3: tie patys reikalavimai kaip H1, palyginus su atsitiktine kontrole.

PAPILDOMAS FILTRAS, kurio anksciau nebuvo: jei kuriai nors grupei
P_min(virsune >= tikslas) < 15%, ta grupe tavo taisyklei netinka,
nesvarbu koks jos vidurkis - nes tikslo ji beveik niekada nepasiekia.

Jei tenkinama tik viename langelyje - kandidatas kitam patikrinimui,
NE rezultatas.

Paleidimas:
    python backtest_prisilietimas.py --rinka eu --metai 10
    python backtest_prisilietimas.py --rinka us --metai 10
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    sys.exit("KLAIDA: reikia yfinance (pip install yfinance)")

# ---------------------------------------------------------------- parametrai
TIKSLAS = 0.010
SKAUSMAS = -0.028
HORIZONTAS = 3
POZICIJA = 18000.0
SANAUDOS_EUR = 10.0
MIN_APYVARTA = 5e6
MIN_IVYKIU = 200
BOOT_N = 2000
Z_LANGAS = 20
HL_MIN_OBS = 250
MIN_PASIEKIAMUMAS = 0.15      # zemiau sios ribos grupe tavo taisyklei netinka


def universas(rinka):
    try:
        import universas as U
        if rinka == "eu":
            if hasattr(U, "visi_tickeriai"):
                t = U.visi_tickeriai()
                if t:
                    return sorted(set(t))
        else:
            for nm in ("JAV_UNIVERSAS", "US_UNIVERSAS", "jav_universas"):
                if hasattr(U, nm):
                    o = getattr(U, nm)
                    if isinstance(o, dict):
                        return sorted({t for v in o.values() for t in v})
                    return sorted(set(o))
    except Exception as e:
        print(f"  (universas.py neprieinamas: {e})")
    if rinka == "eu":
        return ["SAP.DE", "ASML.AS", "MC.PA", "SIE.DE", "ALV.DE", "AIR.PA",
                "BAS.DE", "BAYN.DE", "DTE.DE", "IFX.DE", "OR.PA", "SU.PA",
                "BESI.AS", "RHM.DE", "CAP.PA", "PUM.DE"]
    return ["AAPL", "MSFT", "NVDA", "AMD", "INTC", "JPM", "XOM", "KO"]


def parsiusti(tickers, metai):
    df = yf.download(tickers, period=f"{metai}y", interval="1d",
                     auto_adjust=False, progress=False, group_by="ticker",
                     threads=True)
    out = {}
    for t in tickers:
        try:
            d = df[t] if isinstance(df.columns, pd.MultiIndex) else df
            d = d.dropna(subset=["Open", "High", "Low", "Close"])
            if len(d) > HL_MIN_OBS + 100:
                out[t] = d
        except Exception:
            continue
    return out


def paruosti(d):
    o, h, l, c, v = d["Open"], d["High"], d["Low"], d["Close"], d["Volume"]
    r = pd.DataFrame(index=d.index)
    r["open"], r["high"], r["low"], r["close"] = o, h, l, c
    prev = c.shift(1)
    r["ret"] = c / prev - 1.0
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    r["atr"] = tr.rolling(20).mean() / c
    ma, sd = c.rolling(Z_LANGAS).mean(), c.rolling(Z_LANGAS).std()
    r["z"] = (c - ma) / sd
    rng = (h - l).replace(0, np.nan)
    r["ibs"] = ((c - l) / rng).clip(0, 1)
    r["apyv"] = (c * v).rolling(20).median()
    sma200 = c.rolling(200).mean()
    r["sma200_kyla"] = (sma200 > sma200.shift(20)).astype(float)
    for k in ("atr", "z", "ibs", "apyv", "sma200_kyla", "ret"):
        r[k + "_s"] = r[k].shift(1)
    return r


def pusejimas(z):
    """AR(1) ant Z serijos: half_life = -ln2/ln(phi). Mazas -> greitas grizimas."""
    z = pd.Series(z).dropna()
    if len(z) < HL_MIN_OBS:
        return np.nan
    x, y = z.values[:-1], z.values[1:]
    if x.var() <= 0:
        return np.nan
    phi = np.cov(x, y)[0, 1] / x.var()
    if phi <= 0:
        return 0.1
    if phi >= 0.9999:
        return np.nan
    return float(-np.log(2) / np.log(phi))


def baigtys(r, i, horizontas=HORIZONTAS):
    """Grazina abi isejimo taisykles vienu metu.

    TAIKINYS : tikslas vs skausmo riba per H sesiju (TAVO taisykle)
    NAKTIS   : laikyti per nakti, parduoti kito ryto atidarymu (DABARTINIS modulis)
    VIRSUNE  : auksciausias pasiektas taskas per H sesiju (pagrindinis diagnostinis)
    """
    n = len(r)
    if i + 1 + horizontas > n:
        return None
    ieina = r["open"].iloc[i + 1]
    if not np.isfinite(ieina) or ieina <= 0:
        return None
    hi_b, lo_b = ieina * (1 + TIKSLAS), ieina * (1 + SKAUSMAS)

    virsune = float(r["high"].iloc[i + 1: i + 1 + horizontas].max() / ieina - 1.0)
    dugnas = float(r["low"].iloc[i + 1: i + 1 + horizontas].min() / ieina - 1.0)

    kmin = kmax = 0
    galut = np.nan
    for j in range(i + 1, i + 1 + horizontas):
        t_ok = r["high"].iloc[j] >= hi_b
        s_ok = r["low"].iloc[j] <= lo_b
        if t_ok and s_ok:
            kmin, kmax, galut = 0, 1, np.nan
            break
        if t_ok:
            kmin, kmax, galut = 1, 1, TIKSLAS
            break
        if s_ok:
            kmin, kmax, galut = 0, 0, SKAUSMAS
            break
    else:
        galut = float(r["close"].iloc[i + horizontas] / ieina - 1.0)

    # nakties taisykle: iejimas kito ryto atidarymu -> pardavimas dar kito ryto
    naktis = np.nan
    if i + 2 < n:
        naktis = float(r["open"].iloc[i + 2] / ieina - 1.0)

    return dict(kmin=kmin, kmax=kmax, galut=galut, virsune=virsune,
                dugnas=dugnas, naktis=naktis)


def eur(g):
    return g * POZICIJA - SANAUDOS_EUR if np.isfinite(g) else np.nan


def boot_dienomis(df, stulpelis, n=BOOT_N):
    """Bootstrap pagal DIENAS - eilutes toje pacioje dienoje juda kartu.
    Butent si pataisa parode, kad zurnalo ATR efektas nereiksmingas."""
    g = df.groupby("data")[stulpelis].mean().dropna().values
    if len(g) < 10:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(42)
    bs = [rng.choice(g, len(g), replace=True).mean() for _ in range(n)]
    return float(g.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def surinkti(duom, rezimas):
    eil = []
    for t, d in duom.items():
        r = paruosti(d)
        hl = pusejimas(r["z"].iloc[: len(r) // 2])       # TIK pirma puse
        for i in range(max(Z_LANGAS, 200) + 1, len(r) - HORIZONTAS - 2):
            if not np.isfinite(r["apyv_s"].iloc[i]) or r["apyv_s"].iloc[i] < MIN_APYVARTA:
                continue
            ibs, z, ret = r["ibs_s"].iloc[i], r["z_s"].iloc[i], r["ret_s"].iloc[i]
            if not all(np.isfinite([ibs, z, ret])):
                continue
            if rezimas == "dip":
                if not (ibs <= 0.25 and z <= -0.5):
                    continue
            elif rezimas == "kyla":
                if not (ret >= 0.005 and ibs >= 0.6):
                    continue
            else:                                          # atsitiktine kontrole
                if (i * 2654435761) % 100 >= 3:
                    continue
            b = baigtys(r, i)
            if b is None:
                continue
            eil.append(dict(
                tickeris=t, data=r.index[i], hl=hl,
                atr=r["atr_s"].iloc[i] * 100, z=z, ibs=ibs, ret=ret,
                sma200=r["sma200_kyla_s"].iloc[i],
                kmin=b["kmin"], kmax=b["kmax"],
                pasieke=float(b["virsune"] >= TIKSLAS),
                virsune=b["virsune"] * 100,
                eur_taikinys=eur(b["galut"] if np.isfinite(b["galut"])
                                 else (TIKSLAS if b["kmin"] else SKAUSMAS)),
                eur_naktis=eur(b["naktis"]),
            ))
    return pd.DataFrame(eil)


def lentele(df, stulpelis, ribos, pavadinimai, antraste):
    print()
    print("=" * 108)
    print(antraste)
    print("=" * 108)
    print(f"{'GRUPE':<20}{'N':>7}{'P(tiksl)min':>12}{'P(tiksl)max':>12}"
          f"{'P(virs>=1%)':>12}{'EUR taikinys':>26}{'EUR naktis':>15}")
    print("-" * 108)
    for lo, hi, nm in zip(ribos[:-1], ribos[1:], pavadinimai):
        m = df[(df[stulpelis] >= lo) & (df[stulpelis] < hi)]
        if len(m) < MIN_IVYKIU:
            print(f"{nm:<20}{len(m):>7}   per maza imtis")
            continue
        v, a, b = boot_dienomis(m, "eur_taikinys")
        vn, an, bn = boot_dienomis(m, "eur_naktis")
        pas = m["pasieke"].mean()
        zyme = ""
        if np.isfinite(a) and a > 0 and pas >= MIN_PASIEKIAMUMAS:
            zyme = "  <<<"
        elif pas < MIN_PASIEKIAMUMAS:
            zyme = "  (netinka: tikslo nepasiekia)"
        print(f"{nm:<20}{len(m):>7}{m['kmin'].mean()*100:>11.1f}%"
              f"{m['kmax'].mean()*100:>11.1f}%{pas*100:>11.1f}%"
              f"{v:>10.2f} [{a:>7.2f},{b:>7.2f}]{vn:>15.2f}{zyme}")


def paleisti(rinka, metai):
    tick = universas(rinka)
    print(f"{'EUROPOS' if rinka=='eu' else 'JAV'} universas: {len(tick)}")
    duom = parsiusti(tick, metai)
    print(f"  akciju su duomenimis: {len(duom)}")
    print(f"  tikslas {TIKSLAS*100:+.1f}%  skausmo riba {SKAUSMAS*100:+.1f}%  "
          f"horizontas {HORIZONTAS} sesijos")

    for rezimas, pav in (("dip", "A. DIPAS (dabartinis modelis: IBS<=0.25, Z<=-0.5)"),
                         ("kyla", "B. KYLANTI (tavo hipoteze: +0.5% ir IBS>=0.6)"),
                         ("rnd", "C. ATSITIKTINE KONTROLE")):
        df = surinkti(duom, rezimas)
        if len(df) < MIN_IVYKIU:
            print(f"\n{pav}: per mazai ivykiu ({len(df)})")
            continue
        riba = df["data"].quantile(0.5)
        p1, p2 = df[df["data"] <= riba], df[df["data"] > riba]

        print()
        print("#" * 108)
        print(f"{pav}    ivykiu: {len(df)}   padalijimas: {riba.date()}")
        print("#" * 108)
        print(f"  neaisku (abu barjerai ta pacia diena): "
              f"{(df['kmax']-df['kmin']).mean()*100:.1f}%")
        print(f"  BAZINE LINIJA: P(tikslas pirmiau) min {df['kmin'].mean()*100:.1f}% / "
              f"max {df['kmax'].mean()*100:.1f}%   "
              f"P(virsune>=+{TIKSLAS*100:.0f}%) {df['pasieke'].mean()*100:.1f}%")
        print(f"  virsunes mediana {df['virsune'].median():+.3f}%   "
              f"(gyvame zurnale buvo +0.105% - jei cia panasiai, "
              f"problema ne isejime, o atrankoje)")

        if rezimas == "rnd":
            continue

        for dalis, pv in ((p1, "1-A PUSE (paieska)"), (p2, "2-A PUSE (NEMATYTA)")):
            print()
            print(f"--- {pv}  n={len(dalis)} ---")
            lentele(dalis, "hl", [0, 0.75, 1.5, 3.0, 6.0, 1e9],
                    ["labai greitas <0.75", "greitas 0.75-1.5", "vidutinis 1.5-3",
                     "letas 3-6", "beveik nera >6"],
                    "H1: GRIZIMO PUSEJIMAS (sesijomis, vertintas TIK 1-oje puseje)")
            lentele(dalis, "atr", [0, 1.5, 2.5, 3.5, 99],
                    ["ramios <1.5%", "1.5-2.5%", "2.5-3.5%", "judrios >3.5%"],
                    "H2: ATR RUOZAS (kryptis NEUZDUOTA - zr. 'EUR naktis' stulpeli)")
            lentele(dalis, "sma200", [-0.5, 0.5, 1.5],
                    ["SMA200 nekyla", "SMA200 kyla"],
                    "KONTROLE: SMA200 filtras")

    print()
    print("=" * 108)
    print("KAIP SKAITYTI")
    print("=" * 108)
    print("  1. Pirma ziurek i P(virs>=1%). Jei grupeje jis zemiau 15%, ta grupe")
    print("     tavo taisyklei NETINKA, nesvarbu koks jos EUR vidurkis - tikslo")
    print("     ji beveik niekada nepasiekia. Butent tai rode gyvas zurnalas")
    print("     (virsunes mediana +0.105% prie +1% tikslo).")
    print()
    print("  2. 'EUR taikinys' ir 'EUR naktis' yra DVI SKIRTINGOS taisykles.")
    print("     Jei ju ATR pirmenybe skiriasi - tai ne prestaravimas, o atsakymas:")
    print("     skirtingoms taisyklems tinka skirtingos akcijos.")
    print()
    print("  3. Tikroji P(tikslas pirmiau) yra tarp 'min' ir 'max'. Jei net 'min'")
    print("     tenkina kriteriju - intraday duomenu nereikia.")
    print()
    print("  4. Svarbiausia eilute: greiciausio grizimo grupe 2-oje, NEMATYTOJE")
    print("     puseje. Tai butu pirmas stoties lygio signalas per visa projekta.")
    print()
    print("APRIBOJIMAI")
    print("  - islikimo salismas: Yahoo neturi bankrutavusiu/nupirktu bendroviu")
    print("  - neaiskios dienos: jei ju >30%, dienos barais klausimo neuzdarysim")
    print("    ir reikes 5 min duomenu (EODHD ~20 EUR uz viena menesi)")
    print("  - realus ivykdymas blogesnis: spread'as, dalinis uzpildymas")
    print("  - nakties taisykle matuojama atidarymo kainomis; realiai aukcione")
    print("    spread'as platesnis nei dienos viduryje")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rinka", choices=["eu", "us"], default="eu")
    ap.add_argument("--metai", type=int, default=10)
    ap.add_argument("--tikslas", type=float, default=None)
    ap.add_argument("--skausmas", type=float, default=None)
    ap.add_argument("--horizontas", type=int, default=None)
    a = ap.parse_args()
    if a.tikslas:
        TIKSLAS = a.tikslas / 100.0
    if a.skausmas:
        SKAUSMAS = a.skausmas / 100.0
    if a.horizontas:
        HORIZONTAS = a.horizontas
    paleisti(a.rinka, a.metai)
