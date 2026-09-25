#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PIRMO PRISILIETIMO BACKTESTAS  (v5 - 2026-09-23, tinklelis su realistisku stop vykdymu)
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

NAUJA v3: LAIKYMAS KOL ATSIGAUS
-------------------------------
Iki siol matavome tik variantus su kietu stop'u. Bet tavo tikroji taisykle
stop'o neturi - tu laikai, kol atsistato. Tai matuojama pirma karta:
stulpeliai "atsigavo", "dienu" ir "EUR laukiant" plius uodegos suvestine.
Butent uodega (kiek neatsigavo ir kaip giliai nusileido laukiant) yra
tikroji sios taisykles kaina - YDX -639 EUR tipo epizodai.

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
MAX_LAIKYMAS = 20             # sesiju, "laikyti kol atsigaus" variantui


def universas(rinka):
    """Imame ta pati universa, kuri naudoja skeneris.

    SVARBU: jei universas.py neprieinamas arba grazina per mazai tikeriu,
    testas NUTRAUKIAMAS. Ankstesneje versijoje cia tyliai buvo naudojamas
    16 akciju atsarginis sarasas, ir visas paleidimas tapo bevertis.
    """
    saltinis, t = None, []
    try:
        import universas as U
        if rinka == "eu":
            for nm in ("visi_tickeriai", "visi_tikeriai", "VISI", "EUROPOS_UNIVERSAS",
                       "UNIVERSAS", "europos_universas", "TICKERIAI"):
                if not hasattr(U, nm):
                    continue
                o = getattr(U, nm)
                o = o() if callable(o) else o
                if isinstance(o, dict):
                    t = [x for v in o.values() for x in (v if isinstance(v, (list, tuple, set)) else [v])]
                elif isinstance(o, (list, tuple, set)):
                    t = list(o)
                if t:
                    saltinis = f"universas.{nm}"
                    break
        else:
            for nm in ("UNIVERSAS_US", "JAV_UNIVERSAS", "US_UNIVERSAS",
                       "jav_universas", "jav_tickeriai", "JAV"):
                if not hasattr(U, nm):
                    continue
                o = getattr(U, nm)
                o = o() if callable(o) else o
                if isinstance(o, dict):
                    t = [x for v in o.values() for x in (v if isinstance(v, (list, tuple, set)) else [v])]
                elif isinstance(o, (list, tuple, set)):
                    t = list(o)
                if t:
                    saltinis = f"universas.{nm}"
                    break
        if not t:
            print("  universas.py rasta, bet tinkamo saraso nera. Turimi vardai:")
            print("   ", [n for n in dir(U) if not n.startswith("_")][:40])
    except Exception as e:
        print(f"  universas.py neimportuojamas: {e}")

    t = sorted({str(x).strip() for x in t if x and isinstance(x, str)})
    if len(t) < 40:
        sys.exit(
            f"\nNUTRAUKTA: is universas.py gauta tik {len(t)} tikeriu (reikia >=40).\n"
            f"Be tikro universo testas bevertis - buvusiame paleidime del to\n"
            f"buvo naudojamos vos 16 akciju ir visos pateko i viena pusejimo\n"
            f"kvintili. Pataisyk universas.py arba nurodyk teisinga funkcijos\n"
            f"varda sio failo 'universas()' funkcijoje."
        )
    print(f"  universo saltinis: {saltinis}  ({len(t)} tikeriu)")
    return t


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

    # TAVO TIKROJI TAISYKLE: jokio stop'o, laikom kol paliecia tiksla,
    # bet ne ilgiau kaip MAX_LAIKYMAS sesiju. Tai matuojama pirma karta.
    lauk_d, lauk_g, blog = np.nan, np.nan, np.nan
    kiek = min(MAX_LAIKYMAS, n - i - 1)
    if kiek > 0:
        lan_h = r["high"].iloc[i + 1: i + 1 + kiek].values
        lan_l = r["low"].iloc[i + 1: i + 1 + kiek].values
        pas = np.nonzero(lan_h >= hi_b)[0]
        if len(pas):
            d = int(pas[0]) + 1
            lauk_d, lauk_g = d, TIKSLAS
            blog = float(lan_l[:d].min() / ieina - 1.0)
        else:
            lauk_d = np.nan                       # neatsigavo per langa
            lauk_g = float(r["close"].iloc[i + kiek] / ieina - 1.0)
            blog = float(lan_l.min() / ieina - 1.0)

    # dienu santykiai tinkleliui su REALISTISKU vykdymu (tarpai per nakti)
    dienos = []
    for j in range(i + 1, min(i + 1 + horizontas, n)):
        dienos.append((
            float(r["open"].iloc[j] / ieina - 1.0),
            float(r["high"].iloc[j] / ieina - 1.0),
            float(r["low"].iloc[j] / ieina - 1.0),
            float(r["close"].iloc[j] / ieina - 1.0),
        ))

    return dict(kmin=kmin, kmax=kmax, galut=galut, virsune=virsune,
                dugnas=dugnas, naktis=naktis, dienos=dienos,
                lauk_d=lauk_d, lauk_g=lauk_g, lauk_blog=blog)


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
                dugnas_pct=b["dugnas"] * 100,
                dienos=b["dienos"],
                galut_pct=(b["galut"] * 100 if np.isfinite(b["galut"]) else 0.0),
                eur_taikinys=eur(b["galut"] if np.isfinite(b["galut"])
                                 else (TIKSLAS if b["kmin"] else SKAUSMAS)),
                eur_naktis=eur(b["naktis"]),
                lauk_eur=eur(b["lauk_g"]),
                lauk_dienos=b["lauk_d"],
                lauk_atsigavo=float(np.isfinite(b["lauk_d"])),
                lauk_blog=b["lauk_blog"] * 100 if np.isfinite(b["lauk_blog"]) else np.nan,
            ))
    return pd.DataFrame(eil)


def lentele(df, stulpelis, ribos, pavadinimai, antraste, kvantiliai=False):
    """kvantiliai=True: ruozai sudaromi is PACIU duomenu kvintiliu.

    Buvo butina, nes realiu akciju grizimo pusejimas telpa i siaura ruoza
    (3-6 sesijos) ir su fiksuotomis ribomis VISOS akcijos patekdavo i viena
    langeli - H1 tapdavo neismatuojama.
    """
    if kvantiliai:
        v = df[stulpelis].dropna()
        if len(v) < MIN_IVYKIU * 2:
            print(f"\n{antraste}\n  per maza imtis kvintiliams ({len(v)})")
            return
        q = np.unique(np.percentile(v, [0, 20, 40, 60, 80, 100]))
        if len(q) < 3:
            print(f"\n{antraste}\n  {stulpelis} beveik pastovus - kvintiliu nera")
            return
        ribos = list(q[:-1]) + [q[-1] + 1e-9]
        pavadinimai = [f"{i+1}. {ribos[i]:.2f}-{ribos[i+1]:.2f}"
                       for i in range(len(ribos) - 1)]

    print()
    print("=" * 126)
    print(antraste)
    print("=" * 126)
    print(f"{'GRUPE':<20}{'N':>7}{'P(tiksl)min':>12}{'P(virs>=1%)':>12}"
          f"{'EUR taikinys':>24}{'EUR naktis':>12}"
          f"{'atsigavo':>10}{'dienu':>7}{'EUR laukiant':>22}")
    print("-" * 126)
    for lo, hi, nm in zip(ribos[:-1], ribos[1:], pavadinimai):
        m = df[(df[stulpelis] >= lo) & (df[stulpelis] < hi)]
        if len(m) < MIN_IVYKIU:
            print(f"{nm:<20}{len(m):>7}   per maza imtis")
            continue
        v, a, b = boot_dienomis(m, "eur_taikinys")
        vn, _, _ = boot_dienomis(m, "eur_naktis")
        vl, al, bl = boot_dienomis(m, "lauk_eur")
        pas = m["pasieke"].mean()
        ats = m["lauk_atsigavo"].mean()
        dien = m["lauk_dienos"].median()
        zyme = ""
        if np.isfinite(al) and al > 0:
            zyme = "  <<< laukiant teigiama"
        elif np.isfinite(a) and a > 0 and pas >= MIN_PASIEKIAMUMAS:
            zyme = "  <<<"
        print(f"{nm:<20}{len(m):>7}{m['kmin'].mean()*100:>11.1f}%{pas*100:>11.1f}%"
              f"{v:>9.2f} [{a:>6.2f},{b:>6.2f}]{vn:>12.2f}"
              f"{ats*100:>9.1f}%{dien:>7.1f}{vl:>8.2f} [{al:>6.2f},{bl:>6.2f}]{zyme}")


def paleisti(rinka, metai):
    tick = universas(rinka)
    print(f"{'EUROPOS' if rinka=='eu' else 'JAV'} universas: {len(tick)}")
    duom = parsiusti(tick, metai)
    print(f"  akciju su duomenimis: {len(duom)}")
    print(f"  tikslas {TIKSLAS*100:+.1f}%  skausmo riba {SKAUSMAS*100:+.1f}%  "
          f"horizontas {HORIZONTAS} sesijos")

    sanaudos_pct = SANAUDOS_EUR / POZICIJA * 100
    luzis = (sanaudos_pct + abs(SKAUSMAS) * 100) / (TIKSLAS * 100 + abs(SKAUSMAS) * 100)
    print()
    print("  LUZIO TASKAS: prie siu parametru taisykle teigiama tik jei")
    print(f"  P(tikslas pirmiau) > {luzis*100:.1f}%.  Zemiau - struktūriskai nuostolinga,")
    print("  ir jokia atranka to nepakeis, jei P lieka po sia riba.")

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
        nea = df[df["lauk_atsigavo"] < 0.5]
        print(f"  LAIKYMAS KOL ATSIGAUS (max {MAX_LAIKYMAS} sesiju): "
              f"atsigavo {df['lauk_atsigavo'].mean()*100:.1f}%, "
              f"mediana {df['lauk_dienos'].median():.0f} d.")
        print(f"    blogiausias nuosmukis laukiant: mediana "
              f"{df['lauk_blog'].median():+.2f}%, "
              f"5% blogiausiu {np.nanpercentile(df['lauk_blog'], 5):+.2f}%, "
              f"blogiausias {np.nanmin(df['lauk_blog']):+.2f}%")
        if len(nea):
            print(f"    NEATSIGAVO {len(nea)} ({len(nea)/len(df)*100:.1f}%): "
                  f"vid. rezultatas {nea['galut_pct'].mean():+.2f}%, "
                  f"blogiausias {nea['lauk_blog'].min():+.2f}%  <- cia tavo YDX rizika")

        if rezimas == "rnd":
            continue

        for dalis, pv in ((p1, "1-A PUSE (paieska)"), (p2, "2-A PUSE (NEMATYTA)")):
            print()
            print(f"--- {pv}  n={len(dalis)} ---")
            lentele(dalis, "hl", None, None,
                    "H1: GRIZIMO PUSEJIMAS - KVINTILIAI (vertinta TIK 1-oje puseje)",
                    kvantiliai=True)
            lentele(dalis, "atr", [0, 1.5, 2.5, 3.5, 99],
                    ["ramios <1.5%", "1.5-2.5%", "2.5-3.5%", "judrios >3.5%"],
                    "H2: ATR RUOZAS (kryptis NEUZDUOTA - zr. 'EUR naktis' stulpeli)")
            lentele(dalis, "sma200", [-0.5, 0.5, 1.5],
                    ["SMA200 nekyla", "SMA200 kyla"],
                    "KONTROLE: SMA200 filtras")

    # ------- parametru tinklelis SU PATIKRINIMU NEMATYTOJE PUSEJE
    df = surinkti(duom, "dip")
    if len(df) >= MIN_IVYKIU:
        riba = df["data"].quantile(0.5)
        d1, d2 = df[df["data"] <= riba], df[df["data"] > riba]

        def ivertink(d, t, pain, realistiskai=True):
            """Vykdymas su nakties tarpais.

            Tinklelis iki v5 laike, kad stop'as ivykdomas TIKSLIAI ties savo
            lygiu. Siauriems stop'ams tai netiesa: 0.8% stop'as yra uz nakties
            tarpo ribu ~24-66% atveju (priklausomai nuo ATR), tad realiai
            ivykdomas atidarymo kaina, kuri gali buti daug blogesne.
            Cia: jei diena ATSIDARO uz barjero - iseinam ATIDARYMO kaina.
            """
            rez, pataik = [], 0
            for dienos in d["dienos"]:
                g = None
                for (o, h, l, c) in dienos:
                    if realistiskai and o <= pain:
                        g = o                      # tarpas zemyn - vykdymas blogesnis
                        break
                    if realistiskai and o >= t:
                        g = o                      # tarpas aukstyn - vykdymas geresnis
                        break
                    hit_t, hit_s = h >= t, l <= pain
                    if hit_t and hit_s:
                        g = pain                   # neaisku -> konservatyviai
                        break
                    if hit_t:
                        g = t
                        break
                    if hit_s:
                        g = pain
                        break
                if g is None:
                    g = dienos[-1][3] if dienos else 0.0
                rez.append(g)
                if g >= t - 1e-12:
                    pataik += 1
            if not rez:
                return 0.0, 0.0
            P = pataik / len(rez)
            return P, float(np.mean(rez)) * POZICIJA - SANAUDOS_EUR

        print()
        print("#" * 126)
        print("D. PARAMETRU TINKLELIS - SU PATIKRINIMU NEMATYTOJE PUSEJE")
        print("   Derinys imamas tik jei teigiamas ABIEJOSE pusese. Vienos puses")
        print("   rezultatas yra tas pats slenksciu parinkimas pamacius duomenis,")
        print("   kuris siame projekte jau kelis kartus klaidino.")
        print("#" * 126)
        print("   'idealus' = stop ivykdomas tiksliai ties lygiu (taip buvo iki v5)")
        print("   'realus'   = jei diena atsidaro uz barjero, iseinam ATIDARYMO kaina")
        print(f"{'tikslas':>8}{'skausmas':>10}{'reikia P':>10}"
              f"{'ideal 1-a':>11}{'ideal 2-a':>11}"
              f"{'REAL 1-a':>11}{'REAL 2-a':>11}{'verdiktas':>16}")
        print("-" * 126)
        for t in (0.005, 0.008, 0.010, 0.015, 0.020, 0.030):
            for pain in (-0.008, -0.010, -0.015, -0.020, -0.028, -0.040):
                reikia = (SANAUDOS_EUR / POZICIJA + abs(pain)) / (t + abs(pain))
                _, i1 = ivertink(d1, t, pain, False)
                _, i2 = ivertink(d2, t, pain, False)
                _, r1 = ivertink(d1, t, pain, True)
                _, r2 = ivertink(d2, t, pain, True)
                if r1 > 0 and r2 > 0:
                    v = "ABI TEIGIAMOS"
                elif i1 > 0 and i2 > 0:
                    v = "tik idealiai"
                else:
                    v = ""
                print(f"{t*100:>7.1f}%{pain*100:>9.1f}%{reikia*100:>9.1f}%"
                      f"{i1:>11.2f}{i2:>11.2f}{r1:>11.2f}{r2:>11.2f}{v:>16}")

    print()
    print("=" * 126)
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
