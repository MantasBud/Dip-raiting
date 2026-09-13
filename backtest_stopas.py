#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stop'o vertes backtestas — ar jis pateisina savo kaina.

KLAUSIMAS
Ankstesnis matavimas: isejimas be stop'o +27.31 EUR, su 3% stop'u +10.57 EUR.
Stop kainuoja ~17 EUR sandoriui. Nepriklausomi saltiniai (QuantifiedStrategies)
teigia ta pati: grizimo prie vidurkio strategijose stop'as beveik visada kenkia,
nes ismusa butent tada, kai signalas stipriausias.

Vartotojo argumentas: katastrofiniu judesiu vis tiek nesustabdysim, nes jie
ateina per nakti (naujiena, sektoriaus kritimas, makro), o tada stop suveikia
ties atidarymo kaina, ne ties stop lygiu.

KA CIA MATUOJAM (uzrasyta pries paleidima)
A. Nuostoliu uodega be stop'o: kiek procentu sandoriu baigiasi blogiau nei
   -5%, -10%, -15%; blogiausias atvejis; vidutinis blogiausias menuo.
B. Ar stop'u ismustos pozicijos atsigauna: is tu, kurios pasieke -2/-3/-4%,
   kiek vėliau uzsidare su pelnu iki isejimo salygos.
C. Ar didelis kritimas nuspejamas: ar ji lyde sektoriaus kritimas (bendras
   veiksnys) ar jis buvo pavienis (idiosinkratinis, nenuspejamas).
D. Ar stop suveikia ties savo lygiu, ar per nakti ji peršoka.
E. Stop lygiu tinklelis eurais: 2%, 3%, 4%, 6%, 8%, be stop'o.

Paleidimas:
    python backtest_stopas.py --rinka eu
"""

import argparse
import sys

import numpy as np
import pandas as pd

import universas as U

MIN_IVYKIU = 200
LAIKYMAS = 10          # kaip modulyje: EXIT_MAX_DIENU
EXIT_IBS = 0.80


def flatten(df, sym):
    if isinstance(df.columns, pd.MultiIndex):
        if sym in df.columns.get_level_values(0):
            return df[sym].dropna(how="all")
        if sym in df.columns.get_level_values(-1):
            return df.xs(sym, axis=1, level=-1).dropna(how="all")
    return df.dropna(how="all")


def paruosk(d):
    c, h, l, v, o = d["Close"], d["High"], d["Low"], d["Volume"], d["Open"]
    pc = c.shift(1)
    t = pd.DataFrame(index=d.index)
    t["C"], t["O"], t["H"], t["L"] = c, o, h, l
    t["ret"] = (c / pc - 1) * 100
    rng = (h - l).replace(0, np.nan)
    t["ibs"] = (c - l) / rng
    t["apyv"] = (c * v).rolling(20).median()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    t["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / c * 100
    for n in range(1, LAIKYMAS + 1):
        t[f"O{n}"], t[f"H{n}"], t[f"L{n}"], t[f"C{n}"] = (
            o.shift(-n), h.shift(-n), l.shift(-n), c.shift(-n))
        t[f"ibs{n}"] = ((c.shift(-n) - l.shift(-n))
                        / (h.shift(-n) - l.shift(-n)).replace(0, np.nan))
    return t


def simuliuok(g, stop_pct=None):
    """Vienas praejimas per laikymo langa.

    Grazina: (rezultatas %, kaip baigesi, ar stop peršoktas, giliausias minusas)
    Ijejimas — kitos dienos atidarymu, kaip moduly.
    """
    ijej = g["O1"]
    n = len(g)
    rez = np.full(n, np.nan)
    kaip = np.array(["laikas"] * n, dtype=object)
    persoko = np.zeros(n, dtype=bool)
    giliausia = np.zeros(n)
    baigta = np.zeros(n, dtype=bool)

    stop_k = ijej * (1 - stop_pct / 100) if stop_pct else None

    for i in range(1, LAIKYMAS + 1):
        lo, hi, cl = g[f"L{i}"], g[f"H{i}"], g[f"C{i}"]
        op = g[f"O{i}"]
        # giliausias minusas sekamas visa laika
        dabar = (lo / ijej - 1) * 100
        giliausia = np.where((~baigta) & (dabar < giliausia), dabar, giliausia)

        if stop_k is not None:
            # Ar atidarymas jau zemiau stop'o — tada vykdymas ties atidarymu
            gap_hit = (~baigta) & (op <= stop_k)
            rez = np.where(gap_hit, (op / ijej - 1) * 100, rez)
            kaip = np.where(gap_hit, "stop per nakti", kaip)
            persoko = persoko | gap_hit
            baigta = baigta | gap_hit

            hit = (~baigta) & (lo <= stop_k)
            rez = np.where(hit, -stop_pct, rez)
            kaip = np.where(hit, "stop", kaip)
            baigta = baigta | hit

        # Isejimo salyga: IBS virs ribos -> parduodam kito ryto atidarymu
        sal = (~baigta) & (g[f"ibs{i}"] >= EXIT_IBS)
        if i < LAIKYMAS:
            rez = np.where(sal, (g[f"O{i + 1}"] / ijej - 1) * 100, rez)
            kaip = np.where(sal, "salyga", kaip)
            baigta = baigta | sal

    rez = np.where(baigta, rez, (g[f"C{LAIKYMAS}"] / ijej - 1) * 100)
    return rez, kaip, persoko, giliausia


def main():
    ap = argparse.ArgumentParser(description="Stop'o vertes backtestas")
    ap.add_argument("--metai", type=int, default=10)
    ap.add_argument("--rinka", default="eu", choices=["eu", "us"])
    ap.add_argument("--pozicija", type=float, default=18000.0)
    ap.add_argument("--sanaudos-eur", type=float, default=5.0)
    a = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Paleisk: pip install yfinance pandas numpy")

    uni = U.UNIVERSAS_US if a.rinka == "us" else U.UNIVERSAS
    sekt = U.sektoriai(uni)
    syms = U.visi_tikeriai(uni)
    print(f"{'JAV' if a.rinka == 'us' else 'EUROPOS'} universas: {len(syms)}\n")
    raw = yf.download(syms, period=f"{a.metai}y", interval="1d", group_by="ticker",
                      progress=False, auto_adjust=False, threads=True)

    dalys, be_d = [], 0
    for s in syms:
        try:
            d = flatten(raw, s).dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            if len(d) < 400:
                be_d += 1
                continue
            t = paruosk(d)
            t["tag"], t["sekt"], t["data"] = s, sekt.get(s, "kita"), d.index
            dalys.append(t)
        except Exception:
            be_d += 1
    if not dalys:
        sys.exit("Nepavyko surinkti duomenu.")
    df = pd.concat(dalys, ignore_index=True)
    df = df[(df["ret"] <= -0.8) & (df["ret"] >= -6.0) & (df["apyv"] >= 5e6)].copy()
    df = df.dropna(subset=["O1", f"C{LAIKYMAS}"])

    riba = pd.to_datetime(df["data"]).quantile(0.5)
    IBS_RIBA = float(df.loc[pd.to_datetime(df["data"]) <= riba, "ibs"].quantile(0.20))
    kand = df[df["ibs"] <= IBS_RIBA].copy()
    print(f"  akciju: {df['tag'].nunique()}, kandidatu: {len(kand):,}, be duomenu: {be_d}")
    print(f"  IBS riba: {IBS_RIBA:.3f}, laikymas iki {LAIKYMAS} sesiju\n")

    # ---------- E. STOP LYGIU TINKLELIS ----------
    print("=" * 100)
    print("E. STOP LYGIU TINKLELIS — vidutinis rezultatas EURAIS")
    print("=" * 100)
    print(f"{'STOP':<14} {'VID. EUR':>10} {'>0 dalis':>10} {'BLOGIAUSIAS':>12} "
          f"{'5% blogiausiu':>14} {'STOP DALIS':>11}")
    print("-" * 100)
    rezultatai = {}
    for sp in (2.0, 3.0, 4.0, 6.0, 8.0, None):
        rez, kaip, persoko, gil = simuliuok(kand, sp)
        eur = rez / 100 * a.pozicija - a.sanaudos_eur
        lab = f"{sp:.0f}%" if sp else "be stop'o"
        stop_dalis = float(np.mean([k.startswith("stop") for k in kaip]) * 100)
        rezultatai[lab] = dict(rez=rez, eur=eur, kaip=kaip, persoko=persoko, gil=gil)
        print(f"{lab:<14} {np.nanmean(eur):>+9.2f}€ {float((rez > 0).mean() * 100):>9.0f}% "
              f"{np.nanmin(rez):>11.1f}% {np.nanpercentile(rez, 5):>13.1f}% "
              f"{stop_dalis:>10.0f}%")

    # ---------- A. NUOSTOLIU UODEGA BE STOP'O ----------
    print("\n" + "=" * 100)
    print("A. NUOSTOLIU UODEGA BE STOP'O — kiek daznai ir kiek giliai")
    print("=" * 100)
    be = rezultatai["be stop'o"]["rez"]
    eur_be = rezultatai["be stop'o"]["eur"]
    for riba_p in (-3, -5, -8, -10, -15, -20):
        dalis = float((be <= riba_p).mean() * 100)
        kiek = int((be <= riba_p).sum())
        print(f"  blogiau nei {riba_p:>4}%:  {dalis:>5.2f}% sandoriu ({kiek:>5}), "
              f"vidutiniskai {a.pozicija * riba_p / 100:>8,.0f} EUR")
    print(f"\n  Blogiausias vienas sandoris: {np.nanmin(be):.1f}% "
          f"({np.nanmin(be) / 100 * a.pozicija:,.0f} EUR)")
    print(f"  Blogiausi 1% sandoriu: {np.nanpercentile(be, 1):.1f}%")
    with_stop = rezultatai["3%"]["rez"]
    print(f"  Palyginimui su 3% stop'u: blogiausias {np.nanmin(with_stop):.1f}%")

    # ---------- B. AR ISMUSTOS POZICIJOS ATSIGAUNA ----------
    print("\n" + "=" * 100)
    print("B. AR STOP'U ISMUSTOS POZICIJOS ATSIGAUNA")
    print("Is tu, kurios pasieke stop lygi, kiek BE STOP'O baigesi su pelnu.")
    print("=" * 100)
    print(f"{'STOP LYGIS':<14} {'ISMUSTU':>9} {'ATSIGAVO':>10} {'VID. BE STOP':>14} "
          f"{'VID. SU STOP':>14} {'SKIRTUMAS':>11}")
    print("-" * 100)
    for sp in (2.0, 3.0, 4.0, 6.0):
        r = rezultatai[f"{sp:.0f}%"]
        ismusti = np.array([k.startswith("stop") for k in r["kaip"]])
        if ismusti.sum() < 50:
            continue
        be_i = be[ismusti]
        su_i = r["rez"][ismusti]
        atsig = float((be_i > 0).mean() * 100)
        print(f"{sp:.0f}%{'':<11} {int(ismusti.sum()):>9} {atsig:>9.0f}% "
              f"{np.nanmean(be_i):>+13.2f}% {np.nanmean(su_i):>+13.2f}% "
              f"{np.nanmean(be_i) - np.nanmean(su_i):>+10.2f}")
    print("\n  Jei 'ATSIGAVO' virs ~55% ir skirtumas teigiamas — stop'as kenkia.")

    # ---------- C. AR DIDELIS KRITIMAS NUSPEJAMAS ----------
    print("\n" + "=" * 100)
    print("C. AR DIDELIS KRITIMAS BUVO BENDRAS AR PAVIENIS")
    print("Vartotojo hipoteze: giliausi kritimai ateina su sektoriumi, todel")
    print("ju nesustabdysi jokia atranka.")
    print("=" * 100)
    try:
        kand = kand.reset_index(drop=True)
        kand["_be"] = be
        kand["_gil"] = rezultatai["be stop'o"]["gil"]
        # Sektoriaus judesys per ta pati laikyma
        kand["sekt_rez"] = kand.groupby(["data", "sekt"])["_be"].transform("median")
        kand["rinkos_rez"] = kand.groupby("data")["_be"].transform("median")
        blogi = kand[kand["_be"] <= -8]
        if len(blogi) >= 30:
            su_sekt = float((blogi["sekt_rez"] <= -3).mean() * 100)
            su_rinka = float((blogi["rinkos_rez"] <= -3).mean() * 100)
            print(f"  Sandoriu, kurie baigesi blogiau nei -8%: {len(blogi)}")
            print(f"    is ju kartu krito SEKTORIUS (mediana <= -3%): {su_sekt:.0f}%")
            print(f"    is ju kartu krito VISA RINKA  (mediana <= -3%): {su_rinka:.0f}%")
            print(f"    likusieji — pavieniai, idiosinkratiniai: {100 - su_sekt:.0f}%")
        else:
            print(f"  per maza imtis ({len(blogi)} atveju)")
    except Exception as e:
        print(f"  (analize praleista: {type(e).__name__}: {e})")

    # ---------- D. AR STOP SUVEIKIA TIES SAVO LYGIU ----------
    print("\n" + "=" * 100)
    print("D. AR STOP SUVEIKIA TIES SAVO LYGIU, AR PERSOKAMAS PER NAKTI")
    print("=" * 100)
    for sp in (2.0, 3.0, 4.0):
        r = rezultatai[f"{sp:.0f}%"]
        ismusti = np.array([k.startswith("stop") for k in r["kaip"]])
        nakt = r["persoko"]
        if ismusti.sum() < 50:
            continue
        dalis = float(nakt.sum() / max(1, ismusti.sum()) * 100)
        vid_nakt = float(np.nanmean(r["rez"][nakt])) if nakt.sum() else float("nan")
        print(f"  {sp:.0f}% stop: {int(ismusti.sum())} suveikimu, "
              f"{dalis:.0f}% is ju persokta per nakti, "
              f"ju vidutinis rezultatas {vid_nakt:+.2f}% (vietoj -{sp:.0f}%)")
    print("\n  Kuo didesne 'persokta per nakti' dalis, tuo mazesne stop'o apsauga.")

    # ---------- GALUTINIS ----------
    print("\n" + "=" * 100)
    print("GALUTINIS PALYGINIMAS")
    print("=" * 100)
    b3 = rezultatai["3%"]["eur"]
    print(f"  Su 3% stop'u:  vidurkis {np.nanmean(b3):+.2f}€, "
          f"blogiausias {np.nanmin(b3):,.0f}€")
    print(f"  Be stop'o:     vidurkis {np.nanmean(eur_be):+.2f}€, "
          f"blogiausias {np.nanmin(eur_be):,.0f}€")
    print(f"  Skirtumas vidurkiui: {np.nanmean(eur_be) - np.nanmean(b3):+.2f}€ "
          f"sandoriui be stop'o naudai")
    print(f"  Kaina: blogiausias atvejis pablogeja "
          f"{np.nanmin(eur_be) - np.nanmin(b3):,.0f}€")
    n_year = 50
    print(f"\n  Prie ~{n_year} sandoriu per metus:")
    print(f"    be stop'o duotu {np.nanmean(eur_be) * n_year:+,.0f}€ per metus")
    print(f"    su 3% stop'u   {np.nanmean(b3) * n_year:+,.0f}€ per metus")
    print(f"    bet vienas blogiausias sandoris be stop'o kainuotu "
          f"{np.nanmin(eur_be):,.0f}€ — tai "
          f"{abs(np.nanmin(eur_be)) / max(1, abs(np.nanmean(eur_be) * n_year)) * 100:.0f}% "
          f"metu rezultato")

    print("\nAPRIBOJIMAI: islikimo salismas (bankrutavusiu nera, todel uodega "
          "\nPER SVELNI); stop tikrinamas is dienos H/L, realus vykdymas blogesnis; "
          "\nvienos pozicijos simuliacija — su visu portfeliu viename sandoryje "
          "\nuodegos reiksme daug didesne nei rodo vidurkiai.")


if __name__ == "__main__":
    main()
