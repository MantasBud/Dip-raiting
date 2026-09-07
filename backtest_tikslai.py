#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tikslu ir stop'u tinklelis — koks derinys realiai uzdirba TAVO sandoryje.

Paleidimas:
    pip install yfinance pandas numpy
    python backtest_tikslai.py
    python backtest_tikslai.py --pozicija 18000 --sanaudos-eur 10

KAM SIS TESTAS
Iki siol visi matavimai naudojo 2-3% tiksla. Bet tavo tikras sandoris yra
MC nuo 430 iki 435 = +1.16%. Tai kitas sandoris: mazesnis tikslas pasiekiamas
dazniau, bet reikalauja ankstesnio stop'o, o ankssti stop'ai anksciau matavimuose
buvo pagrindine nuostoliu priezastis. Kuris efektas nugali — nezinia, ir butent
tai cia matuojama.

METODIKA
- Kiekvienas ijejimo taskas vertinamas VISAIS tinklelio deriniais.
- Rezultatas rodomas EURAIS su tavo pozicijos dydziu ir realiomis sanaudomis,
  nes procentiniai punktai cia klaidina: 0.05% su 18 000 EUR yra 9 EUR.
- Modeliuojamas ir stop'o, ir tikslo pasiekimas tame paciame bare: laikoma,
  kad suveike STOP (konservatyvu).
- Ijejimas — kai modulis rodytu kandidata (praeje kietuosius filtrus).
- Dvi laiko puses: tinklelis ziurimas 1-oje, geriausias derinys tikrinamas 2-oje.
  Be sito tinklelis butinai rastu "geriausia" derini vien is atsitiktinumo.
"""

import argparse
import sys

import numpy as np
import pandas as pd

import universas as U

# ----------------------------- TINKLELIS -----------------------------
# Uzrasyta pries matuojant.

TIKSLAI = [0.8, 1.0, 1.2, 1.5, 2.0, 3.0]          # procentais
STOPAI = [("fiks", 0.5), ("fiks", 0.8), ("fiks", 1.2), ("fiks", 1.8),
          ("atr", 0.4), ("atr", 0.6), ("atr", 0.9)]

LAIKYMAS_SESIJU = 3
MIN_APYVARTA_EUR = 5e6
MIN_IVYKIU = 200


def flatten(df, sym):
    if isinstance(df.columns, pd.MultiIndex):
        if sym in df.columns.get_level_values(0):
            return df[sym].dropna(how="all")
        if sym in df.columns.get_level_values(-1):
            return df.xs(sym, axis=1, level=-1).dropna(how="all")
    return df.dropna(how="all")


def paruosk_akcija(d):
    """Dienos barai -> ijejimo salygos ir busimi barai rezultatui."""
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
    pc = c.shift(1)
    t = pd.DataFrame(index=d.index)
    t["C"] = c
    t["O1"] = d["Open"].shift(-1)
    t["ret"] = (c / pc - 1) * 100
    rng = (h - l).replace(0, np.nan)
    t["ibs"] = (c - l) / rng
    t["sma20"] = c.rolling(20).mean()
    t["sma50"] = c.rolling(50).mean()
    t["z20"] = (c - t["sma20"]) / c.rolling(20).std()

    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    t["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / c * 100

    t["apyv"] = (c * v).rolling(20).median()
    raud = (c < pc).astype(int)
    t["raud_serija"] = raud.groupby((raud == 0).cumsum()).cumsum()

    # Busimi barai — pirkimas kitos dienos atidarymu, laikymas LAIKYMAS_SESIJU
    for n in range(1, LAIKYMAS_SESIJU + 1):
        t[f"H{n}"] = h.shift(-n)
        t[f"L{n}"] = l.shift(-n)
        t[f"C{n}"] = c.shift(-n)
    return t


def ijejimo_salyga(t):
    """Kada modulis rodytu kandidata: atsitraukimas, ne peilis, ne virsuje."""
    return (
        (t["ret"] <= -0.8) & (t["ret"] >= -5.0)
        & (t["ibs"] <= 0.45)
        & (t["raud_serija"] <= 2)
        & (t["C"] > t["sma50"] * 0.93)
        & (t["apyv"] >= MIN_APYVARTA_EUR)
        & t["O1"].notna()
    )


def rezultatas(t, tikslas_pct, stop_tipas, stop_dydis, pozicija, sanaudos_eur):
    """Grazina pelna EURAIS kiekvienam ijejimui pagal viena tinklelio derini."""
    ijejimas = t["O1"]
    stop_pct = (stop_dydis if stop_tipas == "fiks" else stop_dydis * t["atr"])
    stop_pct = stop_pct.clip(lower=0.3) if hasattr(stop_pct, "clip") else max(0.3, stop_pct)
    stop_kaina = ijejimas * (1 - stop_pct / 100)
    tikslo_kaina = ijejimas * (1 + tikslas_pct / 100)

    pelnas = pd.Series(np.nan, index=t.index)
    baigta = pd.Series(False, index=t.index)

    for n in range(1, LAIKYMAS_SESIJU + 1):
        hi, lo = t[f"H{n}"], t[f"L{n}"]
        # Tame paciame bare abu — laikom, kad suveike stop (konservatyvu)
        stop_hit = (~baigta) & (lo <= stop_kaina)
        tiksl_hit = (~baigta) & (~stop_hit) & (hi >= tikslo_kaina)
        pelnas = pelnas.where(~stop_hit, -stop_pct)
        pelnas = pelnas.where(~tiksl_hit, tikslas_pct)
        baigta = baigta | stop_hit | tiksl_hit

    # Neuzsidare per laikyma — uzdarom paskutiniu uzdarymu
    likutis = (t[f"C{LAIKYMAS_SESIJU}"] / ijejimas - 1) * 100
    pelnas = pelnas.where(baigta, likutis)

    eur = pelnas / 100 * pozicija - sanaudos_eur
    return eur, baigta


def week_bootstrap(dienu_vid, dienos, n=2000, seed=1):
    if len(dienu_vid) < 15:
        return None
    # Nulines sklaidos apsauga: jei visi rezultatai identiski, intervalas butu
    # nulinio plocio ir verdiktas klaidingai taptu "VERTA". Taip nutinka, kai
    # tikslas per mazas dienos bariniam matavimui — jis "pasiekiamas" visada.
    if float(np.std(dienu_vid)) < 1e-9:
        return None
    rng = np.random.default_rng(seed)
    sav = pd.Series([pd.Timestamp(d).to_period("W") for d in dienos])
    gr = [np.asarray(dienu_vid)[(sav == w).to_numpy()] for w in sav.unique()]
    gr = [g for g in gr if len(g)]
    if len(gr) < 10:
        return None
    boot = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(gr), len(gr))
        boot[i] = np.concatenate([gr[j] for j in pick]).mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(np.mean(dienu_vid)), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description="Tikslu ir stop'u tinklelis")
    ap.add_argument("--metai", type=int, default=8)
    ap.add_argument("--pozicija", type=float, default=18000.0)
    ap.add_argument("--sanaudos-eur", type=float, default=10.0,
                    help="Mokesciai + spread'as EURAIS visam ciklui")
    args = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Paleisk: pip install yfinance pandas numpy")

    syms = U.visi_tikeriai()
    sekt = U.sektoriai()
    print(f"Siunciama {len(syms)} akciju ({args.metai}m)…")
    raw = yf.download(syms, period=f"{args.metai}y", interval="1d", group_by="ticker",
                      progress=False, auto_adjust=False, threads=True)

    dalys, praleista = [], 0
    for s in syms:
        try:
            d = flatten(raw, s).dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            if len(d) < 300:
                praleista += 1
                continue
            t = paruosk_akcija(d)
            t["tag"] = s
            t["sekt"] = sekt.get(s, "kita")
            t["data"] = d.index
            dalys.append(t[ijejimo_salyga(t)])
        except Exception:
            praleista += 1

    if not dalys:
        sys.exit("Nepavyko surinkti duomenu.")
    df = pd.concat(dalys, ignore_index=True).dropna(subset=["O1", "atr"])
    print(f"  akciju su duomenimis: {len(syms) - praleista}; ijejimo tasku: {len(df):,}")
    print(f"  laikotarpis: {df['data'].min():%Y-%m-%d} .. {df['data'].max():%Y-%m-%d}")
    print(f"\nPozicija {args.pozicija:,.0f} EUR, sanaudos {args.sanaudos_eur:.0f} EUR/ciklas "
          f"({args.sanaudos_eur / args.pozicija * 100:.3f}%), laikymas {LAIKYMAS_SESIJU} sesijos")

    riba = df["data"].quantile(0.5)
    p1, p2 = df["data"] <= riba, df["data"] > riba
    print(f"Padalijimas: 1-a puse iki {riba:%Y-%m-%d} "
          f"({int(p1.sum()):,} / {int(p2.sum()):,} tasku)")

    # ---------- Tinklelis 1-oje puseje ----------
    print("\n" + "=" * 100)
    print("TINKLELIS (1-oji puse) — vidutinis pelnas EURAIS vienam sandoriui")
    print("Eilutes = stop, stulpeliai = tikslas. Sanaudos jau atimtos.")
    print("=" * 100)
    antr = "STOP".ljust(12) + "".join(f"{t:>11.1f}%" for t in TIKSLAI)
    print(antr)
    print("-" * len(antr))

    rez1 = {}
    d1 = df[p1]
    for tipas, dydis in STOPAI:
        lab = f"{dydis:.1f}%" if tipas == "fiks" else f"{dydis:.1f}xATR"
        cels = []
        for tk in TIKSLAI:
            eur, baigta = rezultatas(d1, tk, tipas, dydis, args.pozicija, args.sanaudos_eur)
            v = eur.dropna()
            cels.append(v.mean() if len(v) >= MIN_IVYKIU else np.nan)
            rez1[(tipas, dydis, tk)] = v.mean() if len(v) >= MIN_IVYKIU else np.nan
        print(f"{lab:<12}" + "".join(
            f"{c:>+11.2f}" if c == c else f"{'-':>11}" for c in cels))

    geriausi = sorted([(v, k) for k, v in rez1.items() if v == v], reverse=True)[:3]
    if not geriausi:
        sys.exit("\nNepakako duomenu tinkleliui.")

    print("\nTrys geriausi 1-oje puseje:")
    for v, (tipas, dydis, tk) in geriausi:
        lab = f"{dydis:.1f}%" if tipas == "fiks" else f"{dydis:.1f}xATR"
        print(f"  stop {lab:<9} tikslas {tk:.1f}%  ->  {v:+.2f} EUR sandoriui")

    # ---------- Patvirtinimas 2-oje puseje ----------
    print("\n" + "=" * 100)
    print("PATVIRTINIMAS (2-oji puse, nematyta) — tik trys geriausi deriniai")
    print("=" * 100)
    print(f"{'DERINYS':<28} {'N':>7} {'EUR/SANDORIS':>14} {'95% INTERVALAS':>24} "
          f"{'PATAIKE':>9}")
    print("-" * 100)
    d2 = df[p2]
    for _, (tipas, dydis, tk) in geriausi:
        lab = f"stop {dydis:.1f}{'%' if tipas == 'fiks' else 'xATR'}, tikslas {tk:.1f}%"
        eur, baigta = rezultatas(d2, tk, tipas, dydis, args.pozicija, args.sanaudos_eur)
        v = eur.dropna()
        if len(v) < MIN_IVYKIU:
            print(f"{lab:<28} {len(v):>7}   per maza imtis")
            continue
        tmp = d2.assign(_e=eur).dropna(subset=["_e"])
        pagal_diena = tmp.groupby("data")["_e"].mean()
        b = week_bootstrap(pagal_diena.to_numpy(), pagal_diena.index)
        pataike = float((v > 0).mean() * 100)
        if b:
            m, lo, hi = b
            ci = f"{lo:+.2f} .. {hi:+.2f}"
            verd = "VERTA" if lo > 0 else ("nuostolinga" if hi < 0 else "nulis")
            if pataike > 85:
                verd = "NEPATIKIMA"
            print(f"{lab:<28} {len(v):>7} {m:>+13.2f}€ {ci:>24} {pataike:>8.0f}%  {verd}")
            if pataike > 85:
                print(f"{'':<28} pataikymas {pataike:.0f}% neitiketinas — tikslas "
                      f"{tk:.1f}% telpa i dienos barа beveik visada, todel dienos "
                      f"bariniai duomenys ji 'pasiekia' net kai realiai nepasiekta")
        else:
            print(f"{lab:<28} {len(v):>7} {v.mean():>+13.2f}€  "
                  f"(nulinė sklaida arba per maza dienu — rezultatas nepatikimas)")

    # ---------- Kiek sandoriu per metus ----------
    print("\n" + "=" * 100)
    print("KIEK SANDORIU REALIAI BUTU — su 5 kandidatais per diena")
    print("=" * 100)
    dienu = df["data"].nunique()
    metu = dienu / 252
    print(f"  Ijejimo tasku is viso: {len(df):,} per {dienu} dienu ({metu:.1f} metu)")
    print(f"  Vidutiniskai kandidatu per diena: {len(df) / dienu:.1f}")
    print(f"  Jei pirktum 1 per diena: ~{252:.0f} sandoriu per metus")
    v, (tipas, dydis, tk) = geriausi[0]
    print(f"  Prie geriausio derinio ({v:+.2f} EUR): ~{v * 252:+,.0f} EUR per metus")
    print(f"  Sanaudos: {args.sanaudos_eur * 252:,.0f} EUR per metus (jau atimtos)")

    print("\n" + "!" * 100)
    print("SVARBUS APRIBOJIMAS MAZIEMS TIKSLAMS")
    print("Dienos baras turi tik O/H/L/C. Jei tikslas 0.8%, o dienos diapazonas 2%,")
    print("testas laiko tiksla pasiektu beveik visada — bet realiai kaina galejo")
    print("nueiti zemyn PIRMA ir isjungti stop. Todel tikslams zemiau ~1.5% sio")
    print("testo rezultatai yra PER OPTIMISTISKI. Patikimam ju matavimui reikia")
    print("5 min. baru, o ju Yahoo duoda tik 60 dienu.")
    print("!" * 100)
    print("\nAPRIBOJIMAI: dienos barai, todel stop ir tikslas tikrinami is H/L — "
          "\nrealiai vykdymas butu blogesnis; islikimo salismas; "
          "\ntinklelis 1-oje puseje BUTINAI ras 'geriausia' derini vien is atsitiktinumo, "
          "\ntodel sprendziam TIK pagal 2-osios puses intervala.")


if __name__ == "__main__":
    main()
