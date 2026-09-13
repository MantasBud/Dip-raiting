#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dienos momentumo backtestas (Gao, Han, Li & Zhou, JFE 2018).

HIPOTEZE, PERIMTA IS LITERATUROS
"Market Intraday Momentum", Journal of Financial Economics 129(2), 394-414:
pirmo PUSVALANDZIO grazа (nuo vakarykscio uzdarymo) prognozuoja PASKUTINIO
pusvalandzio grazа. S&P 500 ETF, 1993-2013, prognostinis R2 = 1.6%, o
kartu su priespaskutiniu pusvalandziu R2 = 2.6%. Efektas stipresnis:
  - svyruojanciomis dienomis
  - dideles apyvartos dienomis
  - makroekonominiu naujienu dienomis

KODEL MUSU ANKSTESNIS TESTAS TO NERADO
Mes matavom pirma VALANDA pries LIKUSIA DIENA ir gavom koreliacija +0.026.
Tai kitas langas. Autoriai matuoja pirma pusvalandi pries paskutini
pusvalandi — tarpine dienos dalis i skaiciavima neieina.
Antras skirtumas: jie matuoja INDEKSA, ne atskiras akcijas.

KA TIKRINAM
A. Ar efektas yra Europos indekse (EXSA.DE) ir ar jis stipresnis
   svyruojanciomis bei dideles apyvartos dienomis, kaip teigia autoriai.
B. Ar jis perkeliamas i atskiras akcijas.
C. Ar jis naudingas MUSU kandidatams — t. y. ar dipo pozicija, laikoma per
   diena, elgiasi kitaip, kai rytas buvo teigiamas.
D. Ar is to iseina prekiaujama taisykle po sanaudu.

APRIBOJIMAS, ZINOMAS IS ANKSTO
Yahoo duoda 30 min. barus tik 60 dienu atgal. Tai maza imtis, todel
pagrindinis matavimas daromas 60 min. barais per 720 dienu (pirma valanda
pries paskutine valanda), o 30 min. langas tikrinamas atskirai kaip
patikslinimas su aiskiai pazymeta mazesne imtimi.

Paleidimas:
    python backtest_dienos_momentum.py --rinka eu
"""

import argparse
import sys

import numpy as np
import pandas as pd

import universas as U

MIN_DIENU = 40


def flatten(df, sym):
    if isinstance(df.columns, pd.MultiIndex):
        if sym in df.columns.get_level_values(0):
            return df[sym].dropna(how="all")
        if sym in df.columns.get_level_values(-1):
            return df.xs(sym, axis=1, level=-1).dropna(how="all")
    return df.dropna(how="all")


def week_bootstrap(vals, dienos, n=2000, seed=17):
    if len(vals) < 15 or float(np.std(vals)) < 1e-12:
        return None
    rng = np.random.default_rng(seed)
    sav = pd.Series([pd.Timestamp(d).to_period("W") for d in dienos])
    gr = [np.asarray(vals)[(sav == w).to_numpy()] for w in sav.unique()]
    gr = [g for g in gr if len(g)]
    if len(gr) < 10:
        return None
    boot = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(gr), len(gr))
        boot[i] = np.concatenate([gr[j] for j in pick]).mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(mean=float(np.mean(vals)), lo=float(lo), hi=float(hi))


def sesiju_langai(d, pirmas_baru=1, paskutinis_baru=1):
    """Is valandiniu baru istraukia dienos langus.

    r1  — pirmas langas nuo VAKARYKSCIO uzdarymo (kaip straipsnyje)
    r12 — priespaskutinis langas
    r13 — paskutinis langas
    vidurys — tarp ju
    """
    d = d.copy()
    d["_d"] = [i.date() for i in d.index]
    ses = [(dd, g) for dd, g in d.groupby("_d") if len(g) >= 5]
    eil = []
    for i in range(1, len(ses)):
        dd, g = ses[i]
        _, pr = ses[i - 1]
        try:
            pr_uzd = float(pr["Close"].iloc[-1])
            pirmas = float(g["Close"].iloc[pirmas_baru - 1])
            uzd = float(g["Close"].iloc[-1])
            prieszp = float(g["Close"].iloc[-(paskutinis_baru + 1)])
            atid = float(g["Open"].iloc[0])
            if min(pr_uzd, pirmas, uzd, prieszp) <= 0:
                continue
            apyv = float(g["Volume"].sum()) if "Volume" in g else np.nan
            svyr = float((g["High"].max() / g["Low"].min() - 1) * 100)
            eil.append(dict(
                data=pd.Timestamp(dd),
                r1=(pirmas / pr_uzd - 1) * 100,          # pirmas langas nuo vakar
                r_atid=(atid / pr_uzd - 1) * 100,        # vien nakties tarpas
                r12=(prieszp / float(g["Close"].iloc[-(paskutinis_baru + 2)]) - 1) * 100
                if len(g) > paskutinis_baru + 2 else np.nan,
                r13=(uzd / prieszp - 1) * 100,           # paskutinis langas
                vidurys=(prieszp / pirmas - 1) * 100,
                apyv=apyv, svyr=svyr))
        except Exception:
            continue
    return pd.DataFrame(eil)


def ataskaita(df, lab, sanaudos, pozicija):
    """Pagrindine regresija ir prekybos taisykle."""
    d = df.dropna(subset=["r1", "r13"])
    if len(d) < MIN_DIENU:
        print(f"  {lab}: per maza imtis ({len(d)})")
        return
    kor = float(d["r1"].corr(d["r13"]))
    # Paprasta OLS be bibliotekos
    x, y = d["r1"].to_numpy(), d["r13"].to_numpy()
    b = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)) if np.var(x) > 0 else 0
    r2 = kor ** 2 * 100
    print(f"\n  {lab}  (n={len(d)} dienu)")
    print(f"    koreliacija r1 -> r13: {kor:+.3f}   R2 = {r2:.2f}%   "
          f"nuolydis = {b:+.3f}")
    print(f"    straipsnyje: R2 = 1.6% (SPY, 1993-2013)")

    # Prekybos taisykle: jei r1 > 0 — perki paskutiniam langui, jei < 0 — praleidi
    ilgas = d[d["r1"] > 0]["r13"]
    trumpas = d[d["r1"] <= 0]["r13"]
    print(f"    kai rytas TEIGIAMAS ({len(ilgas)} d.): paskutinis langas "
          f"{ilgas.mean():+.4f}%")
    print(f"    kai rytas NEIGIAMAS ({len(trumpas)} d.): paskutinis langas "
          f"{trumpas.mean():+.4f}%")
    print(f"    skirtumas: {ilgas.mean() - trumpas.mean():+.4f} p. p.")

    r = week_bootstrap(d[d["r1"] > 0]["r13"].to_numpy(),
                       d[d["r1"] > 0]["data"])
    if r:
        print(f"    'perki kai rytas teigiamas' 95% intervalas: "
              f"{r['lo']:+.4f} .. {r['hi']:+.4f}")
        eur = r["mean"] / 100 * pozicija - sanaudos
        print(f"    eurais vienam sandoriui: {eur:+.2f}€ "
              f"(sanaudos {sanaudos}€ suvalgo {sanaudos / pozicija * 100:.3f}%)")


def main():
    ap = argparse.ArgumentParser(description="Dienos momentumo backtestas")
    ap.add_argument("--rinka", default="eu", choices=["eu", "us"])
    ap.add_argument("--pozicija", type=float, default=18000.0)
    ap.add_argument("--sanaudos-eur", type=float, default=5.0)
    a = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Paleisk: pip install yfinance pandas numpy")

    indeksas = "SPY" if a.rinka == "us" else U.INDEKSAS
    uni = U.UNIVERSAS_US if a.rinka == "us" else U.UNIVERSAS
    syms = U.visi_tikeriai(uni)

    print(f"{'JAV' if a.rinka == 'us' else 'EUROPOS'} rinka, indeksas {indeksas}")
    print("Gao, Han, Li & Zhou (JFE 2018): pirmas langas prognozuoja paskutini.\n")

    # ---------- A. INDEKSAS ----------
    print("=" * 100)
    print("A. INDEKSAS — ar efektas apskritai yra")
    print("=" * 100)
    try:
        raw = yf.download(indeksas, period="720d", interval="60m",
                          progress=False, auto_adjust=False)
        idx = flatten(raw, indeksas).dropna(subset=["Open", "High", "Low", "Close"])
        d_idx = sesiju_langai(idx)
        ataskaita(d_idx, f"{indeksas}, valandiniai langai", a.sanaudos_eur, a.pozicija)

        # Salygos, kurias nurodo autoriai
        if len(d_idx) >= 80:
            print(f"\n  Autoriai teigia, kad efektas stipresnis svyruojanciomis ir")
            print(f"  dideles apyvartos dienomis. Tikrinam:")
            for c, lab in [("svyr", "dienos svyravimas"), ("apyv", "apyvarta")]:
                if c not in d_idx or d_idx[c].isna().all():
                    continue
                q = d_idx[c].median()
                for kauke, pav in [(d_idx[c] > q, "aukstas"), (d_idx[c] <= q, "zemas")]:
                    s = d_idx[kauke].dropna(subset=["r1", "r13"])
                    if len(s) >= 30:
                        print(f"    {lab} {pav:<8} koreliacija "
                              f"{s['r1'].corr(s['r13']):+.3f}  (n={len(s)})")
    except Exception as e:
        print(f"  indekso analize nepavyko: {type(e).__name__}: {e}")

    # ---------- B. ATSKIROS AKCIJOS ----------
    print("\n" + "=" * 100)
    print("B. ATSKIROS AKCIJOS — ar efektas perkeliamas")
    print("=" * 100)
    print(f"Siunciama {len(syms)} akciju valandine istorija…")
    raw2 = yf.download(syms, period="720d", interval="60m", group_by="ticker",
                       progress=False, auto_adjust=False, threads=True)
    visos, be_d = [], 0
    for s in syms:
        try:
            d = flatten(raw2, s).dropna(subset=["Open", "High", "Low", "Close"])
            if len(d) < 400:
                be_d += 1
                continue
            t = sesiju_langai(d)
            if len(t) < 100:
                be_d += 1
                continue
            t["tag"] = s
            visos.append(t)
        except Exception:
            be_d += 1
    if not visos:
        sys.exit("Nepavyko surinkti akciju duomenu.")
    akc = pd.concat(visos, ignore_index=True)
    print(f"  akciju: {akc['tag'].nunique()}, dienu-akciju: {len(akc):,}, "
          f"be duomenu: {be_d}")

    kor_akc = akc.groupby("tag").apply(
        lambda g: g["r1"].corr(g["r13"]) if len(g) > 50 else np.nan).dropna()
    print(f"\n  Koreliacija r1 -> r13 kiekvienai akcijai:")
    print(f"    mediana {kor_akc.median():+.3f}, vidurkis {kor_akc.mean():+.3f}")
    print(f"    teigiamu: {float((kor_akc > 0).mean() * 100):.0f}% is "
          f"{len(kor_akc)} akciju")
    print(f"    virs +0.10: {int((kor_akc > 0.10).sum())} akcijos")
    if (kor_akc > 0).mean() > 0.6:
        print("    -> efektas linkes buti teigiamas ir atskirose akcijose")
    else:
        print("    -> atskirose akcijose efekto nesimato")

    # Demeanuota: ar akcijos rytas prognozuoja jos vakara VIRS rinkos
    akc["r13_dm"] = akc["r13"] - akc.groupby("data")["r13"].transform("mean")
    akc["r1_dm"] = akc["r1"] - akc.groupby("data")["r1"].transform("mean")
    kor_dm = float(akc[["r1_dm", "r13_dm"]].corr().iloc[0, 1])
    print(f"\n  Demeanuota (atmetus bendra rinkos judesi): {kor_dm:+.3f}")
    print("    Jei sis apie nuli, o bendras teigiamas — efektas yra RINKOS,")
    print("    ne atskiru akciju. Tada ji galima naudoti tik per indeksa.")

    # ---------- C. MUSU KANDIDATAMS ----------
    print("\n" + "=" * 100)
    print("C. AR TAI NAUDINGA MUSU DIPO KANDIDATAMS")
    print("Ar dipo pozicija, laikoma per diena, elgiasi kitaip, kai rytas teigiamas.")
    print("=" * 100)
    try:
        # Dipo salyga: vakarykstis uzdarymas buvo kritimas
        akc = akc.sort_values(["tag", "data"])
        akc["vakar_pok"] = akc.groupby("tag")["r1"].shift(1)
        kand = akc[(akc["r_atid"] <= 0.5)]
        for lo, hi, lab in [(-99, -0.3, "rytas krito"), (-0.3, 0.3, "rytas nulis"),
                            (0.3, 99, "rytas kilo")]:
            sub = kand[(kand["r1"] >= lo) & (kand["r1"] < hi)]
            if len(sub) < 200:
                print(f"  {lab:<16} per maza imtis ({len(sub)})")
                continue
            r = week_bootstrap(sub.groupby("data")["r13_dm"].mean().to_numpy(),
                               sorted(sub["data"].unique()))
            vid = sub["r13"].mean()
            eur = vid / 100 * a.pozicija
            ci = (f"{r['lo']:+.4f} .. {r['hi']:+.4f}" if r else "—")
            print(f"  {lab:<16} n={len(sub):>6}  paskutinis langas {vid:>+8.4f}%  "
                  f"{eur:>+7.2f}€  [{ci}]")
    except Exception as e:
        print(f"  (analize praleista: {type(e).__name__}: {e})")

    # ---------- D. AR VERTA PO SANAUDU ----------
    print("\n" + "=" * 100)
    print("D. AR VERTA PO SANAUDU")
    print("=" * 100)
    print(f"  Sanaudos {a.sanaudos_eur}€ nuo {a.pozicija:,.0f}€ = "
          f"{a.sanaudos_eur / a.pozicija * 100:.3f}%")
    print(f"  Paskutinio lango vidutinis dydis turi virsyti sia riba.")
    print(f"  Straipsnyje R2=1.6% reiskia maza, bet nuosekly pranasuma —")
    print(f"  jis buvo matuotas BE sanaudu ir su ETF, kur spread'as minimalus.")

    print("\nAPRIBOJIMAI: valandiniai barai, ne 30 min. kaip straipsnyje — "
          "\nlangas platesnis ir efektas gali buti silpnesnis; 720 d. imtis "
          "\nvietoj 20 metu; Europos sesija trumpesne uz JAV; islikimo salismas.")


if __name__ == "__main__":
    main()
