#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rytinio tesinio backtestas ant 5 MIN. duomenu.

KODEL PERDARYTA
Valandine versija (backtest_rytinis_tesinys.py) pasileido, bet rezultatu
interpretuoti nebuvo galima: Yahoo valandiniams Europos akciju duomenims
daznai truksta pirmos sesijos valandos apyvartos, todel po filtru liko
MEDIANA 3 AKCIJOS DIENAI. Demeanavimas is trijų akciju beveik nieko neatima,
o "TOP-5 is tos dienos" buvo praktiskai visas tos dienos sarasas.

5 min. duomenys duoda tik ~60 dienu, bet PILNA apyvarta ir visas akcijas,
todel kryzminiai pjuviai tikri. Imtis trumpesne, bet ismatuojama.

KLAUSIMAS TAS PATS
Ar akcija, kuri pradejo kilti dienos pradzioje, kyla toliau ta pacia diena?
Ir ar tai matoma 10:00 — po pirmos valandos?

KA MATUOJAM 10:00 (arba pasirinktu laiku), nieko is ateities:
  kilimas nuo vakar uzdarymo, nakties tarpas, judesys po atidarymo,
  pirmos valandos apyvarta pries iprasta, IBS pirmoje valandoje,
  ir LIKUTIS — minus rinkos mediana, minus sektoriaus mediana.
REZULTATAI nuo ijejimo:
  iki dienos uzdarymo, iki 14:00, paskutine valanda,
  ir PIRMAS PRISILIETIMAS +1% ar -1% (tavo taisykle tiesiogiai).

Paleidimas:
    python backtest_rytinis_5m.py --valanda 10
"""

import argparse
import sys

import numpy as np
import pandas as pd

import universas as U

TZ = "Europe/Berlin"
MIN_DIENU = 20
BOOT_N = 2000


def flatten(df, sym):
    if isinstance(df.columns, pd.MultiIndex):
        if sym in df.columns.get_level_values(0):
            return df[sym].dropna(how="all")
        if sym in df.columns.get_level_values(-1):
            return df.xs(sym, axis=1, level=-1).dropna(how="all")
    return df.dropna(how="all")


def boot_ci(s, n=BOOT_N, seed=43):
    s = pd.Series(s).dropna()
    if len(s) < 10 or float(s.std()) < 1e-12:
        return np.nan, np.nan
    idx = pd.to_datetime(s.index)
    gr = [g.values for _, g in s.groupby(idx.to_period("W"))]
    gr = [g for g in gr if len(g)]
    if len(gr) < 4:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    v = [np.concatenate([gr[p] for p in rng.integers(0, len(gr), len(gr))]).mean()
         for _ in range(n)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def apib(sel, col):
    if sel is None or sel.empty:
        return 0, 0, np.nan, np.nan, np.nan
    per_d = sel.groupby("data")[col].mean()
    lo, hi = boot_ci(per_d)
    return len(sel), len(per_d), float(per_d.mean()), lo, hi


def fmt(n, d, m, lo, hi):
    if n == 0 or m != m:
        return f"{n:>7} {d:>5}   per maza imtis"
    ci = f"{lo:+.3f} .. {hi:+.3f}" if lo == lo else "—"
    return f"{n:>7} {d:>5}   {m:+.3f}%   {ci:<22}"


def paruosti(tag, sekt, d, valanda, tikslas, stop):
    """Vienos akcijos sesijos su ijejimo tasku ties nurodyta valanda."""
    d = d.copy()
    d.index = pd.to_datetime(d.index)
    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC")
    d.index = d.index.tz_convert(TZ)
    d = d[(d.index.hour >= 9) & (d.index.hour <= 17)].dropna(subset=["Close"])
    if d.empty:
        return None
    d["data"] = d.index.date
    eil = []
    dienos = sorted(d["data"].unique())
    for i in range(1, len(dienos)):
        g = d[d["data"] == dienos[i]]
        pr = d[d["data"] == dienos[i - 1]]
        if len(g) < 60 or len(pr) < 30:      # pilna sesija ~102 barai
            continue
        try:
            pr_uzd = float(pr["Close"].iloc[-1])
            atid = float(g["Open"].iloc[0])
            uzd = float(g["Close"].iloc[-1])
            # Ijejimo taskas: paskutinis baras iki nurodytos valandos
            iki = g[g.index.hour < valanda]
            po = g[g.index.hour >= valanda]
            if len(iki) < 6 or len(po) < 12 or pr_uzd <= 0:
                continue
            ent = float(iki["Close"].iloc[-1])
            hi1, lo1 = float(iki["High"].max()), float(iki["Low"].min())
            vol1 = float(iki["Volume"].sum())
            rng_ = hi1 - lo1
            # 14:00 ir paskutine valanda
            p14 = g[g.index.hour < 14]
            c14 = float(p14["Close"].iloc[-1]) if len(p14) else np.nan
            p16 = g[g.index.hour < 16]
            c16 = float(p16["Close"].iloc[-1]) if len(p16) else np.nan
            # Pirmas prisilietimas
            tp, sl = ent * (1 + tikslas / 100), ent * (1 - stop / 100)
            h, l = po["High"].values, po["Low"].values
            ti = np.where(h >= tp)[0]
            si = np.where(l <= sl)[0]
            ti = ti[0] if len(ti) else 10 ** 9
            si = si[0] if len(si) else 10 ** 9
            if si <= ti and si < 10 ** 9:
                liet = -stop
            elif ti < 10 ** 9:
                liet = tikslas
            else:
                liet = (uzd / ent - 1) * 100
            eil.append(dict(
                tag=tag, sekt=sekt, data=pd.Timestamp(dienos[i]),
                ent=ent, vol1=vol1,
                kilo=(ent / pr_uzd - 1) * 100,
                tarpas=(atid / pr_uzd - 1) * 100,
                po_atid=(ent / atid - 1) * 100,
                ibs1=((ent - lo1) / rng_) if rng_ > 0 else np.nan,
                r_uzd=(uzd / ent - 1) * 100,
                r_14=(c14 / ent - 1) * 100 if c14 == c14 else np.nan,
                r_pask=(uzd / c16 - 1) * 100 if c16 == c16 else np.nan,
                r_liet=liet))
        except Exception:
            continue
    if len(eil) < MIN_DIENU:
        return None
    e = pd.DataFrame(eil)
    e["vol1_med"] = e["vol1"].rolling(10, min_periods=5).median().shift(1)
    e["rvol"] = e["vol1"] / e["vol1_med"].replace(0, np.nan)
    return e


def main():
    ap = argparse.ArgumentParser(description="Rytinis tesinys, 5 min. duomenys")
    ap.add_argument("--valanda", type=int, default=10,
                    help="Ijejimo valanda Berlyno laiku (10 = po pirmos valandos)")
    ap.add_argument("--tikslas", type=float, default=1.0)
    ap.add_argument("--stop", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--sanaudos", type=float, default=0.028)
    ap.add_argument("--pozicija", type=float, default=18000.0)
    a = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Paleisk: pip install yfinance pandas numpy")

    syms = U.visi_tikeriai()
    sekt = U.sektoriai()
    print(f"Universas: {len(syms)}; ijejimas {a.valanda}:00 Berlyno laiku; "
          f"pirmas prisilietimas +{a.tikslas}% / -{a.stop}%")
    print("Siunciami 5 min. duomenys (60 d.)…")
    raw = yf.download(syms, period="60d", interval="5m", group_by="ticker",
                      progress=False, auto_adjust=False, threads=True)

    dalys, be_d = [], 0
    for s in syms:
        try:
            d = flatten(raw, s).dropna(subset=["Open", "High", "Low", "Close"])
            if len(d) < 1500:
                be_d += 1
                continue
            e = paruosti(s, sekt.get(s, "kita"), d, a.valanda, a.tikslas, a.stop)
            if e is None:
                be_d += 1
                continue
            dalys.append(e)
        except Exception:
            be_d += 1
    if not dalys:
        sys.exit("Nepavyko surinkti duomenu.")
    df = pd.concat(dalys, ignore_index=True).dropna(subset=["r_uzd", "kilo", "ibs1"])

    per_diena = df.groupby("data")["tag"].nunique()
    per_akcija = df.groupby("tag")["data"].nunique()
    print(f"  akciju: {df['tag'].nunique()}, eiluciu: {len(df):,}, "
          f"dienu: {df['data'].nunique()}, be duomenu: {be_d}")
    print(f"  AKCIJU VIENAI DIENAI: mediana {per_diena.median():.0f} "
          f"(nuo {per_diena.min()} iki {per_diena.max()})")
    print(f"  dienu vienai akcijai: mediana {per_akcija.median():.0f}")
    if per_diena.median() < 20:
        print("  DEMESIO: per mazai akciju dienai — pjuviai nepatikimi.")
    else:
        print("  Kryzminiai pjuviai patikimi.")

    # Likutis: minus rinkos ir sektoriaus mediana ta diena
    df["resid"] = df["kilo"] - df.groupby("data")["kilo"].transform("median")
    df["sekt_resid"] = df["kilo"] - df.groupby(["data", "sekt"])["kilo"].transform("median")
    for c in ("r_uzd", "r_14", "r_pask", "r_liet"):
        df[c + "_dm"] = df[c] - df.groupby("data")[c].transform("mean")

    # ---------- A. KVINTILIAI ----------
    print("\n" + "=" * 100)
    print("A. KVINTILIAI pagal rytini kilima (1 = labiausiai krito, 5 = labiausiai kilo)")
    print("=" * 100)
    df["kv"] = df.groupby("data")["kilo"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1
        if len(s) >= 10 else np.nan)
    print(f"{'KV':<4} {'N':>7} {'IKI UZD.':>10} {'IKI 14:00':>11} {'PASK. VAL.':>11} "
          f"{'LIETIMAS':>10} {'P(+1% pirmiau)':>15}")
    print("-" * 100)
    for kv, g in df.dropna(subset=["kv"]).groupby("kv"):
        pt = float((g["r_liet"] > a.tikslas * 0.9).mean() * 100)
        print(f"{int(kv):<4} {len(g):>7} {g['r_uzd_dm'].mean():>+9.3f}% "
              f"{g['r_14_dm'].mean():>+10.3f}% {g['r_pask_dm'].mean():>+10.3f}% "
              f"{g['r_liet_dm'].mean():>+9.3f}% {pt:>14.1f}%")

    # ---------- B. RINKINIAI ----------
    print("\n" + "=" * 100)
    print("B. TOP-5 RINKINIAI (max 2 is sektoriaus) IR KONTROLES")
    print("=" * 100)
    rng = np.random.default_rng(7)

    def topk(g, col, asc=False):
        ks = g.sort_values(col, ascending=asc)
        imti, cnt = [], {}
        for i, r in ks.iterrows():
            if cnt.get(r["sekt"], 0) >= 2:
                continue
            imti.append(i)
            cnt[r["sekt"]] = cnt.get(r["sekt"], 0) + 1
            if len(imti) >= a.top:
                break
        return g.loc[imti]

    rink = {k: [] for k in ["TOP kilimas", "TOP likutis", "TOP sekt. likutis",
                            "GAP-AND-GO", "BLOGIAUSI (kontrole)",
                            "ATSITIKTINIAI (kontrole)"]}
    for dd, g in df.groupby("data"):
        if len(g) < 15:
            continue
        rink["TOP kilimas"].append(topk(g, "kilo"))
        rink["TOP likutis"].append(topk(g, "resid"))
        rink["TOP sekt. likutis"].append(topk(g, "sekt_resid"))
        rink["BLOGIAUSI (kontrole)"].append(topk(g, "kilo", asc=True))
        rink["ATSITIKTINIAI (kontrole)"].append(
            g.sample(min(a.top, len(g)), random_state=int(rng.integers(1e9))))
        ev = g[(g["kilo"] > 0.5) & (g["rvol"] > 1.5) & (g["ibs1"] > 0.7)
               & (g["tarpas"] > 0)]
        if len(ev):
            rink["GAP-AND-GO"].append(ev)
    rink = {k: (pd.concat(v) if v else pd.DataFrame()) for k, v in rink.items()}

    for col, lab in [("r_uzd_dm", "iki uzdarymo"), ("r_liet_dm", "pirmas liet.")]:
        print(f"\n  {lab}:")
        print(f"  {'RINKINYS':<26} {'N':>7} {'DIENU':>5}   {'DEMEAN.':>9}   "
              f"{'95% INTERVALAS':<24} {'NETO EUR':>9}")
        print("  " + "-" * 96)
        for p, sel in rink.items():
            n, d_, m, lo, hi = apib(sel, col)
            eil = f"  {p:<26} {fmt(n, d_, m, lo, hi)}"
            if n and m == m:
                gr = sel.groupby("data")[col.replace("_dm", "")].mean().mean()
                eur = (gr - a.sanaudos) / 100 * a.pozicija
                eil += f" {eur:>+8.2f}€"
            print(eil)

    if len(rink["GAP-AND-GO"]):
        ev = rink["GAP-AND-GO"]
        print(f"\n  GAP-AND-GO suveikia {len(ev) / len(df) * 100:.1f}% eiluciu; "
              f"P(+{a.tikslas}% pirmiau) = "
              f"{float((ev['r_liet'] > a.tikslas * 0.9).mean() * 100):.0f}%")

    # ---------- C. PIRMO PRISILIETIMO PROPORCIJA ----------
    print("\n" + "=" * 100)
    print("C. PIRMAS PRISILIETIMAS — tavo taisykle tiesiogiai")
    print(f"Kaip daznai pirmiau ateina +{a.tikslas}%, o ne -{a.stop}%")
    print("=" * 100)
    print(f"{'RINKINYS':<26} {'N':>7} {'+ pirmiau':>11} {'- pirmiau':>11} "
          f"{'nei vienas':>12} {'VID. EUR':>10}")
    print("-" * 100)
    for p, sel in list(rink.items()) + [("VISOS eilutes", df)]:
        if sel is None or sel.empty:
            continue
        plus = float((sel["r_liet"] > a.tikslas * 0.9).mean() * 100)
        minus = float((sel["r_liet"] < -a.stop * 0.9).mean() * 100)
        eur = (sel["r_liet"].mean() - a.sanaudos) / 100 * a.pozicija
        print(f"{p:<26} {len(sel):>7} {plus:>10.1f}% {minus:>10.1f}% "
              f"{100 - plus - minus:>11.1f}% {eur:>+9.2f}€")

    print("\n" + "=" * 100)
    print("KRITERIJAI (uzrasyti pries paleidziant)")
    print("=" * 100)
    print(f"  - rytinis tesinys egzistuoja, jei KV5 > KV1 ir TOP intervalo")
    print(f"    apacia > 0, o BLOGIAUSI blogiau uz TOP;")
    print(f"  - prekybai tinka tik jei 'pirmas lietimas' po {a.sanaudos}% sanaudu")
    print(f"    duoda teigiama EUR ir intervalo apacia > 0;")
    print(f"  - jei KV5 < KV1 — ryte kylancios diena atsitraukia.")
    print("\nAPRIBOJIMAI: tik 60 d. istorijos (5 min. duomenu riba); islikimo "
          "\nsalismas; ±1% lietimas is 5 min. H/L — jei abu tame paciame bare, "
          "\nlaikom stop (konservatyvu); vienas laikotarpis, ne dvi puses.")


if __name__ == "__main__":
    main()
