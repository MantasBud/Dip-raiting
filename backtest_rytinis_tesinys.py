#!/usr/bin/env python3
"""
backtest_rytinis_tesinys.py — ar akcija, kuri PRADĖJO kilti dienos pradžioje, kyla toliau
tą pačią dieną? Ir ar galima tai pamatyti 10:00 (po pirmos valandos)?

Klausimas iš praktikos: nupirkti 10:00 (arba 11:00) ir parduoti tą pačią dieną, pakilus ~1 %.

Kas matuojama (60m barai, ~730 sesijų, EUR universas iš bendra_60m.py):
  Sprendimo taškas: pirmo valandinio baro uždarymas (10:00) — variantas ir 11:00.
  Požymiai tuo momentu (nieko iš ateities):
    r_nuo_uzd   = C_10:00 / vakar_uzdarymas − 1     (Gao–Han–Li–Zhou "pirmos pusvalandės" analogas)
    tarpas      = O_9:00 / vakar_uzdarymas − 1
    r_1h        = C_10:00 / O_9:00 − 1               (judesys jau po atidarymo)
    rvol_1h     = pirmos valandos apyvarta / mediana(20 sesijų pirmos valandos)
    ibs_1h      = kur uždarė pirmą valandą jos diapazone
    resid       = r_nuo_uzd − universo mediana tą dieną (akcijos SAVAS judesys, ne rinkos)
    sekt_resid  = r_nuo_uzd − sektoriaus mediana tą dieną
  Rezultatai nuo įėjimo C_10:00 (demeanuoti pagal dieną per visą universą):
    r_uzd       = iki dienos uždarymo
    r_14        = iki 14:00
    r_pask      = paskutinė valanda (16:00→17:30), t. y. Gao stiliaus "paskutinis pusvalandis"
    pirmas_lietimas: +1 % ar −1 % pasiekiamas pirmiau (tavo "šiek tiek pakilus" taisyklė),
                     pnl = +1 / −1 / uždarymas
  Rinkiniai kiekvieną dieną: TOP-5 pagal r_nuo_uzd, pagal resid, pagal sekt_resid, jungtinis
  "gap-and-go" įvykis (r_nuo_uzd > 0,5 %, rvol > 1,5, ibs_1h > 0,7, tarpas > 0), kontrolės
  (blogiausi 5, atsitiktiniai 5), ir rinkos lygio Gao testas su EXSA.DE.
  Statistika: dienų vidurkiai, savaičių blokų bootstrap, dvi laiko pusės, metų lentelė.

Slenksčiai fiksuoti. Nederinti pagal rezultatą.
"""

import argparse
import sys

import numpy as np
import pandas as pd

from bendra_60m import UNIVERSAS, TZ, siusti_60m

BOOT_N = 1000
SEED = 11
MAX_SEKT = 2
INDEKSAS = "EXSA.DE"


# ---------------------------------------------------------------------------
def paruosti(tag, df, taskas, tikslas_pct, stop_pct):
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(TZ)
    df = df[(df.index.hour >= 9) & (df.index.hour <= 17)].dropna(subset=["Close"])
    df["data"] = df.index.date
    df["k"] = df.groupby("data").cumcount()
    pilnos = df.groupby("data")["k"].max()
    df = df[df["data"].isin(pilnos[pilnos >= 7].index)].copy()
    if df["data"].nunique() < 100:
        return None

    g = df.groupby("data")
    sess = pd.DataFrame({
        "open": g["Open"].first(), "close": g["Close"].last(),
        "vol": g["Volume"].sum(), "n": g["k"].max() + 1,
    })
    sess["prev_close"] = sess["close"].shift(1)
    sess["turnover"] = (sess["close"] * sess["vol"]).rolling(20).median().shift(1)

    # pirmas (taskas+1) valandų blokas
    pirmi = df[df["k"] <= taskas]
    gp = pirmi.groupby("data")
    e = pd.DataFrame({
        "ent": gp["Close"].last(), "hi1": gp["High"].max(), "lo1": gp["Low"].min(),
        "vol1": gp["Volume"].sum(),
    })
    e = e.join(sess[["open", "close", "prev_close", "turnover", "n"]])
    e["vol1_med"] = e["vol1"].rolling(20).median().shift(1)
    e = e.dropna(subset=["prev_close", "vol1_med", "turnover"])
    e = e[e["vol1_med"] > 0]

    e["r_nuo_uzd"] = e["ent"] / e["prev_close"] - 1
    e["tarpas"] = e["open"] / e["prev_close"] - 1
    e["r_1h"] = e["ent"] / e["open"] - 1
    e["rvol"] = e["vol1"] / e["vol1_med"]
    rng = (e["hi1"] - e["lo1"]).replace(0, np.nan)
    e["ibs1"] = (e["ent"] - e["lo1"]) / rng
    e["vakar_ret"] = e["prev_close"] / sess["close"].shift(2).reindex(e.index) - 1

    # rezultatai po įėjimo
    e["r_uzd"] = e["close"] / e["ent"] - 1
    c14 = df[df["k"] == 4].groupby("data")["Close"].last()      # 13:00–14:00 baro uždarymas
    e["r_14"] = c14.reindex(e.index) / e["ent"] - 1
    c16 = df[df["k"] == 6].groupby("data")["Close"].last()      # 15:00–16:00 uždarymas
    e["r_pask"] = e["close"] / c16.reindex(e.index) - 1         # paskutinė ~1,5 val.

    # pirmas lietimas +tikslas / −stop per barus po įėjimo
    po = df[df["k"] > taskas]
    rez = {}
    for d, grp in po.groupby("data"):
        if d not in e.index:
            continue
        ent = e.at[d, "ent"]
        tp, sl = ent * (1 + tikslas_pct / 100), ent * (1 - stop_pct / 100)
        h, l = grp["High"].values, grp["Low"].values
        ti = np.where(h >= tp)[0]
        si = np.where(l <= sl)[0]
        ti = ti[0] if len(ti) else 10 ** 9
        si = si[0] if len(si) else 10 ** 9
        if si <= ti and si < 10 ** 9:
            rez[d] = -stop_pct / 100
        elif ti < 10 ** 9:
            rez[d] = tikslas_pct / 100
        else:
            rez[d] = e.at[d, "r_uzd"]
    e["r_lietimas"] = pd.Series(rez)
    e["tag"] = tag
    e["sekt"] = UNIVERSAS.get(tag, "kita")
    e = e.reset_index().rename(columns={"index": "data"})
    return e


# ---------------------------------------------------------------------------
def boot_ci(s: pd.Series, n=BOOT_N, seed=SEED):
    s = s.dropna()
    if len(s) < 10:
        return np.nan, np.nan
    s.index = pd.to_datetime(s.index)
    grupes = [g.values for _, g in s.groupby(s.index.to_period("W"))]
    rng = np.random.default_rng(seed)
    vid = []
    for _ in range(n):
        pick = rng.integers(0, len(grupes), len(grupes))
        vid.append(np.concatenate([grupes[p] for p in pick]).mean())
    return np.percentile(vid, 2.5), np.percentile(vid, 97.5)


def apib(sel, col):
    if sel is None or sel.empty:
        return 0, 0, np.nan, np.nan, np.nan
    per_d = sel.groupby("data")[col].mean()
    lo, hi = boot_ci(per_d)
    return len(sel), len(per_d), per_d.mean(), lo, hi


def fmt(n, d, m, lo, hi):
    if n == 0 or np.isnan(m):
        return f"{'—':>7} {'—':>5}   per maža imtis"
    return f"{n:>7} {d:>5}   {m*100:+.3f}%   {lo*100:+.3f} .. {hi*100:+.3f}"


def atrinkti(df, top, rng):
    out = {p: [] for p in ["TOP r_nuo_uzd", "TOP resid (vs rinka)", "TOP sekt_resid",
                           "GAP-AND-GO ivykis", "BLOGIAUSI (kontrole)", "ATSITIKTINIAI (kontrole)"]}
    for data, g in df.groupby("data", sort=False):
        if len(g) < 10:
            continue
        def topk(col, asc=False):
            ks = g.sort_values(col, ascending=asc)
            imti, cnt = [], {}
            for i, r in ks.iterrows():
                if cnt.get(r["sekt"], 0) >= MAX_SEKT:
                    continue
                imti.append(i); cnt[r["sekt"]] = cnt.get(r["sekt"], 0) + 1
                if len(imti) >= top:
                    break
            return g.loc[imti]
        out["TOP r_nuo_uzd"].append(topk("r_nuo_uzd"))
        out["TOP resid (vs rinka)"].append(topk("resid"))
        out["TOP sekt_resid"].append(topk("sekt_resid"))
        out["BLOGIAUSI (kontrole)"].append(topk("r_nuo_uzd", asc=True))
        out["ATSITIKTINIAI (kontrole)"].append(g.sample(min(top, len(g)), random_state=int(rng.integers(1e9))))
        ev = g[(g["r_nuo_uzd"] > 0.005) & (g["rvol"] > 1.5) & (g["ibs1"] > 0.7) & (g["tarpas"] > 0)]
        if len(ev):
            out["GAP-AND-GO ivykis"].append(ev)
    return {p: (pd.concat(v) if v else pd.DataFrame()) for p, v in out.items()}


def lentele(rink, pav, col, sanaudos, pozicija=18000.0):
    # PATAISYTA: sanaudos buvo perduodamos, bet funkcijos viduje nenaudojamos.
    # Kriterijai reikalauja vertinti PO sanaudu, todel dabar rodomas ir neto
    # rezultatas eurais. 0.028% = 5 EUR nuo 18 000 (vartotojo tikri kastai).
    print(f"\n{'RINKINYS':<26} {'LANGAS':<12} {'N':>7} {'DIENU':>5}   {'DEMEAN.':>9}   "
          f"{'95% INTERVALAS':<24} {'NETO EUR':>9}")
    print("-" * 104)
    for p, sel in rink.items():
        n, d, m, lo, hi = apib(sel, col)
        eil = f"{p:<26} {pav:<12} {fmt(n, d, m, lo, hi)}"
        if n and not np.isnan(m):
            # neto: NEdemeanuota graza minus sanaudos
            gryna = sel.groupby("data")[col.replace("_dm", "")].mean().mean()
            eur = (gryna - sanaudos / 100) * pozicija
            eil = f"{eil:<92} {eur:>+8.2f}€"
        print(eil)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taskas", type=int, default=0, help="0 = sprendimas 10:00, 1 = 11:00")
    ap.add_argument("--tikslas", type=float, default=1.0)
    ap.add_argument("--stop", type=float, default=1.0)
    ap.add_argument("--min-apyvarta", type=float, default=20.0, help="mln EUR")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--sanaudos", type=float, default=0.028,
                    help="% vienam ciklui; 0.028 = 5 EUR nuo 18 000")
    a = ap.parse_args()

    tickers = sorted(UNIVERSAS)
    print(f"Universas: {len(tickers)} EUR akcijų; sprendimo taškas {10 + a.taskas}:00; "
          f"pirmas lietimas +{a.tikslas}% / −{a.stop}%")
    duom = siusti_60m(tickers + [INDEKSAS])
    print(f"  gauta: {len(duom)}")

    dalys, atmesta = [], 0
    for tag, df in duom.items():
        if tag == INDEKSAS:
            continue
        e = paruosti(tag, df, a.taskas, a.tikslas, a.stop)
        if e is None:
            continue
        if e["turnover"].median() < a.min_apyvarta * 1e6:
            atmesta += 1
            continue
        dalys.append(e)
    df = pd.concat(dalys, ignore_index=True).dropna(subset=["r_uzd", "r_nuo_uzd", "ibs1"])
    print(f"  po likvidumo filtro: {df['tag'].nunique()} akcijų (atmesta {atmesta}); "
          f"eilučių {len(df):,}; dienų {df['data'].nunique()}")

    # savas judesys: minus rinkos / sektoriaus mediana tą dieną
    df["resid"] = df["r_nuo_uzd"] - df.groupby("data")["r_nuo_uzd"].transform("median")
    df["sekt_resid"] = df["r_nuo_uzd"] - df.groupby(["data", "sekt"])["r_nuo_uzd"].transform("median")
    for c in ["r_uzd", "r_14", "r_pask", "r_lietimas"]:
        df[c + "_dm"] = df[c] - df.groupby("data")[c].transform("mean")

    # 1) Tiesinė priklausomybė: ar pirmos valandos judesys apskritai ką nors sako
    print("\n" + "=" * 92 + "\nKORELIACIJA (Spearman) tarp požymio 10:00 ir likusios dienos rezultato (demeanuota)\n" + "=" * 92)
    for poz in ["r_nuo_uzd", "resid", "sekt_resid", "r_1h", "tarpas", "rvol", "ibs1", "vakar_ret"]:
        eil = f"{poz:<12}"
        for col in ["r_uzd_dm", "r_14_dm", "r_pask_dm"]:
            per_d = df.groupby("data").apply(lambda g: g[poz].corr(g[col], method="spearman"))
            lo, hi = boot_ci(per_d)
            eil += f"   {col:<10} {per_d.mean():+.3f} [{lo:+.3f}..{hi:+.3f}]"
        print(eil)
    print("  (dienos vidutinė kryžminė koreliacija; |0.02| yra praktiškai nulis, |0.05| jau kažkas)")

    # 2) Kvintiliai pagal r_nuo_uzd — ar ryšys monotoniškas
    print("\n" + "=" * 92 + "\nKVINTILIAI pagal r_nuo_uzd (1 = labiausiai krito 10:00, 5 = labiausiai kilo)\n" + "=" * 92)
    df["kv"] = df.groupby("data")["r_nuo_uzd"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
    print(f"{'KV':<4} {'N':>7}   {'r_uzd':>9}   {'r_14':>9}   {'r_pask':>9}   {'lietimas':>9}   {'P(+1% pirmiau)':>14}")
    for kv, g in df.groupby("kv"):
        pt = (g["r_lietimas"] > 0.009).mean() * 100
        print(f"{int(kv):<4} {len(g):>7}   {g['r_uzd_dm'].mean()*100:+.3f}%   {g['r_14_dm'].mean()*100:+.3f}%"
              f"   {g['r_pask_dm'].mean()*100:+.3f}%   {g['r_lietimas_dm'].mean()*100:+.3f}%   {pt:>13.1f}%")

    # 3) Rinkiniai, dvi pusės
    rng = np.random.default_rng(SEED)
    vid = df["data"].sort_values().iloc[len(df) // 2]
    puses = {"VISA IMTIS": df, "1-oji PUSĖ": df[df["data"] <= vid], "2-oji PUSĖ (nematyta)": df[df["data"] > vid]}
    print(f"\nPadalijimas: 1-a pusė iki {vid}")
    for pav, d in puses.items():
        print("\n" + "=" * 92 + f"\n{pav}\n" + "=" * 92)
        rink = atrinkti(d, a.top, rng)
        lentele(rink, "iki uzdarymo", "r_uzd_dm", a.sanaudos)
        lentele(rink, "iki 14:00", "r_14_dm", a.sanaudos)
        lentele(rink, "pirmas liet.", "r_lietimas_dm", a.sanaudos)
        ev = rink["GAP-AND-GO ivykis"]
        if len(ev):
            print(f"  GAP-AND-GO: suveikia {len(ev)/len(d)*100:.1f}% eilučių; "
                  f"P(+{a.tikslas}% pirmiau nei −{a.stop}%) = {(ev['r_lietimas'] > 0.009).mean()*100:.0f}%")

    # 4) Metų lentelė
    print("\n" + "=" * 92 + "\nPAGAL METUS: TOP r_nuo_uzd, demeanuota iki uždarymo\n" + "=" * 92)
    rink = atrinkti(df, a.top, rng)
    s = rink["TOP r_nuo_uzd"].copy()
    s["metai"] = pd.to_datetime(s["data"]).dt.year
    for m, g in s.groupby("metai"):
        print(f"  {m}: {g['r_uzd_dm'].mean()*100:+.3f}% ({len(g)})")

    # 5) Rinkos lygio Gao testas (EXSA.DE): pirmos valandos ženklas → paskutinės valandos grąža
    if INDEKSAS in duom:
        try:
            ix = duom[INDEKSAS].copy()
            ix.index = pd.to_datetime(ix.index, utc=True).tz_convert(TZ)
            ix = ix[(ix.index.hour >= 9) & (ix.index.hour <= 17)].dropna(subset=["Close"])
            ix["data"] = ix.index.date
            ix["k"] = ix.groupby("data").cumcount()
            gi = ix.groupby("data")
            t = pd.DataFrame({"c0": ix[ix["k"] == 0].groupby("data")["Close"].last(),
                              "c6": ix[ix["k"] == 6].groupby("data")["Close"].last(),
                              "cl": gi["Close"].last()})
            t["prev"] = t["cl"].shift(1)
            t = t.dropna()
            t["pirma"] = t["c0"] / t["prev"] - 1
            t["pask"] = t["cl"] / t["c6"] - 1
            print("\n" + "=" * 92 + "\nRINKOS LYGIS (EXSA.DE): paskutinė valanda pagal pirmos valandos ženklą\n" + "=" * 92)
            for zenk, g in t.groupby(t["pirma"] > 0):
                lo, hi = boot_ci(g["pask"])
                print(f"  pirma valanda {'teigiama' if zenk else 'neigiama'}: n={len(g)}  paskutinė val. "
                      f"{g['pask'].mean()*100:+.3f}% [{lo*100:+.3f}..{hi*100:+.3f}]")
            print(f"  koreliacija pirma→paskutinė: {t['pirma'].corr(t['pask']):+.3f}")
        except Exception as ex:
            print(f"  (indekso testas praleistas: {ex})")

    print("\nKriterijai (užrašyti prieš paleidžiant):")
    print("  - rytinis tęsinys egzistuoja, jei kvintilis 5 > kvintilis 1 (r_uzd_dm) ir TOP r_nuo_uzd")
    print("    ar TOP resid intervalo apačia > 0 abiejose pusėse, o BLOGIAUSI blogiau už TOP;")
    print("  - prekybai tinka tik jei 'pirmas lietimas' po 0.07 % sąnaudų > 0 su apačia > 0;")
    print("  - jei kvintilis 5 < kvintilis 1 — ryte kylančios akcijos dieną ATSITRAUKIA, ir tavo")
    print("    idėja veikia priešingai (pirkti reikėtų kvintilį 1, t. y. dip'ą, kaip ir dabar).")
    print("APRIBOJIMAI: išlikimo šališkumas; 60m barai (~730 d.); ±1 % lietimas iš 60m H/L,")
    print("  todėl 'pirmas lietimas' optimistinis (abu tame pačiame bare → stop, bet nematome eiliškumo).")


if __name__ == "__main__":
    main()
