#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hipoteziu backtestas — viskas, kas uzfiksuota KITAS_BACKTESTAS.md.

Paleidimas:
    pip install yfinance pandas numpy
    python backtest_hipotezes.py --rinka eu
    python backtest_hipotezes.py --rinka us      # nepriklausomas patikrinimas

KAS TIKRINAMA (uzrasyta pries paleidziant, slenksciai nederinami)

A. SVYRAVIMO SALYGA (Nagel 2012, RFS 25(7))
   Trumpalaikio grizimo grazа = likvidumo teikimo atlygis; prognozuojama VIX.
   Hipoteze: musu signalu pranasumas auga kartu su svyravimu.
   Matuojama IR demeanuotai (atmeta rinkos krypti), IR eurais (ka realiai gautum).

B. RSI(2) vietoj RSI(14)
   Anksciau tikrinom RSI(14) — nepasitvirtino. Connors argumentas: RSI(14)
   kraštutiniu reiksmiu pasiekia retai. Tikrinam RSI(2) < 10 ir < 5.

C. SALYGINIS ISEJIMAS vietoj fiksuoto tikslo
   Dokumentuotos IBS strategijos iseina, kai IBS pakyla virs 0.8, o ne pasiekus
   procenta. Musu matavimuose sio isejimo tipo NIEKAD netikrinom.
   Lyginami: IBS>0.8 | uzdarymas virs 5 d. vidurkio | RSI(2)>70 | fiksuotas 2%.

D. TRENDO FILTRAS su konkrecia riba
   Ne "virs SMA200", o "virs SMA200 bet ne daugiau kaip 5% virs".

E. ADX ir OBV — TIK jei koreliacija su patvirtintais signalais < 0.6

F. SLANKIOJO LANGO patikra
   Ar signalai turi periodu, kuriais veikia, ir ar praeitas langas prognozuoja kita.

G. ZALIAVU SEKTORIAI
   Literatura: IBS silpnesnis zaliavu veikiamose rinkose. Tikrinam atskirai.
"""

import argparse
import sys

import numpy as np
import pandas as pd

import universas as U

MIN_IVYKIU = 150
MIN_DIENU = 40
ZALIAVU_SEKT = {"Medziagos", "Energetika"}


# ----------------------------- STATISTIKA -----------------------------

def week_bootstrap(vals, dienos, n=2000, seed=7):
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
    p_pos = float((boot > 0).mean())
    return dict(mean=float(np.mean(vals)), lo=float(lo), hi=float(hi),
                p_two=float(2 * min(p_pos, 1 - p_pos)), dienos=len(vals))


def ivertink(sub, col):
    s = sub.dropna(subset=[col])
    if len(s) < MIN_IVYKIU:
        return None, len(s)
    pd_ = s.groupby("data")[col].mean()
    if len(pd_) < MIN_DIENU:
        return None, len(s)
    r = week_bootstrap(pd_.to_numpy(), pd_.index)
    if r:
        r["n"] = len(s)
    return r, len(s)


def bh(pv, alpha=0.05):
    pv = np.asarray(pv)
    m = len(pv)
    order = np.argsort(pv)
    ok = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order, start=1):
        if pv[idx] <= alpha * rank / m:
            ok[order[:rank]] = True
    return ok


def flatten(df, sym):
    if isinstance(df.columns, pd.MultiIndex):
        if sym in df.columns.get_level_values(0):
            return df[sym].dropna(how="all")
        if sym in df.columns.get_level_values(-1):
            return df.xs(sym, axis=1, level=-1).dropna(how="all")
    return df.dropna(how="all")


# ----------------------------- RODIKLIAI -----------------------------

def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def adx(h, l, c, n=14):
    up, dn = h.diff(), -l.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(plus, index=c.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus, index=c.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def paruosk(d):
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
    pc = c.shift(1)
    t = pd.DataFrame(index=d.index)
    t["C"], t["O"], t["H"], t["L"] = c, d["Open"], h, l
    t["O1"] = d["Open"].shift(-1)
    t["ret"] = (c / pc - 1) * 100
    rng = (h - l).replace(0, np.nan)
    t["ibs"] = (c - l) / rng

    t["sma5"] = c.rolling(5).mean()
    t["sma20"] = c.rolling(20).mean()
    t["sma200"] = c.rolling(200).mean()
    t["z20"] = (c - t["sma20"]) / c.rolling(20).std()
    t["virs200"] = c / t["sma200"] - 1                 # santykinis nuotolis

    t["rsi2"] = rsi(c, 2)
    t["rsi14"] = rsi(c, 14)
    t["adx"] = adx(h, l, c, 14)

    # OBV nuolydis (20 d.)
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    t["obv_slope"] = (obv - obv.shift(20)) / v.rolling(20).mean().replace(0, np.nan)

    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    t["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / c * 100
    t["apyv"] = (c * v).rolling(20).median()

    # Busimi barai — salyginiams isejimams
    for n in range(1, 11):
        t[f"H{n}"], t[f"L{n}"], t[f"C{n}"] = h.shift(-n), l.shift(-n), c.shift(-n)
        t[f"ibs{n}"] = ((c.shift(-n) - l.shift(-n))
                        / (h.shift(-n) - l.shift(-n)).replace(0, np.nan))
        t[f"sma5_{n}"] = t["sma5"].shift(-n)
        t[f"rsi2_{n}"] = t["rsi2"].shift(-n)
    return t


# ----------------------------- ISEJIMAI -----------------------------

def salyginis_isejimas(t, budas, max_dienu=10, stop_pct=None):
    """Grazina grazа % nuo kitos dienos atidarymo iki salygos ivykdymo."""
    ijej = t["O1"]
    baigta = pd.Series(False, index=t.index)
    graza = pd.Series(np.nan, index=t.index)

    for n in range(1, max_dienu + 1):
        if stop_pct is not None:
            sk = ijej * (1 - stop_pct / 100)
            hit = (~baigta) & (t[f"L{n}"] <= sk)
            graza = graza.where(~hit, -stop_pct)
            baigta = baigta | hit

        if budas == "ibs08":
            sal = t[f"ibs{n}"] >= 0.8
        elif budas == "sma5":
            sal = t[f"C{n}"] > t[f"sma5_{n}"]
        elif budas == "rsi70":
            sal = t[f"rsi2_{n}"] >= 70
        else:
            sal = pd.Series(False, index=t.index)

        hit = (~baigta) & sal.fillna(False)
        graza = graza.where(~hit, (t[f"C{n}"] / ijej - 1) * 100)
        baigta = baigta | hit

    graza = graza.where(baigta, (t[f"C{max_dienu}"] / ijej - 1) * 100)
    return graza


def fiksuotas_isejimas(t, tikslas, stop_pct, max_dienu=3):
    ijej = t["O1"]
    tk, sk = ijej * (1 + tikslas / 100), ijej * (1 - stop_pct / 100)
    baigta = pd.Series(False, index=t.index)
    graza = pd.Series(np.nan, index=t.index)
    for n in range(1, max_dienu + 1):
        s_hit = (~baigta) & (t[f"L{n}"] <= sk)
        t_hit = (~baigta) & (~s_hit) & (t[f"H{n}"] >= tk)
        graza = graza.where(~s_hit, -stop_pct)
        graza = graza.where(~t_hit, tikslas)
        baigta = baigta | s_hit | t_hit
    return graza.where(baigta, (t[f"C{max_dienu}"] / ijej - 1) * 100)


def main():
    ap = argparse.ArgumentParser(description="Hipoteziu backtestas")
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
    idx_sym = "SPY" if a.rinka == "us" else U.INDEKSAS
    vix_sym = "^VIX" if a.rinka == "us" else "^V2TX"
    sekt = U.sektoriai(uni)
    syms = U.visi_tikeriai(uni)

    print(f"{'JAV' if a.rinka == 'us' else 'EUROPOS'} universas: {len(syms)} akciju, "
          f"{a.metai} metu\n")
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
    print(f"  duomenu gauta: {df['tag'].nunique()}; be duomenu: {be_d}")

    # --- Svyravimo matas ---
    df["_d"] = pd.to_datetime(df["data"]).dt.date
    svyr = None
    for cand in (vix_sym, idx_sym):
        try:
            v = flatten(yf.download(cand, period=f"{a.metai}y", interval="1d",
                                    progress=False, auto_adjust=False), cand)
            v = v.dropna(subset=["Close"])
            if len(v) > 200:
                if cand == vix_sym:
                    s = pd.Series(v["Close"].values,
                                  index=pd.Index([i.date() for i in v.index]))
                    print(f"  svyravimo matas: {cand}")
                else:
                    r = v["Close"].pct_change()
                    s = pd.Series((r.rolling(20).std() * np.sqrt(252) * 100).values,
                                  index=pd.Index([i.date() for i in v.index]))
                    print(f"  svyravimo matas: {cand} realizuotas 20 d. svyravimas")
                svyr = s
                break
        except Exception:
            continue
    df["svyr"] = df["_d"].map(svyr) if svyr is not None else np.nan

    # --- Ijejimo salyga (bendra visoms hipotezems) ---
    df = df[(df["ret"] <= -0.8) & (df["ret"] >= -6.0)
            & (df["apyv"] >= 5e6) & df["O1"].notna()].copy()
    print(f"  ijejimo tasku: {len(df):,}")

    # Grazos kuriamos IS KARTO — anksciau jos buvo kuriamos tik svyravimo bloke,
    # todel visi kiti skyriai likdavo be duomenu ir rodydavo "per maza imtis".
    df["r3"] = (df["C3"] / df["O1"] - 1) * 100
    df["r3_dm"] = df["r3"] - df.groupby("data")["r3"].transform("mean")
    df = df.dropna(subset=["r3_dm"])
    print(f"  su rezultatu: {len(df):,}; IBS<0.25: {int((df['ibs'] < 0.25).sum()):,}")

    riba = pd.to_datetime(df["data"]).quantile(0.5)
    p1 = pd.to_datetime(df["data"]) <= riba
    p2 = ~p1

    # Kandidato apibrezimas: apatinis IBS kvintilis pagal 1-OS PUSES pasiskirstyma.
    # Fiksuota 0.25 riba priklauso nuo to, kaip placiai svyruoja konkretus rinkinys;
    # percentilis duoda ta pati retuma bet kokiuose duomenyse ir nezvilgcioja i ateiti.
    IBS_RIBA = float(df.loc[p1, "ibs"].quantile(0.20))
    print(f"  kandidato riba (IBS apatinis kvintilis is 1-os puses): {IBS_RIBA:.3f}; "
          f"tokiu atveju: {int((df['ibs'] <= IBS_RIBA).sum()):,}")
    print(f"  padalijimas: 1-a puse iki {riba:%Y-%m-%d}\n")

    # ---------- E. KORELIACIJOS (pirma, nes gali uzdaryti ADX/OBV) ----------
    print("=" * 100)
    print("E. KORELIACIJOS — ar ADX/OBV yra tas pats signalas kitu vardu")
    print("Jei |r| > 0.6 su patvirtintu signalu, toliau netikrinama.")
    print("=" * 100)
    baz = ["z20", "ibs", "rsi2"]
    nauji = ["adx", "obv_slope", "rsi14", "atr"]
    print(f"{'':<14}" + "".join(f"{b:>12}" for b in baz))
    tirti = []
    for nn in nauji:
        eil = f"{nn:<14}"
        maxr = 0.0
        for b in baz:
            r = df[[nn, b]].corr().iloc[0, 1]
            eil += f"{r:>12.3f}"
            maxr = max(maxr, abs(r) if r == r else 0)
        blok = maxr > 0.6
        print(eil + ("   -> BLOKUOTA (dubliuoja)" if blok else "   -> tikrinama"))
        if not blok:
            tirti.append(nn)

    # ---------- A. SVYRAVIMO SALYGA ----------
    print("\n" + "=" * 100)
    print("A. SVYRAVIMO SALYGA (Nagel 2012) — ar pranasumas auga kartu su svyravimu")
    print("=" * 100)
    if df["svyr"].notna().sum() > 1000:
        q = df.loc[p1, "svyr"].quantile([0.33, 0.67])   # ribos IS 1-OS PUSES
        df["svyr_lyg"] = pd.cut(df["svyr"], [-1e9, q.iloc[0], q.iloc[1], 1e9],
                                labels=["zemas", "vidutinis", "aukstas"])
        print(f"{'SIGNALAS':<16} {'SVYRAVIMAS':<12} {'N':>7} {'DEMEAN.':>10} "
              f"{'95% INTERVALAS':>22} {'EUR (nedem.)':>13}")
        print("-" * 100)
        for sig, hb in [("z20", False), ("ibs", False), ("rsi2", False),
                        ("rsi14", False), ("atr", True)]:
            for lyg in ["zemas", "vidutinis", "aukstas"]:
                sub = df[df["svyr_lyg"] == lyg]
                if len(sub) < MIN_IVYKIU:
                    continue
                riba_s = sub[sig].quantile(0.2 if not hb else 0.8)
                sel = sub[sub[sig] <= riba_s] if not hb else sub[sub[sig] >= riba_s]
                r, n = ivertink(sel, "r3_dm")
                eur = sel["r3"].mean() / 100 * a.pozicija - a.sanaudos_eur
                if r is None:
                    print(f"{sig:<16} {lyg:<12} {n:>7}   per maza imtis")
                else:
                    ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
                    print(f"{sig:<16} {lyg:<12} {r['n']:>7} {r['mean']:>+9.3f}% "
                          f"{ci:>22} {eur:>+12.2f}€")
            print()
    else:
        print("  svyravimo duomenu nepakako")

    # ---------- B. RSI(2) vs RSI(14) ----------
    print("=" * 100)
    print("B. RSI(2) vs RSI(14) — ar parametras, o ne rodiklis, buvo problema")
    print("=" * 100)
    print(f"{'SALYGA':<26} {'N':>7} {'DEMEAN. r3':>12} {'95% INTERVALAS':>22}")
    print("-" * 100)
    rez_b = []
    for lab, kauke in [("RSI(2) < 5", df["rsi2"] < 5),
                       ("RSI(2) < 10", df["rsi2"] < 10),
                       ("RSI(2) < 20", df["rsi2"] < 20),
                       ("RSI(14) < 30", df["rsi14"] < 30),
                       ("RSI(14) < 40", df["rsi14"] < 40)]:
        r, n = ivertink(df[kauke], "r3_dm")
        if r is None:
            print(f"{lab:<26} {n:>7}   per maza imtis")
        else:
            ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
            print(f"{lab:<26} {r['n']:>7} {r['mean']:>+11.3f}% {ci:>22}")
            rez_b.append((lab, kauke, r))

    # ---------- D. TRENDO FILTRAS ----------
    print("\n" + "=" * 100)
    print("D. TRENDO FILTRAS — 'virs SMA200, bet ne daugiau kaip 5% virs'")
    print("=" * 100)
    print(f"{'SALYGA':<30} {'N':>7} {'DEMEAN. r3':>12} {'95% INTERVALAS':>22}")
    print("-" * 100)
    # "be filtro" eilutes nera: demeanavus pagal diena viso rinkinio vidurkis yra
    # lygiai nulis, todel ji nieko nepasakytu. Lyginam tik tarpusavyje.
    for lab, kauke in [("virs SMA200", df["virs200"] > 0),
                       ("0-5% virs SMA200", (df["virs200"] > 0) & (df["virs200"] <= 0.05)),
                       ("5-15% virs SMA200", (df["virs200"] > 0.05) & (df["virs200"] <= 0.15)),
                       ("zemiau SMA200", df["virs200"] <= 0)]:
        r, n = ivertink(df[kauke], "r3_dm")
        if r is None:
            print(f"{lab:<30} {n:>7}   per maza imtis")
        else:
            ci = f"{r['lo']:+.3f} .. {r['hi']:+.3f}"
            print(f"{lab:<30} {r['n']:>7} {r['mean']:>+11.3f}% {ci:>22}")

    # ---------- C. ISEJIMO BUDAI ----------
    print("\n" + "=" * 100)
    print("C. ISEJIMO BUDAI — salyginis vs fiksuotas (EURAIS, sanaudos atimtos)")
    print(f"Ijejimas: IBS <= {IBS_RIBA:.3f} (apatinis kvintilis, patvirtintas signalas)")
    print("=" * 100)
    kand = df[df["ibs"] <= IBS_RIBA].copy()
    print(f"{'ISEJIMAS':<34} {'N':>7} {'VID. EUR':>11} {'TRUKME':>8} {'>0 dalis':>10}")
    print("-" * 100)
    isejimai = [
        ("IBS > 0.8 (iki 10 d.)", lambda t: salyginis_isejimas(t, "ibs08")),
        ("IBS > 0.8 + stop 3%", lambda t: salyginis_isejimas(t, "ibs08", stop_pct=3.0)),
        ("uzdarymas virs SMA5", lambda t: salyginis_isejimas(t, "sma5")),
        ("RSI(2) > 70", lambda t: salyginis_isejimas(t, "rsi70")),
        ("fiksuotas 2% / stop 1.5%", lambda t: fiksuotas_isejimas(t, 2.0, 1.5)),
        ("fiksuotas 3% / stop 2%", lambda t: fiksuotas_isejimas(t, 3.0, 2.0)),
    ]
    rez_c = []
    for lab, f in isejimai:
        g = f(kand)
        v = g.dropna()
        if len(v) < MIN_IVYKIU:
            print(f"{lab:<34} {len(v):>7}   per maza imtis")
            continue
        eur = v / 100 * a.pozicija - a.sanaudos_eur
        kand["_g"] = g
        pd_ = kand.dropna(subset=["_g"]).groupby("data")["_g"].mean()
        b = week_bootstrap(pd_.to_numpy(), pd_.index)
        print(f"{lab:<34} {len(v):>7} {eur.mean():>+10.2f}€ {'—':>8} "
              f"{float((v > 0).mean() * 100):>9.0f}%")
        if b:
            rez_c.append((lab, f, b))

    # ---------- G. ZALIAVU SEKTORIAI ----------
    print("\n" + "=" * 100)
    print("G. ZALIAVU SEKTORIAI — literatura sako, kad IBS ten silpnesnis")
    print("=" * 100)
    for lab, kauke in [("visi sektoriai", pd.Series(True, index=df.index)),
                       ("be zaliavu", ~df["sekt"].isin(ZALIAVU_SEKT)),
                       ("tik zaliavos", df["sekt"].isin(ZALIAVU_SEKT))]:
        sub = df[kauke & (df["ibs"] <= IBS_RIBA)]
        r, n = ivertink(sub, "r3_dm")
        if r is None:
            print(f"{lab:<24} {n:>7}   per maza imtis")
        else:
            print(f"{lab:<24} {r['n']:>7} {r['mean']:>+11.3f}% "
                  f"[{r['lo']:+.3f}..{r['hi']:+.3f}]")

    # ---------- F. SLANKIOJO LANGO PATIKRA ----------
    print("\n" + "=" * 100)
    print("F. SLANKUS LANGAS — ar signalas turi periodu, ir ar juos galima numatyti")
    print("Svarbiausia ne 'teigiamu langu dalis', o ar PRAEITAS langas prognozuoja kita.")
    print("=" * 100)
    for sig in ["z20", "ibs", "rsi2"]:
        sel = df[df[sig] <= df[sig].quantile(0.2)].copy()
        if len(sel) < 1000:
            continue
        pagal_diena = sel.groupby("data")["r3_dm"].mean().sort_index()
        for langas in (20, 60):
            rr = pagal_diena.rolling(langas).mean().dropna()
            if len(rr) < 100:
                continue
            teig = float((rr > 0).mean() * 100)
            ats = rr.autocorr(lag=langas) if len(rr) > langas * 2 else float("nan")
            zyma = ("prognozuoja" if ats == ats and abs(ats) > 0.3 else "neprognozuoja")
            print(f"  {sig:<6} {langas:>3} d. langas: teigiamu {teig:>5.1f}%, "
                  f"praeities rysys su ateitimi {ats:>+.3f}  -> {zyma}")

    # ---------- ADX / OBV (jei praejo koreliacijos vartus) ----------
    if tirti:
        print("\n" + "=" * 100)
        print(f"E2. ADX / OBV LANGELIAI (tikrinami: {', '.join(tirti)})")
        print("=" * 100)
        if "adx" in tirti:
            for lab, kauke in [("ADX < 20 (soninis)", df["adx"] < 20),
                               ("ADX 20-30", (df["adx"] >= 20) & (df["adx"] < 30)),
                               ("ADX > 30 (trendas)", df["adx"] >= 30)]:
                sub = df[kauke & (df["ibs"] <= IBS_RIBA)]
                r, n = ivertink(sub, "r3_dm")
                if r:
                    print(f"{lab:<24} {r['n']:>7} {r['mean']:>+11.3f}% "
                          f"[{r['lo']:+.3f}..{r['hi']:+.3f}]")
                else:
                    print(f"{lab:<24} {n:>7}   per maza imtis")
        if "obv_slope" in tirti:
            med = df["obv_slope"].median()
            for lab, kauke in [("OBV kyla", df["obv_slope"] > med),
                               ("OBV krenta", df["obv_slope"] <= med)]:
                sub = df[kauke & (df["ibs"] <= IBS_RIBA)]
                r, n = ivertink(sub, "r3_dm")
                if r:
                    print(f"{lab:<24} {r['n']:>7} {r['mean']:>+11.3f}% "
                          f"[{r['lo']:+.3f}..{r['hi']:+.3f}]")
                else:
                    print(f"{lab:<24} {n:>7}   per maza imtis")

    # ---------- PATVIRTINIMAS 2-OJE PUSEJE ----------
    print("\n" + "=" * 100)
    print("PATVIRTINIMAS 2-OJE PUSEJE — tik tai, kas 1-oje islaike BH pataisa")
    print("=" * 100)
    if rez_b:
        pv = np.array([r["p_two"] for _, _, r in rez_b])
        ok = bh(pv)
        kandidatai = [(lab, k) for (lab, k, _), o in zip(rez_b, ok) if o]
        if not kandidatai:
            print("  1 etape nei viena RSI salyga neislaike pataisos.")
        for lab, kauke in kandidatai:
            r, n = ivertink(df[kauke & p2], "r3_dm")
            if r is None:
                print(f"{lab:<26} {n:>7}   per maza imtis")
                continue
            v = ("PATVIRTINTA" if r["lo"] > 0 else
                 ("PRIESINGA" if r["hi"] < 0 else "nepatvirtinta"))
            print(f"{lab:<26} {r['n']:>7} {r['mean']:>+11.3f}% "
                  f"[{r['lo']:+.3f}..{r['hi']:+.3f}] {v}")
    else:
        print("  nera ka patvirtinti")

    print("\nAPRIBOJIMAI: islikimo salismas; stop tikrinamas is dienos H/L, todel "
          "\nrealus vykdymas butu blogesnis; svyravimo ribos imtos is 1-os puses; "
          "\nsanaudos {:.0f} EUR ciklui. Pjuviu lenteles skirtos hipotezems, "
          "\nne patvirtinimui — sprendziam pagal 2-osios puses eilute."
          .format(a.sanaudos_eur))


if __name__ == "__main__":
    main()
