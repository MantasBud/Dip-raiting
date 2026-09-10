#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sortai — vieso ES sortu registro duomenys.

KAS TAI IR KODEL SVARBU

Pagal ES reglamenta 236/2012 grynosios sortu pozicijos, virsijancios 0.5%
bendroves akciju, PRIVALO buti viesai atskleistos, ir apie kiekviena 0.1%
pokyti virs tos ribos. Tai vienintele mums prieinama informacija, kuri NERA
kainos isvestine — ji rodo, ka realiai daro dideli dalyviai.

Kodel tai svarbu butent siai strategijai: vartotojas perka kritimus ir vengia
trendiniu. Kritimas akcijoje, kurioje fondai DIDINA sortus, yra tiksliai tas
trendinis kritimas, kurio reikia vengti. Iki siol modulis tai spedavo
netiesiogiai (per apyvartos suoli), o dabar gali matyti tiesiogiai.

SALTINIAI
- AMF (Prancuzija): kasdienis istorinis failas nuo 2012-11-01
- AFM (Nyderlandai): einamuju ir archyviniu pranesimu registras
- Bundesanzeiger (Vokietija): paieska, sunkiau nuskaitoma

APRIBOJIMAI, KURIUOS REIKIA ZINOTI
- Vieso registro riba yra 0.5%: mazesnes pozicijos matomos tik reguliatoriui,
  todel "sortu nera" reiskia "nera virs 0.5%", ne "nera is viso".
- Pranesimai vėluoja iki 1 darbo dienos.
- Ne visos salys skelbia vienodai patogiai; ko negaunam, tai tiesiog trukstama.
- Sio signalo NIEKAD nematavom, todel jis rodomas kaip INFORMACIJA, ne kaip
  balo dedamoji. Jei norim ji naudoti sprendimui — pirma i backtesta.
"""

import json
import os
import time
from datetime import datetime, timedelta

import pandas as pd

# Kur laikom ISIN atitikmenis, kad ju netraukti kas karta
ISIN_CACHE = "docs/isin.json"
SORTU_CACHE_MIN = 360          # 6 val. — registrai atnaujinami kartа per diena

AMF_URL = ("https://www.amf-france.org/sites/institutionnel/files/"
           "doctrine/Position%20nette%20courte/"
           "Historique%20des%20positions%20courtes%20nettes%20publiees.csv")

_CACHE = {"laikas": 0, "duomenys": None}


def imk_isin(yf, symbols, kelias=ISIN_CACHE):
    """ISIN kodai tikeriams. Traukiami po viena, todel kesuojami faile.

    Be ISIN registro duomenu prie akciju nepriskirsim: registrai nezino
    Yahoo tikeriu.
    """
    zemelapis = {}
    try:
        if os.path.exists(kelias):
            with open(kelias, encoding="utf-8") as f:
                zemelapis = json.load(f)
    except Exception:
        zemelapis = {}

    truksta = [s for s in symbols if s not in zemelapis]
    if truksta:
        for s in truksta[:40]:            # po truputi, kad neuzkrautume Yahoo
            try:
                isin = yf.Ticker(s).isin
                zemelapis[s] = isin if isin and isin != "-" else ""
            except Exception:
                zemelapis[s] = ""
        try:
            os.makedirs(os.path.dirname(kelias) or ".", exist_ok=True)
            with open(kelias, "w", encoding="utf-8") as f:
                json.dump(zemelapis, f, ensure_ascii=False, indent=0)
        except Exception:
            pass
    return {s: v for s, v in zemelapis.items() if v}


def _amf():
    """Prancuzijos AMF kasdienis failas. Grazina DataFrame arba None."""
    try:
        df = pd.read_csv(AMF_URL, sep=";", encoding="latin-1", on_bad_lines="skip")
    except Exception:
        try:
            df = pd.read_csv(AMF_URL, sep=",", encoding="utf-8", on_bad_lines="skip")
        except Exception:
            return None
    # Stulpeliu pavadinimai skiriasi tarp versiju — ieskom pagal turini
    stulp = {c.lower(): c for c in df.columns}
    isin_c = next((v for k, v in stulp.items() if "isin" in k), None)
    poz_c = next((v for k, v in stulp.items()
                  if "position" in k and ("%" in k or "nette" in k or "short" in k)), None)
    data_c = next((v for k, v in stulp.items() if "date" in k), None)
    if not (isin_c and poz_c):
        return None
    out = pd.DataFrame({
        "isin": df[isin_c].astype(str).str.strip(),
        "poz": pd.to_numeric(df[poz_c].astype(str).str.replace(",", "."),
                             errors="coerce"),
        "data": pd.to_datetime(df[data_c], errors="coerce", dayfirst=True)
        if data_c else pd.NaT,
    }).dropna(subset=["isin", "poz"])
    return out


def surink(yf, symbols, verbose=True):
    """Grazina {tikeris: {"suma": %, "pokytis": p.p., "sk": kiek fondu}}.

    "suma"     — bendra vieso registro sortu dalis siandien
    "pokytis"  — kiek ji pasikeite per 5 darbo dienas (+ reiskia augancius sortus)
    "sk"       — kiek atskiru fondu turi pozicija
    """
    dabar = time.time()
    if _CACHE["duomenys"] is not None and dabar - _CACHE["laikas"] < SORTU_CACHE_MIN * 60:
        baze = _CACHE["duomenys"]
    else:
        baze = _amf()
        _CACHE["duomenys"], _CACHE["laikas"] = baze, dabar
        if verbose:
            print(f"  sortu registras: {'AMF ' + str(len(baze)) + ' irasu' if baze is not None else 'nepavyko'}")

    if baze is None or baze.empty:
        return {}

    isin_map = imk_isin(yf, symbols)
    if not isin_map:
        return {}
    atv = {v: k for k, v in isin_map.items()}

    sub = baze[baze["isin"].isin(atv)]
    if sub.empty:
        return {}

    out = {}
    riba = pd.Timestamp.now().normalize() - pd.Timedelta(days=7)
    for isin, g in sub.groupby("isin"):
        tik = atv.get(isin)
        if not tik:
            continue
        g = g.sort_values("data")
        dabartines = g[g["data"] >= riba] if g["data"].notna().any() else g
        suma = float(dabartines["poz"].sum()) if len(dabartines) else float(g["poz"].sum())
        anksciau = g[g["data"] < riba]
        sena = float(anksciau["poz"].sum()) if len(anksciau) else suma
        out[tik] = dict(suma=round(suma, 2),
                        pokytis=round(suma - sena, 2),
                        sk=int(len(dabartines) if len(dabartines) else len(g)))
    if verbose:
        print(f"  sortu duomenu turi: {len(out)} is {len(symbols)} akciju")
    return out


def zyma(info):
    """Trumpas tekstas kortelei ir zymos lygis."""
    if not info:
        return None
    suma, pok, sk = info.get("suma", 0), info.get("pokytis", 0), info.get("sk", 0)
    if suma <= 0:
        return None
    kryptis = ("auga" if pok > 0.05 else ("mazeja" if pok < -0.05 else "nekinta"))
    lygis = "warn" if (suma >= 3.0 or pok > 0.3) else "info"
    txt = (f"Vieši šortai {suma:.2f}% ({sk} fond{'ai' if sk > 1 else 'as'}), "
           f"per savaitę {pok:+.2f} p. p. — {kryptis}")
    if pok > 0.3:
        txt += ". Fondai didina pozicijas prieš šią akciją — kritimas gali būti trendinis"
    return (lygis, txt)
