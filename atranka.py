#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atranka — kurios akcijos realiai tinka strategijai.

Paleidimas:
    pip install yfinance pandas numpy tzdata
    python atranka.py                    # ivertina visus kandidatus
    python atranka.py --top 45           # kiek imti i galutini sarasa
    python atranka.py --pozicija 18000   # portfelio dydis

KA MATUOJA (viskas is realiu duomenu, ne is vertinimu)

1. JUDRUMAS — ar akcija apskritai gali nueiti tavo tiksla per diena.
   Skaiciuojama: dienu dalis, kai dienos diapazonas >= MIN_TARGET.
   Tai tiesesnis matas nei ATR: ATR yra vidurkis, o tau svarbu, kaip DAZNAI
   proga apskritai atsiranda.

2. LIKVIDUMAS — ar tavo pozicija ispildoma be kainos pastumimo.
   Pozicija lyginama su dienos apyvarta eurais.

3. NAKTIES SUOLIS — ar stop apsaugo laikant per nakti.
   Jei tipinis suolis didesnis uz stop atstuma, akcija rizikinga tavo dydziui.

4. IBS PASISKIRSTYMAS — ar akcijoje apskritai buna zemo IBS dienu.
   Tai vienintelis patvirtintas modulio signalas: jei akcija retai uzdaro
   prie dienos dugno, ji signalo negeneruos, kad ir kokia butu judri.

5. KAINOS GRANULIACIJA — ar 18 000 EUR dalijasi i prasminga akciju skaiciu.
   Perkant 9 akcijas po 1900 EUR, kiekvienas vienetas yra 11% pozicijos.

6. GYVYBINGUMAS — ar duomenys pilni ir prekyba vyksta kasdien.

KO NEMATUOJA
Pelningumo ir kapitalizacijos — tam reikia fundamentaliu duomenu, kurie per
yfinance nepatikimi. Sie du filtrai taikomi rankiniu budu sudarant kandidatu
sarasa, o skriptas tikrina tik prekybines savybes.
"""

import argparse
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ----------------------------- KANDIDATAI -----------------------------
# Formatas: (tikeris, sektorius, subsektorius)
# Visos prekiaujamos eurais ir prieinamos per IBKR.

KANDIDATAI = [
    # Puslaidininkiai
    ("ASML.AS", "Puslaidininkiai", "Litografijos iranga"),
    ("ASM.AS", "Puslaidininkiai", "ALD nusodinimo iranga"),
    ("BESI.AS", "Puslaidininkiai", "Korpusavimo iranga"),
    ("IFX.DE", "Puslaidininkiai", "Galios ir automobiliu lustai"),
    ("STM.PA", "Puslaidininkiai", "Analoginiai ir automobiliu lustai"),
    ("AMS.SW", "Puslaidininkiai", "Jutikliai"),
    # Programine iranga ir IT
    ("SAP.DE", "Programine iranga", "Verslo valdymo sistemos"),
    ("DSY.PA", "Programine iranga", "Projektavimo programine iranga"),
    ("ADYEN.AS", "Programine iranga", "Mokejimu infrastruktura"),
    ("CAP.PA", "IT paslaugos", "Konsultacijos"),
    ("PRX.AS", "Programine iranga", "Interneto holdingas"),
    ("NOKIA.HE", "Telekomu iranga", "Tinklo infrastruktura"),
    ("ERIC-B.ST", "Telekomu iranga", "Tinklo infrastruktura"),
    # Pramone
    ("SIE.DE", "Pramone", "Automatizacija"),
    ("SU.PA", "Pramone", "Elektrifikacija"),
    ("KGX.DE", "Pramone", "Sandeliu automatizacija"),
    ("ABBN.SW", "Pramone", "Elektros iranga"),
    ("VOW3.DE", "Automobiliai", "Gamintojas"),
    # Gynyba ir aviacija
    ("RHM.DE", "Gynyba", "Sausumos sistemos"),
    ("LDO.MI", "Gynyba", "Elektronika ir sraigtasparniai"),
    ("AIR.PA", "Aviacija", "Lektuvu gamyba"),
    ("SAF.PA", "Aviacija", "Varikliai ir komponentai"),
    ("HO.PA", "Gynyba", "Elektronika"),
    # Energetika
    ("ENR.DE", "Energetikos iranga", "Turbinos ir tinklai"),
    ("RWE.DE", "Komunalines", "Elektros gamyba"),
    ("TTE.PA", "Nafta ir dujos", "Integruota"),
    ("ENI.MI", "Nafta ir dujos", "Integruota"),
    ("EOAN.DE", "Komunalines", "Tinklai"),
    # Finansai
    ("BNP.PA", "Bankai", "Universalus"),
    ("UCG.MI", "Bankai", "Universalus"),
    ("DBK.DE", "Bankai", "Investicinis"),
    ("INGA.AS", "Bankai", "Mazmeninis"),
    ("SAN.MC", "Bankai", "Tarptautinis"),
    ("ISP.MI", "Bankai", "Mazmeninis"),
    ("ALV.DE", "Draudimas", "Universalus"),
    ("CS.PA", "Draudimas", "Universalus"),
    # Automobiliai
    ("MBG.DE", "Automobiliai", "Premium gamintojas"),
    ("BMW.DE", "Automobiliai", "Premium gamintojas"),
    ("STLAM.MI", "Automobiliai", "Masinis gamintojas"),
    ("CON.DE", "Auto komponentai", "Padangos ir elektronika"),
    ("P911.DE", "Automobiliai", "Premium gamintojas"),
    # Medziagos ir chemija
    ("BAS.DE", "Chemija", "Bazine chemija"),
    ("MT.AS", "Metalai", "Plienas"),
    ("1COV.DE", "Chemija", "Polimerai"),
    ("AI.PA", "Chemija", "Pramonines dujos"),
    # Vartojimas ir prabanga
    ("MC.PA", "Prabanga", "Konglomeratas"),
    ("KER.PA", "Prabanga", "Konglomeratas"),
    ("MONC.MI", "Prabanga", "Apranga"),
    ("RMS.PA", "Prabanga", "Odos gaminiai"),
    ("ADS.DE", "Vartojimo prekes", "Sportine apranga"),
    ("ITX.MC", "Mazmena", "Apranga"),
    ("AD.AS", "Mazmena", "Maisto prekyba"),
    # Sveikata
    ("SAN.PA", "Farmacija", "Receptiniai vaistai"),
    ("PHIA.AS", "Medicinos technika", "Diagnostika"),
    ("BAYN.DE", "Farmacija", "Vaistai ir agrochemija"),
    ("FRE.DE", "Sveikatos paslaugos", "Ligonines"),
    # Transportas ir kita
    ("LHA.DE", "Aviakompanijos", "Tinklo vezejas"),
    ("DHL.DE", "Logistika", "Ekspres pristatymas"),
    ("MUV2.DE", "Draudimas", "Perdraudimas"),
    ("TEP.PA", "Komunalines", "Elektros gamyba"),
]

# ----------------------------- KRITERIJAI -----------------------------

MIN_TARGET = 2.0          # minimalus tikslas procentais
STOP_ATR_MULT = 0.55      # kaip modulyje
STOP_MIN_PCT = 0.8
HOLD_OVERNIGHT = True     # ar laikoma per nakti (tada svarbu suolio rizika)

# Ribos, kurias reikia perzengti
MIN_PROGU_DALIS = 35.0    # % dienu, kai dienos diapazonas >= MIN_TARGET
MAX_POZ_APYVARTOS = 1.0   # pozicija ne daugiau kaip tiek % dienos apyvartos
MIN_ZEMO_IBS_DIENU = 15.0 # % dienu, kai IBS <= 0.25
MAX_VNT_DALIS = 3.0       # vienas akcijos vienetas ne daugiau kaip tiek % pozicijos
MIN_DIENU = 180           # kiek prekybos dienu turi buti duomenyse


def atr_pct(d, n=14):
    h, l, c = d["High"], d["Low"], d["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    last = float(c.iloc[-1])
    return float(a) / last * 100 if last else None


def ivertink(d, pozicija):
    """Grazina matavimus vienai akcijai arba None, jei duomenu nepakanka."""
    d = d.dropna(subset=["Open", "High", "Low", "Close"])
    if len(d) < MIN_DIENU:
        return None

    price = float(d["Close"].iloc[-1])
    if price <= 0:
        return None

    # 1. Judrumas: kaip DAZNAI dienos diapazonas siekia tiksla
    rng_pct = (d["High"] - d["Low"]) / d["Close"] * 100
    progu_dalis = float((rng_pct >= MIN_TARGET).mean() * 100)
    atr = atr_pct(d)

    # 2. Likvidumas
    apyvarta = float((d["Close"] * d["Volume"]).tail(60).median())
    poz_dalis = pozicija / apyvarta * 100 if apyvarta > 0 else 999

    # 3. Nakties suolis pries stop atstuma
    gaps = ((d["Open"] - d["Close"].shift(1)).abs() / d["Close"].shift(1) * 100).dropna()
    gap = float(gaps.tail(120).median()) if len(gaps) > 20 else None
    stop_atstumas = max(STOP_MIN_PCT, STOP_ATR_MULT * (atr or 2.0))
    suolio_rizika = (gap / stop_atstumas) if (gap and stop_atstumas) else None

    # 4. IBS pasiskirstymas: ar buna dienu, kai uzdaro prie dugno
    rng = (d["High"] - d["Low"]).replace(0, np.nan)
    ibs = ((d["Close"] - d["Low"]) / rng).dropna()
    zemo_ibs = float((ibs <= 0.25).mean() * 100) if len(ibs) > 50 else None

    # 5. Granuliacija
    vnt_dalis = price / pozicija * 100

    # 6. Gyvybingumas: ar nera dienu be prekybos
    tuscios = float((d["Volume"].tail(120) == 0).mean() * 100)

    return dict(kaina=price, atr=atr, progos=progu_dalis, apyvarta=apyvarta,
                poz_dalis=poz_dalis, gap=gap, suolio_rizika=suolio_rizika,
                zemo_ibs=zemo_ibs, vnt_dalis=vnt_dalis, tuscios=tuscios,
                dienu=len(d))


def patikra(m):
    """Grazina (ar_praeina, problemu_sarasas)."""
    p = []
    if m["progos"] < MIN_PROGU_DALIS:
        p.append(f"per rami ({m['progos']:.0f}% dienu siekia {MIN_TARGET}%)")
    if m["poz_dalis"] > MAX_POZ_APYVARTOS:
        p.append(f"pozicija {m['poz_dalis']:.1f}% apyvartos")
    if m["zemo_ibs"] is not None and m["zemo_ibs"] < MIN_ZEMO_IBS_DIENU:
        p.append(f"retai uzdaro prie dugno ({m['zemo_ibs']:.0f}%)")
    if m["vnt_dalis"] > MAX_VNT_DALIS:
        p.append(f"brangus vienetas ({m['vnt_dalis']:.1f}% pozicijos)")
    if HOLD_OVERNIGHT and m["suolio_rizika"] and m["suolio_rizika"] > 1.0:
        p.append(f"nakties suolis virsija stop ({m['suolio_rizika']:.1f}x)")
    if m["tuscios"] > 5:
        p.append("dienu be prekybos")
    return (len(p) == 0), p


def balas(m):
    """Bendras tinkamumo balas. Svoriai atspindi, kas strategijai svarbiausia:
    daugiausia — kaip daznai atsiranda proga ir ar akcija generuoja IBS signala."""
    def norm(v, lo, hi):
        if v is None:
            return 50.0
        return float(np.clip((v - lo) / (hi - lo) * 100, 0, 100))

    dalys = [
        (norm(m["progos"], 20, 75), 35),                    # kaip daznai proga
        (norm(m["zemo_ibs"], 10, 35), 25),                  # ar generuoja signala
        (100 - norm(m["poz_dalis"], 0, 2), 15),             # likvidumas
        (100 - norm(m["suolio_rizika"] or 1.0, 0.3, 2.0), 15),  # nakties rizika
        (100 - norm(m["vnt_dalis"], 0.2, 5), 10),           # granuliacija
    ]
    return sum(v * w for v, w in dalys) / sum(w for _, w in dalys)


def main():
    ap = argparse.ArgumentParser(description="Akciju atranka strategijai")
    ap.add_argument("--top", type=int, default=45, help="Kiek akciju i galutini sarasa")
    ap.add_argument("--pozicija", type=float, default=18000.0, help="Pozicijos dydis EUR")
    ap.add_argument("--rodyti-visus", action="store_true", help="Rodyti ir neislaikiusius")
    args = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Paleisk: pip install yfinance pandas numpy tzdata")

    symbols = [s for s, _, _ in KANDIDATAI]
    print(f"Tikrinama {len(symbols)} kandidatu, pozicija {args.pozicija:,.0f} EUR…\n")
    data = yf.download(symbols, period="2y", interval="1d", group_by="ticker",
                       progress=False, auto_adjust=False, threads=True)

    rezultatai = []
    nepavyko = []
    for sym, sekt, subsekt in KANDIDATAI:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                d = data[sym].dropna(how="all")
            else:
                d = data.dropna(how="all")
            m = ivertink(d, args.pozicija)
            if not m:
                nepavyko.append((sym, "nepakanka duomenu"))
                continue
            ok, problemos = patikra(m)
            rezultatai.append(dict(sym=sym, sekt=sekt, subsekt=subsekt, ok=ok,
                                   problemos=problemos, balas=balas(m), **m))
        except Exception as e:
            nepavyko.append((sym, str(e)[:40]))

    if not rezultatai:
        sys.exit("Nepavyko ivertinti nei vienos akcijos.")

    rezultatai.sort(key=lambda x: -x["balas"])
    praeje = [r for r in rezultatai if r["ok"]]

    print(f"{'AKCIJA':<11} {'SEKTORIUS':<22} {'BALAS':>6} {'KAINA':>9} {'ATR':>6} "
          f"{'PROGOS':>7} {'IBS<0.25':>9} {'APYVARTA':>10} {'SUOLIS':>7}")
    print("-" * 100)
    for r in rezultatai:
        if not r["ok"] and not args.rodyti_visus:
            continue
        zym = "" if r["ok"] else " x"
        print(f"{r['sym']:<11} {r['sekt'][:21]:<22} {r['balas']:>6.1f} "
              f"{r['kaina']:>8.2f}\u20ac {(r['atr'] or 0):>5.1f}% {r['progos']:>6.0f}% "
              f"{(r['zemo_ibs'] or 0):>8.0f}% {r['apyvarta']/1e6:>8.0f}M\u20ac "
              f"{(r['suolio_rizika'] or 0):>6.1f}x{zym}")

    print(f"\nIslaike visas patikras: {len(praeje)} is {len(rezultatai)}")

    if not args.rodyti_visus:
        krito = [r for r in rezultatai if not r["ok"]]
        if krito:
            print(f"\nNEISLAIKE ({len(krito)}):")
            for r in sorted(krito, key=lambda x: -x["balas"])[:20]:
                print(f"  {r['sym']:<11} {'; '.join(r['problemos'][:2])}")

    # --- Galutinis sarasas su sektoriu balansu ---
    print("\n" + "=" * 100)
    print(f"SIULOMAS SARASAS (top {args.top}, ne daugiau kaip 4 is vieno sektoriaus)")
    print("=" * 100)
    galutinis, per_sekt = [], {}
    for r in praeje:
        if len(galutinis) >= args.top:
            break
        if per_sekt.get(r["sekt"], 0) >= 4:
            continue
        galutinis.append(r)
        per_sekt[r["sekt"]] = per_sekt.get(r["sekt"], 0) + 1

    for r in galutinis:
        print(f'    ("{r["sym"].split(".")[0]}", "{r["sym"]}", "{r["subsekt"]}"),'
              f'   # {r["sekt"]}, balas {r["balas"]:.0f}')

    print(f"\nSektoriu pasiskirstymas ({len(per_sekt)} sektoriai):")
    for s, n in sorted(per_sekt.items(), key=lambda x: -x[1]):
        print(f"  {s:<26} {n}")

    print("\nSECTORS zodynas moduliui:")
    for r in galutinis:
        print(f'    "{r["sym"]}": "{r["sekt"].lower()}",')

    if nepavyko:
        print(f"\nNepavyko ivertinti: {', '.join(s for s, _ in nepavyko)}")

    print(f"\nRibos: progu dalis >= {MIN_PROGU_DALIS}%, pozicija <= {MAX_POZ_APYVARTOS}% "
          f"apyvartos,\nzemo IBS dienu >= {MIN_ZEMO_IBS_DIENU}%, vienetas <= {MAX_VNT_DALIS}% "
          f"pozicijos,\nnakties suolis <= stop atstumas. Keisk jas failo virsuje.")


if __name__ == "__main__":
    main()
