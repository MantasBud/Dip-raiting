#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universas — eurais kotiruojamos Europos akcijos, sektoriai ir sektoriu ETF.

Si failа importuoja IR backtestas, IR skeneris, kad nebutu dvieju skirtingu
saraso versiju. Tik EUR birzos: XETRA .DE, Euronext .PA .AS .BR .LS,
Milanas .MI, Madridas .MC, Viena .VI, Helsinkis .HE, Dublinas .IR.
Nera .L (GBP), .SW (CHF), .ST (SEK), .CO (DKK), .OL (NOK).
"""

EUR_GALUNES = {"DE", "PA", "AS", "BR", "LS", "MI", "MC", "VI", "HE", "IR"}

# Birzos prekybos valandos (Berlyno laiku) — skirtingos birzos, skirtingi grafikai
BIRZU_VALANDOS = {
    "DE": (9, 0, 17, 30), "PA": (9, 0, 17, 30), "AS": (9, 0, 17, 30),
    "BR": (9, 0, 17, 30), "LS": (8, 0, 16, 30), "MI": (9, 0, 17, 30),
    "MC": (9, 0, 17, 30), "VI": (9, 0, 17, 30), "HE": (10, 0, 18, 30),
    "IR": (9, 0, 17, 30),
}

UNIVERSAS = {
    "Technologijos": [
        "ASML.AS", "ASM.AS", "BESI.AS", "IFX.DE", "STMPA.PA", "AIXA.DE", "SOI.PA",
        "SAP.DE", "DSY.PA", "CAP.PA", "TEP.PA", "ADYEN.AS", "PRX.AS", "NOKIA.HE",
        "NEM.DE", "SU.PA", "ATE.PA", "EVO.MI", "TKA.DE",
    ],
    "Pramone": [
        "SIE.DE", "KGX.DE", "LR.PA", "PRY.MI", "NEX.PA", "ENR.DE", "VIE.PA",
        "BNR.DE", "GEA.DE", "DUE.DE", "KRZ.IR", "IFX.DE", "ZAL.DE", "NDA.DE",
        "AIXA.DE", "JUN3.DE", "RAA.DE", "PUM.DE", "WCH.DE", "SRT3.DE",
    ],
    "Gynyba ir aviacija": [
        "RHM.DE", "LDO.MI", "HO.PA", "AIR.PA", "SAF.PA", "MTX.DE", "AM.PA",
        "R3NK.DE", "AVIO.MI", "EXA.PA",
    ],
    "Automobiliai": [
        "MBG.DE", "BMW.DE", "VOW3.DE", "P911.DE", "STLAM.MI", "CON.DE", "RNO.PA",
        "PAH3.DE", "FRVIA.PA", "ML.PA", "PIRC.MI", "BRE.MI", "LEO.MI",
    ],
    "Bankai ir finansai": [
        "BNP.PA", "ACA.PA", "GLE.PA", "DBK.DE", "CBK.DE", "UCG.MI", "ISP.MI",
        "SAN.MC", "BBVA.MC", "CABK.MC", "INGA.AS", "ABN.AS", "KBC.BR", "BAMI.MI",
        "BPE.MI", "SAB.MC", "UNI.MI", "BIRG.IR", "AIBG.IR", "DBAN.DE",
    ],
    "Draudimas": [
        "ALV.DE", "CS.PA", "MUV2.DE", "HNR1.DE", "G.MI", "NN.AS", "AGN.AS",
        "ASRNL.AS", "MAP.MC", "UNI.MI", "TLX.DE", "SCR.PA",
    ],
    "Energetika": [
        "TTE.PA", "ENI.MI", "REP.MC", "OMV.VI", "RWE.DE", "EOAN.DE", "ENEL.MI",
        "IBE.MC", "ENGI.PA", "VER.VI", "ELE.MC", "TRN.MI", "SRG.MI", "RED.MC",
        "NEOEN.PA", "EDP.LS", "EDPR.LS", "GALP.LS", "FORTUM.HE", "NESTE.HE",
    ],
    "Medziagos": [
        "BAS.DE", "AI.PA", "MT.AS", "SY1.DE", "LXS.DE", "EVK.DE", "AKZA.AS",
        "DSFIR.AS", "UPM.HE", "STERV.HE", "SOF.BR", "UMI.BR", "ARKEMA.PA",
        "IMCD.AS", "K1R.DE", "SGO.PA", "HEI.DE", "CRH.IR", "BZU.MI",
    ],
    "Sveikata": [
        "SAN.PA", "BAYN.DE", "FRE.DE", "FME.DE", "MRK.DE", "EL.PA", "UCB.BR",
        "PHIA.AS", "DIM.PA", "QIA.DE", "SHL.DE", "GN1.DE", "RCO.PA", "ORNBV.HE",
        "REC.MI", "DIA.MI", "GRF.MC", "ALM.MC", "ROVI.MC", "IPN.PA",
    ],
    "Vartojimas ir prabanga": [
        "MC.PA", "KER.PA", "RMS.PA", "MONC.MI", "ADS.DE", "PUM.DE", "ITX.MC",
        "CPR.MI", "TOD.MI", "SFER.MI", "BRBY.PA", "EO.PA", "RI.PA", "OR.PA",
    ],
    "Maistas ir kasdienes prekes": [
        "ABI.BR", "HEIA.AS", "BN.PA", "AD.AS", "CA.PA", "DANOY.PA", "LDO.MI",
        "CARL.DE", "BEI.DE", "HEN3.DE", "SW.PA", "VIV.PA", "JDEP.AS", "COLR.BR",
    ],
    "Telekomai ir ziniasklaida": [
        "DTE.DE", "ORA.PA", "TEF.MC", "TIT.MI", "KPN.AS", "PUB.PA", "PROX.BR",
        "TEL.VI", "ELISA.HE", "MEO.LS", "NOS.LS", "RCS.MI", "MDG.PA", "TKWY.AS",
    ],
    "Keliones ir laisvalaikis": [
        "LHA.DE", "AF.PA", "RYA.IR", "AC.PA", "EVD.DE", "TUI1.DE", "FDJ.PA",
        "IAG.MC", "AENA.MC", "MEL.MC", "AMS.MC", "FLTR.IR",
    ],
    "Nekilnojamas turtas ir statyba": [
        "VNA.DE", "URW.AS", "LEG.DE", "TEG.DE", "DIC.DE", "COL.MC", "MRL.MC",
        "IGD.MI", "GFC.PA", "ICAD.PA", "KOJAMO.HE", "SPS.MI",
    ],
    # JAV akciju antriniai listingai Frankfurte (kotiruojami EUR). Judrumas
    # didziausias visame sarase, bet: europietiska sesija daugiausia atkartoja
    # tai, kas jau ivyko JAV; likvidumas plonesnis; naktiniai suoliai dazniau
    # virsija stop atstuma. Modulis juos rodo su ispejimais, sprendzia vartotojas.
    "JAV antriniai listingai": [
        "NVD.DE", "AMD.DE", "APC.DE", "MSF.DE", "ABEA.DE", "AMZ.DE", "TL0.DE",
        "FB2A.DE", "NFC.DE", "IBM.DE", "INL.DE", "QCI.DE", "TXN.DE", "AVG.DE",
        "MUB.DE", "AMD.DE", "PLTR.DE", "COIN.DE", "MRNA.DE", "RIVN.DE",
        "SMCI.DE", "ARM.DE", "SNOW.DE", "CRWD.DE", "NOW.DE", "UBER.DE",
    ],
    # Judrios Europos vidutines kapitalizacijos akcijos
    "Judrios Europos": [
        "YDX.DE", "PTX.DE", "EVT.DE", "NDX1.DE", "AFX.DE", "COK.DE", "B4B.DE",
        "SHL.DE", "1U1.DE", "AOF.DE", "SANT.DE", "TMV.DE", "PNE3.DE", "VBK.DE",
        "SBS.DE", "JEN.DE", "SFQ.DE", "ELG.DE", "HYQ.DE", "NA9.DE",
    ],
    # Zaliavos — fiziniais metalais padengti ETC ir naftos instrumentai, visi
    # Xetra, EUR. Jie juda pagal palukanu normas ir geopolitika, t. y. pagal
    # kitus veiksnius nei akcijos, todel yra vienintele nekoreliuojanti dalis
    # sarase. DEMESIO: musu patvirtinti signalai matuoja grizima prie vidurkio
    # AKCIJOSE; zaliavose trendai stipresni, o grizimas silpnesnis, ir to
    # netikrinom. Zurnale jos matuojamos atskirai — jei neveiks, isimsim.
    "Zaliavos": [
        # Patikrinti Yahoo tikeriai (2026-09):
        "4GLD.DE",    # Xetra-Gold, fizinis auksas, fondas ~21.8 mlrd EUR
        "XAD6.DE",    # Xtrackers Physical Silver ETC (EUR)
        "8PSB.DE",    # Invesco Physical Silver ETC
        "IB1T.DE",    # iShares Bitcoin ETP — didziausio judrumo instrumentas
        # Nepatikrinti tikeriai cia neidedami: jei ju nera, Yahoo tiesiog
        # negrazina duomenu, o modulis irasytu i "nepavyko" sarasa.
    ],
    "Logistika ir mazmena": [
        "DHL.DE", "DPW.DE", "GLPG.AS", "PST.MI", "BIM.PA", "ATO.PA", "SES.PA",
        "ZAL.DE", "HFG.DE", "TKWY.AS",
    ],
}

# Sektoriaus ETF (iShares STOXX Europe 600, Xetra, EUR)
SEKTORIU_ETF = {
    "Technologijos": "EXV3.DE",
    "Pramone": "EXH4.DE",
    "Gynyba ir aviacija": "EXH4.DE",
    "Automobiliai": "EXV5.DE",
    "Bankai ir finansai": "EXV1.DE",
    "Draudimas": "EXH5.DE",
    "Energetika": "EXH1.DE",
    "Medziagos": "EXV6.DE",
    "Sveikata": "EXV4.DE",
    "Vartojimas ir prabanga": "EXH6.DE",
    "Maistas ir kasdienes prekes": "EXH7.DE",
    "Telekomai ir ziniasklaida": "EXV2.DE",
    "Keliones ir laisvalaikis": "EXV9.DE",
    "Nekilnojamas turtas ir statyba": "EXV8.DE",
    "Logistika ir mazmena": "EXH4.DE",
    "JAV antriniai listingai": "EXV3.DE",     # daugiausia technologijos
    "Judrios Europos": "EXV3.DE",
    # Zaliavos savo sektoriaus ETF neturi — filtrui naudojam pati auksa,
    # nes visi sio sektoriaus instrumentai juda panasiai
    "Zaliavos": "4GLD.DE",
}
INDEKSAS = "EXSA.DE"

# JAV — nepriklausomam patikrinimui
UNIVERSAS_US = {
    "Technologijos": ["AAPL", "MSFT", "NVDA", "AMD", "AVGO", "INTC", "MU", "AMAT",
                      "LRCX", "KLAC", "ADI", "TXN", "QCOM", "CRM", "ORCL", "ADBE",
                      "NOW", "PANW", "SNPS", "CDNS"],
    "Pramone": ["CAT", "DE", "HON", "GE", "MMM", "EMR", "ETN", "PH", "ITW", "CMI",
                "PCAR", "ROK", "AME", "FTV"],
    "Gynyba ir aviacija": ["LMT", "RTX", "NOC", "GD", "BA", "LHX", "TDG", "HWM"],
    "Automobiliai": ["TSLA", "F", "GM", "APTV", "LEA", "BWA"],
    "Bankai ir finansai": ["JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW", "USB",
                           "PNC", "TFC", "COF"],
    "Draudimas": ["BRK-B", "PGR", "TRV", "ALL", "AIG", "MET", "PRU", "CB"],
    "Energetika": ["XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "OXY", "HAL",
                   "DVN", "FANG"],
    "Medziagos": ["LIN", "APD", "SHW", "FCX", "NEM", "NUE", "DOW", "DD", "PPG"],
    "Sveikata": ["JNJ", "PFE", "MRK", "ABBV", "LLY", "TMO", "ABT", "DHR", "BMY",
                 "AMGN", "GILD", "VRTX", "REGN", "ISRG"],
    "Vartojimas ir prabanga": ["NKE", "SBUX", "MCD", "TJX", "LOW", "HD", "RL", "TPR"],
    "Maistas ir kasdienes prekes": ["PG", "KO", "PEP", "COST", "WMT", "MDLZ",
                                    "CL", "KMB", "GIS", "K"],
    "Telekomai ir ziniasklaida": ["T", "VZ", "TMUS", "DIS", "NFLX", "CMCSA", "META"],
    "Keliones ir laisvalaikis": ["DAL", "UAL", "LUV", "AAL", "MAR", "HLT", "RCL",
                                 "CCL", "BKNG", "ABNB"],
    # JAV akciju antriniai listingai Frankfurte (kotiruojami EUR). Judrumas
    # didziausias visame sarase, bet: europietiska sesija daugiausia atkartoja
    # tai, kas jau ivyko JAV; likvidumas plonesnis; naktiniai suoliai dazniau
    # virsija stop atstuma. Modulis juos rodo su ispejimais, sprendzia vartotojas.
    "JAV antriniai listingai": [
        "NVD.DE", "AMD.DE", "APC.DE", "MSF.DE", "ABEA.DE", "AMZ.DE", "TL0.DE",
        "FB2A.DE", "NFC.DE", "IBM.DE", "INL.DE", "QCI.DE", "TXN.DE", "AVG.DE",
        "MUB.DE", "AMD.DE", "PLTR.DE", "COIN.DE", "MRNA.DE", "RIVN.DE",
        "SMCI.DE", "ARM.DE", "SNOW.DE", "CRWD.DE", "NOW.DE", "UBER.DE",
    ],
    # Judrios Europos vidutines kapitalizacijos akcijos
    "Judrios Europos": [
        "YDX.DE", "PTX.DE", "EVT.DE", "NDX1.DE", "AFX.DE", "COK.DE", "B4B.DE",
        "SHL.DE", "1U1.DE", "AOF.DE", "SANT.DE", "TMV.DE", "PNE3.DE", "VBK.DE",
        "SBS.DE", "JEN.DE", "SFQ.DE", "ELG.DE", "HYQ.DE", "NA9.DE",
    ],
    # Zaliavos — fiziniais metalais padengti ETC ir naftos instrumentai, visi
    # Xetra, EUR. Jie juda pagal palukanu normas ir geopolitika, t. y. pagal
    # kitus veiksnius nei akcijos, todel yra vienintele nekoreliuojanti dalis
    # sarase. DEMESIO: musu patvirtinti signalai matuoja grizima prie vidurkio
    # AKCIJOSE; zaliavose trendai stipresni, o grizimas silpnesnis, ir to
    # netikrinom. Zurnale jos matuojamos atskirai — jei neveiks, isimsim.
    "Zaliavos": [
        # Patikrinti Yahoo tikeriai (2026-09):
        "4GLD.DE",    # Xetra-Gold, fizinis auksas, fondas ~21.8 mlrd EUR
        "XAD6.DE",    # Xtrackers Physical Silver ETC (EUR)
        "8PSB.DE",    # Invesco Physical Silver ETC
        "IB1T.DE",    # iShares Bitcoin ETP — didziausio judrumo instrumentas
        # Nepatikrinti tikeriai cia neidedami: jei ju nera, Yahoo tiesiog
        # negrazina duomenu, o modulis irasytu i "nepavyko" sarasa.
    ],
    "Logistika ir mazmena": ["UPS", "FDX", "CSX", "UNP", "NSC", "ODFL", "AMZN"],
}
US_ETF = {
    "Technologijos": "XLK", "Pramone": "XLI", "Gynyba ir aviacija": "ITA",
    "Automobiliai": "XLY", "Bankai ir finansai": "XLF", "Draudimas": "XLF",
    "Energetika": "XLE", "Medziagos": "XLB", "Sveikata": "XLV",
    "Vartojimas ir prabanga": "XLY", "Maistas ir kasdienes prekes": "XLP",
    "Telekomai ir ziniasklaida": "XLC", "Keliones ir laisvalaikis": "XLY",
    "Logistika ir mazmena": "XLI",
}
US_INDEKSAS = "SPY"


def visi_tikeriai(uni=None):
    uni = uni or UNIVERSAS
    matyti, out = set(), []
    for lst in uni.values():
        for s in lst:
            if s not in matyti:
                matyti.add(s)
                out.append(s)
    return out


def sektoriai(uni=None):
    uni = uni or UNIVERSAS
    out = {}
    for k, lst in uni.items():
        for s in lst:
            out.setdefault(s, k)      # pirmas sektorius laimi
    return out


def tik_eurais(tikeriai):
    return [s for s in tikeriai
            if "." in s and s.rsplit(".", 1)[1].upper() in EUR_GALUNES]
