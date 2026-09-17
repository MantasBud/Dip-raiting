"""PASTABA: sis failas turi ATSKIRA universa (306 tikeriai), kuris NESUTAMPA
su universas.py (214). Backtestui tai priimtina — platesne imtis duoda daugiau
duomenu. Bet skeneris turi naudoti universas.py, kitaip vel matuotume viena,
o naudotume kita. Pasalinti: STM.MI (dublikatas su STMPA.PA), JDEP.AS ir
TKWY.AS (patikrinta — Yahoo duomenu negrazina).

Bendras universas ir 60m atsisiuntimas (naudoja backtest_universas_60m.py ir backtest_rytinis_tesinys.py)."""
import sys, time
import numpy as np
import pandas as pd
import yfinance as yf

TZ = "Europe/Berlin"

UNIVERSAS = {
    # --- XETRA ---
    "SAP.DE": "tech", "SIE.DE": "pramone", "ALV.DE": "finansai", "DTE.DE": "telekom",
    "MUV2.DE": "finansai", "AIR.DE": "gynyba", "BAS.DE": "chemija", "BAYN.DE": "farma",
    "BMW.DE": "auto", "MBG.DE": "auto", "VOW3.DE": "auto", "P911.DE": "auto",
    "DHL.DE": "logistika", "DB1.DE": "finansai", "DBK.DE": "bankai", "CBK.DE": "bankai",
    "IFX.DE": "puslaidininkiai", "ADS.DE": "vartojimas", "HEN3.DE": "vartojimas",
    "BEI.DE": "vartojimas", "MRK.DE": "farma", "FRE.DE": "sveikata", "RHM.DE": "gynyba",
    "MTX.DE": "aviacija", "HEI.DE": "statyba", "SY1.DE": "chemija", "ZAL.DE": "prekyba",
    "EOAN.DE": "energetika", "RWE.DE": "energetika", "VNA.DE": "nt", "1COV.DE": "chemija",
    "BNR.DE": "chemija", "CON.DE": "auto", "SRT3.DE": "sveikata", "QIA.DE": "sveikata",
    "HNR1.DE": "finansai", "ENR.DE": "energetika", "PAH3.DE": "auto", "AIXA.DE": "puslaidininkiai",
    "SHL.DE": "sveikata", "DTG.DE": "auto", "HLAG.DE": "logistika", "NEM.DE": "tech",
    "EVK.DE": "chemija", "LHA.DE": "aviacija", "TKA.DE": "pramone", "PUM.DE": "vartojimas",
    "SDF.DE": "chemija", "G1A.DE": "pramone", "LEG.DE": "nt", "FNTN.DE": "telekom",
    "TEG.DE": "nt", "NDA.DE": "energetika", "KBX.DE": "pramone", "BC8.DE": "tech",
    "SIX2.DE": "sveikata", "WAF.DE": "puslaidininkiai", "KGX.DE": "pramone", "RAA.DE": "sveikata",
    "HFG.DE": "prekyba", "SAX.DE": "pramone", "JEN.DE": "tech", "GXI.DE": "sveikata",
    "AOF.DE": "tech", "S92.DE": "energetika", "BOSS.DE": "vartojimas", "NDX1.DE": "tech",
    # --- Euronext Paris ---
    "MC.PA": "prabanga", "OR.PA": "vartojimas", "TTE.PA": "nafta", "SAN.PA": "farma",
    "AI.PA": "chemija", "SU.PA": "pramone", "BNP.PA": "bankai", "CS.PA": "finansai",
    "RMS.PA": "prabanga", "KER.PA": "prabanga", "DG.PA": "statyba", "EL.PA": "sveikata",
    "SAF.PA": "aviacija", "HO.PA": "gynyba", "DSY.PA": "tech", "CAP.PA": "tech",
    "BN.PA": "vartojimas", "RI.PA": "vartojimas", "ENGI.PA": "energetika", "ORA.PA": "telekom",
    "VIE.PA": "energetika", "SGO.PA": "statyba", "ML.PA": "auto", "STLAP.PA": "auto",
    "RNO.PA": "auto", "ACA.PA": "bankai", "GLE.PA": "bankai", "PUB.PA": "media",
    "LR.PA": "pramone", "STMPA.PA": "puslaidininkiai", "WLN.PA": "tech", "EDEN.PA": "tech",
    "ERF.PA": "sveikata", "TEP.PA": "tech", "ALO.PA": "pramone", "BVI.PA": "pramone",
    "AC.PA": "kelioniu", "ENX.PA": "finansai", "SW.PA": "vartojimas", "CA.PA": "prekyba",
    "FR.PA": "auto", "VIV.PA": "media", "EN.PA": "statyba", "URW.PA": "nt", "RXL.PA": "pramone",
    "DIM.PA": "sveikata", "GTT.PA": "pramone", "NK.PA": "pramone", "COV.PA": "nt",
    "AMUN.PA": "finansai", "SOP.PA": "tech", "IPN.PA": "farma", "MF.PA": "pramone",
    # --- Euronext Amsterdam ---
    "ASML.AS": "puslaidininkiai", "ASM.AS": "puslaidininkiai", "BESI.AS": "puslaidininkiai",
    "ADYEN.AS": "tech", "INGA.AS": "bankai", "PHIA.AS": "sveikata", "PRX.AS": "tech",
    "HEIA.AS": "vartojimas", "AD.AS": "prekyba", "DSFIR.AS": "chemija", "WKL.AS": "media",
    "KPN.AS": "telekom", "RAND.AS": "paslaugos", "NN.AS": "finansai", "ABN.AS": "bankai",
    "AKZA.AS": "chemija", "IMCD.AS": "chemija", "MT.AS": "metalai", "UMG.AS": "media",
    "SHELL.AS": "nafta", "UNA.AS": "vartojimas", "AGN.AS": "finansai", "ASRNL.AS": "finansai",
    "ARCAD.AS": "paslaugos", "FUR.AS": "pramone",
    "OCI.AS": "chemija", "SBMO.AS": "nafta", "LIGHT.AS": "pramone", "AALB.AS": "pramone",
    # --- Borsa Italiana ---
    "ENI.MI": "nafta", "ENEL.MI": "energetika", "ISP.MI": "bankai", "UCG.MI": "bankai",
    "STLAM.MI": "auto", "RACE.MI": "auto", "G.MI": "finansai", "LDO.MI": "gynyba",
    "PRY.MI": "pramone", "TEN.MI": "nafta", "MONC.MI": "prabanga", "BAMI.MI": "bankai",
    "MB.MI": "bankai", "TIT.MI": "telekom", "SRG.MI": "energetika", "TRN.MI": "energetika",
    "A2A.MI": "energetika", "REC.MI": "farma", "CPR.MI": "vartojimas", "AMP.MI": "farma",
    "BMED.MI": "finansai", "PST.MI": "finansai", "FBK.MI": "bankai", "DIA.MI": "sveikata",
    "IP.MI": "pramone", "BPE.MI": "bankai", "UNI.MI": "finansai", "NEXI.MI": "tech",
    "BZU.MI": "statyba", "BC.MI": "prabanga", "IVG.MI": "pramone",
    "MONC.MI": "prabanga", "INW.MI": "telekom", "ERG.MI": "energetika", "AZM.MI": "finansai",
    "BGN.MI": "finansai", "IG.MI": "energetika", "SFER.MI": "prabanga", "HER.MI": "energetika",
    # --- Bolsa de Madrid ---
    "SAN.MC": "bankai", "BBVA.MC": "bankai", "IBE.MC": "energetika", "ITX.MC": "prekyba",
    "TEF.MC": "telekom", "REP.MC": "nafta", "AMS.MC": "tech", "FER.MC": "statyba",
    "CABK.MC": "bankai", "AENA.MC": "kelioniu", "CLNX.MC": "telekom", "ACS.MC": "statyba",
    "IAG.MC": "aviacija", "ENG.MC": "energetika", "RED.MC": "energetika", "GRF.MC": "farma",
    "SAB.MC": "bankai", "BKT.MC": "bankai", "ELE.MC": "energetika", "NTGY.MC": "energetika",
    "ANA.MC": "statyba", "MAP.MC": "finansai", "ACX.MC": "metalai", "SCYR.MC": "statyba",
    "LOG.MC": "logistika", "ROVI.MC": "farma", "PUIG.MC": "vartojimas", "UNI.MC": "bankai",
    "MTS.MC": "metalai", "IDR.MC": "tech", "SLR.MC": "energetika", "FDR.MC": "auto",
    "COL.MC": "nt", "MRL.MC": "nt", "VIS.MC": "sveikata", "CIE.MC": "auto",
    # --- Euronext Brussels ---
    "ABI.BR": "vartojimas", "KBC.BR": "bankai", "UCB.BR": "farma", "SOLB.BR": "chemija",
    "AGS.BR": "finansai", "GBLB.BR": "finansai", "ARGX.BR": "farma", "ELI.BR": "energetika",
    "PROX.BR": "telekom", "COLR.BR": "prekyba", "DIE.BR": "prekyba", "MELE.BR": "tech",
    "WDP.BR": "nt", "ACKB.BR": "finansai", "LOTB.BR": "vartojimas", "UMI.BR": "chemija",
    "AED.BR": "nt", "SYENS.BR": "chemija", "XIOR.BR": "nt", "BEKB.BR": "pramone",
    # --- Nasdaq Helsinki ---
    "NOKIA.HE": "telekom", "NESTE.HE": "nafta", "SAMPO.HE": "finansai", "UPM.HE": "miskas",
    "KNEBV.HE": "pramone", "FORTUM.HE": "energetika", "STERV.HE": "miskas", "WRT1V.HE": "pramone",
    "ELISA.HE": "telekom", "ORNBV.HE": "farma", "METSO.HE": "pramone", "NDA-FI.HE": "bankai",
    "KESKOB.HE": "prekyba", "HUH1V.HE": "miskas", "VALMT.HE": "pramone", "TYRES.HE": "auto",
    "OUT1V.HE": "metalai", "KCR.HE": "pramone", "CGCBV.HE": "pramone", "QTCOM.HE": "tech",
    "TIETO.HE": "tech", "KEMIRA.HE": "chemija", "MANTA.HE": "finansai", "TELIA1.HE": "telekom",
    # --- Euronext Lisbon ---
    "EDP.LS": "energetika", "GALP.LS": "nafta", "JMT.LS": "prekyba", "EDPR.LS": "energetika",
    "BCP.LS": "bankai", "NOS.LS": "telekom", "SON.LS": "prekyba", "CTT.LS": "logistika",
    "NVG.LS": "miskas", "ALTR.LS": "miskas", "SEM.LS": "miskas", "REN.LS": "energetika",
    # --- Wiener Börse ---
    "OMV.VI": "nafta", "EBS.VI": "bankai", "VER.VI": "energetika", "VOE.VI": "metalai",
    "ANDR.VI": "pramone", "RBI.VI": "bankai", "BG.VI": "bankai", "WIE.VI": "statyba",
    "CAI.VI": "nt", "POST.VI": "logistika", "LNZ.VI": "chemija", "SBO.VI": "nafta",
    "UQA.VI": "finansai", "ATS.VI": "tech", "DOC.VI": "pramone", "IIA.VI": "nt",
    # --- Euronext Dublin ---
    "RYA.IR": "aviacija", "KRZ.IR": "vartojimas", "BIRG.IR": "bankai", "A5G.IR": "bankai",
    "GL9.IR": "vartojimas", "KSP.IR": "statyba", "DHG.IR": "kelioniu", "PTSB.IR": "bankai",
}



def siusti_60m(tickers, chunk=25):
    out = {}
    for i in range(0, len(tickers), chunk):
        dalis = tickers[i:i + chunk]
        for bandymas in range(3):
            try:
                df = yf.download(dalis, interval="60m", period="730d", group_by="ticker",
                                 auto_adjust=False, progress=False, threads=True)
                break
            except Exception as e:
                print(f"  klaida ({e}); kartojama…", file=sys.stderr)
                time.sleep(5)
        else:
            continue
        if df is None or df.empty:
            continue
        if len(dalis) == 1:
            out[dalis[0]] = df
            continue
        for t in dalis:
            if t in df.columns.get_level_values(0):
                sub = df[t].dropna(subset=["Close"])
                if len(sub) > 500:
                    out[t] = sub
        print(f"  {min(i + chunk, len(tickers))}/{len(tickers)}", file=sys.stderr)
    return out


def siusti_dieninius(ticker):
    df = yf.download(ticker, period="4y", interval="1d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close"])


