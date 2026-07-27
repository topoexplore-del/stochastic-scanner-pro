"""
Stochastic Scanner Pro — Data Builder
======================================
Generates snapshot.json with Monte Carlo projections for a universe of tickers.

Each ticker output includes:
  - 2 years of daily price history (for SPE model)
  - Basic technical indicators (ATR, ADX proxy, RSI) used by SPE regime detection
  - Full SPE output: expected, prob_up, CI, VaR, CVaR, multi-horizon, regime

NO fundamental data. NO composite scores. NO market analysis layers.
This project is purely about stochastic projections.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add scripts dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stochastic_projector import StochasticProjector

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


# ─────────────────────────────────────────────────────────────
# Ticker universe — organized by group for the web UI
# ─────────────────────────────────────────────────────────────
TICKER_GROUPS = {
    "US Large Cap": [
        "VRT","POWL","ETN","ANET","MPWR","PWR","CAT","FCX",
        "NVDA","PLTR","AVGO","AMD","LMT","NOC","CEG","SMCI",
        "GE","ROK","URI","DE", "SMHI","ALRS","RELY","FVR","CMDB","DAVE","CALY","CPIX","ANGO","XHR",
    "CLBK","FNWB","NODK","SNFCA","BSVN","HNVR","UHAL","UHAL-B","WBI",
    "TVIV","SUNC","SIND","OCAC-U","NHP","JAN","COAG","ARCI","TBI","FA",
    "GTY","TRNO","KRT","PECO","HONA","SPCX","FDXF","DWD","DFAC","CGDV","CBRS","ALAB","QXO","AIRR",
    "HBC2","CGXU","AWM","FENI","TRVC","SPMO","AHR","PTY","BSCS","BSCR",
    "CGIC","FPS","FEOE","ALS","CRWV","QYLD","BSCT","PFFA","SARO","SHC",
    "QDPL","NAD","FEGE","UTH","FT2","TLN","RWL","GTO","JPC","Q","P","EX9",
    "BSCU","BOTZ","FEY","NAVN","MG1","SUX","ONEQ","SAJA","MMT","NZF","LIF",
    "AGNT","INFL"
    ],
    "🤖 AI & Semiconductors": [
        "TSM","ASML","LRCX","KLAC","MU","ARM","ON",
        "QCOM","AMAT","MRVL","ADI","NXPI"
    ],
    # ── LATAM COMPLETO ──
    "🌎 LATAM": [
        "NU","MELI","VALE","PBR","SQM","BSBR","ABEV","ITUB",
        "GLOB","STNE","AMX","FMX","PAC","VIST","SUPV",
        "BAP","KOF","BMA","IFS","TGLS","YPF","BCH","BVN",
        "BSAC","EDN","BWMX","BBAR","DLO","CCU","TEO",
        "CX","AUNA","TV"
    ],
    # ── COLOMBIA ──
    "🇨🇴 Colombia": [
        "EC","CIB","AVAL","CNNE","CRGIY"
    ],
    # ── ESPAÑA ──
    "🇪🇸 España": [
        "SAN","TEF","BBVA","IBE","ITX","REP"
    ],
    # ── REINO UNIDO ──
    "🇬🇧 Reino Unido": [
        "SHEL","AZN","HSBC","BP","RIO","LSEG",
        "GSK","UL","DEO","BCS","NWG","VOD"
    ],
    # ── CHINA CONTINENTAL ──
    "🇨🇳 China Continental": [
        "BABA","PDD","JD","BIDU","NIO","LI",
        "XPEV","ZK","FUTU","TME","BILI","YMM"
    ],
    # ── HONG KONG ──
    "🇭🇰 Hong Kong": [
        "0700.HK","9988.HK","1299.HK","0005.HK","2318.HK",
        "0388.HK","0941.HK","1810.HK","9618.HK","3690.HK"
    ],
    # ── ÍNDICES MUNDIALES ──
    "📊 Índices Mundiales": [
        "SPY","XLP","XLV","XLRE","XLE","XLB","XLU","XLF","XLY","XLI","XLC","XLK","QQQ","DIA","IWM",
        "EWZ","EWW","FXI","MCHI",
        "EWU","EWG","EWQ","EWJ","EWY",
        "EWA","EZA","EWT"
    ],
    # ── ETFs GLOBALES ──
    "💼 ETFs Globales": [
        "VTI","VXUS","VWO","VEA",
        "GLD","SLV","USO",
        "XLE","XLF","XLK","XLV","XLI",
        "SOXX","KWEB","ARKK",
        "TLT","HYG","LQD","VUG","IWF","VO","VGT","MGK","VOO","AGG","NOBL","VYM","IVV","VXUS","VEU","FBND","BNDX","HDV","PYLD","DFCF","XLB","VIG","MUB","IEF","GOVT","DYNF","SCHF","QQQM","AVDV","IALT","VTI","IEFA","XLK","SMH","VTIP","MBB","XLU","VYMI","JEPQ","PULS","VTWO","AIRR","FNDE","IJS","VOOG","ICLN","IJH","SCHD","DON","AVUV","DFAC","XLI","SHYG","DFAS","IGV","IQLT","SCHZ","CORO","DRAM","GGOV","BLCR","DFSD","JIRE","NYF","ANGL","VTES","HYD","PSI","QTUM","DIHP","QTEC","GPIQ","IFRA","BND","SCHX","SOXX","VCIT","SCHG","DFUS","SCHO","STIP","GLDM","IYW","IXUS","DVY","SCHE","GRID","DFAI","HEFA","ESGE","AVUS","VONV","EAGG","VCRB","PAAA","MGV","LMBS","BITO","XAR","CGMU","DUHP","VIS","ROBO","IBDV","IBDU","DBMF","IBTH","GSST","BALT","AOM","HYS","GPIX","FTSM","DVYE","VWO","VOE","JEPI","IWD","XLY","VT","ITA*","VXF","SDVY","ESGU","MDY","IWO","VSGX","GSLC","VGK","PFF","JPIE","VGLT","SCHH","PGX","IXN","VIGI","FDL","NLR","SPSB","IGIB","FLOT","IBDR","IDEV","JPEF","USRT","DFNM","DFEV","XJH","DIVO","SPHY","EFAV","PSK","CMDY","JCPB","JGRO","PMBS","AGGY","RECS","IDEF","KNG","AIPO","FWD","QLD","IBTI","IBTJ","IBTK","QQQI","NASA","FFLC","IBTL","IGPT","CDX","CGDG","TAFM","BUFF","ARTY","FLDR","VWOB","HDEF","JMOM","MINO","RAAX","SUSB","ICF","EMGF","VEA","SPYM","VTEB","BSV","VNQ","MINT","USHY","JPST","IEI","IUSB","CGGR","PAVE","DFAE","DFAT","SCHM","LQD","VGSH","XBI","IGSB","BOND","ESGV","EFG","SCHR","JBND","ACWX","ICSH","JAVA","PTL","GDXJ","CGMS","TBUX","SMMD","DIVI","SPYD","EEMV","BOTZ","IGRO","FREL","INTF","CWB","DFAL","SCHK","FHEQ","WQTM","LGOV","SSO","VPLS","FLHY","QLTA","NUKZ","BBAG","GFLW","DDTA","DDFA","XPH","AGZ","DDTZ","SDS",
        "AIQ","AVDE","AVGE","AVMV","BIV","BUFQ","CGCP","CGHM","CIBR","DCOR",
    "DFAR","DFAW","DFGP","DFGR","DFIC","DFIS","DFIV","DFLV","DFSI","DFSV",
    "DUSB","EMLC","ESGD","ESML","EWL","EWZ","FCOM","FDVV","FMDE","FNDX",
    "HACK","IBDW","IBDX","IYJ","JMST","JMUB","LMUB","LRGF","LVHI","MUNI",
    "NDQ","NEAR","PPA","RDVY","RSP","SCHP","SFLR","SPEM","SPIB","SPMB",
    "SPTL","SPY","SPYI","SPSM","SRLN","SUB","SYSB","TIP","UTES","VAW",
    "VCSH","VGIT","VLUE","VTEI","VTV","WCMI","XLP","XMMO","XOVR"
    ],
    # ── US MARKET (Top ~350 S&P 500 stocks) ──
    "🇺🇸 US Market": [
        # Tech
        "AAPL","MSFT","GOOGL","AMZN","META","TSLA","ORCL","CRM","ADBE","NOW",
        "PANW","CRWD","SNOW","NET","DDOG","INTU","CDNS","SNPS","FTNT","ZS",
        "WDAY","TEAM","HUBS","DOCU","VEEV","ANSS","CPAY","IT","KEYS","TYL",
        "EPAM","PAYC","MANH","MPWR","NXPI","MCHP","SWKS","QRVO","ZBRA","TER",
        "TRMB","GDDY","GEN","CTSH","WIT","ACN","IBM","CSCO","HPQ","HPE","DELL",
        "IMOS","FFIV","RDWR","CNXN","SCSC","NTCT","TRAX","IPCX","OKTA","TENB","QLYS","CYRX","VSTS",
          "KLAC","INTC","MU","MRVL","SNDK","TXN","APH","AVGO","LITE","WDC",
    "AMD","CIEN","AMKR","COHR","VRT","ARM","TSM","CGNX","IONQ","MTSI",
    "LRCX","ASX","JBL","ONTO","NOK","CVLT","QCOM","TWLO","ADI","NTAP",
    "CLS","TOST","MDB","LSCC","CRUS","CLSK","CIFR","PLXS","AUR","ZM",
    "SITM","ADP","MKSI","HIVE","SMCI","QBTS","UMC","AEVA",
        # Financials
        "JPM","V","MA","GS","MS","BLK","SCHW","AXP","C","BAC","WFC","USB",
        "PNC","TFC","COF","ICE","CME","SPGI","MCO","MSCI","FIS","FI","GPN",
        "AIG","MET","PRU","AFL","ALL","TRV","CB","AON","MMC","AJG","CINF","BRO",
        "TIGO","SMFG","NMR","NTRS","STT","VOYA","OPHC","SLF","PRI","SEIC","VIRT",
        "MFG","MUFG","HSBC","RY","BAP","UBS","SHG","KB","BPOP","FHI","GCMG","VCTR","IFS","PSO","QNST","RSI",
        "BNY","STT","IVZ","AMG","JXN","MFC","SLF","BMO","BNS","RY","TD","CM","ING","AEG","CFR","MTB","ZION","RF","FIBK","HWC","WTFC","BOH","CBU",
    "NBTB","BANR","NBHC","FBP","GNW","KRT","NREF","SEZL","PCB","SBFG","FFBC","CTBI","NEWT","PLBC","NTB","OPHC","NTRS","VCTR","GCMG","QNST",
    "NMR","UBS","MUFG","BRK/B","CFG","PNFP","NLY","EWBC","BEN","AIZ","MKL","AFRM","BGC",
    "RKT","ALLY","PIPR","HOOD","NDAQ","SAN","KEY","TROW","ERIE","COLB",
    "FCFS","MTG","BFH","CNS","CBOE","LYG","SNEX","UWMC","LPL",
        # Healthcare
        "UNH","LLY","JNJ","ABBV","MRK","PFE","TMO","ISRG","VRTX","REGN",
        "AMGN","GILD","MDT","SYK","BSX","EW","ZBH","BAX","BDX","HOLX","DXCM",
        "IDXX","MTD","A","WAT","IQV","CRL","TECH","ALGN","PODD","INCY",
        "RLAY","RVMD","DNTH","ACRS","TXG","SEPN","BLZE","TENX","FBRX","URGN",
        "PTGX","CRNX","MNPR","MIRM","IMVT","JAZZ","PSNL","DNLI","NKTX","PHVS","NRIX","PTCT","ICCC","SRRK","TGTX",
        "ABUS","NBIX","CORT","PRVA","ELV","VCEL","PNTG","WDFC","CHE","CVS","IART","GKOS","ACHC","PRVA","PGNY","ADPT","APGE","ELVN","PBYI",
    "ETON","MRVI","RLAY","RVMD","DNTH","ACRS","SEPN","BLZE","TENX","FBRX",
    "URGN","PTGX","CRNX","MNPR","MIRM","IMVT","JAZZ","PSNL","DNLI","NKTX",
    "PHVS","NRIX","PTCT","ICCC","SRRK","TGTX","ABUS","NBIX","CORT","VCEL",
    "PNTG","WDFC","CHE","MCK","BIIB","HIMS","ARGX","GSK","HRMY","DGX","NTRA","CAH","HALO",
    "TMDX","RARE","HQY",
        # Energy
        "XOM","CVX","COP","SLB","EOG","MPC","VLO","PSX","HES",
        "OXY","DVN","HAL","FANG","CTRA","APA","TRGP","WMB","OKE","KMI","POWW","DINO","PARR","PBA","GEV","ET","EPD","OVV","NOV","PBR","LNG","EQT","BKR","ENPH","SM",
    "WES","BE","AR","DTM","RUN","EOSE",
        # Defense & Aerospace
        "RTX","GD","BA","LHX","NOC","LMT","HII","TXT","HWM","TDG","AXON","HXL",
        # Industrials
        "HON","MMM","CMI","PH","ITW","TT","EMR","GE","ETN","ROK","AME",
        "DOV","FTV","XYL","NDSN","ROP","IEX","GWW","FAST","WSO","AOS",
        "IR","CARR","OTIS","JCI","GNRC","HUBB","RBC","SNA","WCC",
        "PENG","CVLG","EXPD","CSX","GTLS","BSET","WKC","SAH","PAG","UNP","UNF","EXPD","GTLS","BSET","SAH","PAG","RUSHA","CVLG","MATX","KFRC",
    "WKC","FA","GTY","TRNO","PDM","AAT","CUZ","HIW","STAG","IIPR","OTTR",
    "LQDT","WILC","SPB","XHR","DE","POWL","WM","LUV","ECHO","JBHT","ENS","DAL","VMI","EME","SWK",
    "MOG/A","MLI","QS","RRX","SPXC","FBIN","BMI","CHRW","OC","OII","AIR",
    "AAL","CR","APG","WTS","VRRM","NSP","FIX","OSK","XPO","R","SAIA",
        # Consumer Discretionary
        "COST","WMT","HD","LOW","NKE","SBUX","MCD","TJX","ROST","DG","DLTR",
        "BKNG","ABNB","MAR","HLT","RCL","CCL","LVS","WYNN","MGM",
        "F","GM","APTV","BWA","LEA","RL","TPR","GRMN","POOL","BBY","TSCO",
        "ORLY","AZO","AAP","KMX","LULU","DECK","ON","ULTA","EL","CPRI",
        "ARMK","JOYY","LTH","EAT","CUE","CAKE","CROX","ETSY","BJRI","CHEF","CROX","BBY","ABNB","BZH","RACD","CVNA","EXPE","TOL","QSR","PII","W","MHK","KTB","BOOT","DASH","M",
    "JD","VIPS","LEN/B",
        # Consumer Staples
        "PG","KO","PEP","PM","MO","STZ","BF-B","MNST","KDP","CLX",
        "CL","KMB","CHD","SJM","HSY","MKC","GIS","CAG","K","HRL","TSN","MDLZ","ATD","UL","FLO","SFM","PFGC","LW",
        # Real Estate
        "AMT","PLD","CCI","EQIX","PSA","SPG","O","DLR","VICI","WELL",
        "AVB","EQR","MAA","ESS","UDR","ARE","BXP","SLG","VNO","IIPR","VTR","HR","NXDT","HTO","PECO","BNL","FR","EPC","LAMR","COMP","CSGP","RYN","GLPI",
        # Utilities
        "NEE","DUK","SO","D","AEP","SRE","EXC","XEL","WEC","ES",
        "AEE","CMS","CNP","PNW","NI","EVRG","ATO","PEG","BNL","FR","PPL","ETR","CWEN","CEG","WTRG","UGI","VST","BIPC","BEPC",
        # Materials
        "LIN","APD","SHW","ECL","NUE","STLD","CF","MOS","ALB","FMC",
        "IFF","CE","PPG","VMC","MLM","NEM","FCX","AA","CPBI","KOP","ASH","GLW","SCCO","CRS","GPK","CBT","VALE","HL","NEU","SCL",
        # Communication Services
        "GOOG","DIS","NFLX","CMCSA","T","VZ","TMUS","CHTR","EA","TTWO",
        "MTCH","ZG","PINS","SNAP","ROKU","SPOT","WBD","PARA","LYV","DHX","JOYY","ANDG","MTCH","EA","JOYY","ANDG","DHX","ASTS","RDDT","LUMN","SE","TRI","FWONK","NTES"
    ],
}



def fetch_ticker(ticker, period="2y"):
    """Fetch 2 years of daily data for a ticker. Returns DataFrame or None."""
    if not HAS_YF:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, auto_adjust=True)
        if hist is None or len(hist) < 250:  # need at least 1 year
            return None
        return hist
    except Exception as e:
        print(f"  [{ticker}] fetch error: {e}")
        return None


def compute_indicators(hist):
    """
    Compute the minimal set of indicators SPE needs for regime detection:
      - ATR (14)
      - RSI (14) as last value
      - ADX proxy as last value
      - Relative volume vs 50-period average
    """
    h, l, c = hist["High"].values, hist["Low"].values, hist["Close"].values
    v = hist["Volume"].values

    # ATR — True Range rolling mean
    tr = np.zeros(len(c))
    tr[0] = h[0] - l[0]
    for i in range(1, len(c)):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    atr_series = pd.Series(tr, index=hist.index).rolling(14).mean().bfill()

    # RSI
    returns = np.diff(c)
    gains = np.where(returns > 0, returns, 0.0)
    losses = np.where(returns < 0, -returns, 0.0)
    avg_gain = pd.Series(gains).rolling(14).mean().iloc[-1] if len(gains) >= 14 else 0
    avg_loss = pd.Series(losses).rolling(14).mean().iloc[-1] if len(losses) >= 14 else 1e-9
    rsi = 100 - 100 / (1 + avg_gain / max(avg_loss, 1e-9))

    # ADX proxy — abs change / atr over last 30 days × 30
    if len(c) >= 30 and atr_series.iloc[-1] > 0:
        abs_change = np.mean(np.abs(np.diff(c[-30:])))
        adx = min(50, (abs_change / atr_series.iloc[-1]) * 30)
    else:
        adx = 20.0

    # Relative volume: last / mean of last 50
    if len(v) >= 50:
        rel_vol = v[-1] / np.mean(v[-50:])
    else:
        rel_vol = 1.0

    return {
        "atr": atr_series,
        "rsi": float(rsi),
        "adx": float(adx),
        "rel_vol": float(rel_vol),
    }


def process_ticker(ticker):
    """
    Full pipeline for one ticker:
      fetch → indicators → SPE projection → packaged dict.
    """
    t0 = time.time()
    hist = fetch_ticker(ticker)
    if hist is None:
        return None

    try:
        ind = compute_indicators(hist)
    except Exception as e:
        print(f"  [{ticker}] indicator error: {e}")
        return None

    try:
        sp = StochasticProjector(hist, ind)
        ens = sp.ensemble_projection(horizon=21)
        multi = sp.multi_horizon_projection()
    except Exception as e:
        print(f"  [{ticker}] SPE error: {e}")
        return None

    mc = ens["monte_carlo"]
    regime = ens["regime"]

    result = {
        "ticker": ticker,
        "close": round(float(hist["Close"].iloc[-1]), 4),
        "fetched_at": datetime.now(timezone.utc).isoformat(),

        # Technicals used by SPE
        "rsi": round(ind["rsi"], 2),
        "adx": round(ind["adx"], 2),
        "rel_vol": round(ind["rel_vol"], 2),
        "atr": round(float(ind["atr"].iloc[-1]), 4),

        # Headline SPE results (1 month default horizon)
        "target": ens["target"],
        "upside_pct": ens["upside_pct"],
        "confidence": ens["confidence"],
        "prob_up": ens["prob_up"],
        "prob_tp3": ens["prob_tp3"],
        "prob_tp5": ens["prob_tp5"],
        "prob_tp10": ens["prob_tp10"],
        "prob_drop_5": ens["prob_drop_5"],

        # Risk metrics
        "var_95": mc["var_95"],
        "cvar_95": mc["cvar_95"],
        "ci_68_low": mc["ci_68_low"],
        "ci_68_high": mc["ci_68_high"],
        "ci_95_low": mc["ci_95_low"],
        "ci_95_high": mc["ci_95_high"],

        # Diagnostics
        "volatility_annualized": mc["volatility_annualized"],
        "skewness": mc["skewness"],
        "kurtosis": mc["kurtosis"],
        "method": mc["method"],
        "vol_method": mc["vol_method"],
        "reliable": mc["reliable"],

        # Regime
        "regime": regime["regime"],
        "regime_desc": regime["description"],
        "atr_ratio": regime["atr_ratio"],

        # Multi-horizon projections
        "horizons": {
            k: {
                "days": v["days"],
                "expected": v["expected"],
                "upside_pct": v["upside_pct"],
                "prob_up": v["prob_up"],
                "ci_68_low": v["ci_68_low"],
                "ci_68_high": v["ci_68_high"],
            }
            for k, v in multi.items()
        },

        # Price history for mini-chart (last 60 days, closes only)
        "history_60d": [round(float(p), 4) for p in hist["Close"].values[-60:]],

        # Timing
        "process_ms": int((time.time() - t0) * 1000),
    }
    return result


def build_snapshot(tickers=None, groups=None, out_dir="data", verbose=True):
    """
    Main entry point. Fetches all tickers, runs SPE, saves snapshot.json.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Determine target tickers
    if groups is None:
        groups = TICKER_GROUPS

    if tickers is not None:
        # Filter to just the given list — put them all under one group
        groups = {"Custom": list(tickers)}

    total = sum(len(v) for v in groups.values())
    print(f"Building snapshot for {total} tickers across {len(groups)} groups\n")

    results = {"built_at": datetime.now(timezone.utc).isoformat(),
               "version": "1.0",
               "groups": {},
               "n_tickers": 0,
               "n_reliable": 0,
               "n_failed": 0}

    idx = 0
    for group_name, tickers_in_group in groups.items():
        if verbose:
            print(f"═══ {group_name} ═══")
        group_results = []
        for ticker in tickers_in_group:
            idx += 1
            if verbose:
                print(f"  [{idx}/{total}] {ticker}...", end=" ", flush=True)
            r = process_ticker(ticker)
            if r is None:
                results["n_failed"] += 1
                if verbose:
                    print("FAILED")
                continue
            group_results.append(r)
            results["n_tickers"] += 1
            if r["reliable"]:
                results["n_reliable"] += 1
            if verbose:
                flag = "★" if r["reliable"] else "⚠"
                print(f"{flag} prob_up={r['prob_up']:.0f}% "
                      f"upside={r['upside_pct']:+.1f}% "
                      f"regime={r['regime']} ({r['process_ms']}ms)")

        results["groups"][group_name] = group_results

    # Save snapshot
    output_file = out_path / "snapshot.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print(f"\n{'═'*60}")
    print(f"SUMMARY")
    print(f"  Tickers processed : {results['n_tickers']} / {total}")
    print(f"  Reliable (≥60d)   : {results['n_reliable']}")
    print(f"  Failed            : {results['n_failed']}")
    print(f"  Output            : {output_file}")
    print(f"  File size         : {os.path.getsize(output_file)/1024:.1f} KB")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Stochastic Scanner Pro — Build snapshot.json from Yahoo data")
    parser.add_argument("--tickers", default=None,
                        help="Comma-separated tickers (overrides groups)")
    parser.add_argument("--out-dir", default="data",
                        help="Output directory (default: data)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-ticker output")
    args = parser.parse_args()

    if not HAS_YF:
        print("ERROR: yfinance not installed.")
        print("Install with: pip install yfinance")
        sys.exit(1)

    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]

    build_snapshot(tickers=tickers, out_dir=args.out_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()
