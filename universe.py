"""Decides which NSE symbols are ETFs / index funds / bonds-funds so they can be kept out of the stock universe.

NSE's daily price file has no instrument type, so this uses (1) naming patterns that ETFs follow and
(2) a short list of real companies whose names merely look like ETFs.  Plain-language rule: if the
symbol looks like a fund name and is not on the KEEP list, it is treated as a fund and left out.
"""
import re

KEEP = {  # real companies whose symbol contains fund-like words
    "JETFREIGHT", "GROWW", "GOLDIAM", "DECNGOLD", "SHANTIGOLD", "SKYGOLD", "GOLDENTOBC", "GOLDTECH",
    "ENERGYDEV", "MIDHANI", "MIDWESTLTD", "MONARCH", "MONEYBOXX", "MONEYVIEW", "MONSANTO", "MONTECARLO",
    "NAM-INDIA", "BSL", "ABSLAMC", "ALPHAGEO", "AONESTEELS", "AONETOTAL", "TATASPONGE", "SBIFUNDS",
    "HDFCBANK", "ICICIBANK", "HDFCAMC", "SENSEXADD",
}
_PAT = re.compile(
    r"(BEES$|ETF|IETF|BETF$|ADD$|BETA$|CASE$|LIQ|^NETF|^NIFTY|^BANKNIFTY|^SENSEX|^NV20|^N100$|^NEXT50"
    r"|^MID150|^MOM(100|50|30|NC|GF|ETAL|IDMTM|MIDCAP|OMENTUM|SBELIEF|ENTUM)|^GROWW|GOLD|SILVER|SILVR|^GILT|GSEC|SDL\d"
    r"|^EBBETF|^SETF|^LICNETF|^HDFC(NIFTY|SML|BSE|SEN)|^ICICI(NIFTY|NV20|B22|IT|TECH|PHARM|SILVE|BANKN?$)|^ICICIB22"
    r"|^UTI(NIFT|BANK|SENS)|^SBI(NIFTY|BANKNIF|NEXT|ETF|VALETF|SMLETF|NEQW|DEF)|^KOTAK(NIFTY|BKETF)|^AXIS(NIFTY|BNK|HC|TEC|BPS|CETF)"
    r"|^AONE(NIFTY|LIQ|GOLD|SILVER|TMMQ)|^DSP|^MAFANG|^MON(100|50EQUAL|EXT50|IFTY|Q50)|^HNGSNG|^LOWVOL|^QUAL(ITY)?\d*$|^IVALUE"
    r"|^ALPHA(ETF)?$|^SMALLCAP$|^MIDCAP$|^MIDSMALL$|^MULTICAP$|^ENERGY$|^INFRA$|^IT$|^PSUBANK|^BETA$|^BANKADD|^FARBEES|^MAFSETF)"
)

def is_etf(sym: str) -> bool:
    s = str(sym).upper()
    if s in KEEP:
        return False
    return bool(_PAT.search(s))
