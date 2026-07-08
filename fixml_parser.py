#!/usr/bin/env python3
"""
fixml_parser.py — MarkitWire FIXML → AAL ExpectedEconomics dict

Converts a raw MarkitWire FIXML trade capture report (TrdCaptRpt) into
the normalized dict that ExpectedEconomics.from_dict() and IRSConfirmation.from_dict()
expect. Deterministic — no LLM required.

Supported asset classes (auto-detected from FIXML):
  - IRS (SecTyp=MLEG or Sym containing LIBOR/SOFR/SONIA/ESTR/OIS)
  - FIXED_INCOME (SecTyp=CORP/MUN/AGENCY/ABS/MBS/UST)
  - CDS (SecTyp=CDS)
  - EQUITY_OPTION (SecTyp=OPT)
  - CAP_FLOOR (SecTyp=MLEG + structure cap/floor/collar)
  - TRS (SecTyp=MLEG + TRS marker)

Usage
-----
    from fixml_parser import parse_fixml

    with open("markitwire_confirm.xml") as f:
        xml_str = f.read()

    result = parse_fixml(xml_str)
    # result = {
    #   "asset_class": "IRS",
    #   "counterparty": "FIRM_SELL",
    #   "trade_date": "2026-07-07",
    #   "notional": 1000000.0,
    #   "currency": "USD",
    #   ...
    # }

Standalone test
---------------
    python fixml_parser.py sample.xml
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Namespace handling — MarkitWire uses fixml-5.0 namespace variants
# ---------------------------------------------------------------------------
FIXML_NS_PATTERNS = [
    "http://fixprotocol.org",
    "http://www.fixprotocol.org",
    "http://www.fixprotocol.org/FIXML-5-0-SP2",
    "http://www.fixprotocol.org/FIXML-4-4",
]

def _detect_ns(root: ET.Element) -> str:
    """Detect FIXML namespace from root tag."""
    tag = root.tag
    if tag.startswith("{"):
        return tag[1:tag.index("}")]
    return ""

def _find(elem: ET.Element, path: str, ns: str) -> Optional[ET.Element]:
    """Namespace-aware element find."""
    if ns:
        parts = path.split("/")
        ns_path = "/".join(f"{{{ns}}}{p}" for p in parts)
        return elem.find(ns_path)
    return elem.find(path)

def _findall(elem: ET.Element, path: str, ns: str) -> list[ET.Element]:
    if ns:
        parts = path.split("/")
        ns_path = "/".join(f"{{{ns}}}{p}" for p in parts)
        return elem.findall(ns_path)
    return elem.findall(path)

def _attr(elem: Optional[ET.Element], attr: str, default=None):
    if elem is None:
        return default
    return elem.get(attr, default)


# ---------------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------------
DATE_FMTS = ["%Y-%m-%d", "%Y%m%d", "%m/%d/%Y", "%d/%m/%Y"]

def _norm_date(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    val = val.strip().split("T")[0]  # strip time component
    for fmt in DATE_FMTS:
        try:
            return datetime.strptime(val, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return val  # return as-is if unparseable


# ---------------------------------------------------------------------------
# Float rate normalization (mirrors FLOAT_ALIASES in main.py)
# ---------------------------------------------------------------------------
FLOAT_ALIASES = {
    "usd-sofr": "SOFR", "sofr compound": "SOFR", "sofr-compound": "SOFR",
    "sofr ois": "SOFR", "us sofr": "SOFR", "sofr": "SOFR",
    "usd-libor-bba": "LIBOR-3M", "usd-libor-bba-3m": "LIBOR-3M",
    "usd-libor-bba-6m": "LIBOR-6M", "usd-libor-bba-1m": "LIBOR-1M",
    "usd-libor": "LIBOR-3M", "libor": "LIBOR-3M", "libor-bba": "LIBOR-3M",
    "usd-libor-3m": "LIBOR-3M",
    "eur-euribor-reuters": "EURIBOR-6M", "euribor": "EURIBOR-6M",
    "gbp-sonia": "SONIA", "sonia": "SONIA",
    "eur-estr": "ESTR", "estr": "ESTR",
    "usd-federal funds-ois compound": "OIS", "ois": "OIS", "fed funds": "OIS",
}

def _norm_float_rate(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    return FLOAT_ALIASES.get(val.strip().lower(), val.strip())


# ---------------------------------------------------------------------------
# Asset class detection from FIXML instrument fields
# ---------------------------------------------------------------------------
def _detect_asset_class(sym: str, sec_type: str) -> str:
    sym_upper = (sym or "").upper()
    sec_type_upper = (sec_type or "").upper()

    if sec_type_upper == "CDS":
        return "CDS"
    if sec_type_upper == "OPT":
        return "EQUITY_OPTION"
    if sec_type_upper in ("CORP", "MUN", "AGENCY", "ABS", "MBS", "UST", "TBOND", "TBILL", "TNOTE"):
        return "FIXED_INCOME"
    if sec_type_upper in ("FUT", "MLEG", "SWAP"):
        # Differentiate IRS / CAP_FLOOR / TRS by symbol
        if any(x in sym_upper for x in ("CAP", "FLOOR", "COLLAR")):
            return "CAP_FLOOR"
        if "TRS" in sym_upper or "TOTAL RETURN" in sym_upper:
            return "TRS"
        # Default swap to IRS
        return "IRS"
    # Fallback: check symbol for rate index markers
    if any(x in sym_upper for x in ("LIBOR", "SOFR", "SONIA", "ESTR", "EURIBOR", "OIS")):
        return "IRS"

    return "IRS"  # safest default


# ---------------------------------------------------------------------------
# Party role codes → semantic labels
# Role R values: 1=Executing Firm, 2=Broker, 4=Clearing Firm, 17=Beneficiary
# ---------------------------------------------------------------------------
def _extract_parties(trd_cap: ET.Element, ns: str) -> dict:
    parties = {}
    for pty in _findall(trd_cap, "Pty", ns):
        role = pty.get("R", "")
        pid = pty.get("ID", "")
        if role == "1":
            parties["executing_firm"] = pid
        elif role == "2":
            parties["broker"] = pid
        elif role == "4":
            parties["clearing_firm"] = pid
        elif role == "17":
            parties["beneficiary"] = pid
    return parties


def _extract_sides(trd_cap: ET.Element, ns: str) -> list[dict]:
    sides = []
    for side in _findall(trd_cap, "Side", ns):
        entry = {
            "direction": "buy" if side.get("BuyDrv") == "1" else "sell",
            "account": side.get("Acct"),
        }
        # Side-level parties
        for pty in _findall(side, "Pty", ns):
            if pty.get("R") == "1":
                entry["trader"] = pty.get("ID")
        sides.append(entry)
    return sides


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------
def parse_fixml(xml_str: str) -> dict:
    """
    Parse a MarkitWire FIXML string and return a normalized dict
    compatible with AAL ExpectedEconomics / IRSConfirmation.from_dict().

    Returns a dict with keys matching the ExpectedEconomics Pydantic model.
    Always includes 'asset_class'.
    """
    try:
        root = ET.fromstring(xml_str.strip())
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML: {e}")

    ns = _detect_ns(root)

    # TrdCaptRpt is either the root or a child
    trd_rpt = root if root.tag.endswith("TrdCaptRpt") else _find(root, "TrdCaptRpt", ns)
    if trd_rpt is None:
        raise ValueError("No TrdCaptRpt element found in FIXML")

    trd_cap = _find(trd_rpt, "TrdCap", ns)
    if trd_cap is None:
        raise ValueError("No TrdCap element found in TrdCaptRpt")

    # -----------------------------------------------------------------------
    # Instrument
    # -----------------------------------------------------------------------
    instrmt = _find(trd_cap, "Instrmt", ns)
    sym      = _attr(instrmt, "Sym", "")
    sec_type = _attr(instrmt, "SecTyp", "")
    currency = _attr(instrmt, "Ccy") or _attr(trd_cap, "Ccy")
    maturity = _norm_date(_attr(instrmt, "MatDt") or _attr(instrmt, "MMY"))

    asset_class = _detect_asset_class(sym, sec_type)

    # -----------------------------------------------------------------------
    # Trade-level fields
    # -----------------------------------------------------------------------
    report_id = _attr(trd_rpt, "RptID") or _attr(_find(trd_cap, "RptID", ns), "ID")
    trade_date = _norm_date(_attr(trd_cap, "TrdDt"))
    effective_date = _norm_date(_attr(trd_cap, "BizDt") or _attr(trd_cap, "EffDt"))
    notional_raw = _attr(trd_cap, "LastQty") or _attr(trd_cap, "Qty")
    notional = float(notional_raw) if notional_raw else None
    price_raw = _attr(trd_cap, "LastPx") or _attr(trd_cap, "Px")
    price = float(price_raw) if price_raw else None

    # UTI / USI
    reg_id_elem = _find(trd_cap, "RegTrdID", ns)
    usi = _attr(reg_id_elem, "ID")

    # -----------------------------------------------------------------------
    # Parties
    # -----------------------------------------------------------------------
    parties = _extract_parties(trd_cap, ns)
    sides   = _extract_sides(trd_cap, ns)

    # Counterparty: prefer broker (role 2), else executing firm (role 1)
    counterparty = parties.get("broker") or parties.get("executing_firm")

    # Account: from sell side (F&G is typically the sell/receiver side)
    account = None
    buyer = seller = None
    for s in sides:
        if s["direction"] == "sell":
            account = s.get("account")
            seller = s.get("trader")
        elif s["direction"] == "buy":
            buyer = s.get("trader")

    # -----------------------------------------------------------------------
    # Floating rate: derive from symbol for IRS/CAP_FLOOR/TRS
    # -----------------------------------------------------------------------
    floating_rate = None
    if asset_class in ("IRS", "CAP_FLOOR", "TRS"):
        floating_rate = _norm_float_rate(sym)

    # -----------------------------------------------------------------------
    # Fixed rate: LastPx is often the fixed rate for IRS in MarkitWire
    # expressed as decimal (0.0150) → convert to %
    # -----------------------------------------------------------------------
    fixed_rate = None
    if asset_class == "IRS" and price is not None:
        # MarkitWire sends as decimal if < 1, as % if >= 1
        fixed_rate = price if price >= 1.0 else round(price * 100, 6)

    # -----------------------------------------------------------------------
    # Assemble output dict — keys match ExpectedEconomics Pydantic model
    # -----------------------------------------------------------------------
    out: dict = {
        "asset_class": asset_class,
        "trade_date":  trade_date,
        "currency":    currency,
        "notional":    notional,
    }

    if counterparty:
        out["counterparty"] = counterparty
    if effective_date:
        out["effective_date"] = effective_date
    if maturity:
        out["maturity_date"] = maturity
    if usi:
        out["usi"] = usi
    if report_id:
        out["trade_id"] = report_id
    if account:
        out["account"] = account

    # IRS-specific
    if asset_class == "IRS":
        if fixed_rate is not None:
            out["fixed_rate"] = fixed_rate
        if floating_rate:
            out["floating_rate"] = floating_rate

    # EQUITY_OPTION-specific
    if asset_class == "EQUITY_OPTION":
        out["underlying"] = sym
        if price is not None:
            out["strike"] = price
        if buyer:
            out["buyer"] = buyer
        if seller:
            out["seller"] = seller

    # FIXED_INCOME-specific
    if asset_class == "FIXED_INCOME":
        cusip = _attr(instrmt, "ID")
        if cusip:
            out["cusip"] = cusip.upper().replace(" ", "")
        if price is not None:
            out["price"] = price
        # Direction from sides
        for s in sides:
            out["direction"] = s["direction"]
            break

    # CDS-specific
    if asset_class == "CDS":
        out["reference_entity"] = sym
        if price is not None:
            # CDS spread — MarkitWire sends as decimal
            out["fixed_rate"] = price if price >= 1.0 else round(price * 100, 4)

    # CAP_FLOOR-specific
    if asset_class == "CAP_FLOOR":
        if floating_rate:
            out["floating_rate"] = floating_rate
        # structure type from symbol
        sym_lower = sym.lower()
        if "collar" in sym_lower:
            out["structure_type"] = "collar"
        elif "floor" in sym_lower:
            out["structure_type"] = "floor"
        else:
            out["structure_type"] = "cap"
        if price is not None:
            out["cap_rate"] = price if price >= 1.0 else round(price * 100, 4)

    return out


# ---------------------------------------------------------------------------
# Standalone CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            xml_str = f.read()
    else:
        # Inline test with the F&G MarkitWire sample
        xml_str = """<?xml version="1.0" encoding="UTF-8"?>
<FIXML v="5.0 SP2"
       xmlns="http://fixprotocol.org"
       xmlns:xsi="http://w3.org"
       xsi:schemaLocation="http://fixprotocol.org fixml-impl-5-0-SP2.xsd">
  <TrdCaptRpt RptID="MW-12345678" TransTyp="0" RptTyp="2" TrdStat="0">
    <Hdr Snt="2026-07-07T15:03:00-05:00" MsgID="MSG-001"/>
    <TrdCap PxTyp="1" LastQty="1000000" LastPx="1.05" TrdDt="2026-07-07"
            BizDt="2026-07-07" TxnTm="2026-07-07T15:03:00-05:00">
      <RptID ID="MW-12345678"/>
      <RegTrdID ID="UTI-ABCDE12345" Src="1010000023"/>
      <Instrmt Sym="USD-LIBOR-3M" SecTyp="FUT" MMY="202609" Ccy="USD"/>
      <Pty ID="FIRM_BUY" R="1" />
      <Pty ID="FIRM_SELL" R="2" />
      <Pty ID="CLEAR_HOUSE_1" R="4" />
      <Side BuyDrv="1" Acct="ACCOUNT_01">
        <Pty ID="TRADER_A" R="1"/>
      </Side>
      <Side BuyDrv="2" Acct="ACCOUNT_02">
        <Pty ID="TRADER_B" R="1"/>
      </Side>
    </TrdCap>
  </TrdCaptRpt>
</FIXML>"""

    result = parse_fixml(xml_str)
    print(json.dumps(result, indent=2))
