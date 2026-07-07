#!/usr/bin/env python3
"""
engine/trs_reconciler.py — AAL Discrepancy Detection API v2
Deterministic reconciliation engine for Total Return Swap (TRS) confirmations.

v1.1 additions (from Standard Chartered IDR zero coupon bond TRS template):
  - ref_isin takes priority over ref_cusip for ISIN-identified securities
  - ref_security_type: "Zero Coupon Bond" | "Fixed Rate Bond" etc
  - local_notional + local_currency: dual-currency notional (e.g. IDR/USD)
  - initial_spot_rate: FX rate at inception (local per 1 USD)
  - current_market_price: reference obligation price % at effective date
  - seniority: "Senior" | "Subordinated"
  - spot_rate_tolerance: 0.0001 (1 pip on IDR/USD scale)

Covers:
  - Rates TRS (UST, agency, corporate bond reference)
  - Cross-currency sovereign TRS (IDR zero coupon, SCB structure)
  - Equity TRS (reference asset = equity index or single name)
  - Credit TRS (reference asset = loan or bond)

Reference documents:
  - Wells Fargo Bank NA TRS template (2021 ISDA Definitions, rates TRS)
  - Standard Chartered Bank IDR Zero Coupon Bond TRS (2006/2014 ISDA)

Architecture: same hybrid pipeline. LLM extracts, code reconciles.
Benchmark: AAL-D-005 (planned)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Exception categories
# ---------------------------------------------------------------------------
class EXC:
    CPTY       = "EXC-CPTY"
    REF_ASSET  = "EXC-CUSIP"
    ISSUER     = "EXC-CPTY"
    COUPON     = "EXC-COUP"
    REF_MAT    = "EXC-MAT"
    INIT_PRICE = "EXC-PRICE"
    SPOT_RATE  = "EXC-FX"
    NOTIONAL   = "EXC-QTY"
    CCY        = "EXC-CCY"
    FLOAT      = "EXC-PROD"
    SPREAD     = "EXC-PRICE"
    FIXED_RATE = "EXC-PRICE"
    DAY_COUNT  = "EXC-PROD"
    EFF_DATE   = "EXC-SDATE"
    TERM_DATE  = "EXC-SDATE"
    FREQ       = "EXC-PROD"
    DIR        = "EXC-DIR"
    SENIORITY  = "EXC-PROD"
    CLEAN      = None

SEVERITY_TABLE: dict[str, str] = {
    EXC.CPTY:       "high",
    EXC.REF_ASSET:  "high",
    EXC.COUPON:     "high",
    EXC.REF_MAT:    "high",
    EXC.INIT_PRICE: "high",
    EXC.SPOT_RATE:  "high",
    EXC.NOTIONAL:   "high",
    EXC.CCY:        "high",
    EXC.EFF_DATE:   "high",
    EXC.TERM_DATE:  "high",
    EXC.DIR:        "high",
    EXC.FLOAT:      "medium",
    EXC.SPREAD:     "medium",
    EXC.FIXED_RATE: "medium",
    EXC.DAY_COUNT:  "medium",
    EXC.SENIORITY:  "medium",
    EXC.FREQ:       "low",
}

def severity_for(category: str) -> str:
    return SEVERITY_TABLE.get(category, "medium")


# ---------------------------------------------------------------------------
# Floating rate normalization
# ---------------------------------------------------------------------------
FLOAT_ALIASES: dict[str, str] = {
    "usd-federal funds-ois compound": "OIS",
    "fed funds": "OIS", "ois": "OIS", "usd-ois": "OIS",
    "federal funds": "OIS", "sofr": "SOFR",
    "usd-sofr": "SOFR", "sofr compound": "SOFR",
    "usd-libor-bba": "LIBOR-3M", "libor": "LIBOR-3M",
    "usd-libor-bba-3m": "LIBOR-3M", "usd-libor-bba-6m": "LIBOR-6M",
    "euribor": "EURIBOR-6M", "sonia": "SONIA",
    "estr": "ESTR", "eur-estr": "ESTR",
}

def normalize_float(rate: Optional[str]) -> Optional[str]:
    if not rate:
        return rate
    return FLOAT_ALIASES.get(rate.strip().lower(), rate)


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------
def _parse_date(v) -> Optional[date]:
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(v.strip(), fmt).date()
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class TRSConfirmation:
    trade_id:             Optional[str]   = None
    trade_date:           Optional[str]   = None
    counterparty:         Optional[str]   = None
    usi:                  Optional[str]   = None
    # Reference asset
    ref_cusip:            Optional[str]   = None
    ref_isin:             Optional[str]   = None
    ref_issuer:           Optional[str]   = None
    ref_security_type:    Optional[str]   = None
    ref_coupon:           Optional[float] = None
    ref_maturity:         Optional[str]   = None
    initial_price:        Optional[float] = None
    seniority:            Optional[str]   = None
    # Economics — dual currency support
    notional:             Optional[float] = None   # USD notional
    local_notional:       Optional[float] = None   # local currency notional
    local_currency:       Optional[str]   = None   # e.g. "IDR"
    initial_spot_rate:    Optional[float] = None   # local per 1 USD
    current_market_price: Optional[float] = None   # reference asset price % at effective date
    currency:             Optional[str]   = None
    direction:            Optional[str]   = None
    # Funding leg
    leg_type:             Optional[str]   = None
    floating_rate:        Optional[str]   = None
    spread:               Optional[float] = None
    fixed_rate:           Optional[float] = None
    day_count:            Optional[str]   = None
    payment_frequency:    Optional[str]   = None
    # Dates
    effective_date:       Optional[str]   = None
    termination_date:     Optional[str]   = None
    valuation_frequency:  Optional[str]   = None

    @classmethod
    def from_dict(cls, d: dict) -> "TRSConfirmation":
        fields = [
            "trade_id", "trade_date", "counterparty", "usi",
            "ref_cusip", "ref_isin", "ref_issuer", "ref_security_type",
            "ref_coupon", "ref_maturity", "initial_price", "seniority",
            "notional", "local_notional", "local_currency", "initial_spot_rate",
            "current_market_price", "currency", "direction", "leg_type",
            "floating_rate", "spread", "fixed_rate", "day_count",
            "payment_frequency", "effective_date", "termination_date",
            "valuation_frequency",
        ]
        obj = cls(**{f: d[f] for f in fields if f in d})
        if obj.floating_rate:
            obj.floating_rate = normalize_float(str(obj.floating_rate))
        if obj.direction:
            obj.direction = str(obj.direction).strip().lower()
        if obj.leg_type:
            obj.leg_type = str(obj.leg_type).strip().lower()
        if obj.seniority:
            obj.seniority = str(obj.seniority).strip().lower()
        if obj.ref_cusip:
            obj.ref_cusip = str(obj.ref_cusip).strip().upper().replace(" ", "")
        if obj.ref_isin:
            obj.ref_isin = str(obj.ref_isin).strip().upper().replace(" ", "")
        if obj.ref_issuer:
            obj.ref_issuer = str(obj.ref_issuer)
        if obj.counterparty:
            obj.counterparty = str(obj.counterparty)
        if obj.currency:
            obj.currency = str(obj.currency).strip().upper()
        if obj.local_currency:
            obj.local_currency = str(obj.local_currency).strip().upper()
        if obj.day_count:
            obj.day_count = str(obj.day_count)
        if obj.valuation_frequency:
            obj.valuation_frequency = str(obj.valuation_frequency)
        return obj


@dataclass
class Discrepancy:
    category:           str
    field:              str
    counterparty_value: object
    internal_value:     object
    difference:         object
    difference_unit:    str
    exposure_usd:       Optional[float]
    severity:           str
    confidence:         float = 1.0


@dataclass
class ReconciliationResult:
    case_id:             Optional[str]
    exception_exists:    bool
    primary:             Optional[Discrepancy]
    secondary:           Optional[Discrepancy]
    overall_severity:    str
    escalation_required: bool
    recommended_action:  str
    processing_notes:    list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------
class TRSReconciler:
    COUPON_TOL    = 0.001
    PRICE_TOL     = 0.01
    NOTIONAL_TOL  = 1_000
    SPREAD_TOL    = 0.001
    RATE_TOL      = 0.001
    SPOT_RATE_TOL = 0.0001  # 1 pip tolerance on FX spot rate

    def reconcile(
        self,
        confirmation: TRSConfirmation | dict,
        internal:     TRSConfirmation | dict,
        case_id:      Optional[str] = None,
    ) -> ReconciliationResult:

        if isinstance(confirmation, dict):
            confirmation = TRSConfirmation.from_dict(confirmation)
        if isinstance(internal, dict):
            internal = TRSConfirmation.from_dict(internal)

        found: list[Discrepancy] = []
        notes: list[str] = []

        for check in [
            self._check_counterparty,
            self._check_ref_asset,
            self._check_ref_issuer,
            self._check_ref_coupon,
            self._check_ref_maturity,
            self._check_initial_price,
            self._check_spot_rate,
            self._check_notional,
            self._check_currency,
            self._check_direction,
            self._check_seniority,
            self._check_floating,
            self._check_spread,
            self._check_fixed_rate,
            self._check_day_count,
            self._check_effective_date,
            self._check_termination_date,
            self._check_valuation_frequency,
        ]:
            d = check(confirmation, internal)
            if d:
                found.append(d)

        primary   = found[0] if found else None
        secondary = found[1] if len(found) > 1 else None
        if len(found) > 2:
            notes.append(f"{len(found)-2} additional discrepancy(ies) found but not scored")

        return ReconciliationResult(
            case_id=case_id,
            exception_exists=primary is not None,
            primary=primary,
            secondary=secondary,
            overall_severity=self._overall_severity(primary, secondary),
            escalation_required=self._escalation_required(primary, secondary),
            recommended_action=self._recommended_action(primary, secondary, confirmation),
            processing_notes=notes,
        )

    # -------------------------------------------------------------------------
    # Checkers
    # -------------------------------------------------------------------------

    def _check_counterparty(self, c, i) -> Optional[Discrepancy]:
        if not (c.counterparty and i.counterparty):
            return None
        if c.counterparty.strip().lower() != i.counterparty.strip().lower():
            return Discrepancy(
                category=EXC.CPTY, field="counterparty",
                counterparty_value=c.counterparty, internal_value=i.counterparty,
                difference="Different legal entity", difference_unit="entity",
                exposure_usd=None, severity="high",
            )
        return None

    def _check_ref_asset(self, c, i) -> Optional[Discrepancy]:
        # ISIN takes priority for sovereign/structured securities
        cv = (c.ref_isin or c.ref_cusip or "").upper().replace(" ", "")
        iv = (i.ref_isin or i.ref_cusip or "").upper().replace(" ", "")
        if not (cv and iv):
            return None
        if cv != iv:
            return Discrepancy(
                category=EXC.REF_ASSET, field="ref_isin" if (c.ref_isin or i.ref_isin) else "ref_cusip",
                counterparty_value=cv, internal_value=iv,
                difference="Reference asset identifier mismatch",
                difference_unit="identifier",
                exposure_usd=i.notional, severity="high",
            )
        return None

    def _check_ref_issuer(self, c, i) -> Optional[Discrepancy]:
        if not (c.ref_issuer and i.ref_issuer):
            return None
        if c.ref_issuer.strip().lower() != i.ref_issuer.strip().lower():
            return Discrepancy(
                category=EXC.ISSUER, field="ref_issuer",
                counterparty_value=c.ref_issuer, internal_value=i.ref_issuer,
                difference="Reference asset issuer mismatch",
                difference_unit="entity",
                exposure_usd=None, severity="high",
            )
        return None

    def _check_ref_coupon(self, c, i) -> Optional[Discrepancy]:
        if c.ref_coupon is None or i.ref_coupon is None:
            return None
        diff = abs(c.ref_coupon - i.ref_coupon)
        if diff > self.COUPON_TOL:
            exposure = (i.notional or 0) * diff / 100
            return Discrepancy(
                category=EXC.COUPON, field="ref_coupon",
                counterparty_value=c.ref_coupon, internal_value=i.ref_coupon,
                difference=round(c.ref_coupon - i.ref_coupon, 4),
                difference_unit="percent",
                exposure_usd=round(exposure, 2) if exposure else None,
                severity="high",
            )
        return None

    def _check_ref_maturity(self, c, i) -> Optional[Discrepancy]:
        if not (c.ref_maturity and i.ref_maturity):
            return None
        cd, id_ = _parse_date(c.ref_maturity), _parse_date(i.ref_maturity)
        if cd and id_ and cd != id_:
            return Discrepancy(
                category=EXC.REF_MAT, field="ref_maturity",
                counterparty_value=str(c.ref_maturity),
                internal_value=str(i.ref_maturity),
                difference=f"{abs((cd - id_).days)} day(s)",
                difference_unit="days",
                exposure_usd=None, severity="high",
            )
        return None

    def _check_initial_price(self, c, i) -> Optional[Discrepancy]:
        if c.initial_price is None or i.initial_price is None:
            return None
        diff = abs(c.initial_price - i.initial_price)
        if diff > self.PRICE_TOL:
            exposure = (i.notional or 0) * diff / 100
            return Discrepancy(
                category=EXC.INIT_PRICE, field="initial_price",
                counterparty_value=c.initial_price, internal_value=i.initial_price,
                difference=round(c.initial_price - i.initial_price, 4),
                difference_unit="per_100",
                exposure_usd=round(exposure, 2) if exposure else None,
                severity="high",
            )
        return None

    def _check_spot_rate(self, c, i) -> Optional[Discrepancy]:
        """Initial FX spot rate — critical for cross-currency TRS (IDR/USD, etc)."""
        if c.initial_spot_rate is None or i.initial_spot_rate is None:
            return None
        # Use percentage tolerance relative to rate magnitude
        tol = max(self.SPOT_RATE_TOL, i.initial_spot_rate * 0.0001)
        diff = abs(c.initial_spot_rate - i.initial_spot_rate)
        if diff > tol:
            # Exposure: difference in USD notional implied by spot rate difference
            local = i.local_notional or c.local_notional or 0
            exposure = local * diff / (i.initial_spot_rate ** 2) if i.initial_spot_rate else None
            return Discrepancy(
                category=EXC.SPOT_RATE, field="initial_spot_rate",
                counterparty_value=c.initial_spot_rate, internal_value=i.initial_spot_rate,
                difference=round(c.initial_spot_rate - i.initial_spot_rate, 6),
                difference_unit="fx_rate",
                exposure_usd=round(exposure, 2) if exposure else None,
                severity="high",
            )
        return None

    def _check_notional(self, c, i) -> Optional[Discrepancy]:
        if c.notional is None or i.notional is None:
            return None
        diff = abs(c.notional - i.notional)
        if diff > self.NOTIONAL_TOL:
            return Discrepancy(
                category=EXC.NOTIONAL, field="notional",
                counterparty_value=c.notional, internal_value=i.notional,
                difference=c.notional - i.notional,
                difference_unit="usd",
                exposure_usd=diff, severity="high",
            )
        return None

    def _check_currency(self, c, i) -> Optional[Discrepancy]:
        if not (c.currency and i.currency):
            return None
        if c.currency.strip().upper() != i.currency.strip().upper():
            return Discrepancy(
                category=EXC.CCY, field="currency",
                counterparty_value=c.currency, internal_value=i.currency,
                difference="Currency mismatch", difference_unit="currency",
                exposure_usd=None, severity="high",
            )
        return None

    def _check_direction(self, c, i) -> Optional[Discrepancy]:
        if not (c.direction and i.direction):
            return None
        if c.direction.lower() != i.direction.lower():
            return Discrepancy(
                category=EXC.DIR, field="direction",
                counterparty_value=c.direction, internal_value=i.direction,
                difference="TRS receiver vs payer mismatch",
                difference_unit="direction",
                exposure_usd=i.notional, severity="high",
            )
        return None

    def _check_seniority(self, c, i) -> Optional[Discrepancy]:
        if not (c.seniority and i.seniority):
            return None
        if c.seniority.strip().lower() != i.seniority.strip().lower():
            return Discrepancy(
                category=EXC.SENIORITY, field="seniority",
                counterparty_value=c.seniority, internal_value=i.seniority,
                difference="Seniority level mismatch",
                difference_unit="seniority",
                exposure_usd=None, severity="medium",
            )
        return None

    def _check_floating(self, c, i) -> Optional[Discrepancy]:
        cv = normalize_float(c.floating_rate)
        iv = normalize_float(i.floating_rate)
        if not (cv and iv):
            return None
        if cv != iv:
            return Discrepancy(
                category=EXC.FLOAT, field="floating_rate",
                counterparty_value=cv, internal_value=iv,
                difference="Floating index mismatch",
                difference_unit="index",
                exposure_usd=None, severity="medium",
            )
        return None

    def _check_spread(self, c, i) -> Optional[Discrepancy]:
        if c.spread is None or i.spread is None:
            return None
        diff = abs(c.spread - i.spread)
        if diff > self.SPREAD_TOL:
            exposure = (i.notional or 0) * diff / 100
            return Discrepancy(
                category=EXC.SPREAD, field="spread",
                counterparty_value=c.spread, internal_value=i.spread,
                difference=round(c.spread - i.spread, 4),
                difference_unit="percent",
                exposure_usd=round(exposure, 2) if exposure else None,
                severity="medium",
            )
        return None

    def _check_fixed_rate(self, c, i) -> Optional[Discrepancy]:
        if c.fixed_rate is None or i.fixed_rate is None:
            return None
        diff = abs(c.fixed_rate - i.fixed_rate)
        if diff > self.RATE_TOL:
            exposure = (i.notional or 0) * diff / 100
            return Discrepancy(
                category=EXC.FIXED_RATE, field="fixed_rate",
                counterparty_value=c.fixed_rate, internal_value=i.fixed_rate,
                difference=round(c.fixed_rate - i.fixed_rate, 4),
                difference_unit="percent",
                exposure_usd=round(exposure, 2) if exposure else None,
                severity="medium",
            )
        return None

    def _check_day_count(self, c, i) -> Optional[Discrepancy]:
        if not (c.day_count and i.day_count):
            return None
        if c.day_count.strip().upper() != i.day_count.strip().upper():
            return Discrepancy(
                category=EXC.DAY_COUNT, field="day_count",
                counterparty_value=c.day_count, internal_value=i.day_count,
                difference="Day count convention mismatch",
                difference_unit="convention",
                exposure_usd=None, severity="medium",
            )
        return None

    def _check_effective_date(self, c, i) -> Optional[Discrepancy]:
        if not (c.effective_date and i.effective_date):
            return None
        cd, id_ = _parse_date(c.effective_date), _parse_date(i.effective_date)
        if cd and id_ and cd != id_:
            return Discrepancy(
                category=EXC.EFF_DATE, field="effective_date",
                counterparty_value=str(c.effective_date),
                internal_value=str(i.effective_date),
                difference=f"{abs((cd - id_).days)} day(s)",
                difference_unit="days",
                exposure_usd=None, severity="high",
            )
        return None

    def _check_termination_date(self, c, i) -> Optional[Discrepancy]:
        if not (c.termination_date and i.termination_date):
            return None
        cd, id_ = _parse_date(c.termination_date), _parse_date(i.termination_date)
        if cd and id_ and cd != id_:
            return Discrepancy(
                category=EXC.TERM_DATE, field="termination_date",
                counterparty_value=str(c.termination_date),
                internal_value=str(i.termination_date),
                difference=f"{abs((cd - id_).days)} day(s)",
                difference_unit="days",
                exposure_usd=None, severity="high",
            )
        return None

    def _check_valuation_frequency(self, c, i) -> Optional[Discrepancy]:
        if not (c.valuation_frequency and i.valuation_frequency):
            return None
        if c.valuation_frequency.strip().lower() != i.valuation_frequency.strip().lower():
            return Discrepancy(
                category=EXC.FREQ, field="valuation_frequency",
                counterparty_value=c.valuation_frequency,
                internal_value=i.valuation_frequency,
                difference="Valuation frequency mismatch",
                difference_unit="frequency",
                exposure_usd=None, severity="low",
            )
        return None

    # -------------------------------------------------------------------------
    # Aggregation
    # -------------------------------------------------------------------------

    def _overall_severity(self, primary, secondary) -> str:
        levels = {"high": 3, "medium": 2, "low": 1, "none": 0}
        sev = "none"
        for d in [primary, secondary]:
            if d and levels.get(d.severity, 0) > levels.get(sev, 0):
                sev = d.severity
        return sev

    def _escalation_required(self, primary, secondary) -> bool:
        no_esc = {EXC.FREQ, EXC.CLEAN}
        return any(d and d.category not in no_esc for d in [primary, secondary])

    def _recommended_action(self, primary, secondary, conf) -> str:
        if primary is None:
            return "Confirmation matches internal record. No action required."
        ref = conf.ref_isin or conf.ref_cusip or conf.ref_issuer or "reference asset TRS"
        parts = []
        for label, d in [("Primary", primary), ("Secondary", secondary)]:
            if d:
                exp_str = f", exposure ~${d.exposure_usd:,.0f}" if d.exposure_usd else ""
                parts.append(
                    f"{label}: {d.category} on {d.field} "
                    f"({d.counterparty_value} vs {d.internal_value}{exp_str})"
                )
        action = f"{ref} TRS confirmation break. " + " | ".join(parts)
        if self._escalation_required(primary, secondary):
            action += " — Escalate; do not affirm."
        return action
