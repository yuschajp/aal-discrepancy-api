#!/usr/bin/env python3
"""
engine/cds_reconciler.py — AAL Discrepancy Detection API v2
Deterministic reconciliation engine for Credit Default Swap (CDS) confirmations.

Reference document: Standard Chartered Bank Single-Name CDS
  (Auction Physical Settlement, 2014 ISDA Credit Derivatives Definitions)

Covers:
  - Single-name CDS (corporate, sovereign, financial)
  - Index CDS tranches (field subset)
  - Loan CDS (LCDS)

Primary ICP: Insurance company credit hedges on general account
(F&G alternatives/credit exposure, CLO/CMBS hedges).

Field priority:
  1. Counterparty (legal entity)
  2. Reference entity
  3. Notional (Fixed Rate Payer Calculation Amount)
  4. Fixed rate (CDS spread)
  5. Buyer / Seller direction (protection buyer vs seller)
  6. Scheduled termination date (maturity)
  7. Effective date
  8. Reference obligation CUSIP/ISIN
  9. Reference obligation coupon
  10. Reference obligation maturity
  11. Seniority level
  12. Day count fraction
  13. Currency
  14. Credit events set
  15. Settlement method

Architecture: same hybrid pipeline. LLM extracts, code reconciles.
Benchmark: AAL-D-007 (planned)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Exception categories
# ---------------------------------------------------------------------------
class EXC:
    CPTY       = "EXC-CPTY"    # counterparty legal entity mismatch
    REF_ENT    = "EXC-CPTY"    # reference entity mismatch
    NOTIONAL   = "EXC-QTY"     # notional mismatch
    SPREAD     = "EXC-PRICE"   # fixed rate/spread mismatch
    DIR        = "EXC-DIR"     # buyer vs seller (protection buyer/seller)
    TERM_DATE  = "EXC-SDATE"   # scheduled termination date mismatch
    EFF_DATE   = "EXC-SDATE"   # effective date mismatch
    REF_OBL    = "EXC-CUSIP"   # reference obligation CUSIP/ISIN mismatch
    REF_COUP   = "EXC-COUP"    # reference obligation coupon mismatch
    REF_MAT    = "EXC-MAT"     # reference obligation maturity mismatch
    SENIORITY  = "EXC-PROD"    # seniority level mismatch
    DAY_COUNT  = "EXC-PROD"    # day count fraction mismatch
    CCY        = "EXC-CCY"     # currency mismatch
    CREDIT_EVT = "EXC-PROD"    # credit events set mismatch
    SETTLE     = "EXC-PROD"    # settlement method mismatch
    CLEAN      = None


# ---------------------------------------------------------------------------
# Severity table
# ---------------------------------------------------------------------------
SEVERITY_TABLE: dict[str, str] = {
    EXC.CPTY:       "high",
    EXC.REF_ENT:    "high",
    EXC.NOTIONAL:   "high",
    EXC.SPREAD:     "high",
    EXC.DIR:        "high",
    EXC.TERM_DATE:  "high",
    EXC.EFF_DATE:   "high",
    EXC.REF_OBL:    "high",
    EXC.REF_COUP:   "medium",
    EXC.REF_MAT:    "medium",
    EXC.CCY:        "high",
    EXC.SENIORITY:  "medium",
    EXC.DAY_COUNT:  "medium",
    EXC.CREDIT_EVT: "high",
    EXC.SETTLE:     "medium",
}

def severity_for(category: str) -> str:
    return SEVERITY_TABLE.get(category, "medium")


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
# Credit event set comparison
# ---------------------------------------------------------------------------
STANDARD_CREDIT_EVENTS = {
    "bankruptcy", "failure to pay", "obligation default",
    "obligation acceleration", "repudiation/moratorium",
    "restructuring", "governmental intervention",
}

def normalize_credit_events(events) -> frozenset:
    """Normalize a list or comma-separated string of credit events."""
    if events is None:
        return frozenset()
    if isinstance(events, str):
        events = [e.strip() for e in events.split(",")]
    return frozenset(e.strip().lower() for e in events if e.strip())


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class CDSConfirmation:
    trade_id:           Optional[str]   = None
    trade_date:         Optional[str]   = None
    effective_date:     Optional[str]   = None
    termination_date:   Optional[str]   = None
    counterparty:       Optional[str]   = None
    usi:                Optional[str]   = None
    # Parties
    protection_buyer:   Optional[str]   = None   # fixed rate payer
    protection_seller:  Optional[str]   = None   # floating rate payer
    # Reference entity
    reference_entity:   Optional[str]   = None
    seniority:          Optional[str]   = None   # "senior" | "subordinated"
    # Reference obligation
    ref_cusip:          Optional[str]   = None
    ref_isin:           Optional[str]   = None
    ref_obligor:        Optional[str]   = None   # primary obligor
    ref_coupon:         Optional[float] = None
    ref_maturity:       Optional[str]   = None
    # Economics
    notional:           Optional[float] = None
    currency:           Optional[str]   = None
    fixed_rate:         Optional[float] = None   # CDS spread (% per annum)
    day_count:          Optional[str]   = None
    payment_frequency:  Optional[str]   = None
    # Credit events
    credit_events:      Optional[list]  = None   # list of credit event strings
    payment_requirement:Optional[float] = None   # default $1M
    default_requirement:Optional[float] = None   # default $10M
    # Settlement
    settlement_method:  Optional[str]   = None   # "auction" | "physical" | "cash"
    reference_price:    Optional[float] = None   # typically 100%

    @classmethod
    def from_dict(cls, d: dict) -> "CDSConfirmation":
        fields = [
            "trade_id", "trade_date", "effective_date", "termination_date",
            "counterparty", "usi", "protection_buyer", "protection_seller",
            "reference_entity", "seniority", "ref_cusip", "ref_isin",
            "ref_obligor", "ref_coupon", "ref_maturity",
            "notional", "currency", "fixed_rate", "day_count",
            "payment_frequency", "credit_events", "payment_requirement",
            "default_requirement", "settlement_method", "reference_price",
        ]
        obj = cls(**{f: d[f] for f in fields if f in d})
        if obj.seniority:
            obj.seniority = obj.seniority.strip().lower()
        if obj.currency:
            obj.currency = obj.currency.strip().upper()
        if obj.settlement_method:
            obj.settlement_method = obj.settlement_method.strip().lower()
        if obj.ref_cusip:
            obj.ref_cusip = obj.ref_cusip.strip().upper().replace(" ", "")
        if obj.ref_isin:
            obj.ref_isin = obj.ref_isin.strip().upper().replace(" ", "")
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
class CDSReconciler:
    """
    Deterministic reconciliation for CDS confirmations.

    Tolerances:
      Spread:    0.001% (0.1bp) — CDS spreads quoted to 0.1bp
      Notional:  $1,000
      Ref coupon: 0.001%
    """

    SPREAD_TOL   = 0.001
    NOTIONAL_TOL = 1_000
    COUPON_TOL   = 0.001

    def reconcile(
        self,
        confirmation: CDSConfirmation | dict,
        internal:     CDSConfirmation | dict,
        case_id:      Optional[str] = None,
    ) -> ReconciliationResult:

        if isinstance(confirmation, dict):
            confirmation = CDSConfirmation.from_dict(confirmation)
        if isinstance(internal, dict):
            internal = CDSConfirmation.from_dict(internal)

        found: list[Discrepancy] = []
        notes: list[str] = []

        for check in [
            self._check_counterparty,
            self._check_reference_entity,
            self._check_notional,
            self._check_spread,
            self._check_direction,
            self._check_termination_date,
            self._check_effective_date,
            self._check_ref_obligation,
            self._check_ref_coupon,
            self._check_ref_maturity,
            self._check_seniority,
            self._check_day_count,
            self._check_currency,
            self._check_credit_events,
            self._check_settlement,
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

    def _check_reference_entity(self, c, i) -> Optional[Discrepancy]:
        if not (c.reference_entity and i.reference_entity):
            return None
        if c.reference_entity.strip().lower() != i.reference_entity.strip().lower():
            return Discrepancy(
                category=EXC.REF_ENT, field="reference_entity",
                counterparty_value=c.reference_entity, internal_value=i.reference_entity,
                difference="Reference entity mismatch",
                difference_unit="entity",
                exposure_usd=i.notional, severity="high",
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

    def _check_spread(self, c, i) -> Optional[Discrepancy]:
        if c.fixed_rate is None or i.fixed_rate is None:
            return None
        diff = abs(c.fixed_rate - i.fixed_rate)
        if diff > self.SPREAD_TOL:
            # Exposure: spread diff × notional × annualized
            exposure = (i.notional or 0) * diff / 100
            return Discrepancy(
                category=EXC.SPREAD, field="fixed_rate",
                counterparty_value=c.fixed_rate, internal_value=i.fixed_rate,
                difference=round(c.fixed_rate - i.fixed_rate, 4),
                difference_unit="percent",
                exposure_usd=round(exposure, 2) if exposure else None,
                severity="high",
            )
        return None

    def _check_direction(self, c, i) -> Optional[Discrepancy]:
        """Protection buyer vs seller — wrong direction = wrong P&L on credit event."""
        cb = (c.protection_buyer or "").strip().lower()
        ib = (i.protection_buyer or "").strip().lower()
        if not (cb and ib):
            return None
        if cb != ib:
            return Discrepancy(
                category=EXC.DIR, field="protection_buyer",
                counterparty_value=c.protection_buyer, internal_value=i.protection_buyer,
                difference="Protection buyer mismatch — wrong direction",
                difference_unit="direction",
                exposure_usd=i.notional, severity="high",
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

    def _check_ref_obligation(self, c, i) -> Optional[Discrepancy]:
        cv = (c.ref_isin or c.ref_cusip or "").upper().replace(" ", "")
        iv = (i.ref_isin or i.ref_cusip or "").upper().replace(" ", "")
        if not (cv and iv):
            return None
        if cv != iv:
            return Discrepancy(
                category=EXC.REF_OBL,
                field="ref_isin" if (c.ref_isin or i.ref_isin) else "ref_cusip",
                counterparty_value=cv, internal_value=iv,
                difference="Reference obligation identifier mismatch",
                difference_unit="identifier",
                exposure_usd=None, severity="high",
            )
        return None

    def _check_ref_coupon(self, c, i) -> Optional[Discrepancy]:
        if c.ref_coupon is None or i.ref_coupon is None:
            return None
        diff = abs(c.ref_coupon - i.ref_coupon)
        if diff > self.COUPON_TOL:
            return Discrepancy(
                category=EXC.REF_COUP, field="ref_coupon",
                counterparty_value=c.ref_coupon, internal_value=i.ref_coupon,
                difference=round(c.ref_coupon - i.ref_coupon, 4),
                difference_unit="percent",
                exposure_usd=None, severity="medium",
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
                exposure_usd=None, severity="medium",
            )
        return None

    def _check_seniority(self, c, i) -> Optional[Discrepancy]:
        if not (c.seniority and i.seniority):
            return None
        if c.seniority.lower() != i.seniority.lower():
            return Discrepancy(
                category=EXC.SENIORITY, field="seniority",
                counterparty_value=c.seniority, internal_value=i.seniority,
                difference="Seniority level mismatch",
                difference_unit="seniority",
                exposure_usd=None, severity="medium",
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

    def _check_credit_events(self, c, i) -> Optional[Discrepancy]:
        """Credit event sets must match — missing or extra events materially
        change the protection being sold/bought."""
        ce_c = normalize_credit_events(c.credit_events)
        ce_i = normalize_credit_events(i.credit_events)
        if not (ce_c and ce_i):
            return None
        if ce_c != ce_i:
            missing = ce_i - ce_c    # in internal but not in confirmation
            extra   = ce_c - ce_i    # in confirmation but not internal
            diff_parts = []
            if missing:
                diff_parts.append(f"missing: {', '.join(sorted(missing))}")
            if extra:
                diff_parts.append(f"extra: {', '.join(sorted(extra))}")
            return Discrepancy(
                category=EXC.CREDIT_EVT, field="credit_events",
                counterparty_value=sorted(ce_c),
                internal_value=sorted(ce_i),
                difference="; ".join(diff_parts),
                difference_unit="events",
                exposure_usd=i.notional, severity="high",
            )
        return None

    def _check_settlement(self, c, i) -> Optional[Discrepancy]:
        if not (c.settlement_method and i.settlement_method):
            return None
        if c.settlement_method.lower() != i.settlement_method.lower():
            return Discrepancy(
                category=EXC.SETTLE, field="settlement_method",
                counterparty_value=c.settlement_method,
                internal_value=i.settlement_method,
                difference="Settlement method mismatch",
                difference_unit="method",
                exposure_usd=None, severity="medium",
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
        no_esc = {EXC.CLEAN}
        return any(d and d.category not in no_esc for d in [primary, secondary])

    def _recommended_action(self, primary, secondary, conf) -> str:
        if primary is None:
            return "Confirmation matches internal record. No action required."
        ref = conf.reference_entity or "unknown reference entity"
        parts = []
        for label, d in [("Primary", primary), ("Secondary", secondary)]:
            if d:
                exp_str = f", exposure ~${d.exposure_usd:,.0f}" if d.exposure_usd else ""
                parts.append(
                    f"{label}: {d.category} on {d.field} "
                    f"({d.counterparty_value} vs {d.internal_value}{exp_str})"
                )
        action = f"{ref} CDS confirmation break. " + " | ".join(parts)
        if self._escalation_required(primary, secondary):
            action += " — Escalate; do not affirm."
        return action
