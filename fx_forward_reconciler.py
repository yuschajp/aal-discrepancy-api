#!/usr/bin/env python3
"""
engine/fx_forward_reconciler.py — AAL Discrepancy Detection API v2
Deterministic reconciliation engine for FX forwards and FX options.

Covers:
  - Vanilla FX forwards (deliverable and non-deliverable)
  - FX options (European calls and puts, cash/deliverable/NDO)
  - Cross-currency FX (NDO cross-currency)

Reference document: Standard Chartered Bank FX Forward/Option template
  (1998 ISDA FX and Currency Option Definitions)
  Fields: Buyer, Seller, Call/Put Currency + Amount, Strike Price,
  Expiration Date, Settlement Date, Settlement type, Premium,
  Currency Option Style/Type, Reference Currency (NDO), Settlement Rate Option

Primary ICP: Insurance company investment ops — currency hedges on
general account assets, reinsurance settlement, premium remittances.

Field priority (FX forward):
  1. Counterparty (legal entity)
  2. Call currency + amount (currency being bought)
  3. Put currency + amount (currency being sold)
  4. Strike / forward rate
  5. Settlement date (value date)
  6. Settlement type (deliverable vs NDO vs cash)
  7. Buyer / seller direction

Field priority (FX option additions):
  8. Option type (call vs put)
  9. Option style (European vs American)
  10. Expiration date
  11. Premium + premium payment date

Architecture: same hybrid pipeline. LLM extracts, code reconciles.
Benchmark: AAL-D-006 (planned)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Exception categories
# ---------------------------------------------------------------------------
class EXC:
    CPTY        = "EXC-CPTY"     # counterparty mismatch
    CALL_CCY    = "EXC-CCY"      # call currency mismatch
    PUT_CCY     = "EXC-CCY"      # put currency mismatch
    CALL_AMT    = "EXC-QTY"      # call amount mismatch
    PUT_AMT     = "EXC-QTY"      # put amount mismatch
    STRIKE      = "EXC-PRICE"    # strike / forward rate mismatch
    SETTLE_DATE = "EXC-SETTLE"   # settlement / value date mismatch
    SETTLE_TYPE = "EXC-PROD"     # deliverable vs NDO vs cash mismatch
    DIR         = "EXC-DIR"      # buyer vs seller direction mismatch
    OPT_TYPE    = "EXC-TYPE"     # call vs put mismatch
    OPT_STYLE   = "EXC-PROD"     # European vs American mismatch
    EXPIRY      = "EXC-EXPIRY"   # expiration date mismatch
    PREM        = "EXC-PREM"     # premium amount mismatch
    PREM_DATE   = "EXC-SETTLE"   # premium payment date mismatch
    REF_CCY     = "EXC-CCY"      # reference currency mismatch (NDO)
    CLEAN       = None


# ---------------------------------------------------------------------------
# Severity table
# ---------------------------------------------------------------------------
SEVERITY_TABLE: dict[str, str] = {
    EXC.CPTY:        "high",
    EXC.CALL_CCY:    "high",
    EXC.PUT_CCY:     "high",
    EXC.CALL_AMT:    "high",
    EXC.PUT_AMT:     "high",
    EXC.STRIKE:      "high",
    EXC.SETTLE_DATE: "high",
    EXC.DIR:         "high",
    EXC.OPT_TYPE:    "high",
    EXC.EXPIRY:      "high",
    EXC.REF_CCY:     "high",
    EXC.SETTLE_TYPE: "medium",
    EXC.OPT_STYLE:   "medium",
    EXC.PREM:        "medium",
    EXC.PREM_DATE:   "medium",
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
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%d-%b-%Y"):
            try:
                return datetime.strptime(v.strip(), fmt).date()
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class FXConfirmation:
    """Normalised FX forward/option confirmation."""
    trade_id:           Optional[str]   = None
    trade_date:         Optional[str]   = None
    counterparty:       Optional[str]   = None
    usi:                Optional[str]   = None
    # FX economics
    call_currency:      Optional[str]   = None   # currency being bought
    call_amount:        Optional[float] = None
    put_currency:       Optional[str]   = None   # currency being sold
    put_amount:         Optional[float] = None
    strike:             Optional[float] = None   # forward rate / option strike
    strike_quote:       Optional[str]   = None   # e.g. "USD per EUR"
    # Direction
    buyer:              Optional[str]   = None   # party buying call currency
    seller:             Optional[str]   = None
    # Settlement
    settlement_date:    Optional[str]   = None   # value date
    settlement_type:    Optional[str]   = None   # "deliverable" | "ndo" | "cash"
    reference_currency: Optional[str]   = None   # for NDO
    settlement_rate_option: Optional[str] = None # fixing source for NDO
    # Option fields (null for vanilla forwards)
    option_type:        Optional[str]   = None   # "call" | "put"
    option_style:       Optional[str]   = None   # "european" | "american"
    expiry_date:        Optional[str]   = None
    expiry_time:        Optional[str]   = None
    premium:            Optional[float] = None
    premium_currency:   Optional[str]   = None
    premium_date:       Optional[str]   = None

    @classmethod
    def from_dict(cls, d: dict) -> "FXConfirmation":
        fields = [
            "trade_id", "trade_date", "counterparty", "usi",
            "call_currency", "call_amount", "put_currency", "put_amount",
            "strike", "strike_quote", "buyer", "seller",
            "settlement_date", "settlement_type", "reference_currency",
            "settlement_rate_option", "option_type", "option_style",
            "expiry_date", "expiry_time", "premium", "premium_currency",
            "premium_date",
        ]
        obj = cls(**{f: d[f] for f in fields if f in d})
        # Normalize
        if obj.call_currency:
            obj.call_currency = obj.call_currency.strip().upper()
        if obj.put_currency:
            obj.put_currency = obj.put_currency.strip().upper()
        if obj.reference_currency:
            obj.reference_currency = obj.reference_currency.strip().upper()
        if obj.premium_currency:
            obj.premium_currency = obj.premium_currency.strip().upper()
        if obj.option_type:
            obj.option_type = obj.option_type.strip().lower()
        if obj.option_style:
            obj.option_style = obj.option_style.strip().lower()
        if obj.settlement_type:
            obj.settlement_type = obj.settlement_type.strip().lower()
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
class FXReconciler:
    """
    Deterministic reconciliation for FX forwards and options.

    Tolerances:
      Strike/rate:  0.00001 (0.1 pip on major pairs, tighter for high-rate EM)
      Call/put amt: $1,000 or 0.01% of amount
      Premium:      $500 or 0.1%
    """

    STRIKE_TOL   = 0.00001   # forward rate tolerance
    AMOUNT_TOL   = 1_000     # absolute currency amount tolerance
    AMOUNT_PCT   = 0.0001    # 0.01% of amount
    PREMIUM_TOL  = 500
    PREMIUM_PCT  = 0.001

    def reconcile(
        self,
        confirmation: FXConfirmation | dict,
        internal:     FXConfirmation | dict,
        case_id:      Optional[str] = None,
    ) -> ReconciliationResult:

        if isinstance(confirmation, dict):
            confirmation = FXConfirmation.from_dict(confirmation)
        if isinstance(internal, dict):
            internal = FXConfirmation.from_dict(internal)

        found: list[Discrepancy] = []
        notes: list[str] = []

        for check in [
            self._check_counterparty,
            self._check_call_currency,
            self._check_put_currency,
            self._check_strike,
            self._check_call_amount,
            self._check_put_amount,
            self._check_settlement_date,
            self._check_settlement_type,
            self._check_direction,
            self._check_option_type,
            self._check_option_style,
            self._check_expiry,
            self._check_premium,
            self._check_reference_currency,
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

    def _check_call_currency(self, c, i) -> Optional[Discrepancy]:
        if not (c.call_currency and i.call_currency):
            return None
        if c.call_currency != i.call_currency:
            return Discrepancy(
                category=EXC.CALL_CCY, field="call_currency",
                counterparty_value=c.call_currency, internal_value=i.call_currency,
                difference="Call currency mismatch", difference_unit="currency",
                exposure_usd=None, severity="high",
            )
        return None

    def _check_put_currency(self, c, i) -> Optional[Discrepancy]:
        if not (c.put_currency and i.put_currency):
            return None
        if c.put_currency != i.put_currency:
            return Discrepancy(
                category=EXC.PUT_CCY, field="put_currency",
                counterparty_value=c.put_currency, internal_value=i.put_currency,
                difference="Put currency mismatch", difference_unit="currency",
                exposure_usd=None, severity="high",
            )
        return None

    def _check_call_amount(self, c, i) -> Optional[Discrepancy]:
        if c.call_amount is None or i.call_amount is None:
            return None
        tol = max(self.AMOUNT_TOL, i.call_amount * self.AMOUNT_PCT)
        diff = abs(c.call_amount - i.call_amount)
        if diff > tol:
            return Discrepancy(
                category=EXC.CALL_AMT, field="call_amount",
                counterparty_value=c.call_amount, internal_value=i.call_amount,
                difference=c.call_amount - i.call_amount,
                difference_unit=i.call_currency or "units",
                exposure_usd=diff if (i.call_currency == "USD") else None,
                severity="high",
            )
        return None

    def _check_put_amount(self, c, i) -> Optional[Discrepancy]:
        if c.put_amount is None or i.put_amount is None:
            return None
        tol = max(self.AMOUNT_TOL, i.put_amount * self.AMOUNT_PCT)
        diff = abs(c.put_amount - i.put_amount)
        if diff > tol:
            return Discrepancy(
                category=EXC.PUT_AMT, field="put_amount",
                counterparty_value=c.put_amount, internal_value=i.put_amount,
                difference=c.put_amount - i.put_amount,
                difference_unit=i.put_currency or "units",
                exposure_usd=diff if (i.put_currency == "USD") else None,
                severity="high",
            )
        return None

    def _check_strike(self, c, i) -> Optional[Discrepancy]:
        if c.strike is None or i.strike is None:
            return None
        # Use percentage tolerance for high EM rates (e.g. IDR/USD ~15000)
        tol = max(self.STRIKE_TOL, i.strike * 0.0001)
        diff = abs(c.strike - i.strike)
        if diff > tol:
            # Exposure: rate diff × notional (approximate using put amount as USD notional)
            notional = i.put_amount if i.put_currency == "USD" else (
                i.call_amount if i.call_currency == "USD" else None
            )
            exposure = (notional * diff / i.strike) if (notional and i.strike) else None
            return Discrepancy(
                category=EXC.STRIKE, field="strike",
                counterparty_value=c.strike, internal_value=i.strike,
                difference=round(c.strike - i.strike, 8),
                difference_unit="rate",
                exposure_usd=round(exposure, 2) if exposure else None,
                severity="high",
            )
        return None

    def _check_settlement_date(self, c, i) -> Optional[Discrepancy]:
        if not (c.settlement_date and i.settlement_date):
            return None
        cd, id_ = _parse_date(c.settlement_date), _parse_date(i.settlement_date)
        if cd and id_ and cd != id_:
            notional = i.put_amount or i.call_amount
            return Discrepancy(
                category=EXC.SETTLE_DATE, field="settlement_date",
                counterparty_value=str(c.settlement_date),
                internal_value=str(i.settlement_date),
                difference=f"{abs((cd - id_).days)} day(s)",
                difference_unit="days",
                exposure_usd=notional, severity="high",
            )
        return None

    def _check_settlement_type(self, c, i) -> Optional[Discrepancy]:
        if not (c.settlement_type and i.settlement_type):
            return None
        if c.settlement_type.lower() != i.settlement_type.lower():
            return Discrepancy(
                category=EXC.SETTLE_TYPE, field="settlement_type",
                counterparty_value=c.settlement_type, internal_value=i.settlement_type,
                difference="Deliverable vs NDO vs cash mismatch",
                difference_unit="type",
                exposure_usd=None, severity="medium",
            )
        return None

    def _check_direction(self, c, i) -> Optional[Discrepancy]:
        if not (c.buyer and i.buyer):
            return None
        if c.buyer.strip().lower() != i.buyer.strip().lower():
            # Only flag if clearly different entity (not just naming variant)
            if not any(word in c.buyer.lower() for word in i.buyer.lower().split()[:2]):
                return Discrepancy(
                    category=EXC.DIR, field="buyer",
                    counterparty_value=c.buyer, internal_value=i.buyer,
                    difference="Buyer direction mismatch",
                    difference_unit="direction",
                    exposure_usd=i.call_amount or i.put_amount, severity="high",
                )
        return None

    def _check_option_type(self, c, i) -> Optional[Discrepancy]:
        if not (c.option_type and i.option_type):
            return None
        if c.option_type.lower() != i.option_type.lower():
            return Discrepancy(
                category=EXC.OPT_TYPE, field="option_type",
                counterparty_value=c.option_type, internal_value=i.option_type,
                difference="Call vs put mismatch", difference_unit="type",
                exposure_usd=None, severity="high",
            )
        return None

    def _check_option_style(self, c, i) -> Optional[Discrepancy]:
        if not (c.option_style and i.option_style):
            return None
        if c.option_style.lower() != i.option_style.lower():
            return Discrepancy(
                category=EXC.OPT_STYLE, field="option_style",
                counterparty_value=c.option_style, internal_value=i.option_style,
                difference="European vs American mismatch", difference_unit="style",
                exposure_usd=None, severity="medium",
            )
        return None

    def _check_expiry(self, c, i) -> Optional[Discrepancy]:
        if not (c.expiry_date and i.expiry_date):
            return None
        cd, id_ = _parse_date(c.expiry_date), _parse_date(i.expiry_date)
        if cd and id_ and cd != id_:
            return Discrepancy(
                category=EXC.EXPIRY, field="expiry_date",
                counterparty_value=str(c.expiry_date),
                internal_value=str(i.expiry_date),
                difference=f"{abs((cd - id_).days)} day(s)",
                difference_unit="days",
                exposure_usd=None, severity="high",
            )
        return None

    def _check_premium(self, c, i) -> Optional[Discrepancy]:
        if c.premium is None or i.premium is None:
            return None
        tol = max(self.PREMIUM_TOL, i.premium * self.PREMIUM_PCT)
        diff = abs(c.premium - i.premium)
        if diff > tol:
            return Discrepancy(
                category=EXC.PREM, field="premium",
                counterparty_value=c.premium, internal_value=i.premium,
                difference=round(c.premium - i.premium, 2),
                difference_unit="usd",
                exposure_usd=round(diff, 2), severity="medium",
            )
        return None

    def _check_reference_currency(self, c, i) -> Optional[Discrepancy]:
        """NDO reference currency check."""
        if not (c.reference_currency and i.reference_currency):
            return None
        if c.reference_currency != i.reference_currency:
            return Discrepancy(
                category=EXC.REF_CCY, field="reference_currency",
                counterparty_value=c.reference_currency,
                internal_value=i.reference_currency,
                difference="NDO reference currency mismatch",
                difference_unit="currency",
                exposure_usd=None, severity="high",
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
        pair = f"{conf.call_currency or '?'}/{conf.put_currency or '?'}"
        struct = "FX option" if conf.option_type else "FX forward"
        parts = []
        for label, d in [("Primary", primary), ("Secondary", secondary)]:
            if d:
                exp_str = f", exposure ~${d.exposure_usd:,.0f}" if d.exposure_usd else ""
                parts.append(
                    f"{label}: {d.category} on {d.field} "
                    f"({d.counterparty_value} vs {d.internal_value}{exp_str})"
                )
        action = f"{pair} {struct} confirmation break. " + " | ".join(parts)
        if self._escalation_required(primary, secondary):
            action += " — Escalate; do not affirm."
        return action
