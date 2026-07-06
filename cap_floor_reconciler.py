#!/usr/bin/env python3
"""
engine/cap_floor_reconciler.py — AAL Discrepancy Detection API v2
Deterministic reconciliation engine for interest rate caps and floors.

Covers:
  - Interest rate caps (buyer receives payment when floating > cap rate)
  - Interest rate floors (buyer receives payment when floating < floor rate)
  - Collars (cap + floor combined)
  - Corridors (cap spread)

Primary ICP: Insurance company liability hedging (F&G general account).
Cap structures used to hedge floating rate exposure on liabilities and
to manage duration/convexity on the general account fixed income book.

Reference document: Bank of America / Goal Capital Funding Trust 2007-1
  - $35M notional, LIBOR cap at 7.00%, quarterly reset, ACT/360, 5yr term
  - Upfront premium: $89,500

Structural difference from IRS:
  - No fixed leg payment stream — single upfront premium
  - Cap/floor rate replaces fixed rate
  - Settlement only occurs when floating exceeds cap (or falls below floor)
  - Day count fraction applied to settlement spread only

Field priority:
  1. Counterparty
  2. Notional
  3. Cap rate (or floor rate)
  4. Floor rate (for collars)
  5. Floating rate index
  6. Currency
  7. Effective date
  8. Termination/maturity date
  9. Payment frequency
  10. Day count fraction
  11. Premium (upfront)
  12. Reset dates / calculation agent

Architecture: same hybrid pipeline as IRS and equity option engines.
Benchmark: AAL-D-005 (planned)
Decision Log: Discrepancy API — Decision Log
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Exception categories — caps/floors
# ---------------------------------------------------------------------------
class EXC:
    CAP_RATE  = "EXC-CAP"      # cap rate mismatch
    FLOOR_RATE= "EXC-FLOOR"    # floor rate mismatch
    NOTIONAL  = "EXC-QTY"      # notional mismatch
    CPTY      = "EXC-CPTY"     # counterparty mismatch
    FLOAT     = "EXC-PROD"     # floating rate index mismatch
    CCY       = "EXC-CCY"      # currency mismatch
    EFF_DATE  = "EXC-SDATE"    # effective date mismatch
    TERM_DATE = "EXC-SDATE"    # termination date mismatch
    FREQ      = "EXC-PROD"     # payment frequency mismatch
    DC        = "EXC-PROD"     # day count convention mismatch
    PREM      = "EXC-PREM"     # premium mismatch
    TYPE      = "EXC-TYPE"     # cap vs floor vs collar mismatch
    CLEAN     = None


# ---------------------------------------------------------------------------
# Severity table
# ---------------------------------------------------------------------------
SEVERITY_TABLE: dict[str, str] = {
    EXC.CPTY:      "high",
    EXC.CAP_RATE:  "high",
    EXC.FLOOR_RATE:"high",
    EXC.NOTIONAL:  "high",
    EXC.CCY:       "high",
    EXC.EFF_DATE:  "high",
    EXC.TERM_DATE: "high",
    EXC.TYPE:      "high",
    EXC.FLOAT:     "medium",
    EXC.FREQ:      "medium",
    EXC.DC:        "medium",
    EXC.PREM:      "medium",
}

def severity_for(category: str) -> str:
    return SEVERITY_TABLE.get(category, "medium")


# ---------------------------------------------------------------------------
# Floating rate normalization (reuse from IRS engine pattern)
# ---------------------------------------------------------------------------
FLOAT_RATE_ALIASES: dict[str, str] = {
    "usd-sofr": "SOFR", "sofr compound": "SOFR", "sofr-compound": "SOFR",
    "sofr ois": "SOFR", "us sofr": "SOFR",
    "usd-libor-bba": "LIBOR-3M", "usd-libor-bba-3m": "LIBOR-3M",
    "usd-libor-bba-6m": "LIBOR-6M", "usd-libor": "LIBOR-3M",
    "libor": "LIBOR-3M", "usd libor": "LIBOR-3M",
    "eur-euribor-reuters": "EURIBOR-6M", "euribor": "EURIBOR-6M",
    "gbp-sonia": "SONIA", "sonia": "SONIA",
    "eur-estr": "ESTR", "estr": "ESTR",
}

def normalize_float(rate: Optional[str]) -> Optional[str]:
    if not rate:
        return rate
    return FLOAT_RATE_ALIASES.get(rate.strip().lower(), rate)


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------
def _parse_date(v) -> Optional[date]:
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                return datetime.strptime(v.strip(), fmt).date()
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class CapFloorConfirmation:
    """Normalised cap/floor confirmation — populated by LLM extraction."""
    trade_id:          Optional[str]   = None
    trade_date:        Optional[str]   = None
    structure_type:    Optional[str]   = None   # "cap" | "floor" | "collar" | "corridor"
    counterparty:      Optional[str]   = None
    notional:          Optional[float] = None
    currency:          Optional[str]   = None
    cap_rate:          Optional[float] = None   # % — strike rate for cap
    floor_rate:        Optional[float] = None   # % — strike rate for floor
    floating_rate:     Optional[str]   = None   # normalized index
    floating_tenor:    Optional[str]   = None   # "3M" | "6M" | "1M"
    day_count:         Optional[str]   = None   # "ACT/360" | "30/360" etc
    effective_date:    Optional[str]   = None
    termination_date:  Optional[str]   = None
    payment_frequency: Optional[str]   = None   # "Quarterly" | "Semi-Annual" | "Monthly"
    premium:           Optional[float] = None   # upfront premium amount
    premium_date:      Optional[str]   = None   # premium payment date
    buyer:             Optional[str]   = None   # cap buyer (pays premium)
    seller:            Optional[str]   = None   # cap seller (receives premium)

    @classmethod
    def from_dict(cls, d: dict) -> "CapFloorConfirmation":
        mapping = {
            "trade_id":         "trade_id",
            "trade_date":       "trade_date",
            "structure_type":   "structure_type",
            "counterparty":     "counterparty",
            "notional":         "notional",
            "currency":         "currency",
            "cap_rate":         "cap_rate",
            "floor_rate":       "floor_rate",
            "floating_rate":    "floating_rate",
            "floating_tenor":   "floating_tenor",
            "day_count":        "day_count",
            "effective_date":   "effective_date",
            "termination_date": "termination_date",
            "payment_frequency":"payment_frequency",
            "premium":          "premium",
            "premium_date":     "premium_date",
            "buyer":            "buyer",
            "seller":           "seller",
        }
        kwargs = {dst: d[src] for src, dst in mapping.items() if src in d}
        obj = cls(**kwargs)
        if obj.floating_rate:
            obj.floating_rate = normalize_float(obj.floating_rate)
        if obj.structure_type:
            obj.structure_type = obj.structure_type.strip().lower()
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
class CapFloorReconciler:
    """
    Deterministic field-by-field reconciliation for interest rate
    caps, floors, and collars.

    Tolerances:
      Cap/floor rate: 0.001% (0.1bp) — standard cap rate precision
      Notional:       $1,000
      Premium:        $500 or 0.1% of premium (whichever larger)
    """

    RATE_TOL    = 0.001     # % — 0.1bp
    NOTIONAL_TOL = 1_000
    PREMIUM_TOL  = 500
    PREMIUM_PCT  = 0.001    # 0.1%

    def reconcile(
        self,
        confirmation: CapFloorConfirmation | dict,
        internal:     CapFloorConfirmation | dict,
        case_id:      Optional[str] = None,
    ) -> ReconciliationResult:

        if isinstance(confirmation, dict):
            confirmation = CapFloorConfirmation.from_dict(confirmation)
        if isinstance(internal, dict):
            internal = CapFloorConfirmation.from_dict(internal)

        found: list[Discrepancy] = []
        notes: list[str] = []

        for check in [
            self._check_counterparty,
            self._check_type,
            self._check_notional,
            self._check_cap_rate,
            self._check_floor_rate,
            self._check_floating,
            self._check_currency,
            self._check_effective_date,
            self._check_termination_date,
            self._check_frequency,
            self._check_day_count,
            self._check_premium,
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

    def _check_type(self, c, i) -> Optional[Discrepancy]:
        if not (c.structure_type and i.structure_type):
            return None
        if c.structure_type.lower() != i.structure_type.lower():
            return Discrepancy(
                category=EXC.TYPE, field="structure_type",
                counterparty_value=c.structure_type, internal_value=i.structure_type,
                difference="Cap vs floor vs collar mismatch",
                difference_unit="type",
                exposure_usd=None, severity="high",
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
                difference_unit="notional_usd",
                exposure_usd=diff,
                severity="high",
            )
        return None

    def _check_cap_rate(self, c, i) -> Optional[Discrepancy]:
        if c.cap_rate is None or i.cap_rate is None:
            return None
        diff = abs(c.cap_rate - i.cap_rate)
        if diff > self.RATE_TOL:
            # Exposure proxy: notional × rate_diff × remaining tenor approximation
            exposure = self._rate_exposure(c, i, diff)
            return Discrepancy(
                category=EXC.CAP_RATE, field="cap_rate",
                counterparty_value=c.cap_rate, internal_value=i.cap_rate,
                difference=round(c.cap_rate - i.cap_rate, 4),
                difference_unit="percent",
                exposure_usd=round(exposure, 2) if exposure else None,
                severity="high",
            )
        return None

    def _check_floor_rate(self, c, i) -> Optional[Discrepancy]:
        if c.floor_rate is None or i.floor_rate is None:
            return None
        diff = abs(c.floor_rate - i.floor_rate)
        if diff > self.RATE_TOL:
            exposure = self._rate_exposure(c, i, diff)
            return Discrepancy(
                category=EXC.FLOOR_RATE, field="floor_rate",
                counterparty_value=c.floor_rate, internal_value=i.floor_rate,
                difference=round(c.floor_rate - i.floor_rate, 4),
                difference_unit="percent",
                exposure_usd=round(exposure, 2) if exposure else None,
                severity="high",
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

    def _check_frequency(self, c, i) -> Optional[Discrepancy]:
        if not (c.payment_frequency and i.payment_frequency):
            return None
        if c.payment_frequency.strip().lower() != i.payment_frequency.strip().lower():
            return Discrepancy(
                category=EXC.FREQ, field="payment_frequency",
                counterparty_value=c.payment_frequency,
                internal_value=i.payment_frequency,
                difference="Payment frequency mismatch",
                difference_unit="frequency",
                exposure_usd=None, severity="medium",
            )
        return None

    def _check_day_count(self, c, i) -> Optional[Discrepancy]:
        if not (c.day_count and i.day_count):
            return None
        if c.day_count.strip().upper() != i.day_count.strip().upper():
            return Discrepancy(
                category=EXC.DC, field="day_count",
                counterparty_value=c.day_count, internal_value=i.day_count,
                difference="Day count convention mismatch",
                difference_unit="convention",
                exposure_usd=None, severity="medium",
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
                exposure_usd=round(diff, 2),
                severity="medium" if diff < 50_000 else "high",
            )
        return None

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _rate_exposure(self, c, i, rate_diff: float) -> Optional[float]:
        """Annualized exposure proxy: notional × rate_diff / 100."""
        n = i.notional or c.notional
        if not n:
            return None
        return n * rate_diff / 100.0

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
        struct = conf.structure_type or "cap/floor"
        rate = f"{conf.cap_rate or conf.floor_rate}% {struct}" if (conf.cap_rate or conf.floor_rate) else struct
        parts = []
        for label, d in [("Primary", primary), ("Secondary", secondary)]:
            if d:
                exp_str = f", exposure ~${d.exposure_usd:,.0f}" if d.exposure_usd else ""
                parts.append(
                    f"{label}: {d.category} on {d.field} "
                    f"({d.counterparty_value} vs {d.internal_value}{exp_str})"
                )
        action = f"{rate} confirmation break. " + " | ".join(parts)
        if self._escalation_required(primary, secondary):
            action += " — Escalate; do not affirm."
        return action
