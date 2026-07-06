#!/usr/bin/env python3
"""
engine/fixed_income_reconciler.py — AAL Discrepancy Detection API v2
Deterministic reconciliation engine for fixed income trade confirmations.

Covers:
  - Investment grade corporate bonds
  - Municipal bonds (MSRB G-15)
  - Agency bonds (FNMA, FHLMC, FHLB)
  - Public structured securities (CMBS, RMBS, CLO — field subset)

Primary ICP: Insurance company investment ops (F&G general account —
26% corporate bonds, 21% structured securities per Q1 2026 filing).

Field priority (checked in order):
  1. CUSIP / ISIN (security identifier)
  2. Counterparty (broker-dealer legal entity)
  3. Buy/Sell direction
  4. Par value / face amount (quantity)
  5. Price (per 100 of par)
  6. Yield (cross-check vs price)
  7. Coupon rate
  8. Maturity date
  9. Settlement date
  10. Accrued interest
  11. Principal amount (computed: price × par / 100)
  12. Total consideration (principal + accrued)
  13. Trade capacity (principal vs agent)

Architecture: same hybrid pipeline.
LLM extracts fields; this module does all arithmetic and comparison.

Regulation context:
  - Corporate bonds: FINRA TRACE reporting, T+2 settlement
  - Munis: MSRB G-15 confirmation requirements, T+2 settlement
  - Structured: dealer-to-dealer bilateral, same field structure

Benchmark: AAL-D-004 (planned)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Exception categories — fixed income
# ---------------------------------------------------------------------------
class EXC:
    CUSIP    = "EXC-CUSIP"    # security identifier mismatch
    CPTY     = "EXC-CPTY"     # counterparty/broker-dealer mismatch
    DIR      = "EXC-DIR"      # buy vs sell direction mismatch
    QTY      = "EXC-QTY"      # par value / face amount mismatch
    PRICE    = "EXC-PRICE"    # price per 100 mismatch
    YIELD    = "EXC-YIELD"    # yield mismatch
    COUP     = "EXC-COUP"     # coupon rate mismatch
    MAT      = "EXC-MAT"      # maturity date mismatch
    SETTLE   = "EXC-SETTLE"   # settlement date mismatch
    ACCR     = "EXC-ACCR"     # accrued interest mismatch
    PRIN     = "EXC-PRIN"     # principal amount mismatch
    TOTAL    = "EXC-TOTAL"    # total consideration mismatch
    CAP      = "EXC-CAP"      # trade capacity mismatch (principal vs agent)
    CCY      = "EXC-CCY"      # currency mismatch
    CLEAN    = None


# ---------------------------------------------------------------------------
# Severity rule table
# ---------------------------------------------------------------------------
SEVERITY_TABLE: dict[str, str] = {
    EXC.CUSIP:  "high",    # wrong security = fundamental error
    EXC.CPTY:   "high",    # wrong counterparty = settlement risk
    EXC.DIR:    "high",    # bought vs sold = complete position error
    EXC.QTY:    "high",    # wrong face amount = wrong position size
    EXC.PRICE:  "high",    # wrong price drives wrong cash flows
    EXC.SETTLE: "high",    # wrong settlement = fails risk
    EXC.MAT:    "high",    # wrong maturity = wrong duration
    EXC.CCY:    "high",    # currency mismatch fundamental
    EXC.COUP:   "medium",  # coupon error — material but not immediate
    EXC.YIELD:  "medium",  # yield cross-check — often rounding
    EXC.ACCR:   "medium",  # accrued interest — material at large notional
    EXC.PRIN:   "medium",  # principal amount — derived, catch rounding
    EXC.TOTAL:  "medium",  # total consideration — derived
    EXC.CAP:    "low",     # capacity difference — operational not economic
}

def severity_for(category: str, exposure: Optional[float] = None) -> str:
    base = SEVERITY_TABLE.get(category, "medium")
    # Upgrade to high if exposure is large
    if base == "medium" and exposure and exposure >= 100_000:
        return "high"
    return base


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------
def _parse_date(v) -> Optional[date]:
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y"):
            try:
                return datetime.strptime(v.strip(), fmt).date()
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class FixedIncomeConfirmation:
    """Normalised fixed income confirmation — populated by LLM extraction."""
    trade_id:          Optional[str]   = None
    trade_date:        Optional[str]   = None
    cusip:             Optional[str]   = None
    isin:              Optional[str]   = None
    issuer:            Optional[str]   = None
    security_desc:     Optional[str]   = None
    counterparty:      Optional[str]   = None
    direction:         Optional[str]   = None   # "buy" | "sell"
    capacity:          Optional[str]   = None   # "principal" | "agent"
    par_value:         Optional[float] = None   # face amount
    currency:          Optional[str]   = None
    price:             Optional[float] = None   # price per 100 of par
    yield_rate:        Optional[float] = None   # yield to maturity (%)
    coupon_rate:       Optional[float] = None   # annual coupon (%)
    coupon_frequency:  Optional[str]   = None   # "semi-annual" | "quarterly" | "monthly"
    maturity_date:     Optional[str]   = None
    settlement_date:   Optional[str]   = None
    accrued_interest:  Optional[float] = None   # dollar amount
    principal_amount:  Optional[float] = None   # price × par / 100
    total_consideration: Optional[float] = None # principal + accrued
    day_count:         Optional[str]   = None   # "30/360" | "ACT/ACT" | "ACT/360"
    callable:          Optional[bool]  = None
    first_call_date:   Optional[str]   = None
    rating:            Optional[str]   = None   # credit rating
    sector:            Optional[str]   = None

    @classmethod
    def from_dict(cls, d: dict) -> "FixedIncomeConfirmation":
        mapping = {
            "trade_id":           "trade_id",
            "trade_date":         "trade_date",
            "cusip":              "cusip",
            "isin":               "isin",
            "issuer":             "issuer",
            "security_desc":      "security_desc",
            "counterparty":       "counterparty",
            "direction":          "direction",
            "capacity":           "capacity",
            "par_value":          "par_value",
            "currency":           "currency",
            "price":              "price",
            "yield_rate":         "yield_rate",
            "coupon_rate":        "coupon_rate",
            "coupon_frequency":   "coupon_frequency",
            "maturity_date":      "maturity_date",
            "settlement_date":    "settlement_date",
            "accrued_interest":   "accrued_interest",
            "principal_amount":   "principal_amount",
            "total_consideration":"total_consideration",
            "day_count":          "day_count",
            "callable":           "callable",
            "first_call_date":    "first_call_date",
            "rating":             "rating",
            "sector":             "sector",
        }
        kwargs = {dst: d[src] for src, dst in mapping.items() if src in d}
        obj = cls(**kwargs)
        # Normalize direction and capacity to lowercase
        if obj.direction:
            obj.direction = obj.direction.strip().lower()
        if obj.capacity:
            obj.capacity = obj.capacity.strip().lower()
        # Normalize CUSIP — strip spaces, uppercase
        if obj.cusip:
            obj.cusip = obj.cusip.strip().upper().replace(" ", "")
        if obj.isin:
            obj.isin = obj.isin.strip().upper().replace(" ", "")
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
class FixedIncomeReconciler:
    """
    Deterministic field-by-field reconciliation for fixed income
    trade confirmations (corporates, munis, agencies, structured).

    Tolerances (calibrated to market conventions):
      Price:        0.001 per 100 (0.1 cent) — standard bond price precision
      Yield:        0.001% — 0.1bp tolerance
      Coupon:       0.001% — sub-bp tolerance
      Accrued:      $100 absolute (rounding in accrual calc)
      Principal:    $500 absolute (rounding)
      Total:        $500 absolute
      Par value:    $1,000 absolute (odd-lot tolerance)
    """

    PRICE_TOL    = 0.001    # per 100
    YIELD_TOL    = 0.001    # percent
    COUPON_TOL   = 0.001    # percent
    ACCR_TOL     = 100      # dollars
    PRIN_TOL     = 500      # dollars
    TOTAL_TOL    = 500      # dollars
    PAR_TOL      = 1_000    # dollars

    def reconcile(
        self,
        confirmation: FixedIncomeConfirmation | dict,
        internal:     FixedIncomeConfirmation | dict,
        case_id:      Optional[str] = None,
    ) -> ReconciliationResult:

        if isinstance(confirmation, dict):
            confirmation = FixedIncomeConfirmation.from_dict(confirmation)
        if isinstance(internal, dict):
            internal = FixedIncomeConfirmation.from_dict(internal)

        found: list[Discrepancy] = []
        notes: list[str] = []

        for check in [
            self._check_cusip,
            self._check_counterparty,
            self._check_direction,
            self._check_par_value,
            self._check_price,
            self._check_yield,
            self._check_coupon,
            self._check_maturity,
            self._check_settlement,
            self._check_accrued,
            self._check_principal,
            self._check_total,
            self._check_capacity,
            self._check_currency,
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

    def _check_cusip(self, c, i) -> Optional[Discrepancy]:
        # Prefer CUSIP; fall back to ISIN
        cv = c.cusip or c.isin
        iv = i.cusip or i.isin
        if not (cv and iv):
            return None
        if cv.upper() != iv.upper():
            return Discrepancy(
                category=EXC.CUSIP, field="cusip",
                counterparty_value=cv, internal_value=iv,
                difference="Security identifier mismatch",
                difference_unit="identifier",
                exposure_usd=i.principal_amount,
                severity="high",
            )
        return None

    def _check_counterparty(self, c, i) -> Optional[Discrepancy]:
        if not (c.counterparty and i.counterparty):
            return None
        if c.counterparty.strip().lower() != i.counterparty.strip().lower():
            return Discrepancy(
                category=EXC.CPTY, field="counterparty",
                counterparty_value=c.counterparty, internal_value=i.counterparty,
                difference="Broker-dealer entity mismatch",
                difference_unit="entity",
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
                difference="Buy vs sell mismatch",
                difference_unit="direction",
                exposure_usd=i.principal_amount, severity="high",
            )
        return None

    def _check_par_value(self, c, i) -> Optional[Discrepancy]:
        if c.par_value is None or i.par_value is None:
            return None
        diff = abs(c.par_value - i.par_value)
        if diff > self.PAR_TOL:
            return Discrepancy(
                category=EXC.QTY, field="par_value",
                counterparty_value=c.par_value, internal_value=i.par_value,
                difference=c.par_value - i.par_value,
                difference_unit="par_usd",
                exposure_usd=diff * (i.price or 100) / 100 if i.price else diff,
                severity="high",
            )
        return None

    def _check_price(self, c, i) -> Optional[Discrepancy]:
        if c.price is None or i.price is None:
            return None
        diff = abs(c.price - i.price)
        if diff > self.PRICE_TOL:
            par = i.par_value or c.par_value or 0
            exposure = par * diff / 100
            return Discrepancy(
                category=EXC.PRICE, field="price",
                counterparty_value=c.price, internal_value=i.price,
                difference=round(c.price - i.price, 6),
                difference_unit="per_100",
                exposure_usd=round(exposure, 2),
                severity=severity_for(EXC.PRICE, exposure),
            )
        return None

    def _check_yield(self, c, i) -> Optional[Discrepancy]:
        if c.yield_rate is None or i.yield_rate is None:
            return None
        diff = abs(c.yield_rate - i.yield_rate)
        if diff > self.YIELD_TOL:
            return Discrepancy(
                category=EXC.YIELD, field="yield_rate",
                counterparty_value=c.yield_rate, internal_value=i.yield_rate,
                difference=round(c.yield_rate - i.yield_rate, 4),
                difference_unit="percent",
                exposure_usd=None,
                severity="medium",
            )
        return None

    def _check_coupon(self, c, i) -> Optional[Discrepancy]:
        if c.coupon_rate is None or i.coupon_rate is None:
            return None
        diff = abs(c.coupon_rate - i.coupon_rate)
        if diff > self.COUPON_TOL:
            par = i.par_value or c.par_value or 0
            exposure = par * diff / 100
            return Discrepancy(
                category=EXC.COUP, field="coupon_rate",
                counterparty_value=c.coupon_rate, internal_value=i.coupon_rate,
                difference=round(c.coupon_rate - i.coupon_rate, 4),
                difference_unit="percent",
                exposure_usd=round(exposure, 2),
                severity=severity_for(EXC.COUP, exposure),
            )
        return None

    def _check_maturity(self, c, i) -> Optional[Discrepancy]:
        if not (c.maturity_date and i.maturity_date):
            return None
        cd, id_ = _parse_date(c.maturity_date), _parse_date(i.maturity_date)
        if cd and id_ and cd != id_:
            return Discrepancy(
                category=EXC.MAT, field="maturity_date",
                counterparty_value=str(c.maturity_date),
                internal_value=str(i.maturity_date),
                difference=f"{abs((cd - id_).days)} day(s)",
                difference_unit="days",
                exposure_usd=None, severity="high",
            )
        return None

    def _check_settlement(self, c, i) -> Optional[Discrepancy]:
        if not (c.settlement_date and i.settlement_date):
            return None
        cd, id_ = _parse_date(c.settlement_date), _parse_date(i.settlement_date)
        if cd and id_ and cd != id_:
            return Discrepancy(
                category=EXC.SETTLE, field="settlement_date",
                counterparty_value=str(c.settlement_date),
                internal_value=str(i.settlement_date),
                difference=f"{abs((cd - id_).days)} day(s)",
                difference_unit="days",
                exposure_usd=i.total_consideration, severity="high",
            )
        return None

    def _check_accrued(self, c, i) -> Optional[Discrepancy]:
        if c.accrued_interest is None or i.accrued_interest is None:
            return None
        diff = abs(c.accrued_interest - i.accrued_interest)
        if diff > self.ACCR_TOL:
            return Discrepancy(
                category=EXC.ACCR, field="accrued_interest",
                counterparty_value=c.accrued_interest,
                internal_value=i.accrued_interest,
                difference=round(c.accrued_interest - i.accrued_interest, 2),
                difference_unit="usd",
                exposure_usd=round(diff, 2),
                severity=severity_for(EXC.ACCR, diff),
            )
        return None

    def _check_principal(self, c, i) -> Optional[Discrepancy]:
        if c.principal_amount is None or i.principal_amount is None:
            return None
        diff = abs(c.principal_amount - i.principal_amount)
        if diff > self.PRIN_TOL:
            return Discrepancy(
                category=EXC.PRIN, field="principal_amount",
                counterparty_value=c.principal_amount,
                internal_value=i.principal_amount,
                difference=round(c.principal_amount - i.principal_amount, 2),
                difference_unit="usd",
                exposure_usd=round(diff, 2),
                severity=severity_for(EXC.PRIN, diff),
            )
        return None

    def _check_total(self, c, i) -> Optional[Discrepancy]:
        if c.total_consideration is None or i.total_consideration is None:
            return None
        diff = abs(c.total_consideration - i.total_consideration)
        if diff > self.TOTAL_TOL:
            return Discrepancy(
                category=EXC.TOTAL, field="total_consideration",
                counterparty_value=c.total_consideration,
                internal_value=i.total_consideration,
                difference=round(c.total_consideration - i.total_consideration, 2),
                difference_unit="usd",
                exposure_usd=round(diff, 2),
                severity=severity_for(EXC.TOTAL, diff),
            )
        return None

    def _check_capacity(self, c, i) -> Optional[Discrepancy]:
        if not (c.capacity and i.capacity):
            return None
        if c.capacity.lower() != i.capacity.lower():
            return Discrepancy(
                category=EXC.CAP, field="capacity",
                counterparty_value=c.capacity, internal_value=i.capacity,
                difference="Principal vs agent mismatch",
                difference_unit="capacity",
                exposure_usd=None, severity="low",
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
        no_esc = {EXC.CAP, EXC.CLEAN}
        return any(d and d.category not in no_esc for d in [primary, secondary])

    def _recommended_action(self, primary, secondary, conf) -> str:
        if primary is None:
            return "Confirmation matches internal record. No action required."
        sec_desc = conf.security_desc or conf.cusip or conf.isin or "fixed income"
        parts = []
        for label, d in [("Primary", primary), ("Secondary", secondary)]:
            if d:
                exp_str = f", exposure ~${d.exposure_usd:,.0f}" if d.exposure_usd else ""
                parts.append(
                    f"{label}: {d.category} on {d.field} "
                    f"({d.counterparty_value} vs {d.internal_value}{exp_str})"
                )
        action = f"{sec_desc} confirmation break. " + " | ".join(parts)
        if self._escalation_required(primary, secondary):
            action += " — Escalate; do not affirm."
        return action
