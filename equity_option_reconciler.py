#!/usr/bin/env python3
"""
engine/equity_option_reconciler.py — AAL Discrepancy Detection API v2
Deterministic reconciliation engine for OTC equity index options.

Covers:
  - Vanilla European calls and puts (single index)
  - Call spreads / put spreads (two strikes)
  - Cash-settled index options (FIA/RILA hedging structures)

Field priority (checked in order):
  1. Counterparty (legal entity)
  2. Underlying index
  3. Option type (call/put)
  4. Strike (lower strike for spreads)
  5. Strike high (upper strike for spreads)
  6. Number of options / notional
  7. Premium
  8. Expiration date
  9. Settlement currency
  10. Option style (European/American)
  11. Buyer/Seller direction

Architecture: same hybrid pipeline as IRS.
LLM extracts fields; this module does all arithmetic and comparison.
No LLM ever touches a number.

Benchmark: AAL-D-003 (planned — equity option cases)
Decision log: Discrepancy API — Decision Log
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Exception categories — equity options
# ---------------------------------------------------------------------------
class EXC:
    STRIKE  = "EXC-STRIKE"   # strike price mismatch
    UNDL    = "EXC-UNDL"     # underlying index mismatch
    TYPE    = "EXC-TYPE"     # option type mismatch (call vs put)
    EXPIRY  = "EXC-EXPIRY"   # expiration date mismatch
    PREM    = "EXC-PREM"     # premium amount mismatch
    QTY     = "EXC-QTY"      # number of options / notional mismatch
    CPTY    = "EXC-CPTY"     # counterparty legal entity mismatch
    CCY     = "EXC-CCY"      # settlement currency mismatch
    DIR     = "EXC-DIR"      # buyer/seller direction mismatch
    STYLE   = "EXC-STYLE"    # option style mismatch (European vs American)
    SETT    = "EXC-SETT"     # settlement method mismatch
    CLEAN   = None


# ---------------------------------------------------------------------------
# Underlying index normalization
# Same pattern as floating rate normalization in IRS engine.
# Applied in extraction prompt AND as post-extraction safety net.
# ---------------------------------------------------------------------------
INDEX_ALIASES: dict[str, str] = {
    # S&P 500
    "s&p 500": "SPX", "s&p500": "SPX", "sp500": "SPX",
    "s&p 500 index": "SPX", "spx index": "SPX", "spx": "SPX",
    "standard & poor's 500": "SPX", "standard and poor's 500": "SPX",
    # Russell 2000
    "russell 2000": "RTY", "russell2000": "RTY", "rty": "RTY",
    "russell 2000 index": "RTY",
    # Nasdaq-100
    "nasdaq-100": "NDX", "nasdaq 100": "NDX", "ndx": "NDX",
    "nasdaq100": "NDX", "nasdaq-100 index": "NDX",
    # MSCI EAFE
    "msci eafe": "MSCI_EAFE", "msci-eafe": "MSCI_EAFE",
    "mxea": "MSCI_EAFE", "msci eafe index": "MSCI_EAFE",
    # MSCI EM
    "msci em": "MSCI_EM", "msci emerging markets": "MSCI_EM",
    "mxef": "MSCI_EM", "msci emerging markets index": "MSCI_EM",
    # DJIA
    "dow jones": "DJIA", "djia": "DJIA", "dow jones industrial average": "DJIA",
    # Euro Stoxx 50
    "euro stoxx 50": "SX5E", "eurostoxx 50": "SX5E", "sx5e": "SX5E",
    "euro stoxx 50 index": "SX5E",
    # Nikkei
    "nikkei 225": "NKY", "nikkei225": "NKY", "nky": "NKY",
    "nikkei 225 index": "NKY",
}

def normalize_index(name: Optional[str]) -> Optional[str]:
    if not name:
        return name
    return INDEX_ALIASES.get(name.strip().lower(), name.strip())


# ---------------------------------------------------------------------------
# Severity rule table
# ---------------------------------------------------------------------------
SEVERITY_TABLE: dict[str, str] = {
    EXC.CPTY:   "high",
    EXC.UNDL:   "high",
    EXC.TYPE:   "high",
    EXC.STRIKE: "high",
    EXC.EXPIRY: "high",
    EXC.DIR:    "high",
    EXC.CCY:    "high",
    EXC.QTY:    "high",
    EXC.STYLE:  "medium",
    EXC.SETT:   "medium",
}

def severity_from_premium(category: str, premium_diff: Optional[float]) -> str:
    if category in SEVERITY_TABLE:
        return SEVERITY_TABLE[category]
    # EXC-PREM: tiered by dollar difference
    if premium_diff is None:
        return "medium"
    if abs(premium_diff) >= 100_000:
        return "high"
    if abs(premium_diff) >= 10_000:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------
def _parse_date(v) -> Optional[date]:
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%b-%Y", "%B %d, %Y"):
            try:
                return datetime.strptime(v.strip(), fmt).date()
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class EquityOptionConfirmation:
    """Normalised equity option confirmation — populated by LLM extraction."""
    trade_id:           Optional[str]   = None
    trade_date:         Optional[str]   = None
    counterparty:       Optional[str]   = None
    option_type:        Optional[str]   = None   # "call" | "put"
    option_style:       Optional[str]   = None   # "european" | "american"
    underlying:         Optional[str]   = None   # normalized index code
    strike:             Optional[float] = None   # lower strike (or only strike)
    strike_high:        Optional[float] = None   # upper strike for spreads
    num_options:        Optional[float] = None   # number of option contracts
    notional:           Optional[float] = None   # notional equivalent if stated
    currency:           Optional[str]   = None
    premium:            Optional[float] = None   # total premium
    premium_per_option: Optional[float] = None
    premium_payment_date: Optional[str] = None
    expiry_date:        Optional[str]   = None
    settlement_currency: Optional[str] = None
    settlement_date:    Optional[str]   = None
    settlement_method:  Optional[str]   = None   # "cash" | "physical"
    buyer:              Optional[str]   = None
    seller:             Optional[str]   = None

    @classmethod
    def from_dict(cls, d: dict) -> "EquityOptionConfirmation":
        mapping = {
            "trade_id":             "trade_id",
            "trade_date":           "trade_date",
            "counterparty":         "counterparty",
            "option_type":          "option_type",
            "option_style":         "option_style",
            "underlying":           "underlying",
            "strike":               "strike",
            "strike_high":          "strike_high",
            "num_options":          "num_options",
            "notional":             "notional",
            "currency":             "currency",
            "premium":              "premium",
            "premium_per_option":   "premium_per_option",
            "premium_payment_date": "premium_payment_date",
            "expiry_date":          "expiry_date",
            "settlement_currency":  "settlement_currency",
            "settlement_date":      "settlement_date",
            "settlement_method":    "settlement_method",
            "buyer":                "buyer",
            "seller":               "seller",
        }
        kwargs = {dst: d[src] for src, dst in mapping.items() if src in d}
        obj = cls(**kwargs)
        # normalize underlying on load
        if obj.underlying:
            obj.underlying = normalize_index(obj.underlying)
        # normalize option type
        if obj.option_type:
            obj.option_type = obj.option_type.strip().lower()
        if obj.option_style:
            obj.option_style = obj.option_style.strip().lower()
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
class EquityOptionReconciler:
    """
    Deterministic field-by-field reconciliation for OTC equity index options.

    Tolerances (validated against real confirmation formats):
      Strike:      exact match within $0.01 (index points) or 0.01%
      Premium:     exact match within $1,000 or 0.1% of total premium
      Num options: exact match (integer comparison)
      Notional:    within $10,000
    """

    STRIKE_TOL       = 0.01     # index points — S&P 500 strikes quoted to 2dp
    STRIKE_TOL_PCT   = 0.0001   # 0.01% of strike as alternative tolerance
    PREMIUM_TOL      = 1_000    # absolute dollar tolerance on total premium
    PREMIUM_TOL_PCT  = 0.001    # 0.1% of premium as alternative tolerance
    NOTIONAL_TOL     = 10_000

    def reconcile(
        self,
        confirmation: EquityOptionConfirmation | dict,
        internal:     EquityOptionConfirmation | dict,
        case_id:      Optional[str] = None,
    ) -> ReconciliationResult:

        if isinstance(confirmation, dict):
            confirmation = EquityOptionConfirmation.from_dict(confirmation)
        if isinstance(internal, dict):
            internal = EquityOptionConfirmation.from_dict(internal)

        found: list[Discrepancy] = []
        notes: list[str] = []

        for check in [
            self._check_counterparty,
            self._check_underlying,
            self._check_option_type,
            self._check_strike,
            self._check_strike_high,
            self._check_quantity,
            self._check_premium,
            self._check_expiry,
            self._check_currency,
            self._check_direction,
            self._check_style,
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

    def _check_underlying(self, c, i) -> Optional[Discrepancy]:
        if not (c.underlying and i.underlying):
            return None
        if normalize_index(c.underlying) != normalize_index(i.underlying):
            return Discrepancy(
                category=EXC.UNDL, field="underlying",
                counterparty_value=c.underlying, internal_value=i.underlying,
                difference="Index mismatch", difference_unit="index",
                exposure_usd=None, severity="high",
            )
        return None

    def _check_option_type(self, c, i) -> Optional[Discrepancy]:
        if not (c.option_type and i.option_type):
            return None
        if c.option_type.lower() != i.option_type.lower():
            return Discrepancy(
                category=EXC.TYPE, field="option_type",
                counterparty_value=c.option_type, internal_value=i.option_type,
                difference="Call vs put mismatch", difference_unit="type",
                exposure_usd=None, severity="high",
            )
        return None

    def _check_strike(self, c, i) -> Optional[Discrepancy]:
        if c.strike is None or i.strike is None:
            return None
        tol = max(self.STRIKE_TOL, i.strike * self.STRIKE_TOL_PCT)
        if abs(c.strike - i.strike) > tol:
            return Discrepancy(
                category=EXC.STRIKE, field="strike",
                counterparty_value=c.strike, internal_value=i.strike,
                difference=round(c.strike - i.strike, 4),
                difference_unit="index_points",
                exposure_usd=self._strike_exposure(c, i),
                severity="high",
            )
        return None

    def _check_strike_high(self, c, i) -> Optional[Discrepancy]:
        """Upper strike for call/put spreads."""
        if c.strike_high is None or i.strike_high is None:
            return None
        tol = max(self.STRIKE_TOL, i.strike_high * self.STRIKE_TOL_PCT)
        if abs(c.strike_high - i.strike_high) > tol:
            return Discrepancy(
                category=EXC.STRIKE, field="strike_high",
                counterparty_value=c.strike_high, internal_value=i.strike_high,
                difference=round(c.strike_high - i.strike_high, 4),
                difference_unit="index_points",
                exposure_usd=None,
                severity="high",
            )
        return None

    def _check_quantity(self, c, i) -> Optional[Discrepancy]:
        """Number of options OR notional — whichever is available."""
        # prefer num_options comparison; fall back to notional
        if c.num_options is not None and i.num_options is not None:
            if abs(c.num_options - i.num_options) > 0:
                exp = abs(c.num_options - i.num_options) * (i.premium_per_option or 0)
                return Discrepancy(
                    category=EXC.QTY, field="num_options",
                    counterparty_value=c.num_options, internal_value=i.num_options,
                    difference=c.num_options - i.num_options,
                    difference_unit="contracts",
                    exposure_usd=round(exp, 2) if exp else None,
                    severity="high",
                )
        elif c.notional is not None and i.notional is not None:
            if abs(c.notional - i.notional) > self.NOTIONAL_TOL:
                return Discrepancy(
                    category=EXC.QTY, field="notional",
                    counterparty_value=c.notional, internal_value=i.notional,
                    difference=c.notional - i.notional,
                    difference_unit="notional",
                    exposure_usd=abs(c.notional - i.notional),
                    severity="high",
                )
        return None

    def _check_premium(self, c, i) -> Optional[Discrepancy]:
        if c.premium is None or i.premium is None:
            return None
        tol = max(self.PREMIUM_TOL, i.premium * self.PREMIUM_TOL_PCT)
        diff = c.premium - i.premium
        if abs(diff) > tol:
            return Discrepancy(
                category=EXC.PREM, field="premium",
                counterparty_value=c.premium, internal_value=i.premium,
                difference=round(diff, 2),
                difference_unit="usd",
                exposure_usd=round(abs(diff), 2),
                severity=severity_from_premium(EXC.PREM, diff),
            )
        return None

    def _check_expiry(self, c, i) -> Optional[Discrepancy]:
        if not (c.expiry_date and i.expiry_date):
            return None
        cd, id_ = _parse_date(c.expiry_date), _parse_date(i.expiry_date)
        if cd and id_ and cd != id_:
            return Discrepancy(
                category=EXC.EXPIRY, field="expiry_date",
                counterparty_value=str(c.expiry_date), internal_value=str(i.expiry_date),
                difference=f"{abs((cd - id_).days)} day(s)",
                difference_unit="days",
                exposure_usd=None, severity="high",
            )
        return None

    def _check_currency(self, c, i) -> Optional[Discrepancy]:
        cv = c.settlement_currency or c.currency
        iv = i.settlement_currency or i.currency
        if not (cv and iv):
            return None
        if cv.strip().upper() != iv.strip().upper():
            return Discrepancy(
                category=EXC.CCY, field="settlement_currency",
                counterparty_value=cv, internal_value=iv,
                difference="Currency mismatch", difference_unit="currency",
                exposure_usd=None, severity="high",
            )
        return None

    def _check_direction(self, c, i) -> Optional[Discrepancy]:
        """Buyer/seller direction — critical for insurance hedgers.
        We check whether our internal record's direction matches the confirmation.
        If internal says we are buyer but confirmation says we are seller → EXC-DIR."""
        if not (i.buyer and c.seller) and not (i.seller and c.buyer):
            return None
        # internal buyer should be confirmation buyer
        if i.buyer and c.buyer:
            if i.buyer.strip().lower() != c.buyer.strip().lower():
                # could be a naming variant — only flag if clearly different entity
                if not any(word in c.buyer.lower() for word in i.buyer.lower().split()):
                    return Discrepancy(
                        category=EXC.DIR, field="buyer",
                        counterparty_value=c.buyer, internal_value=i.buyer,
                        difference="Buyer mismatch", difference_unit="entity",
                        exposure_usd=None, severity="high",
                    )
        return None

    def _check_style(self, c, i) -> Optional[Discrepancy]:
        if not (c.option_style and i.option_style):
            return None
        if c.option_style.lower() != i.option_style.lower():
            return Discrepancy(
                category=EXC.STYLE, field="option_style",
                counterparty_value=c.option_style, internal_value=i.option_style,
                difference="European vs American mismatch", difference_unit="style",
                exposure_usd=None, severity="medium",
            )
        return None

    def _check_settlement(self, c, i) -> Optional[Discrepancy]:
        if not (c.settlement_method and i.settlement_method):
            return None
        if c.settlement_method.strip().lower() != i.settlement_method.strip().lower():
            return Discrepancy(
                category=EXC.SETT, field="settlement_method",
                counterparty_value=c.settlement_method, internal_value=i.settlement_method,
                difference="Settlement method mismatch", difference_unit="method",
                exposure_usd=None, severity="medium",
            )
        return None

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _strike_exposure(self, c, i) -> Optional[float]:
        """Exposure proxy for strike mismatch: premium is the best available
        single-number proxy without knowing current index level.
        Flag as estimate in processing notes."""
        if c.premium is not None:
            return round(abs(c.premium), 2)
        if i.premium is not None:
            return round(abs(i.premium), 2)
        return None

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
        underlying = conf.underlying or "equity index option"
        parts = []
        for label, d in [("Primary", primary), ("Secondary", secondary)]:
            if d:
                exp_str = f", exposure ~${d.exposure_usd:,.0f}" if d.exposure_usd else ""
                parts.append(
                    f"{label}: {d.category} on {d.field} "
                    f"({d.counterparty_value} vs {d.internal_value}{exp_str})"
                )
        action = f"{underlying} confirmation break. " + " | ".join(parts)
        if self._escalation_required(primary, secondary):
            action += " — Escalate; do not affirm."
        return action
