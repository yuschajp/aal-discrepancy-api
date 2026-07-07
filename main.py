#!/usr/bin/env python3
"""
AAL Discrepancy Detection API v3
FastAPI — Full capital markets confirmation discrepancy detection.

Asset classes: IRS, EQUITY_OPTION, FIXED_INCOME, CAP_FLOOR, TRS, CDS

v3 changes:
  - Added FIXED_INCOME, CAP_FLOOR, TRS, CDS engines and extraction prompts
  - Unified asset class router
  - All six engines live behind single endpoint

Architecture (Decision Log 2026-07-04/05):
  POST /v1/confirmations/validate
    → [1] LLM extraction (Claude Haiku) — NER/classification only
    → [2] Asset class router → correct reconciler
    → [3] Severity rule table — no LLM touches arithmetic
    → Response
"""

import os, uuid, time, json, re
from typing import Optional, Literal
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import anthropic

from irs_reconciler          import IRSReconciler
from equity_option_reconciler import EquityOptionReconciler, normalize_index
from fixed_income_reconciler  import FixedIncomeReconciler
from cap_floor_reconciler     import CapFloorReconciler
from trs_reconciler           import TRSReconciler
from cds_reconciler           import CDSReconciler

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
AAL_API_KEY       = os.environ.get("AAL_API_KEY", "dev-key-replace-in-prod")
MODEL             = "claude-haiku-4-5-20251001"
API_VERSION       = "aal-disc-v3.0"

client     = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
ENGINES    = {
    "IRS":           IRSReconciler(),
    "EQUITY_OPTION": EquityOptionReconciler(),
    "FIXED_INCOME":  FixedIncomeReconciler(),
    "CAP_FLOOR":     CapFloorReconciler(),
    "TRS":           TRSReconciler(),
    "CDS":           CDSReconciler(),
}

ASSET_CLASSES = list(ENGINES.keys())

app = FastAPI(
    title="AAL Discrepancy Detection API",
    description="Deterministic capital markets confirmation discrepancy detection. "
                "Covers IRS, equity options, fixed income, caps/floors, TRS, and CDS.",
    version="3.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def require_api_key(key: str = Security(api_key_header)):
    if key != AAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ExpectedEconomics(BaseModel):
    asset_class: Literal[
        "IRS", "EQUITY_OPTION", "FIXED_INCOME", "CAP_FLOOR", "TRS", "CDS"
    ] = "IRS"
    # Shared
    counterparty:    Optional[str]   = None
    trade_date:      Optional[str]   = None
    currency:        Optional[str]   = None
    notional:        Optional[float] = None
    effective_date:  Optional[str]   = None
    # IRS
    fixed_rate:              Optional[float] = None
    floating_rate:           Optional[str]   = None
    maturity_date:           Optional[str]   = None
    payment_frequency_fixed: Optional[str]   = None
    payment_frequency_float: Optional[str]   = None
    day_count_fixed:         Optional[str]   = None
    day_count_float:         Optional[str]   = None
    # Equity option
    option_type:         Optional[str]   = None
    option_style:        Optional[str]   = None
    underlying:          Optional[str]   = None
    strike:              Optional[float] = None
    strike_high:         Optional[float] = None
    num_options:         Optional[float] = None
    premium:             Optional[float] = None
    premium_per_option:  Optional[float] = None
    expiry_date:         Optional[str]   = None
    settlement_currency: Optional[str]   = None
    settlement_method:   Optional[str]   = None
    buyer:               Optional[str]   = None
    seller:              Optional[str]   = None
    # Fixed income
    cusip:               Optional[str]   = None
    isin:                Optional[str]   = None
    direction:           Optional[str]   = None
    par_value:           Optional[float] = None
    price:               Optional[float] = None
    yield_rate:          Optional[float] = None
    coupon_rate:         Optional[float] = None
    settlement_date:     Optional[str]   = None
    accrued_interest:    Optional[float] = None
    principal_amount:    Optional[float] = None
    total_consideration: Optional[float] = None
    # Cap/floor
    structure_type:      Optional[str]   = None
    cap_rate:            Optional[float] = None
    floor_rate:          Optional[float] = None
    termination_date:    Optional[str]   = None
    payment_frequency:   Optional[str]   = None
    day_count:           Optional[str]   = None
    # TRS
    ref_cusip:           Optional[str]   = None
    ref_isin:            Optional[str]   = None
    ref_issuer:          Optional[str]   = None
    ref_coupon:          Optional[float] = None
    ref_maturity:        Optional[str]   = None
    initial_price:       Optional[float] = None
    initial_spot_rate:   Optional[float] = None
    local_notional:      Optional[float] = None
    local_currency:      Optional[str]   = None
    spread:              Optional[float] = None
    leg_type:            Optional[str]   = None
    valuation_frequency: Optional[str]   = None
    # CDS
    reference_entity:    Optional[str]   = None
    protection_buyer:    Optional[str]   = None
    protection_seller:   Optional[str]   = None
    seniority:           Optional[str]   = None
    ref_obligor:         Optional[str]   = None
    credit_events:       Optional[list]  = None
    payment_requirement: Optional[float] = None
    default_requirement: Optional[float] = None
    reference_price:     Optional[float] = None
    # Insurance metadata
    policy_cohort:       Optional[str]   = None
    hedge_program:       Optional[str]   = None

class ValidateOptions(BaseModel):
    severity_threshold:    Literal["low", "medium", "high"] = "low"
    return_raw_extraction: bool = False

class ValidateRequest(BaseModel):
    confirmation_text:  str = Field(..., description="Raw confirmation text")
    expected_economics: ExpectedEconomics
    options:            ValidateOptions = ValidateOptions()

class DiscrepancyDetail(BaseModel):
    field:                 str
    category:              str
    expected:              object
    extracted:             object
    difference:            object
    severity:              str
    exposure_estimate_usd: Optional[float]
    confidence:            float

class ValidateResponse(BaseModel):
    match_id:              str
    asset_class:           str
    overall_status:        Literal["clean", "discrepant", "review_required"]
    discrepancies:         list[DiscrepancyDetail]
    overall_severity:      str
    escalation_required:   bool
    recommended_action:    str
    extraction_confidence: float
    processing_ms:         int
    model_version:         str
    raw_extraction:        Optional[dict] = None
    metadata:              Optional[dict] = None

class HealthResponse(BaseModel):
    status:        str
    version:       str
    model:         str
    llm_ready:     bool
    asset_classes: list[str]

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
FLOAT_ALIASES = {
    "usd-sofr": "SOFR", "sofr compound": "SOFR", "sofr-compound": "SOFR",
    "sofr ois": "SOFR", "us sofr": "SOFR",
    "usd-libor-bba": "LIBOR-3M", "usd-libor-bba-3m": "LIBOR-3M",
    "usd-libor-bba-6m": "LIBOR-6M", "usd-libor-bba-1m": "LIBOR-1M",
    "usd-libor-bba-12m": "LIBOR-12M", "usd-libor": "LIBOR-3M",
    "libor": "LIBOR-3M", "libor-bba": "LIBOR-3M",
    "eur-euribor-reuters": "EURIBOR-6M", "euribor": "EURIBOR-6M",
    "gbp-sonia": "SONIA", "sonia": "SONIA",
    "eur-estr": "ESTR", "estr": "ESTR",
    "usd-federal funds-ois compound": "OIS", "ois": "OIS", "fed funds": "OIS",
}

def norm_float(r): return FLOAT_ALIASES.get((r or "").strip().lower(), r) if r else r

# ---------------------------------------------------------------------------
# Extraction prompts — one per asset class
# ---------------------------------------------------------------------------
PROMPTS = {
"IRS": """Extract IRS confirmation fields as JSON only. Fields:
trade_id, trade_date(YYYY-MM-DD), effective_date, maturity_date, counterparty,
notional(number), currency, fixed_rate(% e.g.4.125), floating_rate(normalized:SOFR/LIBOR-3M/etc),
floating_spread, payment_frequency_fixed(Annual/Semi-Annual/Quarterly),
payment_frequency_float, day_count_fixed(30/360|ACT/360|ACT/365),
day_count_float, usi, extraction_confidence(0-1).
Rules: fixed_rate as % not decimal. Dates YYYY-MM-DD. Normalize floating rate.
Counterparty=the other party not the dealer. Amortizing: use initial notional.""",

"EQUITY_OPTION": """Extract equity option confirmation fields as JSON only. Fields:
trade_id, trade_date(YYYY-MM-DD), counterparty, option_type(call/put),
option_style(european/american), underlying(SPX/RTY/NDX/MSCI_EAFE/MSCI_EM/SX5E/NKY),
strike(number), strike_high(for spreads), num_options, notional, currency,
premium(total$), premium_per_option, premium_payment_date, expiry_date,
settlement_currency, settlement_method(cash/physical), buyer, seller, extraction_confidence.
Rules: normalize underlying. option_type lowercase. Party A=dealer=counterparty.""",

"FIXED_INCOME": """Extract fixed income trade confirmation fields as JSON only. Fields:
trade_id, trade_date(YYYY-MM-DD), cusip(uppercase no spaces), isin,
issuer, security_desc, counterparty, direction(buy/sell), capacity(principal/agent),
par_value, currency, price(per 100), yield_rate(%), coupon_rate(%),
coupon_frequency(semi-annual/quarterly/monthly), maturity_date, settlement_date,
accrued_interest($), principal_amount($), total_consideration($),
day_count(30/360|ACT/ACT|ACT/365), callable(true/false), rating, extraction_confidence.
Rules: CUSIP uppercase no spaces. direction lowercase. Dates YYYY-MM-DD.""",

"CAP_FLOOR": """Extract interest rate cap/floor confirmation fields as JSON only. Fields:
trade_id, trade_date(YYYY-MM-DD), structure_type(cap/floor/collar),
counterparty, notional, currency, cap_rate(%), floor_rate(%),
floating_rate(normalized), floating_tenor(3M/6M/1M), day_count(ACT/360|30/360),
effective_date, termination_date, payment_frequency(Quarterly/Semi-Annual/Monthly),
premium($), premium_date, buyer, seller, extraction_confidence.
Rules: structure_type lowercase. Normalize floating rate. Dates YYYY-MM-DD.""",

"TRS": """Extract total return swap confirmation fields as JSON only. Fields:
trade_id, trade_date, counterparty, ref_cusip(uppercase), ref_isin(uppercase),
ref_issuer, ref_security_type, ref_coupon(%), ref_maturity(YYYY-MM-DD),
initial_price(per 100), seniority(senior/subordinated),
notional(USD amount), local_notional, local_currency, initial_spot_rate,
currency, direction(receiver/payer — total return perspective),
leg_type(floating/fixed), floating_rate(normalized), spread(%),
fixed_rate(%), day_count, payment_frequency, effective_date,
termination_date, valuation_frequency(Monthly/Quarterly/Annual), extraction_confidence.
Rules: ISIN/CUSIP uppercase. Normalize floating. Dates YYYY-MM-DD.
direction=receiver means we receive total return. Party A=dealer=counterparty.""",

"CDS": """Extract credit default swap confirmation fields as JSON only. Fields:
trade_id, trade_date(YYYY-MM-DD), effective_date, termination_date,
counterparty, reference_entity(exact legal name),
protection_buyer(legal entity), protection_seller(legal entity),
notional($), currency, fixed_rate(CDS spread % per annum, e.g.1.50 for 150bp),
day_count(ACT/360|ACT/365), payment_frequency,
seniority(senior/subordinated), ref_cusip(uppercase), ref_isin,
ref_obligor, ref_coupon(%), ref_maturity(YYYY-MM-DD),
credit_events(list: bankruptcy/failure to pay/obligation default/
obligation acceleration/repudiation_moratorium/restructuring/governmental intervention),
payment_requirement($), default_requirement($),
settlement_method(auction/physical/cash), extraction_confidence.
Rules: fixed_rate as % (1.50 not 0.015). Party A=dealer=counterparty.
Reference entity = exact legal name. credit_events as lowercase list.""",
}

# ---------------------------------------------------------------------------
# Salvage JSON parser
# ---------------------------------------------------------------------------
def _salvage_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        if start == -1:
            return {}
        depth = 0; in_str = False; esc = False
        for i in range(start, len(raw)):
            ch = raw[i]
            if in_str:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_str = False
                continue
            if ch == '"': in_str = True
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try: return json.loads(raw[start:i+1])
                    except: return {}
        return {}

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_fields(text: str, asset_class: str) -> tuple[dict, float]:
    if not client:
        raise HTTPException(status_code=503, detail="LLM client not configured")
    prompt = PROMPTS.get(asset_class, PROMPTS["IRS"])
    try:
        msg = client.messages.create(
            model=MODEL, max_tokens=1024, system=prompt,
            messages=[{"role": "user", "content": f"Parse this {asset_class} confirmation:\n\n{text}"}],
        )
        raw = re.sub(r"^```json\s*", "", msg.content[0].text.strip())
        raw = re.sub(r"\s*```$", "", raw)
        extracted = _salvage_json(raw)
        try:
            confidence = float(extracted.pop("extraction_confidence", 0.85))
        except (ValueError, TypeError):
            extracted.pop("extraction_confidence", None)
            confidence = 0.85
        # Post-extraction normalization
        for fld in ("floating_rate", "spread"):
            if fld in extracted and asset_class in ("IRS", "CAP_FLOOR", "TRS"):
                extracted[fld] = norm_float(extracted.get(fld))
        if asset_class == "EQUITY_OPTION" and "underlying" in extracted:
            extracted["underlying"] = normalize_index(extracted["underlying"])
        return extracted, confidence
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM extraction failed: {e}")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="ok", version=API_VERSION, model=MODEL,
        llm_ready=client is not None,
        asset_classes=ASSET_CLASSES,
    )

@app.post(
    "/v1/confirmations/validate",
    response_model=ValidateResponse,
    tags=["Confirmations"],
    dependencies=[Depends(require_api_key)],
)
async def validate_confirmation(req: ValidateRequest):
    t0 = time.time()
    match_id = str(uuid.uuid4())
    asset_class = req.expected_economics.asset_class

    extracted, confidence = extract_fields(req.confirmation_text, asset_class)
    internal_dict = req.expected_economics.model_dump(exclude_none=True)
    engine = ENGINES[asset_class]
    result = engine.reconcile(extracted, internal_dict, case_id=match_id)

    threshold_rank = {"low": 0, "medium": 1, "high": 2}
    threshold = req.options.severity_threshold
    discrepancies = []
    for d in [result.primary, result.secondary]:
        if d is None:
            continue
        if threshold_rank.get(d.severity, 0) >= threshold_rank.get(threshold, 0):
            discrepancies.append(DiscrepancyDetail(
                field=d.field, category=d.category,
                expected=d.internal_value, extracted=d.counterparty_value,
                difference=d.difference, severity=d.severity,
                exposure_estimate_usd=d.exposure_usd, confidence=d.confidence,
            ))

    overall_status = (
        "clean" if not result.exception_exists else
        "discrepant" if result.overall_severity == "high" else
        "review_required"
    )

    metadata = None
    if req.expected_economics.policy_cohort or req.expected_economics.hedge_program:
        metadata = {k: v for k, v in {
            "policy_cohort": req.expected_economics.policy_cohort,
            "hedge_program": req.expected_economics.hedge_program,
        }.items() if v}

    return ValidateResponse(
        match_id=match_id,
        asset_class=asset_class,
        overall_status=overall_status,
        discrepancies=discrepancies,
        overall_severity=result.overall_severity,
        escalation_required=result.escalation_required,
        recommended_action=result.recommended_action,
        extraction_confidence=confidence,
        processing_ms=int((time.time() - t0) * 1000),
        model_version=API_VERSION,
        raw_extraction=extracted if req.options.return_raw_extraction else None,
        metadata=metadata,
    )
