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

import os, uuid, time, json, re, asyncio, random, logging
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

AAL_LLM_MAX_ATTEMPTS         = int(os.environ.get("AAL_LLM_MAX_ATTEMPTS", "4"))          # 1 initial + 3 retries
AAL_LLM_BACKOFF_BASE_S       = float(os.environ.get("AAL_LLM_BACKOFF_BASE_S", "0.5"))
AAL_LLM_BACKOFF_MULTIPLIER   = float(os.environ.get("AAL_LLM_BACKOFF_MULTIPLIER", "2.0"))
AAL_LLM_BACKOFF_MAX_S        = float(os.environ.get("AAL_LLM_BACKOFF_MAX_S", "20.0"))
AAL_LLM_RETRY_AFTER_JITTER_S = float(os.environ.get("AAL_LLM_RETRY_AFTER_JITTER_S", "1.0"))
AAL_LLM_TIMEOUT_S            = float(os.environ.get("AAL_LLM_TIMEOUT_S", "30.0"))

client     = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, max_retries=0, timeout=AAL_LLM_TIMEOUT_S) if ANTHROPIC_API_KEY else None
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

def norm_float(r):
    if not r:
        return r
    try:
        return FLOAT_ALIASES.get(str(r).strip().lower(), str(r) if not isinstance(r, str) else r)
    except Exception:
        return r

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
log = logging.getLogger("aal.extract")

RETRYABLE_LLM_EXC = (
    anthropic.APIConnectionError,   # network failure; APITimeoutError is a subclass
    anthropic.RateLimitError,       # HTTP 429
    anthropic.InternalServerError,  # HTTP >=500, incl. 529 overloaded_error
)

def _retry_after_seconds(exc) -> float | None:
    resp = getattr(exc, "response", None)
    if resp is None:
        return None
    raw = resp.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)  # numeric-seconds form; HTTP-date form falls back to backoff
    except ValueError:
        return None

async def _messages_create_with_retry(**kwargs):
    last_exc = None
    for attempt in range(AAL_LLM_MAX_ATTEMPTS):
        try:
            return await asyncio.to_thread(client.messages.create, **kwargs)
        except RETRYABLE_LLM_EXC as e:
            last_exc = e
            if attempt == AAL_LLM_MAX_ATTEMPTS - 1:
                break
            ra = _retry_after_seconds(e)
            if ra is not None:  # honor server directive + small spread
                delay = min(ra, AAL_LLM_BACKOFF_MAX_S) + random.uniform(0, AAL_LLM_RETRY_AFTER_JITTER_S)
            else:               # exponential backoff with full jitter
                cap = min(AAL_LLM_BACKOFF_MAX_S,
                          AAL_LLM_BACKOFF_BASE_S * (AAL_LLM_BACKOFF_MULTIPLIER ** attempt))
                delay = random.uniform(0, cap)
            log.warning("LLM retry %d/%d in %.2fs after %s: %s",
                        attempt + 1, AAL_LLM_MAX_ATTEMPTS - 1, delay, type(e).__name__, e)
            await asyncio.sleep(delay)
    raise last_exc

async def extract_fields(text: str, asset_class: str) -> tuple[dict, float]:
    if not client:
        raise HTTPException(status_code=503, detail="LLM client not configured")
    prompt = PROMPTS.get(asset_class, PROMPTS["IRS"])
    try:
        msg = await _messages_create_with_retry(
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
        # Sanitize: coerce any non-string values in known string fields to str
        # Prevents 'float object has no attribute strip' in reconciler from_dict methods
        STRING_FIELDS = {
            "counterparty", "trade_date", "effective_date", "maturity_date",
            "termination_date", "settlement_date", "expiry_date", "ref_maturity",
            "floating_rate", "currency", "settlement_currency", "local_currency",
            "direction", "option_type", "option_style", "underlying", "structure_type",
            "leg_type", "day_count", "day_count_fixed", "day_count_float",
            "payment_frequency", "payment_frequency_fixed", "payment_frequency_float",
            "valuation_frequency", "ref_cusip", "ref_isin", "ref_issuer",
            "ref_security_type", "seniority", "settlement_method", "usi",
            "reference_entity", "protection_buyer", "protection_seller",
            "ref_obligor", "cusip", "isin", "issuer", "security_desc",
            "capacity", "coupon_frequency", "rating", "buyer", "seller",
            "premium_payment_date", "policy_cohort", "hedge_program",
        }
        for k in list(extracted.keys()):
            if k in STRING_FIELDS and extracted[k] is not None and not isinstance(extracted[k], str):
                extracted[k] = str(extracted[k])
        for fld in ("floating_rate",):
            if fld in extracted and asset_class in ("IRS", "CAP_FLOOR", "TRS"):
                val = extracted.get(fld)
                if val is not None:
                    extracted[fld] = norm_float(val)
        if asset_class == "EQUITY_OPTION" and "underlying" in extracted:
            extracted["underlying"] = normalize_index(extracted["underlying"])
        return extracted, confidence
    except HTTPException:
        raise
    except RETRYABLE_LLM_EXC as e:
        raise HTTPException(status_code=502,
            detail=f"LLM extraction failed after {AAL_LLM_MAX_ATTEMPTS} attempts: {type(e).__name__}: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM extraction failed: {type(e).__name__}: {e}")

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

    extracted, confidence = await extract_fields(req.confirmation_text, asset_class)
    internal_dict = req.expected_economics.model_dump(exclude_none=True)
    engine = ENGINES[asset_class]
    try:
        result = engine.reconcile(extracted, internal_dict, case_id=match_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM extraction failed: {e} | extracted={extracted}")

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

# ---------------------------------------------------------------------------
# v3.1 — Raw ingestion endpoint (MarkitWire FIXML or PDF text)
# ---------------------------------------------------------------------------
from fixml_parser import parse_fixml

class ValidateRawRequest(BaseModel):
    source_type:   Literal["markitwire", "pdf_text"] = "markitwire"
    source_data:   str = Field(..., description="Raw FIXML string or extracted PDF text")
    internal_type: Literal["markitwire", "pdf_text", "json"] = "json"
    internal_data: str = Field(..., description="Internal record as FIXML, PDF text, or JSON string")
    options:       ValidateOptions = ValidateOptions()

@app.post(
    "/v1/confirmations/validate-raw",
    response_model=ValidateResponse,
    tags=["Confirmations"],
    dependencies=[Depends(require_api_key)],
)
async def validate_raw(req: ValidateRawRequest):
    """
    Accept raw MarkitWire FIXML or PDF text for both counterparty confirmation
    and internal record. Parses both sides, routes to the correct engine.

    source_type / internal_type:
      markitwire  — raw FIXML string (deterministic parse, no LLM)
      pdf_text    — raw text extracted from a PDF confirmation (LLM extraction)
      json        — pre-structured JSON matching ExpectedEconomics fields
    """
    t0 = time.time()
    match_id = str(uuid.uuid4())

    # --- Parse counterparty side ---
    if req.source_type == "markitwire":
        try:
            cpty_dict = parse_fixml(req.source_data)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f"FIXML parse error: {e}")
        asset_class = cpty_dict.pop("asset_class")
        extracted   = cpty_dict
        confidence  = 1.0  # deterministic parse
    elif req.source_type == "pdf_text":
        # Detect asset class via LLM first, then extract
        asset_class = "IRS"  # default; TODO: add asset class detection step
        extracted, confidence = await extract_fields(req.source_data, asset_class)
    else:
        raise HTTPException(status_code=422, detail="Unsupported source_type")

    # --- Parse internal side ---
    if req.internal_type == "json":
        try:
            internal_dict = json.loads(req.internal_data)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=422, detail=f"Internal JSON parse error: {e}")
    elif req.internal_type == "markitwire":
        try:
            internal_dict = parse_fixml(req.internal_data)
            internal_dict.pop("asset_class", None)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f"Internal FIXML parse error: {e}")
    elif req.internal_type == "pdf_text":
        internal_dict, _ = await extract_fields(req.internal_data, asset_class)
    else:
        raise HTTPException(status_code=422, detail="Unsupported internal_type")

    # --- Reconcile ---
    engine = ENGINES.get(asset_class)
    if not engine:
        raise HTTPException(status_code=422, detail=f"Unsupported asset class: {asset_class}")

    try:
        result = engine.reconcile(extracted, internal_dict, case_id=match_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Reconciliation failed: {e}")

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
        metadata=None,
    )

# ---------------------------------------------------------------------------
# v3.1 — Batch endpoint
# ---------------------------------------------------------------------------
import asyncio
from typing import List

class BatchItem(BaseModel):
    source_type:   Literal["markitwire", "pdf_text"] = "markitwire"
    source_data:   str
    internal_type: Literal["markitwire", "pdf_text", "json"] = "json"
    internal_data: str

class BatchRequest(BaseModel):
    items:           List[BatchItem] = Field(..., max_length=500)
    options:         ValidateOptions = ValidateOptions()
    max_concurrent:  int = Field(default=10, ge=1, le=50)

class BatchSummary(BaseModel):
    batch_id:      str
    total:         int
    clean:         int
    discrepant:    int
    review_required: int
    errors:        int
    processing_ms: int
    results:       list[ValidateResponse]

async def _process_one(item: BatchItem, options: ValidateOptions) -> ValidateResponse:
    req = ValidateRawRequest(
        source_type=item.source_type,
        source_data=item.source_data,
        internal_type=item.internal_type,
        internal_data=item.internal_data,
        options=options,
    )
    return await validate_raw(req)

@app.post(
    "/v1/confirmations/batch",
    response_model=BatchSummary,
    tags=["Confirmations"],
    dependencies=[Depends(require_api_key)],
)
async def validate_batch(req: BatchRequest):
    """
    Process up to 500 reconciliations in a single call.
    Items run concurrently (default 10 at a time).
    Returns all results plus a summary breakdown.
    """
    t0 = time.time()
    batch_id = str(uuid.uuid4())
    semaphore = asyncio.Semaphore(req.max_concurrent)

    async def _guarded(item: BatchItem) -> ValidateResponse | dict:
        async with semaphore:
            try:
                return await _process_one(item, req.options)
            except HTTPException as e:
                return {"error": e.detail}
            except Exception as e:
                return {"error": str(e)}

    results_raw = await asyncio.gather(*[_guarded(item) for item in req.items])

    results   = [r for r in results_raw if isinstance(r, dict) and "match_id" in r or isinstance(r, ValidateResponse)]
    errors    = [r for r in results_raw if isinstance(r, dict) and "error" in r]
    valid     = [r for r in results_raw if isinstance(r, ValidateResponse)]

    clean           = sum(1 for r in valid if r.overall_status == "clean")
    discrepant      = sum(1 for r in valid if r.overall_status == "discrepant")
    review_required = sum(1 for r in valid if r.overall_status == "review_required")

    return BatchSummary(
        batch_id=batch_id,
        total=len(req.items),
        clean=clean,
        discrepant=discrepant,
        review_required=review_required,
        errors=len(errors),
        processing_ms=int((time.time() - t0) * 1000),
        results=valid,
    )

# ---------------------------------------------------------------------------
# v3.2 — Margin Call Dispute Analysis endpoint
# ---------------------------------------------------------------------------

DISPUTE_SYSTEM = """You are an expert capital markets collateral operations analyst specializing in margin call dispute classification.

You will be given a margin call case containing:
- A margin call notice from a counterparty
- The firm's internal calculation
- Context including CSA terms and portfolio summary

Your job is CLASSIFICATION and EXTRACTION ONLY. You do NOT calculate, sum, subtract, net, or reconcile any dollar amounts. The system computes every numeric amount deterministically from the two structured inputs. You only identify the dispute category, the field in dispute, and (where noted) extract a rate that is explicitly stated in the CSA text.

DISPUTE CATEGORIES (use exactly these codes):
- DIS-PRICE: MTM/portfolio valuation difference between parties
- DIS-QTY: Quantity, contract count, or notional disagrees between parties
- DIS-HAIRCUT: Collateral IS eligible under the CSA but the wrong haircut percentage was applied. Do NOT use for ineligible collateral - use DIS-ELGBLTY.
- DIS-ELGBLTY: The collateral posted is ineligible under the CSA (wrong asset type, rating, or currency). Do NOT use when eligible collateral has a wrong haircut - use DIS-HAIRCUT.
- DIS-FX: FX rate used for currency conversion or collateral valuation differs from the CSA-specified source
- DIS-THRESH: Threshold or MTA not correctly applied (call below MTA, threshold not deducted, overcollateralization after threshold)
- DIS-TIMING: Settlement date or calculation date error
- DIS-NETTING: Netting set constructed incorrectly - wrong trades included/excluded, or collateral misallocated between CSAs
- DIS-CPTY: Counterparty identity or legal entity mismatch
- DIS-CSA: CSA term disagreement (rounding convention, MTA amendment, collateral currency, call currency, margin ratio)
- DIS-CALC: Calculation error not attributable to another category
- DIS-SIMM: ISDA SIMM methodology dispute - different inputs, versions, or risk factor sensitivities
- DIS-STALE: Stale or unavailable pricing used by one party
- DIS-DUPE: Duplicate margin call - same calculation date, amount, and value date as a prior affirmed call
- DIS-SETTLED: Collateral transfer not yet reflected in counterparty records
- DIS-DIR: Call direction is wrong - counterparty calling for payment in the wrong direction
- DIS-CLEAN: No dispute exists; the call is correct

CLASSIFICATION RULES (these decide the CODE, never a number):
1. DIS-HAIRCUT vs DIS-ELGBLTY: eligible collateral + wrong haircut rate -> DIS-HAIRCUT. Ineligible collateral type -> DIS-ELGBLTY.
2. DIS-NETTING vs DIS-ELGBLTY: eligible collateral in the wrong netting set -> DIS-NETTING. Ineligible collateral -> DIS-ELGBLTY.
3. If the two parties report the same amount from different sources, that is NOT a dispute -> DIS-CLEAN.
4. A collateral or valuation difference below the MTA is NOT a dispute -> DIS-CLEAN.
5. For DIS-THRESH: if the entire call is invalid (below MTA, or threshold makes the correct call zero), set "field" to "call_amount". Otherwise set "field" to the specific threshold/mta field.

FIELD: name the single structured input field most in dispute (e.g. "portfolio_mtm", "collateral_on_hand", "simm_amount", "fx_rate_used", "netting_set", "collateral_type", "call_direction", "call_id", "threshold", "contracts"). Use the exact key name from the inputs when one exists.

EXTRACTION (only when category is DIS-HAIRCUT):
- Read the CSA-required haircut for the disputed collateral directly from the CSA terms / notes text and report it as a decimal in "csa_haircut_rate" (e.g. 2% -> 0.02). If the CSA text does not state a rate, use null. Do NOT compute anything - only copy the stated rate.

ESCALATION TARGET: name the human/desk to escalate to in plain English, or null. This is a judgment, not a number.

Respond with THIS EXACT JSON and nothing else:
{
  "dispute_exists": true or false,
  "primary": {"category": "DIS-CODE", "field": "field_name"} or null,
  "secondary": {"category": "DIS-CODE", "field": "field_name"} or null,
  "csa_haircut_rate": <decimal or null>,
  "escalation_target": "<who, or null>"
}
If there is no dispute: dispute_exists=false, primary=null, secondary=null, csa_haircut_rate=null, escalation_target=null."""


class MarginCallNotice(BaseModel):
    model_config = {"extra": "allow"}
    call_id:          Optional[str]   = None
    call_date:        Optional[str]   = None
    value_date:       Optional[str]   = None
    counterparty:     Optional[str]   = None
    call_direction:   Optional[str]   = None
    call_currency:    Optional[str]   = None
    call_amount:      Optional[float] = None
    calculation_date: Optional[str]   = None
    portfolio_mtm:    Optional[float] = None
    threshold:        Optional[float] = None
    mta:              Optional[float] = None
    rounding:         Optional[float] = None
    collateral_on_hand: Optional[float] = None
    pricing_source:   Optional[str]   = None

class InternalCalculation(BaseModel):
    model_config = {"extra": "allow"}
    call_date:          Optional[str]   = None
    value_date:         Optional[str]   = None
    counterparty:       Optional[str]   = None
    call_direction:     Optional[str]   = None
    call_currency:      Optional[str]   = None
    calculated_amount:  Optional[float] = None
    calculation_date:   Optional[str]   = None
    portfolio_mtm:      Optional[float] = None
    threshold:          Optional[float] = None
    mta:                Optional[float] = None
    rounding:           Optional[float] = None
    collateral_on_hand: Optional[float] = None
    pricing_source:     Optional[str]   = None

class DisputeContext(BaseModel):
    csa_terms:         Optional[str] = None
    portfolio_summary: Optional[str] = None

class DisputeRequest(BaseModel):
    margin_call_notice:  MarginCallNotice
    internal_calculation: InternalCalculation
    context:             DisputeContext = DisputeContext()
    case_id:             Optional[str] = None

class DisputeDetail(BaseModel):
    category:            Optional[str]   = None
    field:               Optional[str]   = None
    counterparty_value:  Optional[float] = None
    internal_value:      Optional[float] = None
    difference:          Optional[float] = None
    disputed_call_amount: Optional[float] = None
    correct_call_amount: Optional[float] = None

class DisputeResponse(BaseModel):
    case_id:             Optional[str]
    dispute_exists:      bool
    primary_dispute:     Optional[DisputeDetail]
    secondary_dispute:   Optional[DisputeDetail]
    recommended_action:  str
    escalation_required: bool
    escalation_target:   Optional[str]
    pay_undisputed:      bool
    undisputed_amount:   Optional[float]
    extraction_confidence: float
    processing_ms:       int
    model_version:       str


# --- deterministic dispute arithmetic (no LLM numbers past this point) -------
_NO_PAY_CATEGORIES = {"DIS-DUPE", "DIS-DIR", "DIS-ELGBLTY"}

def _num(v) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) else None

def _round_to(x: float, inc: Optional[float]) -> float:
    if not inc:
        return x
    return round(x / inc) * inc

def _money(x: Optional[float]) -> str:
    return f"${x:,.0f}" if x is not None else "$0"

def _correct_call_amount(category, field, notice, internal, haircut_rate) -> Optional[float]:
    calc = _num(internal.calculated_amount)
    call = _num(notice.call_amount)
    coll = _num(notice.collateral_on_hand)
    if category == "DIS-DUPE":
        return 0.0
    if category == "DIS-DIR":
        return None if calc is None else -calc
    if category == "DIS-FX":
        return call                                  # counterparty rate governs
    if category == "DIS-ELGBLTY":
        return (call + coll) if (call is not None and coll is not None) else calc
    if category == "DIS-THRESH":
        return 0.0 if field == "call_amount" else calc
    if category == "DIS-HAIRCUT":
        rate = _num(haircut_rate)
        if rate is not None and call is not None and coll is not None:
            inc = _num(notice.rounding) or _num(internal.rounding) or 0
            return _round_to(call + coll * rate, inc)
        return calc                                  # fallback: no rate extractable
    return calc                                      # PRICE/QTY/SETTLED/SIMM/STALE/NETTING/CSA/TIMING/CALC/CPTY

def _disputed_call_amount(category, correct, notice, internal) -> Optional[float]:
    call = _num(notice.call_amount)
    calc = _num(internal.calculated_amount)
    if call is None:
        return None
    if category in ("DIS-DUPE", "DIS-DIR"):
        return call
    if category == "DIS-FX":
        return None if calc is None else call - calc
    if category == "DIS-QTY":
        return None if correct is None else abs(call - correct)
    if correct is None:
        return None
    return call - correct                            # signed: negative for HAIRCUT/ELGBLTY under-calls

def _undisputed_amount(pay, correct, notice) -> Optional[float]:
    if not pay:
        return 0.0
    call = _num(notice.call_amount)
    c = max(0.0, correct) if correct is not None else None
    if call is None:
        return c
    if c is None:
        return call
    return min(call, c)                              # pay the lesser of demanded vs owed

def _pay_undisputed(category, correct, dispute_exists) -> bool:
    if not dispute_exists:
        return True
    if category in _NO_PAY_CATEGORIES:
        return False
    if correct is not None and abs(correct) < 0.5:   # nothing left to pay
        return False
    return True

def _escalation_required(category, correct, dispute_exists) -> bool:
    if not dispute_exists:
        return False
    if category == "DIS-THRESH" and correct is not None and abs(correct) < 0.5:
        return False
    if category == "DIS-TIMING":
        return False
    return True

def _lookup(field, model) -> Optional[float]:
    if not field:
        return None
    return _num(model.model_dump(exclude_none=True).get(field))

def _detail(category, field, notice, internal, correct, disputed) -> DisputeDetail:
    cv = _lookup(field, notice)
    iv = _lookup(field, internal)
    diff = abs(cv - iv) if (cv is not None and iv is not None) else None
    return DisputeDetail(
        category=category, field=field,
        counterparty_value=cv, internal_value=iv, difference=diff,
        disputed_call_amount=disputed, correct_call_amount=correct,
    )

def _recommended_action(dispute_exists, category, pay, disputed, undisputed,
                        call, escalation_target) -> str:
    if not dispute_exists:
        return f"Call is correct. Pay {_money(call)} by value date."
    parts = [f"Dispute {_money(abs(disputed) if disputed is not None else None)} ({category})."]
    if pay:
        parts.append(f"Pay undisputed {_money(undisputed)} by value date to avoid default.")
    else:
        parts.append("Do not pay; the call is invalid pending resolution.")
    if escalation_target:
        parts.append(f"Escalate to {escalation_target}.")
    return " ".join(parts)


@app.post(
    "/v1/disputes/analyze",
    response_model=DisputeResponse,
    tags=["Disputes"],
    dependencies=[Depends(require_api_key)],
)
async def analyze_dispute(req: DisputeRequest):
    """
    Analyze a margin call dispute case.
    Accepts margin_call_notice + internal_calculation + context.
    Returns dispute detection, category, amounts, escalation recommendation.
    Compatible with AAL-D-002 benchmark schema.
    """
    if not client:
        raise HTTPException(status_code=503, detail="LLM client not configured")

    t0 = time.time()
    case_id = req.case_id or str(uuid.uuid4())

    prompt_input = {
        "margin_call_notice":   req.margin_call_notice.model_dump(exclude_none=True),
        "internal_calculation": req.internal_calculation.model_dump(exclude_none=True),
        "context":              req.context.model_dump(exclude_none=True),
    }

    try:
        msg = await _messages_create_with_retry(
            model=MODEL,
            max_tokens=1024,
            system=DISPUTE_SYSTEM,
            messages=[{
                "role": "user",
                "content": "Case:\n" + json.dumps(prompt_input, indent=2)
            }],
        )
        raw = re.sub(r"^```json\s*", "", msg.content[0].text.strip())
        raw = re.sub(r"\s*```$", "", raw)
        pred = _salvage_json(raw)
        if not pred:
            raise HTTPException(status_code=502, detail=f"LLM returned unparseable response: {raw[:200]}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM analysis failed: {e}")

    primary = pred.get("primary") or None
    category = (primary or {}).get("category")
    field    = (primary or {}).get("field")
    dispute_exists = bool(primary) and category != "DIS-CLEAN"

    notice, internal = req.margin_call_notice, req.internal_calculation
    haircut_rate = pred.get("csa_haircut_rate")

    if dispute_exists:
        correct  = _correct_call_amount(category, field, notice, internal, haircut_rate)
        disputed = _disputed_call_amount(category, correct, notice, internal)
        primary_detail = _detail(category, field, notice, internal, correct, disputed)
        sec = pred.get("secondary") or None
        if sec:
            secondary_detail = _detail(sec.get("category"), sec.get("field"),
                                       notice, internal, None, None)
        else:
            secondary_detail = None
    else:
        correct = disputed = None
        primary_detail = secondary_detail = None

    pay = _pay_undisputed(category, correct, dispute_exists)
    esc = _escalation_required(category, correct, dispute_exists)
    call = _num(notice.call_amount)
    undisputed = call if not dispute_exists else _undisputed_amount(pay, correct, notice)
    target = (pred.get("escalation_target") if dispute_exists else None)

    return DisputeResponse(
        case_id=case_id,
        dispute_exists=dispute_exists,
        primary_dispute=primary_detail,
        secondary_dispute=secondary_detail,
        recommended_action=_recommended_action(
            dispute_exists, category, pay, disputed, undisputed, call, target),
        escalation_required=esc,
        escalation_target=target,
        pay_undisputed=pay,
        undisputed_amount=undisputed,
        extraction_confidence=0.9,
        processing_ms=int((time.time() - t0) * 1000),
        model_version=API_VERSION,
    )

