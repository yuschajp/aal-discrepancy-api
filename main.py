#!/usr/bin/env python3
"""
AAL Discrepancy Detection API v2
FastAPI — IRS + Equity Option confirmation discrepancy detection.

v2 changes:
  - EQUITY_OPTION asset class routing to EquityOptionReconciler
  - Equity option extraction prompt (index options, call spreads)
  - Asset-class-aware ExpectedEconomics model
  - Both engines share the same endpoint, response envelope, and auth

Architecture (Decision Log 2026-07-04/05):
  POST /v1/confirmations/validate
    → [1] LLM extraction (Claude Haiku) — NER/classification only
    → [2] Asset class router → IRS or EQUITY_OPTION reconciler
    → [3] Severity rule table — no LLM touches arithmetic
    → Response
"""

import os, uuid, time, json, re
from typing import Optional, Literal, Union
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import anthropic
from irs_reconciler import IRSReconciler
from equity_option_reconciler import EquityOptionReconciler, normalize_index

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
AAL_API_KEY       = os.environ.get("AAL_API_KEY", "dev-key-replace-in-prod")
MODEL             = "claude-haiku-4-5-20251001"
API_VERSION       = "aal-disc-v2.0"

client      = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
irs_engine  = IRSReconciler()
eq_engine   = EquityOptionReconciler()

app = FastAPI(
    title="AAL Discrepancy Detection API",
    description="Deterministic IRS and equity option confirmation discrepancy detection.",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aialphalabs.ai", "http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def require_api_key(key: str = Security(api_key_header)):
    if key != AAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class ExpectedEconomics(BaseModel):
    # Shared
    asset_class:   Literal["IRS", "EQUITY_OPTION"] = "IRS"
    counterparty:  Optional[str]   = None
    trade_date:    Optional[str]   = None
    currency:      Optional[str]   = None
    notional:      Optional[float] = None
    # IRS fields
    fixed_rate:              Optional[float] = None
    floating_rate:           Optional[str]   = None
    effective_date:          Optional[str]   = None
    maturity_date:           Optional[str]   = None
    payment_frequency_fixed: Optional[str]   = None
    payment_frequency_float: Optional[str]   = None
    day_count_fixed:         Optional[str]   = None
    day_count_float:         Optional[str]   = None
    # Equity option fields
    option_type:        Optional[str]   = None   # "call" | "put"
    option_style:       Optional[str]   = None   # "european" | "american"
    underlying:         Optional[str]   = None   # "SPX" | "RTY" etc
    strike:             Optional[float] = None
    strike_high:        Optional[float] = None
    num_options:        Optional[float] = None
    premium:            Optional[float] = None
    premium_per_option: Optional[float] = None
    expiry_date:        Optional[str]   = None
    settlement_currency: Optional[str] = None
    settlement_method:  Optional[str]   = None
    buyer:              Optional[str]   = None
    seller:             Optional[str]   = None
    # Insurance metadata (pass-through, not reconciled)
    policy_cohort:      Optional[str]   = None
    hedge_program:      Optional[str]   = None

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
    status:         str
    version:        str
    model:          str
    llm_ready:      bool
    asset_classes:  list[str]

# ---------------------------------------------------------------------------
# Normalization tables
# ---------------------------------------------------------------------------
FLOAT_RATE_ALIASES = {
    "usd-sofr": "SOFR", "sofr compound": "SOFR", "sofr-compound": "SOFR",
    "sofr ois": "SOFR", "sofr avg": "SOFR", "sofr average": "SOFR", "us sofr": "SOFR",
    "usd-libor-bba": "LIBOR-3M", "usd-libor-bba-3m": "LIBOR-3M",
    "usd-libor-bba-6m": "LIBOR-6M", "usd-libor-bba-1m": "LIBOR-1M",
    "usd-libor-bba-12m": "LIBOR-12M", "usd-libor": "LIBOR-3M",
    "libor": "LIBOR-3M", "libor-bba": "LIBOR-3M", "usd libor": "LIBOR-3M",
    "eur-euribor-reuters": "EURIBOR-6M", "eur-euribor-telerate": "EURIBOR-6M",
    "euribor": "EURIBOR-6M",
    "gbp-sonia": "SONIA", "gbp sonia": "SONIA",
    "eur-estr": "ESTR", "eur estr": "ESTR", "€str": "ESTR",
}

def normalize_floating_rate(rate: Optional[str]) -> Optional[str]:
    if not rate:
        return rate
    return FLOAT_RATE_ALIASES.get(rate.strip().lower(), rate)

# ---------------------------------------------------------------------------
# Extraction prompts
# ---------------------------------------------------------------------------
IRS_EXTRACTION_SYSTEM = """You are a capital markets confirmation parser.
Extract structured fields from an IRS (Interest Rate Swap) confirmation document.
Respond ONLY with a valid JSON object — no prose, no markdown fences.

Extract these fields (use null if absent or ambiguous):
{
  "trade_id": string or null,
  "trade_date": "YYYY-MM-DD" or null,
  "effective_date": "YYYY-MM-DD" or null,
  "maturity_date": "YYYY-MM-DD" or null,
  "counterparty": string or null,
  "notional": number or null,
  "currency": "USD"|"EUR"|"GBP"|"JPY"|etc or null,
  "fixed_rate": number (percentage, e.g. 4.125) or null,
  "floating_rate": string or null,
  "floating_spread": number or null,
  "payment_frequency_fixed": "Annual"|"Semi-Annual"|"Quarterly" or null,
  "payment_frequency_float": "Annual"|"Semi-Annual"|"Quarterly" or null,
  "day_count_fixed": "30/360"|"ACT/360"|"ACT/365"|"ACT/ACT" or null,
  "day_count_float": "30/360"|"ACT/360"|"ACT/365"|"ACT/ACT" or null,
  "usi": string or null,
  "extraction_confidence": number between 0 and 1
}
Rules:
- fixed_rate as percentage (4.125 not 0.04125). "2.01000 percent" → 2.01.
- Dates always YYYY-MM-DD.
- floating_rate normalized: USD-SOFR/SOFR Compound → "SOFR"; USD-LIBOR-BBA → "LIBOR-3M"; EURIBOR → "EURIBOR-6M"; GBP-SONIA → "SONIA"; EUR-ESTR → "ESTR".
- counterparty: exact legal entity name as written. Not the dealer — the other party.
- Amortizing notional schedules: extract the INITIAL (first effective date) notional.
- Cross-currency swaps: extract the USD notional.
- Internal fields (book, account, status) never appear in counterparty confirmations.
"""

EQ_EXTRACTION_SYSTEM = """You are a capital markets confirmation parser specializing in OTC equity derivatives.
Extract structured fields from an equity index option confirmation document.
Respond ONLY with a valid JSON object — no prose, no markdown fences.

Extract these fields (use null if absent or ambiguous):
{
  "trade_id": string or null,
  "trade_date": "YYYY-MM-DD" or null,
  "counterparty": string or null,
  "option_type": "call" or "put" or null,
  "option_style": "european" or "american" or null,
  "underlying": string (normalized index code) or null,
  "strike": number (index level, e.g. 5200.00) or null,
  "strike_high": number (upper strike for spreads) or null,
  "num_options": number or null,
  "notional": number or null,
  "currency": "USD"|"EUR"|"GBP" or null,
  "premium": number (total premium in currency) or null,
  "premium_per_option": number or null,
  "premium_payment_date": "YYYY-MM-DD" or null,
  "expiry_date": "YYYY-MM-DD" or null,
  "settlement_currency": "USD"|"EUR"|"GBP" or null,
  "settlement_method": "cash" or "physical" or null,
  "buyer": string (legal entity name) or null,
  "seller": string (legal entity name) or null,
  "extraction_confidence": number between 0 and 1
}
Rules:
- Dates always YYYY-MM-DD.
- underlying: normalize to canonical code regardless of how written:
    "S&P 500 Index", "S&P500", "SPX Index" → "SPX"
    "Russell 2000", "Russell 2000 Index" → "RTY"
    "Nasdaq-100", "NASDAQ 100" → "NDX"
    "MSCI EAFE", "MXEA" → "MSCI_EAFE"
    "MSCI EM", "MSCI Emerging Markets" → "MSCI_EM"
    "Euro Stoxx 50", "SX5E" → "SX5E"
    "Nikkei 225" → "NKY"
- option_type: always lowercase "call" or "put".
- option_style: always lowercase "european" or "american".
- For call spreads: strike = lower strike, strike_high = upper strike (cap level).
- counterparty: the insurance company or buy-side firm — not the dealer/bank.
  If document says "Party A: Standard Chartered Bank" and "Party B: [client]",
  extract Party B as counterparty.
- premium: total dollar amount of the option premium, not per-option.
- settlement_method: "cash" for cash-settled index options (most common for FIA/RILA hedges).
- Internal fields (book, account, hedge program) never appear in dealer confirmations.
"""

# ---------------------------------------------------------------------------
# Salvage JSON parser (shared)
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
# Extraction dispatcher
# ---------------------------------------------------------------------------
def extract_fields(confirmation_text: str, asset_class: str) -> tuple[dict, float]:
    if not client:
        raise HTTPException(status_code=503, detail="LLM client not configured")
    system = EQ_EXTRACTION_SYSTEM if asset_class == "EQUITY_OPTION" else IRS_EXTRACTION_SYSTEM
    asset_label = "equity option" if asset_class == "EQUITY_OPTION" else "IRS"
    try:
        msg = client.messages.create(
            model=MODEL, max_tokens=1024, system=system,
            messages=[{"role": "user", "content": f"Parse this {asset_label} confirmation:\n\n{confirmation_text}"}],
        )
        raw = re.sub(r"^```json\s*", "", msg.content[0].text.strip())
        raw = re.sub(r"\s*```$", "", raw)
        extracted = _salvage_json(raw)
        confidence = float(extracted.pop("extraction_confidence", 0.85))
        # Post-extraction normalization
        if asset_class == "IRS" and "floating_rate" in extracted:
            extracted["floating_rate"] = normalize_floating_rate(extracted["floating_rate"])
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
        asset_classes=["IRS", "EQUITY_OPTION"],
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

    # Step 1 — extract
    extracted, confidence = extract_fields(req.confirmation_text, asset_class)

    # Step 2 — route to correct engine
    internal_dict = req.expected_economics.model_dump(exclude_none=True)
    if asset_class == "EQUITY_OPTION":
        result = eq_engine.reconcile(extracted, internal_dict, case_id=match_id)
    else:
        result = irs_engine.reconcile(extracted, internal_dict, case_id=match_id)

    # Step 3 — build response
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

    # Insurance metadata pass-through
    metadata = None
    if req.expected_economics.policy_cohort or req.expected_economics.hedge_program:
        metadata = {
            k: v for k, v in {
                "policy_cohort": req.expected_economics.policy_cohort,
                "hedge_program": req.expected_economics.hedge_program,
            }.items() if v
        }

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
