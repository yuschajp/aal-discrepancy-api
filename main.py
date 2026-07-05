#!/usr/bin/env python3
"""
AAL Discrepancy Detection API v1
FastAPI application — IRS confirmation discrepancy detection.

v1.1 changes:
  - Floating rate normalization in extraction prompt + post-extraction safety net
  - Amortizing notional rule: extract initial notional from schedules
  - Counterparty legal entity extraction rule

Architecture (per Decision Log 2026-07-04):
  POST /v1/confirmations/validate
    → [1] LLM extraction layer (Claude Haiku) — NER/classification only
    → [2] IRSReconciler — deterministic arithmetic, never LLM
    → [3] Severity rule table — no LLM
    → Response
"""

import os, uuid, time, json, re
from typing import Optional, Literal
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import anthropic
from irs_reconciler import IRSReconciler

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
AAL_API_KEY       = os.environ.get("AAL_API_KEY", "dev-key-replace-in-prod")
MODEL             = "claude-haiku-4-5-20251001"
API_VERSION       = "aal-disc-v1.1"

client     = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
reconciler = IRSReconciler()

app = FastAPI(
    title="AAL Discrepancy Detection API",
    description="Deterministic IRS confirmation discrepancy detection.",
    version="1.1.0",
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
    asset_class:             Literal["IRS"] = "IRS"
    notional:                Optional[float] = None
    currency:                Optional[str]   = None
    trade_date:              Optional[str]   = None
    counterparty:            Optional[str]   = None
    fixed_rate:              Optional[float] = None
    floating_rate:           Optional[str]   = None
    effective_date:          Optional[str]   = None
    maturity_date:           Optional[str]   = None
    payment_frequency_fixed: Optional[str]   = None
    payment_frequency_float: Optional[str]   = None
    day_count_fixed:         Optional[str]   = None
    day_count_float:         Optional[str]   = None

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
    overall_status:        Literal["clean", "discrepant", "review_required"]
    discrepancies:         list[DiscrepancyDetail]
    overall_severity:      str
    escalation_required:   bool
    recommended_action:    str
    extraction_confidence: float
    processing_ms:         int
    model_version:         str
    raw_extraction:        Optional[dict] = None

class HealthResponse(BaseModel):
    status:    str
    version:   str
    model:     str
    llm_ready: bool

# ---------------------------------------------------------------------------
# Floating rate normalization table
# Applied post-extraction as a safety net (prompt handles most cases)
# ---------------------------------------------------------------------------
FLOAT_RATE_ALIASES = {
    # SOFR variants
    "usd-sofr": "SOFR",
    "sofr compound": "SOFR",
    "sofr-compound": "SOFR",
    "sofr ois": "SOFR",
    "sofr avg": "SOFR",
    "sofr average": "SOFR",
    "us sofr": "SOFR",
    # LIBOR variants
    "usd-libor-bba": "LIBOR-3M",
    "usd-libor-bba-3m": "LIBOR-3M",
    "usd-libor-bba-6m": "LIBOR-6M",
    "usd-libor-bba-1m": "LIBOR-1M",
    "usd-libor-bba-12m": "LIBOR-12M",
    "usd-libor": "LIBOR-3M",
    "libor": "LIBOR-3M",
    "libor-bba": "LIBOR-3M",
    "usd libor": "LIBOR-3M",
    # EURIBOR variants
    "eur-euribor-reuters": "EURIBOR-6M",
    "eur-euribor-telerate": "EURIBOR-6M",
    "euribor": "EURIBOR-6M",
    # SONIA
    "gbp-sonia": "SONIA",
    "gbp sonia": "SONIA",
    # ESTR
    "eur-estr": "ESTR",
    "eur estr": "ESTR",
    "€str": "ESTR",
}

def normalize_floating_rate(rate: Optional[str]) -> Optional[str]:
    if not rate:
        return rate
    key = rate.strip().lower()
    return FLOAT_RATE_ALIASES.get(key, rate)

# ---------------------------------------------------------------------------
# LLM extraction layer
# ---------------------------------------------------------------------------
EXTRACTION_SYSTEM = """You are a capital markets confirmation parser.
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
  "floating_rate": string (e.g. "SOFR", "EURIBOR-6M") or null,
  "floating_spread": number or null,
  "payment_frequency_fixed": "Annual"|"Semi-Annual"|"Quarterly" or null,
  "payment_frequency_float": "Annual"|"Semi-Annual"|"Quarterly" or null,
  "day_count_fixed": "30/360"|"ACT/360"|"ACT/365"|"ACT/ACT" or null,
  "day_count_float": "30/360"|"ACT/360"|"ACT/365"|"ACT/ACT" or null,
  "usi": string or null,
  "extraction_confidence": number between 0 and 1
}

Rules:
- Extract ONLY what is explicitly stated. Never infer or guess field values.
- For fixed_rate: extract as a percentage number (2.01, not 0.0201). If stated as "2.01000 percent", output 2.01.
- For dates: always output YYYY-MM-DD format. "10 November 2009" → "2009-11-10".
- For floating_rate: normalize to canonical short form regardless of how it appears:
    USD-SOFR, SOFR Compound, SOFR OIS → "SOFR"
    USD-LIBOR-BBA, LIBOR, USD-LIBOR → "LIBOR" plus tenor if stated (e.g. "LIBOR-3M")
    EURIBOR, EUR-EURIBOR-Reuters → "EURIBOR" plus tenor if stated (e.g. "EURIBOR-6M")
    GBP-SONIA, SONIA → "SONIA"
    EUR-ESTR, ESTR → "ESTR"
- For counterparty: extract the legal entity name exactly as written in the confirmation.
  If the document identifies the counterparty with a label like "the Counterparty" or names them
  explicitly, use that exact legal name. Do not use JPMorgan or the dealer as the counterparty.
- For amortizing swaps with a notional amount schedule: extract the INITIAL notional amount
  (the first entry in the schedule, corresponding to the effective date).
- For cross-currency swaps: extract the USD notional amount.
- extraction_confidence: your overall confidence in the extraction (0.0 to 1.0).
- Internal workflow fields (book, account, status) never appear in counterparty confirmations.
"""

def extract_fields(confirmation_text: str) -> tuple[dict, float]:
    if not client:
        raise HTTPException(status_code=503, detail="LLM client not configured")
    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=EXTRACTION_SYSTEM,
            messages=[{"role": "user", "content": f"Parse this IRS confirmation:\n\n{confirmation_text}"}],
        )
        raw = msg.content[0].text.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            extracted = json.loads(raw)
        except json.JSONDecodeError:
            extracted = {}
            start = raw.find("{")
            if start != -1:
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
                            try: extracted = json.loads(raw[start:i+1])
                            except: pass
                            break
        confidence = float(extracted.pop("extraction_confidence", 0.85))
        # Post-extraction normalization safety net
        if "floating_rate" in extracted:
            extracted["floating_rate"] = normalize_floating_rate(extracted["floating_rate"])
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

    extracted, extraction_confidence = extract_fields(req.confirmation_text)

    internal_dict = req.expected_economics.model_dump(exclude_none=True)
    result = reconciler.reconcile(
        confirmation=extracted,
        internal=internal_dict,
        case_id=match_id,
    )

    threshold_rank = {"low": 0, "medium": 1, "high": 2}
    threshold = req.options.severity_threshold
    discrepancies = []
    for d in [result.primary, result.secondary]:
        if d is None:
            continue
        if threshold_rank.get(d.severity, 0) >= threshold_rank.get(threshold, 0):
            discrepancies.append(DiscrepancyDetail(
                field=d.field,
                category=d.category,
                expected=d.internal_value,
                extracted=d.counterparty_value,
                difference=d.difference,
                severity=d.severity,
                exposure_estimate_usd=d.exposure_usd,
                confidence=d.confidence,
            ))

    if not result.exception_exists:
        overall_status = "clean"
    elif result.overall_severity == "high":
        overall_status = "discrepant"
    else:
        overall_status = "review_required"

    return ValidateResponse(
        match_id=match_id,
        overall_status=overall_status,
        discrepancies=discrepancies,
        overall_severity=result.overall_severity,
        escalation_required=result.escalation_required,
        recommended_action=result.recommended_action,
        extraction_confidence=extraction_confidence,
        processing_ms=int((time.time() - t0) * 1000),
        model_version=API_VERSION,
        raw_extraction=extracted if req.options.return_raw_extraction else None,
    )
