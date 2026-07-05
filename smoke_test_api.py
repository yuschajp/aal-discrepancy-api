#!/usr/bin/env python3
"""
smoke_test_api.py — local smoke test for the AAL Discrepancy API.

Runs without a live server — calls the endpoint functions directly.
Useful before deploying to verify the full pipeline end-to-end.

Usage:
    python3 smoke_test_api.py              # mock LLM (no API key needed)
    python3 smoke_test_api.py --live       # real LLM call (needs ANTHROPIC_API_KEY)
"""
import sys, json, asyncio, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--live", action="store_true", help="Use real LLM extraction")
args = ap.parse_args()

# Patch the client before importing main
if not args.live:
    import os; os.environ["ANTHROPIC_API_KEY"] = ""

from main import app, reconciler, ValidateRequest, ExpectedEconomics, ValidateOptions
from fastapi.testclient import TestClient

client = TestClient(app)

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
TESTS = [
    {
        "name": "Clean IRS — no exception",
        "text": """
            CONFIRMATION — USD SOFR IRS 5Y
            Trade Date: 2026-07-01
            Effective Date: 2026-07-03
            Maturity Date: 2031-07-03
            Counterparty: Global Bank Capital Markets Inc
            Notional: USD 50,000,000
            Fixed Rate: 4.125%
            Floating Rate: SOFR
            Payment Frequency Fixed: Semi-Annual
            Payment Frequency Float: Quarterly
            Day Count Fixed: 30/360
            Day Count Float: ACT/360
        """,
        "expected": {
            "asset_class": "IRS",
            "notional": 50_000_000,
            "currency": "USD",
            "counterparty": "Global Bank Capital Markets Inc",
            "fixed_rate": 4.125,
            "floating_rate": "SOFR",
            "effective_date": "2026-07-03",
            "maturity_date": "2031-07-03",
        },
        "expect_exception": False,
    },
    {
        "name": "Rate break — 0.5bp on $50M",
        "text": """
            CONFIRMATION — USD SOFR IRS 5Y
            Trade Date: 2026-07-01
            Effective Date: 2026-07-03
            Maturity Date: 2031-07-03
            Counterparty: Global Bank Capital Markets Inc
            Notional: USD 50,000,000
            Fixed Rate: 4.130%
            Floating Rate: SOFR
            Payment Frequency Fixed: Semi-Annual
            Day Count Fixed: 30/360
        """,
        "expected": {
            "asset_class": "IRS",
            "notional": 50_000_000,
            "currency": "USD",
            "counterparty": "Global Bank Capital Markets Inc",
            "fixed_rate": 4.125,
            "floating_rate": "SOFR",
            "effective_date": "2026-07-03",
            "maturity_date": "2031-07-03",
        },
        "expect_exception": True,
        "expect_category": "EXC-PRICE",
    },
    {
        "name": "Counterparty mismatch",
        "text": """
            CONFIRMATION — USD SOFR IRS 5Y
            Trade Date: 2026-07-01
            Counterparty: Global Bank Securities LLC
            Notional: USD 50,000,000
            Fixed Rate: 4.125%
            Floating Rate: SOFR
            Effective Date: 2026-07-03
            Maturity Date: 2031-07-03
        """,
        "expected": {
            "asset_class": "IRS",
            "notional": 50_000_000,
            "currency": "USD",
            "counterparty": "Global Bank Capital Markets Inc",
            "fixed_rate": 4.125,
            "floating_rate": "SOFR",
            "effective_date": "2026-07-03",
            "maturity_date": "2031-07-03",
        },
        "expect_exception": True,
        "expect_category": "EXC-CPTY",
    },
]

# Mock extraction for non-live mode
MOCK_EXTRACTIONS = [
    {"counterparty": "Global Bank Capital Markets Inc", "notional": 50_000_000,
     "currency": "USD", "fixed_rate": 4.125, "floating_rate": "SOFR",
     "effective_date": "2026-07-03", "maturity_date": "2031-07-03",
     "payment_frequency_fixed": "Semi-Annual", "day_count_fixed": "30/360"},
    {"counterparty": "Global Bank Capital Markets Inc", "notional": 50_000_000,
     "currency": "USD", "fixed_rate": 4.130, "floating_rate": "SOFR",
     "effective_date": "2026-07-03", "maturity_date": "2031-07-03"},
    {"counterparty": "Global Bank Securities LLC", "notional": 50_000_000,
     "currency": "USD", "fixed_rate": 4.125, "floating_rate": "SOFR",
     "effective_date": "2026-07-03", "maturity_date": "2031-07-03"},
]

if not args.live:
    # Monkey-patch extract_fields to return mock data
    import main as _main
    _mock_idx = [0]
    def _mock_extract(text):
        idx = _mock_idx[0]; _mock_idx[0] += 1
        return MOCK_EXTRACTIONS[idx], 0.99
    _main.extract_fields = _mock_extract

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
passed = failed = 0
SEP = "─" * 60

print(f"AAL API Smoke Test ({'live LLM' if args.live else 'mock extraction'})\n{SEP}")

for i, t in enumerate(TESTS):
    print(f"\n[{i+1}] {t['name']}")
    payload = {
        "confirmation_text": t["text"],
        "expected_economics": t["expected"],
        "options": {"return_raw_extraction": True},
    }
    resp = client.post(
        "/v1/confirmations/validate",
        json=payload,
        headers={"X-API-Key": "dev-key-replace-in-prod"},
    )
    if resp.status_code != 200:
        print(f"  FAIL — HTTP {resp.status_code}: {resp.text[:200]}")
        failed += 1
        continue

    data = resp.json()
    ok = True

    # Check exception detection
    got_exc = data["overall_status"] != "clean"
    if got_exc != t["expect_exception"]:
        print(f"  FAIL — exception: got={got_exc} expected={t['expect_exception']}")
        ok = False

    # Check category
    if t.get("expect_category"):
        cats = [d["category"] for d in data["discrepancies"]]
        if t["expect_category"] not in cats:
            print(f"  FAIL — category: got={cats} expected={t['expect_category']}")
            ok = False

    if ok:
        print(f"  PASS — status={data['overall_status']} "
              f"severity={data['overall_severity']} "
              f"discrepancies={len(data['discrepancies'])} "
              f"ms={data['processing_ms']}")
        if data["discrepancies"]:
            d = data["discrepancies"][0]
            print(f"         {d['category']} on {d['field']}: "
                  f"{d['extracted']} vs {d['expected']}"
                  + (f" (~${d['exposure_estimate_usd']:,.0f})" if d['exposure_estimate_usd'] else ""))
        passed += 1
    else:
        failed += 1

print(f"\n{SEP}")
print(f"Results: {passed} passed, {failed} failed")
if failed:
    sys.exit(1)
else:
    print("Smoke test clean — ready to start server.")
    print("\nTo start locally:")
    print("  uvicorn main:app --reload")
    print("\nTo test live:")
    print("  curl -X POST http://localhost:8000/v1/confirmations/validate \\")
    print("    -H 'X-API-Key: dev-key-replace-in-prod' \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{...}'")
