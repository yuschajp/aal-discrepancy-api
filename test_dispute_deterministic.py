#!/usr/bin/env python3
"""
test_dispute_deterministic.py — tests for the deterministic margin-call
dispute arithmetic in main.py. The LLM classifies/extracts only; every
numeric field in DisputeResponse is computed in Python.

No live API calls. pytest-asyncio is NOT installed; endpoint tests use
fastapi.testclient.TestClient (sync).
"""
import os
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

import json
import re
import pytest
from unittest.mock import Mock, AsyncMock, patch

from fastapi.testclient import TestClient

import main

client = TestClient(main.app)
HEADERS = {"X-API-Key": "dev-key-replace-in-prod"}


def _stub_llm(payload):
    """Patch main._messages_create_with_retry to return the given JSON payload."""
    return patch.object(
        main, "_messages_create_with_retry",
        new=AsyncMock(return_value=Mock(content=[Mock(text=json.dumps(payload))])),
    )


# ---------------------------------------------------------------------------
# Group A — pure formula tests
# ---------------------------------------------------------------------------

def test_correct_call_price_uses_internal():
    notice = main.MarginCallNotice(call_amount=2450000)
    internal = main.InternalCalculation(calculated_amount=1980000)
    assert main._correct_call_amount("DIS-PRICE", "portfolio_mtm", notice, internal, None) == 1980000


def test_correct_call_dupe_is_zero():
    notice = main.MarginCallNotice(call_amount=500000)
    internal = main.InternalCalculation(calculated_amount=500000)
    correct = main._correct_call_amount("DIS-DUPE", "call_id", notice, internal, None)
    assert correct == 0.0
    disputed = main._disputed_call_amount("DIS-DUPE", correct, notice, internal)
    assert disputed == notice.call_amount


def test_correct_call_dir_negates_internal():
    notice = main.MarginCallNotice(call_amount=400000)
    internal = main.InternalCalculation(calculated_amount=400000)
    correct = main._correct_call_amount("DIS-DIR", "call_direction", notice, internal, None)
    assert correct == -400000
    disputed = main._disputed_call_amount("DIS-DIR", correct, notice, internal)
    assert disputed == notice.call_amount


def test_correct_call_elgblty_adds_collateral():
    notice = main.MarginCallNotice(call_amount=3500000, collateral_on_hand=31500000)
    internal = main.InternalCalculation(calculated_amount=1000000)
    correct = main._correct_call_amount("DIS-ELGBLTY", "collateral_type", notice, internal, None)
    assert correct == 35000000
    disputed = main._disputed_call_amount("DIS-ELGBLTY", correct, notice, internal)
    assert disputed == -31500000


def test_correct_call_haircut_with_rate():
    # with rounding: call + coll*rate is rounded to the increment
    notice = main.MarginCallNotice(call_amount=870000, collateral_on_hand=7880000, rounding=10000)
    internal = main.InternalCalculation(calculated_amount=870000)
    correct = main._correct_call_amount("DIS-HAIRCUT", "collateral_haircut_applied", notice, internal, 0.02)
    assert correct == 1030000.0

    # with no rounding increment on either side, the raw sum is returned unrounded
    notice_nr = main.MarginCallNotice(call_amount=870000, collateral_on_hand=7880000, rounding=None)
    internal_nr = main.InternalCalculation(calculated_amount=870000, rounding=None)
    correct_nr = main._correct_call_amount("DIS-HAIRCUT", "collateral_haircut_applied", notice_nr, internal_nr, 0.02)
    assert correct_nr == 1027600.0


def test_correct_call_haircut_no_rate_falls_back_to_calc():
    notice = main.MarginCallNotice(call_amount=870000, collateral_on_hand=7880000, rounding=10000)
    internal = main.InternalCalculation(calculated_amount=870000)
    correct = main._correct_call_amount("DIS-HAIRCUT", "collateral_haircut_applied", notice, internal, None)
    assert correct == internal.calculated_amount


def test_correct_call_fx_uses_call():
    notice = main.MarginCallNotice(call_amount=1200000)
    internal = main.InternalCalculation(calculated_amount=1150000)
    correct = main._correct_call_amount("DIS-FX", "fx_rate_used", notice, internal, None)
    assert correct == notice.call_amount
    disputed = main._disputed_call_amount("DIS-FX", correct, notice, internal)
    assert disputed == notice.call_amount - internal.calculated_amount


def test_correct_call_thresh_spurious():
    notice = main.MarginCallNotice(call_amount=150000)
    internal = main.InternalCalculation(calculated_amount=0)
    correct = main._correct_call_amount("DIS-THRESH", "call_amount", notice, internal, None)
    assert correct == 0.0
    disputed = main._disputed_call_amount("DIS-THRESH", correct, notice, internal)
    assert disputed == notice.call_amount
    assert main._pay_undisputed("DIS-THRESH", correct, True) is False
    assert main._escalation_required("DIS-THRESH", correct, True) is False


def test_correct_call_thresh_real():
    notice = main.MarginCallNotice(call_amount=150000)
    internal = main.InternalCalculation(calculated_amount=500000)
    correct = main._correct_call_amount("DIS-THRESH", "threshold", notice, internal, None)
    assert correct == internal.calculated_amount
    assert main._escalation_required("DIS-THRESH", correct, True) is True


def test_disputed_qty_is_absolute():
    notice = main.MarginCallNotice(call_amount=125000)
    internal = main.InternalCalculation(calculated_amount=250000)
    disputed = main._disputed_call_amount("DIS-QTY", 250000, notice, internal)
    assert disputed == 125000.0
    assert disputed > 0


@pytest.mark.parametrize("category", ["DIS-DUPE", "DIS-DIR", "DIS-ELGBLTY"])
def test_pay_undisputed_no_pay_categories(category):
    assert main._pay_undisputed(category, 100000, True) is False


def test_undisputed_min_of_call_and_correct():
    notice = main.MarginCallNotice(call_amount=2450000)
    assert main._undisputed_amount(True, 1980000, notice) == 1980000
    assert main._undisputed_amount(False, 1980000, notice) == 0.0


def test_escalation_timing_false():
    assert main._escalation_required("DIS-TIMING", 100000, True) is False


def test_escalation_clean_false():
    assert main._escalation_required("DIS-CLEAN", None, False) is False


# ---------------------------------------------------------------------------
# Group B — endpoint tests via TestClient
# ---------------------------------------------------------------------------

def test_analyze_dispute_price_end_to_end():
    body = {
        "margin_call_notice": {
            "counterparty": "CP",
            "call_amount": 2450000,
            "portfolio_mtm": -24500000,
            "collateral_on_hand": 0,
            "rounding": 10000,
        },
        "internal_calculation": {
            "calculated_amount": 1980000,
            "portfolio_mtm": -23980000,
        },
    }
    stub = {
        "dispute_exists": True,
        "primary": {"category": "DIS-PRICE", "field": "portfolio_mtm"},
        "secondary": None,
        "csa_haircut_rate": None,
        "escalation_target": "Senior collateral manager",
    }
    with _stub_llm(stub):
        resp = client.post("/v1/disputes/analyze", json=body, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["dispute_exists"] is True
    pd = data["primary_dispute"]
    assert pd["correct_call_amount"] == 1980000
    assert pd["disputed_call_amount"] == 470000
    assert data["undisputed_amount"] == 1980000
    assert data["pay_undisputed"] is True
    assert data["escalation_required"] is True
    assert pd["counterparty_value"] == -24500000
    assert pd["internal_value"] == -23980000
    assert pd["difference"] == 520000


def test_analyze_dispute_clean():
    body = {
        "margin_call_notice": {"counterparty": "CP", "call_amount": 500000},
        "internal_calculation": {"calculated_amount": 500000},
    }
    stub = {
        "dispute_exists": False,
        "primary": None,
        "secondary": None,
        "csa_haircut_rate": None,
        "escalation_target": None,
    }
    with _stub_llm(stub):
        resp = client.post("/v1/disputes/analyze", json=body, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["primary_dispute"] is None
    assert data["undisputed_amount"] == 500000
    assert data["pay_undisputed"] is True
    assert data["escalation_required"] is False


def test_llm_numbers_ignored():
    body = {
        "margin_call_notice": {
            "counterparty": "CP",
            "call_amount": 2450000,
            "portfolio_mtm": -24500000,
            "collateral_on_hand": 0,
            "rounding": 10000,
        },
        "internal_calculation": {
            "calculated_amount": 1980000,
            "portfolio_mtm": -23980000,
        },
    }
    stub = {
        "dispute_exists": True,
        "primary": {"category": "DIS-PRICE", "field": "portfolio_mtm"},
        "secondary": None,
        "csa_haircut_rate": None,
        "escalation_target": "Senior collateral manager",
        "correct_call_amount": 999999,
        "undisputed_amount": 123,
    }
    with _stub_llm(stub):
        resp = client.post("/v1/disputes/analyze", json=body, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["primary_dispute"]["correct_call_amount"] == 1980000
    assert data["primary_dispute"]["correct_call_amount"] != 999999
    assert data["undisputed_amount"] != 123


def test_schema_fields_unchanged():
    assert set(main.DisputeResponse.model_fields) == {
        "case_id", "dispute_exists", "primary_dispute", "secondary_dispute",
        "recommended_action", "escalation_required", "escalation_target",
        "pay_undisputed", "undisputed_amount", "extraction_confidence",
        "processing_ms", "model_version",
    }
    assert set(main.DisputeDetail.model_fields) == {
        "category", "field", "counterparty_value", "internal_value",
        "difference", "disputed_call_amount", "correct_call_amount",
    }


def test_unparseable_llm_raises_502():
    body = {
        "margin_call_notice": {"counterparty": "CP", "call_amount": 500000},
        "internal_calculation": {"calculated_amount": 500000},
    }
    with patch.object(
        main, "_messages_create_with_retry",
        new=AsyncMock(return_value=Mock(content=[Mock(text="I cannot answer")])),
    ):
        resp = client.post("/v1/disputes/analyze", json=body, headers=HEADERS)
    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# Group C — regression harness against the AAL-D-002 benchmark
# ---------------------------------------------------------------------------

DATA = "/Users/victoria/Desktop/ai-alpha-labs/aal-benchmark/datasets/AAL-D-002/AAL-D-002-v1.0.json"


def _extract_haircut_rate_for_test(notice_dict, internal_dict, ctx_dict):
    for src in (internal_dict, notice_dict):
        v = src.get("collateral_haircut_required")
        if isinstance(v, (int, float)) and 0 < v < 1:
            return float(v)
    blob = " ".join(str(v) for v in list(internal_dict.values()) + list(notice_dict.values()) + list(ctx_dict.values()))
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*haircut", blob) or re.search(r"haircut[^0-9]{0,25}(\d+(?:\.\d+)?)\s*%", blob)
    if m:
        return float(m.group(1)) / 100.0
    return None


def test_formula_agreement_against_benchmark():
    if not os.path.exists(DATA):
        pytest.skip("benchmark dataset not available")

    with open(DATA) as f:
        cases = json.load(f)

    correct_total = correct_matches = 0
    disputed_total = disputed_matches = 0
    dispute_exists_matches = 0

    for case in cases:
        notice_in = case["input"]["margin_call_notice"]
        internal_in = case["input"]["internal_calculation"]
        notice = main.MarginCallNotice(**notice_in)
        internal = main.InternalCalculation(**internal_in)
        gt = case["ground_truth"]
        gt_dispute_exists = gt["dispute_exists"]
        primary_gt = gt.get("primary_dispute")

        if not gt_dispute_exists or not primary_gt:
            # nothing for the helpers to compute; both sides are None
            dispute_exists_matches += 1
            correct_total += 1
            correct_matches += 1
            disputed_total += 1
            disputed_matches += 1
            continue

        category = primary_gt["category"]
        field = primary_gt["field"]
        if category == "DIS-HAIRCUT":
            ctx_in = case["input"].get("context", {})
            haircut_rate = _extract_haircut_rate_for_test(notice_in, internal_in, ctx_in)
        else:
            haircut_rate = None

        computed_dispute_exists = category != "DIS-CLEAN"
        if computed_dispute_exists == gt_dispute_exists:
            dispute_exists_matches += 1

        correct = main._correct_call_amount(category, field, notice, internal, haircut_rate)
        disputed = main._disputed_call_amount(category, correct, notice, internal)

        gt_correct = primary_gt.get("correct_call_amount")
        gt_disputed = primary_gt.get("disputed_call_amount")

        correct_total += 1
        if correct is None and gt_correct is None:
            correct_matches += 1
        elif correct is not None and gt_correct is not None and abs(correct - gt_correct) <= 1000:
            correct_matches += 1

        disputed_total += 1
        if disputed is None and gt_disputed is None:
            disputed_matches += 1
        elif disputed is not None and gt_disputed is not None and abs(disputed - gt_disputed) <= 1000:
            disputed_matches += 1

    correct_agreement = correct_matches / correct_total
    disputed_agreement = disputed_matches / disputed_total

    print(f"\ncorrect_call agreement: {correct_matches}/{correct_total} = {correct_agreement:.4f}")
    print(f"disputed_call agreement: {disputed_matches}/{disputed_total} = {disputed_agreement:.4f}")
    print(f"dispute_exists agreement: {dispute_exists_matches}/{len(cases)}")

    assert correct_agreement >= 0.95, f"correct_call agreement {correct_agreement:.4f} below 0.95"
    assert disputed_agreement >= 0.92, f"disputed_call agreement {disputed_agreement:.4f} below 0.92"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
