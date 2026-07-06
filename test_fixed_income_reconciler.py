#!/usr/bin/env python3
"""
test_fixed_income_reconciler.py — regression suite for the fixed income engine.

Covers corporate bonds, munis, and structured securities.
Primary ICP scenario: F&G general account (26% corp bonds, 21% structured).

Usage:
    python3 test_fixed_income_reconciler.py
    python3 test_fixed_income_reconciler.py --verbose
"""
import sys, argparse
from fixed_income_reconciler import FixedIncomeReconciler, EXC

ap = argparse.ArgumentParser()
ap.add_argument("--verbose", "-v", action="store_true")
args = ap.parse_args()

reconciler = FixedIncomeReconciler()

CASES = [
    # ------------------------------------------------------------------
    # 1. Clean corporate bond — IG corp, typical F&G general account trade
    # ------------------------------------------------------------------
    {
        "id": "FI-001",
        "desc": "Clean IG corp bond — Apple 3.85% 2043",
        "confirmation": {
            "cusip": "037833DV6",
            "counterparty": "Goldman Sachs & Co. LLC",
            "direction": "buy",
            "capacity": "principal",
            "par_value": 10000000,
            "currency": "USD",
            "price": 95.250,
            "yield_rate": 4.125,
            "coupon_rate": 3.85,
            "maturity_date": "2043-08-04",
            "settlement_date": "2026-07-03",
            "accrued_interest": 120694.44,
            "principal_amount": 9525000.00,
            "total_consideration": 9645694.44,
        },
        "internal": {
            "cusip": "037833DV6",
            "counterparty": "Goldman Sachs & Co. LLC",
            "direction": "buy",
            "capacity": "principal",
            "par_value": 10000000,
            "currency": "USD",
            "price": 95.250,
            "yield_rate": 4.125,
            "coupon_rate": 3.85,
            "maturity_date": "2043-08-04",
            "settlement_date": "2026-07-03",
            "accrued_interest": 120694.44,
            "principal_amount": 9525000.00,
            "total_consideration": 9645694.44,
        },
        "expect_exception": False,
    },

    # ------------------------------------------------------------------
    # 2. CUSIP mismatch — wrong security entirely
    # ------------------------------------------------------------------
    {
        "id": "FI-002",
        "desc": "CUSIP mismatch — confirmation has wrong security",
        "confirmation": {
            "cusip": "037833DX2",
            "counterparty": "Goldman Sachs & Co. LLC",
            "direction": "buy",
            "par_value": 10000000,
            "price": 95.250,
            "coupon_rate": 3.85,
            "maturity_date": "2043-08-04",
            "settlement_date": "2026-07-03",
            "principal_amount": 9525000.00,
        },
        "internal": {
            "cusip": "037833DV6",
            "counterparty": "Goldman Sachs & Co. LLC",
            "direction": "buy",
            "par_value": 10000000,
            "price": 95.250,
            "coupon_rate": 3.85,
            "maturity_date": "2043-08-04",
            "settlement_date": "2026-07-03",
            "principal_amount": 9525000.00,
        },
        "expect_exception": True,
        "expect_category": EXC.CUSIP,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 3. Price break — 0.25 points off on $10M par
    # ------------------------------------------------------------------
    {
        "id": "FI-003",
        "desc": "Price break: 95.500 confirmed vs 95.250 booked ($25K exposure)",
        "confirmation": {
            "cusip": "037833DV6",
            "counterparty": "Goldman Sachs & Co. LLC",
            "direction": "buy",
            "par_value": 10000000,
            "price": 95.500,
            "coupon_rate": 3.85,
            "maturity_date": "2043-08-04",
            "settlement_date": "2026-07-03",
            "principal_amount": 9550000.00,
            "total_consideration": 9670694.44,
        },
        "internal": {
            "cusip": "037833DV6",
            "counterparty": "Goldman Sachs & Co. LLC",
            "direction": "buy",
            "par_value": 10000000,
            "price": 95.250,
            "coupon_rate": 3.85,
            "maturity_date": "2043-08-04",
            "settlement_date": "2026-07-03",
            "principal_amount": 9525000.00,
            "total_consideration": 9645694.44,
        },
        "expect_exception": True,
        "expect_category": EXC.PRICE,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 4. Wrong direction — sold confirmed vs bought booked
    # ------------------------------------------------------------------
    {
        "id": "FI-004",
        "desc": "Direction mismatch: sell confirmed vs buy booked",
        "confirmation": {
            "cusip": "037833DV6",
            "counterparty": "JPMorgan Securities LLC",
            "direction": "sell",
            "par_value": 5000000,
            "price": 95.250,
            "settlement_date": "2026-07-03",
            "principal_amount": 4762500.00,
        },
        "internal": {
            "cusip": "037833DV6",
            "counterparty": "JPMorgan Securities LLC",
            "direction": "buy",
            "par_value": 5000000,
            "price": 95.250,
            "settlement_date": "2026-07-03",
            "principal_amount": 4762500.00,
        },
        "expect_exception": True,
        "expect_category": EXC.DIR,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 5. Settlement date off by 1 day — fail risk
    # ------------------------------------------------------------------
    {
        "id": "FI-005",
        "desc": "Settlement date mismatch: T+3 confirmed vs T+2 booked",
        "confirmation": {
            "cusip": "037833DV6",
            "counterparty": "Morgan Stanley & Co. LLC",
            "direction": "buy",
            "par_value": 10000000,
            "price": 95.250,
            "settlement_date": "2026-07-04",
            "principal_amount": 9525000.00,
            "total_consideration": 9645694.44,
        },
        "internal": {
            "cusip": "037833DV6",
            "counterparty": "Morgan Stanley & Co. LLC",
            "direction": "buy",
            "par_value": 10000000,
            "price": 95.250,
            "settlement_date": "2026-07-03",
            "principal_amount": 9525000.00,
            "total_consideration": 9645694.44,
        },
        "expect_exception": True,
        "expect_category": EXC.SETTLE,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 6. Par value mismatch — $10M confirmed vs $5M booked
    # ------------------------------------------------------------------
    {
        "id": "FI-006",
        "desc": "Par value mismatch: $10M confirmed vs $5M booked",
        "confirmation": {
            "cusip": "037833DV6",
            "counterparty": "Barclays Capital Inc.",
            "direction": "buy",
            "par_value": 10000000,
            "price": 95.250,
            "settlement_date": "2026-07-03",
            "principal_amount": 9525000.00,
        },
        "internal": {
            "cusip": "037833DV6",
            "counterparty": "Barclays Capital Inc.",
            "direction": "buy",
            "par_value": 5000000,
            "price": 95.250,
            "settlement_date": "2026-07-03",
            "principal_amount": 4762500.00,
        },
        "expect_exception": True,
        "expect_category": EXC.QTY,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 7. Coupon rate mismatch — 3.85% vs 3.875%
    # ------------------------------------------------------------------
    {
        "id": "FI-007",
        "desc": "Coupon mismatch: 3.85% confirmed vs 3.875% booked",
        "confirmation": {
            "cusip": "037833DV6",
            "counterparty": "Goldman Sachs & Co. LLC",
            "direction": "buy",
            "par_value": 10000000,
            "price": 95.250,
            "coupon_rate": 3.85,
            "maturity_date": "2043-08-04",
            "settlement_date": "2026-07-03",
        },
        "internal": {
            "cusip": "037833DV6",
            "counterparty": "Goldman Sachs & Co. LLC",
            "direction": "buy",
            "par_value": 10000000,
            "price": 95.250,
            "coupon_rate": 3.875,
            "maturity_date": "2043-08-04",
            "settlement_date": "2026-07-03",
        },
        "expect_exception": True,
        "expect_category": EXC.COUP,
        "expect_severity": "medium",
    },

    # ------------------------------------------------------------------
    # 8. Counterparty mismatch — wrong broker-dealer entity
    # ------------------------------------------------------------------
    {
        "id": "FI-008",
        "desc": "Counterparty mismatch: Goldman International vs Goldman LLC",
        "confirmation": {
            "cusip": "037833DV6",
            "counterparty": "Goldman Sachs International",
            "direction": "buy",
            "par_value": 10000000,
            "price": 95.250,
            "settlement_date": "2026-07-03",
        },
        "internal": {
            "cusip": "037833DV6",
            "counterparty": "Goldman Sachs & Co. LLC",
            "direction": "buy",
            "par_value": 10000000,
            "price": 95.250,
            "settlement_date": "2026-07-03",
        },
        "expect_exception": True,
        "expect_category": EXC.CPTY,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 9. Dual exception — price break + settlement date wrong
    # ------------------------------------------------------------------
    {
        "id": "FI-009",
        "desc": "Dual: price break + wrong settlement date",
        "confirmation": {
            "cusip": "037833DV6",
            "counterparty": "Morgan Stanley & Co. LLC",
            "direction": "buy",
            "par_value": 10000000,
            "price": 96.000,
            "settlement_date": "2026-07-05",
            "principal_amount": 9600000.00,
        },
        "internal": {
            "cusip": "037833DV6",
            "counterparty": "Morgan Stanley & Co. LLC",
            "direction": "buy",
            "par_value": 10000000,
            "price": 95.250,
            "settlement_date": "2026-07-03",
            "principal_amount": 9525000.00,
        },
        "expect_exception": True,
        "expect_category": EXC.PRICE,
        "expect_secondary": EXC.SETTLE,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 10. CUSIP normalization — spaces and lowercase should match
    # ------------------------------------------------------------------
    {
        "id": "FI-010",
        "desc": "CUSIP normalization: '037833 DV 6' == '037833DV6' (clean)",
        "confirmation": {
            "cusip": "037833 DV 6",
            "counterparty": "Goldman Sachs & Co. LLC",
            "direction": "buy",
            "par_value": 10000000,
            "price": 95.250,
            "settlement_date": "2026-07-03",
        },
        "internal": {
            "cusip": "037833DV6",
            "counterparty": "Goldman Sachs & Co. LLC",
            "direction": "buy",
            "par_value": 10000000,
            "price": 95.250,
            "settlement_date": "2026-07-03",
        },
        "expect_exception": False,
    },

    # ------------------------------------------------------------------
    # 11. Muni bond — clean (MSRB G-15 format)
    # ------------------------------------------------------------------
    {
        "id": "FI-011",
        "desc": "Clean muni bond — NYC GO 5.0% 2040",
        "confirmation": {
            "cusip": "64966EYQ8",
            "counterparty": "Citigroup Global Markets Inc.",
            "direction": "buy",
            "capacity": "principal",
            "par_value": 5000000,
            "currency": "USD",
            "price": 103.500,
            "coupon_rate": 5.00,
            "maturity_date": "2040-08-01",
            "settlement_date": "2026-07-08",
            "accrued_interest": 95833.33,
            "principal_amount": 5175000.00,
            "total_consideration": 5270833.33,
        },
        "internal": {
            "cusip": "64966EYQ8",
            "counterparty": "Citigroup Global Markets Inc.",
            "direction": "buy",
            "capacity": "principal",
            "par_value": 5000000,
            "currency": "USD",
            "price": 103.500,
            "coupon_rate": 5.00,
            "maturity_date": "2040-08-01",
            "settlement_date": "2026-07-08",
            "accrued_interest": 95833.33,
            "principal_amount": 5175000.00,
            "total_consideration": 5270833.33,
        },
        "expect_exception": False,
    },

    # ------------------------------------------------------------------
    # 12. Maturity date mismatch
    # ------------------------------------------------------------------
    {
        "id": "FI-012",
        "desc": "Maturity date mismatch: 2043 vs 2044",
        "confirmation": {
            "cusip": "037833DV6",
            "counterparty": "Goldman Sachs & Co. LLC",
            "direction": "buy",
            "par_value": 10000000,
            "price": 95.250,
            "maturity_date": "2044-08-04",
            "settlement_date": "2026-07-03",
        },
        "internal": {
            "cusip": "037833DV6",
            "counterparty": "Goldman Sachs & Co. LLC",
            "direction": "buy",
            "par_value": 10000000,
            "price": 95.250,
            "maturity_date": "2043-08-04",
            "settlement_date": "2026-07-03",
        },
        "expect_exception": True,
        "expect_category": EXC.MAT,
        "expect_severity": "high",
    },
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
passed = failed = 0
SEP = "─" * 72

print(f"Fixed Income Reconciler — {len(CASES)} cases\n{SEP}")

for tc in CASES:
    result = reconciler.reconcile(tc["confirmation"], tc["internal"], case_id=tc["id"])
    ok = True

    if result.exception_exists != tc["expect_exception"]:
        print(f"  FAIL {tc['id']} — detection: got={result.exception_exists} expected={tc['expect_exception']}")
        ok = False

    if tc.get("expect_category"):
        got_cat = result.primary.category if result.primary else None
        if got_cat != tc["expect_category"]:
            print(f"  FAIL {tc['id']} — category: got={got_cat} expected={tc['expect_category']}")
            ok = False

    if tc.get("expect_secondary"):
        got_sec = result.secondary.category if result.secondary else None
        if got_sec != tc["expect_secondary"]:
            print(f"  FAIL {tc['id']} — secondary: got={got_sec} expected={tc['expect_secondary']}")
            ok = False

    if tc.get("expect_severity") and result.primary:
        if result.primary.severity != tc["expect_severity"]:
            print(f"  FAIL {tc['id']} — severity: got={result.primary.severity} expected={tc['expect_severity']}")
            ok = False

    if ok:
        passed += 1
        if args.verbose:
            print(f"\n{tc['id']}  [PASS]  {tc['desc']}")
            print(f"  exception={result.exception_exists}  severity={result.overall_severity}")
            if result.primary:
                exp_str = f"  exposure=${result.primary.exposure_usd:,.0f}" if result.primary.exposure_usd else ""
                print(f"  primary={result.primary.category} on {result.primary.field}{exp_str}")
                print(f"  action={result.recommended_action[:100]}")
        else:
            print(f"  {tc['id']}  PASS  {tc['desc'][:55]}")
    else:
        failed += 1
        if args.verbose:
            print(f"\n{tc['id']}  [FAIL]  {tc['desc']}")

print(f"\n{SEP}")
print(f"Results: {passed} passed, {failed} failed")
if failed:
    sys.exit(1)
else:
    print("All assertions passed — engine regression-clean.")
