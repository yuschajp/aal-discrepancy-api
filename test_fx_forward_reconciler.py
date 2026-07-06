#!/usr/bin/env python3
"""
test_fx_forward_reconciler.py — regression suite for the FX engine.

Covers vanilla FX forwards, NDOs, and FX options.
Reference: Standard Chartered Bank FX Forward/Option template (1998 ISDA FX Definitions).

Usage:
    python3 test_fx_forward_reconciler.py
    python3 test_fx_forward_reconciler.py --verbose
"""
import sys, argparse
from fx_forward_reconciler import FXReconciler, EXC

ap = argparse.ArgumentParser()
ap.add_argument("--verbose", "-v", action="store_true")
args = ap.parse_args()

reconciler = FXReconciler()

CASES = [
    # ------------------------------------------------------------------
    # 1. Clean vanilla EUR/USD forward
    # ------------------------------------------------------------------
    {
        "id": "FX-001",
        "desc": "Clean EUR/USD deliverable forward — €10M at 1.0850",
        "confirmation": {
            "counterparty": "Standard Chartered Bank",
            "call_currency": "EUR", "call_amount": 10000000,
            "put_currency": "USD", "put_amount": 10850000,
            "strike": 1.0850,
            "settlement_date": "2026-09-03",
            "settlement_type": "deliverable",
        },
        "internal": {
            "counterparty": "Standard Chartered Bank",
            "call_currency": "EUR", "call_amount": 10000000,
            "put_currency": "USD", "put_amount": 10850000,
            "strike": 1.0850,
            "settlement_date": "2026-09-03",
            "settlement_type": "deliverable",
        },
        "expect_exception": False,
    },

    # ------------------------------------------------------------------
    # 2. Strike rate mismatch — 1.0850 vs 1.0875
    # ------------------------------------------------------------------
    {
        "id": "FX-002",
        "desc": "Strike mismatch: 1.0850 confirmed vs 1.0875 booked",
        "confirmation": {
            "counterparty": "Goldman Sachs Bank USA",
            "call_currency": "EUR", "call_amount": 10000000,
            "put_currency": "USD", "put_amount": 10850000,
            "strike": 1.0850,
            "settlement_date": "2026-09-03",
            "settlement_type": "deliverable",
        },
        "internal": {
            "counterparty": "Goldman Sachs Bank USA",
            "call_currency": "EUR", "call_amount": 10000000,
            "put_currency": "USD", "put_amount": 10875000,
            "strike": 1.0875,
            "settlement_date": "2026-09-03",
            "settlement_type": "deliverable",
        },
        "expect_exception": True,
        "expect_category": EXC.STRIKE,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 3. Wrong call currency — EUR vs GBP
    # ------------------------------------------------------------------
    {
        "id": "FX-003",
        "desc": "Call currency mismatch: EUR confirmed vs GBP booked",
        "confirmation": {
            "counterparty": "JPMorgan Chase Bank NA",
            "call_currency": "EUR", "call_amount": 10000000,
            "put_currency": "USD", "put_amount": 10850000,
            "strike": 1.0850,
            "settlement_date": "2026-09-03",
        },
        "internal": {
            "counterparty": "JPMorgan Chase Bank NA",
            "call_currency": "GBP", "call_amount": 10000000,
            "put_currency": "USD", "put_amount": 12650000,
            "strike": 1.2650,
            "settlement_date": "2026-09-03",
        },
        "expect_exception": True,
        "expect_category": EXC.CALL_CCY,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 4. Settlement date mismatch — 1 day off
    # ------------------------------------------------------------------
    {
        "id": "FX-004",
        "desc": "Settlement date mismatch: Sep 4 confirmed vs Sep 3 booked",
        "confirmation": {
            "counterparty": "Barclays Bank PLC",
            "call_currency": "EUR", "call_amount": 5000000,
            "put_currency": "USD", "put_amount": 5425000,
            "strike": 1.0850,
            "settlement_date": "2026-09-04",
            "settlement_type": "deliverable",
        },
        "internal": {
            "counterparty": "Barclays Bank PLC",
            "call_currency": "EUR", "call_amount": 5000000,
            "put_currency": "USD", "put_amount": 5425000,
            "strike": 1.0850,
            "settlement_date": "2026-09-03",
            "settlement_type": "deliverable",
        },
        "expect_exception": True,
        "expect_category": EXC.SETTLE_DATE,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 5. Counterparty mismatch
    # ------------------------------------------------------------------
    {
        "id": "FX-005",
        "desc": "Counterparty mismatch: SCB London vs SCB Singapore",
        "confirmation": {
            "counterparty": "Standard Chartered Bank Singapore",
            "call_currency": "USD", "call_amount": 10000000,
            "put_currency": "SGD", "put_amount": 13450000,
            "strike": 1.3450,
            "settlement_date": "2026-09-03",
        },
        "internal": {
            "counterparty": "Standard Chartered Bank",
            "call_currency": "USD", "call_amount": 10000000,
            "put_currency": "SGD", "put_amount": 13450000,
            "strike": 1.3450,
            "settlement_date": "2026-09-03",
        },
        "expect_exception": True,
        "expect_category": EXC.CPTY,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 6. NDO — clean USD/IDR non-deliverable forward
    # ------------------------------------------------------------------
    {
        "id": "FX-006",
        "desc": "Clean USD/IDR NDO — $5M at 15,750",
        "confirmation": {
            "counterparty": "Standard Chartered Bank",
            "call_currency": "USD", "call_amount": 5000000,
            "put_currency": "IDR", "put_amount": 78750000000,
            "strike": 15750.0,
            "settlement_date": "2026-09-03",
            "settlement_type": "ndo",
            "reference_currency": "IDR",
        },
        "internal": {
            "counterparty": "Standard Chartered Bank",
            "call_currency": "USD", "call_amount": 5000000,
            "put_currency": "IDR", "put_amount": 78750000000,
            "strike": 15750.0,
            "settlement_date": "2026-09-03",
            "settlement_type": "ndo",
            "reference_currency": "IDR",
        },
        "expect_exception": False,
    },

    # ------------------------------------------------------------------
    # 7. NDO strike mismatch — IDR/USD rate
    # ------------------------------------------------------------------
    {
        "id": "FX-007",
        "desc": "NDO strike mismatch: 15750 confirmed vs 15800 booked",
        "confirmation": {
            "counterparty": "Standard Chartered Bank",
            "call_currency": "USD", "call_amount": 5000000,
            "put_currency": "IDR", "put_amount": 78750000000,
            "strike": 15750.0,
            "settlement_date": "2026-09-03",
            "settlement_type": "ndo",
            "reference_currency": "IDR",
        },
        "internal": {
            "counterparty": "Standard Chartered Bank",
            "call_currency": "USD", "call_amount": 5000000,
            "put_currency": "IDR", "put_amount": 79000000000,
            "strike": 15800.0,
            "settlement_date": "2026-09-03",
            "settlement_type": "ndo",
            "reference_currency": "IDR",
        },
        "expect_exception": True,
        "expect_category": EXC.STRIKE,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 8. FX option — clean European USD call / EUR put
    # ------------------------------------------------------------------
    {
        "id": "FX-008",
        "desc": "Clean European EUR/USD call option — strike 1.09, €10M",
        "confirmation": {
            "counterparty": "Goldman Sachs Bank USA",
            "call_currency": "EUR", "call_amount": 10000000,
            "put_currency": "USD", "put_amount": 10900000,
            "strike": 1.0900,
            "option_type": "call",
            "option_style": "european",
            "expiry_date": "2026-12-17",
            "settlement_date": "2026-12-19",
            "settlement_type": "deliverable",
            "premium": 125000,
            "premium_date": "2026-07-08",
        },
        "internal": {
            "counterparty": "Goldman Sachs Bank USA",
            "call_currency": "EUR", "call_amount": 10000000,
            "put_currency": "USD", "put_amount": 10900000,
            "strike": 1.0900,
            "option_type": "call",
            "option_style": "european",
            "expiry_date": "2026-12-17",
            "settlement_date": "2026-12-19",
            "settlement_type": "deliverable",
            "premium": 125000,
            "premium_date": "2026-07-08",
        },
        "expect_exception": False,
    },

    # ------------------------------------------------------------------
    # 9. FX option — call vs put mismatch
    # ------------------------------------------------------------------
    {
        "id": "FX-009",
        "desc": "Option type mismatch: call confirmed vs put booked",
        "confirmation": {
            "counterparty": "Morgan Stanley & Co. LLC",
            "call_currency": "EUR", "call_amount": 10000000,
            "put_currency": "USD", "put_amount": 10900000,
            "strike": 1.0900,
            "option_type": "call",
            "option_style": "european",
            "expiry_date": "2026-12-17",
            "premium": 125000,
        },
        "internal": {
            "counterparty": "Morgan Stanley & Co. LLC",
            "call_currency": "EUR", "call_amount": 10000000,
            "put_currency": "USD", "put_amount": 10900000,
            "strike": 1.0900,
            "option_type": "put",
            "option_style": "european",
            "expiry_date": "2026-12-17",
            "premium": 125000,
        },
        "expect_exception": True,
        "expect_category": EXC.OPT_TYPE,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 10. Dual: wrong strike + wrong expiry
    # ------------------------------------------------------------------
    {
        "id": "FX-010",
        "desc": "Dual: strike 1.0850 vs 1.0900 + expiry Dec 16 vs Dec 17",
        "confirmation": {
            "counterparty": "Citibank NA",
            "call_currency": "EUR", "call_amount": 10000000,
            "put_currency": "USD", "put_amount": 10900000,
            "strike": 1.0850,
            "option_type": "call",
            "option_style": "european",
            "expiry_date": "2026-12-16",
            "premium": 125000,
        },
        "internal": {
            "counterparty": "Citibank NA",
            "call_currency": "EUR", "call_amount": 10000000,
            "put_currency": "USD", "put_amount": 10900000,
            "strike": 1.0900,
            "option_type": "call",
            "option_style": "european",
            "expiry_date": "2026-12-17",
            "premium": 125000,
        },
        "expect_exception": True,
        "expect_category": EXC.STRIKE,
        "expect_secondary": EXC.EXPIRY,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 11. Call amount mismatch — €10M vs €9.5M
    # ------------------------------------------------------------------
    {
        "id": "FX-011",
        "desc": "Call amount mismatch: €10M confirmed vs €9.5M booked",
        "confirmation": {
            "counterparty": "Barclays Bank PLC",
            "call_currency": "EUR", "call_amount": 10000000,
            "put_currency": "USD", "put_amount": 10850000,
            "strike": 1.0850,
            "settlement_date": "2026-09-03",
        },
        "internal": {
            "counterparty": "Barclays Bank PLC",
            "call_currency": "EUR", "call_amount": 9500000,
            "put_currency": "USD", "put_amount": 10307500,
            "strike": 1.0850,
            "settlement_date": "2026-09-03",
        },
        "expect_exception": True,
        "expect_category": EXC.CALL_AMT,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 12. Deliverable vs NDO settlement type mismatch
    # ------------------------------------------------------------------
    {
        "id": "FX-012",
        "desc": "Settlement type mismatch: deliverable confirmed vs NDO booked",
        "confirmation": {
            "counterparty": "Standard Chartered Bank",
            "call_currency": "USD", "call_amount": 5000000,
            "put_currency": "BRL", "put_amount": 25000000,
            "strike": 5.0000,
            "settlement_date": "2026-09-03",
            "settlement_type": "deliverable",
        },
        "internal": {
            "counterparty": "Standard Chartered Bank",
            "call_currency": "USD", "call_amount": 5000000,
            "put_currency": "BRL", "put_amount": 25000000,
            "strike": 5.0000,
            "settlement_date": "2026-09-03",
            "settlement_type": "ndo",
        },
        "expect_exception": True,
        "expect_category": EXC.SETTLE_TYPE,
        "expect_severity": "medium",
    },
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
passed = failed = 0
SEP = "─" * 72

print(f"FX Forward/Option Reconciler — {len(CASES)} cases\n{SEP}")

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
