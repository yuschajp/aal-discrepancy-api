#!/usr/bin/env python3
"""
test_equity_option_reconciler.py — regression suite for the equity option engine.

Runs synthetic test cases covering the most common FIA/RILA hedging structures.
Replace with AAL-D-003 ground truth cases when the dataset is built.

Usage:
    python3 test_equity_option_reconciler.py
    python3 test_equity_option_reconciler.py --verbose
"""
import sys, argparse
from equity_option_reconciler import EquityOptionReconciler, EXC

ap = argparse.ArgumentParser()
ap.add_argument("--verbose", "-v", action="store_true")
args = ap.parse_args()

reconciler = EquityOptionReconciler()

# ---------------------------------------------------------------------------
# Test cases — synthetic ground truth covering key FIA/RILA patterns
# Each: confirmation dict, internal dict, expected exception, expected category
# ---------------------------------------------------------------------------
CASES = [
    # ------------------------------------------------------------------
    # 1. Clean vanilla call — no exception
    # ------------------------------------------------------------------
    {
        "id": "EQ-001",
        "desc": "Clean European call on SPX — FIA hedge, no discrepancy",
        "confirmation": {
            "counterparty": "Goldman Sachs Bank USA",
            "option_type": "call",
            "option_style": "european",
            "underlying": "S&P 500 Index",
            "strike": 5200.00,
            "num_options": 100000,
            "premium": 12500000,
            "expiry_date": "2027-07-01",
            "settlement_currency": "USD",
            "settlement_method": "cash",
            "buyer": "F&G Life Insurance",
        },
        "internal": {
            "counterparty": "Goldman Sachs Bank USA",
            "option_type": "call",
            "option_style": "european",
            "underlying": "SPX",
            "strike": 5200.00,
            "num_options": 100000,
            "premium": 12500000,
            "expiry_date": "2027-07-01",
            "settlement_currency": "USD",
            "settlement_method": "cash",
            "buyer": "F&G Life Insurance",
        },
        "expect_exception": False,
        "expect_category": None,
    },

    # ------------------------------------------------------------------
    # 2. Strike mismatch — 5 index points on SPX call
    # ------------------------------------------------------------------
    {
        "id": "EQ-002",
        "desc": "Strike mismatch: confirmation 5205 vs internal 5200",
        "confirmation": {
            "counterparty": "Goldman Sachs Bank USA",
            "option_type": "call",
            "underlying": "SPX",
            "strike": 5205.00,
            "num_options": 100000,
            "premium": 12500000,
            "expiry_date": "2027-07-01",
            "settlement_currency": "USD",
        },
        "internal": {
            "counterparty": "Goldman Sachs Bank USA",
            "option_type": "call",
            "underlying": "SPX",
            "strike": 5200.00,
            "num_options": 100000,
            "premium": 12500000,
            "expiry_date": "2027-07-01",
            "settlement_currency": "USD",
        },
        "expect_exception": True,
        "expect_category": EXC.STRIKE,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 3. Underlying mismatch — Russell 2000 vs S&P 500
    # ------------------------------------------------------------------
    {
        "id": "EQ-003",
        "desc": "Underlying mismatch: RTY vs SPX",
        "confirmation": {
            "counterparty": "Morgan Stanley & Co. LLC",
            "option_type": "call",
            "underlying": "Russell 2000",
            "strike": 2100.00,
            "num_options": 50000,
            "premium": 5000000,
            "expiry_date": "2027-07-01",
            "settlement_currency": "USD",
        },
        "internal": {
            "counterparty": "Morgan Stanley & Co. LLC",
            "option_type": "call",
            "underlying": "SPX",
            "strike": 2100.00,
            "num_options": 50000,
            "premium": 5000000,
            "expiry_date": "2027-07-01",
            "settlement_currency": "USD",
        },
        "expect_exception": True,
        "expect_category": EXC.UNDL,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 4. Counterparty legal entity mismatch
    # ------------------------------------------------------------------
    {
        "id": "EQ-004",
        "desc": "Counterparty mismatch: GS Bank USA vs GS International",
        "confirmation": {
            "counterparty": "Goldman Sachs International",
            "option_type": "call",
            "underlying": "SPX",
            "strike": 5200.00,
            "num_options": 100000,
            "premium": 12500000,
            "expiry_date": "2027-07-01",
            "settlement_currency": "USD",
        },
        "internal": {
            "counterparty": "Goldman Sachs Bank USA",
            "option_type": "call",
            "underlying": "SPX",
            "strike": 5200.00,
            "num_options": 100000,
            "premium": 12500000,
            "expiry_date": "2027-07-01",
            "settlement_currency": "USD",
        },
        "expect_exception": True,
        "expect_category": EXC.CPTY,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 5. Premium mismatch — $250K off on $12.5M premium
    # ------------------------------------------------------------------
    {
        "id": "EQ-005",
        "desc": "Premium mismatch: $12,750,000 vs $12,500,000",
        "confirmation": {
            "counterparty": "Goldman Sachs Bank USA",
            "option_type": "call",
            "underlying": "SPX",
            "strike": 5200.00,
            "num_options": 100000,
            "premium": 12750000,
            "expiry_date": "2027-07-01",
            "settlement_currency": "USD",
        },
        "internal": {
            "counterparty": "Goldman Sachs Bank USA",
            "option_type": "call",
            "underlying": "SPX",
            "strike": 5200.00,
            "num_options": 100000,
            "premium": 12500000,
            "expiry_date": "2027-07-01",
            "settlement_currency": "USD",
        },
        "expect_exception": True,
        "expect_category": EXC.PREM,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 6. Expiry date mismatch — 1 day off
    # ------------------------------------------------------------------
    {
        "id": "EQ-006",
        "desc": "Expiry date mismatch: July 2 vs July 1",
        "confirmation": {
            "counterparty": "Barclays Bank PLC",
            "option_type": "call",
            "underlying": "SPX",
            "strike": 5200.00,
            "num_options": 75000,
            "premium": 9000000,
            "expiry_date": "2027-07-02",
            "settlement_currency": "USD",
        },
        "internal": {
            "counterparty": "Barclays Bank PLC",
            "option_type": "call",
            "underlying": "SPX",
            "strike": 5200.00,
            "num_options": 75000,
            "premium": 9000000,
            "expiry_date": "2027-07-01",
            "settlement_currency": "USD",
        },
        "expect_exception": True,
        "expect_category": EXC.EXPIRY,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 7. Call spread — clean (two strikes, both match)
    # ------------------------------------------------------------------
    {
        "id": "EQ-007",
        "desc": "Clean call spread: 5200/5800 strike, FIA hedge",
        "confirmation": {
            "counterparty": "Citibank NA",
            "option_type": "call",
            "underlying": "S&P 500 Index",
            "strike": 5200.00,
            "strike_high": 5800.00,
            "num_options": 200000,
            "premium": 8000000,
            "expiry_date": "2027-07-01",
            "settlement_currency": "USD",
        },
        "internal": {
            "counterparty": "Citibank NA",
            "option_type": "call",
            "underlying": "SPX",
            "strike": 5200.00,
            "strike_high": 5800.00,
            "num_options": 200000,
            "premium": 8000000,
            "expiry_date": "2027-07-01",
            "settlement_currency": "USD",
        },
        "expect_exception": False,
        "expect_category": None,
    },

    # ------------------------------------------------------------------
    # 8. Call spread — upper strike wrong
    # ------------------------------------------------------------------
    {
        "id": "EQ-008",
        "desc": "Call spread upper strike mismatch: 5850 vs 5800",
        "confirmation": {
            "counterparty": "Citibank NA",
            "option_type": "call",
            "underlying": "SPX",
            "strike": 5200.00,
            "strike_high": 5850.00,
            "num_options": 200000,
            "premium": 8000000,
            "expiry_date": "2027-07-01",
            "settlement_currency": "USD",
        },
        "internal": {
            "counterparty": "Citibank NA",
            "option_type": "call",
            "underlying": "SPX",
            "strike": 5200.00,
            "strike_high": 5800.00,
            "num_options": 200000,
            "premium": 8000000,
            "expiry_date": "2027-07-01",
            "settlement_currency": "USD",
        },
        "expect_exception": True,
        "expect_category": EXC.STRIKE,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 9. Quantity mismatch — 100K vs 75K options
    # ------------------------------------------------------------------
    {
        "id": "EQ-009",
        "desc": "Number of options mismatch: 100,000 vs 75,000",
        "confirmation": {
            "counterparty": "JPMorgan Chase Bank NA",
            "option_type": "call",
            "underlying": "SPX",
            "strike": 5200.00,
            "num_options": 100000,
            "premium": 12500000,
            "expiry_date": "2027-07-01",
            "settlement_currency": "USD",
        },
        "internal": {
            "counterparty": "JPMorgan Chase Bank NA",
            "option_type": "call",
            "underlying": "SPX",
            "strike": 5200.00,
            "num_options": 75000,
            "premium": 12500000,
            "expiry_date": "2027-07-01",
            "settlement_currency": "USD",
        },
        "expect_exception": True,
        "expect_category": EXC.QTY,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 10. Call vs Put mismatch — RILA hedge gone wrong
    # ------------------------------------------------------------------
    {
        "id": "EQ-010",
        "desc": "Option type mismatch: call confirmed vs put booked",
        "confirmation": {
            "counterparty": "Goldman Sachs Bank USA",
            "option_type": "call",
            "underlying": "SPX",
            "strike": 5200.00,
            "num_options": 50000,
            "premium": 6000000,
            "expiry_date": "2027-07-01",
            "settlement_currency": "USD",
        },
        "internal": {
            "counterparty": "Goldman Sachs Bank USA",
            "option_type": "put",
            "underlying": "SPX",
            "strike": 5200.00,
            "num_options": 50000,
            "premium": 6000000,
            "expiry_date": "2027-07-01",
            "settlement_currency": "USD",
        },
        "expect_exception": True,
        "expect_category": EXC.TYPE,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 11. Dual exception: wrong underlying AND wrong expiry
    # ------------------------------------------------------------------
    {
        "id": "EQ-011",
        "desc": "Dual: RTY confirmed vs SPX booked + expiry 1 month off",
        "confirmation": {
            "counterparty": "Morgan Stanley & Co. LLC",
            "option_type": "call",
            "underlying": "Russell 2000",
            "strike": 2100.00,
            "num_options": 50000,
            "premium": 5000000,
            "expiry_date": "2027-08-01",
            "settlement_currency": "USD",
        },
        "internal": {
            "counterparty": "Morgan Stanley & Co. LLC",
            "option_type": "call",
            "underlying": "SPX",
            "strike": 2100.00,
            "num_options": 50000,
            "premium": 5000000,
            "expiry_date": "2027-07-01",
            "settlement_currency": "USD",
        },
        "expect_exception": True,
        "expect_category": EXC.UNDL,
        "expect_secondary": EXC.EXPIRY,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 12. Index normalization — "S&P 500 Index" vs "SPX" should be clean
    # ------------------------------------------------------------------
    {
        "id": "EQ-012",
        "desc": "Index name normalization: S&P 500 Index == SPX (clean)",
        "confirmation": {
            "counterparty": "Barclays Bank PLC",
            "option_type": "call",
            "underlying": "S&P 500 Index",
            "strike": 5200.00,
            "num_options": 100000,
            "premium": 12500000,
            "expiry_date": "2027-07-01",
            "settlement_currency": "USD",
        },
        "internal": {
            "counterparty": "Barclays Bank PLC",
            "option_type": "call",
            "underlying": "SPX",
            "strike": 5200.00,
            "num_options": 100000,
            "premium": 12500000,
            "expiry_date": "2027-07-01",
            "settlement_currency": "USD",
        },
        "expect_exception": False,
        "expect_category": None,
    },
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
passed = failed = 0
SEP = "─" * 72

print(f"Equity Option Reconciler — {len(CASES)} cases\n{SEP}")

for tc in CASES:
    result = reconciler.reconcile(tc["confirmation"], tc["internal"], case_id=tc["id"])

    ok = True
    got_exc = result.exception_exists

    if got_exc != tc["expect_exception"]:
        print(f"  FAIL {tc['id']} — detection: got={got_exc} expected={tc['expect_exception']}")
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
            print(f"  status={result.overall_severity}  exception={result.exception_exists}")
            if result.primary:
                print(f"  primary={result.primary.category} on {result.primary.field}")
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
