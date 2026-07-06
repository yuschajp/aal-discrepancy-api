#!/usr/bin/env python3
"""
test_cds_reconciler.py — regression suite for the CDS engine.

Reference: Standard Chartered Bank Single-Name CDS (Auction Physical Settlement)
2014 ISDA Credit Derivatives Definitions.

Usage:
    python3 test_cds_reconciler.py
    python3 test_cds_reconciler.py --verbose
"""
import sys, argparse
from cds_reconciler import CDSReconciler, EXC

ap = argparse.ArgumentParser()
ap.add_argument("--verbose", "-v", action="store_true")
args = ap.parse_args()

reconciler = CDSReconciler()

CASES = [
    # ------------------------------------------------------------------
    # 1. Clean single-name CDS — IG corporate
    # ------------------------------------------------------------------
    {
        "id": "CDS-001",
        "desc": "Clean Ford Motor Credit 5Y CDS — $10M notional, 150bp",
        "confirmation": {
            "counterparty": "Standard Chartered Bank",
            "reference_entity": "Ford Motor Credit Company LLC",
            "notional": 10000000,
            "currency": "USD",
            "fixed_rate": 1.50,
            "protection_buyer": "F&G Life Insurance Company",
            "protection_seller": "Standard Chartered Bank",
            "effective_date": "2026-07-05",
            "termination_date": "2031-07-20",
            "seniority": "senior",
            "day_count": "ACT/360",
            "credit_events": ["bankruptcy", "failure to pay", "restructuring"],
            "settlement_method": "auction",
        },
        "internal": {
            "counterparty": "Standard Chartered Bank",
            "reference_entity": "Ford Motor Credit Company LLC",
            "notional": 10000000,
            "currency": "USD",
            "fixed_rate": 1.50,
            "protection_buyer": "F&G Life Insurance Company",
            "protection_seller": "Standard Chartered Bank",
            "effective_date": "2026-07-05",
            "termination_date": "2031-07-20",
            "seniority": "senior",
            "day_count": "ACT/360",
            "credit_events": ["bankruptcy", "failure to pay", "restructuring"],
            "settlement_method": "auction",
        },
        "expect_exception": False,
    },

    # ------------------------------------------------------------------
    # 2. Wrong reference entity
    # ------------------------------------------------------------------
    {
        "id": "CDS-002",
        "desc": "Reference entity mismatch: Ford Motor Credit vs Ford Motor Co",
        "confirmation": {
            "counterparty": "Goldman Sachs Bank USA",
            "reference_entity": "Ford Motor Company",
            "notional": 10000000,
            "currency": "USD",
            "fixed_rate": 1.50,
            "effective_date": "2026-07-05",
            "termination_date": "2031-07-20",
            "seniority": "senior",
            "credit_events": ["bankruptcy", "failure to pay", "restructuring"],
        },
        "internal": {
            "counterparty": "Goldman Sachs Bank USA",
            "reference_entity": "Ford Motor Credit Company LLC",
            "notional": 10000000,
            "currency": "USD",
            "fixed_rate": 1.50,
            "effective_date": "2026-07-05",
            "termination_date": "2031-07-20",
            "seniority": "senior",
            "credit_events": ["bankruptcy", "failure to pay", "restructuring"],
        },
        "expect_exception": True,
        "expect_category": EXC.REF_ENT,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 3. Spread mismatch — 150bp confirmed vs 125bp booked
    # ------------------------------------------------------------------
    {
        "id": "CDS-003",
        "desc": "Spread mismatch: 150bp confirmed vs 125bp booked",
        "confirmation": {
            "counterparty": "JPMorgan Chase Bank NA",
            "reference_entity": "Ford Motor Credit Company LLC",
            "notional": 10000000,
            "currency": "USD",
            "fixed_rate": 1.50,
            "effective_date": "2026-07-05",
            "termination_date": "2031-07-20",
            "seniority": "senior",
            "credit_events": ["bankruptcy", "failure to pay", "restructuring"],
        },
        "internal": {
            "counterparty": "JPMorgan Chase Bank NA",
            "reference_entity": "Ford Motor Credit Company LLC",
            "notional": 10000000,
            "currency": "USD",
            "fixed_rate": 1.25,
            "effective_date": "2026-07-05",
            "termination_date": "2031-07-20",
            "seniority": "senior",
            "credit_events": ["bankruptcy", "failure to pay", "restructuring"],
        },
        "expect_exception": True,
        "expect_category": EXC.SPREAD,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 4. Notional mismatch — $10M vs $5M
    # ------------------------------------------------------------------
    {
        "id": "CDS-004",
        "desc": "Notional mismatch: $10M confirmed vs $5M booked",
        "confirmation": {
            "counterparty": "Barclays Bank PLC",
            "reference_entity": "Ford Motor Credit Company LLC",
            "notional": 10000000,
            "currency": "USD",
            "fixed_rate": 1.50,
            "effective_date": "2026-07-05",
            "termination_date": "2031-07-20",
            "credit_events": ["bankruptcy", "failure to pay", "restructuring"],
        },
        "internal": {
            "counterparty": "Barclays Bank PLC",
            "reference_entity": "Ford Motor Credit Company LLC",
            "notional": 5000000,
            "currency": "USD",
            "fixed_rate": 1.50,
            "effective_date": "2026-07-05",
            "termination_date": "2031-07-20",
            "credit_events": ["bankruptcy", "failure to pay", "restructuring"],
        },
        "expect_exception": True,
        "expect_category": EXC.NOTIONAL,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 5. Direction mismatch — buyer vs seller swapped
    # ------------------------------------------------------------------
    {
        "id": "CDS-005",
        "desc": "Direction mismatch: F&G as buyer confirmed vs seller booked",
        "confirmation": {
            "counterparty": "Morgan Stanley & Co. LLC",
            "reference_entity": "Ford Motor Credit Company LLC",
            "notional": 10000000,
            "currency": "USD",
            "fixed_rate": 1.50,
            "protection_buyer": "F&G Life Insurance Company",
            "protection_seller": "Morgan Stanley & Co. LLC",
            "effective_date": "2026-07-05",
            "termination_date": "2031-07-20",
            "credit_events": ["bankruptcy", "failure to pay", "restructuring"],
        },
        "internal": {
            "counterparty": "Morgan Stanley & Co. LLC",
            "reference_entity": "Ford Motor Credit Company LLC",
            "notional": 10000000,
            "currency": "USD",
            "fixed_rate": 1.50,
            "protection_buyer": "Morgan Stanley & Co. LLC",
            "protection_seller": "F&G Life Insurance Company",
            "effective_date": "2026-07-05",
            "termination_date": "2031-07-20",
            "credit_events": ["bankruptcy", "failure to pay", "restructuring"],
        },
        "expect_exception": True,
        "expect_category": EXC.DIR,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 6. Termination date mismatch
    # ------------------------------------------------------------------
    {
        "id": "CDS-006",
        "desc": "Termination date mismatch: 2031 vs 2032",
        "confirmation": {
            "counterparty": "Citibank NA",
            "reference_entity": "Ford Motor Credit Company LLC",
            "notional": 10000000,
            "currency": "USD",
            "fixed_rate": 1.50,
            "effective_date": "2026-07-05",
            "termination_date": "2032-07-20",
            "credit_events": ["bankruptcy", "failure to pay", "restructuring"],
        },
        "internal": {
            "counterparty": "Citibank NA",
            "reference_entity": "Ford Motor Credit Company LLC",
            "notional": 10000000,
            "currency": "USD",
            "fixed_rate": 1.50,
            "effective_date": "2026-07-05",
            "termination_date": "2031-07-20",
            "credit_events": ["bankruptcy", "failure to pay", "restructuring"],
        },
        "expect_exception": True,
        "expect_category": EXC.TERM_DATE,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 7. Credit event set mismatch — restructuring missing
    # ------------------------------------------------------------------
    {
        "id": "CDS-007",
        "desc": "Credit events mismatch: restructuring missing from confirmation",
        "confirmation": {
            "counterparty": "Goldman Sachs Bank USA",
            "reference_entity": "Ford Motor Credit Company LLC",
            "notional": 10000000,
            "currency": "USD",
            "fixed_rate": 1.50,
            "effective_date": "2026-07-05",
            "termination_date": "2031-07-20",
            "credit_events": ["bankruptcy", "failure to pay"],
        },
        "internal": {
            "counterparty": "Goldman Sachs Bank USA",
            "reference_entity": "Ford Motor Credit Company LLC",
            "notional": 10000000,
            "currency": "USD",
            "fixed_rate": 1.50,
            "effective_date": "2026-07-05",
            "termination_date": "2031-07-20",
            "credit_events": ["bankruptcy", "failure to pay", "restructuring"],
        },
        "expect_exception": True,
        "expect_category": EXC.CREDIT_EVT,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 8. Seniority mismatch — senior vs subordinated
    # ------------------------------------------------------------------
    {
        "id": "CDS-008",
        "desc": "Seniority mismatch: senior confirmed vs subordinated booked",
        "confirmation": {
            "counterparty": "Standard Chartered Bank",
            "reference_entity": "Deutsche Bank AG",
            "notional": 5000000,
            "currency": "USD",
            "fixed_rate": 2.50,
            "effective_date": "2026-07-05",
            "termination_date": "2031-07-20",
            "seniority": "senior",
            "credit_events": ["bankruptcy", "failure to pay", "restructuring"],
        },
        "internal": {
            "counterparty": "Standard Chartered Bank",
            "reference_entity": "Deutsche Bank AG",
            "notional": 5000000,
            "currency": "USD",
            "fixed_rate": 2.50,
            "effective_date": "2026-07-05",
            "termination_date": "2031-07-20",
            "seniority": "subordinated",
            "credit_events": ["bankruptcy", "failure to pay", "restructuring"],
        },
        "expect_exception": True,
        "expect_category": EXC.SENIORITY,
        "expect_severity": "medium",
    },

    # ------------------------------------------------------------------
    # 9. Counterparty mismatch — SCB London vs SCB Singapore
    # ------------------------------------------------------------------
    {
        "id": "CDS-009",
        "desc": "Counterparty mismatch: SCB vs JPMorgan",
        "confirmation": {
            "counterparty": "JPMorgan Chase Bank NA",
            "reference_entity": "Ford Motor Credit Company LLC",
            "notional": 10000000,
            "currency": "USD",
            "fixed_rate": 1.50,
            "effective_date": "2026-07-05",
            "termination_date": "2031-07-20",
            "credit_events": ["bankruptcy", "failure to pay", "restructuring"],
        },
        "internal": {
            "counterparty": "Standard Chartered Bank",
            "reference_entity": "Ford Motor Credit Company LLC",
            "notional": 10000000,
            "currency": "USD",
            "fixed_rate": 1.50,
            "effective_date": "2026-07-05",
            "termination_date": "2031-07-20",
            "credit_events": ["bankruptcy", "failure to pay", "restructuring"],
        },
        "expect_exception": True,
        "expect_category": EXC.CPTY,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 10. Dual: wrong reference entity + wrong spread
    # ------------------------------------------------------------------
    {
        "id": "CDS-010",
        "desc": "Dual: wrong reference entity + spread 150bp vs 125bp",
        "confirmation": {
            "counterparty": "Goldman Sachs Bank USA",
            "reference_entity": "Ford Motor Company",
            "notional": 10000000,
            "currency": "USD",
            "fixed_rate": 1.50,
            "effective_date": "2026-07-05",
            "termination_date": "2031-07-20",
            "credit_events": ["bankruptcy", "failure to pay", "restructuring"],
        },
        "internal": {
            "counterparty": "Goldman Sachs Bank USA",
            "reference_entity": "Ford Motor Credit Company LLC",
            "notional": 10000000,
            "currency": "USD",
            "fixed_rate": 1.25,
            "effective_date": "2026-07-05",
            "termination_date": "2031-07-20",
            "credit_events": ["bankruptcy", "failure to pay", "restructuring"],
        },
        "expect_exception": True,
        "expect_category": EXC.REF_ENT,
        "expect_secondary": EXC.SPREAD,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 11. Credit event set — comma-separated string normalization (clean)
    # ------------------------------------------------------------------
    {
        "id": "CDS-011",
        "desc": "Credit event normalization: list vs comma string (clean)",
        "confirmation": {
            "counterparty": "Standard Chartered Bank",
            "reference_entity": "Ford Motor Credit Company LLC",
            "notional": 10000000,
            "currency": "USD",
            "fixed_rate": 1.50,
            "effective_date": "2026-07-05",
            "termination_date": "2031-07-20",
            "credit_events": "Bankruptcy, Failure to Pay, Restructuring",
        },
        "internal": {
            "counterparty": "Standard Chartered Bank",
            "reference_entity": "Ford Motor Credit Company LLC",
            "notional": 10000000,
            "currency": "USD",
            "fixed_rate": 1.50,
            "effective_date": "2026-07-05",
            "termination_date": "2031-07-20",
            "credit_events": ["bankruptcy", "failure to pay", "restructuring"],
        },
        "expect_exception": False,
    },

    # ------------------------------------------------------------------
    # 12. Sovereign CDS — clean (Republic of Indonesia, per SCB template)
    # ------------------------------------------------------------------
    {
        "id": "CDS-012",
        "desc": "Clean sovereign CDS — Republic of Indonesia $5M 200bp",
        "confirmation": {
            "counterparty": "Standard Chartered Bank",
            "reference_entity": "Republic of Indonesia",
            "notional": 5000000,
            "currency": "USD",
            "fixed_rate": 2.00,
            "protection_buyer": "F&G Life Insurance Company",
            "protection_seller": "Standard Chartered Bank",
            "effective_date": "2026-07-05",
            "termination_date": "2031-07-20",
            "seniority": "senior",
            "day_count": "ACT/360",
            "credit_events": [
                "failure to pay", "obligation default",
                "obligation acceleration", "repudiation/moratorium",
                "governmental intervention",
            ],
            "settlement_method": "auction",
        },
        "internal": {
            "counterparty": "Standard Chartered Bank",
            "reference_entity": "Republic of Indonesia",
            "notional": 5000000,
            "currency": "USD",
            "fixed_rate": 2.00,
            "protection_buyer": "F&G Life Insurance Company",
            "protection_seller": "Standard Chartered Bank",
            "effective_date": "2026-07-05",
            "termination_date": "2031-07-20",
            "seniority": "senior",
            "day_count": "ACT/360",
            "credit_events": [
                "failure to pay", "obligation default",
                "obligation acceleration", "repudiation/moratorium",
                "governmental intervention",
            ],
            "settlement_method": "auction",
        },
        "expect_exception": False,
    },
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
passed = failed = 0
SEP = "─" * 72

print(f"CDS Reconciler — {len(CASES)} cases\n{SEP}")

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
