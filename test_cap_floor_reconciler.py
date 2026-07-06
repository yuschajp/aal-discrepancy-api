#!/usr/bin/env python3
"""
test_cap_floor_reconciler.py — regression suite for the cap/floor engine.

Reference confirmation: Bank of America / Goal Capital Funding Trust 2007-1
  $35M notional, LIBOR cap at 7.00%, quarterly, ACT/360, 5yr, $89,500 premium

Usage:
    python3 test_cap_floor_reconciler.py
    python3 test_cap_floor_reconciler.py --verbose
"""
import sys, argparse
from cap_floor_reconciler import CapFloorReconciler, EXC

ap = argparse.ArgumentParser()
ap.add_argument("--verbose", "-v", action="store_true")
args = ap.parse_args()

reconciler = CapFloorReconciler()

CASES = [
    # ------------------------------------------------------------------
    # 1. Clean cap — matches BofA/Goal Capital structure
    # ------------------------------------------------------------------
    {
        "id": "CF-001",
        "desc": "Clean LIBOR cap — $35M 7.00% quarterly ACT/360 5yr",
        "confirmation": {
            "structure_type": "cap",
            "counterparty": "Bank of America NA",
            "notional": 35000000,
            "currency": "USD",
            "cap_rate": 7.00,
            "floating_rate": "USD-LIBOR-BBA",
            "floating_tenor": "3M",
            "day_count": "ACT/360",
            "effective_date": "2007-06-07",
            "termination_date": "2012-06-25",
            "payment_frequency": "Quarterly",
            "premium": 89500,
        },
        "internal": {
            "structure_type": "cap",
            "counterparty": "Bank of America NA",
            "notional": 35000000,
            "currency": "USD",
            "cap_rate": 7.00,
            "floating_rate": "LIBOR-3M",
            "floating_tenor": "3M",
            "day_count": "ACT/360",
            "effective_date": "2007-06-07",
            "termination_date": "2012-06-25",
            "payment_frequency": "Quarterly",
            "premium": 89500,
        },
        "expect_exception": False,
    },

    # ------------------------------------------------------------------
    # 2. Cap rate mismatch — 7.00% vs 6.50%
    # ------------------------------------------------------------------
    {
        "id": "CF-002",
        "desc": "Cap rate mismatch: 7.00% confirmed vs 6.50% booked",
        "confirmation": {
            "structure_type": "cap",
            "counterparty": "JPMorgan Chase Bank NA",
            "notional": 100000000,
            "currency": "USD",
            "cap_rate": 7.00,
            "floating_rate": "SOFR",
            "day_count": "ACT/360",
            "effective_date": "2026-07-01",
            "termination_date": "2031-07-01",
            "payment_frequency": "Quarterly",
            "premium": 450000,
        },
        "internal": {
            "structure_type": "cap",
            "counterparty": "JPMorgan Chase Bank NA",
            "notional": 100000000,
            "currency": "USD",
            "cap_rate": 6.50,
            "floating_rate": "SOFR",
            "day_count": "ACT/360",
            "effective_date": "2026-07-01",
            "termination_date": "2031-07-01",
            "payment_frequency": "Quarterly",
            "premium": 450000,
        },
        "expect_exception": True,
        "expect_category": EXC.CAP_RATE,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 3. Notional mismatch — $35M vs $50M
    # ------------------------------------------------------------------
    {
        "id": "CF-003",
        "desc": "Notional mismatch: $35M confirmed vs $50M booked",
        "confirmation": {
            "structure_type": "cap",
            "counterparty": "Goldman Sachs Bank USA",
            "notional": 35000000,
            "currency": "USD",
            "cap_rate": 5.50,
            "floating_rate": "SOFR",
            "day_count": "ACT/360",
            "effective_date": "2026-07-01",
            "termination_date": "2029-07-01",
            "payment_frequency": "Quarterly",
        },
        "internal": {
            "structure_type": "cap",
            "counterparty": "Goldman Sachs Bank USA",
            "notional": 50000000,
            "currency": "USD",
            "cap_rate": 5.50,
            "floating_rate": "SOFR",
            "day_count": "ACT/360",
            "effective_date": "2026-07-01",
            "termination_date": "2029-07-01",
            "payment_frequency": "Quarterly",
        },
        "expect_exception": True,
        "expect_category": EXC.NOTIONAL,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 4. Cap vs Floor mismatch
    # ------------------------------------------------------------------
    {
        "id": "CF-004",
        "desc": "Structure type mismatch: cap confirmed vs floor booked",
        "confirmation": {
            "structure_type": "cap",
            "counterparty": "Morgan Stanley Bank NA",
            "notional": 50000000,
            "currency": "USD",
            "cap_rate": 5.50,
            "floating_rate": "SOFR",
            "day_count": "ACT/360",
            "effective_date": "2026-07-01",
            "termination_date": "2031-07-01",
            "payment_frequency": "Quarterly",
        },
        "internal": {
            "structure_type": "floor",
            "counterparty": "Morgan Stanley Bank NA",
            "notional": 50000000,
            "currency": "USD",
            "floor_rate": 5.50,
            "floating_rate": "SOFR",
            "day_count": "ACT/360",
            "effective_date": "2026-07-01",
            "termination_date": "2031-07-01",
            "payment_frequency": "Quarterly",
        },
        "expect_exception": True,
        "expect_category": EXC.TYPE,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 5. Termination date mismatch
    # ------------------------------------------------------------------
    {
        "id": "CF-005",
        "desc": "Termination date mismatch: 2031 vs 2032",
        "confirmation": {
            "structure_type": "cap",
            "counterparty": "Barclays Bank PLC",
            "notional": 75000000,
            "currency": "USD",
            "cap_rate": 5.00,
            "floating_rate": "SOFR",
            "day_count": "ACT/360",
            "effective_date": "2026-07-01",
            "termination_date": "2032-07-01",
            "payment_frequency": "Quarterly",
        },
        "internal": {
            "structure_type": "cap",
            "counterparty": "Barclays Bank PLC",
            "notional": 75000000,
            "currency": "USD",
            "cap_rate": 5.00,
            "floating_rate": "SOFR",
            "day_count": "ACT/360",
            "effective_date": "2026-07-01",
            "termination_date": "2031-07-01",
            "payment_frequency": "Quarterly",
        },
        "expect_exception": True,
        "expect_category": EXC.TERM_DATE,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 6. Premium mismatch — $89,500 vs $75,000
    # ------------------------------------------------------------------
    {
        "id": "CF-006",
        "desc": "Premium mismatch: $89,500 confirmed vs $75,000 booked",
        "confirmation": {
            "structure_type": "cap",
            "counterparty": "Bank of America NA",
            "notional": 35000000,
            "currency": "USD",
            "cap_rate": 7.00,
            "floating_rate": "LIBOR-3M",
            "day_count": "ACT/360",
            "effective_date": "2007-06-07",
            "termination_date": "2012-06-25",
            "payment_frequency": "Quarterly",
            "premium": 89500,
        },
        "internal": {
            "structure_type": "cap",
            "counterparty": "Bank of America NA",
            "notional": 35000000,
            "currency": "USD",
            "cap_rate": 7.00,
            "floating_rate": "LIBOR-3M",
            "day_count": "ACT/360",
            "effective_date": "2007-06-07",
            "termination_date": "2012-06-25",
            "payment_frequency": "Quarterly",
            "premium": 75000,
        },
        "expect_exception": True,
        "expect_category": EXC.PREM,
        "expect_severity": "medium",
    },

    # ------------------------------------------------------------------
    # 7. Floating rate normalization — USD-LIBOR-BBA == LIBOR-3M (clean)
    # ------------------------------------------------------------------
    {
        "id": "CF-007",
        "desc": "Floating rate normalization: USD-LIBOR-BBA == LIBOR-3M (clean)",
        "confirmation": {
            "structure_type": "cap",
            "counterparty": "Bank of America NA",
            "notional": 35000000,
            "currency": "USD",
            "cap_rate": 7.00,
            "floating_rate": "USD-LIBOR-BBA",
            "day_count": "ACT/360",
            "effective_date": "2007-06-07",
            "termination_date": "2012-06-25",
            "payment_frequency": "Quarterly",
            "premium": 89500,
        },
        "internal": {
            "structure_type": "cap",
            "counterparty": "Bank of America NA",
            "notional": 35000000,
            "currency": "USD",
            "cap_rate": 7.00,
            "floating_rate": "LIBOR-3M",
            "day_count": "ACT/360",
            "effective_date": "2007-06-07",
            "termination_date": "2012-06-25",
            "payment_frequency": "Quarterly",
            "premium": 89500,
        },
        "expect_exception": False,
    },

    # ------------------------------------------------------------------
    # 8. Counterparty mismatch
    # ------------------------------------------------------------------
    {
        "id": "CF-008",
        "desc": "Counterparty mismatch: BofA NA vs BofA Securities",
        "confirmation": {
            "structure_type": "cap",
            "counterparty": "BofA Securities Inc",
            "notional": 35000000,
            "currency": "USD",
            "cap_rate": 7.00,
            "floating_rate": "LIBOR-3M",
            "day_count": "ACT/360",
            "effective_date": "2007-06-07",
            "termination_date": "2012-06-25",
            "payment_frequency": "Quarterly",
        },
        "internal": {
            "structure_type": "cap",
            "counterparty": "Bank of America NA",
            "notional": 35000000,
            "currency": "USD",
            "cap_rate": 7.00,
            "floating_rate": "LIBOR-3M",
            "day_count": "ACT/360",
            "effective_date": "2007-06-07",
            "termination_date": "2012-06-25",
            "payment_frequency": "Quarterly",
        },
        "expect_exception": True,
        "expect_category": EXC.CPTY,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 9. Clean floor — F&G liability hedge
    # ------------------------------------------------------------------
    {
        "id": "CF-009",
        "desc": "Clean SOFR floor — $100M 3.00% quarterly F&G liability hedge",
        "confirmation": {
            "structure_type": "floor",
            "counterparty": "Goldman Sachs Bank USA",
            "notional": 100000000,
            "currency": "USD",
            "floor_rate": 3.00,
            "floating_rate": "SOFR",
            "day_count": "ACT/360",
            "effective_date": "2026-07-01",
            "termination_date": "2031-07-01",
            "payment_frequency": "Quarterly",
            "premium": 1250000,
        },
        "internal": {
            "structure_type": "floor",
            "counterparty": "Goldman Sachs Bank USA",
            "notional": 100000000,
            "currency": "USD",
            "floor_rate": 3.00,
            "floating_rate": "SOFR",
            "day_count": "ACT/360",
            "effective_date": "2026-07-01",
            "termination_date": "2031-07-01",
            "payment_frequency": "Quarterly",
            "premium": 1250000,
        },
        "expect_exception": False,
    },

    # ------------------------------------------------------------------
    # 10. Dual exception: cap rate wrong + wrong termination date
    # ------------------------------------------------------------------
    {
        "id": "CF-010",
        "desc": "Dual: cap rate 5.50% vs 5.00% + termination date 1yr off",
        "confirmation": {
            "structure_type": "cap",
            "counterparty": "JPMorgan Chase Bank NA",
            "notional": 100000000,
            "currency": "USD",
            "cap_rate": 5.50,
            "floating_rate": "SOFR",
            "day_count": "ACT/360",
            "effective_date": "2026-07-01",
            "termination_date": "2032-07-01",
            "payment_frequency": "Quarterly",
        },
        "internal": {
            "structure_type": "cap",
            "counterparty": "JPMorgan Chase Bank NA",
            "notional": 100000000,
            "currency": "USD",
            "cap_rate": 5.00,
            "floating_rate": "SOFR",
            "day_count": "ACT/360",
            "effective_date": "2026-07-01",
            "termination_date": "2031-07-01",
            "payment_frequency": "Quarterly",
        },
        "expect_exception": True,
        "expect_category": EXC.CAP_RATE,
        "expect_secondary": EXC.TERM_DATE,
        "expect_severity": "high",
    },
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
passed = failed = 0
SEP = "─" * 72

print(f"Cap/Floor Reconciler — {len(CASES)} cases\n{SEP}")

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
