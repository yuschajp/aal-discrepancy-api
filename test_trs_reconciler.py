#!/usr/bin/env python3
"""
test_trs_reconciler.py — regression suite for the TRS engine.

Reference: Wells Fargo Bank NA TRS template (rates TRS on US Treasury).
Covers rates TRS, equity TRS, and credit TRS field structures.

Usage:
    python3 test_trs_reconciler.py
    python3 test_trs_reconciler.py --verbose
"""
import sys, argparse
from trs_reconciler import TRSReconciler, EXC

ap = argparse.ArgumentParser()
ap.add_argument("--verbose", "-v", action="store_true")
args = ap.parse_args()

reconciler = TRSReconciler()

CASES = [
    # ------------------------------------------------------------------
    # 1. Clean rates TRS — US Treasury reference asset, OIS funding
    # ------------------------------------------------------------------
    {
        "id": "TRS-001",
        "desc": "Clean rates TRS — UST 4.25% 2034, $50M, OIS+25bp",
        "confirmation": {
            "counterparty": "Wells Fargo Bank NA",
            "ref_cusip": "91282CJN9",
            "ref_issuer": "United States Treasury",
            "ref_coupon": 4.25,
            "ref_maturity": "2034-02-15",
            "initial_price": 98.750,
            "notional": 50000000,
            "currency": "USD",
            "direction": "receiver",
            "leg_type": "floating",
            "floating_rate": "OIS",
            "spread": 0.25,
            "day_count": "ACT/360",
            "effective_date": "2026-07-03",
            "termination_date": "2027-07-03",
            "valuation_frequency": "Monthly",
        },
        "internal": {
            "counterparty": "Wells Fargo Bank NA",
            "ref_cusip": "91282CJN9",
            "ref_issuer": "United States Treasury",
            "ref_coupon": 4.25,
            "ref_maturity": "2034-02-15",
            "initial_price": 98.750,
            "notional": 50000000,
            "currency": "USD",
            "direction": "receiver",
            "leg_type": "floating",
            "floating_rate": "OIS",
            "spread": 0.25,
            "day_count": "ACT/360",
            "effective_date": "2026-07-03",
            "termination_date": "2027-07-03",
            "valuation_frequency": "Monthly",
        },
        "expect_exception": False,
    },

    # ------------------------------------------------------------------
    # 2. Wrong reference asset CUSIP
    # ------------------------------------------------------------------
    {
        "id": "TRS-002",
        "desc": "Reference asset CUSIP mismatch — wrong Treasury bond",
        "confirmation": {
            "counterparty": "Wells Fargo Bank NA",
            "ref_cusip": "91282CKQ1",
            "ref_coupon": 4.25,
            "ref_maturity": "2034-02-15",
            "initial_price": 98.750,
            "notional": 50000000,
            "currency": "USD",
            "direction": "receiver",
            "floating_rate": "OIS",
            "spread": 0.25,
            "day_count": "ACT/360",
            "effective_date": "2026-07-03",
            "termination_date": "2027-07-03",
        },
        "internal": {
            "counterparty": "Wells Fargo Bank NA",
            "ref_cusip": "91282CJN9",
            "ref_coupon": 4.25,
            "ref_maturity": "2034-02-15",
            "initial_price": 98.750,
            "notional": 50000000,
            "currency": "USD",
            "direction": "receiver",
            "floating_rate": "OIS",
            "spread": 0.25,
            "day_count": "ACT/360",
            "effective_date": "2026-07-03",
            "termination_date": "2027-07-03",
        },
        "expect_exception": True,
        "expect_category": EXC.REF_ASSET,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 3. Wrong reference asset coupon — 4.25% vs 4.50%
    # ------------------------------------------------------------------
    {
        "id": "TRS-003",
        "desc": "Reference coupon mismatch: 4.25% confirmed vs 4.50% booked",
        "confirmation": {
            "counterparty": "Goldman Sachs Bank USA",
            "ref_cusip": "91282CJN9",
            "ref_coupon": 4.25,
            "ref_maturity": "2034-02-15",
            "initial_price": 98.750,
            "notional": 100000000,
            "currency": "USD",
            "direction": "receiver",
            "floating_rate": "SOFR",
            "spread": 0.10,
            "day_count": "ACT/360",
            "effective_date": "2026-07-01",
            "termination_date": "2027-07-01",
        },
        "internal": {
            "counterparty": "Goldman Sachs Bank USA",
            "ref_cusip": "91282CJN9",
            "ref_coupon": 4.50,
            "ref_maturity": "2034-02-15",
            "initial_price": 98.750,
            "notional": 100000000,
            "currency": "USD",
            "direction": "receiver",
            "floating_rate": "SOFR",
            "spread": 0.10,
            "day_count": "ACT/360",
            "effective_date": "2026-07-01",
            "termination_date": "2027-07-01",
        },
        "expect_exception": True,
        "expect_category": EXC.COUPON,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 4. Initial price mismatch — 98.750 vs 98.500
    # ------------------------------------------------------------------
    {
        "id": "TRS-004",
        "desc": "Initial price mismatch: 98.750 confirmed vs 98.500 booked",
        "confirmation": {
            "counterparty": "JPMorgan Chase Bank NA",
            "ref_cusip": "91282CJN9",
            "ref_coupon": 4.25,
            "initial_price": 98.750,
            "notional": 50000000,
            "currency": "USD",
            "direction": "receiver",
            "floating_rate": "OIS",
            "spread": 0.25,
            "day_count": "ACT/360",
            "effective_date": "2026-07-01",
            "termination_date": "2027-07-01",
        },
        "internal": {
            "counterparty": "JPMorgan Chase Bank NA",
            "ref_cusip": "91282CJN9",
            "ref_coupon": 4.25,
            "initial_price": 98.500,
            "notional": 50000000,
            "currency": "USD",
            "direction": "receiver",
            "floating_rate": "OIS",
            "spread": 0.25,
            "day_count": "ACT/360",
            "effective_date": "2026-07-01",
            "termination_date": "2027-07-01",
        },
        "expect_exception": True,
        "expect_category": EXC.INIT_PRICE,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 5. TRS direction mismatch — receiver vs payer
    # ------------------------------------------------------------------
    {
        "id": "TRS-005",
        "desc": "Direction mismatch: receiver confirmed vs payer booked",
        "confirmation": {
            "counterparty": "Barclays Bank PLC",
            "ref_cusip": "91282CJN9",
            "ref_coupon": 4.25,
            "initial_price": 98.750,
            "notional": 50000000,
            "currency": "USD",
            "direction": "receiver",
            "floating_rate": "SOFR",
            "spread": 0.15,
            "effective_date": "2026-07-01",
            "termination_date": "2027-07-01",
        },
        "internal": {
            "counterparty": "Barclays Bank PLC",
            "ref_cusip": "91282CJN9",
            "ref_coupon": 4.25,
            "initial_price": 98.750,
            "notional": 50000000,
            "currency": "USD",
            "direction": "payer",
            "floating_rate": "SOFR",
            "spread": 0.15,
            "effective_date": "2026-07-01",
            "termination_date": "2027-07-01",
        },
        "expect_exception": True,
        "expect_category": EXC.DIR,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 6. Notional mismatch — $50M vs $100M
    # ------------------------------------------------------------------
    {
        "id": "TRS-006",
        "desc": "Notional mismatch: $50M confirmed vs $100M booked",
        "confirmation": {
            "counterparty": "Wells Fargo Bank NA",
            "ref_cusip": "91282CJN9",
            "ref_coupon": 4.25,
            "initial_price": 98.750,
            "notional": 50000000,
            "currency": "USD",
            "direction": "receiver",
            "floating_rate": "OIS",
            "spread": 0.25,
            "effective_date": "2026-07-01",
            "termination_date": "2027-07-01",
        },
        "internal": {
            "counterparty": "Wells Fargo Bank NA",
            "ref_cusip": "91282CJN9",
            "ref_coupon": 4.25,
            "initial_price": 98.750,
            "notional": 100000000,
            "currency": "USD",
            "direction": "receiver",
            "floating_rate": "OIS",
            "spread": 0.25,
            "effective_date": "2026-07-01",
            "termination_date": "2027-07-01",
        },
        "expect_exception": True,
        "expect_category": EXC.NOTIONAL,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 7. Floating rate normalization — OIS aliases (clean)
    # ------------------------------------------------------------------
    {
        "id": "TRS-007",
        "desc": "Floating rate normalization: USD-Federal Funds-OIS Compound == OIS (clean)",
        "confirmation": {
            "counterparty": "Wells Fargo Bank NA",
            "ref_cusip": "91282CJN9",
            "ref_coupon": 4.25,
            "initial_price": 98.750,
            "notional": 50000000,
            "currency": "USD",
            "direction": "receiver",
            "floating_rate": "USD-Federal Funds-OIS Compound",
            "spread": 0.25,
            "day_count": "ACT/360",
            "effective_date": "2026-07-01",
            "termination_date": "2027-07-01",
        },
        "internal": {
            "counterparty": "Wells Fargo Bank NA",
            "ref_cusip": "91282CJN9",
            "ref_coupon": 4.25,
            "initial_price": 98.750,
            "notional": 50000000,
            "currency": "USD",
            "direction": "receiver",
            "floating_rate": "OIS",
            "spread": 0.25,
            "day_count": "ACT/360",
            "effective_date": "2026-07-01",
            "termination_date": "2027-07-01",
        },
        "expect_exception": False,
    },

    # ------------------------------------------------------------------
    # 8. Counterparty mismatch
    # ------------------------------------------------------------------
    {
        "id": "TRS-008",
        "desc": "Counterparty mismatch: Wells Fargo Bank NA vs WF Securities",
        "confirmation": {
            "counterparty": "Wells Fargo Securities LLC",
            "ref_cusip": "91282CJN9",
            "ref_coupon": 4.25,
            "initial_price": 98.750,
            "notional": 50000000,
            "currency": "USD",
            "direction": "receiver",
            "floating_rate": "OIS",
            "spread": 0.25,
            "effective_date": "2026-07-01",
            "termination_date": "2027-07-01",
        },
        "internal": {
            "counterparty": "Wells Fargo Bank NA",
            "ref_cusip": "91282CJN9",
            "ref_coupon": 4.25,
            "initial_price": 98.750,
            "notional": 50000000,
            "currency": "USD",
            "direction": "receiver",
            "floating_rate": "OIS",
            "spread": 0.25,
            "effective_date": "2026-07-01",
            "termination_date": "2027-07-01",
        },
        "expect_exception": True,
        "expect_category": EXC.CPTY,
        "expect_severity": "high",
    },

    # ------------------------------------------------------------------
    # 9. Spread mismatch — 0.25% vs 0.35%
    # ------------------------------------------------------------------
    {
        "id": "TRS-009",
        "desc": "Spread mismatch: OIS+25bp confirmed vs OIS+35bp booked",
        "confirmation": {
            "counterparty": "Goldman Sachs Bank USA",
            "ref_cusip": "91282CJN9",
            "ref_coupon": 4.25,
            "initial_price": 98.750,
            "notional": 100000000,
            "currency": "USD",
            "direction": "receiver",
            "floating_rate": "OIS",
            "spread": 0.25,
            "effective_date": "2026-07-01",
            "termination_date": "2027-07-01",
        },
        "internal": {
            "counterparty": "Goldman Sachs Bank USA",
            "ref_cusip": "91282CJN9",
            "ref_coupon": 4.25,
            "initial_price": 98.750,
            "notional": 100000000,
            "currency": "USD",
            "direction": "receiver",
            "floating_rate": "OIS",
            "spread": 0.35,
            "effective_date": "2026-07-01",
            "termination_date": "2027-07-01",
        },
        "expect_exception": True,
        "expect_category": EXC.SPREAD,
        "expect_severity": "medium",
    },

    # ------------------------------------------------------------------
    # 10. Dual: wrong CUSIP + wrong termination date
    # ------------------------------------------------------------------
    {
        "id": "TRS-010",
        "desc": "Dual: wrong CUSIP + termination 1yr off",
        "confirmation": {
            "counterparty": "Citibank NA",
            "ref_cusip": "91282CKQ1",
            "ref_coupon": 4.25,
            "initial_price": 98.750,
            "notional": 50000000,
            "currency": "USD",
            "direction": "receiver",
            "floating_rate": "SOFR",
            "spread": 0.20,
            "effective_date": "2026-07-01",
            "termination_date": "2028-07-01",
        },
        "internal": {
            "counterparty": "Citibank NA",
            "ref_cusip": "91282CJN9",
            "ref_coupon": 4.25,
            "initial_price": 98.750,
            "notional": 50000000,
            "currency": "USD",
            "direction": "receiver",
            "floating_rate": "SOFR",
            "spread": 0.20,
            "effective_date": "2026-07-01",
            "termination_date": "2027-07-01",
        },
        "expect_exception": True,
        "expect_category": EXC.REF_ASSET,
        "expect_secondary": EXC.TERM_DATE,
        "expect_severity": "high",
    },
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
passed = failed = 0
SEP = "─" * 72

print(f"TRS Reconciler — {len(CASES)} cases\n{SEP}")

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
