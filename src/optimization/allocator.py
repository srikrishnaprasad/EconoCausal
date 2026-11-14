"""
allocator.py - budget optimizer with three selectable algorithms.
Pick ONE discount per customer to maximize expected incremental profit while the
expected discount spend stays within the budget.

For customer i and discount k (0, 10, 20):
    p_ik   = clip(p0_i + tau_ik, 0, 1)    chance of buying with discount k
    cost   = k * p_ik                     discount is only paid when they buy
    profit = tau_ik * V - cost            extra sales profit minus discount paid
"""
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, identity, kron, vstack

METHODS = {"lagrangian": "Lagrangian (fast, near-optimal)",
           "greedy": "Greedy by profit per dollar",
           "milp": "MILP (exact, slower)"}


def _m(v):
    return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"


def _tables(ite_df, levels, profit_per_sale):
    n = len(ite_df)
    ks = np.array([0.0, *levels])
    p0 = ite_df["p0"].to_numpy(float)
    tau = np.column_stack([np.zeros(n)] + [ite_df[f"ite_{k}"].to_numpy(float) for k in levels])
    p_k = np.clip(p0[:, None] + tau, 0.0, 1.0)
    tau = p_k - p0[:, None]
    cost = ks[None, :] * p_k
    profit = tau * profit_per_sale - cost
    return ks, tau, cost, profit


def _lagrangian(profit, cost, budget):
    rows = np.arange(len(profit))
    pick = lambda lmb: np.argmax(profit - lmb * cost, axis=1)
    spend = lambda c: cost[rows, c].sum()
    choice = pick(0.0)
    if spend(choice) > budget:
        lo, hi = 0.0, 1.0
        while spend(pick(hi)) > budget and hi < 1e12:
            hi *= 2.0
        for _ in range(50):
            mid = (lo + hi) / 2.0
            if spend(pick(mid)) > budget:
                lo = mid
            else:
                hi = mid
        choice = pick(hi)
    return choice


def _greedy(profit, cost, budget):
    rows = np.arange(len(profit))
    choice = np.argmax(profit, axis=1)
    choice = np.where(profit[rows, choice] > 0, choice, 0)
    bp, bc = profit[rows, choice], cost[rows, choice]
    order = np.argsort(-(bp / np.maximum(bc, 1e-9)))
    keep = np.zeros(len(choice), dtype=bool)
    keep[order[np.cumsum(bc[order]) <= budget]] = True
    return np.where(keep, choice, 0)


def _milp(profit, cost, budget, time_limit=15, max_candidates=15000):
    n, K = profit.shape
    best = profit[:, 1:].max(axis=1)
    cand = np.where(best > 0)[0]
    note = ""
    if len(cand) > max_candidates:
        cand = cand[np.argsort(-best[cand])[:max_candidates]]
        note = f"MILP was limited to the {max_candidates:,} most promising customers."
    choice = np.zeros(n, dtype=int)
    m = len(cand)
    if m == 0:
        return choice, note
    P, C = profit[cand], cost[cand]
    A = vstack([kron(identity(m), np.ones((1, K)), format="csr"), csr_matrix(C.reshape(1, -1))], format="csr")
    lb = np.r_[np.ones(m), -np.inf]
    ub = np.r_[np.ones(m), budget]
    res = milp(c=-P.reshape(-1), constraints=LinearConstraint(A, lb, ub),
               integrality=np.ones(m * K), bounds=Bounds(0, 1), options={"time_limit": time_limit})
    if res.x is None:
        return None, "MILP found no solution within the time limit."
    choice[cand] = res.x.reshape(m, K).argmax(axis=1)
    return choice, note


def recommend(ite_df, levels=(10, 20), profit_per_sale=100.0):
    """Best discount per customer ignoring the budget (0 when no discount pays off)."""
    ks, _, _, profit = _tables(ite_df, tuple(sorted(levels)), profit_per_sale)
    rows = np.arange(len(profit))
    best = profit.argmax(axis=1)
    best = np.where(profit[rows, best] > 0, best, 0)
    return ks[best].astype(int)


def allocate(ite_df, budget, levels=(10, 20), profit_per_sale=100.0, method="lagrangian"):
    levels = tuple(sorted(levels))
    n = len(ite_df)
    ks, tau, cost, profit = _tables(ite_df, levels, profit_per_sale)
    rows = np.arange(n)
    note, used = "", method
    t0 = time.perf_counter()
    if method == "greedy":
        choice = _greedy(profit, cost, budget)
    elif method == "milp":
        choice, note = _milp(profit, cost, budget)
        if choice is None:
            choice, used = _lagrangian(profit, cost, budget), "lagrangian"
            note += " Used the Lagrangian method instead."
    else:
        choice, used = _lagrangian(profit, cost, budget), "lagrangian"
    seconds = time.perf_counter() - t0

    keep = [c for c in ("user_id", "p0", "ite_10", "ite_20") if c in ite_df.columns]
    out = ite_df[keep].copy().reset_index(drop=True)
    out["discount"] = ks[choice].astype(int)
    out["expected_cost"] = cost[rows, choice]
    out["incremental_profit"] = profit[rows, choice]
    out["extra_purchases"] = tau[rows, choice]

    total_cost = float(cost[rows, choice].sum())
    total_profit = float(profit[rows, choice].sum())
    n_targeted = int((choice > 0).sum())
    baselines = {f"blanket_{int(k)}": {"cost": float(cost[:, j + 1].sum()),
                                       "profit": float(profit[:, j + 1].sum())}
                 for j, k in enumerate(levels)}
    key = "blanket_10" if "blanket_10" in baselines else next(iter(baselines))
    b, label = baselines[key], key.split("_")[1]
    diff = b["cost"] - total_cost
    spend_txt = (f"saves {_m(diff)} in discount spend" if diff >= 0
                 else f"spends {_m(-diff)} more on discounts")
    text = (f"Targeting {n_targeted:,} of {n:,} customers ({100 * n_targeted / max(n, 1):.0f}%) with expected "
            f"discount spend of {_m(total_cost)} (budget {_m(budget)}) is expected to add "
            f"{_m(total_profit)} in profit. A blanket ${label} discount for everyone would spend "
            f"{_m(b['cost'])} and return {_m(b['profit'])}, so this plan {spend_txt} and changes "
            f"profit by {_m(total_profit - b['profit'])}.")
    totals = {"n_customers": n, "n_targeted": n_targeted, "budget": float(budget),
              "profit_per_sale": float(profit_per_sale), "total_cost": total_cost,
              "incremental_profit": total_profit,
              "extra_purchases": float(tau[rows, choice].sum()),
              "roi": float(total_profit / total_cost * 100) if total_cost > 0 else None,
              "baselines": baselines, "seconds": float(seconds), "method": used,
              "method_label": METHODS[used], "note": note.strip(),
              "mix": {str(int(k)): int((choice == i).sum()) for i, k in enumerate(ks)},
              "summary_text": text}
    return out, totals