"""mendelcheck 代价扩展（costs）的单元测试。

参考实现（ref_*）为独立暴力枚举：对所有 3^n 完整基因型指派逐一检验
孟德尔一致性、硬观测、代价禁止与总代价，绝不调用被测核心。
"""

import itertools
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mendelcheck import ValidationError, check

TRANSMIT = ((0,), (0, 1), (1,))


# ------------------------------------------------- 独立参考实现（暴力枚举）

def ref_consistent(pedigree, assignment):
    for cid, (fa, mo) in pedigree.items():
        if fa is None and mo is None:
            continue
        gc = assignment[cid]
        fa_alleles = TRANSMIT[assignment[fa]] if fa is not None else (0, 1)
        mo_alleles = TRANSMIT[assignment[mo]] if mo is not None else (0, 1)
        if not any(a + b == gc for a in fa_alleles for b in mo_alleles):
            return False
    return True


def ref_cost(costs, assignment):
    total = 0
    for pid, g in assignment.items():
        if pid in costs:
            c = costs[pid][g]
            if c is None:
                return None
            total += c
    return total


def ref_optimal(pedigree, observations, costs):
    """返回 (最优解列表, 最小代价)；无可行解返回 ([], None)。"""
    ids = list(pedigree)
    sols = []
    for combo in itertools.product((0, 1, 2), repeat=len(ids)):
        assignment = dict(zip(ids, combo))
        if any(assignment[p] != g for p, g in observations.items()):
            continue
        total = ref_cost(costs, assignment)
        if total is None or not ref_consistent(pedigree, assignment):
            continue
        sols.append((assignment, total))
    if not sols:
        return [], None
    optimum = min(t for _, t in sols)
    best = [a for a, t in sols if t == optimum]
    return best, optimum


def ref_optimal_sets(best_sols):
    ids = sorted({pid for a in best_sols for pid in a})
    return {pid: frozenset(a[pid] for a in best_sols) for pid in ids}


def ref_lex_witness(best_sols):
    ids = sorted({pid for a in best_sols for pid in a})
    return min(best_sols, key=lambda a: tuple(a[p] for p in ids))


def ref_min_corrections(pedigree, observations, costs):
    obs_ids = sorted(observations)
    for k in range(len(obs_ids) + 1):
        found = []
        for combo in itertools.combinations(obs_ids, k):
            kept = {p: g for p, g in observations.items() if p not in combo}
            best, _ = ref_optimal(pedigree, kept, costs)
            if best:
                found.append(tuple(combo))
        if found:
            return sorted(found)
    raise AssertionError("unreachable")


def make_rng_pedigree(rng, n):
    ids = [f"I{i}" for i in range(n)]
    ped = {ids[0]: (None, None)}
    for i in range(1, n):
        earlier = ids[:i]

        def pick():
            return rng.choice(earlier) if rng.random() < 0.6 else None

        fa, mo = pick(), pick()
        if fa == mo:
            mo = None
        ped[ids[i]] = (fa, mo)
    return ped


# ------------------------------------------------- 校验

class TestCostValidation(unittest.TestCase):
    PED = {"A": (None, None), "B": (None, None), "C": ("A", "B")}

    def test_bad_cost_shapes(self):
        for bad in ([1, 2], (1, 2, 3, 4), "x", 7):
            with self.assertRaises(ValidationError, msg=f"{bad!r}"):
                check(self.PED, {}, costs={"A": bad})
        with self.assertRaises(ValidationError):
            check(self.PED, {}, costs="not-a-dict")
        with self.assertRaises(ValidationError):
            check(self.PED, {}, costs={"Z": [0, 0, 0]})

    def test_bad_cost_values(self):
        for bad in (-1, True, 1.0, float("nan"), float("inf"), "0"):
            with self.assertRaises(ValidationError, msg=f"{bad!r}"):
                check(self.PED, {}, costs={"A": [bad, 0, 0]})

    def test_none_means_forbidden_accepted(self):
        res = check(self.PED, {}, costs={"A": [None, None, 0]})
        self.assertEqual(res.status, "optimal")
        self.assertEqual(res.optimal_genotypes["A"], frozenset({2}))


# ------------------------------------------------- 最优代价

class TestOptimal(unittest.TestCase):
    def test_basic_optimal_and_reversal(self):
        ped = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
        res = check(ped, {}, costs={"C": [0, 10, 10]})
        self.assertEqual(res.status, "optimal")
        self.assertEqual(res.optimal_cost, 0)
        self.assertEqual(res.optimal_genotypes["C"], frozenset({0}))
        self.assertEqual(res.witness["C"], 0)
        res2 = check(ped, {}, costs={"C": [10, 10, 0]})
        self.assertEqual(res2.optimal_cost, 0)
        self.assertEqual(res2.optimal_genotypes["C"], frozenset({2}))
        self.assertEqual(res2.witness["C"], 2)

    def test_tie_multiple_solutions_lex_witness(self):
        ped = {"A": (None, None), "B": (None, None)}
        costs = {"A": [0, 0, 5], "B": [0, 0, 5]}
        res = check(ped, {}, costs=costs)
        self.assertEqual(res.status, "optimal")
        self.assertEqual(res.optimal_cost, 0)
        self.assertEqual(res.optimal_genotypes["A"], frozenset({0, 1}))
        self.assertEqual(res.optimal_genotypes["B"], frozenset({0, 1}))
        self.assertEqual(res.witness, {"A": 0, "B": 0})

    def test_tie_with_pedigree_constraint(self):
        ped = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
        costs = {"C": [10, 10, 0], "A": [5, 5, 0], "B": [5, 5, 0]}
        res = check(ped, {}, costs=costs)
        self.assertEqual(res.status, "optimal")
        self.assertEqual(res.optimal_cost, 0)
        self.assertEqual(res.witness, {"A": 2, "B": 2, "C": 2})

    def test_both_parents_missing(self):
        ped = {"C": (None, None), "D": (None, None), "E": ("C", "D")}
        costs = {"E": [7, 0, 7]}
        res = check(ped, {}, costs=costs)
        self.assertEqual(res.status, "optimal")
        self.assertEqual(res.optimal_cost, 0)
        self.assertEqual(res.optimal_genotypes["E"], frozenset({1}))
        self.assertIn(res.witness["C"], (0, 1))
        self.assertIn(res.witness["D"], (1, 2))

    def test_zero_costs_equivalent_to_open(self):
        ped = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
        res = check(ped, {}, costs={})
        self.assertEqual(res.status, "optimal")
        self.assertEqual(res.optimal_cost, 0)
        for pid in ped:
            self.assertEqual(res.optimal_genotypes[pid], frozenset({0, 1, 2}))


# ------------------------------------------------- 禁止与硬观测交互

class TestForbidden(unittest.TestCase):
    def test_all_candidates_forbidden_is_infeasible(self):
        ped = {"A": (None, None)}
        res = check(ped, {}, costs={"A": [None, None, None]})
        self.assertEqual(res.status, "infeasible")
        self.assertIsNone(res.conflict)
        self.assertIn("structurally infeasible", res.infeasible_reason)

    def test_all_forbidden_independent_of_budget(self):
        ped = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
        costs = {"B": [None, None, None]}
        res = check(ped, {}, costs=costs, node_budget=1)
        self.assertEqual(res.status, "infeasible")

    def test_observed_value_forbidden_becomes_conflict(self):
        ped = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
        obs = {"C": 0}
        costs = {"C": [None, 5, 5]}
        res = check(ped, obs, costs=costs)
        self.assertEqual(res.status, "infeasible")
        self.assertEqual(res.conflict.removals, ("C",))
        witness = res.conflict.witnesses["C"]
        self.assertIn(witness["C"], (1, 2))

    def test_high_cost_does_not_relax_hard_observation(self):
        ped = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
        costs = {"C": [0, 100, 0]}
        res = check(ped, {"C": 1}, costs=costs)
        self.assertEqual(res.status, "optimal")
        self.assertEqual(res.optimal_cost, 100)
        self.assertEqual(res.witness["C"], 1)


# ------------------------------------------------- 预算语义

class TestCostBudget(unittest.TestCase):
    def test_small_budget_unknown_never_claims_optimum(self):
        ped = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
        costs = {"C": [3, 0, 3]}
        saw_upper_bound = False
        saw_plain_unknown = False
        for budget in range(1, 40):
            res = check(ped, {}, costs=costs, node_budget=budget)
            self.assertIn(res.status, ("optimal", "unknown"))
            if res.status == "unknown":
                self.assertIsNone(res.optimal_genotypes)
                self.assertIsNone(res.optimal_cost)
                if res.witness is not None:
                    saw_upper_bound = True
                    self.assertGreaterEqual(res.witness_cost, 0)
                    self.assertEqual(res.nodes_used, budget)
                else:
                    saw_plain_unknown = True
        self.assertTrue(saw_upper_bound and saw_plain_unknown)

    def test_full_budget_proves_optimum(self):
        ped = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
        costs = {"C": [3, 0, 3]}
        res = check(ped, {}, costs=costs)
        self.assertEqual(res.status, "optimal")
        self.assertEqual(res.optimal_cost, 0)


# ------------------------------------------------- 穷举参考对照（随机小实例）

class TestCostsAgainstBruteForce(unittest.TestCase):
    def test_random_small_instances_with_costs(self):
        n_feasible = n_conflict = 0
        for seed in range(80):
            rng = random.Random(4530000 + seed)
            n = rng.randint(2, 7)
            ped = make_rng_pedigree(rng, n)
            costs = {}
            for pid in ped:
                triple = [
                    None if rng.random() < 0.12 else rng.randint(0, 4)
                    for _ in range(3)
                ]
                if rng.random() < 0.7:
                    costs[pid] = triple
            obs = {
                pid: rng.randint(0, 2)
                for pid in ped
                if rng.random() < 0.5
            }
            best, optimum = ref_optimal(ped, obs, costs)
            res = check(ped, obs, costs=costs)
            self.assertNotEqual(res.status, "unknown", f"seed={seed}")
            if best:
                n_feasible += 1
                self.assertEqual(res.status, "optimal", f"seed={seed}")
                self.assertEqual(res.optimal_cost, optimum, f"seed={seed}")
                self.assertEqual(
                    res.optimal_genotypes, ref_optimal_sets(best), f"seed={seed}"
                )
                self.assertEqual(res.witness, ref_lex_witness(best), f"seed={seed}")
                self.assertEqual(res.feasible_genotypes, res.optimal_genotypes)
            else:
                n_conflict += 1
                self.assertEqual(res.status, "infeasible", f"seed={seed}")
                struct = _structurally_infeasible(ped, costs)
                if struct:
                    self.assertIsNone(res.conflict, f"seed={seed}")
                    self.assertIsNotNone(res.infeasible_reason, f"seed={seed}")
                else:
                    got = sorted([res.conflict.removals, *res.conflict.alternatives])
                    want = ref_min_corrections(ped, obs, costs)
                    self.assertEqual(got, want, f"seed={seed}")
                    covered = set().union(*map(set, want))
                    self.assertEqual(
                        set(res.conflict.witnesses), covered, f"seed={seed}"
                    )
                    for obs_id, witness in res.conflict.witnesses.items():
                        owner = next(s for s in want if obs_id in s)
                        self.assertTrue(
                            ref_consistent(ped, witness), f"seed={seed}"
                        )
                        kept = {p: g for p, g in obs.items() if p not in owner}
                        self.assertEqual(
                            ref_cost(costs, witness),
                            ref_optimal(ped, kept, costs)[1],
                            f"seed={seed} witness not min-cost",
                        )
                        for pid, g in kept.items():
                            self.assertEqual(witness[pid], g, f"seed={seed}")
        self.assertGreater(n_feasible, 10)
        self.assertGreater(n_conflict, 3)


def _structurally_infeasible(pedigree, costs):
    best, _ = ref_optimal(pedigree, {}, costs)
    return not best


if __name__ == "__main__":
    unittest.main()
