"""mendelcheck.optimize 的单元测试。

参考实现（ref_*）为独立的暴力枚举：对所有 3^n 完整基因型指派逐一检验
孟德尔一致性并计算总代价，绝不调用被测核心 mendelcheck 得到期望值。
"""

import itertools
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mendelcheck import OptimizeResult, ValidationError, check, optimize

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


def ref_optimize(pedigree, observations, costs):
    """暴力枚举全部指派，返回 (min_cost, optimal_sets, lex_witness) 或 None。

    lex_witness 为按 sorted(ID) 基因型序列字典序最小的最优指派。
    """
    ids = list(pedigree)
    best = None
    best_sols = []
    for combo in itertools.product((0, 1, 2), repeat=len(ids)):
        assignment = dict(zip(ids, combo))
        if any(assignment[pid] != g for pid, g in observations.items()):
            continue
        total = 0
        ok = True
        for pid in ids:
            c = costs.get(pid, (0, 0, 0))[assignment[pid]]
            if c is None:
                ok = False
                break
            total += c
        if not ok or not ref_consistent(pedigree, assignment):
            continue
        if best is None or total < best:
            best = total
            best_sols = [assignment]
        elif total == best:
            best_sols.append(assignment)
    if best is None:
        return None
    sets = {pid: frozenset(s[pid] for s in best_sols) for pid in ids}
    order = sorted(ids)
    witness = min(best_sols, key=lambda s: tuple(s[pid] for pid in order))
    return best, sets, witness


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


# ------------------------------------------------- 输入校验

class TestCostValidation(unittest.TestCase):
    def ped(self):
        return {"A": (None, None), "B": (None, None), "C": ("A", "B")}

    def test_costs_type_errors(self):
        ped = self.ped()
        with self.assertRaises(ValidationError):
            optimize(ped, {}, "not-a-dict")
        with self.assertRaises(ValidationError):
            optimize(ped, {}, {"Z": (0, 0, 0)})  # 未知个体
        with self.assertRaises(ValidationError):
            optimize(ped, {}, {"A": (0, 0)})  # 长度不为 3
        with self.assertRaises(ValidationError):
            optimize(ped, {}, {"A": 5})  # 非三元组
        with self.assertRaises(ValidationError):
            optimize(ped, {}, {"A": (True, 0, 0)})  # bool 拒绝
        with self.assertRaises(ValidationError):
            optimize(ped, {}, {"A": (0.5, 0, 0)})  # float 拒绝
        with self.assertRaises(ValidationError):
            optimize(ped, {}, {"A": (-1, 0, 0)})  # 负数拒绝

    def test_costs_none_and_empty_ok(self):
        ped = self.ped()
        self.assertEqual(optimize(ped, {}).status, "optimal")
        self.assertEqual(optimize(ped, {}, None).min_cost, 0)
        self.assertEqual(optimize(ped, {}, {}).min_cost, 0)

    def test_result_shape(self):
        res = optimize(self.ped(), {}, {"A": (1, 0, 2)})
        self.assertIsInstance(res, OptimizeResult)
        self.assertEqual(res.status, "optimal")
        self.assertEqual(set(res.optimal_genotypes), {"A", "B", "C"})
        self.assertEqual(set(res.witness), {"A", "B", "C"})
        self.assertGreaterEqual(res.nodes_used, 1)


# ------------------------------------------------- 基本优化语义

class TestOptimize(unittest.TestCase):
    def test_single_person_min_cost(self):
        res = optimize({"A": (None, None)}, {}, {"A": (3, 1, 2)})
        self.assertEqual(res.status, "optimal")
        self.assertEqual(res.min_cost, 1)
        self.assertEqual(res.optimal_genotypes["A"], frozenset({1}))
        self.assertEqual(res.witness, {"A": 1})
        self.assertEqual(res.best_cost_bound, 1)

    def test_both_parents_missing(self):
        # 两位父母信息均缺失：子女亲本槽位各自是独立未知创始者，
        # 创始者不计代价，子女按自身代价取最优。
        ped = {"C": (None, None)}
        res = optimize(ped, {}, {"C": (2, 0, 5)})
        self.assertEqual(res.min_cost, 0)
        self.assertEqual(res.witness, {"C": 1})
        # 父母均缺失的三人结构：缺失父母可传任意等位，不约束子女
        ped2 = {"M": (None, None), "C": (None, "M")}
        res2 = optimize(ped2, {"M": 0}, {"C": (4, 1, 9)})
        self.assertEqual(res2.status, "optimal")
        # 母=0 时子女只能为 0 或 1 -> 取代价较小的 1
        self.assertEqual(res2.optimal_genotypes["C"], frozenset({1}))
        self.assertEqual(res2.min_cost, 1)

    def test_tie_multiple_optima(self):
        # 同分多解：0 与 1 代价相同且皆可行
        res = optimize({"A": (None, None)}, {}, {"A": (0, 0, 5)})
        self.assertEqual(res.min_cost, 0)
        self.assertEqual(res.optimal_genotypes["A"], frozenset({0, 1}))
        # 确定性见证：字典序最小 -> 0
        self.assertEqual(res.witness, {"A": 0})
        # 重复调用结果确定
        res2 = optimize({"A": (None, None)}, {}, {"A": (0, 0, 5)})
        self.assertEqual(res, res2)

    def test_tie_witness_lexicographic_over_ids(self):
        # 两人同分：见证按 ID 字典序的基因型序列取最小
        ped = {"B": (None, None), "A": (None, None)}
        costs = {"A": (0, 0, 9), "B": (0, 0, 9)}
        res = optimize(ped, {}, costs)
        self.assertEqual(res.min_cost, 0)
        self.assertEqual(res.witness, {"A": 0, "B": 0})
        costs2 = {"A": (0, 0, 9), "B": (5, 0, 0)}
        res2 = optimize(ped, {}, costs2)
        # A 取 0；B 的最优为 {1, 2}，见证取字典序最小 -> 1
        self.assertEqual(res2.optimal_genotypes["B"], frozenset({1, 2}))
        self.assertEqual(res2.witness, {"A": 0, "B": 1})

    def test_all_candidates_forbidden(self):
        res = optimize({"A": (None, None)}, {}, {"A": (None, None, None)})
        self.assertEqual(res.status, "infeasible")
        self.assertIsNone(res.min_cost)
        self.assertIsNone(res.optimal_genotypes)
        self.assertIsNone(res.witness)

    def test_forbidden_genotype_respected(self):
        res = optimize({"A": (None, None)}, {}, {"A": (None, 4, None)})
        self.assertEqual(res.status, "optimal")
        self.assertEqual(res.witness, {"A": 1})
        self.assertEqual(res.min_cost, 4)

    def test_hard_observation_not_bypassed_by_high_cost(self):
        # 硬观测 A=0 且取 0 代价很高：仍必须取 0，不能用高代价偷偷放过
        res = optimize({"A": (None, None)}, {"A": 0}, {"A": (99, 0, 0)})
        self.assertEqual(res.status, "optimal")
        self.assertEqual(res.witness, {"A": 0})
        self.assertEqual(res.min_cost, 99)
        # 硬观测与禁止项冲突 -> 证明无解
        res2 = optimize({"A": (None, None)}, {"A": 0}, {"A": (None, 0, 0)})
        self.assertEqual(res2.status, "infeasible")

    def test_cost_change_reverses_optimum(self):
        # 同一约束（B=0 观测 + C=0 观测迫使 A ∈ {0, 1}），代价改变后最优反转
        ped = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
        obs = {"B": 0, "C": 0}
        res1 = optimize(ped, obs, {"A": (5, 1, 0)})
        self.assertEqual(res1.optimal_genotypes["A"], frozenset({1}))
        self.assertEqual(res1.min_cost, 1)
        res2 = optimize(ped, obs, {"A": (0, 7, 0)})
        self.assertEqual(res2.optimal_genotypes["A"], frozenset({0}))
        self.assertEqual(res2.min_cost, 0)
        self.assertNotEqual(res1.witness["A"], res2.witness["A"])

    def test_transmission_constraint_strict(self):
        # 传递约束严格：父母皆 0（观测）-> 子女不可能为 2，再高代价也不行
        ped = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
        obs = {"A": 0, "B": 0}
        costs = {"C": (9, 9, 0)}
        res = optimize(ped, obs, costs)
        self.assertEqual(res.status, "optimal")
        self.assertEqual(res.optimal_genotypes["C"], frozenset({0}))
        self.assertEqual(res.min_cost, 9)

    def test_synthetic_founders_cost_nothing(self):
        # 缺失父母为独立合成创始者，不计代价：只有实个体计费
        ped = {"C": (None, None)}
        res = optimize(ped, {}, {"C": (0, 0, 0)})
        self.assertEqual(res.min_cost, 0)
        self.assertEqual(res.optimal_genotypes["C"], frozenset({0, 1, 2}))


# ------------------------------------------------- 预算语义

class TestOptimizeBudget(unittest.TestCase):
    def test_budget_exhaustion_is_unknown(self):
        ped = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
        res = optimize(ped, {}, {"A": (2, 1, 0)}, node_budget=1)
        self.assertEqual(res.status, "unknown")
        self.assertIsNone(res.min_cost)
        self.assertIsNone(res.optimal_genotypes)
        self.assertIsNone(res.witness)

    def test_unknown_reports_feasible_upper_bound_only(self):
        # 预算只够找到可行解、不够证明最优：best_cost_bound 是可行上界，
        # 且绝不给出未经证明的 optimal_genotypes / witness。
        ped = {f"I{i}": (None, None) for i in range(6)}
        ped["C"] = ("I0", "I1")
        costs = {pid: (3, 2, 1) for pid in ped}
        full = optimize(ped, {}, costs)
        self.assertEqual(full.status, "optimal")
        found_unknown = False
        for budget in range(1, full.nodes_used + 2):
            res = optimize(ped, {}, costs, node_budget=budget)
            if res.status == "unknown":
                found_unknown = True
                self.assertIsNone(res.optimal_genotypes)
                self.assertIsNone(res.witness)
                if res.best_cost_bound is not None:
                    # 可行上界绝不小于真实最优
                    self.assertGreaterEqual(res.best_cost_bound, full.min_cost)
                if res.min_cost is not None:
                    # 已证明最优时才允许给 min_cost，且必须等于真实最优
                    self.assertEqual(res.min_cost, full.min_cost)
            else:
                self.assertEqual(res.status, "optimal")
                self.assertEqual(res.min_cost, full.min_cost)
        self.assertTrue(found_unknown)

    def test_proven_infeasible_distinct_from_unknown(self):
        ped = {"A": (None, None)}
        costs = {"A": (None, None, None)}
        self.assertEqual(optimize(ped, {}, costs).status, "infeasible")
        # 同一可行实例：预算充足 -> optimal；预算=1 -> unknown。两者严格分开。
        ped2 = {f"I{i}": (None, None) for i in range(4)}
        costs2 = {pid: (2, 1, 0) for pid in ped2}
        self.assertEqual(optimize(ped2, {}, costs2).status, "optimal")
        self.assertEqual(optimize(ped2, {}, costs2, node_budget=1).status, "unknown")


# ------------------------------------------------- 穷举参考对照（随机小实例）

class TestOptimizeAgainstBruteForce(unittest.TestCase):
    def test_random_small_instances(self):
        for seed in range(50):
            rng = random.Random(20260925 + seed)
            n = rng.randint(1, 7)
            ped = make_rng_pedigree(rng, n)
            obs = {pid: rng.randint(0, 2) for pid in ped if rng.random() < 0.4}
            costs = {}
            for pid in ped:
                if rng.random() < 0.7:
                    costs[pid] = tuple(
                        None if rng.random() < 0.15 else rng.randint(0, 5)
                        for _ in range(3)
                    )
            res = optimize(ped, obs, costs)
            self.assertNotEqual(res.status, "unknown", f"seed={seed}")
            ref = ref_optimize(ped, obs, costs)
            if ref is None:
                self.assertEqual(res.status, "infeasible", f"seed={seed}")
                continue
            want_cost, want_sets, want_witness = ref
            self.assertEqual(res.status, "optimal", f"seed={seed}")
            self.assertEqual(res.min_cost, want_cost, f"seed={seed}")
            self.assertEqual(res.optimal_genotypes, want_sets, f"seed={seed}")
            self.assertEqual(res.witness, want_witness, f"seed={seed}")
            # 见证独立校验：满足观测、孟德尔约束且代价等于最优
            self.assertTrue(ref_consistent(ped, res.witness), f"seed={seed}")
            for pid, g in obs.items():
                self.assertEqual(res.witness[pid], g, f"seed={seed}")
            total = sum(costs.get(p, (0, 0, 0))[g] for p, g in res.witness.items())
            self.assertEqual(total, want_cost, f"seed={seed}")

    def test_old_check_regression_still_works(self):
        # 旧硬观测接口不受影响
        ped = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
        res = check(ped, {"A": 2, "B": 2, "C": 0})
        self.assertEqual(res.status, "infeasible")
        self.assertEqual(res.conflict.removals, ("C",))


if __name__ == "__main__":
    unittest.main()
