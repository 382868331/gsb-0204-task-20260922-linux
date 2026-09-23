"""mendelcheck 的单元测试。

参考实现（ref_*）为独立的暴力枚举：对所有 3^n 完整基因型指派逐一检验
孟德尔一致性，绝不调用被测核心 mendelcheck 得到期望值。
"""

import itertools
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mendelcheck
from mendelcheck import CheckResult, ValidationError, check

TRANSMIT = ((0,), (0, 1), (1,))


# ------------------------------------------------- 独立参考实现（暴力枚举）

def ref_consistent(pedigree, assignment):
    """完整指派是否满足全部孟德尔约束（缺失父母=独立未知创始者）。

    缺失父母可传任意单等位 {0,1}，且对每个子女独立——这正好等价于
    “每个缺失亲本槽位一个不受约束的创始者”。
    """
    for cid, (fa, mo) in pedigree.items():
        if fa is None and mo is None:
            continue
        gc = assignment[cid]
        fa_alleles = TRANSMIT[assignment[fa]] if fa is not None else (0, 1)
        mo_alleles = TRANSMIT[assignment[mo]] if mo is not None else (0, 1)
        if not any(a + b == gc for a in fa_alleles for b in mo_alleles):
            return False
    return True


def ref_solutions(pedigree, observations):
    """枚举全部满足观测与孟德尔约束的完整指派。"""
    ids = list(pedigree)
    sols = []
    for combo in itertools.product((0, 1, 2), repeat=len(ids)):
        assignment = dict(zip(ids, combo))
        if any(assignment[pid] != g for pid, g in observations.items()):
            continue
        if ref_consistent(pedigree, assignment):
            sols.append(assignment)
    return sols


def ref_feasible_sets(pedigree, observations):
    sols = ref_solutions(pedigree, observations)
    if not sols:
        return None
    return {pid: frozenset(s[pid] for s in sols) for pid in pedigree}


def ref_min_corrections(pedigree, observations):
    """按基数递增枚举全部最小移除集，返回升序元组的字典序列表。"""
    obs_ids = sorted(observations)
    for k in range(len(obs_ids) + 1):
        found = []
        for combo in itertools.combinations(obs_ids, k):
            kept = {p: g for p, g in observations.items() if p not in combo}
            if ref_solutions(pedigree, kept):
                found.append(tuple(combo))
        if found:
            return sorted(found)
    raise AssertionError("unreachable: removing all observations is always feasible")


def make_rng_pedigree(rng, n):
    """确定性生成一个 n 人无环家系（父母取自更早的个体）。"""
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


# ------------------------------------------------- 校验与接口语义

class TestValidation(unittest.TestCase):
    def ped(self):
        return {"A": (None, None), "B": (None, None), "C": ("A", "B")}

    def test_genre_type_errors(self):
        with self.assertRaises(ValidationError):
            check("not-a-dict", {})
        with self.assertRaises(ValidationError):
            check({}, {})  # 空家系
        with self.assertRaises(ValidationError):
            check({"A": (None, None)}, "not-a-dict")
        with self.assertRaises(ValidationError):
            check({"A": None}, {})  # parents 非二元组
        with self.assertRaises(ValidationError):
            check({"A": (None, None, None)}, {})  # 长度不为 2
        with self.assertRaises(ValidationError):
            check({"": (None, None)}, {})  # 空 ID
        with self.assertRaises(ValidationError):
            check({1: (None, None)}, {})  # 非 str ID

    def test_parent_reference_errors(self):
        with self.assertRaises(ValidationError) as cm:
            check({"A": ("Z", None)}, {})
        self.assertIn("Z", str(cm.exception))
        with self.assertRaises(ValidationError):
            check({"A": ("A", None)}, {})  # 自环
        with self.assertRaises(ValidationError):
            check({"A": (1, None)}, {})  # 非 str 父母

    def test_cycle_detected_and_located(self):
        ped = {"A": ("B", None), "B": ("C", None), "C": ("A", None)}
        with self.assertRaises(ValidationError) as cm:
            check(ped, {})
        self.assertIn("cycle", str(cm.exception))

    def test_individual_limit(self):
        ped = {"I0": (None, None)}
        for i in range(1, mendelcheck.MAX_INDIVIDUALS + 1):
            ped[f"I{i}"] = (f"I{i-1}", None)
        with self.assertRaises(ValidationError) as cm:
            check(ped, {})
        self.assertIn("21", str(cm.exception))
        ped_ok = {k: v for k, v in ped.items() if k != "I20"}
        self.assertEqual(check(ped_ok, {}).status, "feasible")

    def test_observation_errors(self):
        ped = self.ped()
        with self.assertRaises(ValidationError):
            check(ped, {"Z": 1})  # 未知个体
        with self.assertRaises(ValidationError):
            check(ped, {"A": 3})  # 越界
        with self.assertRaises(ValidationError):
            check(ped, {"A": -1})
        with self.assertRaises(ValidationError):
            check(ped, {"A": True})  # bool 拒绝
        with self.assertRaises(ValidationError):
            check(ped, {"A": 1.0})  # float 拒绝
        with self.assertRaises(ValidationError):
            check(ped, {"A": float("nan")})  # NaN 拒绝
        with self.assertRaises(ValidationError):
            check(ped, {"A": float("inf")})  # Infinity 拒绝

    def test_budget_errors(self):
        ped = self.ped()
        for bad in (True, 0, -3, 2.5, float("nan"), "100"):
            with self.assertRaises(ValidationError, msg=f"node_budget={bad!r}"):
                check(ped, {}, node_budget=bad)

    def test_result_shape(self):
        res = check(self.ped(), {"A": 0, "B": 2})
        self.assertIsInstance(res, CheckResult)
        self.assertEqual(res.status, "feasible")
        self.assertIsNone(res.conflict)
        self.assertEqual(set(res.feasible_genotypes), {"A", "B", "C"})
        self.assertGreaterEqual(res.nodes_used, 1)


# ------------------------------------------------- 可行基因型集合

class TestFeasibleSets(unittest.TestCase):
    def test_trio_homozygous_parents(self):
        ped = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
        res = check(ped, {"A": 0, "B": 2})
        self.assertEqual(res.status, "feasible")
        self.assertEqual(res.feasible_genotypes["A"], frozenset({0}))
        self.assertEqual(res.feasible_genotypes["B"], frozenset({2}))
        self.assertEqual(res.feasible_genotypes["C"], frozenset({1}))

    def test_trio_heterozygous_parents(self):
        ped = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
        res = check(ped, {"A": 1, "B": 1})
        self.assertEqual(res.feasible_genotypes["C"], frozenset({0, 1, 2}))

    def test_no_observations_all_open(self):
        ped = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
        res = check(ped, {})
        self.assertEqual(res.status, "feasible")
        for pid in ped:
            self.assertEqual(res.feasible_genotypes[pid], frozenset({0, 1, 2}))

    def test_single_parent_missing(self):
        # 父缺失（独立未知创始者）：母=0 时子女只能为 0 或 1
        ped = {"M": (None, None), "C": (None, "M")}
        res = check(ped, {"M": 0})
        self.assertEqual(res.feasible_genotypes["C"], frozenset({0, 1}))
        # 两个同母异父（父均缺失）的半同胞：缺失父各自独立
        ped2 = {"M": (None, None), "C1": (None, "M"), "C2": (None, "M")}
        res2 = check(ped2, {"M": 1, "C1": 0, "C2": 2})
        self.assertEqual(res2.status, "feasible")

    def test_multigeneration_propagation(self):
        # 祖父母 0x0 迫使父母为 0，进而限制孙代
        ped = {
            "G1": (None, None), "G2": (None, None),
            "P": ("G1", "G2"), "D": (None, None),
            "E": ("P", "D"),
        }
        res = check(ped, {"G1": 0, "G2": 0})
        self.assertEqual(res.feasible_genotypes["P"], frozenset({0}))
        self.assertEqual(res.feasible_genotypes["E"], frozenset({0, 1}))


# ------------------------------------------------- 冲突定位

class TestConflicts(unittest.TestCase):
    def assert_witnesses_valid(self, pedigree, observations, conflict):
        """独立校验：每个见证满足孟德尔约束与其对应移除集之外的观测。"""
        all_sets = [conflict.removals, *conflict.alternatives]
        for obs_id, witness in conflict.witnesses.items():
            owner = next(s for s in all_sets if obs_id in s)
            self.assertEqual(set(witness), set(pedigree))
            self.assertTrue(all(w in (0, 1, 2) for w in witness.values()))
            self.assertTrue(ref_consistent(pedigree, witness))
            for pid, g in observations.items():
                if pid not in owner:
                    self.assertEqual(witness[pid], g)

    def test_simple_trio_conflict(self):
        # 父母皆 2 -> 子女必为 2，与观测 0 冲突；最小解释为 {C}
        ped = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
        res = check(ped, {"A": 2, "B": 2, "C": 0})
        self.assertEqual(res.status, "infeasible")
        self.assertIsNone(res.feasible_genotypes)
        self.assertEqual(res.conflict.removals, ("C",))
        self.assertEqual(res.conflict.alternatives, ())
        self.assertFalse(res.conflict.truncated)
        self.assert_witnesses_valid(ped, {"A": 2, "B": 2, "C": 0}, res.conflict)
        # 见证中 C 被重新赋值为唯一可行值 2
        self.assertEqual(res.conflict.witnesses["C"]["C"], 2)

    def test_multigeneration_implicit_conflict(self):
        # G1=G2=0 -> P=0；D=2；E=2 需要 P 传 1：跨两代才显现的隐含冲突。
        # 最小移除集（大小 1）：{E}、{G1}、{G2} —— 多个最小解释。
        ped = {
            "G1": (None, None), "G2": (None, None),
            "P": ("G1", "G2"), "D": (None, None),
            "E": ("P", "D"),
        }
        obs = {"G1": 0, "G2": 0, "D": 2, "E": 2}
        res = check(ped, obs)
        self.assertEqual(res.status, "infeasible")
        self.assertEqual(res.conflict.removals, ("E",))  # 字典序最小
        self.assertEqual(sorted(res.conflict.alternatives), [("G1",), ("G2",)])
        self.assertEqual(set(res.conflict.witnesses), {"E", "G1", "G2"})
        self.assert_witnesses_valid(ped, obs, res.conflict)

    def test_inbred_pedigree(self):
        # 同胞交配（近亲家系，图仍无环）：A=B=2 -> C=D=2 -> E 必为 2。
        ped = {
            "A": (None, None), "B": (None, None),
            "C": ("A", "B"), "D": ("A", "B"),
            "E": ("C", "D"),
        }
        obs = {"A": 2, "B": 2, "E": 0}
        res = check(ped, obs)
        self.assertEqual(res.status, "infeasible")
        self.assertEqual(res.conflict.removals, ("A",))
        self.assertEqual(sorted(res.conflict.alternatives), [("B",), ("E",)])
        self.assert_witnesses_valid(ped, obs, res.conflict)

    def test_inbred_feasible_consanguinity_loop(self):
        # 表亲婚配但数据一致：验证近亲结构本身不误报
        ped = {
            "A": (None, None), "B": (None, None),
            "C": ("A", "B"), "D": ("A", "B"),
            "E": (None, None), "F": ("C", "E"),
            "G": (None, None), "H": ("D", "G"),
            "I": ("F", "H"),
        }
        obs = {"A": 0, "B": 0, "E": 2, "G": 2, "I": 1}
        res = check(ped, obs)
        self.assertEqual(res.status, "feasible")
        self.assertEqual(res.feasible_genotypes["C"], frozenset({0}))
        self.assertEqual(res.feasible_genotypes["F"], frozenset({1}))
        self.assertEqual(res.feasible_genotypes["I"], frozenset({1}))

    def test_single_parent_missing_conflict(self):
        # 母=0、缺失父未知：子=2 不可行；{C} 与 {M} 都是最小解释
        ped = {"M": (None, None), "C": (None, "M")}
        obs = {"M": 0, "C": 2}
        res = check(ped, obs)
        self.assertEqual(res.status, "infeasible")
        self.assertEqual(res.conflict.removals, ("C",))
        self.assertEqual(list(res.conflict.alternatives), [("M",)])
        self.assert_witnesses_valid(ped, obs, res.conflict)

    def test_two_independent_conflicts_need_two_removals(self):
        # 两个互不相干的三人冲突 -> 最小移除集大小为 2
        ped = {
            "A": (None, None), "B": (None, None), "C": ("A", "B"),
            "D": (None, None), "E": (None, None), "F": ("D", "E"),
        }
        obs = {"A": 2, "B": 2, "C": 0, "D": 0, "E": 0, "F": 2}
        res = check(ped, obs)
        self.assertEqual(res.status, "infeasible")
        all_sets = [res.conflict.removals, *res.conflict.alternatives]
        # 每个三人冲突的唯一最小修正是移除子观测 -> 全局唯一最小集 {C, F}
        self.assertEqual(sorted(all_sets), [("C", "F")])
        # 与独立穷举参考核对
        self.assertEqual(sorted(all_sets), ref_min_corrections(ped, obs))
        self.assert_witnesses_valid(ped, obs, res.conflict)


# ------------------------------------------------- 预算语义

class TestBudget(unittest.TestCase):
    def test_budget_exhaustion_is_unknown_not_infeasible(self):
        ped = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
        res = check(ped, {}, node_budget=1)
        self.assertEqual(res.status, "unknown")
        self.assertIsNone(res.feasible_genotypes)
        self.assertIsNone(res.conflict)

    def test_proven_infeasible_distinct_from_unknown(self):
        # 同一实例：预算充足 -> 证明无解；预算=1 -> unknown。两者严格分开。
        ped = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
        obs = {"A": 0, "B": 0, "C": 1}
        self.assertEqual(check(ped, obs).status, "infeasible")
        self.assertEqual(check(ped, obs, node_budget=1).status, "unknown")

    def test_nodes_used_within_budget(self):
        ped = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
        res = check(ped, {"A": 1}, node_budget=500)
        self.assertLessEqual(res.nodes_used, 500)


# ------------------------------------------------- 穷举参考对照（随机小实例）

class TestAgainstBruteForce(unittest.TestCase):
    def test_random_small_instances(self):
        n_cases = 0
        for seed in range(60):
            rng = random.Random(20260922 + seed)
            n = rng.randint(2, 8)
            ped = make_rng_pedigree(rng, n)
            obs = {
                pid: rng.randint(0, 2)
                for pid in ped
                if rng.random() < 0.6
            }
            res = check(ped, obs)
            self.assertNotEqual(res.status, "unknown", f"seed={seed}")
            ref_sets = ref_feasible_sets(ped, obs)
            if ref_sets is not None:
                self.assertEqual(res.status, "feasible", f"seed={seed}")
                self.assertEqual(res.feasible_genotypes, ref_sets, f"seed={seed}")
            else:
                self.assertEqual(res.status, "infeasible", f"seed={seed}")
                got = sorted([res.conflict.removals, *res.conflict.alternatives])
                want = ref_min_corrections(ped, obs)
                self.assertEqual(got, want, f"seed={seed}")
                # 每个被移除观测都有见证，且见证独立校验通过
                covered = set().union(*map(set, want))
                self.assertEqual(set(res.conflict.witnesses), covered)
                for obs_id, witness in res.conflict.witnesses.items():
                    owner = next(s for s in want if obs_id in s)
                    self.assertTrue(ref_consistent(ped, witness))
                    for pid, g in obs.items():
                        if pid not in owner:
                            self.assertEqual(witness[pid], g)
            n_cases += 1
        self.assertEqual(n_cases, 60)


if __name__ == "__main__":
    unittest.main()
