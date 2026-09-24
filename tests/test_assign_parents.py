"""assign_parents（多位点候选父母判定）的单元测试。

参考实现 ref_locus_feasible 为独立暴力枚举：对固定联合分配构造家系，
在单一位点上枚举全部 3^n 完整基因型指派逐一检验孟德尔一致性，
绝不调用被测核心 mendelcheck 得到期望值。
"""

import itertools
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mendelcheck  # noqa: F401  -- 导入即验证模块可加载
from mendelcheck import (
    AssignmentConflict,
    AssignmentResult,
    ValidationError,
    assign_parents,
)

TRANSMIT = ((0,), (0, 1), (1,))


# ------------------------------------------------- 独立参考实现（暴力枚举）

def ref_trio_ok(gf, gm, gc):
    return any(a + b == gc for a in TRANSMIT[gf] for b in TRANSMIT[gm])


def ref_consistent(pedigree, assignment):
    """完整指派是否满足全部孟德尔约束（与旧测试同一套独立规则）。"""
    for cid, (fa, mo) in pedigree.items():
        if fa is None and mo is None:
            continue
        gc = assignment[cid]
        fa_alleles = TRANSMIT[assignment[fa]] if fa is not None else (0, 1)
        mo_alleles = TRANSMIT[assignment[mo]] if mo is not None else (0, 1)
        if not any(a + b == gc for a in fa_alleles for b in mo_alleles):
            return False
    return True


def ref_is_acyclic(pedigree):
    color = {pid: 0 for pid in pedigree}

    def dfs(node):
        color[node] = 1
        for par in pedigree[node]:
            if par is None:
                continue
            if color[par] == 1 or (color[par] == 0 and not dfs(par)):
                return False
        color[node] = 2
        return True

    return all(color[n] != 0 or dfs(n) for n in pedigree)


def ref_locus_feasible(members, children, choice, profiles, locus):
    """给定一个联合分配（每孩子一对父母），单一位点上是否存在可行完整指派。"""
    pedigree = {pid: (None, None) for pid in members}
    pedigree.update(choice)
    if not ref_is_acyclic(pedigree):
        return False
    ids = list(members)
    for combo in itertools.product((0, 1, 2), repeat=len(ids)):
        assignment = dict(zip(ids, combo))
        if any(assignment[pid] != profiles[pid][locus] for pid in ids):
            continue
        if ref_consistent(pedigree, assignment):
            return True
    return False


def ref_enumerate(profiles, candidates):
    """返回 ({可行分配元组}, {分配元组: (好位点元组, 坏位点元组)})，全独立计算。"""
    children = sorted(candidates)
    loci = sorted(next(iter(profiles.values())))
    choices = itertools.product(*[range(len(candidates[c])) for c in children])
    feasible, per_locus = [], {}
    for sel in choices:
        choice = {
            cid: tuple(candidates[cid][pi]) for cid, pi in zip(children, sel)
        }
        key = tuple((cid, choice[cid]) for cid in children)
        good, bad = [], []
        for locus in loci:
            (good if ref_locus_feasible(
                list(profiles), children, choice, profiles, locus) else bad
            ).append(locus)
        per_locus[key] = (tuple(good), tuple(bad))
        if good == loci:
            feasible.append(choice)
    return feasible, per_locus


def make_random_case(rng):
    """确定性生成小规模随机输入；跳过任何含成环联合选择的情形。"""
    n = rng.randint(4, 7)
    members = [f"I{i}" for i in range(n)]
    loci = [f"L{i}" for i in range(rng.randint(1, 4))]
    profiles = {
        pid: {locus: rng.randint(0, 2) for locus in loci} for pid in members
    }
    n_children = rng.randint(1, min(3, n - 2))
    chosen_children = rng.sample(members, n_children)
    candidates = {}
    for cid in chosen_children:
        pool = [p for p in members if p != cid]
        pairs, seen = [], set()
        for _ in range(rng.randint(1, 3)):
            fa, mo = rng.sample(pool, 2)
            if (fa, mo) not in seen:
                seen.add((fa, mo))
                pairs.append((fa, mo))
        candidates[cid] = pairs
    # 全部联合选择都必须无环，否则本题输入会被拒绝（另测）。
    selections = itertools.product(*[range(len(candidates[c])) for c in sorted(candidates)])
    for sel in selections:
        ped = {pid: (None, None) for pid in members}
        for cid, pi in zip(sorted(candidates), sel):
            ped[cid] = candidates[cid][pi]
        if not ref_is_acyclic(ped):
            return None
    return profiles, candidates


# ------------------------------------------------- 手算正常用例

class TestAssignHandComputed(unittest.TestCase):
    def profiles(self):
        # F=0 x M=2：L1 上子女只能为 1；F=2 x M=2：L2 上子女只能为 2。
        # G=0 x H=0：两位点上子女都只能为 0。
        return {
            "F": {"L1": 0, "L2": 2},
            "M": {"L1": 2, "L2": 2},
            "G": {"L1": 0, "L2": 0},
            "H": {"L1": 0, "L2": 0},
            "C": {"L1": 1, "L2": 2},
        }

    def candidates(self):
        return {"C": [("F", "M"), ("G", "H")]}

    def test_unique_feasible_assignment_hand_calc(self):
        res = assign_parents(self.profiles(), self.candidates())
        self.assertEqual(res.status, "feasible")
        self.assertEqual(
            res.feasible_assignments, ({"C": ("F", "M")},)
        )
        self.assertEqual(res.conflicts, ())
        self.assertGreaterEqual(res.nodes_used, 1)

    def test_result_is_frozen_shape(self):
        res = assign_parents(self.profiles(), self.candidates())
        self.assertIsInstance(res, AssignmentResult)
        with self.assertRaises(Exception):
            res.status = "x"  # frozen dataclass

    def test_ordered_parents_distinct_pairs(self):
        # (F,M) 与 (M,F) 视为两个不同有序对，二者都可行时都应返回。
        profiles = self.profiles()
        candidates = {"C": [("F", "M"), ("M", "F")]}
        res = assign_parents(profiles, candidates)
        self.assertEqual(res.status, "feasible")
        self.assertEqual(
            res.feasible_assignments,
            ({"C": ("F", "M")}, {"C": ("M", "F")}),
        )

    def test_does_not_mutate_inputs(self):
        import copy
        profiles, candidates = self.profiles(), self.candidates()
        p0, c0 = copy.deepcopy(profiles), copy.deepcopy(candidates)
        assign_parents(profiles, candidates)
        self.assertEqual(profiles, p0)
        self.assertEqual(candidates, c0)


# ------------------------------------------------- 不可逐位点换父母 / 冲突位点

class TestAssignConflicts(unittest.TestCase):
    def test_per_locus_switching_rejected(self):
        # A 对只在 L1 相容，B 对只在 L2 相容；逐位点换父母才能救 -> 必须无解。
        profiles = {
            "F1": {"L1": 0, "L2": 2}, "M1": {"L1": 2, "L2": 2},
            "F2": {"L1": 2, "L2": 0}, "M2": {"L1": 2, "L2": 0},
            "C1": {"L1": 1, "L2": 0},
        }
        candidates = {"C1": [("F1", "M1"), ("F2", "M2")]}
        res = assign_parents(profiles, candidates)
        self.assertEqual(res.status, "infeasible")
        self.assertEqual(res.feasible_assignments, ())
        by_pair = {c.assignment["C1"]: c for c in res.conflicts}
        c_a = by_pair[("F1", "M1")]
        self.assertEqual(c_a.feasible_loci, ("L1",))
        self.assertEqual(c_a.conflicting_loci, ("L2",))
        self.assertEqual(c_a.unknown_loci, ())
        c_b = by_pair[("F2", "M2")]
        self.assertEqual(c_b.feasible_loci, ("L2",))
        self.assertEqual(c_b.conflicting_loci, ("L1",))
        for c in res.conflicts:
            self.assertIsInstance(c, AssignmentConflict)

    def test_both_loci_conflict(self):
        profiles = {
            "F": {"L1": 2, "L2": 0}, "M": {"L1": 2, "L2": 0},
            "C": {"L1": 0, "L2": 2},  # 2x2 只能生 2；0x0 只能生 0
        }
        res = assign_parents(profiles, {"C": [("F", "M")]})
        self.assertEqual(res.status, "infeasible")
        self.assertEqual(len(res.conflicts), 1)
        c = res.conflicts[0]
        self.assertEqual(c.assignment, {"C": ("F", "M")})
        self.assertEqual(c.feasible_loci, ())
        self.assertEqual(c.conflicting_loci, ("L1", "L2"))

    def test_multigeneration_joint_constraint(self):
        # D 的候选父母含 C，而 C 自身也在选父母：单独三人组都相容时，
        # 联合选择仍可能跨孩子矛盾（C=2 时 (C,A) 无法生出 D=0）。
        profiles = {
            "A": {"L1": 2}, "B": {"L1": 2},
            "E": {"L1": 0},
            "C": {"L1": 2},   # 只能来自 (A,B)
            "D": {"L1": 0},   # (C=?, A=2) 生 0 需要 C=0，但 C 被观测为 2
        }
        candidates = {
            "C": [("A", "B"), ("A", "E")],
            "D": [("C", "A"), ("E", "A")],
        }
        res = assign_parents(profiles, candidates)
        feasible, per_locus = ref_enumerate(profiles, candidates)
        self.assertFalse(feasible)
        got = {tuple(sorted(c.assignment.items())):
               (c.feasible_loci, c.conflicting_loci) for c in res.conflicts}
        self.assertEqual(res.status, "infeasible")
        for key, (good, bad) in per_locus.items():
            self.assertEqual(got[tuple(sorted(key))], (good, bad))


# ------------------------------------------------- 边界规模

class TestAssignBoundaries(unittest.TestCase):
    def _all_discordant(self, n_members, n_children, n_pairs, n_loci):
        """构造题面上限规模、所有联合分配都冲突的确定性输入。"""
        members = [f"I{i}" for i in range(n_members)]
        loci = [f"L{k}" for k in range(n_loci)]
        # 父母全 2/2 档案，孩子全 0 -> 任何 (2,2) 对在任一位点都不能生 0。
        profiles = {}
        for i, pid in enumerate(members):
            profiles[pid] = {locus: (0 if i >= n_members - n_children else 2)
                             for locus in loci}
        candidates = {}
        for ci in range(n_children):
            cid = members[n_members - n_children + ci]
            pool = members[:n_members - n_children]
            candidates[cid] = [
                (pool[2 * k % len(pool)], pool[(2 * k + 1) % len(pool)])
                for k in range(n_pairs)
            ]
        return profiles, candidates

    def test_max_box_27_assignments_4_loci(self):
        profiles, candidates = self._all_discordant(
            n_members=9, n_children=3, n_pairs=3, n_loci=4
        )
        res = assign_parents(profiles, candidates)
        self.assertEqual(res.status, "infeasible")
        self.assertEqual(len(res.conflicts), 27)  # 3^3 全枚举
        for c in res.conflicts:
            self.assertEqual(len(c.conflicting_loci), 4)
            self.assertEqual(c.feasible_loci, ())

    def test_minimal_size_one_child_one_pair_one_locus(self):
        profiles = {"A": {"L": 0}, "B": {"L": 2}, "C": {"L": 1}}
        res = assign_parents(profiles, {"C": [("A", "B")]})
        self.assertEqual(res.status, "feasible")
        self.assertEqual(res.feasible_assignments, ({"C": ("A", "B")},))

    def test_neutral_members_without_candidates_allowed(self):
        # 有档案但既不是孩子也不出现在候选对中的个体允许存在。
        profiles = {
            "A": {"L": 0}, "B": {"L": 2}, "C": {"L": 1}, "X": {"L": 0},
        }
        res = assign_parents(profiles, {"C": [("A", "B")]})
        self.assertEqual(res.status, "feasible")


# ------------------------------------------------- 输入校验

class TestAssignValidation(unittest.TestCase):
    def good(self):
        return (
            {"A": {"L": 0}, "B": {"L": 2}, "C": {"L": 1}},
            {"C": [("A", "B")]},
        )

    def test_type_errors(self):
        with self.assertRaises(ValidationError):
            assign_parents("nope", {})
        with self.assertRaises(ValidationError):
            assign_parents({}, {})  # 空档案
        p, c = self.good()
        with self.assertRaises(ValidationError):
            assign_parents(p, "nope")
        with self.assertRaises(ValidationError):
            assign_parents(p, {})  # 0 个孩子
        with self.assertRaises(ValidationError):
            assign_parents(p, {"C": ("A", "B")})  # 候选须为序列
        with self.assertRaises(ValidationError):
            assign_parents(p, {"C": [("A",)]})  # 长度不为 2
        with self.assertRaises(ValidationError):
            assign_parents(p, {"C": [(1, "B")]})  # 非 str 父母
        with self.assertRaises(ValidationError):
            assign_parents(p, {"C": [("A", None)]})  # 不支持未知父母候选
        with self.assertRaises(ValidationError):
            assign_parents(p, {"C": [("", "B")]})  # 空 ID

    def test_list_pair_is_accepted(self):
        p, _ = self.good()
        res = assign_parents(p, {"C": [["A", "B"]]})
        self.assertEqual(res.status, "feasible")
        self.assertEqual(res.feasible_assignments[0]["C"], ("A", "B"))

    def test_missing_references(self):
        p, _ = self.good()
        with self.assertRaises(ValidationError):
            assign_parents(p, {"Z": [("A", "B")]})  # 孩子无档案
        with self.assertRaises(ValidationError):
            assign_parents(p, {"C": [("A", "Z")]})  # 父母无档案

    def test_duplicate_pair(self):
        p, _ = self.good()
        with self.assertRaises(ValidationError) as cm:
            assign_parents(p, {"C": [("A", "B"), ("A", "B")]})
        self.assertIn("duplicate", str(cm.exception))

    def test_locus_set_mismatch_and_bad_genotypes(self):
        p = {"A": {"L": 0}, "B": {"L": 2, "L2": 1}, "C": {"L": 1}}
        with self.assertRaises(ValidationError):
            assign_parents(p, {"C": [("A", "B")]})
        for bad in (True, 1.0, float("nan"), float("inf"), 3, -1, "1"):
            p2 = {"A": {"L": bad}, "B": {"L": 2}, "C": {"L": 1}}
            with self.assertRaises(ValidationError, msg=f"genotype={bad!r}"):
                assign_parents(p2, {"C": [("A", "B")]})

    def test_bad_locus_name(self):
        p = {"A": {"": 0}, "B": {"": 2}, "C": {"": 1}}
        with self.assertRaises(ValidationError):
            assign_parents(p, {"C": [("A", "B")]})

    def test_limits(self):
        def build(n_children=1, n_pairs=1, n_loci=1, n_profiles=3):
            loci = [f"L{k}" for k in range(n_loci)]
            members = [f"I{i}" for i in range(n_profiles)]
            profiles = {pid: {l: 0 for l in loci} for pid in members}
            candidates = {}
            for ci in range(n_children):
                cid = f"K{ci}" if f"K{ci}" in profiles else members[-1 - ci]
                pool = [m for m in members if m != cid]
                candidates[cid] = [(pool[(2 * k) % len(pool)],
                                    pool[(2 * k + 1) % len(pool)])
                                   for k in range(n_pairs)]
            return profiles, candidates

        with self.assertRaises(ValidationError):  # 4 个孩子
            assign_parents(*build(n_children=4, n_profiles=8))
        with self.assertRaises(ValidationError):  # 每孩子 4 对
            assign_parents(*build(n_pairs=4, n_profiles=10))
        with self.assertRaises(ValidationError):  # 5 个位点
            assign_parents(*build(n_loci=5))
        loci = [f"L{k}" for k in range(5)]
        p21 = {f"I{i}": {l: 0 for l in loci[:1]} for i in range(21)}
        with self.assertRaises(ValidationError):  # 21 个档案
            assign_parents(p21, {"I0": [("I1", "I2")]})
        with self.assertRaises(ValidationError):  # 0 个位点
            assign_parents({"A": {}, "B": {}, "C": {}}, {"C": [("A", "B")]})

    def test_cyclic_selection_rejected(self):
        # 自环：C 候选自己为父
        profiles = {"A": {"L": 0}, "C": {"L": 1}}
        with self.assertRaises(ValidationError):
            assign_parents(profiles, {"C": [("C", "A")]})
        # A、C 互为对方父母 -> 某个联合选择成环
        profiles2 = {"A": {"L": 0}, "B": {"L": 0}, "C": {"L": 0}}
        candidates2 = {"A": [("C", "B")], "C": [("A", "B"), ("B", "B")]}
        with self.assertRaises(ValidationError) as cm:
            assign_parents(profiles2, candidates2)
        self.assertIn("cyclic", str(cm.exception))

    def test_budget_validation(self):
        p, c = self.good()
        for bad in (True, 0, -1, 2.0, float("nan"), "5"):
            with self.assertRaises(ValidationError, msg=f"node_budget={bad!r}"):
                assign_parents(p, c, node_budget=bad)


# ------------------------------------------------- 预算语义

class TestAssignBudget(unittest.TestCase):
    def test_unknown_distinct_from_infeasible(self):
        # 全员观测、三人结构：每个位点的判定恰耗 1 个搜索节点。
        profiles = {
            "F": {"L1": 2, "L2": 2, "L3": 0, "L4": 0},
            "M": {"L1": 2, "L2": 2, "L3": 0, "L4": 0},
            "C": {"L1": 0, "L2": 0, "L3": 1, "L4": 1},
        }
        candidates = {"C": [("F", "M")]}
        full = assign_parents(profiles, candidates)
        self.assertEqual(full.status, "infeasible")  # 预算充足时证明无解
        tiny = assign_parents(profiles, candidates, node_budget=3)
        self.assertEqual(tiny.status, "unknown")
        self.assertEqual(tiny.feasible_assignments, ())
        self.assertEqual(len(tiny.conflicts), 1)
        c = tiny.conflicts[0]
        self.assertEqual(len(c.feasible_loci), 0)
        # 前三个位点预算内判定（L1/L2 冲突、L3...），L4 起为 unknown
        self.assertEqual(len(c.conflicting_loci) + len(c.unknown_loci), 4)
        self.assertGreaterEqual(len(c.unknown_loci), 1)
        self.assertEqual(tiny.nodes_used, 3)

    def test_feasible_found_before_budget_runs_out(self):
        profiles = {"A": {"L": 0}, "B": {"L": 2}, "C": {"L": 1}}
        res = assign_parents(profiles, {"C": [("A", "B")]}, node_budget=100)
        self.assertEqual(res.status, "feasible")
        self.assertLessEqual(res.nodes_used, 100)


# ------------------------------------------------- 穷举参考对照（随机小实例）

class TestAgainstBruteForce(unittest.TestCase):
    def test_random_small_instances(self):
        n_cases = 0
        for seed in range(120):
            rng = random.Random(20260924 + seed)
            case = make_random_case(rng)
            if case is None:
                continue
            profiles, candidates = case
            res = assign_parents(profiles, candidates)
            self.assertNotEqual(res.status, "unknown", f"seed={seed}")
            feasible, per_locus = ref_enumerate(profiles, candidates)
            want_keys = {tuple(sorted(a.items())) for a in feasible}
            got_keys = {
                tuple(sorted(a.items())) for a in res.feasible_assignments
            }
            self.assertEqual(got_keys, want_keys, f"seed={seed}")
            if feasible:
                self.assertEqual(res.status, "feasible", f"seed={seed}")
                self.assertEqual(res.conflicts, ())
            else:
                self.assertEqual(res.status, "infeasible", f"seed={seed}")
                got_diag = {
                    tuple(sorted(c.assignment.items())):
                    (c.feasible_loci, c.conflicting_loci, c.unknown_loci)
                    for c in res.conflicts
                }
                # 每个联合分配都至少有一个坏位点，诊断与独立枚举逐位点一致。
                self.assertEqual(len(got_diag), len(per_locus), f"seed={seed}")
                for key, (good, bad) in per_locus.items():
                    self.assertTrue(bad, f"seed={seed}")
                    self.assertEqual(
                        got_diag[tuple(sorted(key))], (good, bad, ()),
                        f"seed={seed}",
                    )
            n_cases += 1
        self.assertGreaterEqual(n_cases, 80)


if __name__ == "__main__":
    unittest.main()
