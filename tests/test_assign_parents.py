"""assign_parents（多位点候选父母对分配）的单元测试。

参考实现 ref_* 为独立暴力枚举：对每种候选组合、每个位点，枚举家系
（孩子 + 被引用父母）的全部 3^n 基因型指派并检验孟德尔一致性，
绝不调用被测核心 mendelcheck 得到期望值。
"""

import itertools
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mendelcheck
from mendelcheck import AssignmentResult, ParentAssignment, ValidationError, assign_parents

TRANSMIT = ((0,), (0, 1), (1,))


# ------------------------------------------------- 独立参考实现（暴力枚举）

def ref_combo_pedigree(child_ids, chosen):
    """组合 chosen={child:(fa,mo)} 下的单位点家系（父母为创始者）。"""
    ped = {}
    for cid in child_ids:
        fa, mo = chosen[cid]
        ped[cid] = (fa, mo)
        if fa is not None and fa not in ped:
            ped[fa] = (None, None)
        if mo is not None and mo not in ped:
            ped[mo] = (None, None)
    return ped


def ref_locus_consistent(pedigree, assignment):
    for cid, (fa, mo) in pedigree.items():
        if fa is None and mo is None:
            continue
        gc = assignment[cid]
        fa_alleles = TRANSMIT[assignment[fa]] if fa is not None else (0, 1)
        mo_alleles = TRANSMIT[assignment[mo]] if mo is not None else (0, 1)
        if not any(a + b == gc for a in fa_alleles for b in mo_alleles):
            return False
    return True


def ref_locus_feasible(child_ids, chosen, gmap, locus):
    """该组合在该位点是否可行：枚举家系全部 3^n 指派。"""
    ped = ref_combo_pedigree(child_ids, chosen)
    ids = list(ped)
    for combo in itertools.product((0, 1, 2), repeat=len(ids)):
        assignment = dict(zip(ids, combo))
        if any(
            assignment[pid] != gmap[pid][locus]
            for pid in ids
            if locus in gmap.get(pid, {})
        ):
            continue
        if ref_locus_consistent(ped, assignment):
            return True
    return False


def ref_combo_feasible(child_ids, chosen, gmap, loci):
    return all(ref_locus_feasible(child_ids, chosen, gmap, l) for l in loci)


def ref_conflict_cores(child_ids, chosen, gmap, locus):
    """返回该位点不可行的极小孩子子集集合（极小不可行核）。"""
    bad = []
    for mask in range(1, 1 << len(child_ids)):
        sub = [child_ids[b] for b in range(len(child_ids)) if mask & (1 << b)]
        chosen_t = {c: chosen[c] for c in sub}
        if not ref_locus_feasible(sub, chosen_t, gmap, locus):
            bad.append(frozenset(sub))
    return {s for s in bad if not any(t < s for t in bad)}


def ref_solve(genotypes, candidates):
    """参考求解：返回 (feasible_combos, {idx_tuple: {child: [冲突位点]}}, loci)。"""
    child_ids = sorted(candidates)
    loci = tuple(sorted(
        {l for row in genotypes.values() for l in row},
        key=lambda l: (0, l) if isinstance(l, int) else (1, l),
    ))
    feasible = []
    bad_loci = {}
    ranges = [range(len(candidates[cid])) for cid in child_ids]
    for idxs in itertools.product(*ranges):
        chosen = {cid: candidates[cid][k] for cid, k in zip(child_ids, idxs)}
        if ref_combo_feasible(child_ids, chosen, genotypes, loci):
            feasible.append(idxs)
        else:
            flags = {cid: [] for cid in child_ids}
            for locus in loci:
                for core in ref_conflict_cores(child_ids, chosen, genotypes, locus):
                    for cid in core:
                        if locus not in flags[cid]:
                            flags[cid].append(locus)
            bad_loci[idxs] = flags
    return feasible, bad_loci, loci


# ------------------------------------------------- 手算例子

class TestHandComputed(unittest.TestCase):
    def test_single_child_picks_only_consistent_pair(self):
        # C：位点 1=1、位点 2=0。
        # 对 0 (A,B)=(0,2)@l1,(0,0)@l2：l1 子女必为 1 ✓；l2 必为 0 ✓。
        # 对 1 (D,E)=(2,2)@l1,(2,2)@l2：子女两位点都必为 2，全冲突。
        # 对 2 (None,F)，F=(2)@l1,(2)@l2：l1 子女 ∈{1,2} ✓；l2 ∈{1,2}，观测 0 冲突。
        genotypes = {
            "C": {1: 1, 2: 0},
            "A": {1: 0, 2: 0}, "B": {1: 2, 2: 0},
            "D": {1: 2, 2: 2}, "E": {1: 2, 2: 2},
            "F": {1: 2, 2: 2},
        }
        candidates = {"C": [("A", "B"), ("D", "E"), (None, "F")]}
        res = assign_parents(genotypes, candidates)
        self.assertEqual(res.status, "feasible")
        self.assertEqual(len(res.assignments), 1)
        combo = res.assignments[0]
        self.assertEqual(len(combo), 1)
        a = combo[0]
        self.assertEqual((a.child, a.pair_index, a.pair, a.loci_conflicts),
                         ("C", 0, ("A", "B"), ()))
        self.assertEqual(res.loci, (1, 2))

    def test_infeasible_reports_conflict_loci_per_rejected_pair(self):
        # 唯一孩子，三对候选全部不行：
        # (A,B)=2x2 -> C 必 2，观测 0/0：两位点都冲突；
        # (D,E)：D=0,E=0 @l1 -> C 必 0 ✓；D=2,E=2 @l2 -> C 必 2，观测 0：仅 l2 冲突；
        # (None,F)，F=0@l1 -> C∈{0,1} ✓；F=2@l2 -> C∈{1,2}，观测 0：仅 l2 冲突。
        genotypes = {
            "C": {1: 0, 2: 0},
            "A": {1: 2, 2: 2}, "B": {1: 2, 2: 2},
            "D": {1: 0, 2: 2}, "E": {1: 0, 2: 2},
            "F": {1: 0, 2: 2},
        }
        candidates = {"C": [("A", "B"), ("D", "E"), (None, "F")]}
        res = assign_parents(genotypes, candidates)
        self.assertEqual(res.status, "infeasible")
        self.assertEqual(res.assignments, ())
        self.assertEqual(len(res.conflicts), 3)  # 3 种被否决组合
        by_idx = {combo[0].pair_index: combo[0] for combo in res.conflicts}
        self.assertEqual(by_idx[0].loci_conflicts, (1, 2))
        self.assertEqual(by_idx[1].loci_conflicts, (2,))
        self.assertEqual(by_idx[2].loci_conflicts, (2,))
        for combo in res.conflicts:
            self.assertIsInstance(combo[0], ParentAssignment)
            self.assertEqual(combo[0].child, "C")

    def test_same_pair_must_hold_across_all_loci(self):
        # 对 0=(A,B)=0x0：l1 出 0 ✓，但 l2 出不了 2；
        # 对 1=(D,E)=2x2：l2 出 2 ✓，但 l1 出不了 0。
        # 逐位点换父母就能过，但同一选择必须跨全部位点 -> 整体 infeasible。
        genotypes = {
            "C": {1: 0, 2: 2},
            "A": {1: 0, 2: 0}, "B": {1: 0, 2: 0},
            "D": {1: 2, 2: 2}, "E": {1: 2, 2: 2},
        }
        candidates = {"C": [("A", "B"), ("D", "E")]}
        res = assign_parents(genotypes, candidates)
        self.assertEqual(res.status, "infeasible")
        by_idx = {combo[0].pair_index: combo[0] for combo in res.conflicts}
        self.assertEqual(by_idx[0].loci_conflicts, (2,))
        self.assertEqual(by_idx[1].loci_conflicts, (1,))

    def test_shared_observed_parent_evaluated_jointly(self):
        # 手算（P=2, Q=1, T=0；C1=1, C2=0）：
        #   C1 选 (P,Q)=(2,1)：子女 1 = P 传 1 + Q 传 0，可行；
        #   C1 选 (Q,T)=(1,0)：子女 1 = Q 传 1 + T 传 0，可行；
        #   C2 选 (P,T)=(2,0)：P=2 只能传 1，子女最小为 1，观测 0 不可行；
        #   C2 选 (Q,T)=(1,0)：Q 传 0 + T 传 0 = 0，可行。
        # 故可行组合为 (0,1)、(1,1)；两个不可行组合的极小核都是单元素 {C2}。
        genotypes = {
            "P": {1: 2}, "Q": {1: 1}, "T": {1: 0},
            "C1": {1: 1}, "C2": {1: 0},
        }
        candidates = {
            "C1": [("P", "Q"), ("Q", "T")],
            "C2": [("P", "T"), ("Q", "T")],
        }
        res = assign_parents(genotypes, candidates)
        self.assertEqual(res.status, "feasible")
        feasible_idxs = {tuple(a.pair_index for a in combo) for combo in res.assignments}
        self.assertEqual(feasible_idxs, {(0, 1), (1, 1)})

    def test_infeasible_combo_flags_only_individually_bad_child(self):
        # 唯一组合：C2 的三元组单独即不可行（P=2 只能传 1，T=0 传 0，
        # 子女最小为 1，观测 C2=0）；C1=(P,Q)=(2,1)->子女1 可行。
        # 极小核为 {C2}，不连坐 C1。
        genotypes = {
            "P": {1: 2}, "Q": {1: 1}, "T": {1: 0},
            "C1": {1: 1}, "C2": {1: 0},
        }
        candidates = {"C1": [("P", "Q")], "C2": [("P", "T")]}
        res = assign_parents(genotypes, candidates)
        self.assertEqual(res.status, "infeasible")
        self.assertEqual(len(res.conflicts), 1)
        flags = {a.child: a.loci_conflicts for a in res.conflicts[0]}
        self.assertEqual(flags, {"C1": (), "C2": (1,)})


# ------------------------------------------------- 边界

class TestBoundaries(unittest.TestCase):
    def test_max_size_three_children_three_pairs_four_loci(self):
        loci = (1, 2, 3, 4)
        genotypes = {}
        candidates = {}
        parents = [f"P{i}" for i in range(6)]
        for p in parents:
            genotypes[p] = {l: 1 for l in loci}
        for k in range(3):
            cid = f"C{k}"
            genotypes[cid] = {l: 1 for l in loci}  # 1x1 -> 子女 0/1/2 均可
            candidates[cid] = [
                (parents[2 * k], parents[2 * k + 1]),
                (None, parents[2 * k]),
                (parents[2 * k + 1], None),
            ]
        res = assign_parents(genotypes, candidates)
        self.assertEqual(res.status, "feasible")
        self.assertEqual(len(res.assignments), 27)
        self.assertEqual(res.loci, loci)
        for combo in res.assignments:
            self.assertEqual([a.child for a in combo], ["C0", "C1", "C2"])

    def test_partially_observed_parents(self):
        # 父母只在部分位点有观测：未观测位点自由。
        genotypes = {
            "C": {1: 2, 2: 2},
            "A": {1: 2},          # 缺 l2
            "B": {1: 2, 2: 2},
        }
        res = assign_parents(genotypes, {"C": [("A", "B")]})
        self.assertEqual(res.status, "feasible")

    def test_unobserved_parent_id_is_fine(self):
        # 父母 ID 出现于候选但完全无观测：创始者自由。
        genotypes = {"C": {1: 0}}
        res = assign_parents(genotypes, {"C": [("X", "Y"), (None, None)]})
        self.assertEqual(res.status, "feasible")
        self.assertEqual(len(res.assignments), 2)

    def test_string_loci_sorted_lexicographically(self):
        genotypes = {
            "C": {"b": 0, "a": 0},
            "X": {"b": 0, "a": 0}, "Y": {"b": 0, "a": 0},
        }
        res = assign_parents(genotypes, {"C": [("X", "Y")]})
        self.assertEqual(res.loci, ("a", "b"))

    def test_mixed_int_str_loci_ordering(self):
        genotypes = {
            "C": {10: 0, "2": 0, 1: 0},
            "X": {10: 0, "2": 0, 1: 0}, "Y": {10: 0, "2": 0, 1: 0},
        }
        res = assign_parents(genotypes, {"C": [("X", "Y")]})
        self.assertEqual(res.loci, (1, 10, "2"))

    def test_too_many_children(self):
        genotypes = {f"C{i}": {1: 0} for i in range(4)}
        candidates = {f"C{i}": [(None, None)] for i in range(4)}
        with self.assertRaises(ValidationError) as cm:
            assign_parents(genotypes, candidates)
        self.assertIn("children", str(cm.exception))

    def test_no_children(self):
        with self.assertRaises(ValidationError):
            assign_parents({"X": {1: 0}}, {})

    def test_too_many_or_zero_pairs(self):
        genotypes = {"C": {1: 0}}
        with self.assertRaises(ValidationError):
            assign_parents(genotypes, {"C": []})
        with self.assertRaises(ValidationError):
            assign_parents(genotypes, {"C": [(None, None)] * 4})

    def test_too_many_loci(self):
        genotypes = {"C": {i: 0 for i in range(5)},
                     "X": {i: 0 for i in range(5)}, "Y": {i: 0 for i in range(5)}}
        with self.assertRaises(ValidationError) as cm:
            assign_parents(genotypes, {"C": [("X", "Y")]})
        self.assertIn("loci", str(cm.exception))

    def test_duplicate_pair_rejected(self):
        genotypes = {"C": {1: 0}, "X": {1: 0}, "Y": {1: 0}}
        with self.assertRaises(ValidationError):
            assign_parents(genotypes, {"C": [("X", "Y"), ("X", "Y")]})
        # 槽位顺序不同视为不同对
        res = assign_parents(genotypes, {"C": [("X", "Y"), ("Y", "X")]})
        self.assertEqual(res.status, "feasible")

    def test_child_missing_locus_rejected(self):
        genotypes = {"C": {1: 0}, "X": {1: 0, 2: 0}, "Y": {1: 0, 2: 0}}
        with self.assertRaises(ValidationError) as cm:
            assign_parents(genotypes, {"C": [("X", "Y")]})
        self.assertIn("C", str(cm.exception))

    def test_unobserved_child_rejected(self):
        with self.assertRaises(ValidationError):
            assign_parents({"X": {1: 0}}, {"C": [("X", None)]})

    def test_parent_that_is_another_child_rejected(self):
        genotypes = {"C1": {1: 0}, "C2": {1: 0}, "X": {1: 0}}
        with self.assertRaises(ValidationError):
            assign_parents(genotypes, {"C1": [("C2", "X")], "C2": [(None, None)]})

    def test_self_parent_rejected(self):
        with self.assertRaises(ValidationError):
            assign_parents({"C": {1: 0}}, {"C": [("C", None)]})

    def test_empty_id_rejected(self):
        with self.assertRaises(ValidationError):
            assign_parents({"C": {1: 0}, "": {1: 0}}, {"C": [("", None)]})


# ------------------------------------------------- 输入校验

class TestAssignmentValidation(unittest.TestCase):
    def good(self):
        return (
            {"C": {1: 1}, "A": {1: 0}, "B": {1: 2}},
            {"C": [("A", "B")]},
        )

    def test_genotypes_type_errors(self):
        g, c = self.good()
        with self.assertRaises(ValidationError):
            assign_parents("nope", c)
        with self.assertRaises(ValidationError):
            assign_parents([], c)
        with self.assertRaises(ValidationError):
            assign_parents({}, c)
        with self.assertRaises(ValidationError):
            assign_parents({1: {1: 0}}, c)  # 非 str ID

    def test_profile_type_errors(self):
        with self.assertRaises(ValidationError):
            assign_parents({"C": [0, 1]}, {"C": [(None, None)]})
        with self.assertRaises(ValidationError):
            assign_parents({"C": {}}, {"C": [(None, None)]})  # 空 profile

    def test_locus_key_errors(self):
        with self.assertRaises(ValidationError):
            assign_parents({"C": {True: 0}}, {"C": [(None, None)]})
        with self.assertRaises(ValidationError):
            assign_parents({"C": {1.0: 0}}, {"C": [(None, None)]})
        with self.assertRaises(ValidationError):
            assign_parents({"C": {"": 0}}, {"C": [(None, None)]})

    def test_genotype_value_errors(self):
        for bad in (3, -1, True, 1.0, float("nan"), float("inf")):
            with self.assertRaises(ValidationError, msg=f"g={bad!r}"):
                assign_parents({"C": {1: bad}}, {"C": [(None, None)]})

    def test_candidates_type_errors(self):
        g, _ = self.good()
        with self.assertRaises(ValidationError):
            assign_parents(g, "nope")
        with self.assertRaises(ValidationError):
            assign_parents(g, {"C": ("A", "B")})     # 非列表
        with self.assertRaises(ValidationError):
            assign_parents(g, {"C": ["AB"]})         # 非二元组
        with self.assertRaises(ValidationError):
            assign_parents(g, {"C": [("A", "B", "X")]})
        with self.assertRaises(ValidationError):
            assign_parents(g, {"C": [(1, "B")]})     # 非 str/None 父母
        with self.assertRaises(ValidationError):
            assign_parents(g, {1: [("A", "B")]})     # 非 str 孩子 ID

    def test_budget_errors(self):
        g, c = self.good()
        for bad in (True, 0, -3, 2.5, float("nan"), "100"):
            with self.assertRaises(ValidationError, msg=f"node_budget={bad!r}"):
                assign_parents(g, c, node_budget=bad)

    def test_result_shape(self):
        g, c = self.good()
        res = assign_parents(g, c)
        self.assertIsInstance(res, AssignmentResult)
        self.assertEqual(res.status, "feasible")
        self.assertEqual(res.conflicts, ())
        self.assertGreaterEqual(res.nodes_used, 1)

    def test_inputs_not_mutated_feasible_and_infeasible(self):
        import copy
        g, c = self.good()
        g0, c0 = copy.deepcopy(g), copy.deepcopy(c)
        assign_parents(g, c)
        self.assertEqual((g, c), (g0, c0))
        g2 = {"C": {1: 0, 2: 2},
              "A": {1: 0, 2: 0}, "B": {1: 0, 2: 0},
              "D": {1: 2, 2: 2}, "E": {1: 2, 2: 2}}
        c2 = {"C": [("A", "B"), ("D", "E")]}
        g2_0, c2_0 = copy.deepcopy(g2), copy.deepcopy(c2)
        assign_parents(g2, c2)
        self.assertEqual((g2, c2), (g2_0, c2_0))

    def test_failed_validation_leaves_inputs_intact(self):
        import copy
        g = {"C": {1: 0}, "X": {1: 0, 2: 0}, "Y": {1: 0, 2: 0}}
        c = {"C": [("X", "Y")]}
        g0, c0 = copy.deepcopy(g), copy.deepcopy(c)
        with self.assertRaises(ValidationError):
            assign_parents(g, c)
        self.assertEqual((g, c), (g0, c0))


# ------------------------------------------------- 预算语义

class TestAssignmentBudget(unittest.TestCase):
    def test_tiny_budget_is_unknown(self):
        genotypes = {"C": {1: 1, 2: 1}, "A": {1: 0, 2: 0}, "B": {1: 2, 2: 2}}
        res = assign_parents(genotypes, {"C": [("A", "B")]}, node_budget=1)
        self.assertEqual(res.status, "unknown")
        self.assertEqual(res.assignments, ())
        self.assertEqual(res.conflicts, ())
        self.assertEqual(res.nodes_used, 1)

    def test_same_input_feasible_with_default_budget(self):
        genotypes = {"C": {1: 1, 2: 1}, "A": {1: 0, 2: 0}, "B": {1: 2, 2: 2}}
        self.assertEqual(assign_parents(genotypes, {"C": [("A", "B")]}).status,
                         "feasible")


# ------------------------------------------------- 穷举参考对照（随机小实例）

def make_random_problem(rng):
    """确定性生成一个题面上限内的分配问题（父母池与孩子完全分开）。"""
    n_children = rng.randint(1, 3)
    n_loci = rng.randint(1, 4)
    loci = tuple(range(n_loci))
    child_ids = [f"C{i}" for i in range(n_children)]
    pool = [f"P{i}" for i in range(rng.randint(2, 6))]

    def random_parent():
        if rng.random() < 0.25:
            return None
        return rng.choice(pool)

    candidates = {}
    for cid in child_ids:
        pairs = []
        while len(pairs) < rng.randint(1, 3):
            fa, mo = random_parent(), random_parent()
            if fa is not None and mo is not None and fa == mo:
                continue
            if (fa, mo) not in pairs:
                pairs.append((fa, mo))
        candidates[cid] = pairs

    referenced = {
        p for pairs in candidates.values() for pair in pairs for p in pair if p
    }
    genotypes = {}
    for p in sorted(referenced):
        if rng.random() < 0.85:
            genotypes[p] = {l: rng.randint(0, 2) for l in loci if rng.random() < 0.8}
            if not genotypes[p]:
                genotypes[p] = {loci[0]: rng.randint(0, 2)}
    for cid in child_ids:  # 孩子必须全位点观测
        genotypes[cid] = {l: rng.randint(0, 2) for l in loci}
    return genotypes, candidates, child_ids, loci


class TestAgainstBruteForce(unittest.TestCase):
    def test_random_small_instances(self):
        n_cases = 0
        for seed in range(120):
            rng = random.Random(20260924 + seed)
            genotypes, candidates, child_ids, loci = make_random_problem(rng)
            res = assign_parents(genotypes, candidates)
            self.assertNotEqual(res.status, "unknown", f"seed={seed}")
            self.assertEqual(res.loci, loci, f"seed={seed}")

            want_feasible, want_bad, _ = ref_solve(genotypes, candidates)

            if want_feasible:
                self.assertEqual(res.status, "feasible", f"seed={seed}")
                got = {
                    tuple(a.pair_index for a in combo)
                    for combo in res.assignments
                }
                self.assertEqual(got, set(want_feasible), f"seed={seed}")
                for combo in res.assignments:
                    self.assertTrue(all(a.loci_conflicts == () for a in combo))
            else:
                self.assertEqual(res.status, "infeasible", f"seed={seed}")
                self.assertEqual(len(res.conflicts), len(want_bad), f"seed={seed}")
                for combo in res.conflicts:
                    idxs = tuple(a.pair_index for a in combo)
                    got_flags = {a.child: list(a.loci_conflicts) for a in combo}
                    self.assertEqual(got_flags, want_bad[idxs],
                                     f"seed={seed} combo={idxs}")
            n_cases += 1
        self.assertEqual(n_cases, 120)


if __name__ == "__main__":
    unittest.main()
