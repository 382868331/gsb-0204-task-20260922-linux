"""结构化家系用例 + 随机小家系对拍独立穷举参考。

独立参考见 tests/_brute_reference.py（不 import 被测核心）。
"""

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

import _brute_reference as brute  # noqa: E402

from mendel_conflict import (  # noqa: E402
    STATUS_FEASIBLE,
    STATUS_INCONSISTENT,
    STATUS_UNKNOWN,
    check,
)
from mendel_conflict.core import verify_witness  # noqa: E402


def ped(rows):
    """rows: [(id, father, mother), ...] -> pedigree dict"""
    return {
        "individuals": [
            {"id": i, "father": f, "mother": m} for i, f, m in rows
        ]
    }


class TestHandCases(unittest.TestCase):
    def test_trio_dominant_x_recessive_forces_heterozygote(self):
        # 父 0（aa）× 母 2（AA）→ 子女只能是 1（Aa）
        p = ped([("f", None, None), ("m", None, None), ("c", "f", "m")])
        r = check(p, {"f": 0, "m": 2})
        self.assertEqual(r.status, STATUS_FEASIBLE)
        self.assertEqual(r.feasible_genotypes["c"], (1,))
        self.assertEqual(r.feasible_genotypes["f"], (0,))
        self.assertEqual(r.feasible_genotypes["m"], (2,))

    def test_observed_child_0_back_propagates_parent_domains(self):
        # 子女 0（aa）→ 父母都必须携带 a：取值 ⊆ {0,1}
        p = ped([("f", None, None), ("m", None, None), ("c", "f", "m")])
        r = check(p, {"c": 0})
        self.assertEqual(r.status, STATUS_FEASIBLE)
        self.assertEqual(r.feasible_genotypes["f"], (0, 1))
        self.assertEqual(r.feasible_genotypes["m"], (0, 1))
        self.assertEqual(r.feasible_genotypes["c"], (0,))

    def test_direct_conflict_22_cannot_make_0(self):
        p = ped([("f", None, None), ("m", None, None), ("c", "f", "m")])
        r = check(p, {"f": 2, "m": 2, "c": 0})
        self.assertEqual(r.status, STATUS_INCONSISTENT)
        # 穷举参考给出的最小解释
        exp, wits = brute.minimum_removals(p, {"f": 2, "m": 2, "c": 0})
        self.assertEqual(list(r.minimal_removals), exp)
        # 每个解释都是单观测移除，且见证确实改变了该观测
        for removal in r.minimal_removals:
            for rid in removal:
                w = r.witnesses[(removal, rid)]
                self.assertTrue(
                    brute.witness_check(
                        p, {"f": 2, "m": 2, "c": 0}, w, removal
                    )
                )
                self.assertNotEqual(w[rid], {"f": 2, "m": 2, "c": 0}[rid])

    def test_multigeneration_implicit_conflict(self):
        # a,b 均为 2 => c 必为 2；d,e 均为 0 => f 必为 0；
        # c(2) × f(0) => g 必为 1；观测 g=0 冲突，
        # 且冲突在直接三元组层面不可见（c/f 本身未观测）。
        p = ped(
            [
                ("a", None, None),
                ("b", None, None),
                ("d", None, None),
                ("e", None, None),
                ("c", "a", "b"),
                ("f", "d", "e"),
                ("g", "c", "f"),
            ]
        )
        obs = {"a": 2, "b": 2, "d": 0, "e": 0, "g": 0}

        # 去掉 g 的冲突观测前：c=2,f=0 被隐含强制
        ok = check(p, {k: v for k, v in obs.items() if k != "g"})
        self.assertEqual(ok.status, STATUS_FEASIBLE)
        self.assertEqual(ok.feasible_genotypes["c"], (2,))
        self.assertEqual(ok.feasible_genotypes["f"], (0,))
        self.assertEqual(ok.feasible_genotypes["g"], (1,))

        r = check(p, obs)
        self.assertEqual(r.status, STATUS_INCONSISTENT)
        exp, _ = brute.minimum_removals(p, obs)
        self.assertEqual(list(r.minimal_removals), exp)
        for removal in r.minimal_removals:
            for rid in removal:
                w = r.witnesses[(removal, rid)]
                self.assertTrue(brute.witness_check(p, obs, w, removal))
                self.assertNotEqual(w[rid], obs[rid])

    def test_single_missing_parent_two_minimum_explanations(self):
        # 父 2（AA）必传 A；母缺失（独立未知创始者）。
        # 子女 0（aa）不可能 => 移除 {f} 或 {c} 都可恢复，两个最小解释。
        p = ped([("f", None, None), ("c", "f", None)])
        obs = {"f": 2, "c": 0}
        r = check(p, obs)
        self.assertEqual(r.status, STATUS_INCONSISTENT)
        self.assertEqual(
            list(r.minimal_removals), [("c",), ("f",)]
        )  # 字典序
        w_c = r.witnesses[(("c",), "c")]
        w_f = r.witnesses[(("f",), "f")]
        self.assertTrue(brute.witness_check(p, obs, w_c, {"c"}))
        self.assertTrue(brute.witness_check(p, obs, w_f, {"f"}))
        self.assertNotEqual(w_c["c"], 0)
        self.assertNotEqual(w_f["f"], 2)
        self.assertTrue(verify_witness(p, obs, w_c, frozenset({"c"})))

    def test_witness_for_helper(self):
        p = ped([("f", None, None), ("c", "f", None)])
        obs = {"f": 2, "c": 0}
        r = check(p, obs)
        # 同一观测只在一个解释中时可省略 removal_set
        self.assertEqual(r.witness_for("c")["c"], 1)
        self.assertEqual(r.witness_for("f")["f"], 0)
        # 显式指定也行
        self.assertEqual(r.witness_for("c", ("c",))["c"], 1)
        with self.assertRaises(KeyError):
            r.witness_for("nope")

    def test_missing_parent_is_independent_founder(self):
        # 同一缺失母亲位置出现在两个单亲子女中时，二者不要求共享等位来源：
        # f1=2 -> c1 可 1（缺失母传 a）；f2=0 -> c2 可 1（缺失母传 A）。
        p = ped(
            [
                ("f1", None, None),
                ("f2", None, None),
                ("c1", "f1", None),
                ("c2", "f2", None),
            ]
        )
        r = check(p, {"f1": 2, "f2": 0, "c1": 1, "c2": 1})
        self.assertEqual(r.status, STATUS_FEASIBLE)

    def test_inbreeding_half_sib_mating_feasible_sets(self):
        # 近亲：a 的两个子女 c（a×b）与 d（a×缺失母）生育 e。
        # 直接与穷举参考对拍可行集合。
        p = ped(
            [
                ("a", None, None),
                ("b", None, None),
                ("c", "a", "b"),
                ("d", "a", None),
                ("e", "c", "d"),
            ]
        )
        obs = {"a": 1, "e": 0}
        r = check(p, obs)
        self.assertEqual(r.status, STATUS_FEASIBLE)
        exp = brute.feasible_sets(p, obs)
        for iid, vals in r.feasible_genotypes.items():
            self.assertEqual(frozenset(vals), exp[iid], iid)

    def test_inbreeding_conflict_vs_brute(self):
        p = ped(
            [
                ("a", None, None),
                ("b", None, None),
                ("c", "a", "b"),
                ("d", "a", None),
                ("e", "c", "d"),
            ]
        )
        obs = {"a": 2, "b": 2, "c": 2, "e": 0}
        r = check(p, obs)
        self.assertEqual(r.status, STATUS_INCONSISTENT)
        exp, _ = brute.minimum_removals(p, obs)
        self.assertEqual(list(r.minimal_removals), exp)
        for removal in r.minimal_removals:
            for rid in removal:
                w = r.witnesses[(removal, rid)]
                self.assertTrue(brute.witness_check(p, obs, w, removal))

    def test_no_observations_everything_feasible(self):
        p = ped([("f", None, None), ("m", None, None), ("c", "f", "m")])
        r = check(p)
        self.assertEqual(r.status, STATUS_FEASIBLE)
        for vals in r.feasible_genotypes.values():
            self.assertEqual(vals, (0, 1, 2))


class TestRandomizedAgainstBrute(unittest.TestCase):
    def test_random_small_pedigrees(self):
        rng = random.Random(20260923)
        trials = 40
        for t in range(trials):
            n = rng.randint(1, 7)
            ids = [f"I{i:02d}" for i in range(n)]
            rows = []
            for i, iid in enumerate(ids):
                earlier = ids[:i]
                fa = rng.choice(earlier) if earlier and rng.random() < 0.7 else None
                mo = None
                if earlier and rng.random() < 0.7:
                    pool = [x for x in earlier if x != fa]
                    mo = rng.choice(pool) if pool else None
                rows.append((iid, fa, mo))
            p = ped(rows)

            # 观测个体 ≤4，保证穷举参考极快
            k = rng.randint(0, min(4, n))
            chosen = rng.sample(ids, k)
            obs = {x: rng.randint(0, 2) for x in chosen}

            r = check(p, obs)
            brute_feasible = brute.is_feasible(p, obs)
            self.assertEqual(
                r.status == STATUS_FEASIBLE,
                brute_feasible,
                msg=f"trial {t}: {p} {obs}",
            )

            if brute_feasible:
                exp = brute.feasible_sets(p, obs)
                self.assertEqual(set(r.feasible_genotypes), set(exp))
                for iid in ids:
                    self.assertEqual(
                        frozenset(r.feasible_genotypes[iid]),
                        exp[iid],
                        msg=f"trial {t} id={iid}: {p} {obs}",
                    )
            else:
                exp, _ = brute.minimum_removals(p, obs)
                self.assertEqual(
                    list(r.minimal_removals),
                    exp,
                    msg=f"trial {t}: {p} {obs}",
                )
                for removal in r.minimal_removals:
                    self.assertEqual(removal, tuple(sorted(removal)))
                    for rid in removal:
                        w = r.witnesses[(removal, rid)]
                        self.assertTrue(
                            brute.witness_check(p, obs, w, removal),
                            msg=f"bad witness {w} for {removal}",
                        )
                        # 最小基数移除集的见证必须在每个被移除观测上都不同，
                        # 否则少移除一个也可行，与最小性矛盾。
                        self.assertNotEqual(w[rid], obs[rid])


if __name__ == "__main__":
    unittest.main()
