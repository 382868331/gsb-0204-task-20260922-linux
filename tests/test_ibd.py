"""ibd_sharing 单位点 IBD 查询的单元测试。

ref_* 为独立参考：自行构造创始人标签并递归枚举每次减数分裂选择，
用 Counter 多重集交计算最大匹配；绝不调用被测核心 mendelcheck 得到期望值。
"""

import itertools
import os
import random
import sys
import unittest
from collections import Counter
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mendelcheck import (
    MAX_IBD_TRANSMISSIONS,
    IBDResult,
    ValidationError,
    ibd_sharing,
)
import mendelcheck
CoreFraction = mendelcheck.Fraction


# ----------------------------------------------- 独立参考实现（递归枚举）

def ref_ibd_counts(pedigree, target1, target2):
    """独立穷举：返回 {共享份数: 组合数} 与传递次数、总组合数。

    递归按拓扑序确定每个体；无记录创始人给两份互不相同、且全局唯一的标签；
    缺失亲本槽位现场生成独立创始人。每个亲本方向独立抛一枚均匀硬币。
    """
    ids = list(pedigree)
    done = set()
    labels = {}
    counter = Counter()
    founder_counter = [0]

    def new_founder():
        n = founder_counter[0]
        founder_counter[0] += 1
        return (f"R{n}a", f"R{n}b")

    def match(x, y):
        cx, cy = Counter(x), Counter(y)
        return sum((cx & cy).values())

    def fill_next(prefix_idx):
        if prefix_idx == len(ids):
            counter[match(labels[target1], labels[target2])] += 1
            return
        pid = ids[prefix_idx]  # 父母只能取自更早成员（make_dag 保证）
        fa, mo = pedigree[pid]
        if fa is None and mo is None:
            labels[pid] = new_founder()
            done.add(pid)
            fill_next(prefix_idx + 1)
            done.discard(pid)
            del labels[pid]
            return
        pair_f = labels[fa] if fa is not None else new_founder()
        pair_m = labels[mo] if mo is not None else new_founder()
        for bf, bm in itertools.product((0, 1), (0, 1)):
            labels[pid] = (pair_f[bf], pair_m[bm])
            done.add(pid)
            fill_next(prefix_idx + 1)
            done.discard(pid)
            del labels[pid]

    fill_next(0)
    n_nonfounders = sum(
        1 for p in pedigree if pedigree[p][0] is not None or pedigree[p][1] is not None
    )
    transmissions = 2 * n_nonfounders
    return dict(counter), transmissions, 1 << transmissions


def ref_ibd(pedigree, a, b):
    counts, transmissions, total = ref_ibd_counts(pedigree, a, b)
    return {k: Fraction(counts.get(k, 0), total) for k in (0, 1, 2)}, transmissions


def make_dag(rng, n):
    """确定性生成 n 个成员的无有向环家系（父母只能取自更早的成员）。"""
    ids = [f"V{i}" for i in range(n)]
    ped = {ids[0]: (None, None)}
    for i in range(1, n):
        earlier = ids[:i]

        def pick():
            return rng.choice(earlier) if rng.random() < 0.55 else None

        fa, mo = pick(), pick()
        if fa is not None and mo is not None and fa == mo:
            mo = None
        ped[ids[i]] = (fa, mo)
    return ped


# ----------------------------------------------- 题目指定关系用例

class TestCanonicalRelationships(unittest.TestCase):
    def assert_distribution(self, result, want):
        got = (result.p0, result.p1, result.p2)
        self.assertEqual(got, tuple(Fraction(x) for x in want))
        self.assertEqual(result.shared_counts, {0: got[0], 1: got[1], 2: got[2]})
        self.assertEqual(result.p0 + result.p1 + result.p2, 1)
        self.assertIsInstance(result.p0, CoreFraction)
        self.assertIsInstance(result, IBDResult)

    def test_unrelated_founders(self):
        ped = {"A": (None, None), "B": (None, None)}
        res = ibd_sharing(ped, "A", "B")
        self.assert_distribution(res, (1, 0, 0))
        self.assertEqual(res.transmissions, 0)  # 无传递组合时枚举只有 1 个世界

    def test_parent_child(self):
        ped = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
        res = ibd_sharing(ped, "A", "C")
        self.assert_distribution(res, (0, 1, 0))
        self.assertEqual(res.transmissions, 2)

    def test_full_siblings(self):
        ped = {
            "A": (None, None), "B": (None, None),
            "C": ("A", "B"), "D": ("A", "B"),
        }
        res = ibd_sharing(ped, "C", "D")
        self.assert_distribution(res, (Fraction(1, 4), Fraction(1, 2), Fraction(1, 4)))

    def test_half_siblings(self):
        # 同母异父：各自缺失的父亲是独立创始人
        ped = {"M": (None, None), "C1": (None, "M"), "C2": (None, "M")}
        res = ibd_sharing(ped, "C1", "C2")
        self.assert_distribution(res, (Fraction(1, 2), Fraction(1, 2), 0))

    def test_sibling_mating_inbreeding(self):
        # 同胞交配：E 有 1/4 概率两份拷贝来自同一祖先标签（近交纯合）。
        ped = {
            "A": (None, None), "B": (None, None),
            "C": ("A", "B"), "D": ("A", "B"),
            "E": ("C", "D"),
        }
        # E 与祖父 A：p2 = P(父系传 A) * P(母系也传同一份 A) = 1/2 * 1/4 = 1/8
        res_a = ibd_sharing(ped, "E", "A")
        self.assert_distribution(
            res_a, (Fraction(1, 4), Fraction(5, 8), Fraction(1, 8))
        )
        # E 与父亲 C：必然共享 1；另有 1/4 概率 E 的母系拷贝也与 C 同源
        res_c = ibd_sharing(ped, "E", "C")
        self.assert_distribution(
            res_c, (0, Fraction(3, 4), Fraction(1, 4))
        )
        # 同一成员：自己的两份拷贝与自身完全匹配，恒为 2
        res_self = ibd_sharing(ped, "E", "E")
        self.assert_distribution(res_self, (0, 0, 1))

    def test_cousin_mating_multiple_paths(self):
        # 表亲婚配（多条祖先路径、无有向环）：与独立参考逐一核对
        ped = {
            "A": (None, None), "B": (None, None),
            "C": ("A", "B"), "D": ("A", "B"),
            "E": (None, None), "F": ("C", "E"),
            "G": (None, None), "H": ("D", "G"),
            "I": ("F", "H"),
        }
        for x, y in (("I", "A"), ("I", "C"), ("I", "F"), ("F", "H"),
                     ("A", "B"), ("C", "D")):
            res = ibd_sharing(ped, x, y)
            want, transmissions = ref_ibd(ped, x, y)
            self.assertEqual(res.shared_counts, want, f"{x}-{y}")
            self.assertEqual(res.transmissions, transmissions)

    def test_symmetry(self):
        ped = {
            "A": (None, None), "B": (None, None),
            "C": ("A", "B"), "D": ("A", "B"),
            "E": ("C", "D"),
        }
        self.assertEqual(
            ibd_sharing(ped, "E", "A").shared_counts,
            ibd_sharing(ped, "A", "E").shared_counts,
        )


# ----------------------------------------------- 独立穷举对照（随机小 DAG）

class TestAgainstIndependentEnumeration(unittest.TestCase):
    def test_random_small_dags(self):
        cases = 0
        for seed in range(80):
            rng = random.Random(4242 + seed)
            n = rng.randint(2, 8)
            ped = make_dag(rng, n)
            n_nonfounders = sum(
                1 for p in ped if ped[p][0] is not None or ped[p][1] is not None
            )
            if 2 * n_nonfounders > MAX_IBD_TRANSMISSIONS:
                continue
            ids = list(ped)
            a, b = rng.sample(ids, 2)
            res = ibd_sharing(ped, a, b)
            want, transmissions = ref_ibd(ped, a, b)
            self.assertEqual(res.shared_counts, want, f"seed={seed} {a}-{b}")
            self.assertEqual(res.transmissions, transmissions)
            self.assertEqual(sum(res.shared_counts.values()), 1)
            cases += 1
        self.assertGreater(cases, 40)


# ----------------------------------------------- 输入校验与边界

class TestIBDValidation(unittest.TestCase):
    def test_unknown_member_rejected(self):
        ped = {"A": (None, None)}
        with self.assertRaises(ValidationError):
            ibd_sharing(ped, "A", "Z")
        with self.assertRaises(ValidationError):
            ibd_sharing(ped, 1, "A")  # 非 str

    def test_cycle_still_rejected(self):
        ped = {"A": ("B", None), "B": ("A", None)}
        with self.assertRaises(ValidationError) as cm:
            ibd_sharing(ped, "A", "B")
        self.assertIn("cycle", str(cm.exception))

    def test_transmission_limit_boundary(self):
        # 6 个非创始子女 => 12 次传递，恰好允许；7 个 => 14 次，拒绝。
        ped_ok = {"F": (None, None), "M": (None, None)}
        for i in range(6):
            ped_ok[f"C{i}"] = ("F", "M")
        res = ibd_sharing(ped_ok, "C0", "C5")
        self.assertEqual(res.transmissions, MAX_IBD_TRANSMISSIONS)
        self.assertEqual(sum(res.shared_counts.values()), 1)

        ped_bad = dict(ped_ok)
        ped_bad["C6"] = ("F", "M")
        with self.assertRaises(ValidationError) as cm:
            ibd_sharing(ped_bad, "C0", "C6")
        self.assertIn("14", str(cm.exception))
        self.assertIn(str(MAX_IBD_TRANSMISSIONS), str(cm.exception))

    def test_empty_pedigree_rejected(self):
        with self.assertRaises(ValidationError):
            ibd_sharing({}, "A", "B")


if __name__ == "__main__":
    unittest.main()
