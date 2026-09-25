"""ibd_sharing 的单元测试。

参考实现 ref_ibd 为独立暴力枚举：对全部 2**T 种传递组合逐一传播
标签并计算多重集最大匹配，绝不调用被测核心 mendelcheck 得到期望值。
"""

import itertools
import os
import sys
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mendelcheck import IBDResult, ValidationError, ibd_sharing


# ------------------------------------------------- 独立参考实现（暴力枚举）

def ref_ibd(pedigree, id1, id2):
    """独立枚举所有传递组合，返回 (P0, P1, P2) 与组合数。"""
    # 拓扑序（Kahn，父母先于子女）
    indegree = {pid: 0 for pid in pedigree}
    children = {pid: [] for pid in pedigree}
    for cid, (fa, mo) in pedigree.items():
        for par in (fa, mo):
            if par is not None:
                indegree[cid] += 1
                children[par].append(cid)
    queue = [pid for pid in pedigree if indegree[pid] == 0]
    order = []
    while queue:
        node = queue.pop()
        order.append(node)
        for ch in children[node]:
            indegree[ch] -= 1
            if indegree[ch] == 0:
                queue.append(ch)
    assert len(order) == len(pedigree)

    slots = [(cid, s) for cid, (fa, mo) in pedigree.items()
             for s, p in enumerate((fa, mo)) if p is not None]
    counts = [0, 0, 0]
    for picks in itertools.product((0, 1), repeat=len(slots)):
        choice = dict(zip(slots, picks))
        labels = {}
        next_label = [0]

        def fresh():
            v = next_label[0]
            next_label[0] += 1
            return v

        for cid in order:
            fa, mo = pedigree[cid]
            if fa is None and mo is None:
                labels[cid] = (fresh(), fresh())
                continue
            pair = []
            for slot, par in enumerate((fa, mo)):
                if par is None:
                    pair.append(fresh())
                else:
                    pair.append(labels[par][choice[(cid, slot)]])
            labels[cid] = tuple(pair)
        # 多重集最大匹配（贪心，2 元素足够）
        b = list(labels[id2])
        shared = 0
        for x in labels[id1]:
            if x in b:
                b.remove(x)
                shared += 1
        counts[shared] += 1
    total = 2 ** len(slots)
    return tuple(Fraction(c, total) for c in counts), total


def assert_matches_ref(testcase, pedigree, id1, id2):
    res = ibd_sharing(pedigree, id1, id2)
    (r0, r1, r2), total = ref_ibd(pedigree, id1, id2)
    testcase.assertEqual((res.p0, res.p1, res.p2), (r0, r1, r2))
    testcase.assertEqual(res.combinations, total)
    testcase.assertEqual(res.p0 + res.p1 + res.p2, Fraction(1))
    return res


# ------------------------------------------------- 家系

UNRELATED = {"A": (None, None), "B": (None, None)}

PARENT_CHILD = {"A": (None, None), "C": ("A", None)}

SIBS = {
    "A": (None, None), "B": (None, None),
    "C": ("A", "B"), "D": ("A", "B"),
}

# 近交：同胞 C、D 婚配生 E（允许多条祖先路径，但无有向环）
INBRED = {
    "A": (None, None), "B": (None, None),
    "C": ("A", "B"), "D": ("A", "B"),
    "E": ("C", "D"),
}


class TestKnownValues(unittest.TestCase):
    def test_unrelated(self):
        res = assert_matches_ref(self, UNRELATED, "A", "B")
        self.assertEqual((res.p0, res.p1, res.p2),
                         (Fraction(1), Fraction(0), Fraction(0)))
        self.assertEqual(res.transmissions, 0)

    def test_parent_child(self):
        res = assert_matches_ref(self, PARENT_CHILD, "A", "C")
        self.assertEqual((res.p0, res.p1, res.p2),
                         (Fraction(0), Fraction(1), Fraction(0)))

    def test_siblings(self):
        res = assert_matches_ref(self, SIBS, "C", "D")
        self.assertEqual((res.p0, res.p1, res.p2),
                         (Fraction(1, 4), Fraction(1, 2), Fraction(1, 4)))

    def test_self(self):
        res = assert_matches_ref(self, SIBS, "C", "C")
        self.assertEqual(res.p2, Fraction(1))

    def test_inbred_parent_child(self):
        # E 必从 C 继承一份 -> 至少共享 1；精确分布由独立枚举对照。
        res = assert_matches_ref(self, INBRED, "E", "C")
        self.assertEqual(res.p0, Fraction(0))
        self.assertEqual((res.p1, res.p2), (Fraction(3, 4), Fraction(1, 4)))

    def test_inbred_grandparent(self):
        res = assert_matches_ref(self, INBRED, "E", "A")
        self.assertEqual(res.p0 + res.p1 + res.p2, Fraction(1))

    def test_half_sibs(self):
        ped = {"A": (None, None), "B": (None, None), "D": (None, None),
               "C": ("A", "B"), "E": ("A", "D")}
        res = assert_matches_ref(self, ped, "C", "E")
        self.assertEqual((res.p0, res.p1, res.p2),
                         (Fraction(1, 2), Fraction(1, 2), Fraction(0)))


class TestFractionExactness(unittest.TestCase):
    def test_probabilities_are_fractions(self):
        res = ibd_sharing(SIBS, "C", "D")
        for p in (res.p0, res.p1, res.p2):
            self.assertIsInstance(p, Fraction)

    def test_result_fields(self):
        res = ibd_sharing(SIBS, "C", "D")
        self.assertIsInstance(res, IBDResult)
        self.assertEqual((res.id1, res.id2), ("C", "D"))
        self.assertEqual(res.transmissions, 4)
        self.assertEqual(res.combinations, 16)


class TestValidation(unittest.TestCase):
    def test_unknown_member(self):
        with self.assertRaises(ValidationError):
            ibd_sharing(SIBS, "C", "ZZ")
        with self.assertRaises(ValidationError):
            ibd_sharing(SIBS, 123, "C")

    def test_cycle_rejected(self):
        ped = {"A": ("B", None), "B": ("C", None), "C": ("A", None)}
        with self.assertRaises(ValidationError):
            ibd_sharing(ped, "A", "B")

    def test_bad_parent_reference(self):
        ped = {"A": ("ZZ", None)}
        with self.assertRaises(ValidationError):
            ibd_sharing(ped, "A", "A")

    def test_too_many_transmissions(self):
        # 7 个子女 x 2 次传递 = 14 > 12
        ped = {"A": (None, None), "B": (None, None)}
        for i in range(7):
            ped[f"K{i}"] = ("A", "B")
        with self.assertRaises(ValidationError):
            ibd_sharing(ped, "K0", "K1")

    def test_exactly_twelve_transmissions_ok(self):
        ped = {"A": (None, None), "B": (None, None)}
        for i in range(6):
            ped[f"K{i}"] = ("A", "B")
        res = ibd_sharing(ped, "K0", "K1")
        self.assertEqual(res.transmissions, 12)
        self.assertEqual(res.combinations, 4096)
        self.assertEqual(res.p0 + res.p1 + res.p2, Fraction(1))


if __name__ == "__main__":
    unittest.main()
