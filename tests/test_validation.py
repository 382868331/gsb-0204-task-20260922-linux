"""输入校验、错误语义与预算（unknown 与证明无解严格分开）测试。"""

import math
import unittest

from mendel_conflict import (
    STATUS_FEASIBLE,
    STATUS_INCONSISTENT,
    STATUS_UNKNOWN,
    PedigreeError,
    check,
)
from mendel_conflict.core import triple_is_mendelian


def ped(rows):
    return {
        "individuals": [
            {"id": i, "father": f, "mother": m} for i, f, m in rows
        ]
    }


TRIO = ped([("f", None, None), ("m", None, None), ("c", "f", "m")])


class TestInputValidation(unittest.TestCase):
    def test_bad_pedigree_type(self):
        with self.assertRaises(PedigreeError) as cm:
            check([("f", None, None)])
        self.assertEqual(cm.exception.reason, "bad_type")

    def test_individuals_must_be_list(self):
        with self.assertRaises(PedigreeError) as cm:
            check({"individuals": {}})
        self.assertEqual(cm.exception.path, "individuals")

    def test_bad_individual_record(self):
        with self.assertRaises(PedigreeError):
            check({"individuals": ["f"]})

    def test_empty_id(self):
        with self.assertRaises(PedigreeError) as cm:
            check({"individuals": [{"id": "", "father": None, "mother": None}]})
        self.assertEqual(cm.exception.path, "individuals[0].id")

    def test_duplicate_id(self):
        with self.assertRaises(PedigreeError) as cm:
            check(
                {
                    "individuals": [
                        {"id": "f", "father": None, "mother": None},
                        {"id": "f", "father": None, "mother": None},
                    ]
                }
            )
        self.assertEqual(cm.exception.reason, "duplicate_id")

    def test_unknown_parent(self):
        with self.assertRaises(PedigreeError) as cm:
            check({"individuals": [{"id": "c", "father": "ghost", "mother": None}]})
        self.assertEqual(cm.exception.reason, "unknown_parent")
        self.assertIn("father", cm.exception.path)

    def test_self_parent(self):
        with self.assertRaises(PedigreeError) as cm:
            check({"individuals": [{"id": "c", "father": "c", "mother": None}]})
        self.assertEqual(cm.exception.reason, "self_parent")

    def test_same_father_mother_rejected(self):
        p = ped(
            [
                ("a", None, None),
                ("c", "a", "a"),
            ]
        )
        with self.assertRaises(PedigreeError) as cm:
            check(p)
        self.assertEqual(cm.exception.reason, "same_parents")

    def test_cyclic_pedigree_detected(self):
        # a 父为 b，b 父为 a（声明顺序无关，靠环检测）
        p = {
            "individuals": [
                {"id": "a", "father": "b", "mother": None},
                {"id": "b", "father": "a", "mother": None},
            ]
        }
        with self.assertRaises(PedigreeError) as cm:
            check(p)
        self.assertEqual(cm.exception.reason, "cyclic_pedigree")

    def test_indirect_cycle_detected(self):
        p = {
            "individuals": [
                {"id": "a", "father": "c", "mother": None},
                {"id": "b", "father": "a", "mother": None},
                {"id": "c", "father": "b", "mother": None},
            ]
        }
        with self.assertRaises(PedigreeError) as cm:
            check(p)
        self.assertEqual(cm.exception.reason, "cyclic_pedigree")

    def test_too_many_individuals(self):
        rows = [(f"I{i:02d}", None, None) for i in range(21)]
        with self.assertRaises(PedigreeError) as cm:
            check(ped(rows))
        self.assertEqual(cm.exception.reason, "too_many_individuals")

    def test_empty_string_parent_rejected_not_silently_null(self):
        p = {"individuals": [{"id": "c", "father": "", "mother": None}]}
        with self.assertRaises(PedigreeError) as cm:
            check(p)
        self.assertEqual(cm.exception.reason, "bad_parent")

    def test_bad_parent_type(self):
        p = {"individuals": [{"id": "c", "father": 3, "mother": None}]}
        with self.assertRaises(PedigreeError) as cm:
            check(p)
        self.assertEqual(cm.exception.reason, "bad_parent")

    def test_observation_unknown_individual(self):
        with self.assertRaises(PedigreeError) as cm:
            check(TRIO, {"ghost": 1})
        self.assertEqual(cm.exception.reason, "unknown_individual")

    def test_observation_float_rejected(self):
        for bad in (1.0, 1.5, float("nan"), float("inf"), -float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(PedigreeError) as cm:
                    check(TRIO, {"f": bad})
                self.assertEqual(cm.exception.reason, "bad_genotype")

    def test_observation_bool_rejected(self):
        with self.assertRaises(PedigreeError) as cm:
            check(TRIO, {"f": True})
        self.assertEqual(cm.exception.reason, "bad_genotype")

    def test_observation_out_of_range(self):
        for bad in (-1, 3, 100):
            with self.subTest(bad=bad):
                with self.assertRaises(PedigreeError):
                    check(TRIO, {"f": bad})

    def test_observations_bad_type(self):
        with self.assertRaises(PedigreeError):
            check(TRIO, [("f", 1)])

    def test_budget_rejects_bool_zero_negative_float(self):
        for bad in (True, False, 0, -1, 1.0, math.nan):
            with self.subTest(bad=bad):
                with self.assertRaises(PedigreeError):
                    check(TRIO, {"f": 0}, node_budget=bad)

    def test_error_is_value_error(self):
        with self.assertRaises(ValueError):
            check({"individuals": [{"id": "c", "father": 9, "mother": None}]})

    def test_empty_pedigree_is_vacuously_feasible(self):
        r = check({"individuals": []})
        self.assertEqual(r.status, STATUS_FEASIBLE)
        self.assertEqual(dict(r.feasible_genotypes), {})

    def test_inputs_are_not_mutated(self):
        import copy

        p = copy.deepcopy(TRIO)
        obs = {"f": 2, "m": 2, "c": 0}
        obs_copy = copy.deepcopy(obs)
        check(p, obs)
        check(p, {"f": 0})
        self.assertEqual(p, copy.deepcopy(TRIO))
        self.assertEqual(obs, obs_copy)


class TestBudgetSemantics(unittest.TestCase):
    def test_tiny_budget_gives_unknown_not_inconsistent(self):
        # 预算 1：只够根节点，绝不能误报 inconsistent。
        p = ped([("f", None, None), ("m", None, None), ("c", "f", "m")])
        r = check(p, {"f": 2, "m": 2, "c": 0}, node_budget=1)
        self.assertEqual(r.status, STATUS_UNKNOWN)
        self.assertTrue(r.budget_exhausted)
        self.assertEqual(r.nodes_used, 1)

    def test_proven_inconsistent_never_unknown(self):
        # 即使预算很小，只要在预算内完成了无解证明与枚举，就是 inconsistent。
        p = ped([("f", None, None), ("m", None, None), ("c", "f", "m")])
        r = check(p, {"f": 2, "m": 2, "c": 0}, node_budget=50)
        self.assertEqual(r.status, STATUS_INCONSISTENT)

    def test_enough_budget_feasible(self):
        r = check(TRIO, {"f": 0, "m": 2, "c": 1})
        self.assertEqual(r.status, STATUS_FEASIBLE)
        self.assertGreaterEqual(r.nodes_used, 1)
        self.assertLessEqual(r.nodes_used, r.node_budget)

    def test_unknown_does_not_return_partial_removals(self):
        r = check(
            ped([("f", None, None), ("m", None, None), ("c", "f", "m")]),
            {"f": 2, "m": 2, "c": 0},
            node_budget=1,
        )
        self.assertEqual(r.minimal_removals, ())
        self.assertEqual(dict(r.witnesses), {})

    def test_two_unknown_phases_are_distinguished_in_note(self):
        trio = ped([("f", None, None), ("m", None, None), ("c", "f", "m")])
        # 可行问题预算耗尽：尚未判定相容性
        r1 = check(trio, {"f": 1}, node_budget=1)
        self.assertEqual(r1.status, STATUS_UNKNOWN)
        self.assertIn("整体可行性", r1.note)
        # 已证无解，但枚举最小解释时耗尽：note 明确区别
        r2 = check(trio, {"f": 2, "m": 2, "c": 0}, node_budget=1)
        self.assertEqual(r2.status, STATUS_UNKNOWN)
        self.assertIn("最小移除集", r2.note)
        self.assertNotEqual(r1.note, r2.note)


class TestTripleRule(unittest.TestCase):
    def test_table(self):
        self.assertTrue(triple_is_mendelian(0, 2, 1))
        self.assertFalse(triple_is_mendelian(2, 2, 0))
        self.assertFalse(triple_is_mendelian(0, 0, 1))
        self.assertTrue(triple_is_mendelian(1, 1, 0))
        self.assertTrue(triple_is_mendelian(1, 1, 1))
        self.assertTrue(triple_is_mendelian(1, 1, 2))
        # 缺失亲本：创始者可传任一等位
        self.assertTrue(triple_is_mendelian(2, None, 1))
        self.assertFalse(triple_is_mendelian(2, None, 0))
        self.assertTrue(triple_is_mendelian(0, None, 0))
        self.assertTrue(triple_is_mendelian(None, None, 2))


if __name__ == "__main__":
    unittest.main()
