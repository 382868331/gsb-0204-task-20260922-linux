"""独立穷举参考（不调用被测核心的任何求解逻辑）。

用于在小家系（<=7 人）上对照：
- 枚举 3^n 个完整基因型指派，独立重写孟德尔三元组规则；
- 独立计算“全部可行基因型集合”；
- 按移除集大小逐层枚举，独立求全部最小移除集与见证。

本模块只服务于测试，是期望值的唯一来源；被测核心只被调用，
其输出不参与构造期望值。
"""

from __future__ import annotations

from itertools import product
from typing import Optional

ALLELES = {0: (0,), 1: (0, 1), 2: (1,)}


def _triple_ok(father: Optional[int], mother: Optional[int], child: int) -> bool:
    fa = ALLELES[father] if father is not None else (0, 1)
    ma = ALLELES[mother] if mother is not None else (0, 1)
    for a in fa:
        for b in ma:
            if a + b == child:
                return True
    return False


def _normalized(pedigree: dict):
    inds = [(x["id"], x.get("father"), x.get("mother")) for x in pedigree["individuals"]]
    return inds


def _assignments_valid(inds, obs: dict, assignment: dict) -> bool:
    for iid, gt in obs.items():
        if assignment[iid] != gt:
            return False
    for iid, fa, ma in inds:
        f = assignment[fa] if fa is not None else None
        m = assignment[ma] if ma is not None else None
        if not _triple_ok(f, m, assignment[iid]):
            return False
    return True


def _all_solutions(inds, obs: dict):
    ids = [row[0] for row in inds]
    for vals in product((0, 1, 2), repeat=len(ids)):
        assignment = dict(zip(ids, vals))
        if _assignments_valid(inds, obs, assignment):
            yield assignment


def is_feasible(pedigree: dict, observations: dict) -> bool:
    inds = _normalized(pedigree)
    for _ in _all_solutions(inds, observations):
        return True
    return False


def feasible_sets(pedigree: dict, observations: dict) -> dict[str, frozenset[int]]:
    inds = _normalized(pedigree)
    sets = {row[0]: set() for row in inds}
    found = False
    for assignment in _all_solutions(inds, observations):
        found = True
        for iid, gt in assignment.items():
            sets[iid].add(gt)
    if not found:
        return {}
    return {k: frozenset(v) for k, v in sets.items()}


def minimum_removals(pedigree: dict, observations: dict):
    """返回 (排序后的最小移除集列表, {移除集: 见证指派})。"""
    inds = _normalized(pedigree)
    ids = sorted(observations)
    for size in range(0, len(ids) + 1):
        from itertools import combinations

        hits: list[tuple[str, ...]] = []
        witnesses: dict[tuple[str, ...], dict] = {}
        for combo in combinations(ids, size):
            kept = {k: observations[k] for k in ids if k not in combo}
            for assignment in _all_solutions(inds, kept):
                key = tuple(sorted(combo))
                hits.append(key)
                witnesses[key] = assignment
                break
        if hits:
            return sorted(hits), witnesses
    raise AssertionError("unreachable")


def witness_check(pedigree: dict, observations: dict, witness: dict, removed) -> bool:
    """独立验证一份见证（与核心 verify_witness 互相独立的实现）。"""
    inds = _normalized(pedigree)
    ids = {row[0] for row in inds}
    if set(witness) != ids:
        return False
    if any(gt not in (0, 1, 2) for gt in witness.values()):
        return False
    removed = set(removed)
    for iid, gt in observations.items():
        if iid in removed:
            continue
        if witness[iid] != gt:
            return False
    for iid, fa, ma in inds:
        f = witness[fa] if fa is not None else None
        m = witness[ma] if ma is not None else None
        if not _triple_ok(f, m, witness[iid]):
            return False
    return True
