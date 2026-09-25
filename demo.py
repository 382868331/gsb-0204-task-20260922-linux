"""mendelcheck 固定输入演示：真实计算，无 sleep、无预录结果。

展示：
1. 正常可行家系 -> 全部个体的可行基因型集合；
2. 多代隐含冲突 -> 最小移除集、多个最小解释与修正见证；
3. 实际触发的拒绝边界（环状家系 -> ValidationError）；
4. 节点预算耗尽 -> 明确的 unknown（与证明无解分开）。
5. 软代价优化 -> 最小总代价、每人全部最优基因型与确定性见证；
6. 硬观测 + 禁止项冲突 -> 证明无解，不能用高代价偷偷放过。
"""

from mendelcheck import ValidationError, check, optimize


def show_feasible():
    print("=== 1. 正常结果：三代家系的可行基因型集合 ===")
    pedigree = {
        "G1": (None, None), "G2": (None, None),
        "P": ("G1", "G2"),
        "D": (None, None),
        "E": ("P", "D"),
    }
    observations = {"G1": 0, "G2": 2, "D": 1}
    res = check(pedigree, observations)
    print(f"status = {res.status} (nodes_used = {res.nodes_used})")
    for pid, vals in res.feasible_genotypes.items():
        print(f"  {pid}: 可行基因型 {sorted(vals)}")


def show_conflict():
    print("\n=== 2. 多代隐含冲突：最小移除集与修正见证 ===")
    # G1=G2=0 迫使 P=0；D=2；E 观测为 2 需要 P 传出等位 1 —— 跨两代才显现。
    pedigree = {
        "G1": (None, None), "G2": (None, None),
        "P": ("G1", "G2"),
        "D": (None, None),
        "E": ("P", "D"),
    }
    observations = {"G1": 0, "G2": 0, "D": 2, "E": 2}
    res = check(pedigree, observations)
    print(f"status = {res.status} (nodes_used = {res.nodes_used})")
    rep = res.conflict
    print(f"最少需移除 {len(rep.removals)} 个观测；最小解释共 "
          f"{1 + len(rep.alternatives)} 个：")
    for s in [rep.removals, *rep.alternatives]:
        print(f"  移除集 {list(s)}")
    print("每个被移除观测的重新可行见证（修正后完整指派）：")
    for obs_id, witness in rep.witnesses.items():
        old = observations[obs_id]
        new = witness[obs_id]
        mark = f"（{obs_id}: 观测 {old} -> 见证 {new}）" if old != new else ""
        print(f"  移除 {obs_id}: {witness} {mark}")


def show_rejection():
    print("\n=== 3. 实际触发的拒绝边界：环状家系 ===")
    cyclic = {"A": ("B", None), "B": ("C", None), "C": ("A", None)}
    try:
        check(cyclic, {})
    except ValidationError as exc:
        print(f"ValidationError: {exc}")


def show_unknown():
    print("\n=== 4. 节点预算耗尽 -> 明确 unknown（与证明无解分开）===")
    pedigree = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
    res = check(pedigree, {}, node_budget=1)
    print(f"status = {res.status} (node_budget = 1)")


def show_optimize():
    print("\n=== 5. 软代价优化：最小总代价与全部最优基因型 ===")
    # 三代家系；G1、D 有硬观测（严格），P、E 带软代价，G2 全部候选禁止。
    pedigree = {
        "G1": (None, None), "G2": (None, None),
        "P": ("G1", "G2"),
        "D": (None, None),
        "E": ("P", "D"),
    }
    observations = {"G1": 0, "D": 1}
    costs = {
        "P": (0, 2, 9),
        "E": (3, 0, 4),
        "G2": (None, 1, None),  # 禁止基因型 0 和 2
    }
    res = optimize(pedigree, observations, costs)
    print(f"status = {res.status} (nodes_used = {res.nodes_used})")
    print(f"最小总代价 min_cost = {res.min_cost}")
    for pid, vals in res.optimal_genotypes.items():
        print(f"  {pid}: 最优可行基因型 {sorted(vals)}")
    print(f"确定性最优见证（同分按 ID 字典序取最小）: {res.witness}")
    total = sum(costs.get(p, (0, 0, 0))[g] for p, g in res.witness.items())
    print(f"见证总代价复核 = {total}")


def show_optimize_conflict():
    print("\n=== 6. 硬观测与禁止项冲突：证明无解，高代价不能偷偷放过 ===")
    pedigree = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
    # A 硬观测为 0，但 costs 禁止 A 取 0 -> 不可行（不是“付高代价取 0”）
    res = optimize(pedigree, {"A": 0}, {"A": (None, 1, 1)})
    print(f"status = {res.status} (min_cost = {res.min_cost})")
    # 对照：允许取 0 但代价很高 -> 仍必须取 0
    res2 = optimize(pedigree, {"A": 0}, {"A": (99, 0, 0)})
    print(f"对照（允许高代价取 0）: status = {res2.status}, "
          f"witness = {res2.witness}, min_cost = {res2.min_cost}")


if __name__ == "__main__":
    show_feasible()
    show_conflict()
    show_rejection()
    show_unknown()
    show_optimize()
    show_optimize_conflict()
