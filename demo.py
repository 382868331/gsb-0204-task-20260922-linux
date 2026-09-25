"""mendelcheck 固定输入演示：真实计算，无 sleep、无预录结果。

展示：
1. 正常可行家系 -> 全部个体的可行基因型集合；
2. 代价扩展 -> 最小总代价、每人的全部最优基因型与字典序确定见证；
3. 同分多解与代价反转；
4. 硬观测与禁止冲突 -> 最小移除集（禁止/高代价不能放行观测）；
5. 全部候选禁止 -> 结构性 infeasible；预算耗尽 -> 可行上界 unknown；
6. 实际触发的拒绝边界（环状家系 -> ValidationError）。
"""

from mendelcheck import ValidationError, check


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
    print("\n=== 2. （旧功能）多代隐含冲突：最小移除集与修正见证 ===")
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
    print("\n=== 6. 实际触发的拒绝边界：环状家系 ===")
    cyclic = {"A": ("B", None), "B": ("C", None), "C": ("A", None)}
    try:
        check(cyclic, {})
    except ValidationError as exc:
        print(f"ValidationError: {exc}")


def show_unknown():
    print("\n=== 5b. 节点预算耗尽 -> unknown；带可行上界但不宣称最优 ===")
    pedigree = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
    costs = {"A": [3, 1, 4], "B": [1, 5, 9], "C": [2, 6, 0]}
    res = check(pedigree, {}, costs=costs, node_budget=4)
    print(f"status = {res.status} (node_budget = 4)")
    if res.witness is not None:
        print(f"  仅为可行上界见证 {res.witness}，上界总代价 = {res.witness_cost}")
        print("  （optimal_genotypes 为 None：未证明，绝不列“所有最优”）")


def show_costs():
    print("\n=== 3. 代价扩展：最小总代价、全部最优基因型与确定见证 ===")
    # 两位父母信息缺失（独立未知创始者）；E 的三种基因型代价不同。
    pedigree = {"M": (None, None), "F": (None, None), "E": ("F", "M")}
    costs = {
        "M": [0, 2, 5],
        "F": [5, 2, 0],
        "E": [10, 0, 10],
    }
    res = check(pedigree, {}, costs=costs)
    print(f"status = {res.status} (nodes_used = {res.nodes_used})")
    print(f"已证明最小总代价 optimal_cost = {res.optimal_cost}")
    for pid in sorted(pedigree):
        print(f"  {pid} 的全部最优基因型: {sorted(res.optimal_genotypes[pid])}")
    print(f"确定见证（ID 字典序最小最优指派）: {res.witness}")

    print("\n  -- 同分多解：A/B 取 0 或 1 代价相同 --")
    ped2 = {"A": (None, None), "B": (None, None)}
    costs2 = {"A": [0, 0, 7], "B": [0, 0, 7]}
    res2 = check(ped2, {}, costs=costs2)
    print(f"  optimal_cost = {res2.optimal_cost}；A 最优值 "
          f"{sorted(res2.optimal_genotypes['A'])}；见证 {res2.witness}")

    print("\n  -- 代价改变后最优反转：C 从偏好 0 翻转为偏好 2 --")
    ped3 = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
    r3a = check(ped3, {}, costs={"C": [0, 9, 9]})
    r3b = check(ped3, {}, costs={"C": [9, 9, 0]})
    print(f"  代价表 [0,9,9] -> 见证 {r3a.witness}（C={r3a.witness['C']}）")
    print(f"  代价表 [9,9,0] -> 见证 {r3b.witness}（C={r3b.witness['C']}）")


def show_forbidden():
    print("\n=== 4. 禁止（None）与硬观测的交互 ===")
    pedigree = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
    # C 的硬观测是 1，但代价表禁止 1；只能移除该观测，不能高代价放行。
    obs = {"C": 1}
    costs = {"C": [0, None, 0]}
    res = check(pedigree, obs, costs=costs)
    print(f"status = {res.status}；最小移除集 = {list(res.conflict.removals)}")
    print(f"移除后的最小代价见证: {res.conflict.witnesses['C']}")
    print("（硬观测先限制候选再加代价：禁止项不会被当作高代价选项放过）")

    print("\n=== 5a. 全部候选禁止 -> 结构性 infeasible（无需移除观测）===")
    res2 = check({"X": (None, None)}, {}, costs={"X": [None, None, None]})
    print(f"status = {res2.status}；reason = {res2.infeasible_reason}")


if __name__ == "__main__":
    show_feasible()
    show_conflict()
    show_costs()
    show_forbidden()
    show_rejection()
    show_unknown()
