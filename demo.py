"""mendelcheck 固定输入演示：真实计算，无 sleep、无预录结果。

展示：
1. 正常可行家系 -> 全部个体的可行基因型集合；
2. 多代隐含冲突 -> 最小移除集、多个最小解释与修正见证；
3. 实际触发的拒绝边界（环状家系 -> ValidationError）；
4. 节点预算耗尽 -> 明确的 unknown（与证明无解分开）；
5. 多位点候选父母判定 -> 可行的父母分配（同一选择贯穿全部位点）；
6. 候选父母冲突 -> 每个联合分配的逐位点诊断（逐位点换父母也救不了）；
7. 候选父母输入的实际拒绝（重复候选对 -> ValidationError）。
"""

from mendelcheck import ValidationError, assign_parents, check


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


def show_assignment_feasible():
    print("\n=== 5. 多位点候选父母判定：唯一可行联合分配（2 孩子 × 2 位点）===")
    # 手算（父母各传一等位，g=0 只传 0、g=2 只传 1、g=1 两者皆可）：
    #   C 的候选 (F,M)：L1 0x2 -> 子只能为 1；L2 2x0 -> 子只能为 1。
    #                   C=(1,1) 两位点都相容。
    #   C 的候选 (G,H)：L1 0x0 -> 子只能为 0，与 C=1 冲突，被排除。
    #   D 的候选 (R,S)：L1 0x2 -> 子只能为 1；L2 2x2 -> 子只能为 2。
    #                   D=(1,2) 两位点都相容。
    #   D 的候选 (F,M)：L2 2x0 -> 子只能为 1，与 D=2 冲突，被排除。
    # 故联合枚举 2x2=4 个分配中只有 {C:(F,M), D:(R,S)} 可行。
    profiles = {
        "F": {"L1": 0, "L2": 2}, "M": {"L1": 2, "L2": 0},
        "G": {"L1": 0, "L2": 2}, "H": {"L1": 0, "L2": 2},
        "R": {"L1": 0, "L2": 2}, "S": {"L1": 2, "L2": 2},
        "C": {"L1": 1, "L2": 1},
        "D": {"L1": 1, "L2": 2},
    }
    candidates = {
        "C": [("F", "M"), ("G", "H")],
        "D": [("R", "S"), ("F", "M")],
    }
    res = assign_parents(profiles, candidates)
    print(f"status = {res.status}，可行联合分配 {len(res.feasible_assignments)} 个"
          f"（nodes_used = {res.nodes_used}）：")
    for a in res.feasible_assignments:
        print(f"  C -> {a['C']}，D -> {a['D']}")


def show_assignment_conflict():
    print("\n=== 6. 逐位点换父母也救不了：每个联合分配的冲突位点 ===")
    # 对 (F1,M1)：L1 0x2 -> 子只能为 1（C=1 相容）；L2 2x2 -> 子只能为 2
    # （C=0 冲突）。
    # 对 (F2,M2)：L1 2x2 -> 子只能为 2（C=1 冲突）；L2 0x0 -> 子只能为 0
    # （C=0 相容）。
    # 同一对必须贯穿两位点 -> 没有任何分配可行。
    profiles = {
        "F1": {"L1": 0, "L2": 2}, "M1": {"L1": 2, "L2": 2},
        "F2": {"L1": 2, "L2": 0}, "M2": {"L1": 2, "L2": 0},
        "C": {"L1": 1, "L2": 0},
    }
    candidates = {"C": [("F1", "M1"), ("F2", "M2")]}
    res = assign_parents(profiles, candidates)
    print(f"status = {res.status}（nodes_used = {res.nodes_used}）")
    for c in res.conflicts:
        print(f"  C -> {c.assignment['C']}：相容位点 {list(c.feasible_loci)}，"
              f"冲突位点 {list(c.conflicting_loci)}")


def show_assignment_rejection():
    print("\n=== 7. 候选父母输入的实际拒绝：重复候选对 ===")
    profiles = {"A": {"L": 0}, "B": {"L": 2}, "C": {"L": 1}}
    try:
        assign_parents(profiles, {"C": [("A", "B"), ("A", "B")]})
    except ValidationError as exc:
        print(f"ValidationError: {exc}")


if __name__ == "__main__":
    show_feasible()
    show_conflict()
    show_rejection()
    show_unknown()
    show_assignment_feasible()
    show_assignment_conflict()
    show_assignment_rejection()
