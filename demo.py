"""mendelcheck 固定输入演示：真实计算，无 sleep、无预录结果。

展示：
1. 正常可行家系 -> 全部个体的可行基因型集合；
2. 多代隐含冲突 -> 最小移除集、多个最小解释与修正见证；
3. 实际触发的拒绝边界（环状家系 -> ValidationError）；
4. 节点预算耗尽 -> 明确的 unknown（与证明无解分开）；
5. assign_parents 正常结果：多位点下同一父母选择的可行分配；
6. assign_parents 边界：逐位点都能过、但同一选择必须跨全部位点 -> 冲突位点；
7. assign_parents 实际错误：孩子缺少位点观测 -> ValidationError。
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
    print("\n=== 5. assign_parents 正常结果：同一选择跨两位点的可行分配 ===")
    # X：l1=1、l2=1。候选 (A,B)=(0,2)@两位点：两位点子女都必为 1，可行；
    #                候选 (E,F)=(0,0)@l1：l1 子女必为 0，即被否决。
    # Y：l1=0、l2=2。候选 (E,F)=(0,0)@l1,(2,2)@l2：可行；
    #   候选 (None,E)：缺失父自由；l1 E=0 -> 子女∈{0,1} 含 0，
    #                            l2 E=2 -> 子女∈{1,2} 含 2，可行。
    # 故可行组合为 X->(A,B) 配 Y 的任一种，共 2 种。
    genotypes = {
        "X": {1: 1, 2: 1},
        "Y": {1: 0, 2: 2},
        "A": {1: 0, 2: 0}, "B": {1: 2, 2: 2},
        "E": {1: 0, 2: 2}, "F": {1: 0, 2: 2},
    }
    candidates = {
        "X": [("A", "B"), ("E", "F")],
        "Y": [("E", "F"), (None, "E")],
    }
    res = assign_parents(genotypes, candidates)
    print(f"status = {res.status}，位点 {list(res.loci)}，"
          f"可行分配 {len(res.assignments)} 种 (nodes_used = {res.nodes_used})：")
    for combo in res.assignments:
        desc = ", ".join(
            f"{a.child}->候选{a.pair_index}{list(a.pair)}" for a in combo
        )
        print(f"  {desc}")


def show_assignment_infeasible():
    print("\n=== 6. 边界：逐位点换父母能过，但同一选择必须跨全部位点 ===")
    # 对 0=(A,B)=0x0：l1 出 0 ✓，l2 出不了 2；
    # 对 1=(D,E)=2x2：l2 出 2 ✓，l1 出不了 0。
    genotypes = {
        "C": {1: 0, 2: 2},
        "A": {1: 0, 2: 0}, "B": {1: 0, 2: 0},
        "D": {1: 2, 2: 2}, "E": {1: 2, 2: 2},
    }
    candidates = {"C": [("A", "B"), ("D", "E")]}
    res = assign_parents(genotypes, candidates)
    print(f"status = {res.status}，每种被否决选择的冲突位点：")
    for combo in res.conflicts:
        a = combo[0]
        print(f"  候选{a.pair_index} {list(a.pair)} -> 冲突位点 {list(a.loci_conflicts)}")


def show_assignment_rejection():
    print("\n=== 7. assign_parents 实际错误：孩子缺少位点 2 的观测 ===")
    genotypes = {
        "C": {1: 0},
        "X": {1: 0, 2: 0}, "Y": {1: 0, 2: 0},
    }
    try:
        assign_parents(genotypes, {"C": [("X", "Y")]})
    except ValidationError as exc:
        print(f"ValidationError: {exc}")


if __name__ == "__main__":
    show_feasible()
    show_conflict()
    show_rejection()
    show_unknown()
    show_assignment_feasible()
    show_assignment_infeasible()
    show_assignment_rejection()
