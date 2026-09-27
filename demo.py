"""mendelcheck 固定输入演示：真实计算，无 sleep、无预录结果。

展示：
1. 正常可行家系 -> 全部个体的可行基因型集合；
2. 多代隐含冲突 -> 最小移除集、多个最小解释与修正见证；
3. 候选亲本枚举（父亲未定，仅可能来自给定名单）；
4. 实际触发的拒绝边界（环状家系 -> ValidationError）；
5. 节点预算耗尽 -> 明确的 unknown（与证明无解分开）。
"""

from mendelcheck import ValidationError, check, resolve_links


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


def show_candidate_father():
    print("\n=== 3. 候选亲本：父亲未定，只可能来自给定名单 ===")
    # X 的父亲尚未确定，但只可能是 A 或 B；母亲 M 已确定。
    pedigree = {
        "A": (None, None), "B": (None, None),
        "M": (None, None),
        "X": (["A", "B"], "M"),
    }
    # A=0、B=2、M=0、X=1：父亲必须能传出等位 1 -> 仅 B 相容（观测消歧）。
    observations = {"A": 0, "B": 2, "M": 0, "X": 1}
    res = resolve_links(pedigree, observations)
    print(f"status = {res.status} (nodes_used = {res.nodes_used})")
    print(f"不确定槽位 {res.slots}")
    for link in res.feasible:
        print(f"  相容完整选择 selection={link.selection}")
        print(f"    parents={link.parents}")
        print(f"    基因型见证={link.assignment}")
    pretty_possible = {
        c: {role: sorted(vals) for role, vals in by_role.items()}
        for c, by_role in res.possible.items()
    }
    print(f"  各槽位仍可能的候选: {pretty_possible}")
    # 改成让两个候选都无法满足的观测 -> 已确认不可能（而非预算问题）。
    impossible = resolve_links(pedigree, {"A": 0, "B": 0, "M": 0, "X": 2})
    print(f"  改为 X=2（A、B 均为0）-> status = {impossible.status}，"
          f"相容选择数 = {len(impossible.feasible)}")


def show_rejection():
    print("\n=== 4. 实际触发的拒绝边界：环状家系 ===")
    cyclic = {"A": ("B", None), "B": ("C", None), "C": ("A", None)}
    try:
        check(cyclic, {})
    except ValidationError as exc:
        print(f"ValidationError: {exc}")


def show_budget_unknown():
    print("\n=== 5. 节点预算耗尽 -> 明确 unknown（与证明无解分开）===")
    pedigree = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
    res = check(pedigree, {}, node_budget=1)
    print(f"status = {res.status} (node_budget = 1)")
    ped_choice = {"A": (None, None), "B": (None, None),
                  "M": (None, None), "X": (["A", "B"], "M")}
    res2 = resolve_links(ped_choice, {}, node_budget=1)
    print(f"resolve_links node_budget=1 -> status = {res2.status}（搜索未完成）")


if __name__ == "__main__":
    show_feasible()
    show_conflict()
    show_candidate_father()
    show_rejection()
    show_budget_unknown()
