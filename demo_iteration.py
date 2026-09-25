"""单位点 IBD 查询固定输入演示：真实枚举计算，无 sleep、无预录输出。

展示：
1. 无亲缘、亲子、同胞三类基础关系的 0/1/2 份同祖拷贝概率；
2. 同胞交配近交家系：多条祖先路径下的精确概率（含近交纯合）；
3. 边界：0 次传递的创始人家系仍枚举成功；
4. 实际触发的拒绝边界：超过 12 次二选一传递的家系。

本演示是有限标签传递模型的教学/校验用途，不是真实个体遗传风险评估。
"""

from fractions import Fraction

from mendelcheck import IBDResult, ValidationError, ibd_sharing


def show(result: IBDResult, label: str) -> None:
    p = result.shared_counts
    print(
        f"  {label}: P(0)={p[0]}, P(1)={p[1]}, P(2)={p[2]} "
        f"（小数近似 {float(p[0]):.4f}/{float(p[1]):.4f}/{float(p[2]):.4f}，"
        f"枚举 {result.transmissions} 次传递、{1 << result.transmissions} 个组合）"
    )
    assert sum(p.values()) == 1


def show_basic_relationships():
    print("=== 1. 基础关系：无亲缘 / 亲子 / 同胞 ===")
    unrelated = {"A": (None, None), "B": (None, None)}
    show(ibd_sharing(unrelated, "A", "B"), "无亲缘创始人 A-B")

    trio = {"A": (None, None), "B": (None, None), "C": ("A", "B")}
    show(ibd_sharing(trio, "A", "C"), "亲子 A-C")

    sibs = {
        "A": (None, None), "B": (None, None),
        "C": ("A", "B"), "D": ("A", "B"),
    }
    show(ibd_sharing(sibs, "C", "D"), "同胞 C-D")


def show_inbreeding():
    print("\n=== 2. 近交例子：同胞交配（多条祖先路径，家系无有向环）===")
    ped = {
        "A": (None, None), "B": (None, None),
        "C": ("A", "B"), "D": ("A", "B"),
        "E": ("C", "D"),
    }
    show(ibd_sharing(ped, "E", "A"), "近交子代 E 与祖父 A")
    show(ibd_sharing(ped, "E", "C"), "近交子代 E 与父亲 C")
    show(ibd_sharing(ped, "C", "D"), "同胞父母 C-D")
    res = ibd_sharing(ped, "E", "E")
    print(
        f"  同一成员 E-E: P(2)={res.p2} "
        "（自己两份拷贝与自身完全匹配；近交纯合信息体现在与他人的分布中）"
    )
    # 独立核对：E 的两份拷贝同标签（近交纯合）的概率恰为 1/4
    assert ibd_sharing(ped, "E", "A").p2 == Fraction(1, 8)


def show_boundary_ok():
    print("\n=== 3. 正常边界：0 次传递（仅创始人）与恰好 12 次传递 ===")
    founders = {"X": (None, None), "Y": (None, None), "Z": (None, None)}
    res = ibd_sharing(founders, "X", "Z")
    print(f"  创始人家系 X-Z: {res.shared_counts}，transmissions={res.transmissions}")

    ped = {"F": (None, None), "M": (None, None)}
    for i in range(6):  # 6 个子女 * 2 次传递 = 12（上限）
        ped[f"C{i}"] = ("F", "M")
    res = ibd_sharing(ped, "C0", "C5")
    print(
        f"  12 次传递边界（{1 << 12} 个组合）: "
        f"P(0)={res.p0}, P(1)={res.p1}, P(2)={res.p2}"
    )


def show_rejection():
    print("\n=== 4. 实际触发的拒绝边界：14 次传递超过 12 次上限 ===")
    ped = {"F": (None, None), "M": (None, None)}
    for i in range(7):  # 7 个子女 * 2 = 14 次
        ped[f"C{i}"] = ("F", "M")
    try:
        ibd_sharing(ped, "C0", "C6")
    except ValidationError as exc:
        print(f"  ValidationError: {exc}")


if __name__ == "__main__":
    show_basic_relationships()
    show_inbreeding()
    show_boundary_ok()
    show_rejection()
