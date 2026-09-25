"""ibd_sharing 固定输入演示：真实计算，无 sleep、无预录结果。

展示：
1. 正常结果：同胞与近交家系的 IBD 共享概率（Fraction 精确值）；
2. 边界一：传递次数恰为上限 12 的最大家系，真实枚举 4096 种组合；
3. 边界二：实际触发的拒绝（传递次数 14 > 12 -> ValidationError）。
"""

from mendelcheck import MAX_TRANSMISSIONS, ValidationError, ibd_sharing


def show_normal():
    print("--- 1. 正常结果：同胞与近交家系的 IBD 共享概率 ---")
    sibs = {
        "A": (None, None), "B": (None, None),
        "C": ("A", "B"), "D": ("A", "B"),
    }
    res = ibd_sharing(sibs, "C", "D")
    print(f"同胞 C-D：P(共享0)={res.p0}  P(共享1)={res.p1}  "
          f"P(共享2)={res.p2}  （枚举 {res.combinations} 种传递组合）")

    inbred = dict(sibs)
    inbred["E"] = ("C", "D")  # 同胞婚配：多条祖先路径，无有向环
    res = ibd_sharing(inbred, "E", "C")
    print(f"近交 E-C（E 为同胞 C、D 之子）：P0={res.p0}  P1={res.p1}  "
          f"P2={res.p2}  （枚举 {res.combinations} 种组合）")
    res = ibd_sharing(inbred, "E", "A")
    print(f"近交 E-A（孙与祖父，双路径）：P0={res.p0}  P1={res.p1}  P2={res.p2}")


def show_boundary_ok():
    print(f"\n--- 2. 边界：传递次数恰为上限 {MAX_TRANSMISSIONS} ---")
    ped = {"A": (None, None), "B": (None, None)}
    for i in range(6):
        ped[f"K{i}"] = ("A", "B")
    res = ibd_sharing(ped, "K0", "K5")
    print(f"6 同胞家系 K0-K5：P0={res.p0}  P1={res.p1}  P2={res.p2}  "
          f"（{res.transmissions} 次传递，真实枚举 {res.combinations} 种组合）")


def show_boundary_rejected():
    print(f"\n--- 3. 边界：传递次数超限（>{MAX_TRANSMISSIONS}）实际触发拒绝 ---")
    ped = {"A": (None, None), "B": (None, None)}
    for i in range(7):
        ped[f"K{i}"] = ("A", "B")
    try:
        ibd_sharing(ped, "K0", "K1")
    except ValidationError as exc:
        print(f"ValidationError: {exc}")


if __name__ == "__main__":
    show_normal()
    show_boundary_ok()
    show_boundary_rejected()
