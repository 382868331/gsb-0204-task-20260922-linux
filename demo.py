"""固定输入演示：孟德尔家系最小冲突定位。

运行：python demo.py（约 1 秒内完成，无网络、无第三方依赖、无 sleep/预录结果）。

三个段落均为当场真实计算：
  1. 正常结果：三代家系相容，打印每个个体的全部可行基因型；
  2. 冲突定位：单亲缺失家系的多个最小解释 + 每个被移除观测的修正见证；
  3. 拒绝边界：观测值传入 NaN，实际触发 PedigreeError（可定位错误）。
"""

import time

from mendel_conflict import PedigreeError, check
from mendel_conflict.core import verify_witness

GT_NAME = {0: "aa(0)", 1: "Aa(1)", 2: "AA(2)"}


def family(rows):
    return {
        "individuals": [
            {"id": i, "father": f, "mother": m} for i, f, m in rows
        ]
    }


def section(title):
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def main():
    # ---- 1. 正常结果：三代家系，观测相容 ---------------------------------
    section("【1】正常结果：三代家系，全部观测相容")
    ped = family(
        [
            ("gf", None, None),   # 爷爷
            ("gm", None, None),   # 奶奶
            ("mf", None, None),   # 外公
            ("mm", None, None),   # 外婆
            ("F", "gf", "gm"),    # 父亲
            ("M", "mf", "mm"),    # 母亲
            ("C", "F", "M"),      # 孩子
        ]
    )
    # gf=gm=0 => F 只能 0；mf=mm=2 => M 只能 2；F×M => C 只能 1。
    obs = {"gf": 0, "gm": 0, "mf": 2, "mm": 2, "C": 1}
    t0 = time.perf_counter()
    result = check(ped, obs)
    elapsed = time.perf_counter() - t0
    print(f"status = {result.status}（搜索节点 {result.nodes_used}，耗时 {elapsed*1000:.1f} ms）")
    print("观测:", {k: GT_NAME[v] for k, v in obs.items()})
    print("每个个体的全部可行基因型：")
    for iid, vals in result.feasible_genotypes.items():
        print(f"  {iid:>3}: {', '.join(GT_NAME[v] for v in vals)}")

    # ---- 2. 冲突定位：单亲缺失，多个最小解释 + 修正见证 ------------------
    section("【2】冲突定位：父亲=AA、母亲缺失（独立创始者）、孩子=aa")
    ped2 = family([("f", None, None), ("c", "f", None)])
    obs2 = {"f": 2, "c": 0}  # AA 父亲必传 A，孩子不可能是 aa
    t0 = time.perf_counter()
    result2 = check(ped2, obs2)
    elapsed2 = time.perf_counter() - t0
    print(f"status = {result2.status}（搜索节点 {result2.nodes_used}，耗时 {elapsed2*1000:.1f} ms）")
    print(f"共 {len(result2.minimal_removals)} 个最小解释（基数最小，按 ID 字典序）：")
    for removal in result2.minimal_removals:
        print(f"\n  移除观测集合: {list(removal)}")
        for rid in removal:
            witness = result2.witnesses[(removal, rid)]
            old = obs2[rid]
            new = witness[rid]
            ok = verify_witness(ped2, obs2, witness, removed=frozenset(removal))
            print(f"    被移除观测 {rid}={GT_NAME[old]} → 修正见证建议 {rid}={GT_NAME[new]}")
            print(f"    完整见证: {{'f': {GT_NAME[witness['f']]}, 'c': {GT_NAME[witness['c']]}}}")
            print(f"    见证独立复核（三元组规则）: {'通过' if ok else '失败'}")

    # ---- 3. 拒绝边界：NaN 观测必须被拒绝，错误可定位 ---------------------
    section("【3】异常边界：观测值为 NaN（真实计算触发，非预录）")
    try:
        check(ped, {"gf": float("nan")})
    except PedigreeError as exc:
        print(f"捕获 {type(exc).__name__}（ValueError 子类）")
        print(f"  可定位字段 path   = {exc.path}")
        print(f"  机器原因 reason   = {exc.reason}")
        print(f"  错误信息          = {exc}")
    else:
        raise AssertionError("NaN 观测必须被拒绝")

    print()
    print("演示完成：1 个正常结果、1 组冲突修正见证、1 个实际触发的拒绝边界。")


if __name__ == "__main__":
    main()
