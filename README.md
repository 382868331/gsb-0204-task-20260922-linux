# mendel_conflict — 孟德尔家系的最小冲突定位

单二等位位点（biallelic, single locus）的家系基因型校验库：判断一组基因型观测是否
符合孟德尔遗传；相容时给出**每个个体的全部可行基因型**；不相容时找出**基数最小、
需移除的观测集合（全部最小解释）**，并为每个被移除观测提供一份移除后恢复可行的
**修正见证（witness）**。搜索预算耗尽时明确返回 `unknown`，与“证明无解”严格区分。

- **Python 版本**：在 Python **3.14.7**（Linux）上开发与验证；仅使用标准库，
  语法层面兼容 3.10+。
- **依赖**：无第三方依赖；安装 Python 后直接运行，不需要网络、账号、密钥、
  数据库、中间件、Docker 或任何硬件。
- 不调用任何第三方同类求解实现；核心算法（传播、回溯、命中集枚举）全部为本库自写。

## 运行命令

```bash
# 必要测试（结构化用例 + 随机小家系对拍独立穷举参考 + 输入校验 + 预算语义）
python -m unittest discover -s tests -v

# 固定输入演示（约 0.1 秒，远低于 8 秒；当场真实计算，无 sleep、无预录结果）
python demo.py
```

## 数据结构与接口

```python
from mendel_conflict import check, PedigreeError
from mendel_conflict import STATUS_FEASIBLE, STATUS_INCONSISTENT, STATUS_UNKNOWN

pedigree = {
    "individuals": [
        {"id": "f", "father": None, "mother": None},   # 缺失亲本用 null
        {"id": "m", "father": None, "mother": None},
        {"id": "c", "father": "f",  "mother": "m"},
    ]
}
observations = {"f": 0, "m": 2, "c": 1}   # 0/1/2 = 替代等位计数；可省略
result = check(pedigree, observations, node_budget=100_000)
```

### 基因型编码

| 值 | 含义（A 为替代等位） |
|----|----------------------|
| 0  | aa（参考纯合）        |
| 1  | Aa（杂合）            |
| 2  | AA（替代纯合）        |

遗传规则：父母各传一个等位，和为子女计数（0×0→只能 0；0×2→只能 1；2×2→只能 2；
1×1→0/1/2 均可，余此类推）。

### 家系规则

- 至多 **20** 个个体（接口硬上限，超出报 `too_many_individuals`）。
- `father`/`mother` 为个体 ID 字符串或 `null`；`null` 表示**缺失亲本**，
  其位置视为**独立未知创始者**（可自由传出任一等位，不与任何人共享来源；
  即使多个单亲子女都缺母亲，这些缺失位置也互不相关）。
- 所有被引用的亲本必须在 `individuals` 中声明；亲子有向图**必须无环**
  （DFS 三色检测，环报 `cyclic_pedigree`，消息含环路径）。
- 允许近亲婚配（半同胞、共同祖先等），此时约束无向图可有环，算法不依赖
  “图为树”的假设。
- 同一人不能同时是 father 和 mother；不能以自己为亲本。

### `CheckResult` 字段

| 字段 | 说明 |
|------|------|
| `status` | `"feasible"` / `"inconsistent"` / `"unknown"` |
| `feasible` / `inconsistent` / `budget_exhausted` | 状态布尔属性 |
| `feasible_genotypes` | feasible：`{个体ID: (可行基因型…)}`，键与值均排序；**逐值试解得出的完整集合**（近亲图下弧一致不充分，故不以传播域代替） |
| `minimal_removals` | inconsistent：全部**基数最小**移除集，每个为按 ID 字典序排列的元组，移除集之间也按字典序 |
| `witnesses` | inconsistent：`{(移除集元组, 被移除观测ID): {全部个体ID: 基因型}}`；该指派满足全部亲子三元组与所有**未被移除**的观测，且被移除观测处的取值与原观测不同 |
| `witness_for(removed_id, removal_set=None)` | 便捷取见证；同一观测出现在多个解释中时需指定 `removal_set` |
| `nodes_used` / `node_budget` | 实际使用/允许的分支节点数 |
| `note` | 人类可读说明；unknown 时区分“整体可行性阶段耗尽”与“已证无解但最小解释枚举阶段耗尽” |

### 三种状态的严格语义

1. **feasible**：找到完整见证（交付前用三元组规则独立复核），且可行基因型集合枚举完整。
2. **inconsistent**：搜索在预算内**证明无解**，并在预算内完成全部最小移除集枚举与见证构造。
3. **unknown**：预算耗尽。**绝不**把预算耗尽报成无解；也不返回半截的可行集合或
   半截解释（无部分变更）。两种 unknown 阶段在 `note` 中明确区分。

`inconsistent` 的“最小”按**移除观测的基数**（不是按数值偏差）。

## 错误语义（可定位、不静默纠正）

非法输入抛 `PedigreeError`（`ValueError` 子类），带：

- `path`：可定位字段路径，如 `individuals[3].father`、`observations['c']`、`node_budget`；
- `reason`：机器可读原因码（`bad_type` / `bad_id` / `duplicate_id` / `bad_parent` /
  `unknown_parent` / `self_parent` / `same_parents` / `cyclic_pedigree` /
  `too_many_individuals` / `unknown_individual` / `bad_genotype` / `bad_budget`）。

数值与布尔规则：

- 基因型只接受整数 `0/1/2`：**拒绝 `bool`**（`True`/`False`）、拒绝浮点
  （含 `1.0`）、拒绝 `NaN`/`+Infinity`/`-Infinity`、拒绝越界整数；
- `node_budget` 只接受正整数，同样拒绝 `bool`、浮点、0、负数；
- 空字符串亲本不会被静默当作 `null`，报 `bad_parent`；
- 观测引用了不在家系中的 ID 报 `unknown_individual`。

本库**没有浮点输出**（所有结果都是 0/1/2 整数指派与离散集合），因此不存在浮点
误差口径/容差；任何比较均为精确整数比较，不以宽容差掩盖边界。

## 设计取舍

1. **传播**：对每个亲子三元组（child, father, mother）维护域 `⊆ {0,1,2}`，
   按“亲本可传等位并集 → 子女可能和”联合剪枝三方域，队列传播到不动点。
   缺失亲本不参与剪枝（创始者可传任一等位）。
2. **搜索**：MRV（最小剩余域）分支，每次赋值一个基因型并重新传播；单例节点
   还会用逐条三元组规则校验最终见证。预算按分支节点计数。
3. **全部可行基因型**：对每个个体的每个候选值单独试解；这样在近亲（约束图有
   无向环）情形下依然精确——关系弧一致对有环图不是完备推断。
4. **最小移除集**：先求一个**删除极小的不可满足观测核 C**（逐个尝试删观测，
   删后仍不可行就永久删除）；任何能恢复可行性的移除集都必须命中 C
   （否则 C 全部保留、仍不可行）。再按基数升层枚举命中 C 的候选并严格验证
   整问题可行性；首个有解层即为全部最小基数解释，子问题解做缓存。
5. **见证可信**：每份见证在返回前用独立于搜索的 `verify_witness`
   （完整覆盖 + 剩余观测 + 全部三元组）复核，测试中再用 `tests/_brute_reference.py`
   的第二套独立穷举实现复核。
6. **独立测试参考**：`tests/_brute_reference.py` 直接枚举 3ⁿ 个指派，
   **不 import 被测核心**；期望值（可行性、可行集合、最小移除集、见证）全部
   由它独立产生。40 组 ≤7 人随机家系（含随机单亲缺失与近亲结构）对拍。
   “往返一致”只作补充，不作为期望值来源。

## 已知限制（本题数学/协议子集，不宣称行业标准兼容）

- 仅**单个二等位位点**；不处理单倍型、连锁、多等位基因、性染色体、
  线粒体、突变、基因分型错误概率/似然等。
- “最小”指移除观测的**基数最小**；不计算带权最小集或概率最可能解释。
- 最小移除集在**观测**之间选取（错误归因到观测，不重写家系拓扑）；
  家系结构本身的错误只报输入校验错误，不在冲突定位范围内。
- 上限 20 人；观测数同样受此约束（观测必须引用家系个体）。
- 预算是节点数而非墙钟时间；极小预算下相容问题也可能返回 `unknown`
  （这是契约规定的诚实行为，不是异常）。
- 见证是“存在一份可行修正”的证书；同一移除集的多个可行修正中返回搜索找到的
  第一个，不枚举修正本身（只枚举最小移除集）。

## 目录

```
mendel_conflict/__init__.py   公开接口导出
mendel_conflict/core.py       解析校验、传播、回溯、命中集枚举、见证复核
demo.py                       固定输入演示（正常 + 冲突修正见证 + NaN 拒绝边界）
tests/test_mendel.py          结构化用例 + 随机对拍
tests/test_validation.py      输入校验、错误语义、预算语义
tests/_brute_reference.py     独立 3^n 穷举参考（不调用被测核心）
```
