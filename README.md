# mendelcheck — 孟德尔家系单基因座最小冲突定位

离线遗传数据校验的可复用模块：对单二等位位点校验家系观测的孟德尔一致性；
不一致时定位最少需移除的观测集合，并给出每个被移除观测的修正见证。

## 环境

- Python 3.14.7，仅标准库，**无需安装任何第三方依赖**（`requirements.txt` 为空约定说明）。
- Linux 原生运行；核心、测试、演示全部离线，不需要网络、账号、数据库或 Docker。

## 接口

```python
from mendelcheck import check, CheckResult, ConflictReport, ValidationError

result = check(pedigree, observations, *, node_budget=100_000)
```

候选亲子关系（部分亲缘未定、仅知有限候选名单）：

```python
from mendelcheck import resolve, PlacementResult

result = resolve(
    pedigree,
    observations,
    {("C", "father"): ("F1", "F2")},   # C 的父亲未定，只可能来自名单
    node_budget=100_000,
)
```

### 输入契约

- `pedigree: dict[str, tuple[str | None, str | None]]`
  键为个体 ID（非空 `str`，**至多 20 人**），值为 `(父亲ID或None, 母亲ID或None)`。
  父母 ID 必须引用家系内成员；亲子图必须**无环**。
  缺失父母（`None`）对每个子女、每个亲本槽位视为**独立未知创始者**。
- `observations: dict[str, int]`
  个体 ID → 基因型。基因型为替代等位计数，严格 `int` 且取值 `0/1/2`；
  未观测个体基因型未知，可取三者。
- `node_budget: int`（关键字参数，默认 `100_000`）
  整个 `check` 调用共享的搜索节点预算，须为 `int >= 1`。

### `resolve` 的额外输入

- `parent_candidates: dict[tuple[str, str], tuple[str, ...]]`
  键为 `(子女ID, "father" | "mother")`，值为候选父母 ID 的非空元组/列表。
  对应槽位在 `pedigree` 中必须为 `None`（尚未确定）。
  每个候选必须是家系内成员：**同一个人在整个家系中只有一份身份与观测**——
  被选为父母后，其观测与亲缘约束在所有关系里同时生效，
  不能在不同关系里采用不同解释（与缺失父母的“独立未知创始者”语义不同）。
- 各候选列表长度的乘积（完整组合总数）不得超过
  `MAX_CHOICE_COMBINATIONS = 100_000`，超出抛出 `ValidationError`。

### 返回 `CheckResult`

| 字段 | 含义 |
|---|---|
| `status` | `"feasible"` / `"infeasible"` / `"unknown"` |
| `feasible_genotypes` | 可行时：`{ID: frozenset(可行基因型)}`，覆盖全部个体；否则 `None` |
| `conflict` | 证明无解时：`ConflictReport`；否则 `None` |
| `nodes_used` | 实际消耗的搜索节点数 |

`ConflictReport`：

- `removals: tuple[str, ...]` —— 一个最少需移除的观测 ID 元组（升序；在所有
  最小基数移除集中取字典序最小者，结果确定）。
- `alternatives: tuple[tuple[str, ...], ...]` —— 其余同基数最小移除集（多个最小解释）。
- `witnesses: dict[str, dict[str, int]]` —— 每个被移除观测 ID → 一份完整基因型
  指派。该指派满足全部孟德尔约束及“原观测减去包含该 ID 的首个最小移除集”后
  保留的观测，即修正后的**重新可行见证**。
- `truncated: bool` —— 最小移除集总数超过 `MAX_MINIMAL_SETS`（256）被截断时为 `True`。

### 返回 `PlacementResult`（`resolve`）

| 字段 | 含义 |
|---|---|
| `status` | `"feasible"` / `"infeasible"` / `"unknown"` |
| `choices` | 可行时：全部与观测相容的完整选择（每个为 `{(子女ID, 角色): 父母ID}` 字典，确定性排序）；证明不可能时为空元组；`unknown` 时为 `None` |
| `cyclic` | 因形成祖先循环而被排除的完整选择（同结构元组），与预算无关、始终完整枚举 |
| `nodes_used` | 实际消耗的搜索节点数 |

- `"feasible"`：仍有可行选择（`choices` 非空，已穷尽全部组合）。
- `"infeasible"`：全部组合已穷尽且无一相容——**已确认不可能**（证明，非预算耗尽）。
- `"unknown"`：节点预算耗尽，**搜索尚未完成**，`choices` 为 `None`。

### 状态与错误语义

- `"infeasible"` 是**证明无解**（搜索空间穷尽），`"unknown"` 是**节点预算耗尽**，
  两者严格分开，绝不混淆。
- 非法输入抛出 `ValidationError`（`ValueError` 子类），消息定位到具体个体/字段：
  - 基因型非 `int`（拒绝 `bool`、`float`、`NaN`、`Infinity`）或不在 `{0,1,2}`；
  - `node_budget` 非 `int`、为 `bool` 或 `< 1`；
  - 家系为空、超过 20 人、ID 非法、父母引用未知成员、亲子图有环（报出环路径）；
  - 观测引用未知个体。
- 不做任何静默纠正；`check` 为纯函数式调用，失败不会留下部分变更的状态。

## 运行

```bash
python -m unittest discover -s tests -v   # 必要测试（含独立穷举参考对照）
python demo.py                            # 固定输入演示，约 0.1 秒内完成
```

## 设计取舍

- **约束传播 + 分支搜索**：对每个 (父, 母, 子) 三元约束做 AC-3 弧一致
  （27 个合法三元组预计算为位掩码），收缩域后按最小域变量分支。
  每个递归调用计 1 个节点，预算在整个 `check` 调用内共享。
- **可行基因型集合**：先弧一致预剪，再对每个个体每个候选值做一次可满足性
  检查（至多 20×3 次），精确给出每个个体在某完整解中可取到的值。
- **最小冲突定位**：Reiter 碰撞集算法——删除式极小化求得极小冲突，
  在其成员上分支构建碰撞树，按当前最优基数剪枝，枚举**全部**最小基数
  移除集（上限 256 个，超出置 `truncated`）。见证由确认可行性的那次求解
  直接产出，不额外消耗搜索。
- **缺失父母**：每个 (子女, 缺失亲本槽位) 引入一个独立合成创始者变量，
  严格实现题面“独立未知创始者”语义。
- **候选亲子关系**：`resolve` 先对全部完整组合做无预算的环检测（成环组合
  记入 `cyclic` 并排除），再把每组选择具体化为确定家系，复用同一
  AC-3 + 分支求解器在共享预算内逐一检验；候选父母是家系内的真实个体，
  其观测与身份在所有关系中一致生效。

## 已知限制

- 仅单二等位基因座、经典孟德尔传递；不建模突变、重组、外显率或多位点连锁，
  不宣称兼容任何行业标准格式（如 PED/LINKAGE）。
- 规模上限 20 人；最坏情况搜索为指数级，预算耗尽返回 `unknown` 而非错误结论。
- 最小移除集枚举上限 256 个（超出时 `truncated=True`，已返回的集合仍保证最小基数）。
- 候选亲子关系的完整组合总数上限 `MAX_CHOICE_COMBINATIONS = 100_000`（输入规模限制，
  超出抛 `ValidationError`）；组合逐一求解共享同一节点预算，预算耗尽返回
  `unknown` 而非部分结论。
- 见证语义为“相对包含该观测的首个最小移除集重新可行”，不保证单独移除
  该观测即可行（多个不相交冲突时单移除一个不充分）。

## 误差口径

本库输出全部为**精确离散结果**（基因型集合、ID 元组、整数指派），
不涉及任何浮点计算，无数值误差、无容差判定。
