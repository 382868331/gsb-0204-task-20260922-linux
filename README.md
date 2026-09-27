# mendelcheck — 孟德尔家系单基因座最小冲突定位与候选亲本解析

离线遗传数据校验的可复用模块：对单二等位位点校验家系观测的孟德尔一致性；
不一致时定位最少需移除的观测集合，并给出每个被移除观测的修正见证。
同时支持“尚未确定、但只可能来自给定名单”的候选亲子关系，枚举与观测相容的
完整亲本选择，排除会形成祖先循环的选择。

## 环境

- Python 3.14.7，仅标准库，**无需安装任何第三方依赖**（`requirements.txt` 为空约定说明）。
- Linux 原生运行；核心、测试、演示全部离线，不需要网络、账号、数据库或 Docker。

## 接口

```python
from mendelcheck import (
    check, resolve_links,
    CheckResult, ConflictReport, LinkResult, FeasibleLink, ValidationError,
)

result = check(pedigree, observations, *, node_budget=100_000)
```

候选亲本解析：

```python
result = resolve_links(pedigree, observations, *,
                       node_budget=100_000, max_feasible=256)
```

### 输入契约

- `pedigree: dict[str, tuple[str | None, str | None]]`
  键为个体 ID（非空 `str`，**至多 20 人**），值为 `(父亲ID或None, 母亲ID或None)`。
  父母 ID 必须引用家系内成员；亲子图必须**无环**。
  缺失父母（`None`）对每个子女、每个亲本槽位视为**独立未知创始者**。
  `resolve_links` 额外允许某个亲本槽位填**非空候选 ID 序列**，如
  `"X": (["A", "B"], "M")` 表示 X 的母亲确定为 M、父亲尚未确定但只可能是
  A 或 B；重复候选按首次出现去重。候选为个体自身或引用未知成员将报错；
  `check()` 不接受候选序列（必须先 `resolve_links` 解析）。
- `observations: dict[str, int]`
  个体 ID → 基因型。基因型为替代等位计数，严格 `int` 且取值 `0/1/2`；
  未观测个体基因型未知，可取三者。
- `node_budget: int`（关键字参数，默认 `100_000`）
  整个调用共享的搜索节点预算，须为 `int >= 1`。
- `resolve_links` 的 `max_feasible: int`（默认 `256`，须为 `int >= 1`，拒绝
  `bool`）：最多返回的相容完整选择数；达到上限提前停止时 `truncated=True`。

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

### `resolve_links` 返回 `LinkResult`

| 字段 | 含义 |
|---|---|
| `status` | `"feasible"` / `"infeasible"` / `"unknown"` |
| `slots` | 不确定亲本槽位元组 `(子女ID, "father"|"mother")`，按家系插入序确定 |
| `feasible` | 相容的 `FeasibleLink` 元组（DFS 确定性顺序） |
| `possible` | `{子女ID: {"father"|"mother": frozenset(候选)}}`，仅含在某相容选择中出现过的候选 |
| `possible_genotypes` | `{个体ID: frozenset(在某相容见证中取到的基因型)}` |
| `truncated` | 相容选择达到 `max_feasible` 上限、提前停止时为 `True` |
| `nodes_used` | 实际消耗的搜索节点数 |

`FeasibleLink`：

- `selection`：与 `slots` 对齐的被选亲本 ID 元组；
- `parents`：该组完整选择下 `{子女ID: (父亲ID|None, 母亲ID|None)}` 的具体家系；
- `assignment`：满足全部孟德尔约束与观测的一份完整基因型指派（重新可行见证）。

同一个体在所有亲子关系中共享同一个变量与观测，不会在不同关系里采用不同解释；
会形成祖先循环的完整选择在枚举中被剪枝。`status="infeasible"` 表示**全部**
完整选择均已检验且无一相容（确认不可能）；`"feasible"` 表示搜索穷尽且至少一组
相容；`"unknown"` 表示 `node_budget` 耗尽、搜索未完成，此时 `feasible` /
`possible` 仅为已找到的部分结果，不据此断言穷尽。

### 状态与错误语义

- `"infeasible"` 是**证明无解**（搜索空间穷尽），`"unknown"` 是**节点预算耗尽**，
  两者严格分开，绝不混淆。
- 非法输入抛出 `ValidationError`（`ValueError` 子类），消息定位到具体个体/字段：
  - 基因型非 `int`（拒绝 `bool`、`float`、`NaN`、`Infinity`）或不在 `{0,1,2}`；
  - `node_budget` 非 `int`、为 `bool` 或 `< 1`；
  - 家系为空、超过 20 人、ID 非法、父母引用未知成员、亲子图有环（报出环路径）；
  - 观测引用未知个体；
  - `resolve_links` 候选名单为空、含非 `str`、引用未知成员或列出个体自身；
  - `max_feasible` 非 `int`、为 `bool` 或 `< 1`。
- 不做任何静默纠正；两个入口均为纯函数式调用，失败不会留下部分变更的状态。

## 运行

```bash
python -m unittest discover -s tests -v   # 必要测试（含独立穷举参考对照）
python demo.py                            # 固定输入演示，约 0.1 秒内完成
```

## 设计取舍

- **约束传播 + 分支搜索**：对每个 (父, 母, 子) 三元约束做 AC-3 弧一致
  （27 个合法三元组预计算为位掩码），收缩域后按最小域变量分支。
  每个递归调用计 1 个节点，预算在整个调用内共享。
- **可行基因型集合**：先弧一致预剪，再对每个个体每个候选值做一次可满足性
  检查（至多 20×3 次），精确给出每个个体在某完整解中可取到的值。
- **最小冲突定位**：Reiter 碰撞集算法——删除式极小化求得极小冲突，
  在其成员上分支构建碰撞树，按当前最优基数剪枝，枚举**全部**最小基数
  移除集（上限 256 个，超出置 `truncated`）。见证由确认可行性的那次求解
  直接产出，不额外消耗搜索。
- **缺失父母**：每个 (子女, 缺失亲本槽位) 引入一个独立合成创始者变量，
  严格实现题面“独立未知创始者”语义。
- **候选亲本枚举**：将候选槽位按确定顺序展开为一棵选择树，深度优先枚举完整
  组合；每加入一条 child->parent 边先做沿亲本边的可达性成环剪枝；叶节点用与
  `check` 完全相同的传播+搜索检验孟德尔一致性，记录选择与一份见证。选择树
  节点与叶节点求解共享同一 `node_budget`。仅由确定亲本边构成的图若有环，
  在输入校验阶段直接拒绝并报出环路径。

## 已知限制

- 仅单二等位基因座、经典孟德尔传递；不建模突变、重组、外显率或多位点连锁，
  不宣称兼容任何行业标准格式（如 PED/LINKAGE）。
- 规模上限 20 人；最坏情况搜索为指数级，预算耗尽返回 `unknown` 而非错误结论。
- 最小移除集枚举上限 256 个（超出时 `truncated=True`，已返回的集合仍保证最小基数）。
- 候选完整选择枚举上限默认 `max_feasible=256`（`MAX_LINK_SELECTIONS`）；达到
  上限时 `status` 仍为 `"feasible"` 但 `truncated=True`，其余组合未检验。
- 候选组合数随不确定槽位数指数增长；预算耗尽返回 `"unknown"` 而非穷尽结论。
- `check` 的见证语义为“相对包含该观测的首个最小移除集重新可行”，不保证单独移除
  该观测即可行（多个不相交冲突时单移除一个不充分）。

## 误差口径

本库输出全部为**精确离散结果**（基因型集合、ID 元组、整数指派），
不涉及任何浮点计算，无数值误差、无容差判定。
