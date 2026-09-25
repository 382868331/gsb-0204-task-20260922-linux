# mendelcheck — 孟德尔家系单基因座最小冲突定位

离线遗传数据校验的可复用模块：对单二等位位点校验家系观测的孟德尔一致性；
不一致时定位最少需移除的观测集合，并给出每个被移除观测的修正见证。
新增软代价优化入口 `optimize`：把硬观测扩展为可选的每人三项非负整数
代价（`None` 为禁止），在严格传递约束下最小化全家系总代价，给出每人
的全部最优可行基因型与一份确定性最优见证。

## 环境

- Python 3.14.7，仅标准库，**无需安装任何第三方依赖**（`requirements.txt` 为空约定说明）。
- Linux 原生运行；核心、测试、演示全部离线，不需要网络、账号、数据库或 Docker。

## 接口一：硬观测校验 `check`

```python
from mendelcheck import check, CheckResult, ConflictReport, ValidationError

result = check(pedigree, observations, *, node_budget=100_000)
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

### 状态与错误语义

- `"infeasible"` 是**证明无解**（搜索空间穷尽），`"unknown"` 是**节点预算耗尽**，
  两者严格分开，绝不混淆。
- 非法输入抛出 `ValidationError`（`ValueError` 子类），消息定位到具体个体/字段：
  - 基因型非 `int`（拒绝 `bool`、`float`、`NaN`、`Infinity`）或不在 `{0,1,2}`；
  - `node_budget` 非 `int`、为 `bool` 或 `< 1`；
  - 家系为空、超过 20 人、ID 非法、父母引用未知成员、亲子图有环（报出环路径）；
  - 观测引用未知个体。
- 不做任何静默纠正；`check` 为纯函数式调用，失败不会留下部分变更的状态。

## 接口二：软代价优化 `optimize`

```python
from mendelcheck import optimize, OptimizeResult

result = optimize(pedigree, observations, costs=None, *, node_budget=100_000)
```

### 代价契约

- `costs: dict[str, tuple[int | None, int | None, int | None]] | None`
  个体 ID → `(c0, c1, c2)`，`cv` 为该个体取基因型 `v` 的代价：严格
  `int >= 0`（拒绝 `bool`/`float`/负数）；`None` 表示**禁止**该基因型。
  未列入 `costs` 的个体视为 `(0, 0, 0)`；`costs=None` 等价于全免费。
- **硬观测仍然严格**：先按 `observations` 把候选限制为观测值，再施加
  禁止项；观测与禁止冲突、或传递约束不可满足，一律证明无解
  （`infeasible`），**不能用高代价偷偷放过**。
- 目标：在所有满足孟德尔传递约束与硬观测的完整指派上最小化全家系总代价。
  仅实个体计费；缺失父母的合成创始者不计代价（身份规则同 `check`）。

### 返回 `OptimizeResult`

| 字段 | 含义 |
|---|---|
| `status` | `"optimal"` / `"infeasible"` / `"unknown"` |
| `min_cost` | 已证明的最小总代价；未证明时为 `None` |
| `optimal_genotypes` | 已证明最优时：`{ID: frozenset(在最优解中可取的基因型)}`，覆盖全部个体；否则 `None` |
| `witness` | 已证明最优时：一份确定性最优见证（同分按人物 ID 字典序的基因型序列取字典序最小者）；否则 `None` |
| `best_cost_bound` | 已找到的最好可行总代价（可行上界；未找到可行解为 `None`） |
| `nodes_used` | 实际消耗的搜索节点数 |

- `"optimal"`：最优性已证明，`min_cost == best_cost_bound`，
  `optimal_genotypes` 为**全部**最优可行基因型（逐人逐值做一次有界
  可满足性检查，精确而非采样）。
- `"infeasible"`：证明无可行指派（传递约束 / 硬观测 / 禁止项冲突）。
- `"unknown"`：节点预算耗尽。此时 `best_cost_bound` 只是**可行上界**；
  仅当分支定界已完整结束（最优代价已证明、但最优基因型枚举未完成）
  时 `min_cost` 才非 `None`。**绝不给出未经证明的“所有最优”**：
  `unknown` 下 `optimal_genotypes` 与 `witness` 必为 `None`。

## 运行

```bash
./.venv/bin/python -m unittest discover -s tests -v   # 必要测试（含独立穷举参考对照）
./.venv/bin/python demo.py                            # 固定输入演示，约 0.1 秒内完成
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
- **软代价优化（optimize）**：分支定界——下界为各变量当前域内最小单项
  代价之和，严格小于当前最优才继续分支；证明最小代价后，对每人每值
  做“代价 ≤ 最优值”的有界可满足性检查，得到全部最优可行基因型；
  见证按实个体 ID 字典序固定变量序、基因型值升序取第一个最优解，
  保证同分时确定性。禁止项（`None`）直接从初始域剔除并与硬观测取
  交集，因此冲突只会导致证明无解，不会被高代价绕过。

## 已知限制

- 仅单二等位基因座、经典孟德尔传递；不建模突变、重组、外显率或多位点连锁，
  不宣称兼容任何行业标准格式（如 PED/LINKAGE），不构成临床诊断。
- 规模上限 20 人；最坏情况搜索为指数级，预算耗尽返回 `unknown` 而非错误结论。
- 最小移除集枚举上限 256 个（超出时 `truncated=True`，已返回的集合仍保证最小基数）。
- 见证语义为“相对包含该观测的首个最小移除集重新可行”，不保证单独移除
  该观测即可行（多个不相交冲突时单移除一个不充分）。
- `optimize` 的代价模型为每人每基因型独立加性代价，不建模互作或上位效应。

## 误差口径

本库输出全部为**精确离散结果**（基因型集合、ID 元组、整数指派），
不涉及任何浮点计算，无数值误差、无容差判定。
