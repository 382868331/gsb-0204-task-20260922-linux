"""mendelcheck — 孟德尔家系单基因座可行性 / 最小冲突 / 最小代价指派。

公开接口（向后兼容旧契约）：

    check(pedigree, observations, *, costs=None, node_budget=DEFAULT_NODE_BUDGET)
        -> CheckResult

- pedigree: dict，键为个体 ID（非空 str，至多 MAX_INDIVIDUALS=20 个），
  值为 (father_id_or_None, mother_id_or_None) 二元组/列表。
  父母 ID 必须引用家系内成员；亲子图必须无环。
  缺失父母（None）对每个子女、每个亲本槽位视为“独立未知创始者”。
- observations: dict，个体 ID -> 基因型。基因型为替代等位计数，
  严格 int 且取值 0/1/2（拒绝 bool、float、NaN、Infinity）。
  未观测个体基因型未知，可取 0/1/2。
- costs: 可选，dict，个体 ID -> 长度为 3 的序列
  [选择基因型 0 的代价, 选择 1 的代价, 选择 2 的代价]。
  每个代价为非负严格 int（拒绝 bool）；元素为 None 表示禁止该基因型。
  未出现在 costs 中的个体，三个基因型代价均为 0（均不禁止）。
  硬观测与代价同时存在时，先按硬观测把候选限制为观测值，再取其代价；
  若观测基因型恰被该人代价表禁止，则该观测只能作为冲突被移除——
  高代价绝不能让其“偷偷”放行。
- node_budget: 严格 int >= 1（拒绝 bool），为整个 check 调用共享的
  搜索节点预算；耗尽时 status 取 "unknown"（未证明最优，可能给出可行上界），
  与“证明无解 / 已证明最优”严格区分。

status 语义：

- "feasible"：无 costs 输入（旧模式），已找到可行指派并枚举全部
  可行基因型；feasible_genotypes 覆盖全部个体；optimal_genotypes 为 None。
- "optimal"：有 costs 输入且已证明找到全家系总代价最小指派。
  - optimal_cost: 已证明的最小总代价（int >= 0）；
  - optimal_genotypes: {ID: frozenset(在某个最优指派中可取到的基因型)}，
    覆盖全部实个体；
  - witness: 一份确定见证 {ID: 基因型}，在所有最优指派中按人物 ID
    （sorted）序列上基因型字典序最小者；
  - feasible_genotypes: 与 optimal_genotypes 同值（便于统一读取）。
- "unknown"：节点预算耗尽，未证明最优性。
  有 costs 时若已找到至少一个可行指派，则 witness 为一份可行上界见证，
  witness_cost 为其总代价（上界，不保证最优），optimal_genotypes 为 None；
  未找到任何可行指派时这些字段为 None。绝不给出未经证明的“所有最优”。
- "infeasible"：证明无解（非预算耗尽）。若去掉全部硬观测后仍无解
  （例如某个体三个基因型全被禁止、或家系本身矛盾），infeasible_reason
  给出原因，conflict 为 None；否则 conflict 为 ConflictReport：
    - removals:     一个最少需移除的硬观测 ID 元组（升序；在所有最小
                    基数移除集中取字典序最小者，确定性）；
    - alternatives: 其余同基数最小移除集（每个升序，整体字典序）；
    - witnesses:    {被移除观测 ID: 完整基因型指派}。指派满足全部
                    孟德尔约束与“原观测减去包含该 ID 的首个最小移除集”
                    后保留的观测，且在该前提下总代价最小、字典序最小；
    - truncated:    若最小移除集总数超过 MAX_MINIMAL_SETS 被截断则为 True。

所有非法输入抛出 ValidationError（ValueError 子类），消息定位到具体
个体 ID / 字段。本库不做任何静默纠正，输出全部为精确离散结果，
不涉及浮点计算。仅单基因座、经典孟德尔传递，不做任何临床诊断声明。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "check",
    "CheckResult",
    "ConflictReport",
    "MendelError",
    "ValidationError",
    "MAX_INDIVIDUALS",
    "DEFAULT_NODE_BUDGET",
    "MAX_MINIMAL_SETS",
    "GENOTYPES",
]

MAX_INDIVIDUALS = 20
DEFAULT_NODE_BUDGET = 100_000
MAX_MINIMAL_SETS = 256
GENOTYPES = (0, 1, 2)

# 基因型 g 在减数分裂中可传出的等位（0=参考，1=替代）。
_TRANSMIT = ((0,), (0, 1), (1,))

# 合法 (父, 母, 子) 基因型三元组：存在可传等位 a、b 使 a+b == 子基因型。
_COMPAT = tuple(
    (gf, gm, gc)
    for gf in GENOTYPES
    for gm in GENOTYPES
    for gc in GENOTYPES
    if any(a + b == gc for a in _TRANSMIT[gf] for b in _TRANSMIT[gm])
)

_FULL_MASK = 0b111  # 域位掩码：bit v 置位 <=> 基因型 v 可行
_INF = 10 ** 30


class MendelError(Exception):
    """mendelcheck 错误基类。"""


class ValidationError(MendelError, ValueError):
    """输入不合法；消息定位到具体个体/字段。"""


class _BudgetExhausted(Exception):
    """内部：节点预算耗尽（与证明无解严格区分）。"""


class _Budget:
    __slots__ = ("remaining", "limit")

    def __init__(self, limit: int) -> None:
        self.remaining = limit
        self.limit = limit

    def tick(self) -> None:
        self.remaining -= 1
        if self.remaining < 0:
            raise _BudgetExhausted

    @property
    def used(self) -> int:
        return self.limit - max(self.remaining, 0)


@dataclass(frozen=True)
class ConflictReport:
    """证明无解时的最小冲突定位报告。字段语义见模块 docstring。"""

    removals: tuple
    alternatives: tuple
    witnesses: dict
    truncated: bool


@dataclass(frozen=True)
class CheckResult:
    """check() 返回值；status 语义见模块 docstring。"""

    status: str
    feasible_genotypes: Optional[dict]
    conflict: Optional[ConflictReport]
    nodes_used: int
    optimal_genotypes: Optional[dict] = None
    optimal_cost: Optional[int] = None
    witness: Optional[dict] = None
    witness_cost: Optional[int] = None
    infeasible_reason: Optional[str] = None


class _Model:
    """内部求解模型：实个体变量 + 每个缺失亲本槽位一个独立合成创始者变量。"""

    __slots__ = (
        "ids", "index", "domains0", "constraints", "adjacency",
        "obs", "obs_ids", "costs", "has_costs",
    )

    def __init__(self) -> None:
        self.ids = []          # 实个体 ID，按 pedigree 插入序
        self.index = {}        # ID -> 变量下标
        self.domains0 = []     # 无硬观测时的初始域（已应用代价禁止）
        self.constraints = []  # [(父变量, 母变量, 子变量)]
        self.adjacency = []    # 变量 -> 关联约束下标列表
        self.obs = {}          # ID -> 观测基因型
        self.obs_ids = []      # 观测 ID，字典序
        self.costs = {}        # ID -> (c0, c1, c2)，None 表示禁止
        self.has_costs = False


def _values(mask: int) -> list:
    return [v for v in GENOTYPES if mask & (1 << v)]


# ---------------------------------------------------------------- 输入校验

def _check_acyclic(pedigree: Mapping) -> None:
    """DFS 三色标记检测亲子图环；发现环时报出环路径。"""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {pid: WHITE for pid in pedigree}
    for start in pedigree:
        if color[start] != WHITE:
            continue
        color[start] = GRAY
        path = [start]
        stack = [(start, iter(p for p in pedigree[start] if p is not None))]
        while stack:
            node, it = stack[-1]
            advanced = False
            for par in it:
                if color[par] == GRAY:
                    cycle = path[path.index(par):] + [par]
                    raise ValidationError(
                        "pedigree parent-child graph has a cycle: " + " -> ".join(cycle)
                    )
                if color[par] == WHITE:
                    color[par] = GRAY
                    path.append(par)
                    stack.append((par, iter(p for p in pedigree[par] if p is not None)))
                    advanced = True
                    break
            if not advanced:
                color[node] = BLACK
                stack.pop()
                path.pop()


def _validate_costs(costs, pedigree) -> dict:
    if isinstance(costs, (str, bytes)) or not isinstance(costs, Mapping):
        raise ValidationError(
            f"costs must be a mapping {{id: [c0, c1, c2]}}, "
            f"got {type(costs).__name__}"
        )
    out = {}
    for pid, triple in costs.items():
        if not isinstance(pid, str) or pid not in pedigree:
            raise ValidationError(f"costs[{pid!r}]: no such individual in pedigree")
        if isinstance(triple, (str, bytes)) or not isinstance(triple, (tuple, list)):
            raise ValidationError(
                f"costs[{pid!r}] must be a length-3 sequence of int/None, "
                f"got {triple!r}"
            )
        if len(triple) != 3:
            raise ValidationError(
                f"costs[{pid!r}] must have exactly 3 entries (costs for 0, 1, 2), "
                f"got length {len(triple)}"
            )
        parsed = []
        for g, c in zip(GENOTYPES, triple):
            if c is None:
                parsed.append(None)
                continue
            if isinstance(c, bool) or not isinstance(c, int):
                raise ValidationError(
                    f"costs[{pid!r}][{g}] must be a non-negative int or None "
                    f"(bool/float rejected), got {c!r}"
                )
            if c < 0:
                raise ValidationError(
                    f"costs[{pid!r}][{g}] must be non-negative, got {c!r}"
                )
            parsed.append(c)
        out[pid] = tuple(parsed)
    return out


def _validate_and_build(pedigree, observations, costs, node_budget) -> _Model:
    if isinstance(node_budget, bool) or not isinstance(node_budget, int):
        raise ValidationError(
            f"node_budget must be an int >= 1 (bool/float rejected), got {node_budget!r}"
        )
    if node_budget < 1:
        raise ValidationError(f"node_budget must be >= 1, got {node_budget!r}")

    if isinstance(pedigree, (str, bytes)) or not isinstance(pedigree, Mapping):
        raise ValidationError(
            f"pedigree must be a mapping {{id: (father, mother)}}, got {type(pedigree).__name__}"
        )
    if len(pedigree) == 0:
        raise ValidationError("pedigree must contain at least one individual")
    if len(pedigree) > MAX_INDIVIDUALS:
        raise ValidationError(
            f"pedigree has {len(pedigree)} individuals, limit is {MAX_INDIVIDUALS}"
        )
    for pid in pedigree:
        if not isinstance(pid, str) or not pid:
            raise ValidationError(f"individual id must be a non-empty str, got {pid!r}")
    for pid, parents in pedigree.items():
        if (
            isinstance(parents, (str, bytes))
            or not isinstance(parents, (tuple, list))
            or len(parents) != 2
        ):
            raise ValidationError(
                f"pedigree[{pid!r}] must be a (father, mother) pair, got {parents!r}"
            )
        for role, par in zip(("father", "mother"), parents):
            if par is None:
                continue
            if not isinstance(par, str):
                raise ValidationError(
                    f"pedigree[{pid!r}] {role} must be None or a member id str, got {par!r}"
                )
            if par not in pedigree:
                raise ValidationError(
                    f"pedigree[{pid!r}] {role} {par!r} is not a pedigree member"
                )
            if par == pid:
                raise ValidationError(f"pedigree[{pid!r}] lists itself as {role}")
    _check_acyclic(pedigree)

    if isinstance(observations, (str, bytes)) or not isinstance(observations, Mapping):
        raise ValidationError(
            f"observations must be a mapping {{id: genotype}}, got {type(observations).__name__}"
        )
    obs = {}
    for pid, g in observations.items():
        if not isinstance(pid, str) or pid not in pedigree:
            raise ValidationError(f"observations[{pid!r}]: no such individual in pedigree")
        if isinstance(g, bool) or not isinstance(g, int):
            raise ValidationError(
                f"observations[{pid!r}] genotype must be an int in {{0, 1, 2}} "
                f"(bool/float/NaN/Infinity rejected), got {g!r}"
            )
        if g not in GENOTYPES:
            raise ValidationError(
                f"observations[{pid!r}] genotype must be in {{0, 1, 2}}, got {g!r}"
            )
        obs[pid] = g

    cost_map = _validate_costs(costs, pedigree) if costs is not None else {}
    has_costs = costs is not None

    model = _Model()
    model.ids = list(pedigree.keys())
    model.index = {pid: i for i, pid in enumerate(model.ids)}
    model.has_costs = has_costs
    model.costs = cost_map
    # 初始域先应用代价禁止；硬观测稍后作为单值域叠加。二者冲突时观测
    # 只能作为最小移除冲突被移除，禁止/高代价绝不放行它。
    model.domains0 = [_FULL_MASK] * len(model.ids)
    for pid, triple in cost_map.items():
        mask = 0
        for g, c in zip(GENOTYPES, triple):
            if c is not None:
                mask |= 1 << g
        model.domains0[model.index[pid]] = mask
    model.adjacency = [[] for _ in model.ids]

    def synth_founder() -> int:
        idx = len(model.domains0)
        model.domains0.append(_FULL_MASK)
        model.adjacency.append([])
        return idx

    for cid, (fa, mo) in pedigree.items():
        if fa is None and mo is None:
            continue  # 创始者，无三元约束
        fi = model.index[fa] if fa is not None else synth_founder()
        mi = model.index[mo] if mo is not None else synth_founder()
        ci = model.index[cid]
        cno = len(model.constraints)
        model.constraints.append((fi, mi, ci))
        model.adjacency[fi].append(cno)
        model.adjacency[mi].append(cno)
        model.adjacency[ci].append(cno)

    model.obs = obs
    model.obs_ids = sorted(obs)
    return model


# ---------------------------------------------------------------- 约束传播（AC-3）

def _propagate(domains: list, model: _Model) -> bool:
    """对所有三元约束做弧一致；任一域为空返回 False。就地修改 domains。"""
    if any(mask == 0 for mask in domains):
        return False
    queue = list(range(len(model.constraints)))
    in_queue = [True] * len(model.constraints)
    while queue:
        cno = queue.pop()
        in_queue[cno] = False
        fi, mi, ci = model.constraints[cno]
        df, dm, dc = domains[fi], domains[mi], domains[ci]
        nf = nm = nc = 0
        for gf, gm, gc in _COMPAT:
            if (df >> gf) & 1 and (dm >> gm) & 1 and (dc >> gc) & 1:
                nf |= 1 << gf
                nm |= 1 << gm
                nc |= 1 << gc
        if nf == 0 or nm == 0 or nc == 0:
            return False
        for var, new in ((fi, nf), (mi, nm), (ci, nc)):
            if new != domains[var]:
                domains[var] = new
                for c2 in model.adjacency[var]:
                    if c2 != cno and not in_queue[c2]:
                        queue.append(c2)
                        in_queue[c2] = True
    return True


# ---------------------------------------------------------------- 代价工具

def _fixed_domains(model: _Model, kept_obs_ids) -> list:
    """初始域（已含代价禁止）叠加保留的硬观测。"""
    domains = list(model.domains0)
    keep = set(kept_obs_ids)
    for pid, g in model.obs.items():
        if pid in keep:
            domains[model.index[pid]] &= 1 << g
    return domains


def _value_cost(model: _Model, var: int, value: int) -> int:
    """实个体变量取 value 的代价；合成创始者与未给代价者代价为 0。"""
    if var >= len(model.ids):
        return 0
    triple = model.costs.get(model.ids[var])
    if triple is None:
        return 0
    c = triple[value]
    return 0 if c is None else c


def _bound_from(model: _Model, domains: list, fixed_cost: int) -> int:
    """所有实个体域内最小代价之和（单值域取其唯一值）+ fixed_cost。

    调用方保证：fixed_cost 只统计已经不在域中体现的已固定实个体；
    当前实现中域始终体现所有实个体，故 fixed_cost 恒为 0。
    """
    total = fixed_cost
    for i in range(len(model.ids)):
        vals = _values(domains[i])
        if not vals:
            return _INF
        total += min(_value_cost(model, i, v) for v in vals)
    return total


def _bound_to(model: _Model, domains: list) -> int:
    """所有实个体域内最大代价之和（单值域取其唯一值）。"""
    total = 0
    for i in range(len(model.ids)):
        vals = _values(domains[i])
        if not vals:
            return -1
        total += max(_value_cost(model, i, v) for v in vals)
    return total


def _target_possible(model: _Model, domains: list, target: int) -> bool:
    """安全判定：在当前域内能否为实个体选出代价总和恰为 target 的组合。

    传播可能进一步缩域，故通过只可能造成“漏报可行”——本函数只在
    能证明不可能（False）时用于剪枝；True 时继续搜索，保持正确性。
    """
    reachable = {0}
    for i in range(len(model.ids)):
        vals = _values(domains[i])
        if not vals:
            return False
        costs_here = frozenset(_value_cost(model, i, v) for v in vals)
        reachable = {t + c for t in reachable for c in costs_here if t + c <= target}
    return target in reachable


# ---------------------------------------------------------------- 最优搜索（分支定界）

def _optimize(domains: list, model: _Model, budget: _Budget):
    """分支定界求代价最小完整指派。

    返回 (best_sol, best_cost, proved_optimal)。
    proved_optimal=False 仅可能因 _BudgetExhausted：此时 best_sol 是已找到
    的最优（可行上界），不保证全局最优；best_sol is None 表示尚未找到可行解。
    """
    best = {"sol": None, "cost": _INF}

    def recurse(doms) -> None:
        budget.tick()
        if not _propagate(doms, model):
            return
        lb = _bound_from(model, doms, 0)
        if lb >= best["cost"]:
            return  # 不可能严格更优；等代价解由 _lex_equal 单独处理
        var = -1
        best_pc = 4
        for i, mask in enumerate(doms):
            pc = mask.bit_count()
            if 1 < pc < best_pc:
                best_pc = pc
                var = i
                if pc == 2:
                    break
        if var < 0:
            sol = [mask.bit_length() - 1 for mask in doms]
            cost = _bound_to(model, doms)
            if cost < best["cost"]:
                best["sol"] = sol
                best["cost"] = cost
            return
        for v in _values(doms[var]):
            child = list(doms)
            child[var] = 1 << v
            recurse(child)

    try:
        recurse(domains)
    except _BudgetExhausted:
        return best["sol"], best["cost"], False
    return best["sol"], best["cost"], True


def _find_equal(domains: list, model: _Model, budget: _Budget, target: int,
                lex_key: bool):
    """寻找总代价恰为 target 的完整指派。

    lex_key=True 时额外要求该指派在实个体（sorted ID 序，即 model.ids 之外
    再按 sorted 顺序）上字典序最小：按 sorted ID 逐个固定基因型 0,1,2 试探。
    返回 (sol, cost) 或 None。
    """

    def recurse(doms):
        budget.tick()
        if not _propagate(doms, model):
            return None
        # 域内各值代价能否凑出 target（单调安全剪枝；下界在传播后不单调，
        # 不能用于等值存在性判断）。
        if not _target_possible(model, doms, target):
            return None
        var = -1
        best_pc = 4
        for i, mask in enumerate(doms):
            pc = mask.bit_count()
            if 1 < pc < best_pc:
                best_pc = pc
                var = i
                if pc == 2:
                    break
        if var < 0:
            sol = [mask.bit_length() - 1 for mask in doms]
            cost = _bound_to(model, doms)  # 全部单值：最大==最小==实际代价
            return (sol, cost) if cost == target else None
        for v in _values(doms[var]):
            child = list(doms)
            child[var] = 1 << v
            found = recurse(child)
            if found is not None:
                return found
        return None

    if not lex_key:
        return recurse(domains)

    # 逐个 sorted 实个体固定最小可行基因型，保证见证全局字典序最小。
    trial = list(domains)
    assignment = {}
    for pid in sorted(model.ids):
        i = model.index[pid]
        chosen = None
        for v in _values(trial[i]):
            probe = list(trial)
            probe[i] = 1 << v
            found = recurse(probe)
            if found is not None:
                chosen = v
                trial[i] = 1 << v
                assignment[pid] = v
                break
        if chosen is None:
            return None
    final_sol = recurse(trial)
    return final_sol


def _optimal_sets(model: _Model, budget: _Budget, domains: list, optimum: int):
    """对每个实个体求“在某个总代价==optimum 的指派中可取到”的基因型集合。

    每个候选值单独做一次“其余人完全自由”的等值存在性检查，枚举全部
    最优指派中的并集；字典序最小见证由 _find_equal(lex_key=True) 求。
    """
    out = {}
    for pid in sorted(model.ids):
        i = model.index[pid]
        feasible_values = []
        for v in _values(domains[i]):
            probe = _fixed_domains(model, model.obs_ids)
            probe[i] = 1 << v
            found = _find_equal(probe, model, budget, optimum, False)
            if found is not None:
                feasible_values.append(v)
        out[pid] = frozenset(feasible_values)
        if not feasible_values:
            raise AssertionError("internal: optimal value vanished during enumeration")
    got = _find_equal(
        _fixed_domains(model, model.obs_ids), model, budget, optimum, True
    )
    if got is None:
        raise _BudgetExhausted
    sol = got[0]
    witness = {pid: sol[model.index[pid]] for pid in model.ids}
    return out, witness


# ---------------------------------------------------------------- 旧模式：可行性枚举

def _solve_feasible(domains: list, model: _Model, budget: _Budget):
    """只求一个满足约束与单值域的完整解（无代价视角，旧模式用）。"""
    budget.tick()
    if not _propagate(domains, model):
        return None
    var = -1
    best = 4
    for i, mask in enumerate(domains):
        pc = mask.bit_count()
        if 1 < pc < best:
            best = pc
            var = i
            if pc == 2:
                break
    if var < 0:
        return [mask.bit_length() - 1 for mask in domains]
    for v in _values(domains[var]):
        child = list(domains)
        child[var] = 1 << v
        sol = _solve_feasible(child, model, budget)
        if sol is not None:
            return sol
    return None


def _feasible_sets(model: _Model, budget: _Budget) -> dict:
    """旧模式：对每个实个体，返回其在某个完整解中可取到的基因型集合。"""
    base = _fixed_domains(model, model.obs_ids)
    if not _propagate(base, model):  # 已知可行，传播不可能失败
        raise AssertionError("internal: propagation failed on a feasible instance")
    out = {}
    for pid in sorted(model.ids):
        i = model.index[pid]
        vals = set()
        for v in _values(base[i]):  # 弧一致已剪掉的值必然不可行，跳过
            trial = list(base)
            trial[i] = 1 << v
            if _solve_feasible(trial, model, budget) is not None:
                vals.add(v)
        out[pid] = frozenset(vals)
    return out


# ---------------------------------------------------------------- 最小冲突定位

def _solve_obs(model: _Model, kept_obs_ids, budget: _Budget):
    """在仅保留 kept_obs_ids 硬观测的条件下求一个代价最优完整解。

    返回 (sol, cost, proved)。代价已开启时为分支定界最优结果；
    旧模式下退化为可行性求解（cost 为 None）。
    """
    domains = _fixed_domains(model, kept_obs_ids)
    if model.has_costs:
        sol, cost, proved = _optimize(domains, model, budget)
        if sol is None:
            return None
        return sol, cost, proved
    sol = _solve_feasible(domains, model, budget)
    if sol is None:
        return None
    return sol, None, True


def _find_minimal_conflict(model: _Model, obs_ids, budget: _Budget) -> frozenset:
    """删除式极小化：返回一个（关于集合包含）极小的不可行观测子集。"""
    current = sorted(obs_ids)
    for o in list(current):
        trial = [x for x in current if x != o]
        if _solve_obs(model, trial, budget) is None:
            current = trial
    return frozenset(current)


def _minimum_corrections(model: _Model, budget: _Budget):
    """Reiter 碰撞集搜索：枚举全部最小基数的观测移除集及其可行见证。

    有代价时，每个保留观测下的见证取总代价最小者；字典序最小见证由
    _find_equal(lex_key=True) 在相同最优代价上挑选。
    返回 (sets_sorted, witnesses, truncated)。
    """
    best = [len(model.obs_ids) + 1]
    found = set()          # frozenset(移除的观测ID)
    cost_by_set = {}       # frozenset -> 该前提下的最优总代价（或 None）
    expanded = set()
    truncated = [False]
    stack = [frozenset()]
    while stack:
        removed = stack.pop()
        if len(removed) > best[0]:
            continue
        kept = [o for o in model.obs_ids if o not in removed]
        result = _solve_obs(model, kept, budget)
        if result is not None:
            sol, cost, _proved = result
            if len(removed) < best[0]:
                best[0] = len(removed)
                found.clear()
                cost_by_set.clear()
            if removed not in found:
                if len(found) >= MAX_MINIMAL_SETS:
                    truncated[0] = True
                else:
                    found.add(removed)
                    cost_by_set[removed] = cost
            continue
        if len(removed) >= best[0] or removed in expanded:
            continue
        expanded.add(removed)
        conflict = _find_minimal_conflict(model, kept, budget)
        for o in sorted(conflict):
            nxt = removed | {o}
            if nxt not in expanded:
                stack.append(nxt)

    sets_sorted = sorted(tuple(sorted(s)) for s in found)
    witnesses = {}
    for s in sets_sorted:
        kept = [o for o in model.obs_ids if o not in s]
        domains = _fixed_domains(model, kept)
        if model.has_costs:
            optimum = cost_by_set[frozenset(s)]
            got = _find_equal(domains, model, budget, optimum, True)
            if got is None:  # 预算在此耗尽不应发生：最优解刚被找到
                raise _BudgetExhausted
            sol = got[0]
        else:
            sol = _solve_feasible(domains, model, budget)
            if sol is None:
                raise AssertionError("internal: witness lost during enumeration")
        assignment = {pid: sol[model.index[pid]] for pid in model.ids}
        for o in s:
            if o not in witnesses:
                witnesses[o] = assignment
    return sets_sorted, witnesses, truncated[0]


# ---------------------------------------------------------------- 公开接口

def check(pedigree, observations, *, costs=None,
          node_budget: int = DEFAULT_NODE_BUDGET) -> CheckResult:
    """校验单基因座孟德尔家系；可选每人三项非负整数代价。契约见模块 docstring。"""
    model = _validate_and_build(pedigree, observations, costs, node_budget)
    budget = _Budget(node_budget)
    try:
        if model.has_costs:
            domains = _fixed_domains(model, model.obs_ids)
            sol, best_cost, proved = _optimize(domains, model, budget)
            if sol is not None and proved:
                # 已证明最优：枚举每个实个体的全部最优可行基因型，并取
                # sorted 人物 ID 序列上基因型字典序最小的最优指派为见证。
                optimal_sets, witness = _optimal_sets(
                    model, budget, domains, best_cost
                )
                return CheckResult(
                    status="optimal",
                    feasible_genotypes=optimal_sets,
                    conflict=None,
                    nodes_used=budget.used,
                    optimal_genotypes=optimal_sets,
                    optimal_cost=best_cost,
                    witness=witness,
                    witness_cost=best_cost,
                )
            if sol is not None:
                # 预算耗尽：只有可行上界，绝不宣称最优或枚举“所有最优”。
                upper = {pid: sol[model.index[pid]] for pid in model.ids}
                return CheckResult(
                    status="unknown",
                    feasible_genotypes=None,
                    conflict=None,
                    nodes_used=node_budget,
                    witness=upper,
                    witness_cost=best_cost,
                )
            # 尚未找到任何可行解：可能是硬观测冲突、结构性不可行，或仅预算不足。
            # 1) 先用弧一致做零节点结构性检查（某人域为空即与观测无关地无解）。
            struct_domains = _fixed_domains(model, ())
            if not _propagate(struct_domains, model):
                return CheckResult(
                    status="infeasible",
                    feasible_genotypes=None,
                    conflict=None,
                    nodes_used=node_budget,
                    infeasible_reason=(
                        "structurally infeasible: some individual has no "
                        "allowed genotype (all three values are forbidden, or "
                        "eliminated by pedigree constraints)"
                    ),
                )
            # 2) 无硬观测却找不到解：弧一致成功只意味着可能可行，需证明。
            if not model.obs_ids:
                return CheckResult(
                    status="unknown",
                    feasible_genotypes=None,
                    conflict=None,
                    nodes_used=node_budget,
                )
            # 3) 有硬观测：做最小移除定位（预算耗尽由外层兜底为 unknown）。
            sets_sorted, witnesses, truncated = _minimum_corrections(model, budget)
            if not sets_sorted:
                return CheckResult(
                    status="unknown",
                    feasible_genotypes=None,
                    conflict=None,
                    nodes_used=node_budget,
                )
            report = ConflictReport(
                removals=sets_sorted[0],
                alternatives=tuple(sets_sorted[1:]),
                witnesses=witnesses,
                truncated=truncated,
            )
            return CheckResult(
                status="infeasible",
                feasible_genotypes=None,
                conflict=report,
                nodes_used=node_budget,
            )

        # 旧模式（无 costs）
        domains = _fixed_domains(model, model.obs_ids)
        sol = _solve_feasible(domains, model, budget)
        if sol is not None:
            return CheckResult(
                status="feasible",
                feasible_genotypes=_feasible_sets(model, budget),
                conflict=None,
                nodes_used=budget.used,
            )
        # 先判定是否结构性不可行（去掉全部观测仍无解）。
        struct_domains = _fixed_domains(model, ())
        if not _propagate(struct_domains, model):
            return CheckResult(
                status="infeasible",
                feasible_genotypes=None,
                conflict=None,
                nodes_used=budget.used,
                infeasible_reason=(
                    "structurally infeasible: pedigree constraints leave no "
                    "complete genotype assignment"
                ),
            )
        sets_sorted, witnesses, truncated = _minimum_corrections(model, budget)
        report = ConflictReport(
            removals=sets_sorted[0],
            alternatives=tuple(sets_sorted[1:]),
            witnesses=witnesses,
            truncated=truncated,
        )
        return CheckResult(
            status="infeasible",
            feasible_genotypes=None,
            conflict=report,
            nodes_used=budget.used,
        )
    except _BudgetExhausted:
        return CheckResult(
            status="unknown",
            feasible_genotypes=None,
            conflict=None,
            nodes_used=node_budget,
        )
