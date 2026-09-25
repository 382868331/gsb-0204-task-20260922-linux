"""mendelcheck — 孟德尔家系单基因座最小冲突定位。

契约（固定公开接口）：

    check(pedigree, observations, *, node_budget=DEFAULT_NODE_BUDGET) -> CheckResult

- pedigree: dict，键为个体 ID（非空 str，至多 MAX_INDIVIDUALS=20 个），
  值为 (father_id_or_None, mother_id_or_None) 二元组/列表。
  父母 ID 必须引用家系内成员；亲子图必须无环。
  缺失父母（None）对每个子女、每个亲本槽位视为“独立未知创始者”。
- observations: dict，个体 ID -> 基因型。基因型为替代等位计数，
  严格 int 且取值 0/1/2（拒绝 bool、float、NaN、Infinity）。
  未观测个体基因型未知，可取 0/1/2。
- node_budget: 严格 int >= 1（拒绝 bool），为整个 check 调用共享的
  搜索节点预算；耗尽时返回 status="unknown"，与“证明无解”严格区分。

返回 CheckResult：
- status == "feasible":   feasible_genotypes 为 {ID: frozenset(可行基因型)}，
                          覆盖全部个体；conflict 为 None。
- status == "infeasible": conflict 为 ConflictReport（证明无解，非预算耗尽）：
    - removals:     一个最少需移除的观测 ID 元组（升序；在所有最小
                    基数移除集中取字典序最小者，确定性）；
    - alternatives: 其余同基数最小移除集（每个升序，整体字典序）；
    - witnesses:    {被移除观测 ID: 完整基因型指派}。每个指派满足全部
                    孟德尔约束与“原观测减去包含该 ID 的首个最小移除集”
                    后保留的观测，即修正后的重新可行见证；
    - truncated:    若最小移除集总数超过 MAX_MINIMAL_SETS 被截断则为 True。
- status == "unknown":    节点预算耗尽，既未证明可行也未证明无解；
                          feasible_genotypes 与 conflict 均为 None。

所有非法输入抛出 ValidationError（ValueError 子类），消息定位到具体
个体 ID / 字段。本库不做任何静默纠正，输出全部为精确离散结果，
不涉及浮点计算。

软代价优化（新增公开接口）：

    optimize(pedigree, observations, costs=None, *, node_budget=DEFAULT_NODE_BUDGET)
        -> OptimizeResult

- costs: None 或 dict，个体 ID -> (c0, c1, c2) 三项序列。cv 为严格 int 且
  >= 0（拒绝 bool/float/负数），表示该个体取基因型 v 的代价；cv 为 None
  表示禁止该基因型。未列入 costs 的个体视为 (0, 0, 0)。
- 硬观测仍然严格：先按 observations 把候选限制为观测值，再施加禁止项；
  观测与禁止冲突（或传递约束不可满足）一律证明无解，不能用高代价放过。
- 目标：在所有满足孟德尔传递约束与硬观测的完整指派上，最小化全家系
  总代价（仅实个体计费，缺失父母的合成创始者不计代价）。

返回 OptimizeResult：
- status == "optimal":    已证明最优。min_cost 为最小总代价；
                          optimal_genotypes 为 {ID: frozenset(在最优解中
                          可取的基因型)}，覆盖全部个体；witness 为一份确定
                          性最优见证（同分时按人物 ID 字典序的基因型序列
                          取字典序最小者）；best_cost_bound == min_cost。
- status == "infeasible": 证明无可行指派（传递约束/硬观测/禁止项冲突）。
- status == "unknown":    节点预算耗尽。best_cost_bound 为已找到的最好
                          可行总代价（可行上界，未找到可行解则为 None）；
                          若最优代价已证明但最优基因型集合未枚举完，
                          min_cost 为该已证明最优值，否则为 None；
                          optimal_genotypes 与 witness 必为 None —— 绝不
                          给出未经证明的“所有最优”。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "check",
    "optimize",
    "CheckResult",
    "OptimizeResult",
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
_COMPAT = frozenset(
    (gf, gm, gc)
    for gf in GENOTYPES
    for gm in GENOTYPES
    for gc in GENOTYPES
    if any(a + b == gc for a in _TRANSMIT[gf] for b in _TRANSMIT[gm])
)

_FULL_MASK = 0b111  # 域位掩码：bit v 置位 <=> 基因型 v 可行


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
    """check() 的返回值。status ∈ {"feasible", "infeasible", "unknown"}。"""

    status: str
    feasible_genotypes: Optional[dict]
    conflict: Optional[ConflictReport]
    nodes_used: int


@dataclass(frozen=True)
class OptimizeResult:
    """optimize() 的返回值。status ∈ {"optimal", "infeasible", "unknown"}。"""

    status: str
    min_cost: Optional[int]
    optimal_genotypes: Optional[dict]
    witness: Optional[dict]
    best_cost_bound: Optional[int]
    nodes_used: int


class _Model:
    """内部求解模型：实个体变量 + 每个缺失亲本槽位一个独立合成创始者变量。"""

    __slots__ = ("ids", "index", "domains0", "constraints", "adjacency", "obs", "obs_ids")

    def __init__(self) -> None:
        self.ids = []          # 实个体 ID，按 pedigree 插入序
        self.index = {}        # ID -> 变量下标
        self.domains0 = []     # 无观测时的初始域（全 FULL_MASK）
        self.constraints = []  # [(父变量, 母变量, 子变量)]
        self.adjacency = []    # 变量 -> 关联约束下标列表
        self.obs = {}          # ID -> 观测基因型
        self.obs_ids = []      # 观测 ID，字典序


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


def _validate_and_build(pedigree, observations, node_budget) -> _Model:
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

    model = _Model()
    model.ids = list(pedigree.keys())
    model.index = {pid: i for i, pid in enumerate(model.ids)}
    model.domains0 = [_FULL_MASK] * len(model.ids)
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


# ---------------------------------------------------------------- 分支搜索

def _solve(domains: list, model: _Model, budget: _Budget) -> Optional[list]:
    """约束传播 + 最小域分支。返回全部变量（含合成创始者）的一个解，或 None。"""
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
        sol = _solve(child, model, budget)
        if sol is not None:
            return sol
    return None


def _solve_obs(model: _Model, kept_obs_ids, budget: _Budget) -> Optional[list]:
    """在仅保留 kept_obs_ids 观测的条件下求一个解。"""
    domains = list(model.domains0)
    keep = set(kept_obs_ids)
    for pid, g in model.obs.items():
        if pid in keep:
            domains[model.index[pid]] = 1 << g
    return _solve(domains, model, budget)


def _feasible_sets(model: _Model, budget: _Budget) -> dict:
    """对每个实个体，返回其在某个完整解中可取到的基因型集合。"""
    base = list(model.domains0)
    for pid, g in model.obs.items():
        base[model.index[pid]] = 1 << g
    if not _propagate(base, model):  # 已知可行，传播不可能失败
        raise AssertionError("internal: propagation failed on a feasible instance")
    out = {}
    for i, pid in enumerate(model.ids):
        vals = set()
        for v in _values(base[i]):  # 弧一致已剪掉的值必然不可行，跳过
            trial = list(base)
            trial[i] = 1 << v
            if _solve(trial, model, budget) is not None:
                vals.add(v)
        out[pid] = frozenset(vals)
    return out


# ---------------------------------------------------------------- 最小冲突定位

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

    返回 (sets_sorted, witnesses, truncated)：
    sets_sorted 为升序元组的字典序列表；witnesses 为 {观测ID: 完整指派}。
    """
    best = [len(model.obs_ids) + 1]
    found = set()          # frozenset(移除的观测ID)
    sol_by_set = {}        # frozenset -> 保留观测下的一个完整解
    expanded = set()
    truncated = [False]
    stack = [frozenset()]
    while stack:
        removed = stack.pop()
        if len(removed) > best[0]:
            continue
        kept = [o for o in model.obs_ids if o not in removed]
        sol = _solve_obs(model, kept, budget)
        if sol is not None:
            if len(removed) < best[0]:
                best[0] = len(removed)
                found.clear()
                sol_by_set.clear()
            if removed not in found:
                if len(found) >= MAX_MINIMAL_SETS:
                    truncated[0] = True
                else:
                    found.add(removed)
                    sol_by_set[removed] = sol
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
        sol = sol_by_set[frozenset(s)]
        assignment = {pid: sol[model.index[pid]] for pid in model.ids}
        for o in s:
            if o not in witnesses:
                witnesses[o] = assignment
    return sets_sorted, witnesses, truncated[0]


# ---------------------------------------------------------------- 公开接口

def check(pedigree, observations, *, node_budget: int = DEFAULT_NODE_BUDGET) -> CheckResult:
    """校验单基因座孟德尔家系，定位最小观测冲突。契约见模块 docstring。"""
    model = _validate_and_build(pedigree, observations, node_budget)
    budget = _Budget(node_budget)
    try:
        sol = _solve_obs(model, model.obs_ids, budget)
        if sol is not None:
            return CheckResult(
                status="feasible",
                feasible_genotypes=_feasible_sets(model, budget),
                conflict=None,
                nodes_used=budget.used,
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


# ---------------------------------------------------------------- 软代价优化

def _validate_costs(pedigree, costs) -> dict:
    """校验 costs，返回 {ID: (c0, c1, c2)}（元素为 None 或非负 int）。"""
    if costs is None:
        return {}
    if isinstance(costs, (str, bytes)) or not isinstance(costs, Mapping):
        raise ValidationError(
            f"costs must be a mapping {{id: (c0, c1, c2)}} or None, "
            f"got {type(costs).__name__}"
        )
    out = {}
    for pid, triple in costs.items():
        if not isinstance(pid, str) or pid not in pedigree:
            raise ValidationError(f"costs[{pid!r}]: no such individual in pedigree")
        if (
            isinstance(triple, (str, bytes))
            or not isinstance(triple, (tuple, list))
            or len(triple) != 3
        ):
            raise ValidationError(
                f"costs[{pid!r}] must be a (c0, c1, c2) triple, got {triple!r}"
            )
        vals = []
        for v, c in zip(GENOTYPES, triple):
            if c is None:
                vals.append(None)
                continue
            if isinstance(c, bool) or not isinstance(c, int):
                raise ValidationError(
                    f"costs[{pid!r}][{v}] must be None or an int >= 0 "
                    f"(bool/float rejected), got {c!r}"
                )
            if c < 0:
                raise ValidationError(
                    f"costs[{pid!r}][{v}] must be >= 0, got {c!r}"
                )
            vals.append(c)
        out[pid] = tuple(vals)
    return out


def _domain_min_cost(mask: int, costs) -> int:
    """域内最小单项代价。域已剔除 None（禁止）项，故取最小值安全。"""
    return min(costs[v] for v in GENOTYPES if (mask >> v) & 1)


def _lower_bound(domains, var_costs) -> int:
    return sum(_domain_min_cost(mask, var_costs[i]) for i, mask in enumerate(domains))


def _bnb(domains, model: _Model, var_costs, budget: _Budget, best: list) -> None:
    """分支定界求最小总代价，结果写入 best[0]（无可行解则保持 None）。"""
    budget.tick()
    if not _propagate(domains, model):
        return
    lb = 0
    var = -1
    best_pc = 4
    for i, mask in enumerate(domains):
        if mask == 0:
            return
        lb += _domain_min_cost(mask, var_costs[i])
        pc = mask.bit_count()
        if 1 < pc < best_pc:
            best_pc = pc
            var = i
    if best[0] is not None and lb >= best[0]:
        return
    if var < 0:
        best[0] = lb  # 全部变量已定型，lb 即该指派的精确总代价
        return
    for v in _values(domains[var]):
        child = list(domains)
        child[var] = 1 << v
        _bnb(child, model, var_costs, budget, best)


def _bounded_sat(domains, model: _Model, var_costs, budget: _Budget, bound: int) -> bool:
    """是否存在总代价 <= bound 的可行指派。"""
    budget.tick()
    if not _propagate(domains, model):
        return False
    lb = 0
    var = -1
    best_pc = 4
    for i, mask in enumerate(domains):
        if mask == 0:
            return False
        lb += _domain_min_cost(mask, var_costs[i])
        pc = mask.bit_count()
        if 1 < pc < best_pc:
            best_pc = pc
            var = i
    if lb > bound:
        return False
    if var < 0:
        return True
    for v in _values(domains[var]):
        child = list(domains)
        child[var] = 1 << v
        if _bounded_sat(child, model, var_costs, budget, bound):
            return True
    return False


def _lex_witness(domains, model: _Model, var_costs, budget: _Budget, bound: int, order):
    """按固定变量序（实个体按 ID 字典序在前）、值升序找第一个代价 <= bound 的解。

    第一个找到的完整指派即“人物 ID 序列上基因型字典序最小”的最优见证。
    """
    budget.tick()
    if not _propagate(domains, model):
        return None
    if any(mask == 0 for mask in domains):
        return None
    if _lower_bound(domains, var_costs) > bound:
        return None
    var = -1
    for i in order:
        if domains[i].bit_count() > 1:
            var = i
            break
    if var < 0:
        return [mask.bit_length() - 1 for mask in domains]
    for v in _values(domains[var]):
        child = list(domains)
        child[var] = 1 << v
        sol = _lex_witness(child, model, var_costs, budget, bound, order)
        if sol is not None:
            return sol
    return None


def optimize(pedigree, observations, costs=None, *, node_budget: int = DEFAULT_NODE_BUDGET) -> OptimizeResult:
    """带软代价的单基因座孟德尔家系优化。契约见模块 docstring。"""
    model = _validate_and_build(pedigree, observations, node_budget)
    cost_map = _validate_costs(pedigree, costs)
    n_real = len(model.ids)
    n_var = len(model.domains0)
    var_costs = [
        cost_map.get(model.ids[i], (0, 0, 0)) if i < n_real else (0, 0, 0)
        for i in range(n_var)
    ]
    # 硬观测先限制候选，再剔除禁止项；两者冲突 -> 域为空 -> 证明无解。
    domains = list(model.domains0)
    for i in range(n_real):
        pid = model.ids[i]
        mask = 0
        for v in GENOTYPES:
            if var_costs[i][v] is not None:
                mask |= 1 << v
        if pid in model.obs:
            mask &= 1 << model.obs[pid]
        domains[i] = mask

    budget = _Budget(node_budget)
    best = [None]
    proven = False
    try:
        _bnb(list(domains), model, var_costs, budget, best)
        proven = True
        if best[0] is None:
            return OptimizeResult(
                status="infeasible",
                min_cost=None,
                optimal_genotypes=None,
                witness=None,
                best_cost_bound=None,
                nodes_used=budget.used,
            )
        bound = best[0]
        base = list(domains)
        if not _propagate(base, model):  # 已知可行，传播不可能失败
            raise AssertionError("internal: propagation failed on a feasible instance")
        optimal = {}
        for i in range(n_real):
            vals = set()
            for v in _values(base[i]):  # 弧一致已剪掉的值必然不在任何最优解中
                trial = list(base)
                trial[i] = 1 << v
                if _bounded_sat(trial, model, var_costs, budget, bound):
                    vals.add(v)
            optimal[model.ids[i]] = frozenset(vals)
        order = sorted(range(n_real), key=lambda i: model.ids[i]) + list(
            range(n_real, n_var)
        )
        sol = _lex_witness(list(domains), model, var_costs, budget, bound, order)
        if sol is None:  # 最优性已证明，必有解
            raise AssertionError("internal: no optimal witness found")
        witness = {pid: sol[model.index[pid]] for pid in model.ids}
        return OptimizeResult(
            status="optimal",
            min_cost=bound,
            optimal_genotypes=optimal,
            witness=witness,
            best_cost_bound=bound,
            nodes_used=budget.used,
        )
    except _BudgetExhausted:
        # 区分：best[0] 是可行上界；仅当分支定界已完整结束（proven）时
        # 它才是已证明最优。optimal_genotypes/witness 绝不给出未证明结果。
        return OptimizeResult(
            status="unknown",
            min_cost=best[0] if proven else None,
            optimal_genotypes=None,
            witness=None,
            best_cost_bound=best[0],
            nodes_used=node_budget,
        )
