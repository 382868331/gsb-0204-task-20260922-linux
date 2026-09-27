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

另提供候选亲子关系解析：

    resolve(pedigree, observations, parent_candidates, *, node_budget=...) -> PlacementResult

- parent_candidates: dict，键为 (子女ID, "father"|"mother")，值为候选父母 ID
  的非空元组/列表。对应槽位在 pedigree 中必须为 None（未确定）。
  每个候选必须是家系内成员——同一个人在整个家系中只有一份身份与观测，
  被选为父母后其观测与亲缘约束在所有关系里同时生效，不做按槽位的独立解释。
- resolve 枚举候选的全部完整组合：会形成祖先循环的组合被排除（记入 cyclic），
  其余组合逐一在共享节点预算内检验孟德尔一致性。
- 返回 PlacementResult：
    - status == "feasible":   choices 为全部与观测相容的完整选择（每个为
                              {(子女ID, 角色): 父母ID} 字典，确定性排序）；
    - status == "infeasible": 全部组合已穷尽且无一相容（证明不可能），
                              choices 为空元组；
    - status == "unknown":    节点预算耗尽，搜索尚未完成，choices 为 None；
    - cyclic:                 因形成祖先循环而被排除的完整选择（与预算无关，
                              始终完整）；unknown 时 choices 无法保证完整故为 None。
  组合总数（各候选列表长度的乘积）超过 MAX_CHOICE_COMBINATIONS 时抛出
  ValidationError（输入规模上限，与 MAX_INDIVIDUALS 同类）。

所有非法输入抛出 ValidationError（ValueError 子类），消息定位到具体
个体 ID / 字段。本库不做任何静默纠正，输出全部为精确离散结果，
不涉及浮点计算。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from typing import Optional

__all__ = [
    "check",
    "resolve",
    "CheckResult",
    "PlacementResult",
    "ConflictReport",
    "MendelError",
    "ValidationError",
    "MAX_INDIVIDUALS",
    "DEFAULT_NODE_BUDGET",
    "MAX_MINIMAL_SETS",
    "MAX_CHOICE_COMBINATIONS",
    "GENOTYPES",
]

MAX_INDIVIDUALS = 20
DEFAULT_NODE_BUDGET = 100_000
MAX_MINIMAL_SETS = 256
MAX_CHOICE_COMBINATIONS = 100_000
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
class PlacementResult:
    """resolve() 的返回值。status ∈ {"feasible", "infeasible", "unknown"}。

    choices: 与观测相容的全部完整选择（{(子女ID, 角色): 父母ID} 字典元组）；
             unknown 时为 None。infeasible 时为空元组。
    cyclic:  因形成祖先循环被排除的完整选择（同结构元组），始终完整枚举。
    """

    status: str
    choices: Optional[tuple]
    cyclic: tuple
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


# ---------------------------------------------------------------- 候选亲子关系解析

_ROLES = ("father", "mother")


def _validate_candidates(pedigree: Mapping, parent_candidates) -> list:
    """校验 parent_candidates，返回 [(子女ID, 角色, 候选ID元组)]，确定性排序。"""
    if isinstance(parent_candidates, (str, bytes)) or not isinstance(
        parent_candidates, Mapping
    ):
        raise ValidationError(
            "parent_candidates must be a mapping {(child_id, role): (candidate_ids...)}, "
            f"got {type(parent_candidates).__name__}"
        )
    slots = []
    total = 1
    for key, cands in parent_candidates.items():
        if (
            isinstance(key, (str, bytes))
            or not isinstance(key, (tuple, list))
            or len(key) != 2
        ):
            raise ValidationError(
                f"parent_candidates key must be a (child_id, role) pair, got {key!r}"
            )
        child, role = key
        if not isinstance(child, str) or child not in pedigree:
            raise ValidationError(
                f"parent_candidates key {key!r}: child {child!r} is not a pedigree member"
            )
        if role not in _ROLES:
            raise ValidationError(
                f"parent_candidates key {key!r}: role must be 'father' or 'mother', "
                f"got {role!r}"
            )
        if pedigree[child][_ROLES.index(role)] is not None:
            raise ValidationError(
                f"parent_candidates key {key!r}: pedigree[{child!r}] already specifies "
                f"a {role}; candidates require an unspecified (None) slot"
            )
        if (
            isinstance(cands, (str, bytes))
            or not isinstance(cands, (tuple, list))
            or len(cands) == 0
        ):
            raise ValidationError(
                f"parent_candidates[{key!r}] must be a non-empty tuple/list of "
                f"candidate ids, got {cands!r}"
            )
        seen = set()
        for cand in cands:
            if not isinstance(cand, str) or cand not in pedigree:
                raise ValidationError(
                    f"parent_candidates[{key!r}] candidate {cand!r} is not a "
                    "pedigree member"
                )
            if cand == child:
                raise ValidationError(
                    f"parent_candidates[{key!r}] candidate {cand!r} is the child itself"
                )
            if cand in seen:
                raise ValidationError(
                    f"parent_candidates[{key!r}] lists candidate {cand!r} more than once"
                )
            seen.add(cand)
        ordered = tuple(sorted(seen))
        total *= len(ordered)
        if total > MAX_CHOICE_COMBINATIONS:
            raise ValidationError(
                f"parent_candidates yield more than {MAX_CHOICE_COMBINATIONS} "
                "complete combinations (input size limit)"
            )
        slots.append((child, role, ordered))
    slots.sort(key=lambda s: (s[0], s[1]))
    return slots


def _apply_choice(pedigree: Mapping, slots: list, picked: tuple) -> dict:
    """按一组选择填充候选槽位，返回新的具体 pedigree 字典。"""
    concrete = {pid: [fa, mo] for pid, (fa, mo) in pedigree.items()}
    for (child, role, _cands), parent in zip(slots, picked):
        concrete[child][_ROLES.index(role)] = parent
    return {pid: (fa, mo) for pid, (fa, mo) in concrete.items()}


def resolve(
    pedigree,
    observations,
    parent_candidates,
    *,
    node_budget: int = DEFAULT_NODE_BUDGET,
) -> PlacementResult:
    """枚举候选亲子关系的全部完整选择，找出与观测相容者。契约见模块 docstring。"""
    _validate_and_build(pedigree, observations, node_budget)  # 复用同一输入契约
    slots = _validate_candidates(pedigree, parent_candidates)

    # 先完整枚举组合并排除祖先循环（不消耗搜索预算），再逐一检验孟德尔一致性。
    cyclic = []
    pending = []
    for picked in product(*(s[2] for s in slots)) if slots else [()]:
        concrete = _apply_choice(pedigree, slots, picked)
        try:
            _check_acyclic(concrete)
        except ValidationError:
            cyclic.append(dict(zip(((s[0], s[1]) for s in slots), picked)))
        else:
            pending.append(concrete)

    budget = _Budget(node_budget)
    choices = []
    try:
        for concrete in pending:
            model = _validate_and_build(concrete, observations, node_budget)
            if _solve_obs(model, model.obs_ids, budget) is not None:
                choices.append(concrete)
    except _BudgetExhausted:
        return PlacementResult(
            status="unknown",
            choices=None,
            cyclic=tuple(cyclic),
            nodes_used=node_budget,
        )

    def _choice_of(concrete):
        return {
            (child, role): concrete[child][_ROLES.index(role)]
            for child, role, _cands in slots
        }

    chosen = tuple(_choice_of(c) for c in choices)
    return PlacementResult(
        status="feasible" if chosen else "infeasible",
        choices=chosen,
        cyclic=tuple(cyclic),
        nodes_used=budget.used,
    )
