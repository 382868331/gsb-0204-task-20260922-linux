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

候选亲子关系（resolve_links）：
pedigree 的某个亲本槽位除可填 str（确定）或 None（缺失=独立未知创始者）外，
还可填一个非空 str 序列 [id1, id2, ...]，表示该父亲/母亲“尚未确定，但只可能
来自给定名单”。resolve_links 枚举所有不确定槽位的完整组合，逐组：
(1) 排除会形成祖先循环的选择；(2) 用与 check 完全相同的传播+搜索检验孟德尔
一致性；(3) 记录相容选择的完整亲本指派与一份完整基因型见证。同一个体在所有
关系约束中共享同一个变量，身份与观测天然全局一致。状态三档严格分开：
“infeasible”=全部选择均被证伪（已确认不可能）；“feasible”=搜索穷尽且至少
一组相容；“unknown”=节点预算耗尽、搜索未完成（可同时返回已找到的部分结果）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "check",
    "resolve_links",
    "CheckResult",
    "ConflictReport",
    "FeasibleLink",
    "LinkResult",
    "MendelError",
    "ValidationError",
    "MAX_INDIVIDUALS",
    "DEFAULT_NODE_BUDGET",
    "MAX_MINIMAL_SETS",
    "MAX_LINK_SELECTIONS",
    "GENOTYPES",
]

MAX_INDIVIDUALS = 20
DEFAULT_NODE_BUDGET = 100_000
MAX_MINIMAL_SETS = 256
MAX_LINK_SELECTIONS = 256
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
class FeasibleLink:
    """一组相容的完整亲本选择及其基因型见证。"""

    selection: tuple          # 与 LinkResult.slots 对齐的被选亲本 ID 元组
    parents: dict             # {子女ID: (父亲ID|None, 母亲ID|None)} 完整指派
    assignment: dict          # {实个体ID: 基因型} 的一份完整可行见证


@dataclass(frozen=True)
class LinkResult:
    """resolve_links() 的返回值。status ∈ {"feasible", "infeasible", "unknown"}。"""

    status: str
    slots: tuple              # 不确定亲本槽位 (子女ID, role)，确定顺序
    feasible: tuple           # FeasibleLink 元组（DFS 确定性顺序）
    possible: dict            # {子女ID: {role: frozenset(在某相容选择中出现过的候选)}}
    possible_genotypes: dict  # {个体ID: frozenset(在某相容见证中出现过的基因型)}
    truncated: bool           # 相容选择数达到 max_feasible 上限而提前停止
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
    _validate_budget(node_budget)
    concrete, _choices, ids = _parse_pedigree(pedigree, allow_choices=False)
    obs = _validate_observations(observations, ids)
    return _build_model(ids, concrete, obs)


def _validate_budget(node_budget) -> None:
    if isinstance(node_budget, bool) or not isinstance(node_budget, int):
        raise ValidationError(
            f"node_budget must be an int >= 1 (bool/float rejected), got {node_budget!r}"
        )
    if node_budget < 1:
        raise ValidationError(f"node_budget must be >= 1, got {node_budget!r}")


def _parse_pedigree(pedigree, *, allow_choices: bool):
    """校验家系形状，返回 (concrete, choices, ids)。

    concrete[cid] 为 (fa|None, mo|None)：确定槽位为 str，不确定槽位与缺失
    亲本均暂为 None（枚举时再替换不确定槽位）。
    choices[(cid, role)] 为去重并按首次出现排序的候选 ID 元组。
    allow_choices=False（check 路径）时，候选序列被明确拒绝。
    """
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

    concrete = {}
    choices = {}
    for pid, parents in pedigree.items():
        if (
            isinstance(parents, (str, bytes))
            or not isinstance(parents, (tuple, list))
            or len(parents) != 2
        ):
            raise ValidationError(
                f"pedigree[{pid!r}] must be a (father, mother) pair, got {parents!r}"
            )
        slots = []
        for role, par in zip(("father", "mother"), parents):
            if par is None:
                slots.append(None)
                continue
            if isinstance(par, str):
                if par not in pedigree:
                    raise ValidationError(
                        f"pedigree[{pid!r}] {role} {par!r} is not a pedigree member"
                    )
                if par == pid:
                    raise ValidationError(f"pedigree[{pid!r}] lists itself as {role}")
                slots.append(par)
                continue
            if isinstance(par, (bytes, bytearray)) or not isinstance(par, (tuple, list)):
                raise ValidationError(
                    f"pedigree[{pid!r}] {role} must be None, a member id str, or a "
                    f"non-empty candidate list, got {par!r}"
                )
            if not allow_choices:
                raise ValidationError(
                    f"pedigree[{pid!r}] {role}: candidate parent lists require "
                    f"resolve_links(); check() accepts only a single id or None, got {par!r}"
                )
            if len(par) == 0:
                raise ValidationError(
                    f"pedigree[{pid!r}] {role} candidate list must be non-empty"
                )
            ordered = []
            for cand in par:
                if not isinstance(cand, str):
                    raise ValidationError(
                        f"pedigree[{pid!r}] {role} candidate must be a member id str, "
                        f"got {cand!r}"
                    )
                if cand not in pedigree:
                    raise ValidationError(
                        f"pedigree[{pid!r}] {role} candidate {cand!r} is not a pedigree member"
                    )
                if cand == pid:
                    raise ValidationError(
                        f"pedigree[{pid!r}] {role} candidate lists the individual itself"
                    )
                if cand not in ordered:
                    ordered.append(cand)
            choices[(pid, role)] = tuple(ordered)
            slots.append(None)
        concrete[pid] = tuple(slots)
    # 仅由确定边构成的图若有环，则没有任何选择能挽救：直接拒绝并定位环。
    _check_acyclic(concrete)
    return concrete, choices, list(pedigree.keys())


def _validate_observations(observations, ids):
    if isinstance(observations, (str, bytes)) or not isinstance(observations, Mapping):
        raise ValidationError(
            f"observations must be a mapping {{id: genotype}}, got {type(observations).__name__}"
        )
    obs = {}
    for pid, g in observations.items():
        if not isinstance(pid, str) or pid not in ids:
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
    return obs


def _build_model(ids, concrete, obs) -> _Model:
    """由所有槽位均已具体的家系与观测构建求解模型（输入已校验）。"""
    index = {pid: i for i, pid in enumerate(ids)}
    model = _Model()
    model.ids = ids
    model.index = index
    model.domains0 = [_FULL_MASK] * len(ids)
    model.adjacency = [[] for _ in ids]

    def synth_founder() -> int:
        idx = len(model.domains0)
        model.domains0.append(_FULL_MASK)
        model.adjacency.append([])
        return idx

    for cid, (fa, mo) in concrete.items():
        if fa is None and mo is None:
            continue  # 创始者，无三元约束
        fi = index[fa] if fa is not None else synth_founder()
        mi = index[mo] if mo is not None else synth_founder()
        ci = index[cid]
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

def _creates_cycle(concrete, child: str, parent: str) -> bool:
    """在 child->parent 这条边加入后是否成环（沿当前已选亲本边向上可达）。"""
    stack = [parent]
    seen = set()
    while stack:
        node = stack.pop()
        if node == child:
            return True
        if node in seen:
            continue
        seen.add(node)
        fa, mo = concrete[node]
        if fa is not None:
            stack.append(fa)
        if mo is not None:
            stack.append(mo)
    return False


def resolve_links(
    pedigree,
    observations,
    *,
    node_budget: int = DEFAULT_NODE_BUDGET,
    max_feasible: int = MAX_LINK_SELECTIONS,
) -> LinkResult:
    """枚举不确定亲本槽位的完整选择，返回与观测相容者。

    pedigree 中亲本槽位可为 None / str / 非空候选 str 序列。缺失亲本 None
    仍表示独立未知创始者，不会与候选名单混淆。同一个体在所有三元关系约束中
    共享同一变量与观测。会形成祖先循环的完整选择在枚举中剪枝排除。

    返回 LinkResult：
    - "infeasible": 全部完整选择均已检验且无一相容（确认不可能）；
    - "feasible":   搜索穷尽且至少一组相容（feasible 给出全部相容选择，
                    达到 max_feasible 上限提前停止时 truncated=True）；
    - "unknown":    node_budget 耗尽、搜索未完成；feasible/possible 为已找到的
                    部分结果，不据此断言穷尽。
    """
    _validate_budget(node_budget)
    if isinstance(max_feasible, bool) or not isinstance(max_feasible, int) or max_feasible < 1:
        raise ValidationError(
            f"max_feasible must be an int >= 1 (bool rejected), got {max_feasible!r}"
        )
    concrete, choices, ids = _parse_pedigree(pedigree, allow_choices=True)
    obs = _validate_observations(observations, ids)

    slots = tuple((cid, role) for cid in ids for role in ("father", "mother")
                  if (cid, role) in choices)
    budget = _Budget(node_budget)
    found = []
    possible = {cid: {"father": set(), "mother": set()} for cid in ids}
    possible_genotypes = {cid: set() for cid in ids}
    truncated = False

    def record(concrete_sel, sol):
        picked = []
        parents = {}
        for cid in ids:
            fa, mo = concrete_sel[cid]
            parents[cid] = (fa, mo)
        for cid, role in slots:
            picked_id = parents[cid][0 if role == "father" else 1]
            picked.append(picked_id)
            possible[cid][role].add(picked_id)
        assignment = {cid: sol[k] for k, cid in enumerate(ids)}
        for cid, g in assignment.items():
            possible_genotypes[cid].add(g)
        found.append(FeasibleLink(
            selection=tuple(picked),
            parents=parents,
            assignment=assignment,
        ))

    def search(depth: int) -> bool:
        """返回 True 表示达到 max_feasible 上限、提前停止。"""
        budget.tick()
        if depth == len(slots):
            model = _build_model(ids, concrete, obs)
            sol = _solve_obs(model, model.obs_ids, budget)
            if sol is not None:
                record(concrete, sol)
                if len(found) >= max_feasible:
                    return True
            return False
        cid, role = slots[depth]
        pos = 0 if role == "father" else 1
        old = concrete[cid][pos]
        for cand in choices[(cid, role)]:
            if _creates_cycle(concrete, cid, cand):
                continue
            concrete[cid] = (
                (cand, concrete[cid][1]) if pos == 0 else (concrete[cid][0], cand)
            )
            stop = search(depth + 1)
            concrete[cid] = (
                (old, concrete[cid][1]) if pos == 0 else (concrete[cid][0], old)
            )
            if stop:
                return True
        return False

    try:
        capped = search(0)
        if not found:
            status = "infeasible"
        else:
            status = "feasible"
        truncated = capped
    except _BudgetExhausted:
        status = "unknown"

    possible_out = {
        cid: {role: frozenset(vals) for role, vals in by_role.items() if vals}
        for cid, by_role in possible.items()
        if any(by_role.values())
    }
    return LinkResult(
        status=status,
        slots=slots,
        feasible=tuple(found),
        possible=possible_out,
        possible_genotypes={cid: frozenset(vals) for cid, vals in
                            possible_genotypes.items() if vals},
        truncated=truncated,
        nodes_used=budget.used,
    )


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
