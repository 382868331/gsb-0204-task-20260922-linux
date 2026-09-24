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

第二入口（多位点候选父母对分配）：

    assign_parents(genotypes, candidates, *, node_budget=DEFAULT_NODE_BUDGET)
        -> AssignmentResult

- genotypes: dict，个体 ID -> {位点键: 基因型}。位点键为非 bool 的 int
  或非空 str，全体位点共 1..MAX_ASSIGN_LOCI=4 个；基因型规则同 check。
  父母可只在部分位点有观测（未观测位点取 0/1/2），也可完全不出现；
  每个孩子必须在全部位点上有观测。
- candidates: dict，孩子 ID -> 1..MAX_ASSIGN_PAIRS=3 个候选父母对，
  每对为 (father_id_or_None, mother_id_or_None)；孩子数 1..MAX_ASSIGN_CHILDREN=3。
  None 槽位语义同 check（独立未知创始者）。候选父母不能是另一个被分配
  的孩子（本接口只覆盖两层分配，多代请用 check）。
- 每个孩子选定**一对**父母，该选择对全部位点生效，不允许逐位点换父母。
  至多 3^3=27 种组合，全部枚举；组合内各位点独立调用旧遗传规则求解
  （未观测共享父母的耦合仍在单个位点内精确处理）。
- 返回 AssignmentResult：
    status="feasible":   assignments 为全部可行选择 ParentAssignment（按
                         候选下标元组字典序），每个含 choice/pairs/空冲突；
    status="infeasible": 没有任何选择可行；conflicts 给出每种被否决选择
                         及其不一致位点元组 loci_conflicts（非空）；
    status="unknown":    节点预算耗尽，assignments/conflicts 均为空元组。
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "check",
    "CheckResult",
    "ConflictReport",
    "assign_parents",
    "AssignmentResult",
    "ParentAssignment",
    "MendelError",
    "ValidationError",
    "MAX_INDIVIDUALS",
    "DEFAULT_NODE_BUDGET",
    "MAX_MINIMAL_SETS",
    "MAX_ASSIGN_CHILDREN",
    "MAX_ASSIGN_PAIRS",
    "MAX_ASSIGN_LOCI",
    "GENOTYPES",
]

MAX_INDIVIDUALS = 20
DEFAULT_NODE_BUDGET = 100_000
MAX_MINIMAL_SETS = 256
GENOTYPES = (0, 1, 2)

# 候选父母对分配的题面上限。
MAX_ASSIGN_CHILDREN = 3
MAX_ASSIGN_PAIRS = 3
MAX_ASSIGN_LOCI = 4

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
class ParentAssignment:
    """assign_parents() 中一个孩子的父母选择。"""

    child: str
    pair_index: int                     # 该孩子候选列表中的下标（0 起）
    pair: tuple                         # (father_id_or_None, mother_id_or_None)
    loci_conflicts: tuple = ()          # 该选择被否决时：不一致位点键元组；可行时为空


@dataclass(frozen=True)
class AssignmentResult:
    """assign_parents() 的返回值。status ∈ {"feasible", "infeasible", "unknown"}。"""

    status: str
    assignments: tuple   # feasible: ParentAssignment 元组的元组（全部可行组合）
    conflicts: tuple     # infeasible: 每个被否决组合一条（每孩一个 ParentAssignment）
    loci: tuple          # 参与校验的全部位点键，按规范顺序
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


def _validate_budget(node_budget) -> None:
    if isinstance(node_budget, bool) or not isinstance(node_budget, int):
        raise ValidationError(
            f"node_budget must be an int >= 1 (bool/float rejected), got {node_budget!r}"
        )
    if node_budget < 1:
        raise ValidationError(f"node_budget must be >= 1, got {node_budget!r}")


def _validate_and_build(pedigree, observations, node_budget) -> _Model:
    _validate_budget(node_budget)

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


# ------------------------------------------------ 多位点候选父母对分配

def _locus_sort_key(locus):
    # int 位点按数值在前，str 位点按字典序在后；同类型内自然排序。
    return (0, locus) if isinstance(locus, int) else (1, locus)


def _validate_assignment_input(genotypes, candidates):
    """校验 assign_parents 输入，返回 (child_ids, loci, pair_lists, genotype_map)。

    child_ids 升序；loci 为规范顺序的位点元组；pair_lists 为 {child: [(fa,mo),...]}；
    genotype_map 为 {id: {locus: g}}（已是合法 genotype dict 视图）。
    """
    if isinstance(genotypes, (str, bytes)) or not isinstance(genotypes, Mapping):
        raise ValidationError(
            f"genotypes must be a mapping {{id: {{locus: genotype}}}}, "
            f"got {type(genotypes).__name__}"
        )
    if not genotypes:
        raise ValidationError("genotypes must contain at least one genotyped individual")
    for pid in genotypes:
        if not isinstance(pid, str) or not pid:
            raise ValidationError(f"individual id must be a non-empty str, got {pid!r}")

    loci = set()
    gmap = {}
    for pid, prof in genotypes.items():
        if isinstance(prof, (str, bytes)) or not isinstance(prof, Mapping):
            raise ValidationError(
                f"genotypes[{pid!r}] must be a mapping {{locus: 0|1|2}}, "
                f"got {type(prof).__name__}"
            )
        if not prof:
            raise ValidationError(f"genotypes[{pid!r}] has no locus observations")
        row = {}
        for locus, g in prof.items():
            if isinstance(locus, bool) or not isinstance(locus, (int, str)):
                raise ValidationError(
                    f"genotypes[{pid!r}] locus key must be an int or non-empty str, "
                    f"got {locus!r}"
                )
            if isinstance(locus, str) and not locus:
                raise ValidationError(f"genotypes[{pid!r}] has an empty str locus key")
            if locus in row:
                raise ValidationError(f"genotypes[{pid!r}] lists locus {locus!r} twice")
            if isinstance(g, bool) or not isinstance(g, int):
                raise ValidationError(
                    f"genotypes[{pid!r}][{locus!r}] genotype must be an int in "
                    f"{{0, 1, 2}} (bool/float rejected), got {g!r}"
                )
            if g not in GENOTYPES:
                raise ValidationError(
                    f"genotypes[{pid!r}][{locus!r}] genotype must be in {{0, 1, 2}}, "
                    f"got {g!r}"
                )
            row[locus] = g
            loci.add(locus)
        gmap[pid] = row
    if len(loci) > MAX_ASSIGN_LOCI:
        raise ValidationError(
            f"got {len(loci)} loci, limit is {MAX_ASSIGN_LOCI}"
        )
    loci_tuple = tuple(sorted(loci, key=_locus_sort_key))

    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Mapping):
        raise ValidationError(
            f"candidates must be a mapping {{child_id: [(father, mother), ...]}}, "
            f"got {type(candidates).__name__}"
        )
    if not 1 <= len(candidates) <= MAX_ASSIGN_CHILDREN:
        raise ValidationError(
            f"number of children must be in 1..{MAX_ASSIGN_CHILDREN}, "
            f"got {len(candidates)}"
        )
    child_ids = sorted(candidates)
    pair_lists = {}
    for cid in child_ids:
        if not isinstance(cid, str) or not cid:
            raise ValidationError(f"child id must be a non-empty str, got {cid!r}")
        pairs = candidates[cid]
        if isinstance(pairs, (str, bytes)) or not isinstance(pairs, (tuple, list)):
            raise ValidationError(
                f"candidates[{cid!r}] must be a list of (father, mother) pairs, "
                f"got {pairs!r}"
            )
        if not 1 <= len(pairs) <= MAX_ASSIGN_PAIRS:
            raise ValidationError(
                f"candidates[{cid!r}] must have 1..{MAX_ASSIGN_PAIRS} pairs, "
                f"got {len(pairs)}"
            )
        plist = []
        seen = set()
        for j, pair in enumerate(pairs):
            if (
                isinstance(pair, (str, bytes))
                or not isinstance(pair, (tuple, list))
                or len(pair) != 2
            ):
                raise ValidationError(
                    f"candidates[{cid!r}][{j}] must be a (father, mother) pair, "
                    f"got {pair!r}"
                )
            fa, mo = pair
            for role, par in (("father", fa), ("mother", mo), ):
                if par is not None and not isinstance(par, str):
                    raise ValidationError(
                        f"candidates[{cid!r}][{j}] {role} must be None or a "
                        f"member id str, got {par!r}"
                    )
                if par == "":
                    raise ValidationError(
                        f"candidates[{cid!r}][{j}] {role} is an empty id"
                    )
            if fa == cid or mo == cid:
                raise ValidationError(
                    f"candidates[{cid!r}][{j}] lists the child itself as a parent"
                )
            t = (fa, mo)
            if t in seen:
                raise ValidationError(
                    f"candidates[{cid!r}] contains duplicate pair {list(t)}"
                )
            seen.add(t)
            plist.append(t)
        pair_lists[cid] = plist

    child_set = set(child_ids)
    for cid in child_ids:
        if cid not in gmap:
            raise ValidationError(
                f"child {cid!r} has no genotype observations (children must be "
                f"observed at every locus)"
            )
        missing = [l for l in loci_tuple if l not in gmap[cid]]
        if missing:
            raise ValidationError(
                f"child {cid!r} is missing observations at loci {list(missing)}; "
                f"children must be observed at every locus"
            )
        for j, (fa, mo) in enumerate(pair_lists[cid]):
            if fa in child_set or mo in child_set:
                par = fa if fa in child_set else mo
                raise ValidationError(
                    f"candidates[{cid!r}][{j}] parent {par!r} is itself an "
                    f"assigned child (multi-generation assignment is not supported; "
                    f"use check())"
                )
    return child_ids, loci_tuple, pair_lists, gmap


def _locus_model(child_ids, chosen, gmap, locus):
    """为给定候选组合构造单位点的 (pedigree, observations)。

    chosen: {child: (fa, mo)}。父母无论是否观测都是家系创始者。
    """
    pedigree = {}
    for cid in child_ids:
        fa, mo = chosen[cid]
        pedigree[cid] = (fa, mo)
        if fa is not None and fa not in pedigree:
            pedigree[fa] = (None, None)
        if mo is not None and mo not in pedigree:
            pedigree[mo] = (None, None)
    observations = {}
    for pid in pedigree:  # 只观测家系成员；未被引用的基因型个体与该组合无关
        row = gmap.get(pid)
        if row is not None and locus in row:
            observations[pid] = row[locus]
    return pedigree, observations


def _locus_feasible(child_ids, chosen, gmap, locus, budget) -> bool:
    pedigree, observations = _locus_model(child_ids, chosen, gmap, locus)
    model = _validate_and_build(pedigree, observations, budget.limit)
    return _solve_obs(model, model.obs_ids, budget) is not None


def assign_parents(genotypes, candidates, *, node_budget: int = DEFAULT_NODE_BUDGET):
    """在有限候选父母对中为每个孩子选择同一对（跨全部位点）父母。

    最多 3 个孩子、每人 3 组候选、4 个位点；枚举至多 27 种组合，逐组合
    校验全部位点。契约见模块 docstring。
    """
    _validate_budget(node_budget)
    child_ids, loci_tuple, pair_lists, gmap = _validate_assignment_input(
        genotypes, candidates
    )
    budget = _Budget(node_budget)
    try:
        feasible_combos = []
        ranges = [range(len(pair_lists[cid])) for cid in child_ids]
        for idxs in itertools.product(*ranges):
            chosen = {
                cid: pair_lists[cid][k] for cid, k in zip(child_ids, idxs)
            }
            ok = True
            for locus in loci_tuple:
                if not _locus_feasible(child_ids, chosen, gmap, locus, budget):
                    ok = False
                    break
            if ok:
                feasible_combos.append(
                    tuple(
                        ParentAssignment(
                            child=cid,
                            pair_index=k,
                            pair=chosen[cid],
                            loci_conflicts=(),
                        )
                        for cid, k in zip(child_ids, idxs)
                    )
                )
        if feasible_combos:
            return AssignmentResult(
                status="feasible",
                assignments=tuple(feasible_combos),
                conflicts=(),
                loci=loci_tuple,
                nodes_used=budget.used,
            )

        # 全部组合被否决：对每个被否决组合做逐位点归因。孩子三元组只有在
        # 共享被选父母时才互相耦合，故枚举孩子的全部子集（至多 2^3=8 个），
        # 找出使该位点不可行的极小孩子子集（极小不可行核）：独立三元组冲突
        # 时只标记孩子本人，共享父母导致的联合冲突则同时标记相关孩子。
        rejected = []
        n_children = len(child_ids)
        ranges = [range(len(pair_lists[cid])) for cid in child_ids]
        for idxs in itertools.product(*ranges):
            chosen = {
                cid: pair_lists[cid][k] for cid, k in zip(child_ids, idxs)
            }
            per_child_bad = {cid: [] for cid in child_ids}
            for locus in loci_tuple:
                infeasible_subsets = []
                for mask in range(1, 1 << n_children):
                    sub = [child_ids[b] for b in range(n_children) if mask & (1 << b)]
                    chosen_t = {c: chosen[c] for c in sub}
                    ped_t, obs_t = _locus_model(sub, chosen_t, gmap, locus)
                    m_t = _validate_and_build(ped_t, obs_t, budget.limit)
                    if _solve_obs(m_t, m_t.obs_ids, budget) is None:
                        infeasible_subsets.append(frozenset(sub))
                if not infeasible_subsets:
                    continue  # 该位点在此组合下可行
                cores = [
                    s for s in infeasible_subsets
                    if not any(t < s for t in infeasible_subsets)
                ]
                for core in cores:
                    for cid in core:
                        if locus not in per_child_bad[cid]:
                            per_child_bad[cid].append(locus)
            rejected.append(
                tuple(
                    ParentAssignment(
                        child=cid,
                        pair_index=k,
                        pair=chosen[cid],
                        loci_conflicts=tuple(per_child_bad[cid]),
                    )
                    for cid, k in zip(child_ids, idxs)
                )
            )
        return AssignmentResult(
            status="infeasible",
            assignments=(),
            conflicts=tuple(rejected),
            loci=loci_tuple,
            nodes_used=budget.used,
        )
    except _BudgetExhausted:
        return AssignmentResult(
            status="unknown",
            assignments=(),
            conflicts=(),
            loci=loci_tuple,
            nodes_used=node_budget,
        )
