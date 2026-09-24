"""mendelcheck — 孟德尔家系单基因座最小冲突定位 / 多位点候选父母判定。

契约（固定公开接口）：

1. check(pedigree, observations, *, node_budget=DEFAULT_NODE_BUDGET) -> CheckResult

   单基因座家系校验与最小冲突定位（详见下文）。

2. assign_parents(profiles, candidates, *, node_budget=DEFAULT_NODE_BUDGET)
       -> AssignmentResult

   多位点候选父母判定：每个孩子给至多 MAX_CANDIDATE_PAIRS=3 组候选
   （父亲, 母亲）有序对，位点至多 MAX_LOCI=4 个、孩子至多
   MAX_CHILDREN=3 个。枚举“每个孩子选定一对”的全部组合（至多 27 个），
   每个组合在**全部位点**上用与 check 完全相同的孟德尔规则判定；
   一个组合一经选定即用于该孩子的全部位点，**不允许逐位点更换父母**。
   返回全部可行分配；不存在时按联合分配给出逐位点冲突诊断。

   - profiles: dict，ID（非空 str，至多 20 个）-> {位点名（非空 str）:
     基因型 int 0/1/2}。所有个体必须给出**完全相同**的位点集合
     （1..4 个）。
   - candidates: dict，孩子 ID（必须有 profile）-> 非空的 (父亲, 母亲)
     有序对序列（1..3 对，不可重复；父母均须有 profile；不支持 None
     候选——无法对档案外的创始者做检验）。孩子 1..3 个；任何联合选择
     形成的亲子图都必须无环（含自环）。
   - 返回 AssignmentResult（字段与状态语义见该 dataclass）。
   - 非法输入抛 ValidationError；纯函数，不修改入参、不保留状态。

check 的契约：

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
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "check",
    "assign_parents",
    "CheckResult",
    "ConflictReport",
    "AssignmentResult",
    "AssignmentConflict",
    "MendelError",
    "ValidationError",
    "MAX_INDIVIDUALS",
    "DEFAULT_NODE_BUDGET",
    "MAX_MINIMAL_SETS",
    "MAX_CHILDREN",
    "MAX_CANDIDATE_PAIRS",
    "MAX_LOCI",
    "GENOTYPES",
]

MAX_INDIVIDUALS = 20
DEFAULT_NODE_BUDGET = 100_000
MAX_MINIMAL_SETS = 256
GENOTYPES = (0, 1, 2)

# assign_parents 的题面规模上限。
MAX_CHILDREN = 3
MAX_CANDIDATE_PAIRS = 3
MAX_LOCI = 4

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
class AssignmentConflict:
    """一个完整联合候选分配未能通过全部位点时的逐位点诊断。

    - assignment: {孩子ID: (父亲ID, 母亲ID)} —— 每个孩子选定的同一对
      候选父母贯穿全部位点（不允许逐位点更换）；
    - feasible_loci:   该分配与全员观测相容的位点名元组；
    - conflicting_loci: 被孟德尔规则直接证伪的位点名元组；
    - unknown_loci:    因节点预算耗尽未能判定的位点名元组。
    """

    assignment: dict
    feasible_loci: tuple
    conflicting_loci: tuple
    unknown_loci: tuple


@dataclass(frozen=True)
class AssignmentResult:
    """assign_parents() 的返回值。status ∈ {"feasible", "infeasible", "unknown"}。

    - feasible_assignments: 可行时为非空元组，每项为
        {孩子ID: (父亲ID, 母亲ID)} 的一个完整联合分配；
      多个孩子的选择做笛卡尔积（至多 3^3=27 项），顺序确定（孩子按 ID
      排序、候选按给入顺序）。
    - conflicts: 不可行时为每个联合分配的 AssignmentConflict（每项都
      至少有一个 conflicting_loci）；预算耗尽时仅含已完成判定的分配，
      其中至少一项含 unknown_loci；可行时为 ()。
    - nodes_used: 实际消耗的搜索节点数（全部位点、全部分配共享预算）。

    status 语义：
    - "feasible"：至少一个联合分配在全部位点上可行；
    - "infeasible"：全部联合分配都被证明在某位点不可行；
    - "unknown"：尚未找到可行分配，且判定因预算耗尽未完成。
    """

    status: str
    feasible_assignments: tuple
    conflicts: tuple
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


# ---------------------------------------------------------------- 多位点候选父母判定

def _require_genotype(label, g) -> None:
    """校验单个基因型（严格 int、0/1/2，拒绝 bool/float/NaN/Infinity）。"""
    if isinstance(g, bool) or not isinstance(g, int):
        raise ValidationError(
            f"{label} genotype must be an int in {{0, 1, 2}} "
            f"(bool/float/NaN/Infinity rejected), got {g!r}"
        )
    if g not in GENOTYPES:
        raise ValidationError(f"{label} genotype must be in {{0, 1, 2}}, got {g!r}")


def _validate_assignment_input(profiles, candidates, node_budget):
    """校验 assign_parents 输入，返回确定性排序后的 (children, loci, pairs)。"""
    if isinstance(node_budget, bool) or not isinstance(node_budget, int):
        raise ValidationError(
            f"node_budget must be an int >= 1 (bool/float rejected), got {node_budget!r}"
        )
    if node_budget < 1:
        raise ValidationError(f"node_budget must be >= 1, got {node_budget!r}")

    if isinstance(profiles, (str, bytes)) or not isinstance(profiles, Mapping):
        raise ValidationError(
            f"profiles must be a mapping {{id: {{locus: genotype}}}}, "
            f"got {type(profiles).__name__}"
        )
    if not (1 <= len(profiles) <= MAX_INDIVIDUALS):
        raise ValidationError(
            f"profiles must contain 1..{MAX_INDIVIDUALS} individuals, got {len(profiles)}"
        )
    for pid in profiles:
        if not isinstance(pid, str) or not pid:
            raise ValidationError(f"individual id must be a non-empty str, got {pid!r}")
    loci = None
    for pid, prof in profiles.items():
        if isinstance(prof, (str, bytes)) or not isinstance(prof, Mapping):
            raise ValidationError(
                f"profiles[{pid!r}] must be a mapping {{locus: genotype}}, "
                f"got {type(prof).__name__}"
            )
        if not (1 <= len(prof) <= MAX_LOCI):
            raise ValidationError(
                f"profiles[{pid!r}] must contain 1..{MAX_LOCI} loci, got {len(prof)}"
            )
        for locus, g in prof.items():
            if not isinstance(locus, str) or not locus:
                raise ValidationError(
                    f"profiles[{pid!r}] locus name must be a non-empty str, got {locus!r}"
                )
            _require_genotype(f"profiles[{pid!r}][{locus!r}]", g)
        keys = set(prof)
        if loci is None:
            loci = keys
        elif keys != loci:
            missing = sorted(loci - keys)
            extra = sorted(keys - loci)
            raise ValidationError(
                f"profiles[{pid!r}] locus set differs; missing {missing}, extra {extra}"
            )

    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Mapping):
        raise ValidationError(
            f"candidates must be a mapping {{child_id: [(father, mother), ...]}}, "
            f"got {type(candidates).__name__}"
        )
    if not (1 <= len(candidates) <= MAX_CHILDREN):
        raise ValidationError(
            f"candidates must declare 1..{MAX_CHILDREN} children, got {len(candidates)}"
        )
    for cid in candidates:
        if not isinstance(cid, str) or not cid:
            raise ValidationError(f"child id must be a non-empty str, got {cid!r}")
        if cid not in profiles:
            raise ValidationError(
                f"candidates[{cid!r}]: child has no profile in profiles"
            )
    children = sorted(candidates)
    pairs_by_child = {}
    for cid in children:
        plist = candidates[cid]
        if isinstance(plist, (str, bytes)) or not isinstance(plist, (tuple, list)):
            raise ValidationError(
                f"candidates[{cid!r}] must be a sequence of (father, mother) pairs, "
                f"got {plist!r}"
            )
        if not (1 <= len(plist) <= MAX_CANDIDATE_PAIRS):
            raise ValidationError(
                f"candidates[{cid!r}] must contain 1..{MAX_CANDIDATE_PAIRS} pairs, "
                f"got {len(plist)}"
            )
        seen = set()
        norm = []
        for k, pair in enumerate(plist):
            if isinstance(pair, (str, bytes)) or not isinstance(pair, (tuple, list)):
                raise ValidationError(
                    f"candidates[{cid!r}][{k}] must be a (father, mother) pair, "
                    f"got {pair!r}"
                )
            if len(pair) != 2:
                raise ValidationError(
                    f"candidates[{cid!r}][{k}] must have exactly 2 members, got {pair!r}"
                )
            for role, par in zip(("father", "mother"), pair):
                if not isinstance(par, str) or not par:
                    raise ValidationError(
                        f"candidates[{cid!r}][{k}] {role} must be a non-empty member id "
                        f"str (missing/unknown founders are not allowed as candidates; "
                        f"parents outside the candidate set cannot be tested), got {par!r}"
                    )
                if par not in profiles:
                    raise ValidationError(
                        f"candidates[{cid!r}][{k}] {role} {par!r} has no profile"
                    )
            tpair = (pair[0], pair[1])
            if tpair in seen:
                raise ValidationError(
                    f"candidates[{cid!r}] contains duplicate pair {list(tpair)}"
                )
            seen.add(tpair)
            norm.append(tpair)
        pairs_by_child[cid] = norm

    loci_sorted = sorted(loci)

    # 任一可选组合都不得使亲子图成环（含自环）：候选枚举只产生可行结构，
    # 结构非法在此一次性明确拒绝，不混同位点冲突诊断。
    base_pedigree = {pid: (None, None) for pid in profiles}
    for sel in itertools.product(*(range(len(pairs_by_child[c])) for c in children)):
        pedigree = dict(base_pedigree)
        for cid, pi in zip(children, sel):
            pedigree[cid] = pairs_by_child[cid][pi]
        try:
            _check_acyclic(pedigree)
        except ValidationError as exc:
            chosen = ", ".join(
                f"{cid}->{list(pairs_by_child[cid][pi])}"
                for cid, pi in zip(children, sel)
            )
            raise ValidationError(
                f"candidate selection ({chosen}) forms a cyclic parent-child graph: {exc}"
            ) from None

    return children, loci_sorted, pairs_by_child


def _solve_combo_locus(profiles, pedigree, locus, budget: _Budget):
    """在给定家系结构与单一位点的全员观测下求一个解（复用 check 的内核）。"""
    obs = {pid: profiles[pid][locus] for pid in profiles}
    model = _validate_and_build(pedigree, obs, budget.limit)
    return _solve_obs(model, model.obs_ids, budget)


def assign_parents(profiles, candidates, *, node_budget: int = DEFAULT_NODE_BUDGET) -> AssignmentResult:
    """多位点候选父母判定。契约见模块 docstring 与 AssignmentResult。

    枚举每个孩子选定一对候选父母的全部联合分配（至多 3^3=27 个），同一
    分配在全部位点（至多 4 个）上用与 check 完全相同的孟德尔规则判定。
    纯函数：非法输入抛 ValidationError，不修改任何入参或外部状态。
    """
    children, loci, pairs_by_child = _validate_assignment_input(
        profiles, candidates, node_budget
    )
    budget = _Budget(node_budget)

    # 枚举完整联合分配：同一选择贯穿全部位点，不允许逐位点换父母。
    # 对每个分配逐位点判定，记录好/坏位点；预算耗尽时剩余位点记 unknown。
    feasible_assignments = []
    conflict_reports = []
    exhausted = False
    base_pedigree = {pid: (None, None) for pid in profiles}
    selections = list(
        itertools.product(*(range(len(pairs_by_child[c])) for c in children))
    )
    for sel in selections:
        pedigree = dict(base_pedigree)
        assignment = {}
        for cid, pi in zip(children, sel):
            pair = pairs_by_child[cid][pi]
            pedigree[cid] = pair
            assignment[cid] = pair

        good, bad = [], []
        unknown_loci = []
        for li, locus in enumerate(loci):
            try:
                sol = _solve_combo_locus(profiles, pedigree, locus, budget)
            except _BudgetExhausted:
                exhausted = True
            if exhausted:
                unknown_loci = loci[li:]
                break
            (good if sol is not None else bad).append(locus)

        if not bad and not unknown_loci:
            feasible_assignments.append(dict(assignment))
        else:
            conflict_reports.append(
                AssignmentConflict(
                    assignment=dict(assignment),
                    feasible_loci=tuple(good),
                    conflicting_loci=tuple(bad),
                    unknown_loci=tuple(unknown_loci),
                )
            )
        if exhausted:
            break

    if feasible_assignments:
        # 预算可能在枚举后期耗尽，但任何已返回的分配都已逐位点证明可行。
        return AssignmentResult(
            status="feasible",
            feasible_assignments=tuple(feasible_assignments),
            conflicts=(),
            nodes_used=budget.used,
        )

    return AssignmentResult(
        status="unknown" if exhausted else "infeasible",
        feasible_assignments=(),
        conflicts=tuple(conflict_reports),
        nodes_used=budget.used,
    )
