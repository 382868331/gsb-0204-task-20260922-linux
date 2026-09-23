"""核心算法：孟德尔可行性（三元关系传播 + 回溯分支）与最小冲突定位（命中集枚举）。

接口契约（详见 README）：

- ``check(pedigree, observations=None, *, node_budget=100_000)`` 返回 ``CheckResult``。
- 状态三选一：``STATUS_FEASIBLE`` / ``STATUS_INCONSISTENT`` / ``STATUS_UNKNOWN``。
- 预算耗尽只产生 ``unknown``；``inconsistent`` 仅在证明无解时给出，二者严格分开。

本模块仅使用 Python 标准库。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import FrozenSet, Mapping, Optional

STATUS_FEASIBLE = "feasible"
STATUS_INCONSISTENT = "inconsistent"
STATUS_UNKNOWN = "unknown"

GENOTYPES = (0, 1, 2)
_MAX_INDIVIDUALS = 20
_DEFAULT_BUDGET = 100_000


class PedigreeError(ValueError):
    """输入结构或取值违反契约。

    ``path`` 给出可定位的字段路径（如 ``individuals[3].father``），
    ``reason`` 为机器可读的原因代码。
    """

    def __init__(self, message: str, path: str = "<root>", reason: str = "invalid"):
        self.path = path
        self.reason = reason
        full = f"{path}: {message}" if path else message
        super().__init__(full)


# ---------------------------------------------------------------------------
# 孟德尔关系（二等位，基因型为替代等位计数：0=aa, 1=Aa, 2=AA）
# ---------------------------------------------------------------------------

# 单个已知亲本 p 可直接传递给子女的等位集合
_ALLELES = {0: (0,), 1: (0, 1), 2: (1,)}


def triple_is_mendelian(father: int, mother: int, child: int) -> bool:
    """三元组是否满足：父母各传一个等位，相加重计等于子女基因型。

    父/母为 ``None`` 表示该位置是独立未知创始者（任意单等位）。
    """
    paternal = _ALLELES[father] if father is not None else (0, 1)
    maternal = _ALLELES[mother] if mother is not None else (0, 1)
    for a in paternal:
        for b in maternal:
            if a + b == child:
                return True
    return False


def verify_witness(
    pedigree: Mapping,
    observations: Mapping[str, int],
    witness: Mapping[str, int],
    removed: FrozenSet[str] = frozenset(),
) -> bool:
    """独立校验一份见证：覆盖全部个体、满足观测、满足每个亲子三元组。

    供测试与演示交叉验证使用；``check`` 内部返回见证前也会调用它。
    """
    try:
        inds = _parse_pedigree(pedigree)
    except PedigreeError:
        return False
    removed_set = frozenset(removed)
    by_id = {i.id: i for i in inds}
    if set(witness) != set(by_id):
        return False
    for iid, gt in witness.items():
        if gt not in GENOTYPES:
            return False
    for iid, gt in observations.items():
        if iid in removed_set:
            continue
        if witness.get(iid) != gt:
            return False
    for ind in inds:
        f = witness[ind.father] if ind.father is not None else None
        m = witness[ind.mother] if ind.mother is not None else None
        if not triple_is_mendelian(f, m, witness[ind.id]):
            return False
    return True


# ---------------------------------------------------------------------------
# 输入解析与校验
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Individual:
    id: str
    father: Optional[str]
    mother: Optional[str]


def _reject_non_int(value, path: str, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PedigreeError(
            f"{what} 必须是 0/1/2 的整数（拒绝 bool、浮点与 NaN/Infinity），得到 {value!r}",
            path,
            "bad_genotype",
        )
    if value not in GENOTYPES:
        raise PedigreeError(
            f"{what} 必须是 0、1 或 2，得到 {value!r}", path, "bad_genotype"
        )
    return value


def _parse_pedigree(pedigree: object) -> list[Individual]:
    if not isinstance(pedigree, Mapping):
        raise PedigreeError(
            f"pedigree 必须是映射（dict），得到 {type(pedigree).__name__}",
            "<root>",
            "bad_type",
        )
    raw = pedigree.get("individuals")
    if not isinstance(raw, list):
        raise PedigreeError(
            "pedigree['individuals'] 必须是列表", "individuals", "bad_type"
        )
    if len(raw) > _MAX_INDIVIDUALS:
        raise PedigreeError(
            f"个体数 {len(raw)} 超过上限 {_MAX_INDIVIDUALS}",
            "individuals",
            "too_many_individuals",
        )
    inds: list[Individual] = []
    seen: set[str] = set()
    for idx, item in enumerate(raw):
        p = f"individuals[{idx}]"
        if not isinstance(item, Mapping):
            raise PedigreeError("个体记录必须是映射（dict）", p, "bad_type")
        iid = item.get("id")
        if not isinstance(iid, str) or not iid:
            raise PedigreeError("id 必须是非空字符串", f"{p}.id", "bad_id")
        if iid in seen:
            raise PedigreeError(f"个体 ID {iid!r} 重复", f"{p}.id", "duplicate_id")
        seen.add(iid)
        parents: list[Optional[str]] = []
        for side in ("father", "mother"):
            v = item[side] if side in item else None
            if v is not None and not isinstance(v, str):
                raise PedigreeError(
                    f"{side} 必须是字符串或 null，得到 {v!r}",
                    f"{p}.{side}",
                    "bad_parent",
                )
            if v == "":
                raise PedigreeError(
                    f"{side} 为空字符串；缺失父/母必须用 null 表示",
                    f"{p}.{side}",
                    "bad_parent",
                )
            parents.append(v)
        father, mother = parents
        if father is not None and father == iid:
            raise PedigreeError("个体不能是自己的父亲", f"{p}.father", "self_parent")
        if mother is not None and mother == iid:
            raise PedigreeError("个体不能是自己的母亲", f"{p}.mother", "self_parent")
        inds.append(Individual(iid, father, mother))

    by_id = {i.id: i for i in inds}
    for idx, ind in enumerate(inds):
        p = f"individuals[{idx}]"
        for side, pid in (("father", ind.father), ("mother", ind.mother)):
            if pid is not None and pid not in by_id:
                raise PedigreeError(
                    f"{side} {pid!r} 未在 individuals 中声明",
                    f"{p}.{side}",
                    "unknown_parent",
                )
        if ind.father is not None and ind.father == ind.mother:
            raise PedigreeError(
                "father 与 mother 不能是同一个体", f"{p}.father", "same_parents"
            )

    _ensure_acyclic(inds)
    return inds


def _ensure_acyclic(inds: list[Individual]) -> None:
    by_id = {i.id: i for i in inds}
    # 0=未访问 1=在当前 DFS 栈 2=完成
    color: dict[str, int] = {i.id: 0 for i in inds}

    def visit(iid: str, stack: list[str]) -> None:
        color[iid] = 1
        stack.append(iid)
        ind = by_id[iid]
        for pid in (ind.father, ind.mother):
            if pid is None:
                continue
            if color[pid] == 1:
                cycle = stack[stack.index(pid):] + [pid]
                raise PedigreeError(
                    "亲子图存在环: " + " -> ".join(cycle),
                    "individuals",
                    "cyclic_pedigree",
                )
            if color[pid] == 0:
                visit(pid, stack)
        stack.pop()
        color[iid] = 2

    for iid in color:
        if color[iid] == 0:
            visit(iid, [])


def _parse_observations(
    observations: object, known_ids: set[str]
) -> dict[str, int]:
    if observations is None:
        return {}
    if not isinstance(observations, Mapping):
        raise PedigreeError(
            f"observations 必须是映射（dict），得到 {type(observations).__name__}",
            "observations",
            "bad_type",
        )
    out: dict[str, int] = {}
    for key, value in observations.items():
        if not isinstance(key, str):
            raise PedigreeError(
                f"观测 ID 必须是字符串，得到 {key!r}", "observations", "bad_id"
            )
        if key not in known_ids:
            raise PedigreeError(
                f"观测个体 {key!r} 不在家系中",
                f"observations[{key!r}]",
                "unknown_individual",
            )
        out[key] = _reject_non_int(value, f"observations[{key!r}]", "基因型观测")
    return out


# ---------------------------------------------------------------------------
# 弧一致（AC-3）与带见证的回溯搜索
# ---------------------------------------------------------------------------


@dataclass
class _Problem:
    inds: list[Individual]
    by_id: dict[str, Individual]
    observations: dict[str, int]
    # 每个亲子三元组：(child, father_or_None, mother_or_None)
    families: list[tuple[str, Optional[str], Optional[str]]]
    # 个体参与的所有三元组编号（作为子女或亲本）
    fam_index: dict[str, list[int]]


def _build_problem(
    inds: list[Individual], observations: dict[str, int]
) -> _Problem:
    by_id = {i.id: i for i in inds}
    families: list[tuple[str, Optional[str], Optional[str]]] = []
    fam_index: dict[str, list[int]] = {i.id: [] for i in inds}
    for idx, ind in enumerate(inds):
        families.append((ind.id, ind.father, ind.mother))
        fam_index[ind.id].append(idx)
        if ind.father is not None:
            fam_index[ind.father].append(idx)
        if ind.mother is not None:
            fam_index[ind.mother].append(idx)
    return _Problem(inds, by_id, observations, families, fam_index)


def _initial_domains(prob: _Problem) -> dict[str, set[int]]:
    domains: dict[str, set[int]] = {i.id: set(GENOTYPES) for i in prob.inds}
    for iid, gt in prob.observations.items():
        domains[iid] = {gt}
    return domains


def _possible_alleles(domain: set[int]) -> frozenset[int]:
    """该域中所有基因型可传出的等位并集。"""
    out: set[int] = set()
    for v in domain:
        out.update(_ALLELES[v])
    return frozenset(out)


def _revise_family(
    child: str,
    father: Optional[str],
    mother: Optional[str],
    domains: dict[str, set[int]],
) -> bool:
    """对一个三元组做关系弧一致：联合剪枝子女与每个已知亲本的域。

    缺失亲本（None）按可传任一等位的独立创始者处理。
    """
    changed = False
    dc = domains[child]
    df = domains[father] if father is not None else None
    dm = domains[mother] if mother is not None else None
    pa = _possible_alleles(df) if df is not None else frozenset({0, 1})
    ma = _possible_alleles(dm) if dm is not None else frozenset({0, 1})

    possible_child = {a + b for a in pa for b in ma}
    new_c = dc & possible_child
    if len(new_c) != len(dc):
        domains[child] = new_c
        changed = True
        dc = new_c

    if not dc:
        return True

    if df is not None:
        kept = {
            p
            for p in df
            if any(
                a + b in dc for a in _ALLELES[p] for b in ma
            )
        }
        if len(kept) != len(df):
            domains[father] = kept
            changed = True
            df = kept

    if dm is not None and df is not None and not df:
        return True
    if dm is not None:
        # 母本剪枝使用剪枝后的父本域，信息更强。
        pa2 = _possible_alleles(df) if df is not None else frozenset({0, 1})
        kept = {
            q
            for q in dm
            if any(
                a + b in dc for a in pa2 for b in _ALLELES[q]
            )
        }
        if len(kept) != len(dm):
            domains[mother] = kept
            changed = True
    return changed


def _propagate(prob: _Problem, domains: dict[str, set[int]]) -> bool:
    """三元关系传播到不动点；任一域擦空返回 False。"""
    queue: list[int] = list(range(len(prob.families)))
    queued = [True] * len(prob.families)
    while queue:
        fi = queue.pop()
        queued[fi] = False
        child, father, mother = prob.families[fi]
        participants = [child]
        if father is not None:
            participants.append(father)
        if mother is not None:
            participants.append(mother)
        if any(not domains[x] for x in participants):
            return False
        before = {x: len(domains[x]) for x in participants}
        _revise_family(child, father, mother, domains)
        for x in participants:
            if not domains[x]:
                return False
            if len(domains[x]) != before[x]:
                for nfi in prob.fam_index[x]:
                    if not queued[nfi]:
                        queued[nfi] = True
                        queue.append(nfi)
    return True


class _Budget:
    __slots__ = ("remaining", "used")

    def __init__(self, budget: int):
        self.remaining = budget
        self.used = 0

    def tick(self) -> None:
        # 在进入第 budget+1 个节点前抛出；used 只计实际执行的节点数。
        if self.remaining == 0:
            raise _BudgetExhausted(self.used)
        self.remaining -= 1
        self.used += 1


class _BudgetExhausted(Exception):
    def __init__(self, used: int):
        self.used = used
        super().__init__(f"搜索节点预算耗尽（已用 {used}）")


def _search(
    prob: _Problem, domains: dict[str, set[int]], budget: _Budget
) -> Optional[dict[str, int]]:
    """MRV 分支搜索；每个节点先做三元关系传播。返回完整见证或 None。"""
    budget.tick()
    local = {k: set(v) for k, v in domains.items()}
    if not _propagate(prob, local):
        return None
    if all(len(v) == 1 for v in local.values()):
        witness = {k: next(iter(v)) for k, v in local.items()}
        if _triple_check(prob, witness):
            return witness
        return None
    # MRV：选域最小且未定的个体；观察域为单值者天然跳过。
    var = min(
        (k for k in local if len(local[k]) > 1),
        key=lambda k: (len(local[k]), k),
    )
    for val in sorted(local[var]):
        child_domains = {k: set(v) for k, v in local.items()}
        child_domains[var] = {val}
        found = _search(prob, child_domains, budget)
        if found is not None:
            return found
    return None


def _triple_check(prob: _Problem, assignment: Mapping[str, int]) -> bool:
    for ind in prob.inds:
        f = assignment[ind.father] if ind.father is not None else None
        m = assignment[ind.mother] if ind.mother is not None else None
        if not triple_is_mendelian(f, m, assignment[ind.id]):
            return False
    return True


def _solve_without(
    prob: _Problem, removed: frozenset[str], budget: _Budget
) -> Optional[dict[str, int]]:
    obs = {k: v for k, v in prob.observations.items() if k not in removed}
    sub = _Problem(prob.inds, prob.by_id, obs, prob.families, prob.fam_index)
    return _search(sub, _initial_domains(sub), budget)


# ---------------------------------------------------------------------------
# 最小移除集（最小命中集）枚举
# ---------------------------------------------------------------------------


def _unsat_core(
    prob: _Problem, budget: _Budget
) -> frozenset[str]:
    """求一个删除极小的不可满足观测子集（必然非空）。"""
    core = list(prob.observations)  # 已知整体不可行
    for oid in list(core):
        trial = [x for x in core if x != oid]
        obs = {k: prob.observations[k] for k in trial}
        sub = _Problem(prob.inds, prob.by_id, obs, prob.families, prob.fam_index)
        if _search(sub, _initial_domains(sub), budget) is None:
            # 去掉它仍不可行 => 它对此 core 冗余，安全删除。
            core = trial
    return frozenset(core)


def _minimum_removals(
    prob: _Problem, budget: _Budget
) -> tuple[
    list[tuple[str, ...]], dict[tuple[tuple[str, ...], str], dict[str, int]]
]:
    """枚举全部最小基数移除集，每个返回一份重新可行见证。

    定理：设 C 是任一不可满足观测子集，则任何能恢复可行性的移除集 R
    都必须与 C 相交（否则 C 完整保留，仍不可行）。因此只需枚举命中 C 的
    候选，再用整问题可行性严格验证。
    """
    core = _unsat_core(prob, budget)
    all_ids = sorted(prob.observations)
    core_ids = sorted(core)
    other_ids = [x for x in all_ids if x not in core]
    m = len(all_ids)

    cache: dict[frozenset[str], Optional[dict[str, int]]] = {}

    def solve_cached(removed: frozenset[str]):
        if removed not in cache:
            cache[removed] = _solve_without(prob, removed, budget)
        return cache[removed]

    for k in range(1, m + 1):
        solutions: list[tuple[str, ...]] = []
        witnesses: dict[tuple[tuple[str, ...], str], dict[str, int]] = {}
        # 至少 1 个、至多 min(k, |C|) 个成员来自 core。
        hit_lo, hit_hi = 1, min(k, len(core_ids))
        for from_core in range(hit_lo, hit_hi + 1):
            from_other = k - from_core
            if from_other > len(other_ids):
                continue
            for ccombo in combinations(core_ids, from_core):
                for ocombo in combinations(other_ids, from_other):
                    removed = frozenset(ccombo).union(ocombo)
                    wit = solve_cached(removed)
                    if wit is not None:
                        key = tuple(sorted(removed))
                        solutions.append(key)
                        # 每个被移除观测记录“移除后恢复可行”的同一份完整见证；
                        # 不同最小解释各自独立记录。
                        for rid in key:
                            witnesses[(key, rid)] = wit
        if solutions:
            solutions.sort()
            return solutions, witnesses
    # 理论不可达：移除全部观测后创始者自由，必可行。
    raise RuntimeError("internal error: removing all observations must be feasible")


# ---------------------------------------------------------------------------
# 公开结果对象
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """``check`` 的结果。

    属性：
      status:             feasible / inconsistent / unknown
      feasible_genotypes: feasible 时每个体全部可行基因型（按 ID、值排序）
      minimal_removals:   inconsistent 时全部最小移除集（每个为按 ID 字典序
                          排列的元组；移除集之间也按字典序）
      witnesses:          inconsistent 时每个被移除观测的重新可行见证，
                          键为 ``(移除集元组, 被移除观测ID)``，值为覆盖全部
                          个体、满足剩余观测与全部三元组的完整基因型指派
      nodes_used:         搜索实际使用的分支节点数
      node_budget:        本次预算
      note:               人类可读补充说明（unknown 时说明耗尽发生的阶段）
    """

    status: str
    feasible_genotypes: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    minimal_removals: tuple[tuple[str, ...], ...] = ()
    witnesses: Mapping[tuple[tuple[str, ...], str], Mapping[str, int]] = field(
        default_factory=dict
    )
    nodes_used: int = 0
    node_budget: int = _DEFAULT_BUDGET
    note: str = ""

    @property
    def feasible(self) -> bool:
        return self.status == STATUS_FEASIBLE

    @property
    def inconsistent(self) -> bool:
        return self.status == STATUS_INCONSISTENT

    @property
    def budget_exhausted(self) -> bool:
        return self.status == STATUS_UNKNOWN

    def witness_for(self, removed_id: str, removal_set: tuple[str, ...] | None = None):
        """取某个被移除观测的重新可行见证。

        当同一观测出现在多个最小移除集中时需用 ``removal_set`` 指定；
        只有一个时可省略。
        """
        if removal_set is None:
            hits = [k for k in self.witnesses if k[1] == removed_id]
            if len(hits) != 1:
                raise KeyError(
                    f"观测 {removed_id!r} 出现在 {len(hits)} 个最小移除集中，"
                    "请显式指定 removal_set"
                )
            return self.witnesses[hits[0]]
        key = (tuple(sorted(removal_set)), removed_id)
        if key not in self.witnesses:
            raise KeyError(f"见证不存在: {key}")
        return self.witnesses[key]


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------


def check(
    pedigree: Mapping,
    observations: Optional[Mapping[str, int]] = None,
    *,
    node_budget: int = _DEFAULT_BUDGET,
) -> CheckResult:
    """检验单二等位位点的孟德尔家系观测。

    参数：
      pedigree:     ``{"individuals": [{"id", "father", "mother"}, ...]}``，
                    缺失父/母用 null（或省略），最多 20 人，亲子图必须无环。
      observations: ``{个体ID: 0|1|2}``，可省略/为空。
      node_budget:  分支搜索节点正整数预算（拒绝 bool）；耗尽时返回
                    status="unknown"，与“证明无解”严格区分。

    返回：``CheckResult``。非法输入抛 ``PedigreeError``（``ValueError`` 子类）。
    """
    if isinstance(node_budget, bool) or not isinstance(node_budget, int):
        raise PedigreeError(
            f"node_budget 必须是正整数（拒绝 bool），得到 {node_budget!r}",
            "node_budget",
            "bad_budget",
        )
    if node_budget <= 0:
        raise PedigreeError(
            f"node_budget 必须是正整数，得到 {node_budget}",
            "node_budget",
            "bad_budget",
        )

    inds = _parse_pedigree(pedigree)
    obs = _parse_observations(observations, {i.id for i in inds})
    prob = _build_problem(inds, obs)
    budget = _Budget(node_budget)

    # 阶段 1：整体可行性 + 可行基因型集合（树状 CSP 上弧一致即全局一致，
    # 但见证仍通过三元组逐一校验，不把该性质当作免验假设）。
    try:
        witness = _search(prob, _initial_domains(prob), budget)
    except _BudgetExhausted as exc:
        return CheckResult(
            STATUS_UNKNOWN,
            nodes_used=exc.used,
            node_budget=node_budget,
            note="预算在整体可行性搜索阶段耗尽；未证明可行，也未证明冲突。",
        )

    if witness is not None:
        # 逐个体、逐值求“全部可行基因型”。每个候选值单独试解：
        # 无近亲（亲子关系图为树）时弧一致域即充分，但近亲婚配会使约束图
        # 出现无向环，弧一致不足以保证全局一致，故一律以试解为准，
        # 不把树结构性质当作免验假设。≤20 人、域 ≤3，成本有界。
        base = _initial_domains(prob)
        try:
            feasible_genotypes: dict[str, tuple[int, ...]] = {}
            for iid in sorted(prob.by_id):
                kept: list[int] = []
                for val in GENOTYPES:
                    if val not in base[iid]:
                        continue
                    trial = {k: set(v) for k, v in base.items()}
                    trial[iid] = {val}
                    if _search(prob, trial, budget) is not None:
                        kept.append(val)
                feasible_genotypes[iid] = tuple(kept)
        except _BudgetExhausted as exc:
            return CheckResult(
                STATUS_UNKNOWN,
                nodes_used=exc.used,
                node_budget=node_budget,
                note="已找到可行见证（问题相容），但预算在枚举全部可行基因型阶段"
                "耗尽；可行集合不保证完整，故不返回部分结果。",
            )
        return CheckResult(
            STATUS_FEASIBLE,
            feasible_genotypes=feasible_genotypes,
            nodes_used=budget.used,
            node_budget=node_budget,
            note="全部观测相容；feasible_genotypes 为每个体的全部可行基因型。",
        )

    # 阶段 2：证明无解后，求极小冲突核 + 枚举全部最小移除集。
    try:
        removals, wit_map = _minimum_removals(prob, budget)
    except _BudgetExhausted as exc:
        return CheckResult(
            STATUS_UNKNOWN,
            nodes_used=exc.used,
            node_budget=node_budget,
            note="已证明观测整体不相容，但预算在最小移除集枚举阶段耗尽；"
            "最小解释未完整确定。",
        )

    # 交付前用独立三元组规则复核每份见证，不满足绝不返回。
    for (removal_set, rid), wit in wit_map.items():
        if not verify_witness(
            pedigree, obs, wit, removed=frozenset(removal_set)
        ):
            raise RuntimeError(
                f"internal error: witness failed verification for "
                f"{removal_set}::{rid}"
            )

    return CheckResult(
        STATUS_INCONSISTENT,
        minimal_removals=tuple(removals),
        witnesses=dict(wit_map),
        nodes_used=budget.used,
        node_budget=node_budget,
        note="观测不相容；minimal_removals 为按 ID 字典序排列的全部最小移除集，"
        "witnesses 给出移除后恢复可行的完整基因型见证。",
    )
