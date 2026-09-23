"""mendel_conflict — 孟德尔家系的最小冲突定位（单二等位位点）。

公开接口见 ``check``；数据结构与错误语义见 README。
本包只依赖 Python 3.14 标准库。
"""

from .core import (
    CheckResult,
    PedigreeError,
    STATUS_FEASIBLE,
    STATUS_INCONSISTENT,
    STATUS_UNKNOWN,
    check,
)

__all__ = [
    "check",
    "CheckResult",
    "PedigreeError",
    "STATUS_FEASIBLE",
    "STATUS_INCONSISTENT",
    "STATUS_UNKNOWN",
]

__version__ = "1.0.0"
