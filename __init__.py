"""
scE2TM: Single-cell Embedded Topic Model.
"""

# 导入高层函数，并作为公开的 scE2TM
from .api import scE2TM

# 模型类（如果用户需要，可以以其他名字导出，例如 _scE2TMModel）
from .models.scE2TM import scE2TM as _scE2TMModel
from .runners.Runner import Runner
from .utils.data.SingleCellDataHandler import SingleCellDataHandler

__all__ = ["scE2TM", "Runner", "SingleCellDataHandler", "_scE2TMModel"]