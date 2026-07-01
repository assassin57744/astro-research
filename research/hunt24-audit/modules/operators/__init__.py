# modules/operators/__init__.py

"""
模块算子汇聚大盘。
通过包内命名空间合并，对外维持原原单体 operators.py 的平铺引用属性，实现无缝平滑迁移！
"""

from .asset_manager import AssetManager, SchemaAligner
from .param_registry import ParamRegistry
from .data_selector import DataSelector

# 显式声明暴露的契约矩阵
__all__ = [
    "AssetManager",
    "SchemaAligner",
    "ParamRegistry",
    "DataSelector"
]