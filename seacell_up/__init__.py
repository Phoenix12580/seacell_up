"""seacell_up: 百万级单细胞 Metacell 全自适应极速构建系统.

纯 Python 生态（numpy / scipy.sparse / sklearn / anndata），分治 + 迭代清洗
架构，所有影响 Metacell 数量/尺寸/质量的参数均由数据自适应推导。

快速上手:
    from seacell_up import build_metacells
    adata_out, mc_adata, report = build_metacells("cells.h5ad", sample_key="sampleID")

可选模块（按需 import，不强依赖）:
    from seacell_up.cnv_infer import infer_cnv          # 需 infercnvpy 环境
    from seacell_up.plotting import plot_metacell_umap  # 需 pip install seacell_up[plot]

方案评估见仓库根目录 EVALUATION.md。
"""
from .api import build_metacells, build_metacells_cnv
from .cnv_branch import CNVBranchConfig, run_cnv_branch
from .gnmf import FastGNMF, enforce_capacity
from .optimizer import AutoMetaCellOptimizer
from .pipeline import IterativeMetaCellPipeline, PipelineConfig, score_metacells
from .preprocessing import (
    aggregate_metacells,
    normalize_log,
    select_hvgs,
)

__version__ = "0.3.1"

__all__ = [
    "build_metacells",
    "build_metacells_cnv",
    "FastGNMF",
    "enforce_capacity",
    "AutoMetaCellOptimizer",
    "IterativeMetaCellPipeline",
    "PipelineConfig",
    "score_metacells",
    "aggregate_metacells",
    "normalize_log",
    "select_hvgs",
    "run_cnv_branch",
    "CNVBranchConfig",
]
