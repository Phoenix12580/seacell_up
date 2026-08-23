"""高层函数式 API：一行调用完成 Metacell 构建。

快速上手:
    from seacell_up import build_metacells
    adata_out, mc_adata, report = build_metacells("cells.h5ad", sample_key="sampleID")

上皮/肿瘤 CNV 分支（需先在 infercnvpy 环境跑 infer_cnv）:
    from seacell_up import build_metacells_cnv
    cnv_out, cnv_report = build_metacells_cnv("epi_cnv.h5ad", sample_key="sampleID")

可视化（需 pip install seacell_up[plot]）:
    from seacell_up.plotting import plot_metacell_umap
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import anndata as ad

from .cnv_branch import CNVBranchConfig, run_cnv_branch
from .pipeline import IterativeMetaCellPipeline, PipelineConfig

PathLike = Union[str, Path]


def _load(x: Union[ad.AnnData, PathLike]) -> ad.AnnData:
    """AnnData 直通；路径则读入。"""
    if isinstance(x, ad.AnnData):
        return x
    return ad.read_h5ad(str(x))


def build_metacells(
    adata: Union[ad.AnnData, PathLike],
    sample_key: str = "sample_id",
    input_is_raw: Optional[bool] = None,
    n_jobs: int = -1,
    n_top_genes: int = 2048,
    capacity_lo: int = 20,
    capacity_hi: int = 35,
    max_rounds: int = 3,
    pass_rule: str = "mean",
    config: Optional[PipelineConfig] = None,
    **config_kwargs,
) -> tuple[ad.AnnData, ad.AnnData, dict]:
    """构建 Metacell（主流水线：分治 + GNMF + 迭代清洗）。

    每个 Metacell 严格 capacity_lo..capacity_hi 个细胞（默认 [20,35]）；
    每个样本独立自适应选 NMF 维数 k_dim、Metacell 尺寸 gamma、数量 K。

    Parameters
    ----------
    adata:
        AnnData 对象或 .h5ad 路径。X 为原始计数（或已归一化，自动识别）。
    sample_key:
        obs 中的样本列名；缺失时整体视为单样本。
    input_is_raw:
        None=自动启发式判断 X 是否原始计数；True/False 强制声明。
    n_jobs:
        并行进程数，-1 = 全核。
    n_top_genes:
        全局 HVG 数（所有阶段共用基因空间）。
    capacity_lo / capacity_hi:
        Metacell 细胞数硬界（默认 20-35）。
    max_rounds:
        第二阶段跨样本回收的最大轮数。
    pass_rule:
        通过线规则: "mean"（默认，全局平均分）| "median" | "quantile"
        （配 pass_quantile，如 0.85 表示只回收最差 15%）。
    config:
        完整 PipelineConfig（给出时忽略上面的散参数，除 config_kwargs）。
    **config_kwargs:
        透传 PipelineConfig 其余字段（如 alpha、n_neighbors、search_rows）。

    Returns
    -------
    adata_out:
        细胞级 AnnData（输入副本），obs 新增:
        - metacell_id: "MC000123" 或 "noise"
        - metacell_stage: stage1 / rescue1..3 / noise
        - mc_score: 该 Metacell 的质量分
        - is_noise: bool
    mc_adata:
        Metacell x genes 伪体（原始计数之和），obs 含
        n_cells / stage / score / gamma / dominant_sample / sample_purity。
    report:
        运行报告 dict：blocks（各样本 k_dim/gamma/K）、rounds、summary。

    Examples
    --------
    >>> adata_out, mc, report = build_metacells("cells.h5ad", sample_key="sampleID")
    >>> print(report["summary"])          # n_metacells / noise_rate / total_seconds
    >>> mc.obs["n_cells"].describe()      # 严格 20-35
    """
    cfg = config or PipelineConfig(
        n_top_genes=n_top_genes,
        n_jobs=n_jobs,
        capacity_lo=capacity_lo,
        capacity_hi=capacity_hi,
        max_rounds=max_rounds,
        pass_rule=pass_rule,
        **config_kwargs,
    )
    pipe = IterativeMetaCellPipeline(
        sample_key=sample_key, input_is_raw=input_is_raw, config=cfg
    )
    return pipe.run(_load(adata))


def build_metacells_cnv(
    cnv_adata: Union[ad.AnnData, PathLike],
    sample_key: str = "sampleID",
    n_top_features: int = 3000,
    n_jobs: int = -1,
    min_block_cells: int = 40,
    config: Optional[PipelineConfig] = None,
    **config_kwargs,
) -> tuple[ad.AnnData, dict]:
    """上皮/肿瘤细胞在 CNV 空间按病人分治构建 Metacell。

    适用场景：上皮（尤其恶性）在表达空间是连续谱，常规评分天然吃亏；
    CNV 空间克隆结构离散且病人特异。输入需先经 infer_cnv()（infercnvpy）
    产生。容量沿用主流水线的 [20,35]（可经 config 覆盖）。

    流程：方差 top-N 特征 -> 非负平移 -> 按样本分治（每样本自己的
    k_dim/gamma/K）-> 全局 mean 线 -> 失败池跨样本 rescue 1 轮 ->
    rescue 失败者内 median 分层放行 -> 剩余 is_noise。

    Parameters
    ----------
    cnv_adata:
        infercnvpy 输出的 AnnData 或 .h5ad 路径（X = 基因级 CNV 矩阵，
        可含负值；obs 需含 sample_key 列，建议含 cnv_score）。
    n_top_features:
        方差筛选的特征数（默认 3000；实测与全维质量持平、快 ~36%）。
    min_block_cells:
        小于该数的样本整块进失败池。

    Returns
    -------
    cnv_out:
        输入 AnnData 副本，obs 新增:
        - metacell_id: "CMC000123" 或 "noise"
        - metacell_stage: cnv_pass1 / cnv_rescue / cnv_stratified / cnv_noise
        - is_noise: bool
    cnv_report:
        dict：labels / stage / per_sample（各样本 k_dim/gamma/K）/
        n_metacells / noise_rate / n_pool。

    Examples
    --------
    >>> cnv_out, rep = build_metacells_cnv("epi_cnv.h5ad")
    >>> rep["noise_rate"]                    # 上皮 unresolved 比例（实测 ~15%）
    """
    data = _load(cnv_adata).copy()
    bcfg = CNVBranchConfig(
        n_top_features=n_top_features,
        sample_key=sample_key,
        n_jobs=n_jobs,
        min_block_cells=min_block_cells,
        pipeline=config or PipelineConfig(n_jobs=n_jobs, **config_kwargs),
    )
    res = run_cnv_branch(data, bcfg)
    data.obs["metacell_id"] = [
        f"CMC{i:06d}" if i >= 0 else "noise" for i in res["labels"]]
    data.obs["metacell_stage"] = [f"cnv_{s}" for s in res["stage"]]
    data.obs["is_noise"] = res["is_noise"]
    return data, res
