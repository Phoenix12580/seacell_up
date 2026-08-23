"""CNV 推断（可选依赖：infercnvpy + scanpy 环境）。

函数式封装（原 scripts/cnv_infer_epithelial.py）。注意 infercnvpy
与主流水线可在不同环境（如各自的 kernel/venv）：本模块只负责产出
cnv AnnData，之后在任何装有 seacell_up 核心依赖的环境里用
build_metacells_cnv() 消费它。

    from seacell_up.cnv_infer import infer_cnv
    cnv_adata = infer_cnv("cells.h5ad", "hg38_gene_coords.csv.gz",
                          reference_key="ct.main", reference_cat="Immune",
                          target_cat="Epithelia")
    cnv_adata.write_h5ad("epi_cnv.h5ad", compression="gzip")

坐标 CSV 的列: gene_name, chromosome, start, end（可从 Gencode GTF 生成，
见 README）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def load_gene_coords(coords: Union[pd.DataFrame, PathLike]) -> pd.DataFrame:
    """读入/直通基因坐标表（列: gene_name, chromosome, start, end）。"""
    if isinstance(coords, pd.DataFrame):
        return coords.set_index("gene_name")
    return pd.read_csv(str(coords)).set_index("gene_name")


def annotate_genes(adata, m: pd.DataFrame):
    """给 adata.var 挂 chromosome/start/end（sc.concat 会丢 var 列，需重挂）。"""
    base = pd.Series(adata.var_names).str.replace(r"\.\d+$", "", regex=True)
    adata.var["chromosome"] = [m.loc[b, "chromosome"] if b in m.index else None
                               for b in base]
    adata.var["start"] = [m.loc[b, "start"] if b in m.index else np.nan for b in base]
    adata.var["end"] = [m.loc[b, "end"] if b in m.index else np.nan for b in base]
    return adata


def infer_cnv(
    adata,
    coords: Union[pd.DataFrame, PathLike],
    reference_key: str,
    reference_cat: str,
    target_cat: str,
    n_ref: int = 20_000,
    window_size: int = 250,
    random_state: int = 0,
):
    """以参考群（如免疫细胞）推断目标群（如上皮）的 CNV 矩阵。

    Parameters
    ----------
    adata:
        AnnData（或 .h5ad 路径），原始计数，obs 含 reference_key 列
        （取值含 reference_cat 与 target_cat）。
    coords:
        基因坐标表（DataFrame 或 CSV(.gz) 路径）。
    reference_key / reference_cat / target_cat:
        如 reference_key="ct.main", reference_cat="Immune",
        target_cat="Epithelia"。
    n_ref:
        参考群下采样数（CNV 参考无需全量；0 = 不下采样）。
    window_size:
        infercnvpy 平滑窗口（基因数）。

    Returns
    -------
    AnnData:
        目标+参考细胞，X = 基因级 CNV 矩阵，obs 含 cnv_ref
        ("tumor"/"ref_...")、cnv_leiden、cnv_score。
        传给 build_metacells_cnv() 前建议过滤 obs["cnv_ref"]=="tumor"。
    """
    try:
        import scanpy as sc
        import infercnvpy as cnv
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "infer_cnv 需要 infercnvpy + scanpy（与核心包可装在不同环境）。"
            "请在含这两个包的环境运行，或参考 scripts/cnv_infer_epithelial.py。"
        ) from e

    m = load_gene_coords(coords)
    if not isinstance(adata, sc.AnnData):
        adata = sc.read_h5ad(str(adata))

    base = pd.Series(adata.var_names).str.replace(r"\.\d+$", "", regex=True)
    adata = adata[:, base.isin(m.index).values].copy()

    tgt = adata[adata.obs[reference_key] == target_cat].copy()
    ref = adata[adata.obs[reference_key] == reference_cat]
    if n_ref and ref.n_obs > n_ref:
        sel = np.sort(np.random.default_rng(random_state).choice(
            ref.n_obs, n_ref, replace=False))
        ref = ref[sel].copy()
    tgt.obs["cnv_ref"] = "tumor"
    ref.obs["cnv_ref"] = f"ref_{reference_cat}"
    out = sc.concat([tgt, ref], join="inner")
    del tgt, ref
    annotate_genes(out, m)  # concat 丢 var 列, 重挂坐标

    sc.pp.normalize_total(out, target_sum=1e4)
    sc.pp.log1p(out)
    cnv.tl.infercnv(out, reference_key="cnv_ref",
                    reference_cat=[f"ref_{reference_cat}"],
                    window_size=window_size, n_jobs=8)

    # per-cell CNV score（infercnvpy 0.6 要求先在 CNV 空间聚类）
    cnv.tl.pca(out, n_comps=30, random_state=random_state)
    cnv.pp.neighbors(out, n_neighbors=10, random_state=random_state)
    cnv.tl.leiden(out, resolution=1.0, key_added="cnv_leiden")
    cnv.tl.cnv_score(out)
    return out
