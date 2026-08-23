"""Metacell 可视化（可选依赖：pip install seacell_up[plot]）。

    from seacell_up.plotting import plot_metacell_umap
    mc, figs = plot_metacell_umap(mc_adata, batch_key="dominant_sample",
                                  color=["stage"], save="mc_umap.png")

从 scripts/plot_metacell_umap.py 固化的函数式封装。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import anndata as ad
import numpy as np

PathLike = Union[str, Path]


def _ensure_plot_deps():
    try:
        import scanpy as sc
        import harmonypy as hm
        import matplotlib.pyplot as plt
        return sc, hm, plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "可视化需要 scanpy + harmonypy + matplotlib + umap-learn："
            "pip install 'seacell_up[plot]'"
        ) from e


def harmonize_and_embed(
    mc: "ad.AnnData",
    batch_key: Optional[str],
    n_comps: int = 30,
    n_neighbors: int = 15,
    random_state: int = 0,
) -> "ad.AnnData":
    """PCA -> (可选 Harmony) -> neighbors -> UMAP。

    batch_key 为 None 时只做 PCA+UMAP；否则额外产出校正版嵌入
    obsm["X_pca_harmony"] / obsm["X_umap_harmony"]，并把 obsm["X_umap"]
    置为校正版（原始版存于 obsm["X_umap_raw"]）。

    输入 mc 建议: 表达空间伪体先 normalize_total+log1p+HVG+scale；
    CNV 矩阵伪体直接进（跳过 normalize）。
    """
    sc, hm, _ = _ensure_plot_deps()
    sc.tl.pca(mc, n_comps=n_comps, random_state=random_state,
              svd_solver="randomized")
    if batch_key is None:
        sc.pp.neighbors(mc, n_neighbors=n_neighbors, random_state=random_state)
        sc.tl.umap(mc, random_state=random_state)
        return mc
    ho = hm.run_harmony(mc.obsm["X_pca"], mc.obs, [batch_key],
                        max_iter_harmony=20, random_state=random_state)
    mc.obsm["X_pca_harmony"] = np.asarray(ho.Z_corr)  # (n_obs, n_pcs)
    sc.pp.neighbors(mc, n_neighbors=n_neighbors, use_rep="X_pca_harmony",
                    random_state=random_state)
    sc.tl.umap(mc, random_state=random_state)
    mc.obsm["X_umap_harmony"] = mc.obsm["X_umap"].copy()
    sc.pp.neighbors(mc, n_neighbors=n_neighbors, use_rep="X_pca",
                    random_state=random_state)
    sc.tl.umap(mc, random_state=random_state)
    mc.obsm["X_umap_raw"] = mc.obsm["X_umap"].copy()
    mc.obsm["X_umap"] = mc.obsm["X_umap_harmony"].copy()
    return mc


def plot_metacell_umap(
    mc_adata: "ad.AnnData",
    color: Sequence[str] = ("stage",),
    batch_key: Optional[str] = None,
    size: int = 10,
    cmap: Optional[str] = None,
    normalize_first: bool = False,
    save: Optional[PathLike] = None,
    show: bool = False,
):
    """Metacell UMAP 绘图（一步到位：PCA/Harmony/UMAP + 面板图）。

    Parameters
    ----------
    mc_adata:
        Metacell 伪体 AnnData（build_metacells 的 mc_adata，或 CNV 伪体）。
    color:
        obs 列名列表（如 ["dominant_sample", "stage", "n_cells"]），
        每列一个面板。
    batch_key:
        Harmony 批次列（如 "dominant_sample"）；None = 不校正。
    normalize_first:
        True 时先 normalize_total+log1p（表达空间伪体用；CNV 矩阵伪体
        保持 False，直接 PCA）。
    save:
        图片保存路径前缀（每列存一张 <save>_<col>.png）；None = 不保存。

    Returns
    -------
    (mc, figs):
        mc 为带 obsm["X_umap"]（及 _harmony/_raw）的 AnnData；
        figs 为 matplotlib figure 列表（每列一个单面板图）。

    Examples
    --------
    >>> adata_out, mc, report = build_metacells(adata, sample_key="sampleID")
    >>> mc.obs["n_cells"] = mc.obs["n_cells"].astype(float)
    >>> mc, figs = plot_metacell_umap(mc, color=["stage", "n_cells"],
    ...                               batch_key="dominant_sample", save="mc.png")
    """
    sc, _, plt = _ensure_plot_deps()
    mc = mc_adata.copy()
    if normalize_first:
        sc.pp.normalize_total(mc, target_sum=1e4)
        sc.pp.log1p(mc)
    harmonize_and_embed(mc, batch_key=batch_key)
    figs = []
    for col in color:
        fig, ax = plt.subplots(figsize=(7, 6))
        sc.pl.umap(mc, color=col, ax=ax, show=False, size=size,
                   cmap=cmap if mc.obs[col].dtype.kind in "fi" else None,
                   title=str(col))
        figs.append(fig)
        if save:
            fig.savefig(str(save).replace(".png", f"_{col}.png"),
                        bbox_inches="tight")
        if not show:
            plt.close(fig)
    return mc, figs
