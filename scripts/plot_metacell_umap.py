#!/usr/bin/env python3
"""Metacell UMAP 可视化（Harmony 批次校正版）。

图 A: 主流水线 MC（表达空间伪体）—— Harmony(dominant_sample) 前后对照,
      着色 ct.main / ct.sub / stage / ct.main 纯度
图 B: CNV 分支上皮 MC（CNV 矩阵伪体）—— Harmony(dominant_sample) 前后对照,
      着色 mean CNV score / stage / n_cells
输出: results/zhao_222k/figures/*.png
前置: run_large_pbmc.py + run_cnv_branch_top3000.py 的产物
"""
from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import harmonypy as hm  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logging.getLogger("harmonypy").setLevel(logging.WARNING)
log = logging.getLogger("umap")

HERE = Path(__file__).resolve().parents[1]
ZHAO_OUT = HERE / "results" / "zhao_222k"
CNV_DIR = ZHAO_OUT / "cnv"
FIG = ZHAO_OUT / "figures"


def embed_both(mc: ad.AnnData, batch_key: str) -> None:
    """PCA -> Harmony 校正与未校正两套 neighbors+UMAP, 存 obsm X_umap_harmony/_raw。"""
    sc.tl.pca(mc, n_comps=30, random_state=0, svd_solver="randomized")
    ho = hm.run_harmony(mc.obsm["X_pca"], mc.obs, [batch_key],
                        max_iter_harmony=20, random_state=0)
    mc.obsm["X_pca_harmony"] = np.asarray(ho.Z_corr)  # (n_obs, n_pcs)
    sc.pp.neighbors(mc, n_neighbors=15, use_rep="X_pca_harmony", random_state=0)
    sc.tl.umap(mc, random_state=0)
    mc.obsm["X_umap_harmony"] = mc.obsm["X_umap"].copy()
    sc.pp.neighbors(mc, n_neighbors=15, use_rep="X_pca", random_state=0)
    sc.tl.umap(mc, random_state=0)
    mc.obsm["X_umap_raw"] = mc.obsm["X_umap"].copy()


def use(mc: ad.AnnData, which: str) -> None:
    mc.obsm["X_umap"] = mc.obsm[f"X_umap_{which}"].copy()


def dominant_ct(cells_obs: pd.DataFrame, key: str) -> pd.DataFrame:
    g = cells_obs.groupby("metacell_id", observed=True)[key]
    return pd.DataFrame({
        f"dom_{key}": g.agg(lambda x: x.value_counts().index[0]),
        f"pur_{key}": g.agg(lambda x: x.value_counts().iloc[0] / len(x)),
    })


def fig_a() -> None:
    t0 = time.time()
    mc = ad.read_h5ad(ZHAO_OUT / "zhao_metacells.h5ad")
    cells = ad.read_h5ad(ZHAO_OUT / "zhao_cells.h5ad", backed="r")
    obs = cells.obs[["metacell_id", "ct.main", "ct.sub"]]
    cells.file.close()
    mc.obs = mc.obs.join(dominant_ct(obs, "ct.main")).join(dominant_ct(obs, "ct.sub"))
    log.info("图 A 输入: %s MC", mc.shape)

    sc.pp.normalize_total(mc, target_sum=1e4)
    sc.pp.log1p(mc)
    sc.pp.highly_variable_genes(mc, n_top_genes=1000, flavor="seurat")
    mc = mc[:, mc.var["highly_variable"]].copy()
    sc.pp.scale(mc, max_value=10)
    embed_both(mc, "dominant_sample")
    log.info("图 A 嵌入完成 (%.0fs)", time.time() - t0)

    keep = mc.obs["dom_ct.sub"].value_counts().index[:14]
    mc.obs["ct.sub.simp"] = mc.obs["dom_ct.sub"].where(
        mc.obs["dom_ct.sub"].isin(keep), "other")

    sc.settings.set_figure_params(dpi=120, frameon=False)
    panels = [("dom_ct.main", {}, "dominant ct.main"),
              ("ct.sub.simp", {}, "dominant ct.sub (top14+other)"),
              ("stage", {}, "stage (stage1→rescue1/2/3)"),
              ("pur_ct.main", {"cmap": "viridis"}, "ct.main purity")]
    for which, tag, fname in [("harmony", "Harmony 校正 (dominant_sample)",
                               "A_metacell_umap_harmony.png"),
                              ("raw", "无校正对照", "A_metacell_umap_notharmony.png")]:
        use(mc, which)
        fig, axes = plt.subplots(2, 2, figsize=(17, 15))
        for ax, (color, kw, title) in zip(axes.ravel(), panels):
            sc.pl.umap(mc, color=color, ax=ax, show=False, size=8, title=title, **kw)
        fig.suptitle(f"主流水线 Metacell UMAP — 表达空间 | {tag}",
                     fontsize=15, y=0.995)
        fig.tight_layout()
        fig.savefig(FIG / fname, bbox_inches="tight")
        plt.close(fig)
        log.info("图 A[%s] -> %s", which, fname)


def fig_b() -> None:
    t0 = time.time()
    cnv = ad.read_h5ad(CNV_DIR / "epi_cnv.h5ad")
    cnv = cnv[(cnv.obs["cnv_ref"] == "tumor").values].copy()
    lab = ad.read_h5ad(CNV_DIR / "epi_cnv_top3000_labels.h5ad").obs
    cnv = cnv[cnv.obs_names.isin(lab.index)].copy()
    cnv.obs = cnv.obs.join(lab)
    cnv = cnv[(~cnv.obs["is_noise"].values)].copy()

    codes, uniq = pd.factorize(cnv.obs["metacell_id"])
    ind = sp.csr_matrix((np.ones(len(codes), dtype=np.float32),
                         (np.arange(len(codes)), codes)), shape=(len(codes), len(uniq)))
    X = (ind.T @ sp.csr_matrix(cnv.X, dtype=np.float32)).tocsr()
    g = cnv.obs.groupby("metacell_id", observed=True)
    mc = ad.AnnData(X=X, obs=pd.DataFrame({
        "n_cells": g.size(), "stage": g["stage"].agg(lambda x: x.iloc[0]),
        "mean_cnv_score": g["cnv_score"].mean(),
        "n_samples": g["sampleID"].nunique(),
        "dominant_sample": g["sampleID"].agg(lambda x: x.value_counts().index[0]),
    }, index=uniq))
    log.info("图 B MC: %s (%.0fs)", mc.shape, time.time() - t0)

    embed_both(mc, "dominant_sample")
    log.info("图 B 嵌入完成 (%.0fs)", time.time() - t0)

    panels = [("mean_cnv_score", {"cmap": "Reds"}, "mean CNV score (恶性信号)"),
              ("stage", {}, "stage (pass1/rescue/stratified)"),
              ("n_cells", {"cmap": "Blues"}, "metacell size [20,35]")]
    for which, tag, fname in [("harmony", "Harmony 校正 (dominant_sample)",
                               "B_metacell_umap_cnv_harmony.png"),
                              ("raw", "无校正对照", "B_metacell_umap_cnv_notharmony.png")]:
        use(mc, which)
        fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
        for ax, (color, kw, title) in zip(axes, panels):
            sc.pl.umap(mc, color=color, ax=ax, show=False, size=12, title=title, **kw)
        fig.suptitle(f"上皮 CNV 分支 Metacell UMAP — 按病人分治/top3000/[20,35] (CNV 空间) | {tag}",
                     fontsize=15, y=1.02)
        fig.tight_layout()
        fig.savefig(FIG / fname, bbox_inches="tight")
        plt.close(fig)
        log.info("图 B[%s] -> %s", which, fname)

    hi = mc.obs[mc.obs["mean_cnv_score"] > mc.obs["mean_cnv_score"].quantile(0.9)]
    log.info("top10%% 恶性 MC %d 个, 平均来自 %.2f 个样本 (期望 ~1: 克隆病人特异)",
             len(hi), hi["n_samples"].mean())


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig_a()
    fig_b()
    log.info("全部完成")


if __name__ == "__main__":
    main()
