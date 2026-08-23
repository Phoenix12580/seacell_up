#!/usr/bin/env python3
"""上皮 CNV 推断（需 infercnvpy 环境，如 infercnvpy 0.6.1）。

流程：全量 -> 基因坐标注释(Gencode v44 CSV: gene_name,chromosome,start,end)
-> log-normalize -> infercnvpy.tl.infercnv（Immune 为参考, 下采样 20K +
Epithelia 全量）-> per-cell cnv score -> 保存。

坐标 CSV 可从 Gencode GTF 生成（见 README）。inference 与下游分支在
不同环境跑时, 输出 h5ad 直接传给 run_cnv_branch / --cnv-input。

用法: python scripts/cnv_infer_epithelial.py input.h5ad hg38_gene_coords.csv.gz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import infercnvpy as cnv

OUTDIR = Path(__file__).resolve().parents[1] / "results" / "zhao_222k" / "cnv"
N_REF = 20_000  # 免疫参考下采样（CNV 参考无需全量）


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="AnnData .h5ad（原始计数, obs 含 ct.main 与 sampleID）")
    p.add_argument("coords", help="基因坐标 CSV(.gz): 列 gene_name,chromosome,start,end")
    args = p.parse_args()
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(args.input)
    print(f"读入 {adata.shape} ({time.time()-t0:.0f}s)", flush=True)

    # ---- 基因坐标注释（保存映射，concat 会丢 var 列，之后重挂）
    coords = pd.read_csv(args.coords)
    m = coords.set_index("gene_name")

    def annotate(ad_obj) -> None:
        base = pd.Series(ad_obj.var_names).str.replace(r"\.\d+$", "", regex=True)
        ad_obj.var["chromosome"] = [m.loc[b, "chromosome"] if b in m.index else None
                                    for b in base]
        ad_obj.var["start"] = [m.loc[b, "start"] if b in m.index else np.nan for b in base]
        ad_obj.var["end"] = [m.loc[b, "end"] if b in m.index else np.nan for b in base]

    base = pd.Series(adata.var_names).str.replace(r"\.\d+$", "", regex=True)
    keep = base.isin(m.index).values
    adata = adata[:, keep].copy()
    print(f"坐标可注释基因: {adata.shape[1]} ({time.time()-t0:.0f}s)", flush=True)

    # ---- 免疫参考(下采样) + 上皮(全量)
    epi = adata[adata.obs["ct.main"] == "Epithelia"].copy()
    imm = adata[adata.obs["ct.main"] == "Immune"]
    imm = imm[np.sort(np.random.default_rng(0).choice(imm.n_obs, N_REF, replace=False))].copy()
    imm.obs["cnv_ref"] = "ref_immune"
    epi.obs["cnv_ref"] = "tumor"
    adata = sc.concat([epi, imm], join="inner")
    del epi, imm
    annotate(adata)  # concat 丢 var 列, 重挂坐标
    print(f"目标矩阵 {adata.shape}: 上皮 + {N_REF} 免疫参考 ({time.time()-t0:.0f}s)", flush=True)

    # ---- log-normalize (infercnv 输入要求)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ---- CNV 推断
    cnv.tl.infercnv(
        adata,
        reference_key="cnv_ref",
        reference_cat=["ref_immune"],
        window_size=250,
        n_jobs=8,
    )
    print(f"infercnv 完成 ({time.time()-t0:.0f}s), X={adata.shape} 窗口化", flush=True)
    adata.write_h5ad(OUTDIR / "epi_cnv_partial.h5ad", compression="gzip")  # 防后续步骤失败重跑

    # ---- per-cell CNV score（infercnvpy 0.6 要求先在 CNV 空间聚类）
    cnv.tl.pca(adata, n_comps=30, random_state=0)
    cnv.pp.neighbors(adata, n_neighbors=10, random_state=0)
    cnv.tl.leiden(adata, resolution=1.0, key_added="cnv_leiden")
    cnv.tl.cnv_score(adata)
    print("cnv_score 上皮 vs 免疫参考:")
    print(adata.obs.groupby("cnv_ref")["cnv_score"].agg(["mean", "median", "count"]))

    adata.write_h5ad(OUTDIR / "epi_cnv.h5ad", compression="gzip")
    print(f"输出 -> {OUTDIR / 'epi_cnv.h5ad'} (总耗时 {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
