#!/usr/bin/env python3
"""PBMC 小测试：pbmc3k（2700 细胞，原始计数，无样本列）。

无真实样本注释，确定性对半切分为 2 个伪样本以驱动两阶段流程；
用经典 PBMC marker 基因做事后合理性检查（非算法输入）。

用法: python scripts/run_small_pbmc.py pbmc3k_raw.h5ad
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seacell_up import IterativeMetaCellPipeline, PipelineConfig  # noqa: E402

OUTDIR = Path(__file__).resolve().parents[1] / "results" / "pbmc3k"

MARKERS = {  # 细胞类型 -> marker 基因（pbmc3k 参考图谱经典组合）
    "T_cell": ["CD3D", "CD3E", "IL7R"],
    "B_cell": ["MS4A1", "CD79A"],
    "NK": ["NKG7", "GNLY"],
    "Monocyte": ["LST1", "FCN1", "CD14"],
    "DC": ["FCER1A", "CLEC10A"],
    "Platelet": ["PPBP", "PF4"],
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="pbmc3k 原始计数 .h5ad（无样本列, 自动伪 2 样本）")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    t0 = time.time()
    adata = ad.read_h5ad(args.input)
    logging.info("读入 pbmc3k: %s (%.1fs)", adata.shape, time.time() - t0)

    # 伪样本对半分（确定性）
    half = adata.n_obs // 2
    adata.obs["sample_id"] = ["S0"] * half + ["S1"] * (adata.n_obs - half)
    adata.obs_names_make_unique()

    cfg = PipelineConfig(n_jobs=2)
    pipe = IterativeMetaCellPipeline(sample_key="sample_id", config=cfg)
    adata_out, mc_adata, report = pipe.run(adata)

    # ---- Metacell 尺寸与阶段分布
    print("\n===== Metacell 概况 =====")
    print(mc_adata.obs.groupby("stage", observed=True)["n_cells"].agg(["count", "sum", "mean", "min", "max"]))
    s = report["summary"]
    print(f"噪音率: {100 * s['noise_rate']:.2f}% | Metacells: {s['n_metacells']} | "
          f"总耗时 {s['total_seconds']:.1f}s")

    # ---- marker 事后检查: 每个 Metacell 的伪体中 marker 均值
    print("\n===== 每 Metacell 最高 marker（事后验证，非算法输入）=====")
    raw = np.asarray(adata_out.X.todense()) if hasattr(adata_out.X, "todense") else adata_out.X
    genes = list(adata_out.var_names)
    gid = {g: i for i, g in enumerate(genes)}
    mc_ids = adata_out.obs["metacell_id"].values
    import pandas as pd

    order = [n for n in mc_adata.obs_names]
    rows = []
    for n in order[:12]:  # 展示前 12 个
        members = np.where(mc_ids == n)[0]
        if len(members) == 0 or "noise" in n:
            continue
        best_type, best_score = "?", -np.inf
        for ctype, markers in MARKERS.items():
            idx = [gid[m] for m in markers if m in gid]
            if not idx:
                continue
            sc = raw[members][:, idx].mean()
            if sc > best_score:
                best_type, best_score = ctype, sc
        rows.append({"metacell": n, "n_cells": len(members),
                     "top_marker_type": best_type, "mean_expr": round(best_score, 2)})
    print(pd.DataFrame(rows).to_string(index=False))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    adata_out.write_h5ad(OUTDIR / "pbmc3k_metacells_cells.h5ad", compression="gzip")
    mc_adata.write_h5ad(OUTDIR / "pbmc3k_metacells.h5ad", compression="gzip")
    print(f"\n输出已写入 {OUTDIR}")


if __name__ == "__main__":
    main()
