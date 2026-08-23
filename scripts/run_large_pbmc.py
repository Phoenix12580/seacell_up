#!/usr/bin/env python3
"""PBMC 大测试：Zhao_AllCells（222,529 细胞 x 20,055 基因, 93 样本）。

验证点：端到端时间 / 峰值内存 / 噪音率 / Metacell 细胞类型纯度
（ct.main 与 ct.sub 为独立外部标注，仅用于事后验证）。

用法: python scripts/run_large_pbmc.py big.h5ad --sample-key sampleID
"""
from __future__ import annotations

import argparse
import json
import logging
import resource
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seacell_up import IterativeMetaCellPipeline, PipelineConfig  # noqa: E402

OUTDIR = Path(__file__).resolve().parents[1] / "results" / "zhao_222k"


def peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def purity_of(adata_out: ad.AnnData, key: str) -> pd.Series:
    passed = adata_out.obs[~adata_out.obs["is_noise"]]
    return passed.groupby("metacell_id", observed=True)[key].agg(
        lambda x: x.value_counts().iloc[0] / len(x)
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="AnnData .h5ad（原始计数, obs 含 sampleID 与 ct.main/ct.sub）")
    p.add_argument("--sample-key", default="sampleID")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    t0 = time.time()
    adata = ad.read_h5ad(args.input)
    logging.info("读入: %s, X dtype=%s (%.1fs, 峰值 %.1f GB)",
                 adata.shape, adata.X.dtype, time.time() - t0, peak_gb())

    cfg = PipelineConfig(n_jobs=16, n_top_genes=2048)
    pipe = IterativeMetaCellPipeline(sample_key=args.sample_key, config=cfg)
    adata_out, mc_adata, report = pipe.run(adata)
    logging.info("流水线完成, 峰值内存 %.1f GB", peak_gb())

    # ================= 事后验证 =================
    print("\n" + "=" * 62)
    print("大规模测试结果")
    print("=" * 62)
    s = report["summary"]
    print(f"细胞 {s['n_cells']:,} | 样本 {s['n_samples']} | Metacells {s['n_metacells']:,} | "
          f"噪音 {s['n_noise']:,} ({100 * s['noise_rate']:.2f}%)")
    print(f"第一阶段并行 {s['stage1_seconds']:.0f}s | 总耗时 {s['total_seconds']:.0f}s | "
          f"峰值内存 {peak_gb():.1f} GB")

    print("\n----- Metacell 阶段/尺寸分布 -----")
    print(mc_adata.obs.groupby("stage", observed=True)["n_cells"]
          .agg(["count", "sum", "mean", "min", "max"]).round(1))

    print("\n----- 细胞类型纯度（外部标注事后验证）-----")
    for key in ("ct.main", "ct.sub"):
        if key not in adata_out.obs:
            continue
        p = purity_of(adata_out, key)
        print(f"{key}: mean={p.mean():.4f} median={p.median():.4f} | "
              f">=0.95 的 Metacell 占比 {(p >= 0.95).mean() * 100:.1f}% | "
              f">=0.99 占比 {(p >= 0.99).mean() * 100:.1f}%")

    # 噪音细胞的类型构成：应偏向稀有/过渡/连续谱
    if "ct.sub" in adata_out.obs:
        noise_ct = adata_out.obs.loc[adata_out.obs["is_noise"], "ct.sub"].value_counts(normalize=True)
        keep_ct = adata_out.obs.loc[~adata_out.obs["is_noise"], "ct.sub"].value_counts(normalize=True)
        enr = (noise_ct / keep_ct.clip(lower=1e-6)).sort_values(ascending=False)
        print("\n----- 噪音细胞中富集的 ct.sub (噪音占比/全体占比, 前 8) -----")
        print(enr.head(8).round(2).to_string())

    # 第一阶段 gamma 分布
    gammas = [b["gamma"] for b in report["blocks"].values()]
    print(f"\n----- 第一阶段 gamma: 中位 {np.median(gammas):.1f}, "
          f"IQR [{np.percentile(gammas, 25):.1f}, {np.percentile(gammas, 75):.1f}], "
          f"范围 [{min(gammas):.1f}, {max(gammas):.1f}] -----")
    for r in report["rounds"]:
        print(f"第 {r['round']} 轮: 池 {r['n_pool']:,} 细胞 -> gamma={r['gamma']:.1f}, "
              f"K={r['n_metacells']}, {r['seconds']:.0f}s")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    adata_out.write_h5ad(OUTDIR / "cells.h5ad", compression="gzip")
    mc_adata.write_h5ad(OUTDIR / "metacells.h5ad", compression="gzip")
    with open(OUTDIR / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n输出已写入 {OUTDIR}")


if __name__ == "__main__":
    main()
