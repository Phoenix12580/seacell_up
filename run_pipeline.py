#!/usr/bin/env python3
"""seacell_up 入口脚本：输入 AnnData (对象或 .h5ad 路径)，输出打好
Metacell 标签与噪音标记的细胞级 AnnData + Metacell 伪体 AnnData。

用法示例:
    python run_pipeline.py input.h5ad --sample-key sampleID \
        --output out_cells.h5ad --metacell-output out_metacells.h5ad
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="输入 .h5ad 路径")
    p.add_argument("--sample-key", default="sample_id", help="obs 中的样本列名")
    p.add_argument("--output", required=True, help="细胞级输出 .h5ad（含 metacell 标签/噪音标记）")
    p.add_argument("--metacell-output", default=None, help="Metacell 伪体输出 .h5ad")
    p.add_argument("--report-json", default=None, help="运行报告 JSON 输出路径")
    p.add_argument("--input-is-raw", default="auto", choices=["auto", "true", "false"],
                   help="X 是否原始计数（默认自动启发式判断）")
    p.add_argument("--n-top-genes", type=int, default=2048)
    p.add_argument("--hvg-mode", default="global", choices=["global", "per_sample"],
                   help="HVG 选择: global=全样本合并选一次(默认, 跨块特征统一); "
                        "per_sample=第一轮各样本独立选、回收轮用剩余细胞合并选")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--pass-rule", default="mean", choices=["mean", "median", "quantile"],
                   help="通过线规则（默认 mean=规范原文：低于全局平均分通过）")
    p.add_argument("--pass-quantile", type=float, default=0.75)
    p.add_argument("--gamma-min", type=float, default=5.0)
    p.add_argument("--gamma-max", type=float, default=500.0)
    p.add_argument("--log-level", default="INFO")
    # ---- 上皮 CNV 分支（可选; 输入为 infercnvpy 输出的 CNV 矩阵 h5ad,
    #      由 scripts/cnv_infer_epithelial.py 产生, 需 infercnvpy 环境） ----
    p.add_argument("--cnv-input", default=None,
                   help="上皮 CNV 矩阵 .h5ad（infercnvpy 输出）; 提供时对其中"
                        "细胞按样本分治重建 Metacell（容量 [20,35]），标签覆盖主输出")
    p.add_argument("--cnv-top-features", type=int, default=3000,
                   help="CNV 分支方差筛选的特征数（默认 3000）")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # worker 进程的日志也可见
    logging.getLogger("seacell_up").setLevel(args.log_level.upper())

    from seacell_up import IterativeMetaCellPipeline, PipelineConfig

    cfg = PipelineConfig(
        n_top_genes=args.n_top_genes,
        hvg_mode=args.hvg_mode,
        n_jobs=args.n_jobs,
        max_rounds=args.max_rounds,
        pass_rule=args.pass_rule,
        pass_quantile=args.pass_quantile,
        gamma_bounds=(args.gamma_min, args.gamma_max),
    )
    raw_flag = {"auto": None, "true": True, "false": False}[args.input_is_raw]

    t0 = time.time()
    adata = ad.read_h5ad(args.input)
    logging.info("读入 %s: %s, %.1fs", args.input, adata.shape, time.time() - t0)

    pipe = IterativeMetaCellPipeline(
        sample_key=args.sample_key, input_is_raw=raw_flag, config=cfg
    )
    adata_out, mc_adata, report = pipe.run(adata)

    # ---- 上皮 CNV 分支（可选）: 覆盖对应细胞的 Metacell 标签
    if args.cnv_input:
        from seacell_up import CNVBranchConfig, run_cnv_branch

        cnv = ad.read_h5ad(args.cnv_input)
        bcfg = CNVBranchConfig(n_top_features=args.cnv_top_features,
                               sample_key=args.sample_key,
                               n_jobs=args.n_jobs)
        bres = run_cnv_branch(cnv, bcfg)
        common = adata_out.obs_names.intersection(cnv.obs_names)
        pos = pd.Series(np.arange(cnv.n_obs), index=cnv.obs_names)
        rows = pos.reindex(common).astype(int).values
        adata_out.obs.loc[common, "metacell_id"] = [
            f"CMC{i:06d}" if i >= 0 else "noise" for i in bres["labels"][rows]]
        adata_out.obs.loc[common, "metacell_stage"] = [
            f"cnv_{s}" for s in bres["stage"][rows]]
        adata_out.obs.loc[common, "is_noise"] = bres["is_noise"][rows]
        report["cnv_branch"] = {
            "n_cells": int(cnv.n_obs), "n_metacells": bres["n_metacells"],
            "noise_rate": bres["noise_rate"], "n_pool": bres["n_pool"],
        }
        logging.info("CNV 分支: %d 细胞 -> %d Metacells, 噪音率 %.1f%%（标签已覆盖）",
                     cnv.n_obs, bres["n_metacells"], 100 * bres["noise_rate"])
        del cnv

    adata_out.write_h5ad(args.output, compression="gzip")
    print(f"细胞级输出 -> {args.output}")
    if args.metacell_output:
        mc_adata.write_h5ad(args.metacell_output, compression="gzip")
        print(f"Metacell 伪体输出 -> {args.metacell_output}")
    if args.report_json:
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"运行报告 -> {args.report_json}")

    s = report["summary"]
    print(
        f"\n===== 摘要 =====\n细胞 {s['n_cells']} | 样本 {s['n_samples']} | "
        f"Metacells {s['n_metacells']} | 噪音 {s['n_noise']} ({100 * s['noise_rate']:.2f}%)\n"
        f"第一阶段并行 {s['stage1_seconds']:.1f}s | 总耗时 {s['total_seconds']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
