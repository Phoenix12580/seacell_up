#!/usr/bin/env python3
"""hvg_mode 对比实验：global（全样本合并选一次）vs per_sample（每块独立选）。

Zhao 抽 10 个样本（混合大小/组织/类型构成），同参数各跑一遍主流水线，
对比：ct.main / ct.sub 的 MC 纯度、噪音率、每样本 γ、耗时。

per_sample 语义 = 第一轮各样本独立选 HVG；回收轮在合并后的剩余细胞上
选 HVG（剩余细胞的"全局"HVG）——由 _process_block 在块内选 HVG 自然实现。

用法: python scripts/exp_hvg_mode.py [zhao_raw.h5ad]
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seacell_up import build_metacells  # noqa: E402

INPUT = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/y413007/lihaosci/data_collection/Zhao_AllCells.h5ad"

# 确定性抽样: 覆盖 Tumor/Benign/Normal
PICK = ["TUMOR", "BMET11-Tumor", "D5.tumor", "BMM5-Benign",
        "LN_normal_LEFT", "GSM4773522.primary", "CRPC4", "PRAD175.Primary",
        "BPH1", "UREPR1.Normal"]


def evaluate(adata_out: ad.AnnData, tag: str, dt: float) -> dict:
    obs = adata_out.obs
    passed = obs[~obs["is_noise"]]
    out = {"mode": tag, "seconds": round(dt)}
    for key in ("ct.main", "ct.sub"):
        p = passed.groupby("metacell_id", observed=True)[key].agg(
            lambda x: x.value_counts().iloc[0] / len(x))
        out[f"{key}_median"] = round(p.median(), 4)
        out[f"{key}_mean"] = round(p.mean(), 4)
        out[f"{key}>=0.95"] = round((p >= 0.95).mean(), 4)
    out["noise_rate"] = round(obs["is_noise"].mean(), 4)
    # 稀有 ct.sub 回收（子集中占比 <2% 的类型）
    vc = obs["ct.sub"].value_counts()
    rare_types = vc[vc / len(obs) < 0.02].index
    if len(rare_types):
        rare = obs[obs["ct.sub"].isin(rare_types)]
        out["rare_kept"] = round(1 - rare["is_noise"].mean(), 4)
        out["n_rare_cells"] = int(len(rare))
    return out


def main() -> None:
    adata = ad.read_h5ad(INPUT, backed="r")
    vc = adata.obs["sampleID"].value_counts()
    picks = [s for s in PICK if s in vc.index]
    while len(picks) < 10:  # 不足则补
        for s in vc.index:
            if s not in picks:
                picks.append(s)
                break
    idx = np.where(adata.obs["sampleID"].astype(str).isin(picks).values)[0]
    sub = ad.AnnData(X=sp.csr_matrix(adata[idx].X), obs=adata.obs.iloc[idx].copy())
    sub.obs["sampleID"] = sub.obs["sampleID"].astype(str)
    adata.file.close()
    print(f"子集: {sub.shape}, {sub.obs['sampleID'].nunique()} 样本, "
          f"ct.main: {dict(sub.obs['ct.main'].value_counts())}", flush=True)

    rows, gammas_by_mode = [], {}
    for mode in ("global", "per_sample"):
        print(f"\n===== hvg_mode={mode} =====", flush=True)
        t0 = time.time()
        out, mc, rep = build_metacells(sub, sample_key="sampleID", n_jobs=8,
                                       hvg_mode=mode)
        dt = time.time() - t0
        rows.append(evaluate(out, mode, dt))
        gammas_by_mode[mode] = {b: v["gamma"] for b, v in rep["blocks"].items()}
        rows[-1]["n_metacells"] = rep["summary"]["n_metacells"]

    df = pd.DataFrame(rows).set_index("mode")
    print("\n" + "=" * 70)
    print("hvg_mode 对比 (同参数)")
    print(df.T.to_string())

    g = pd.DataFrame(gammas_by_mode)
    both = g.dropna()
    if len(both):
        print(f"\n每样本 gamma 相关性 (Pearson): {both['global'].corr(both['per_sample']):.3f}")
        print(f"gamma 差异: median|Δ|={abs(both['global']-both['per_sample']).median():.1f}")


if __name__ == "__main__":
    main()
