#!/usr/bin/env python3
"""上皮细胞两方案对比实验（方案 3 粗容量 vs 方案 4 CNV 空间）。

实验 A（方案 4）: 上皮在 CNV 空间（infercnvpy 基因级矩阵）建 Metacell，
                对照基线 = 表达空间已有全量结果的上皮统计。
实验 B（方案 3）: 表达空间 unresolved(noise) 细胞用粗容量 [80,160] 建
                super-Metacell，检验"连续谱而非噪音"假设。

评估：单轮通过率、cnv_score 组内一致性、ct.sub 纯度、找回率。
前置: scripts/cnv_infer_epithelial.py 已产生 results/zhao_222k/cnv/epi_cnv.h5ad
用法: python scripts/exp_epithelial_experiments.py input_raw.h5ad
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seacell_up import PipelineConfig, normalize_log  # noqa: E402
from seacell_up.pipeline import _process_block, _pass_threshold  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
ZHAO_OUT = HERE / "results" / "zhao_222k"
CNV_DIR = ZHAO_OUT / "cnv"


def mc_consistency(scores: pd.Series, labels: np.ndarray) -> float:
    """cnv_score 的组内/组间方差比（越小越一致）。"""
    df = pd.DataFrame({"s": scores.values, "mc": labels})
    within = df.groupby("mc")["s"].var().mean()
    total = df["s"].var()
    return float(within / max(total, 1e-12))


def purity(ct: pd.Series, labels: np.ndarray) -> float:
    df = pd.DataFrame({"ct": ct.values, "mc": labels})
    return float((df.groupby("mc")["ct"].agg(lambda x: x.value_counts().iloc[0] / len(x))).median())


def run_block(X, cfg_over: dict, seed: int, tag: str) -> dict:
    cfg = replace(PipelineConfig(), **cfg_over)
    t0 = time.time()
    r = _process_block(X.tocsr(), asdict(cfg), tag, seed)
    thr = _pass_threshold(r["mc_scores"], cfg)
    ok = r["mc_scores"] <= thr
    print(f"[{tag}] N={r['n']} gamma={r['gamma']:.1f} K={r['n_metacells']} "
          f"容量[{r['counts'].min()},{r['counts'].max()}] 单轮通过 {ok.sum()}/{len(ok)} "
          f"({ok.mean()*100:.1f}%) {time.time()-t0:.0f}s", flush=True)
    r["pass_mask"] = ok
    r["cfg"] = cfg
    return r


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("zhao_raw", help="原始 AnnData .h5ad（实验 B 需要原始计数）")
    args = p.parse_args()
    t0 = time.time()
    # ================= 基线：表达空间全量结果的上皮统计 =================
    cells = ad.read_h5ad(ZHAO_OUT / "zhao_cells.h5ad", backed="r")
    obs = cells.obs[:]
    cells.file.close()
    epi_obs = obs[obs["ct.main"] == "Epithelia"]
    stage_counts = epi_obs["metacell_stage"].value_counts()
    base_unresolved = (epi_obs["metacell_stage"] == "noise").mean()
    base_pass = (epi_obs["metacell_stage"].isin(["stage1", "rescue1"])).mean()
    print("=" * 62)
    print("基线（表达空间, 全量 3 轮迭代后的上皮）")
    print(f"  阶段分布: {dict(stage_counts)}")
    print(f"  unresolved(noise) 率: {base_unresolved*100:.1f}% | stage1+rescue1 通过率: {base_pass*100:.1f}%")

    # ================= 读 CNV 推断结果 =================
    cnv_adata = ad.read_h5ad(CNV_DIR / "epi_cnv.h5ad")
    cnv_adata = cnv_adata[cnv_adata.obs["cnv_ref"] == "tumor"].copy()
    print(f"\nCNV 矩阵: {cnv_adata.shape}, cnv_score 上皮: "
          f"median={cnv_adata.obs['cnv_score'].median():.3f}", flush=True)
    score_by_bc = cnv_adata.obs["cnv_score"].copy()

    # ================= 实验 A（方案 4）: CNV 空间建 Metacell =================
    print("\n" + "=" * 62)
    print("实验 A（方案 4）: 上皮在 CNV 空间建 Metacell [20,35]")
    X_cnv = cnv_adata.X
    X_cnv = sp.csr_matrix(X_cnv) if not sp.issparse(X_cnv) else X_cnv
    X_cnv = X_cnv.astype(np.float32)
    shift = abs(float(X_cnv.min()))  # NMF 需非负：全局平移保持相对结构
    X_cnv.data = X_cnv.data + shift
    del cnv_adata
    rA = run_block(X_cnv, {}, seed=101, tag="A_cnv_space")

    labelsA = rA["labels"]
    print(f"  cnv_score 组内/组间方差比: {mc_consistency(score_by_bc, labelsA):.4f} (越小越一致)")

    # 表达空间 unresolved 的上皮在 CNV 空间的找回率
    epi_noise_idx = epi_obs.index[epi_obs["metacell_stage"] == "noise"]
    common = score_by_bc.index.intersection(epi_noise_idx)
    pos = pd.Series(np.arange(len(score_by_bc)), index=score_by_bc.index)
    noise_rows = pos.loc[common].values
    recovered = rA["pass_mask"][labelsA[noise_rows]]
    print(f"  表达空间 {len(noise_rows)} 个 unresolved 上皮: "
          f"CNV 空间被找回 {recovered.sum()} ({recovered.mean()*100:.1f}%)")

    # ================= 实验 B（方案 3）: 连续谱粗容量 super-MC =================
    print("\n" + "=" * 62)
    print("实验 B（方案 3）: 表达空间 unresolved 细胞建粗容量 [80,160] super-MC")
    noise_obs = obs[obs["metacell_stage"] == "noise"]
    print(f"  输入: {len(noise_obs)} 细胞, ct.main 构成: "
          f"{dict(noise_obs['ct.main'].value_counts().head(3))}")
    raw = ad.read_h5ad(args.zhao_raw, backed="r")
    noise_pos = pd.Series(np.arange(raw.n_obs), index=raw.obs_names).loc[noise_obs.index].values
    Xn = sp.csr_matrix(raw[noise_pos].X)
    raw.file.close()
    Xn = normalize_log(Xn.astype(np.float32))

    rB = run_block(Xn, {"capacity_lo": 80, "capacity_hi": 160}, seed=202, tag="B_coarse")
    labelsB = rB["labels"]
    cnv_scores_B = score_by_bc.reindex(noise_obs.index).dropna()
    posB = pd.Series(np.arange(len(noise_obs)), index=noise_obs.index)
    rowsB = posB.loc[cnv_scores_B.index].values
    print(f"  cnv_score 组内/组间方差比(粗MC): {mc_consistency(cnv_scores_B, labelsB[rowsB]):.4f}")
    print(f"  ct.sub 纯度(中位): {purity(noise_obs['ct.sub'], labelsB):.3f}")
    print(f"  单轮通过 {rB['pass_mask'].sum()}/{len(rB['pass_mask'])} "
          f"({rB['pass_mask'].mean()*100:.1f}%) -> 通过者即'连续谱可结构化'细胞")

    # ================= 对照表 =================
    print("\n" + "=" * 62)
    print("结论对照")
    print(f"  基线(表达空间):  上皮 unresolved 率 {base_unresolved*100:.1f}%")
    print(f"  方案4(CNV空间):  单轮通过率 {rA['pass_mask'].mean()*100:.1f}%, "
          f"表达空间噪音找回率 {recovered.mean()*100:.1f}%")
    print(f"  方案3(粗容量):   单轮通过率 {rB['pass_mask'].mean()*100:.1f}% "
          f"(容量[{rB['counts'].min()},{rB['counts'].max()}])")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
