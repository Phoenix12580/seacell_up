#!/usr/bin/env python3
"""实验 A-v2（方案 4 按样本分治）：上皮在 CNV 空间、按样本独立建 Metacell。

与混合版（exp_epithelial_experiments.py 实验 A）的区别：
- 每个样本独立选 k_dim / gamma / K（CNV 克隆结构是病人特异的）
- 流程：按样本分治单轮 -> 失败者跨样本 rescue 1 轮 -> 评分分层回收
  （失败者分布内 median 线）-> 剩余 is_noise
- 全程细容量 [20,35]，无粗 MC

对比基线：混合版单轮 44.7% / 找回率 21.3% / 方差比 0.343；
表达空间基线上皮 unresolved 27.0%。
前置: scripts/cnv_infer_epithelial.py 已产生 results/zhao_222k/cnv/epi_cnv.h5ad
"""
from __future__ import annotations

import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed, parallel_backend

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seacell_up import PipelineConfig  # noqa: E402
from seacell_up.pipeline import _process_block, _stable_seed  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
CNV_ADATA = HERE / "results" / "zhao_222k" / "cnv" / "epi_cnv.h5ad"
ZHAO_OUT = HERE / "results" / "zhao_222k"


def main() -> None:
    t0 = time.time()
    # ================= 数据 =================
    cnv = ad.read_h5ad(CNV_ADATA)
    keep = (cnv.obs["cnv_ref"] == "tumor").values
    names = cnv.obs_names[keep].copy()  # X 行序对应的 barcode
    cnv = cnv[keep].copy()
    score = cnv.obs["cnv_score"].copy()
    sample_ids = cnv.obs["sampleID"].astype(str).values
    ct_sub = cnv.obs["ct.sub"].astype(str).values
    X = sp.csr_matrix(cnv.X, dtype=np.float32)
    del cnv
    shift = abs(float(X.min()))  # NMF 需非负：全局平移（与混合版一致，保持跨块可比）
    X.data = X.data + shift
    print(f"上皮 CNV 矩阵: {X.shape}, {pd.unique(sample_ids).size} 样本 ({time.time()-t0:.0f}s)",
          flush=True)

    # 表达空间 unresolved 索引（找回率统计用）
    cells = ad.read_h5ad(ZHAO_OUT / "zhao_cells.h5ad", backed="r")
    epi_obs = cells.obs[(cells.obs["ct.main"] == "Epithelia").values]
    cells.file.close()
    expr_noise = set(epi_obs.index[epi_obs["metacell_stage"] == "noise"])
    pos_all = pd.Series(np.arange(X.shape[0]), index=names)
    print(f"表达空间上皮 unresolved: {len(expr_noise)}", flush=True)

    # ================= 按样本分治单轮 =================
    cfg_dict = asdict(PipelineConfig())  # 默认容量 [20,35]
    blocks: dict = {}
    too_small: dict = {}
    for sid in pd.unique(sample_ids):
        idx = np.where(sample_ids == sid)[0]
        if len(idx) >= 40:
            blocks[sid] = idx
        else:
            too_small[sid] = idx
    print(f"分块: {len(blocks)} 个样本块(>=40 细胞), {len(too_small)} 个微样本直接进失败池",
          flush=True)

    t1 = time.time()
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=16)(
            delayed(_process_block)(X[idx].tocsr(), cfg_dict, str(sid),
                                    _stable_seed(0, f"Acnv|{sid}"))
            for sid, idx in blocks.items()
        )
    print(f"分治单轮并行完成: {time.time()-t1:.0f}s", flush=True)

    labels = np.full(X.shape[0], -1, dtype=np.int64)   # 全局 MC 标签
    mc_of_cell = np.full(X.shape[0], -1, dtype=np.int64)
    scores_by_mc: list[float] = []
    fail_cells: list[np.ndarray] = []
    kdims, gammas, ks = [], [], []
    for r in results:
        sid = r["block_id"]
        idx = blocks[sid]
        kdims.append(r["k_dim"]); gammas.append(r["gamma"]); ks.append(r["n_metacells"])
        for m in range(r["n_metacells"]):
            members = idx[r["labels"] == m]
            scores_by_mc.append(float(r["mc_scores"][m]))
            mc_of_cell[members] = len(scores_by_mc) - 1
            labels[members] = len(scores_by_mc) - 1
    all_scores = np.array(scores_by_mc)
    thr1 = float(all_scores.mean())
    pass1 = all_scores <= thr1
    for mc in np.where(~pass1)[0]:
        fail_cells.append(np.where(mc_of_cell == mc)[0])
    for sid_idx in too_small.values():
        fail_cells.append(sid_idx)
    pool = np.concatenate(fail_cells) if fail_cells else np.array([], dtype=int)
    pass_rate1 = pass1.sum() / len(pass1)
    print(f"\n[1] 分治单轮: K={len(all_scores)} MC, k_dim 中位 {np.median(kdims):.0f} "
          f"(范围 {min(kdims)}-{max(kdims)}), gamma 中位 {np.median(gammas):.1f} "
          f"(范围 {min(gammas):.1f}-{max(gammas):.1f})", flush=True)
    print(f"    全局通过线(mean)={thr1:.4f} -> 单轮通过 {pass1.sum()}/{len(pass1)} "
          f"({pass_rate1*100:.1f}%) | 失败池 {len(pool)} 细胞", flush=True)

    # ================= 失败者跨样本 rescue 1 轮 =================
    rescued = np.zeros(X.shape[0], dtype=bool)
    stage = np.full(X.shape[0], "fail", dtype=object)
    stage[np.isin(mc_of_cell, np.where(pass1)[0])] = "pass1"
    if len(pool) >= 60:
        t2 = time.time()
        r2 = _process_block(X[pool].tocsr(), cfg_dict, "rescue_cnv", _stable_seed(0, "Rcnv"))
        thr2 = float(r2["mc_scores"].mean())
        pass2 = r2["mc_scores"] <= thr2
        mc_of_cell2 = np.full(X.shape[0], -1, dtype=np.int64)
        for m in range(r2["n_metacells"]):
            members = pool[r2["labels"] == m]
            mc_of_cell2[members] = m
        rescued[np.isin(mc_of_cell2, np.where(pass2)[0])] = True
        print(f"[2] rescue 1 轮: 池 {len(pool)} -> K={r2['n_metacells']}, "
              f"mean 线={thr2:.4f}, 通过 {pass2.sum()}/{len(pass2)} "
              f"({pass2.mean()*100:.1f}%) ({time.time()-t2:.0f}s)", flush=True)

        # ================= 评分分层: rescue 失败者在自身分布内 median 再过一次 =================
        fail2_scores = r2["mc_scores"][~pass2]
        thr3 = float(np.median(fail2_scores))
        pass3 = r2["mc_scores"] <= thr3
        stratified = np.isin(mc_of_cell2, np.where(pass3)[0])
        print(f"[3] 评分分层: 失败者内 median 线={thr3:.4f} -> 累计放行 "
              f"{pass3.sum()}/{len(pass3)} ({pass3.mean()*100:.1f}%)", flush=True)
    else:
        mc_of_cell2 = np.full(X.shape[0], -1, dtype=np.int64)
        stratified = np.zeros(X.shape[0], dtype=bool)
        print(f"[2] 失败池 {len(pool)} < 60, 跳过 rescue", flush=True)

    # ================= 最终统计 =================
    final_pass = (stage == "pass1") | rescued | stratified
    final_noise = ~final_pass
    print(f"\n最终: 通过 {final_pass.sum()} ({final_pass.mean()*100:.1f}%), "
          f"is_noise {final_noise.sum()} ({final_noise.mean()*100:.1f}%) "
          f"[对比: 表达空间基线上皮 unresolved 27.0%]", flush=True)

    rows = pos_all.loc[list(expr_noise.intersection(pos_all.index))].values
    recovered = final_pass[rows]
    print(f"表达空间 {len(rows)} unresolved 上皮: 本流程找回 {recovered.sum()} "
          f"({recovered.mean()*100:.1f}%) [对比: 混合版 21.3%]", flush=True)

    df = pd.DataFrame({"s": score.values, "mc": labels, "pass1": np.isin(labels, np.where(pass1)[0])})
    d1 = df[df["pass1"] & (df["mc"] >= 0)]
    w1 = d1.groupby("mc")["s"].var().mean()
    print(f"分治单轮 MC 的 cnv_score 组内/组间方差比: "
          f"{w1/max(score.var(), 1e-12):.4f} [对比: 混合版 0.343]", flush=True)

    dsub = pd.DataFrame({"ct": ct_sub, "mc": labels})
    dsub = dsub[dsub["mc"] >= 0]
    dsub = dsub[np.isin(dsub["mc"].values, np.where(pass1)[0])]
    pur = dsub.groupby("mc")["ct"].agg(lambda x: x.value_counts().iloc[0] / len(x))
    print(f"分治单轮 MC 的 ct.sub 纯度: median={pur.median():.3f} mean={pur.mean():.3f}",
          flush=True)

    per_s = pd.DataFrame({"sample": list(blocks.keys()), "k_dim": kdims,
                          "gamma": gammas, "K": ks}).sort_values("K", ascending=False)
    print("\n每样本明细 (按 K 降序, 前 8):")
    print(per_s.head(8).to_string(index=False))
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
