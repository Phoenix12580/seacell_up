"""上皮 CNV 分治分支（固化自实验 A-v2，用户决策 2026-08-18）。

背景：上皮（尤其恶性）在表达空间是连续谱，gini+零值率评分天然吃亏；
CNV 空间中克隆结构是离散的且**病人特异**，因此按样本分治建 Metacell。

输入: infercnvpy 输出的 AnnData（X = 基因级 CNV 矩阵，obs 含样本列与
     cnv_score；产生方式见 scripts/cnv_infer_epithelial.py）。

流程:
  1. 特征筛选: 按列方差取 top n_top_features（默认 3000，用户决策;
     16,338 全维 NMF 是分治阶段的主要成本，top3000 保留克隆信号）
  2. 非负平移（NMF 要求，保持相对结构）
  3. 按样本分治: 每样本独立 auto_select k_dim + gamma 搜索 + GNMF,
     容量严格 [20,35]（<min_block_cells 的微样本直接进失败池）
  4. 全局 mean 线单轮筛选（与主流水线同规则）
  5. 失败池跨样本 rescue 1 轮（大池子走 _process_block 的子采样搜索路径）
  6. rescue 失败者在自身分布内 median 分层放行（纯评分操作, 不改 MC 尺寸）
  7. 剩余细胞 is_noise

参考实现指标（Zhao 222K 上皮 90,626 细胞）: 单轮通过 48.5%, 上皮
unresolved 13.4%（表达空间 27.0%）, ct.sub 纯度 median 1.000,
cnv_score 组内/组间方差比 0.19。
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed, parallel_backend

from .pipeline import PipelineConfig, _process_block, _stable_seed

logger = logging.getLogger(__name__)


@dataclass
class CNVBranchConfig:
    """上皮 CNV 分支配置。容量界沿用主流水线的 [20,35]。"""

    n_top_features: int = 3000   # 方差 top-N 特征（用户决策 2026-08-18）
    sample_key: str = "sampleID"
    min_block_cells: int = 40    # < 2*capacity_lo 无法建 2 个 MC, 整块进失败池
    n_jobs: int = -1
    random_state: int = 0
    # 透传给 _process_block 的主流水线配置（容量等）
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


def column_variance(X: sp.csr_matrix) -> np.ndarray:
    """稀疏列方差: Var = E[x^2] - E[x]^2。"""
    Xc = X.tocoo()
    n = X.shape[0]
    mean = np.asarray(X.mean(axis=0)).ravel()
    sq = np.bincount(Xc.col, weights=(Xc.data.astype(np.float64) ** 2),
                     minlength=X.shape[1]) / n
    # E[x^2] = nnz 部分 + 0 部分(=0)，bincount/n 已是全部
    return np.maximum(sq - mean**2, 0.0)


def run_cnv_branch(
    cnv_adata: ad.AnnData,
    config: Optional[CNVBranchConfig] = None,
) -> Dict[str, object]:
    """执行上皮 CNV 分治分支。

    Parameters
    ----------
    cnv_adata:
        infercnvpy 输出（仅目标细胞，如上皮；obs 含 sample_key 列）。
        X 为基因级 CNV 矩阵（可含负值）。

    Returns
    -------
    dict:
        labels (N,) int64: Metacell 编号, -1 = is_noise;
        stage (N,) object: "pass1" / "rescue" / "stratified" / "noise";
        n_metacells, noise_rate, per_sample (每样本 k_dim/gamma/K),
        mc_scores (K,) float。
    """
    cfg = config or CNVBranchConfig()
    pcfg = cfg.pipeline
    X = sp.csr_matrix(cnv_adata.X, dtype=np.float32)
    n = X.shape[0]
    sample_ids = cnv_adata.obs[cfg.sample_key].astype(str).values

    # ---- 1. 方差 top-N 特征
    if X.shape[1] > cfg.n_top_features:
        var = column_variance(X)
        feat = np.sort(np.argsort(-var)[: cfg.n_top_features])
        X = X[:, feat].tocsr()
        logger.info("CNV 分支: 方差筛选 %d -> %d 特征", cnv_adata.n_vars, len(feat))

    # ---- 2. 非负平移（NMF 要求; 平移不改变方差与相对结构）
    shift = abs(float(X.min())) if X.nnz and X.min() < 0 else 0.0
    if shift:
        X.data = X.data + shift
        X.eliminate_zeros()

    # ---- 3. 按样本分治
    cfg_dict = asdict(pcfg)
    blocks: Dict[str, np.ndarray] = {}
    small: list[np.ndarray] = []
    for sid in pd.unique(sample_ids):
        idx = np.where(sample_ids == sid)[0]
        if len(idx) >= cfg.min_block_cells:
            blocks[str(sid)] = idx
        else:
            small.append(idx)
    logger.info("CNV 分支: %d 个样本块, %d 个微样本直接进失败池",
                len(blocks), len(small))
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=cfg.n_jobs)(
            delayed(_process_block)(
                X[idx].tocsr(), cfg_dict, str(sid),
                _stable_seed(cfg.random_state, f"cnv|{sid}"),
            )
            for sid, idx in blocks.items()
        )

    labels = np.full(n, -1, dtype=np.int64)
    stage = np.full(n, "fail", dtype=object)
    mc_scores: list[float] = []
    per_sample = []
    for r in results:
        sid = r["block_id"]
        per_sample.append({"sample": sid, "k_dim": r["k_dim"],
                           "gamma": r["gamma"], "K": r["n_metacells"]})
        idx = blocks[sid]
        for m in range(r["n_metacells"]):
            members = idx[r["labels"] == m]
            labels[members] = len(mc_scores)
            mc_scores.append(float(r["mc_scores"][m]))
    scores = np.array(mc_scores)

    # ---- 4. 全局 mean 线
    thr1 = float(scores.mean()) if len(scores) else 0.0
    pass1 = scores <= thr1
    stage[np.isin(labels, np.where(pass1)[0])] = "pass1"
    fail_mc = set(np.where(~pass1)[0].tolist())
    fail_cells = [np.where(labels == m)[0] for m in fail_mc]
    fail_cells.extend(small)
    pool = (np.concatenate(fail_cells) if fail_cells
            else np.array([], dtype=int))
    logger.info("CNV 分支单轮: K=%d, mean 线=%.4f, 通过 %d/%d (%.1f%%), 失败池 %d",
                len(scores), thr1, pass1.sum(), len(scores),
                100 * pass1.mean() if len(scores) else 0.0, len(pool))

    # ---- 5. 失败池 rescue 1 轮
    if len(pool) >= pcfg.min_pool_cells:
        r2 = _process_block(X[pool].tocsr(), cfg_dict, "cnv_rescue",
                            _stable_seed(cfg.random_state, "cnv_rescue"))
        pool_labels_local = np.full(n, -1, dtype=np.int64)
        pool_scores: list[float] = []
        for m in range(r2["n_metacells"]):
            members = pool[r2["labels"] == m]
            pool_labels_local[members] = m
            pool_scores.append(float(r2["mc_scores"][m]))
        ps = np.array(pool_scores)
        thr2 = float(ps.mean()) if len(ps) else 0.0
        pass2 = ps <= thr2
        stage[np.isin(pool_labels_local, np.where(pass2)[0])] = "rescue"

        # ---- 6. rescue 失败者内 median 分层放行
        fail2 = ps[~pass2]
        if len(fail2):
            thr3 = float(np.median(fail2))
            pass3 = ps <= thr3
            stage[(stage == "fail")
                  & np.isin(pool_labels_local, np.where(pass3)[0])] = "stratified"
            logger.info("CNV rescue: 池 %d -> K=%d, mean 线=%.4f 通过 %.1f%%; "
                        "分层(失败者内 median=%.4f) 累计放行 %.1f%%",
                        len(pool), r2["n_metacells"], thr2, 100 * pass2.mean(),
                        thr3, 100 * pass3.mean())
    else:
        logger.info("CNV 分支: 失败池 %d < %d, 跳过 rescue", len(pool),
                    pcfg.min_pool_cells)

    # ---- 7. 剩余 = noise
    stage[stage == "fail"] = "noise"
    is_noise = stage == "noise"
    n_mc = int((labels >= 0).sum() and len(np.unique(labels[labels >= 0])))
    logger.info("CNV 分支完成: 通过 %.1f%%, is_noise %d/%d (%.1f%%)",
                100 * (1 - is_noise.mean()), is_noise.sum(), n,
                100 * is_noise.mean())
    return {
        "labels": labels,
        "stage": stage,
        "is_noise": is_noise,
        "n_metacells": n_mc,
        "noise_rate": float(is_noise.mean()),
        "mc_scores": scores,
        "per_sample": per_sample,
        "n_pool": int(len(pool)),
    }
