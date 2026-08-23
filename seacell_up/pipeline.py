"""模块 3：主调度流水线 (IterativeMetaCellPipeline).

三层漏斗：分治构建 -> 跨样本混合迭代清洗 -> 终局剔除。

第一阶段：按 sample_id 拆分，各样本并行执行
    自适应降维 -> 三分法优化 gamma -> GNMF 构建 -> 硬容量约束 -> 逐 Metacell 打分
    Score 低于全局平均分的 Metacell 通过（默认；pass_rule 可调，评估 P5）。
第二阶段：回收细胞合并，最多 max_rounds(3) 轮同流程清洗。
第三阶段：第 3 轮仍未通过的细胞标记 is_noise=True 剔除。

复杂度说明（E1 修正）：gamma 搜索阶段的 GNMF 与软打分在超过 search_rows 的
池子上用种子固定行子采样执行；gamma 定格后在全量数据做一次完整重拟合。
逐 Metacell 最终打分用硬成员伪体聚合（O(nnz)）而非软聚合（O(nnz*K)）。
"""
from __future__ import annotations

import logging
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed, parallel_backend

from .gnmf import FastGNMF, enforce_capacity
from .optimizer import AutoMetaCellOptimizer
from .preprocessing import (
    aggregate_metacells,
    looks_like_raw_counts,
    normalize_log,
    select_hvgs,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """流水线配置。所有影响 Metacell 数量/尺寸/质量的量由数据自适应推导；
    此处仅含资源/数值稳定/文档化的配置项（见 EVALUATION.md §三）。"""

    n_top_genes: int = 2048
    target_sum: float = 1e4
    # HVG 选择模式（用户可选，2026-08-23）:
    # - "global": 全部样本合并选一次 HVG，各块共用（跨块特征统一，
    #   但可能漏掉样本特异的高变基因，如病人特异克隆基因）
    # - "per_sample": 每个块（样本/回收池）内部独立选 HVG，即第一轮
    #   各样本自己的 HVG、回收轮用剩余细胞合并选 HVG。忠实于每块的
    #   变异结构。MC 伪体始终在全基因上聚合，下游跨样本可比性不受
    #   构建期 HVG 选择影响，两种模式的伪体可直接对比。
    hvg_mode: str = "global"
    # gamma（Metacell 尺寸）搜索范围与三分法迭代数
    gamma_bounds: Tuple[float, float] = (5.0, 500.0)
    n_ternary_iters: int = 6
    # NMF 选维
    max_k: int = 100
    # GNMF
    n_neighbors: int = 10
    alpha: float = 0.5
    gnmf_search_iter: int = 60   # gamma 搜索阶段迭代数（E1）
    gnmf_final_iter: int = 200   # 终局重拟合迭代数（tol 提前停止）
    # 硬容量约束（用户决策 2026-08-17）：每个 Metacell 严格 20-35 个细胞。
    # 可行性 K*lo <= N <= K*hi 由 K 钳制到 [ceil(N/hi), floor(N/lo)] 保证，
    # gamma 搜索的可行域随之自动收敛到 [lo, hi]。
    target_size: Optional[int] = None
    capacity_lo: int = 20
    capacity_hi: int = 35
    # 迭代清洗
    max_rounds: int = 3
    pass_rule: str = "mean"      # "mean" | "median" | "quantile"
    pass_quantile: float = 0.75  # pass_rule="quantile" 时：最差 (1-q) 比例被回收
    min_block_cells: int = 40    # < 2*capacity_lo 的样本无法建 2 个 Metacell，整块进回收池（E2）
    min_pool_cells: int = 60     # 回收池低于该数停止迭代
    # 资源
    max_metacells: int = 5000    # 单块 Metacell 数上限（内存护栏，评估 P2/E1）
    search_rows: int = 5_000     # gamma 搜索阶段最大行数（超出则种子固定子采样；γ 是尺寸参数，可迁移）
    max_rows_for_nmf: int = 4_000
    nmf_max_iter: int = 300
    n_jobs: int = -1
    random_state: int = 0


def _stable_seed(base: int, key: str) -> int:
    return (base + zlib.crc32(key.encode())) % (2**31)


def _pass_threshold(scores: np.ndarray, cfg: PipelineConfig) -> float:
    """数据驱动的通过线（无经验阈值）：默认全局平均分（规范原文）。"""
    if cfg.pass_rule == "mean":
        return float(scores.mean())
    if cfg.pass_rule == "median":
        return float(np.median(scores))
    return float(np.quantile(scores, cfg.pass_quantile))


def score_metacells(V: np.ndarray, X: sp.csr_matrix, labels: np.ndarray) -> np.ndarray:
    """逐 Metacell 打分：Score_m = 0.5*(1-纯度_m) + 0.5*(零值率_m)。

    纯度_m = 成员细胞软分配行基尼的均值；
    零值率_m = 硬成员伪体 (X 按标签求和) 第 m 行的零值比例，O(nnz) 计算。
    """
    K = int(labels.max()) + 1
    gini = AutoMetaCellOptimizer.gini_rows(V)
    purity = np.bincount(labels, weights=gini, minlength=K) / np.maximum(
        np.bincount(labels, minlength=K), 1
    )
    agg = aggregate_metacells(X, labels, K)  # 稀疏 K x genes
    zero_rate = 1.0 - agg.getnnz(axis=1) / X.shape[1]
    return 0.5 * (1.0 - purity) + 0.5 * zero_rate


def _process_block(
    X_block: sp.csr_matrix,
    cfg_dict: dict,
    block_id: str,
    seed: int,
) -> Dict[str, object]:
    """处理一个细胞块（一个样本，或一轮回收池）。可在 worker 进程内执行。"""
    # worker 进程不继承主进程 logging 配置，这里补上使块级日志可回传（Step 6）
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s: %(message)s",
                            datefmt="%H:%M:%S", stream=sys.stdout)
    cfg = PipelineConfig(**cfg_dict)
    # 显式可写副本：joblib 多进程下大数组以只读 memmap 传入，原地写会报错
    X = sp.csr_matrix(X_block, dtype=np.float32).copy()
    X.sort_indices()
    if cfg.hvg_mode == "per_sample" and X.shape[1] > cfg.n_top_genes:
        # 每块独立选 HVG（样本块=本样本 HVG; 回收池块=剩余细胞合并 HVG）。
        # 伪体仍在全基因聚合, 不影响下游可比性。
        X = X[:, select_hvgs(X, cfg.n_top_genes)].tocsr()
    n = X.shape[0]
    t0 = time.time()

    opt = AutoMetaCellOptimizer(
        max_k=cfg.max_k,
        gamma_bounds=cfg.gamma_bounds,
        n_ternary_iters=cfg.n_ternary_iters,
        max_rows_for_nmf=cfg.max_rows_for_nmf,
        random_state=seed,
        nmf_max_iter=cfg.nmf_max_iter,
    )

    # 1) 自适应降维（NMF 重构误差曲率选 k_dim）
    k_dim, _ = opt.auto_select_n_components(X)
    emb = opt.fit_embedding(X, k_dim)

    # 2) gamma 搜索子样本（大池子搜索降规模，E1；小数据子样本=全量）
    if n > cfg.search_rows:
        sel = np.sort(
            np.random.default_rng(seed).choice(n, cfg.search_rows, replace=False)
        )
    else:
        sel = None
    X_s = X if sel is None else X[sel].tocsr()
    emb_s = emb if sel is None else emb[sel]

    # 硬容量 [lo, hi] 的 K 可行域（子样本行数下）: K*lo <= N_s <= K*hi
    def _k_bounds(n_rows: int) -> Tuple[int, int]:
        return (
            max(2, int(np.ceil(n_rows / cfg.capacity_hi))),
            int(np.clip(n_rows // cfg.capacity_lo, 2, cfg.max_metacells)),
        )

    k_bounds_s = _k_bounds(emb_s.shape[0])

    def runner(embedding: np.ndarray, K: int, final: bool) -> np.ndarray:
        K = int(np.clip(K, k_bounds_s[0], min(k_bounds_s[1], embedding.shape[0] - 1)))
        if K < 2:
            raise ValueError(f"K={K} 过小 (N={embedding.shape[0]})")
        gnmf = FastGNMF(
            n_metacells=K,
            n_neighbors=min(cfg.n_neighbors, embedding.shape[0] - 1),
            alpha=cfg.alpha,
            max_iter=cfg.gnmf_final_iter if final else cfg.gnmf_search_iter,
            random_state=seed,
            n_jobs=1,
        )
        return gnmf.fit_transform(embedding)

    gamma, _, gamma_info = opt.optimize_gamma(
        X_s, emb_s, runner, final_refit=False, k_bounds=k_bounds_s
    )
    # gamma 定格 -> 全量行数下的 K（同样钳制到容量可行域）
    kb = _k_bounds(n)
    best_K = int(np.clip(int(n / max(gamma, 1e-9)), kb[0], min(kb[1], n - 1)))

    # 3) gamma 定格后在全量数据上完整重拟合 GNMF。
    #    大池子（N*K > 1e8）时迭代预算自适应下调：scipy 稀疏乘 A@W 是大 K 下
    #    每迭代的主导成本且单线程，60 迭代已足以稳定 argmax 分配（资源护栏）。
    final_iter = cfg.gnmf_final_iter
    if n * best_K > 1e8:
        final_iter = max(60, int(final_iter * 1e8 / (n * best_K)))
    V = FastGNMF(
        n_metacells=best_K,
        n_neighbors=min(cfg.n_neighbors, n - 1),
        alpha=cfg.alpha,
        max_iter=final_iter,
        random_state=seed,
        n_jobs=1,
    ).fit_transform(emb)

    # 4) 硬容量约束 -> 逐 Metacell 打分
    labels, counts = enforce_capacity(
        V, target_size=cfg.target_size, lo=cfg.capacity_lo, hi=cfg.capacity_hi
    )
    mc_scores = score_metacells(V, X, labels)

    logger.info(
        "[%s] N=%d k_dim=%d gamma=%.1f K=%d | Metacell score mean=%.4f "
        "| 容量范围 [%d, %d] | gamma 评估 %d 次 | %.1fs",
        block_id, n, k_dim, gamma, len(counts), float(mc_scores.mean()),
        int(counts.min()), int(counts.max()),
        int(gamma_info.get("evaluations", -1)), time.time() - t0,
    )
    return {
        "block_id": block_id,
        "n": n,
        "k_dim": int(k_dim),
        "gamma": float(gamma),
        "n_metacells": int(len(counts)),
        "labels": labels,
        "counts": counts,
        "mc_scores": mc_scores,
        "score_mean": float(mc_scores.mean()),
        "n_gamma_evals": int(gamma_info.get("evaluations", -1)),
        "seconds": time.time() - t0,
    }


class IterativeMetaCellPipeline:
    """分治 + 迭代清洗的 Metacell 构建流水线。

    Parameters
    ----------
    sample_key:
        obs 中的样本列名；缺失时整体视为单样本（告警）。
    input_is_raw:
        None=自动启发式判断；True/False 强制声明 X 是否原始计数。
    """

    def __init__(
        self,
        sample_key: str = "sample_id",
        input_is_raw: Optional[bool] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.sample_key = sample_key
        self.input_is_raw = input_is_raw
        self.cfg = config or PipelineConfig()

    # ------------------------------------------------------------------ utils
    def _preprocess(
        self, adata: ad.AnnData
    ) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
        """返回 (X_norm_hvg, X_for_pseudobulk, hvg_gene_idx)。"""
        X = adata.X
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        X = sp.csr_matrix(X, dtype=np.float32)

        is_raw = self.input_is_raw
        if is_raw is None:
            is_raw = looks_like_raw_counts(X)
            logger.info("输入 X 自动判断为%s", "原始计数" if is_raw else "已归一化数据")
        if is_raw:
            X_raw = X.copy()
            X_norm = normalize_log(X, self.cfg.target_sum)
        else:
            X_raw = None  # 已归一化：伪体直接聚合归一化矩阵
            X_norm = X

        t0 = time.time()
        if self.cfg.hvg_mode == "per_sample":
            # 块内独立选 HVG（_process_block 内执行），此处返回全基因归一化矩阵
            logger.info("HVG 模式 per_sample: 各块内部独立选 %d 基因", self.cfg.n_top_genes)
            return X_norm.tocsr(), (X_raw if X_raw is not None else X_norm), np.arange(X.shape[1])
        hvg = select_hvgs(X_norm, n_top=self.cfg.n_top_genes)
        logger.info(
            "全局 HVG: %d/%d 基因, %.1fs（全阶段共用基因空间）",
            len(hvg), X.shape[1], time.time() - t0,
        )
        return X_norm[:, hvg].tocsr(), (X_raw if X_raw is not None else X_norm), hvg

    # -------------------------------------------------------------------- run
    def run(
        self, adata: ad.AnnData
    ) -> Tuple[ad.AnnData, ad.AnnData, Dict[str, object]]:
        """执行完整流水线。

        Returns
        -------
        adata_out:
            输入 AnnData 副本 + obs 列:
            metacell_id / metacell_stage / mc_score / is_noise。
        metacell_adata:
            Metacell 伪体 (K x genes) + obs:
            n_cells / stage / gamma / k_dim / score / dominant_sample / sample_purity。
        report:
            结构化运行报告（各块 gamma/K/得分、池规模、时间、噪音率）。
        """
        cfg = self.cfg
        t_start = time.time()
        X_hvg, X_pb, hvg = self._preprocess(adata)
        n_cells = adata.n_obs

        # ---- 样本拆分
        if self.sample_key in adata.obs.columns:
            sample_ids = adata.obs[self.sample_key].astype(str).values
        else:
            logger.warning("obs 缺少 '%s'，整体作为单样本处理", self.sample_key)
            sample_ids = np.full(n_cells, "all", dtype=object)
        blocks: Dict[str, np.ndarray] = {}
        leftover_mask = np.zeros(n_cells, dtype=bool)
        for sid in pd.unique(sample_ids):
            idx = np.where(sample_ids == sid)[0]
            if len(idx) < cfg.min_block_cells:
                logger.info("样本 %s 仅 %d 细胞，直接进回收池", sid, len(idx))
                leftover_mask[idx] = True
            else:
                blocks[str(sid)] = idx
        report: Dict[str, object] = {"blocks": {}, "rounds": [], "summary": {}}

        # ============ 第一阶段：各样本并行 ============
        cfg_dict = asdict(cfg)
        n_jobs = cfg.n_jobs
        t_stage1 = time.time()
        with parallel_backend("loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=n_jobs)(
                delayed(_process_block)(
                    X_hvg[idx].tocsr(),
                    cfg_dict,
                    str(sid),
                    _stable_seed(cfg.random_state, f"S1|{sid}"),
                )
                for sid, idx in blocks.items()
            )
        stage1_seconds = time.time() - t_stage1

        # ---- 全局通过线（规范：Score 低于全局平均分的 Metacell 通过）
        all_scores = np.concatenate([r["mc_scores"] for r in results])
        threshold = _pass_threshold(all_scores, cfg)
        logger.info(
            "第一阶段: %d 样本, %d Metacells, 全局通过线 score<=%.4f, 并行 %.1fs",
            len(results), len(all_scores), threshold, stage1_seconds,
        )

        mc_id = np.full(n_cells, "noise", dtype=object)
        mc_stage = np.full(n_cells, "noise", dtype=object)
        mc_score = np.full(n_cells, np.nan, dtype=np.float64)
        mc_names: List[str] = []           # 创建顺序 = metacell_adata 的行序
        mc_meta: List[Dict[str, object]] = []  # 每个已接受 Metacell 的 {stage, score, gamma}

        def _accept(members: np.ndarray, stage: str, score: float, gamma: float) -> None:
            name = f"MC{len(mc_names):06d}"
            mc_names.append(name)
            mc_meta.append({"stage": stage, "score": score, "gamma": gamma})
            mc_id[members] = name
            mc_stage[members] = stage
            mc_score[members] = score

        for r in results:
            sid = r["block_id"]
            cell_idx = blocks[sid]
            report["blocks"][sid] = {
                k: v for k, v in r.items()
                if k not in ("labels", "mc_scores", "counts")
            }
            report["blocks"][sid]["counts_minmax"] = (
                int(r["counts"].min()), int(r["counts"].max()),
            )
            for m in range(r["n_metacells"]):
                members = cell_idx[r["labels"] == m]
                if r["mc_scores"][m] <= threshold:
                    _accept(members, "stage1", float(r["mc_scores"][m]), float(r["gamma"]))
                else:
                    leftover_mask[members] = True
        stage1_gamma = [r["gamma"] for r in results]
        del results

        # ============ 第二/三阶段：跨样本混合迭代 ============
        for rnd in range(1, cfg.max_rounds + 1):
            pool_idx = np.where(leftover_mask)[0]
            if len(pool_idx) < cfg.min_pool_cells:
                logger.info(
                    "第 %d 轮: 回收池 %d 细胞不足 (%d)，停止迭代",
                    rnd, len(pool_idx), cfg.min_pool_cells,
                )
                break
            logger.info("第 %d 轮: 回收池 %d 细胞", rnd, len(pool_idx))
            t_r = time.time()
            r = _process_block(
                X_hvg[pool_idx].tocsr(),
                cfg_dict,
                f"rescue{rnd}",
                _stable_seed(cfg.random_state, f"R{rnd}"),
            )
            thr = _pass_threshold(r["mc_scores"], cfg)
            report["rounds"].append(
                {"round": rnd, "n_pool": int(len(pool_idx)), "gamma": r["gamma"],
                 "k_dim": r["k_dim"], "n_metacells": r["n_metacells"],
                 "threshold": thr, "seconds": time.time() - t_r}
            )
            logger.info(
                "第 %d 轮: gamma=%.1f K=%d 通过线=%.4f, %.1fs",
                rnd, r["gamma"], r["n_metacells"], thr, time.time() - t_r,
            )
            n_pass = 0
            for m in range(r["n_metacells"]):
                members = pool_idx[r["labels"] == m]
                if r["mc_scores"][m] <= thr:
                    _accept(members, f"rescue{rnd}", float(r["mc_scores"][m]), float(r["gamma"]))
                    leftover_mask[members] = False
                    n_pass += 1
            logger.info("第 %d 轮: %d/%d Metacells 通过", rnd, n_pass, r["n_metacells"])
            if n_pass == r["n_metacells"]:
                break

        # ---- 终局剔除
        noise_idx = np.where(leftover_mask)[0]
        is_noise = np.zeros(n_cells, dtype=bool)
        is_noise[noise_idx] = True
        logger.info(
            "噪音剔除: %d/%d 细胞 (%.2f%%)",
            len(noise_idx), n_cells, 100 * len(noise_idx) / max(n_cells, 1),
        )

        # ---- 输出 1：细胞级 AnnData
        adata_out = adata.copy()
        adata_out.obs["metacell_id"] = pd.Categorical(mc_id)
        adata_out.obs["metacell_stage"] = pd.Categorical(mc_stage)
        adata_out.obs["mc_score"] = mc_score
        adata_out.obs["is_noise"] = is_noise

        # ---- 输出 2：Metacell 伪体 AnnData（按 mc_names 顺序）
        passed_mask = ~is_noise
        passed_idx = np.where(passed_mask)[0]
        name_to_int = {nm: i for i, nm in enumerate(mc_names)}
        pb_labels = np.array(
            [name_to_int[x] for x in np.asarray(mc_id)[passed_mask]], dtype=np.int64
        )
        agg = aggregate_metacells(X_pb[passed_mask].tocsr(), pb_labels, len(mc_names))
        agg = agg.astype(np.float32)

        sid_pass = np.asarray(sample_ids, dtype=object)[passed_mask]
        rows = []
        for i, (nm, meta) in enumerate(zip(mc_names, mc_meta)):
            members = np.where(pb_labels == i)[0]
            vc = pd.Series(sid_pass[members]).value_counts()
            rows.append(
                {"n_cells": int(len(members)),
                 "stage": meta["stage"],
                 "score": meta["score"],
                 "gamma": meta["gamma"],
                 "dominant_sample": str(vc.index[0]),
                 "sample_purity": float(vc.iloc[0] / len(members))}
            )
        mc_obs = pd.DataFrame(rows, index=pd.Index(mc_names, name="metacell_id"))
        metacell_adata = ad.AnnData(X=agg, obs=mc_obs)
        metacell_adata.var_names = adata.var_names

        report["summary"] = {
            "n_cells": int(n_cells),
            "n_samples": len(blocks),
            "n_metacells": len(mc_names),
            "n_noise": int(len(noise_idx)),
            "noise_rate": float(len(noise_idx) / max(n_cells, 1)),
            "stage1_gamma_median": float(np.median(stage1_gamma)) if stage1_gamma else None,
            "stage1_seconds": stage1_seconds,
            "total_seconds": time.time() - t_start,
        }
        logger.info(
            "完成: %d Metacells, 噪音率 %.2f%%, 总耗时 %.1fs",
            len(mc_names), 100 * report["summary"]["noise_rate"],
            report["summary"]["total_seconds"],
        )
        self.report_ = report
        return adata_out, metacell_adata, report
