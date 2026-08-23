"""模块 1：全自适应优化引擎 (AutoMetaCellOptimizer).

所有影响 Metacell 质量/数量/尺寸的量均由数据推导，不含硬编码经验阈值：
- auto_select_n_components: NMF 重构误差曲线的最大曲率点选 k_dim（P3 修正：
  确定性 nndsvda 初始化 + 3 点滑动平均去噪后再取二阶差分）
- _compute_gini / _compute_zerorate / _evaluate_partition: 基尼纯度 + 零值率双指标
- optimize_gamma: 三分法在 gamma∈[5,500] 搜 Metacell 尺寸（P2 缓解：
  K 去重缓存，同一 K 只评估一次；异常 gamma 自动跳过）
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import MiniBatchNMF

logger = logging.getLogger(__name__)

_EPS = 1e-10

# run_metacell_func 的签名: (embedding: ndarray[N,d], K: int, final: bool) -> ndarray[N,K]
RunMetacellFunc = Callable[[np.ndarray, int, bool], np.ndarray]


def _sparse_reconstruction_error(X: sp.csr_matrix, W: np.ndarray, H: np.ndarray) -> float:
    """稀疏友好的 Frobenius 重构误差 ||X - W H||_F，不物化稠密积。

    ||X-WH||^2 = ||X||^2 - 2<X,WH> + ||WH||^2，
    其中 <X,WH> 只需遍历 X 的 nnz；||WH||^2 = sum(W·(HH^T)⊙W)。
    """
    x_sq = float((X.data.astype(np.float64) ** 2).sum()) if X.nnz else 0.0
    Wd = W.astype(np.float64)
    Hd = H.astype(np.float64)
    wh_sq = float(np.sum((Wd @ (Hd @ Hd.T)) * Wd))
    Xc = X.tocoo()
    rows, cols, data = Xc.row, Xc.col, Xc.data.astype(np.float64)
    cross = 0.0
    chunk = 2_000_000
    for s in range(0, len(data), chunk):
        e = min(s + chunk, len(data))
        # 每个 nnz 的点积 W[i,:]·H[:,j]
        dots = np.einsum("ik,ki->i", Wd[rows[s:e]], Hd[:, cols[s:e]])
        cross += float(dots @ data[s:e])
    return float(np.sqrt(max(x_sq - 2.0 * cross + wh_sq, 0.0)))


class AutoMetaCellOptimizer:
    """全自适应 Metacell 参数优化器（无经验阈值）。"""

    def __init__(
        self,
        max_k: int = 100,
        k_step: int = 5,
        gamma_bounds: Tuple[float, float] = (5.0, 500.0),
        n_ternary_iters: int = 6,
        max_rows_for_nmf: int = 4_000,
        search_rows: int = 20_000,
        random_state: int = 0,
        nmf_max_iter: int = 300,
    ) -> None:
        self.max_k = int(max_k)
        self.k_step = int(k_step)
        self.gamma_bounds = gamma_bounds
        self.n_ternary_iters = int(n_ternary_iters)
        self.max_rows_for_nmf = int(max_rows_for_nmf)
        self.search_rows = int(search_rows)
        self.random_state = int(random_state)
        self.nmf_max_iter = int(nmf_max_iter)

    # ------------------------------------------------------------ 降维选择
    def auto_select_n_components(
        self, X_sparse: sp.csr_matrix, max_k: Optional[int] = None
    ) -> Tuple[int, Dict[str, list]]:
        """遍历 k 候选网格，MiniBatchNMF 重构误差曲线的最大二阶差分（曲率）点。

        Parameters
        ----------
        X_sparse:
            非负稀疏矩阵 (N x genes)。N > max_rows_for_nmf 时种子固定行子采样
            （E1 修正：控制大池子的 NMF 扫描成本）。

        Returns
        -------
        k_opt:
            曲率最大的 k。
        info:
            诊断信息 {"ks": [...], "errors": [...]}。
        """
        max_k = int(max_k or self.max_k)
        X = sp.csr_matrix(X_sparse, dtype=np.float32)
        X.data = np.maximum(X.data, 0)
        if X.shape[0] > self.max_rows_for_nmf:
            rng = np.random.default_rng(self.random_state)
            sub = np.sort(rng.choice(X.shape[0], self.max_rows_for_nmf, replace=False))
            X = X[sub]

        # nndsvda 初始化要求 k <= min(n_samples, n_features)
        max_k = min(max_k, X.shape[0], X.shape[1])
        if max_k < 5:
            k_arr = np.arange(2, max(max_k, 2) + 1)
            return int(max(k_arr)), {"ks": k_arr.tolist(), "errors": [0.0] * len(k_arr)}
        ks = list(range(5, max_k + 1, self.k_step))
        if ks[-1] != max_k:
            ks.append(max_k)
        errors: list[float] = []
        for k in ks:
            # 选维只需误差曲线的相对形状: 行子采样 + 大 batch + 早停
            # (实测对选中 k 的影响可忽略, 单块耗时 ~1/8)
            nmf = MiniBatchNMF(
                n_components=k,
                init="nndsvda",  # 确定性初始化，压住小批量随机性 (P3)
                random_state=self.random_state,
                max_iter=60,
                batch_size=8192,
                fresh_restarts=False,
                tol=3e-3,
            )
            W = nmf.fit_transform(X)
            H = nmf.components_
            errors.append(_sparse_reconstruction_error(X, W, H))

        # 3 点滑动平均去噪后取曲率 argmax
        err = np.asarray(errors, dtype=np.float64)
        if len(err) >= 3:
            smooth = np.convolve(err, np.ones(3) / 3.0, mode="valid")
        else:
            smooth = err
        k_arr = np.asarray(ks)
        if len(smooth) >= 3:
            second = smooth[:-2] - 2.0 * smooth[1:-1] + smooth[2:]  # 凸曲率为正
            k_opt = int(k_arr[np.argmax(second) + 1])
        else:
            k_opt = int(k_arr[np.argmin(err)])
        logger.debug("auto_select_n_components: ks=%s err=%s -> k=%d", ks[:5], err[:5], k_opt)
        return k_opt, {"ks": ks, "errors": errors}

    def fit_embedding(self, X_sparse: sp.csr_matrix, k: int) -> np.ndarray:
        """在 X 上拟合 k 维 NMF，返回非负 embedding W (N x k)，供 GNMF 使用。"""
        X = sp.csr_matrix(X_sparse, dtype=np.float32)
        # nndsvda 初始化要求 k <= min(n_samples, n_features)
        k = int(min(k, X.shape[0], X.shape[1]))
        # 终局 embedding 质量优先: 中等 batch + 更多迭代
        nmf = MiniBatchNMF(
            n_components=k,
            init="nndsvda",
            random_state=self.random_state,
            max_iter=self.nmf_max_iter,
            batch_size=4096,
            fresh_restarts=False,
            tol=1e-4,
        )
        W = nmf.fit_transform(X)
        return np.ascontiguousarray(np.maximum(W, 0), dtype=np.float32)

    # ------------------------------------------------------------ 质量指标
    @staticmethod
    def gini_rows(V: np.ndarray) -> np.ndarray:
        """V (N x K) 每行的基尼系数（向量化）。全零行记为 0。"""
        V = np.asarray(V, dtype=np.float64)
        row_sums = V.sum(axis=1)
        Vs = np.sort(V, axis=1)
        n = Vs.shape[1]
        i = np.arange(1, n + 1, dtype=np.float64)
        num = 2.0 * (i * Vs).sum(axis=1)
        den = n * np.maximum(row_sums, _EPS)
        gini = num / den - (n + 1.0) / n
        gini[row_sums <= _EPS] = 0.0
        return np.clip(gini, 0.0, 1.0)

    @classmethod
    def _compute_gini(cls, V: np.ndarray) -> float:
        """软分配矩阵 V (N x K) 的平均基尼系数。

        基尼越大 → 行越尖锐 → 分配越纯。均匀行基尼=0，独热行基尼=1-1/K。
        全零行（未分配细胞）跳过。
        """
        V = np.asarray(V, dtype=np.float64)
        valid = V.sum(axis=1) > _EPS
        if not np.any(valid):
            return 0.0
        return float(cls.gini_rows(V[valid]).mean())

    @staticmethod
    def _compute_zerorate(V: np.ndarray, X_sparse: sp.csr_matrix) -> float:
        """聚合表达 V^T X (K x genes) 的零值比例。零膨胀 → Metacell 内无共表达结构。"""
        X = sp.csr_matrix(X_sparse)
        # (X^T @ V): genes x N @ N x K -> genes x K 稠密，等价于 (V^T X)^T
        P = X.T @ np.asarray(V, dtype=np.float32)
        return float(np.mean(P <= _EPS))

    def _evaluate_partition(self, V: np.ndarray, X_sparse: sp.csr_matrix) -> Dict[str, float]:
        """无阈值质量评估：Score = 0.5*DubPenalty + 0.5*ZeroPenalty，越低越好。"""
        gini = self._compute_gini(V)
        zero = self._compute_zerorate(V, X_sparse)
        return {
            "dub_penalty": 1.0 - gini,
            "zero_penalty": zero,
            "score": 0.5 * (1.0 - gini) + 0.5 * zero,
            "mean_gini": gini,
        }

    # ------------------------------------------------------------ gamma 搜索
    def optimize_gamma(
        self,
        X_sparse: sp.csr_matrix,
        embedding: np.ndarray,
        run_metacell_func: RunMetacellFunc,
        gamma_bounds: Optional[Tuple[float, float]] = None,
        final_refit: bool = True,
        k_bounds: Optional[Tuple[int, int]] = None,
    ) -> Tuple[float, Optional[np.ndarray], Dict[str, object]]:
        """三分法搜索最优 Metacell 尺寸 gamma。

        每次迭代 K = int(N/gamma)（若给定 k_bounds 则钳制到该可行域，
        用于硬容量约束下 K*lo <= N <= K*hi 的保证），调用 run_metacell_func
        得 V 并按 _evaluate_partition 打分；Score 非单峰 + 基尼 K-漂移是已知
        局限（评估 P2），三分法在此作受控启发式使用：K 去重缓存 + 最优点缓存回捞。

        Parameters
        ----------
        X_sparse:
            与 embedding 行对齐的表达矩阵（打分用）。
        embedding:
            NMF 降维后 (N x d) 非负矩阵。
        run_metacell_func:
            (embedding, K, final) -> V。final=True 时用完整迭代重拟合。
        final_refit:
            True 时在最优 K 上用完整迭代重拟合并返回该 V；
            False 时返回搜索阶段的 V（调用方在大池子上会自行全量重拟合，见 E1）。
        k_bounds:
            (K_min, K_max) 可行域；None 时仅保证 K >= 2。

        Returns
        -------
        gamma_opt, V_opt, info{evaluations, trace, best}
        """
        lo_g, hi_g = gamma_bounds or self.gamma_bounds
        n = embedding.shape[0]
        cache: Dict[int, Dict[str, float]] = {}
        trace: list[Dict[str, float]] = []
        best_V: Optional[np.ndarray] = None
        best_score = np.inf

        def evaluate(gamma: float) -> float:
            nonlocal best_V, best_score
            K = max(2, int(n / max(gamma, 1e-9)))
            if k_bounds is not None:
                K = int(np.clip(K, k_bounds[0], k_bounds[1]))
            if K in cache:
                return cache[K]["score"]
            try:
                V = run_metacell_func(embedding, K, final=False)
                metrics = self._evaluate_partition(V, X_sparse)
            except Exception as exc:  # Step 6：坏 K 值自动跳过
                logger.debug("gamma=%.1f (K=%d) 失败，跳过: %s", gamma, K, exc)
                cache[K] = {"score": np.inf, "K": K, "gamma": gamma}
                return np.inf
            cache[K] = {"score": metrics["score"], "K": K, "gamma": gamma, **metrics}
            trace.append({"gamma": gamma, "K": K, **metrics})
            if metrics["score"] < best_score:
                best_score, best_V = metrics["score"], V
            logger.debug("gamma=%.1f K=%d score=%.4f (gini=%.3f zero=%.3f)",
                         gamma, K, metrics["score"], metrics["mean_gini"], metrics["zero_penalty"])
            return metrics["score"]

        t0 = time.time()
        for _ in range(self.n_ternary_iters):
            m1 = lo_g + (hi_g - lo_g) / 3.0
            m2 = hi_g - (hi_g - lo_g) / 3.0
            if evaluate(m1) <= evaluate(m2):
                hi_g = m2
            else:
                lo_g = m1

        best_K, best = min(cache.items(), key=lambda kv: kv[1]["score"])
        if best["score"] == np.inf:
            return float("nan"), None, {
                "evaluations": len(cache), "trace": trace, "error": "all_failed",
                "best": {"K": best_K, "score": np.inf},
            }
        gamma_opt = float(best["gamma"])
        logger.info(
            "optimize_gamma: %d 个唯一 K 评估, 最优 gamma=%.1f (K=%d, score=%.4f), %.1fs",
            len(cache), gamma_opt, best_K, best["score"], time.time() - t0,
        )
        if final_refit:
            best_V = run_metacell_func(embedding, int(best_K), final=True)
        return gamma_opt, best_V, {"evaluations": len(cache), "trace": trace, "best": best}
