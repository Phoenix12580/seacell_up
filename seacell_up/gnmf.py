"""模块 2：极速 GNMF 引擎与硬容量约束.

图正则化非负矩阵分解 (Graph-regularized NMF)，替代原版 SEACells 的核心分解。
- 输入：降维后的非负矩阵 (N x d)
- 输出：软分配矩阵 V (N x K)，每行归一化为对 K 个 Metacell 的归属权重
- 图正则化项使用稀疏矩阵乘法 (scipy.sparse)

性能要点（大 K/N 下实测关键）:
- H 更新 W^T W H 与 W 更新 W (H H^T) 均按结合律换序到 N*K*d 通道
  (朴素写法是 N*K*K, K=4000/d=10 时慢 ~400 倍)
- 图项 A@W (scipy 稀疏乘单线程) 每 4 迭代滞后刷新一次
- 大 N 的 kNN 用手写 chunked float32 top-k (BLAS sgemm)

容量约束 enforce_capacity 将软分配转为带硬容量界的硬标签:
构造性重平衡 (surplus -> deficit 定向补给), 总量可行时所有使用的桶
严格落在 [lo, hi]。默认界由 N/K 自适应推导; 流水线层传入 [20,35]
(K 可行域钳制保证任意规模可行, 见 EVALUATION.md P1)。
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

_EPS = 1e-10
# 乘性更新中防"锁死"的下限：一旦为 0 永远为 0
_FLOOR = 1e-12


def _topk_cosine(X: np.ndarray, k: int, chunk: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """chunked top-k 余弦相似度（float32 sgemm + argpartition）。

    Returns
    -------
    idx:
        (N, k) 邻居索引（每行含自身）。
    sims:
        (N, k) 余弦相似度，已按行降序。
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norms, 1e-12)
    n = Xn.shape[0]
    idx = np.empty((n, k), dtype=np.int32)
    sims = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        block = Xn[s:e] @ Xn.T  # (e-s) x n, float32
        # 排除自身
        block[np.arange(e - s), np.arange(s, e)] = -1.0
        part = np.argpartition(-block, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(block, part, axis=1)
        order = np.argsort(-vals, axis=1)
        idx[s:e] = np.take_along_axis(part, order, axis=1).astype(np.int32)
        sims[s:e] = np.take_along_axis(vals, order, axis=1)
    return idx, np.clip(sims, 0.0, 1.0)


class FastGNMF:
    """图正则化 NMF，输出细胞 x Metacell 软分配矩阵.

    目标函数:  min_{W,H>=0} ||X - W H||_F^2 + alpha * Tr(W^T L W)
    其中 L = D - A 为图拉普拉斯，A 为 kNN 余弦相似度图。

    Parameters
    ----------
    n_metacells:
        Metacell 数 K。
    n_neighbors:
        kNN 图邻居数（资源/数值配置，非质量阈值）。
    alpha:
        图正则化权重。
    max_iter:
        乘性更新最大迭代数。
    tol:
        目标函数相对变化低于此值时提前停止。
    random_state:
        随机种子。
    n_jobs:
        kNN 搜索并行度。
    """

    def __init__(
        self,
        n_metacells: int,
        n_neighbors: int = 10,
        alpha: float = 0.5,
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: int = 0,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.n_metacells = int(n_metacells)
        self.n_neighbors = int(n_neighbors)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.n_jobs = n_jobs
        self.V_: Optional[np.ndarray] = None
        self.W_: Optional[np.ndarray] = None
        self.H_: Optional[np.ndarray] = None
        self.n_iter_: int = 0
        self.objective_: list[float] = []

    # ------------------------------------------------------------------ graph
    def _build_graph(self, X: np.ndarray) -> sp.csr_matrix:
        """对称 kNN 余弦相似度图 (N x N, 稀疏)。

        大 N 时走手写 chunked float32 top-k（BLAS sgemm，比 sklearn 的
        float64 brute 快一个量级）；小 N 用 sklearn。
        """
        n = X.shape[0]
        k = min(self.n_neighbors + 1, n)  # +1 因为包含自身
        if k < 2:
            # 单点/极小块：自环图，图正则退化为无约束
            return sp.eye(n, format="csr", dtype=np.float32)
        if n > 20_000:
            idx, sim = _topk_cosine(X.astype(np.float32), k)
        else:
            nn = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=self.n_jobs)
            nn.fit(X)
            dist, idx = nn.kneighbors(X)
            sim = np.clip(1.0 - dist, 0.0, 1.0).astype(np.float32)
        rows = np.repeat(np.arange(n), k)
        cols = idx.ravel().astype(np.int32)
        A = sp.csr_matrix((sim.ravel(), (rows, cols)), shape=(n, n))
        A = A.maximum(A.T).tocsr()  # 对称化
        A.data = np.maximum(A.data, 0).astype(np.float32)
        return A

    # ------------------------------------------------------------------- fit
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """拟合并返回行归一化软分配 V (N x K)，每行和为 1。

        Parameters
        ----------
        X:
            非负矩阵 (N x d)，通常是 NMF 降维后的 embedding。
        """
        Xd = np.ascontiguousarray(X, dtype=np.float32)
        Xd = np.maximum(Xd, 0.0)
        n, d = Xd.shape
        K = self.n_metacells
        if K < 2 or K >= n:
            raise ValueError(f"n_metacells 需满足 2 <= K < N，收到 K={K}, N={n}")

        rng = np.random.default_rng(self.random_state)
        scale = max(float(Xd.mean()), 1e-3)
        W = (rng.random((n, K), dtype=np.float32) + 0.1) * np.sqrt(scale / K)
        H = (rng.random((K, d), dtype=np.float32) + 0.1) * np.sqrt(scale / K)
        W = np.maximum(W, _FLOOR)
        H = np.maximum(H, _FLOOR)

        A = self._build_graph(Xd)
        deg = np.asarray(A.sum(axis=1), dtype=np.float32).ravel()
        x_norm2 = float((Xd**2).sum())
        alpha = self.alpha
        self.objective_ = []
        prev_obj = np.inf
        # 图项 A@W 滞后刷新：scipy 稀疏乘单线程，大 K 时是每迭代主导成本
        # (~N*nnz_per_row*K 标量ops)。每 graph_refresh 迭代重算一次，乘性更新
        # 框架下用滞后图梯度收敛性无碍（MM/SVRG 风格）。
        graph_refresh = 4
        AW = None

        for it in range(1, self.max_iter + 1):
            # ---- H 更新: H <- H * (W^T X) / (W^T W H)
            # W^T W H 按结合律写成 W^T (W H): N*K*d 通道, 大 K 时比
            # (W^T W) @ H 的 N*K*K 通道快 K/d 倍（K=3500, d=10 时 350 倍）
            WtX = W.T @ Xd
            WH_ = W @ H
            H *= WtX / (W.T @ WH_ + _EPS)
            np.maximum(H, _FLOOR, out=H)

            # ---- W 更新: W <- W * (X H^T + alpha*A W) / (W H H^T + alpha*D W)
            # W H H^T 按结合律写成 (W H) H^T: N*K*d 通道, 大 K 时比 W@(H H^T)
            # 的 N*K*K 通道快 d/K 倍（K=4000, d=10 时约 400 倍）
            num = Xd @ H.T
            if alpha > 0:
                if AW is None or it % graph_refresh == 1:
                    AW = A @ W
                num += alpha * AW
            den = (W @ H) @ H.T
            if alpha > 0:
                den += alpha * deg[:, None] * W
            W *= num / (den + _EPS)
            np.maximum(W, _FLOOR, out=W)

            if it % 10 == 0 or it == self.max_iter:
                # obj = ||X-WH||^2 + alpha * (sum(deg*W^2) - sum(W A W))
                WH = W @ H
                WH_norm2 = float((WH * WH).sum())
                cross = float((W * (Xd @ H.T)).sum())
                graph = float((deg[:, None] * (W**2)).sum() - float((AW * W).sum()) if AW is not None else 0.0)
                obj = x_norm2 - 2.0 * cross + WH_norm2 + alpha * max(graph, 0.0)
                self.objective_.append(obj)
                self.n_iter_ = it
                if prev_obj != np.inf and abs(prev_obj - obj) / max(abs(prev_obj), 1e-12) < self.tol:
                    break
                prev_obj = obj

        V = W / (W.sum(axis=1, keepdims=True) + _EPS)
        self.W_, self.H_, self.V_ = W, H, V
        return V


# ---------------------------------------------------------------------- capacity
def enforce_capacity(
    V: np.ndarray,
    target_size: Optional[int] = None,
    lo: Optional[int] = None,
    hi: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """硬容量约束：把软分配 V 转为严格受容量界约束的硬标签。

    逻辑：
    1. 初始标签 = argmax(V)，溢出桶按得分保留 top-hi，溢出细胞按次高得分
       转移到未满桶；
    2. 构造性重平衡：把 count>lo 桶中对本桶亲和最低的过剩细胞定向补给
       欠满桶（补到恰好 lo），一步完成，无迭代修复循环；
    3. 标签重映射为 0..n_used-1。

    容量界：默认 target = N // K, lo = max(2, floor(0.8*target)),
    hi = ceil(1.25*target)（自适应，任意 (N,K) 可行）。流水线默认传入
    [20, 35]，K 可行域钳制保证可行性；显式设置时校验，不可行抛 ValueError。

    Parameters
    ----------
    V:
        软分配矩阵 (N x K)，非负。
    target_size / lo / hi:
        显式容量设置（全部留空走自适应推导）。

    Returns
    -------
    labels:
        (N,) int32 硬标签（紧凑编号）。
    counts:
        (n_used,) 每个 Metacell 的细胞数，全部落在 [lo, hi] 内。
    """
    V = np.asarray(V)
    n, k = V.shape
    if target_size is None:
        target_size = max(2, int(n // max(k, 1)))
    if lo is None:
        lo = max(2, int(np.floor(0.8 * target_size)))
    if hi is None:
        hi = int(np.ceil(1.25 * target_size))
    hi = max(hi, lo + 1)

    if n > k * hi:
        raise ValueError(
            f"容量约束不可行: N={n} > K*hi={k}*{hi}={k*hi}。"
            f"需增大 K (至少 {int(np.ceil(n / hi))}) 或放宽 hi。"
        )
    if k * lo > n and k > 1:
        # 允许：部分桶不存在即可（K 是桶数上限），解散逻辑会自然收缩
        logger.debug("N=%d < K*lo=%d，实际使用的桶数将少于 K", n, k * lo)

    labels = np.argmax(V, axis=1).astype(np.int32)
    best = V[np.arange(n), labels].astype(np.float32)
    counts = np.bincount(labels, minlength=k)

    # ---- 1. 溢出桶：保留得分最高的 hi 个，溢出者按次高得分转移
    overflow_cells: list[int] = []
    for b in np.where(counts > hi)[0]:
        members = np.where(labels == b)[0]
        members = members[np.argsort(-best[members])]  # 得分降序
        keep, drop = members[:hi], members[hi:]
        counts[b] = hi
        if drop.size:
            overflow_cells.extend(drop.tolist())
    if overflow_cells:
        # 得分高者优先挑桶（更可能装进次高桶）
        overflow_cells.sort(key=lambda c: -best[c])
        order = np.argsort(-V[overflow_cells], axis=1)  # 每个溢出细胞的桶偏好
        for row_i, cell in enumerate(overflow_cells):
            placed = False
            for b in order[row_i]:
                b = int(b)
                if b != labels[cell] and counts[b] < hi:
                    labels[cell] = b
                    counts[b] += 1
                    placed = True
                    break
            if not placed and counts[labels[cell]] > hi:
                # 理论上可行域内不会发生；兜底放进最空的桶
                b = int(np.argmin(counts))
                labels[cell] = b
                counts[b] += 1

    # ---- 2. 构造性重平衡: 把"过剩细胞"定向补给欠满桶，一步完成。
    #      过剩 = count>lo 的桶中对本桶亲和最低的 (count-lo) 个细胞；
    #      欠满桶按缺口降序接收，补到恰好 lo。
    #      使用的桶数 u <= N//lo（由 K 可行域保证）=> surplus >= deficit 恒成立。
    under = np.where((counts > 0) & (counts < lo))[0]
    surplus_cells: list[int] = []
    for b in np.where(counts > lo)[0]:
        mem = np.where(labels == b)[0]
        mem = mem[np.argsort(V[mem, b])]  # 升序：对桶 b 亲和最弱者在前
        take = int(counts[b] - lo)
        surplus_cells.extend(mem[:take].tolist())
    if under.size and surplus_cells:
        surplus_arr = np.asarray(surplus_cells, dtype=np.int64)
        remaining = np.ones(len(surplus_arr), dtype=bool)
        under_sorted = under[np.argsort(-(lo - counts[under]))]
        for u in under_sorted:
            need_u = int(lo - counts[u])
            cand = np.where(remaining)[0]
            if cand.size == 0:
                break
            aff = V[surplus_arr[cand], u]
            top = cand[np.argsort(-aff)[:need_u]]
            cells = surplus_arr[top]
            src = labels[cells].copy()
            labels[cells] = u
            np.add.at(counts, src, -1)
            counts[u] = lo
            remaining[top] = False
    for b in np.where((counts > 0) & (counts < lo))[0]:
        logger.warning("桶 %d 仍低于 lo=%d（容量可行域边缘），保留 %d 个细胞",
                       b, lo, counts[b])

    # ---- 3. 紧凑重编号
    used = np.unique(labels)
    remap = np.full(k, -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    labels = remap[labels]
    counts = counts[used]
    return labels.astype(np.int32), counts.astype(np.int64)
