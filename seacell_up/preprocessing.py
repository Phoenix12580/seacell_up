"""预处理与聚合工具（评估 P4 修正：原方案缺失的环节）.

- log-normalize：CPM(target_sum) + log1p，稀疏友好
- select_hvgs：Seurat 式离散度 HVG（纯 numpy 分箱 z-score，不依赖 scanpy）
- 关键约束：HVG 在全部样本上**全局**选择一次，保证第一阶段各样本与
  第二阶段回收池共用同一基因空间
- aggregate_metacells：按硬标签聚合 Metacell 伪体 (pseudobulk)
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def looks_like_raw_counts(X: sp.csr_matrix) -> bool:
    """启发式判断 X 是否像原始计数（整数且幅值大）。仅用于日志提示，不影响流程。"""
    if X.nnz == 0:
        return False
    data = X.data[: min(X.nnz, 1_000_000)]
    return bool(np.allclose(data, np.round(data)) and data.max() > 50)


def normalize_log(X: sp.csr_matrix, target_sum: float = 1e4) -> sp.csr_matrix:
    """每细胞 library-size 归一化到 target_sum 再 log1p。返回新矩阵。"""
    X = sp.csr_matrix(X, dtype=np.float32)
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    scale = np.divide(target_sum, row_sums, out=np.zeros_like(row_sums), where=row_sums > 0)
    out = sp.diags(scale) @ X
    out.data = np.log1p(out.data)
    return out.tocsr()


def select_hvgs(
    X: sp.csr_matrix, n_top: int = 2048, n_bins: int = 20, min_mean: float = 0.01
) -> np.ndarray:
    """Seurat 式 HVG：按平均表达分箱，箱内对离散度做 z-score，取截断标准化离散度最高者。

    Parameters
    ----------
    X:
        已 log 归一化的矩阵 (N x genes)。
    n_top:
        选择的基因数（文档化配置，非质量阈值）。
    n_bins:
        表达分箱数。
    min_mean:
        过滤近零表达基因（数据推导的噪声地板，min_mean=0.01 对应 log1p 后
        平均 < 1e-2，即群体中几乎不表达）。

    Returns
    -------
    基因列索引 (n_top,)。
    """
    X = sp.csr_matrix(X, dtype=np.float64)
    n_cells = X.shape[0]
    gene_mean = np.asarray(X.mean(axis=0)).ravel()
    # 方差 = E[x^2] - mean^2（稀疏：data^2 归到列）
    Xc = X.tocoo()
    gene_sq = np.bincount(Xc.col, weights=(Xc.data**2).astype(np.float64), minlength=X.shape[1]) / n_cells
    gene_var = np.maximum(gene_sq - gene_mean**2, 0.0)

    dispersion = gene_var / np.maximum(gene_mean, 1e-12)
    log_disp = np.log1p(dispersion)
    log_mean = np.log1p(gene_mean)

    ok = gene_mean > min_mean
    if ok.sum() < n_top:
        logger.warning("过表达基因仅 %d < n_top=%d，放宽到全部非零基因", ok.sum(), n_top)
        ok = gene_mean > 0

    idx_ok = np.where(ok)[0]
    # 分箱（箱内 z-score），等频分箱基于 log_mean 排序
    order = idx_ok[np.argsort(log_mean[idx_ok])]
    bins = np.array_split(order, min(n_bins, len(order)))
    z = np.full(X.shape[1], -np.inf)
    for b in bins:
        if len(b) < 3:
            zd = np.zeros(len(b))
        else:
            ld = log_disp[b]
            mu, sd = ld.mean(), ld.std()
            zd = (ld - mu) / (sd + 1e-12)
        z[b] = zd
    return np.argsort(-z)[: min(n_top, len(idx_ok))].astype(np.int64)


def aggregate_metacells(
    X: sp.csr_matrix, labels: np.ndarray, n_metacells: Optional[int] = None
) -> sp.csr_matrix:
    """按标签聚合伪体：返回 (n_metacells x genes) 稀疏矩阵，每行 = 成员表达之和。"""
    X = sp.csr_matrix(X)
    labels = np.asarray(labels, dtype=np.int64)
    if n_metacells is None:
        n_metacells = int(labels.max()) + 1
    ind = sp.csr_matrix(
        (np.ones(len(labels), dtype=np.float32), (np.arange(len(labels)), labels)),
        shape=(len(labels), n_metacells),
    )
    return (ind.T @ X).tocsr()  # (K x N) @ (N x G) = K x G


def dominant_label_purity(labels: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """每个 Metacell 内占多数的组别纯度（用于事后验证，如 ct.main 细胞类型）。

    Returns
    -------
    purity:
        (K,) 每个 Metacell 的 majority fraction。
    dominant:
        (K,) 占多数的组别值。
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    K = int(labels.max()) + 1
    ug = np.unique(groups[~pd_isna(groups)])
    idx = np.searchsorted(ug, groups)
    mat = sp.csr_matrix(
        (np.ones(len(labels)), (labels, idx)), shape=(K, len(ug))
    ).toarray()
    dominant = ug[np.argmax(mat, axis=1)]
    purity = mat.max(axis=1) / np.maximum(mat.sum(axis=1), 1)
    return purity, dominant


def pd_isna(arr: np.ndarray) -> np.ndarray:
    """numpy/分组列的缺失值掩码（兼容 object 数组与 pandas categorical）。"""
    if arr.dtype == object:
        return np.array([x is None or (isinstance(x, float) and np.isnan(x)) for x in arr])
    if np.issubdtype(arr.dtype, np.floating):
        return np.isnan(arr)
    return np.zeros(len(arr), dtype=bool)
