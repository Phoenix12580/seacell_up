"""Step 1 基础设施：模拟稀疏计数 AnnData（2 样本 x 5000 细胞）。

构造 5 个细胞类型，各带独立 marker 基因块；类型在样本间分布不均
（稀有类型约占 2-4%），用于检验第二阶段回收池能否找回稀有群体。
"""
from __future__ import annotations

import numpy as np
import anndata as ad
import scipy.sparse as sp


def make_simulated_data(
    n_samples: int = 2,
    n_cells_per_sample: int = 5000,
    n_genes: int = 2000,
    n_types: int = 5,
    rare_fraction: float = 0.03,
    seed: int = 0,
) -> ad.AnnData:
    """生成带 obs['sample_id'] / obs['true_celltype'] 的原始计数 AnnData。"""
    rng = np.random.default_rng(seed)
    marker_per_type = max(40, n_genes // (2 * n_types))

    type_rates = []
    for t in range(n_types):
        lam = np.full(n_genes, 0.02)
        marker = slice(t * marker_per_type, (t + 1) * marker_per_type)
        lam[marker] = 2.5 + rng.random(marker_per_type) * 2.0
        type_rates.append(lam)

    # 类型构成：主类型 + 每样本一个稀有类型
    types_per_sample = []
    for s in range(n_samples):
        main = list(range(n_types))
        rare = main.pop(s % n_types)
        types_per_sample.append((main, rare))

    counts_rows, cell_types, sample_ids = [], [], []
    for s in range(n_samples):
        main, rare = types_per_sample[s]
        probs = [(1.0 - rare_fraction) / len(main)] * len(main) + [rare_fraction]
        t_draw = rng.choice(main + [rare], size=n_cells_per_sample, p=probs)
        lam = np.stack([type_rates[t] for t in t_draw])
        lib = rng.lognormal(0, 0.4, size=(n_cells_per_sample, 1))
        counts = rng.poisson(lam * lib)
        counts_rows.append(counts)
        cell_types.extend(t_draw.tolist())
        sample_ids.extend([f"sample_{s}"] * n_cells_per_sample)

    X = sp.csr_matrix(np.vstack(counts_rows).astype(np.float32))
    gene_names = [f"Gene{i:05d}" for i in range(n_genes)]
    adata = ad.AnnData(X=X)
    adata.var_names = gene_names
    adata.obs["sample_id"] = pd_series(sample_ids)
    adata.obs["true_celltype"] = pd_series([f"type_{t}" for t in cell_types])
    return adata


def pd_series(values: list) -> "np.ndarray":
    import pandas as pd

    return pd.Categorical(values)
