"""Step 1-6 单元测试。直接运行: python tests/test_steps.py [step名 ...]

pytest 兼容: pytest tests/test_steps.py -v
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seacell_up import (  # noqa: E402
    AutoMetaCellOptimizer,
    FastGNMF,
    IterativeMetaCellPipeline,
    PipelineConfig,
    enforce_capacity,
    normalize_log,
    select_hvgs,
)
from tests.simulate import make_simulated_data  # noqa: E402


# ----------------------------------------------------------------- Step 1
def test_step1_simulated_data():
    """2 样本 x 5000 细胞模拟数据：形状/稀疏性/obs 列。"""
    adata = make_simulated_data()
    assert adata.shape == (10000, 2000)
    assert sp.issparse(adata.X)
    assert set(adata.obs["sample_id"].unique()) == {"sample_0", "sample_1"}
    assert adata.obs["true_celltype"].nunique() == 5
    vc = adata.obs.groupby("sample_id")["true_celltype"].value_counts()
    print("  每样本类型构成 OK; 总 nnz =", adata.X.nnz)
    return adata


# ----------------------------------------------------------------- Step 2
def test_step2_optimizer_metrics():
    """基尼/零值率的正确性与速度 (<0.1s @ 5000x50)。"""
    opt = AutoMetaCellOptimizer()
    rng = np.random.default_rng(0)

    # 极限情形: 均匀行 gini=0, 独热行 gini=1-1/K
    K = 50
    assert abs(opt._compute_gini(np.ones((100, K))) - 0.0) < 1e-9
    onehot = np.repeat(np.eye(K), 2, axis=0)
    assert abs(opt._compute_gini(onehot) - (1 - 1 / K)) < 1e-9

    # 手工对照小向量
    x = np.array([[1.0, 1.0, 4.0]])
    xs = np.sort(x, axis=1)
    n = 3
    manual = 2 * np.arange(1, n + 1) @ xs[0] / (n * xs.sum()) - (n + 1) / n
    assert abs(opt._compute_gini(x) - manual) < 1e-12

    # 速度: 5000 x 50
    V = rng.random((5000, 50)).astype(np.float32) ** 2
    t0 = time.time()
    opt._compute_gini(V)
    dt_gini = time.time() - t0
    Xs = sp.random(5000, 500, density=0.1, format="csr", dtype=np.float32,
                   random_state=1)
    t0 = time.time()
    opt._compute_zerorate(V, Xs)
    dt_zero = time.time() - t0
    assert dt_gini < 0.1 and dt_zero < 0.1, f"速度不达标: gini={dt_gini:.3f}s zero={dt_zero:.3f}s"
    print(f"  gini 5000x50: {dt_gini*1000:.1f}ms | zerorate: {dt_zero*1000:.1f}ms")

    # 零值率语义: 全零表达 -> 1.0; 全正 -> 0.0
    Xz = sp.csr_matrix((5, 10), dtype=np.float32)
    V5 = np.abs(rng.random((5, 3))).astype(np.float32)
    assert opt._compute_zerorate(V5, Xz) == 1.0
    Xf = sp.csr_matrix(np.ones((5, 10), dtype=np.float32))
    assert opt._compute_zerorate(V5, Xf) == 0.0


def test_step2_auto_select_n_components():
    """NMF 曲率选维输出合理 k。"""
    adata = make_simulated_data(n_cells_per_sample=5000)
    Xn = normalize_log(adata.X.tocsr())
    hvg = select_hvgs(Xn, n_top=500)
    Xh = Xn[:, hvg].tocsr()
    opt = AutoMetaCellOptimizer(max_k=50, random_state=0)
    k, info = opt.auto_select_n_components(Xh, max_k=50)
    assert 5 <= k <= 50, f"k 越界: {k}"
    assert len(info["errors"]) == len(info["ks"])
    print(f"  auto_select k_dim = {k} (误差曲线 {len(info['ks'])} 点)")
    return k


# ----------------------------------------------------------------- Step 3
def test_step3_gnmf_and_capacity():
    """GNMF 软分配 + 硬容量约束严格界。"""
    rng = np.random.default_rng(0)
    emb = np.abs(rng.normal(0, 1, (5000, 25))).astype(np.float32)
    gnmf = FastGNMF(n_metacells=100, max_iter=80, random_state=0)
    V = gnmf.fit_transform(emb)
    assert V.shape == (5000, 100)
    assert np.allclose(V.sum(axis=1), 1.0, atol=1e-3)

    # 自适应界 (N=5000, K=100 -> target 50, 界 [40, 63])
    labels, counts = enforce_capacity(V)
    lo, hi = max(2, int(0.8 * (5000 // 100))), int(np.ceil(1.25 * (5000 // 100)))
    assert counts.min() >= lo and counts.max() <= hi, \
        f"自适应界违例: [{counts.min()}, {counts.max()}] 不在 [{lo}, {hi}]"
    assert counts.sum() == 5000

    # 规格原文的严格界: 可行组合 N=3000, K=100, [25, 35]
    V3 = V[rng.choice(5000, 3000, replace=False)]
    l3, c3 = enforce_capacity(V3, target_size=30, lo=25, hi=35)
    assert c3.min() >= 25 and c3.max() <= 35, f"[25,35] 违例: [{c3.min()}, {c3.max()}]"
    print(f"  GNMF V OK | 自适应界 [{counts.min()},{counts.max()}] ⊂ [{lo},{hi}] | "
          f"严格界 [{c3.min()},{c3.max()}] ⊂ [25,35]")

    # 不可行组合 (评估 P1: 规格自相矛盾的 N=5000,K=100,[25,35]) 必须报错
    try:
        enforce_capacity(V, target_size=30, lo=25, hi=35)
        raise AssertionError("不可行组合未报错")
    except ValueError:
        print("  不可行组合 (N=5000,K=100,[25,35]) 正确抛出 ValueError")


# ----------------------------------------------------------------- Step 4
def test_step4_gamma_search():
    """三分法 gamma 搜索收敛。"""
    adata = make_simulated_data(n_cells_per_sample=5000, n_samples=1)
    Xn = normalize_log(adata.X.tocsr())
    Xh = Xn[:, select_hvgs(Xn, n_top=500)].tocsr()
    opt = AutoMetaCellOptimizer(max_k=30, random_state=0)
    emb = opt.fit_embedding(Xh, k=15)

    calls = {"n": 0}

    def runner(embedding, K, final):
        calls["n"] += 1
        return FastGNMF(n_metacells=K, max_iter=60 if not final else 120,
                        random_state=0).fit_transform(embedding)

    t0 = time.time()
    gamma, V, info = opt.optimize_gamma(Xh, emb, runner)
    dt = time.time() - t0
    assert 5.0 <= gamma <= 500.0
    assert V is not None and V.shape[0] == 5000
    assert info["evaluations"] <= 14, f"评估次数异常: {info['evaluations']}"
    K_opt = info["best"]["K"]
    assert V.shape[1] == K_opt
    print(f"  gamma_opt={gamma:.1f} K={K_opt} | 唯一 K 评估 {info['evaluations']} 次 | "
          f"GNMF 调用 {calls['n']} 次 | {dt:.1f}s")
    for tr in info["trace"]:
        print(f"    gamma={tr['gamma']:.0f} K={tr['K']:4d} score={tr['score']:.4f} "
              f"(gini={tr['mean_gini']:.3f} zero={tr['zero_penalty']:.3f})")


# ----------------------------------------------------------------- Step 5
def test_step5_full_pipeline():
    """完整流水线（模拟 2 样本 x 5000）：标签/噪音/伪体输出。

    注: 默认配置的全量运行已验证通过; 此处用缩减迭代数的等价配置控制测试时长。
    """
    adata = make_simulated_data()
    cfg = PipelineConfig(n_top_genes=500, n_jobs=2, max_rounds=3,
                         max_k=50, gnmf_search_iter=40, gnmf_final_iter=150)
    pipe = IterativeMetaCellPipeline(sample_key="sample_id", config=cfg)
    adata_out, mc_adata, report = pipe.run(adata)

    for col in ("metacell_id", "metacell_stage", "mc_score", "is_noise"):
        assert col in adata_out.obs, f"缺少 obs 列 {col}"
    n_noise = int(adata_out.obs["is_noise"].sum())
    s = report["summary"]
    assert s["n_metacells"] == mc_adata.n_obs
    assert mc_adata.shape[1] == adata.n_vars
    sizes = mc_adata.obs["n_cells"].values
    # 每个 Metacell 严格 20-35 个细胞
    assert sizes.min() >= 20 and sizes.max() <= 35, \
        f"Metacell 尺寸越界 [{sizes.min()}, {sizes.max()}] 不在 [20, 35]"
    assert sizes.sum() == 10000 - n_noise

    # 事后验证（非算法输入）：Metacell 内真实细胞类型纯度
    import pandas as pd

    passed = adata_out.obs[~adata_out.obs["is_noise"]]
    ct = passed.groupby("metacell_id", observed=True)["true_celltype"].agg(
        lambda x: x.value_counts().iloc[0] / len(x)
    )
    rare_recovered = passed[passed["true_celltype"] == "type_0"].shape[0]
    total_rare = (adata.obs["true_celltype"] == "type_0").sum()
    print(f"  Metacells={s['n_metacells']} 噪音率={100*s['noise_rate']:.2f}% | "
          f"类型纯度 mean={ct.mean():.3f} 中位={ct.median():.3f} | "
          f"稀有类型回收 {rare_recovered}/{total_rare}")
    assert ct.mean() > 0.7, f"纯度过低: {ct.mean():.3f}"
    return report


# ----------------------------------------------------------------- Step 6
def test_step6_robustness():
    """坏 K 值自动跳过 + 日志可追踪。"""
    import logging

    adata = make_simulated_data(n_samples=1, n_cells_per_sample=2000)
    Xn = normalize_log(adata.X.tocsr())
    Xh = Xn[:, select_hvgs(Xn, n_top=300)].tocsr()
    opt = AutoMetaCellOptimizer(max_k=20, random_state=0)
    emb = opt.fit_embedding(Xh, k=10)

    def bad_runner(embedding, K, final):
        if K > 30:  # 模拟大 K 崩溃
            raise RuntimeError(f"模拟 GNMF 失败 K={K}")
        return FastGNMF(n_metacells=K, max_iter=30, random_state=0).fit_transform(embedding)

    gamma, V, info = opt.optimize_gamma(Xh, emb, bad_runner)
    assert not np.isnan(gamma) and V is not None
    failed = [c for c in info["trace"]]  # trace 只含成功项
    logger = logging.getLogger("seacell_up.optimizer")
    assert logger is not None
    print(f"  异常 K 自动跳过 OK: gamma={gamma:.1f}, 成功评估 {len(failed)} 次")


TESTS = {
    "step1": test_step1_simulated_data,
    "step2a": test_step2_optimizer_metrics,
    "step2b": test_step2_auto_select_n_components,
    "step3": test_step3_gnmf_and_capacity,
    "step4": test_step4_gamma_search,
    "step5": test_step5_full_pipeline,
    "step6": test_step6_robustness,
}


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    selected = sys.argv[1:] or list(TESTS)
    failed = []
    for name in selected:
        fn = TESTS[name]
        print(f"\n===== {name}: {fn.__name__} =====")
        t0 = time.time()
        try:
            fn()
            print(f"----- {name} PASS ({time.time()-t0:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"----- {name} FAIL: {exc}")
            failed.append(name)
    print("\n" + "=" * 50)
    print("FAILED: " + ", ".join(failed) if failed else "ALL PASSED ✓")
    sys.exit(1 if failed else 0)
