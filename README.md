# seacell_up — 百万级单细胞 Metacell 全自适应极速构建系统

纯 Python（numpy / scipy.sparse / sklearn / anndata）的 Metacell 构建流水线，
"分治 + 迭代清洗"三层漏斗架构，面向 ~1.3M 细胞 / 100 样本规模。

> **先读 [EVALUATION.md](EVALUATION.md)**：原方案有 5 处技术硬伤（γ 搜索与固定
> 容量区间矛盾、软聚合复杂度爆炸、预处理缺失等），本实现保留方案全部核心设计
> 并做了必要修正。**每个 Metacell 严格 20-35 个细胞**（用户决策 2026-08-17），
> 由 K 可行域钳制 `[⌈N/35⌉, ⌊N/20⌋]` 保证任意规模严格可行。

## 架构

```
原始数据 (N cells x S samples)
  │
  ├─ 预处理: log-normalize + 全局 HVG（全阶段共用基因空间）
  │
  ├─ 第一阶段: 各样本并行 [自适应降维 -> 三分法搜γ -> GNMF -> 容量约束 -> 打分]
  │      ├─ Score ≤ 全局平均分 ──> 高质量 Metacell 池
  │      └─ 否则 ───────────────> 回收细胞池
  │
  ├─ 第二阶段: 回收池合并, 同流程清洗 (≤3 轮)
  │      ├─ 通过 -> 稀有 Metacell 池
  │      └─ 第 3 轮仍失败 -> is_noise=True 剔除
  │
  └─ 输出: 细胞级标签 + Metacell 伪体 AnnData + 运行报告

上皮/肿瘤专项分支 (--cnv-input):
  infercnvpy CNV 矩阵 -> 方差 top3000 -> 按病人分治建 MC [20,35]
  -> rescue 1 轮 -> 失败者内 median 分层放行 -> is_noise
  (恶性克隆结构病人特异, 按样本分治远优于混合建块, 见 docs/EXPERIMENTS.md)
```

## 安装

```bash
pip install -e .            # 核心: numpy/scipy/scikit-learn/anndata/joblib
pip install -e ".[plot]"    # + UMAP/Harmony 可视化
```

## 快速开始

```python
import anndata as ad
from seacell_up import IterativeMetaCellPipeline, PipelineConfig

adata = ad.read_h5ad("cells.h5ad")          # X 为原始计数（或已归一化，自动识别）
pipe = IterativeMetaCellPipeline(
    sample_key="sampleID",                  # obs 中的样本列
    config=PipelineConfig(n_jobs=16),
)
adata_out, mc_adata, report = pipe.run(adata)
# adata_out.obs: metacell_id / metacell_stage / mc_score / is_noise
# mc_adata:      Metacell x genes 伪体, obs 含 n_cells/stage/gamma/score/样本纯度
```

命令行：

```bash
python run_pipeline.py input.h5ad --sample-key sampleID \
    --output cells_with_mc.h5ad --metacell-output metacells.h5ad \
    --n-jobs 16 --report-json report.json \
    [--cnv-input epi_cnv.h5ad]   # 上皮 CNV 分支（可选）
```

## 模块

| 模块 | 文件 | 内容 |
|---|---|---|
| AutoMetaCellOptimizer | `seacell_up/optimizer.py` | NMF 曲率选维 `auto_select_n_components`、基尼/零值率评估、三分法 `optimize_gamma` |
| FastGNMF | `seacell_up/gnmf.py` | 图正则 NMF 软分配 + `enforce_capacity` 硬容量约束 |
| IterativeMetaCellPipeline | `seacell_up/pipeline.py` | 分治并行 / 回收迭代 / 噪音剔除 / 输出组装 |
| 预处理 | `seacell_up/preprocessing.py` | log-normalize、全局 HVG、伪体聚合 |
| 上皮 CNV 分支 | `seacell_up/cnv_branch.py` | 按病人分治的 CNV 空间 Metacell（`run_cnv_branch`） |

## 测试

```bash
python tests/test_steps.py                 # Step 1-6 全部（模拟 2 样本 x 5000 细胞）
python tests/test_steps.py step3           # 单跑某个 step
python scripts/run_small_pbmc.py pbmc3k_raw.h5ad       # PBMC 小测试
python scripts/run_large_pbmc.py big.h5ad --sample-key sampleID  # 大规模测试
python scripts/plot_metacell_umap.py       # Metacell UMAP (Harmony 批次校正)
```

**实测结果摘要**（固定容量 [20,35]）：

| 数据 | Metacells | 噪音率 | 质量 | 耗时 / 内存 |
|---|---|---|---|---|
| 模拟 2样本×5000 | 274（全部 20-35） | 4.7% | 类型纯度中位 0.971；稀有类型(3%)回收 **99.9%** | 10 min |
| pbmc3k 2700 细胞 | 79（全部 20-35） | 2.7% | NK/T/B/Mono marker 干净富集 | 9 min |
| Zhao 222,529 细胞 / 90 样本 | **6,000（全部 20-35）** | 11.0%（几乎全为 Epithelia 肿瘤连续谱，其余类型≈0） | ct.main 纯度 median **1.000**（≥0.95 占 85.9%） | 84 min / 29.9 GB，16 核 |
| 上皮 CNV 分支（top3000, 按病人分治） | 2,756 | 15.5% | ct.sub 纯度 1.000；**83.7% 的表达空间"噪音"上皮以细 MC 找回**；高恶性 MC 严格病人特异 | 61 min |

详见 [EVALUATION.md](EVALUATION.md)（方案评估与修正）与 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)（上皮实验全记录）。

## "全自适应"的边界

由数据推导（无经验阈值）：NMF 维数 k_dim、Metacell 尺寸 γ 与数量 K、
质量 Score、通过线（全局平均分）、容量界（由 N/K 导出，保证任意组合可行）。
文档化配置（非隐藏魔数）：HVG 数、kNN 邻居数、图正则权重 α、迭代数、
资源上限等，见 `PipelineConfig` docstring 与 EVALUATION.md §三。

## 上皮 CNV 分支的坐标准备

`scripts/cnv_infer_epithelial.py` 需要基因坐标 CSV（列: gene_name, chromosome, start, end）。
可从 Gencode GTF 生成：

```bash
python - <<'EOF'
import gzip, re, pandas as pd
rows = []
with gzip.open('gencode.v44.annotation.gtf.gz', 'rt') as f:
    for line in f:
        if line.startswith('#'): continue
        p = line.split('\t')
        if p[2] != 'gene' or not re.fullmatch(r'chr[0-9XY]+', p[0]): continue
        m = re.search(r'gene_name "([^"]+)"', p[8])
        if m: rows.append((m.group(1), p[0], int(p[3]), int(p[4])))
pd.DataFrame(rows, columns=['gene_name','chromosome','start','end'])\
  .drop_duplicates('gene_name').to_csv('hg38_gene_coords.csv.gz', index=False, compression='gzip')
EOF
```
