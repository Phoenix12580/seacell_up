# seacell_up — 百万级单细胞 Metacell 全自适应极速构建系统

纯 Python（numpy / scipy.sparse / sklearn / anndata）的 Metacell 构建流水线，
"分治 + 迭代清洗"三层漏斗架构，面向 ~1.3M 细胞 / 100 样本规模。
**每个 Metacell 严格 20-35 个细胞**，所有影响 Metacell 数量/尺寸/质量的
参数（NMF 维数、尺寸 γ、数量 K、质量线）均由数据自适应推导。

> 设计评估与 5 处方案硬伤的修正见 [EVALUATION.md](EVALUATION.md)；
> 上皮/肿瘤 CNV 分支实验记录见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 安装

```bash
pip install -e .              # 核心: numpy/scipy/scikit-learn/anndata/joblib/pandas
pip install -e ".[plot]"      # + UMAP/Harmony 可视化 (scanpy/harmonypy/matplotlib/umap-learn)
```

CNV 推断（可选）另需 `infercnvpy + scanpy` 环境（可与核心包不同环境/机器）。

---

## 使用方法

### 1. 一行构建 Metacell（最常用）

```python
import seacell_up

adata_out, mc_adata, report = seacell_up.build_metacells(
    "cells.h5ad",              # AnnData 对象或 .h5ad 路径; 原始计数/已归一化均可(自动识别)
    sample_key="sampleID",     # obs 中的样本列
    n_jobs=16,                 # 并行进程数
)
```

返回三个对象：

| 返回 | 内容 |
|---|---|
| `adata_out` | 细胞级（输入副本）。obs 新增 `metacell_id`（"MC000123"/"noise"）、`metacell_stage`（stage1/rescue1..3/noise）、`mc_score`、`is_noise` |
| `mc_adata` | Metacell x genes 伪体。obs 含 `n_cells`、`stage`、`score`、`gamma`、`dominant_sample`、`sample_purity` |
| `report` | 运行报告：`report["summary"]`（n_metacells/noise_rate/total_seconds）、`report["blocks"]`（各样本 k_dim/γ/K）、`report["rounds"]`（回收轮次） |

常用可选参数：

```python
adata_out, mc, report = seacell_up.build_metacells(
    adata,
    sample_key="sampleID",
    n_top_genes=2048,          # 全局 HVG 数（全阶段共用基因空间）
    capacity_lo=20,            # Metacell 细胞数硬界（默认 20-35）
    capacity_hi=35,
    max_rounds=3,              # 跨样本回收最大轮数
    pass_rule="mean",          # 通过线: mean(默认)/median/quantile
    # pass_quantile=0.85,      # pass_rule="quantile" 时只回收最差 15%
    # alpha=0.5, n_neighbors=10, search_rows=5000, ...  # 透传 PipelineConfig
)
```

命令行等价：

```bash
python run_pipeline.py cells.h5ad --sample-key sampleID \
    --output cells_with_mc.h5ad --metacell-output metacells.h5ad \
    --n-jobs 16 --report-json report.json
```

### 2. 上皮/肿瘤细胞：CNV 空间按病人分治（可选分支）

上皮（尤其恶性）在表达空间是连续谱，评分天然吃亏；CNV 空间克隆结构离散
且**病人特异**。两步：

```python
# 步骤 A: CNV 推断（需 infercnvpy 环境；产出与核心包环境解耦）
from seacell_up.cnv_infer import infer_cnv
cnv_adata = infer_cnv(
    "cells.h5ad",
    coords="hg38_gene_coords.csv.gz",   # 列: gene_name,chromosome,start,end
    reference_key="ct.main", reference_cat="Immune",   # 免疫作参考
    target_cat="Epithelia",             # 上皮全量
)
cnv_adata.write_h5ad("epi_cnv.h5ad", compression="gzip")

# 步骤 B: CNV 空间按病人分治建 MC（核心包环境即可）
from seacell_up import build_metacells_cnv
cnv_out, cnv_report = build_metacells_cnv("epi_cnv.h5ad", sample_key="sampleID")
print(cnv_report["noise_rate"])     # 上皮 unresolved 比例（实测 ~15%）
```

`infer_cnv` 需要基因坐标 CSV，从 Gencode GTF 一行生成：

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

流程细节：方差 top3000 特征 → 非负平移 → 每个病人独立 k_dim/γ/K（[20,35]）
→ mean 线 → 失败池跨样本 rescue 1 轮 → 失败者内 median 分层放行 → is_noise。

### 3. Metacell UMAP（Harmony 批次校正）

```python
from seacell_up.plotting import plot_metacell_umap

mc.obs["n_cells"] = mc.obs["n_cells"].astype(float)
mc, figs = plot_metacell_umap(
    mc,                                # build_metacells 返回的 mc_adata
    color=["stage", "n_cells"],        # obs 列, 每列一个面板
    batch_key="dominant_sample",       # Harmony 批次列; None=不校正
    normalize_first=True,              # 表达伪体 True; CNV 矩阵伪体 False
    save="mc_umap.png",                # 每列存一张 mc_umap_<col>.png
)
# mc.obsm: X_umap (harmony 版) / X_umap_harmony / X_umap_raw / X_pca_harmony
```

### 4. 典型完整工作流

```python
import seacell_up
from seacell_up import build_metacells, build_metacells_cnv
from seacell_up.plotting import plot_metacell_umap

# ① 全量 Metacell（免疫/基质等离散类型表现最佳）
adata_out, mc, report = build_metacells("cells.h5ad", sample_key="sampleID", n_jobs=16)

# ② 上皮走 CNV 分支（可选; 表达空间噪音细胞 84% 可在 CNV 空间以细 MC 找回）
cnv_out, cnv_report = build_metacells_cnv("epi_cnv.h5ad", sample_key="sampleID")

# ③ 合并标签: CNV 分支覆盖上皮细胞
mask = adata_out.obs_names.isin(cnv_out.obs_names)
adata_out.obs.loc[mask, ["metacell_id", "metacell_stage", "is_noise"]] = \
    cnv_out.obs.loc[adata_out.obs_names[mask],
                    ["metacell_id", "metacell_stage", "is_noise"]]

# ④ 可视化
mc.obs["n_cells"] = mc.obs["n_cells"].astype(float)
mc, _ = plot_metacell_umap(mc, color=["stage", "n_cells"],
                           batch_key="dominant_sample", normalize_first=True,
                           save="mc_umap.png")
```

命令行一步到位（含 CNV 分支）：

```bash
python run_pipeline.py cells.h5ad --sample-key sampleID \
    --output out.h5ad --metacell-output metacells.h5ad --cnv-input epi_cnv.h5ad
```

---

## API 参考

| 函数/类 | 模块 | 说明 |
|---|---|---|
| `build_metacells(adata, sample_key, ...)` | `seacell_up` | **主入口**。分治+GNMF+迭代清洗，返回 (细胞级 AnnData, Metacell 伪体, 报告) |
| `build_metacells_cnv(cnv_adata, sample_key, ...)` | `seacell_up` | 上皮 CNV 空间按病人分治，返回 (带标签 AnnData, 报告) |
| `infer_cnv(adata, coords, reference_key, ...)` | `seacell_up.cnv_infer` | infercnvpy 推断（可选依赖），返回含 CNV 矩阵/cnv_score 的 AnnData |
| `plot_metacell_umap(mc, color, batch_key, ...)` | `seacell_up.plotting` | UMAP+Harmony（可选依赖），返回 (带嵌入 AnnData, figs) |
| `IterativeMetaCellPipeline` / `PipelineConfig` | `seacell_up` | 底层类（build_metacells 即其封装；需要精细控制时直接用） |
| `FastGNMF` / `enforce_capacity` | `seacell_up` | GNMF 引擎与硬容量约束（研究用） |
| `AutoMetaCellOptimizer` | `seacell_up` | 自适应参数优化器（选维/评分/γ 搜索） |
| `run_cnv_branch` / `CNVBranchConfig` | `seacell_up` | CNV 分支底层（build_metacells_cnv 即其封装） |

输出 obs 字段速查：

- 主流水线：`metacell_id` = MC000123/noise；`metacell_stage` = stage1/rescue1..3/noise
- CNV 分支：`metacell_id` = CMC000123/noise；`metacell_stage` = cnv_pass1/cnv_rescue/cnv_stratified/cnv_noise
- 两者都有：`is_noise`（bool）、`mc_score`（主流水线）

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
  │      └─ 第 3 轮仍失败 -> is_noise=True
  │
  └─ 输出: 细胞级标签 + Metacell 伪体 AnnData + 运行报告

上皮/肿瘤专项分支 (build_metacells_cnv / --cnv-input):
  infercnvpy CNV 矩阵 -> 方差 top3000 -> 按病人分治建 MC [20,35]
  -> rescue 1 轮 -> 失败者内 median 分层放行 -> is_noise
```

## 测试

```bash
python tests/test_steps.py                 # Step 1-6 全部（模拟 2 样本 x 5000 细胞）
python tests/test_steps.py step3           # 单跑某个 step
pytest tests/test_steps.py -v              # pytest 兼容
```

**实测结果摘要**（固定容量 [20,35]）：

| 数据 | Metacells | 噪音率 | 质量 | 耗时 / 内存 |
|---|---|---|---|---|
| 模拟 2样本×5000 | 274（全部 20-35） | 4.7% | 类型纯度中位 0.971；稀有类型(3%)回收 **99.9%** | 10 min |
| pbmc3k 2700 细胞 | 79（全部 20-35） | 2.7% | NK/T/B/Mono marker 干净富集 | 9 min |
| Zhao 222,529 细胞 / 90 样本 | **6,000（全部 20-35）** | 11.0%（几乎全为 Epithelia 肿瘤连续谱，其余类型≈0） | ct.main 纯度 median **1.000**（≥0.95 占 85.9%） | 84 min / 29.9 GB，16 核 |
| 上皮 CNV 分支（top3000, 按病人分治） | 2,756 | 15.5% | ct.sub 纯度 1.000；**83.7% 的表达空间"噪音"上皮以细 MC 找回**；高恶性 MC 严格病人特异 | 61 min |

百万级外推（1.3M / 100 样本 / 16 核）：约 2.5-4 小时，内存 <100 GB。

## "全自适应"的边界

由数据推导（无经验阈值）：NMF 维数 k_dim、Metacell 尺寸 γ 与数量 K、
质量 Score、通过线（全局平均分）。文档化配置（非隐藏魔数）：HVG 数、
kNN 邻居数、图正则权重 α、迭代数、资源上限等，见 `PipelineConfig`
docstring 与 EVALUATION.md §三。
