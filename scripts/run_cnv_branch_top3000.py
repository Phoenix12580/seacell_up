#!/usr/bin/env python3
"""TOP3000 特征验证：固化版 cnv_branch 按样本分治 + 方差 top3000。

对照 v2 全维(16,338)指标: 单轮 48.5% / 找回 79.7% / 方差比 0.192 /
纯度 1.000 / unresolved 13.4% / 耗时 95 分钟。
前置: scripts/cnv_infer_epithelial.py 已产生 results/zhao_222k/cnv/epi_cnv.h5ad
"""
from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seacell_up import CNVBranchConfig, run_cnv_branch  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
CNV_ADATA = HERE / "results" / "zhao_222k" / "cnv" / "epi_cnv.h5ad"
ZHAO_OUT = HERE / "results" / "zhao_222k"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    t0 = time.time()
    cnv = ad.read_h5ad(CNV_ADATA)
    cnv = cnv[(cnv.obs["cnv_ref"] == "tumor").values].copy()
    names = cnv.obs_names.copy()
    score = cnv.obs["cnv_score"].values
    ct_sub = cnv.obs["ct.sub"].astype(str).values
    print(f"输入: {cnv.shape}", flush=True)

    res = run_cnv_branch(cnv, CNVBranchConfig(n_top_features=3000, n_jobs=16))
    labels, stage = res["labels"], res["stage"]
    print(f"\n===== TOP3000 结果 (vs v2 全维) =====")
    print(f"总耗时: {time.time()-t0:.0f}s [v2: 5726s]")
    print(f"最终 unresolved: {res['noise_rate']*100:.1f}% [v2: 13.4%]")

    # cnv_score 方差比（单轮通过的 MC）
    ok = stage == "pass1"
    df = pd.DataFrame({"s": score, "mc": labels})[ok]
    w = df.groupby("mc")["s"].var().mean() / max(np.var(score), 1e-12)
    print(f"单轮 MC cnv_score 方差比: {w:.4f} [v2: 0.192]")

    d = pd.DataFrame({"ct": ct_sub, "mc": labels})[ok]
    pur = d.groupby("mc")["ct"].agg(lambda x: x.value_counts().iloc[0] / len(x))
    print(f"单轮 MC ct.sub 纯度: median={pur.median():.3f} [v2: 1.000]")

    # 找回率（对照主流水线表达空间的噪音上皮）
    cells = ad.read_h5ad(ZHAO_OUT / "zhao_cells.h5ad", backed="r")
    epi = cells.obs[(cells.obs["ct.main"] == "Epithelia").values]
    cells.file.close()
    noise_bc = epi.index[epi["metacell_stage"] == "noise"]
    pos = pd.Series(np.arange(len(names)), index=names)
    rows = pos.reindex(noise_bc).dropna().astype(int).values
    rec = ~res["is_noise"][rows]
    print(f"表达空间噪音找回率: {rec.mean()*100:.1f}% [v2: 79.7%]")

    out = ad.AnnData(X=None, obs=pd.DataFrame(
        {"metacell_id": [f"CMC{i:06d}" if i >= 0 else "noise" for i in labels],
         "stage": stage, "is_noise": res["is_noise"]}, index=names))
    out.write_h5ad(CNV_ADATA.parent / "epi_cnv_top3000_labels.h5ad", compression="gzip")
    print(f"\n标签已保存 -> {CNV_ADATA.parent / 'epi_cnv_top3000_labels.h5ad'}")


if __name__ == "__main__":
    main()
