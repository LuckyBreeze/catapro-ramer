# 实际使用的cluster40数据集，seed668

本目录是已完成原CataPro＋RAMER实验实际使用样本的可读CSV导出。没有重新划分、删除、去重、增广或重算标签。

| 文件 | 样本数 | 簇数 | 顺序 |
|---|---:|---:|---|
| [train.csv](train.csv) | 21585 | 2995 | 原rowspec.train |
| [valid.csv](valid.csv) | 3037 | 161 | 原rowspec.valid |
| [test.csv](test.csv) | 3036 | 163 | 原rowspec.eval.test |

合计27658行。三者行ID及cluster_id均不交叉；这只是对既有簇分配的核验，本次没有重新运行序列相似度聚类。

## 字段

| 列名 | 含义 |
|---|---|
| `author_row_id` | 原数据的稳定行ID，对应训练rowspec中的kcat_* |
| `fold` | **作者原始10折标注，不是本实验train/valid/test划分** |
| `sequence` | 完整原始蛋白序列，原字符串保留 |
| `smiles` | 原始底物SMILES，未规范化或改写 |
| `kcat_s_minus1` | 原始kcat，单位s⁻¹；原字符串保留 |
| `log10_kcat` | 实际输入表的log10(kcat)标签；训练再转换为float32 |
| `split` | 本次实际train / valid / test归属 |
| `cluster_id` | 原assignments.csv的balance_group；是已有cluster40簇标识 |

前六列逐字段字符串与服务器`catapro_cluster40_six_seed668_20261025/inputs/rows.csv`一致，仅增加split和cluster_id。CSV采用UTF-8、LF换行。训练集按CSV顺序构造Dataset后，原RandomSampler每轮打乱；CSV顺序不是每轮实际抽样顺序。验证和测试不shuffle。

## 序列与embedding

CSV保留完整原序列。生成蛋白embedding时，原流程对长度大于1000的序列取首500+尾500，并将U/Z/O/B替换为X，然后由RAMER官方inference分支提取有效残基表示，做float32均值池化。

CSV不含1959维embedding特征。这些表是原样本的文本导出；冻结runner实际读取已核验的`feature1959.npy`、`feature_store.json`和`rowspec.json`，并不会直接读取这三个新CSV。若需要在另一环境运行，仍需准备对应embedding与特征元数据，不能把CSV直接当作模型张量输入。

## 可核验来源

- 原始数据来源：[zchwang/CataPro datasets/kcat-data_0.4simi-10fold.csv](https://github.com/zchwang/CataPro/blob/master/datasets/kcat-data_0.4simi-10fold.csv)。本实验使用既有cluster40单次划分，不是作者10折逐折训练。
- 原实际行表SHA256：`a0f15c13c715006513040387a5de600fdffc42481af5830277e683d5a6dc1710`。
- [manifest.json](manifest.json)保存三个导出CSV的SHA、大小、计数、首末ID以及顺序ID摘要。
- 冻结划分：[rowspec.json](../../experiments/catapro_baseline_ramer_cluster40_20261029/inputs/rowspec.json)、[assignments.csv](../../experiments/catapro_baseline_ramer_cluster40_20261029/assignments.csv)。
- 导出实现：[scripts/export_dataset.py](../../scripts/export_dataset.py)。只从来源SHA完全匹配的行表按rowspec取行，不生成新划分。原字段、ID覆盖、log10关系、簇不交叉均检查通过。

离线检查仓库数据完整性：

```bash
python scripts/verify_archive.py
```

来源许可与归属见[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)和[上游许可证](../../third_party/CataPro-LICENSE.txt)。
