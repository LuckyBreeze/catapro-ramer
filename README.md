# catapro-ramer

原始 **CataPro MLP 仅替换 RAMER 蛋白 embedding** 的已完成实验代码归档。
数据：CataPro原始表；划分：既有固定cluster40 train/valid/test；seed：668。
本仓库保留实际执行源码、特征准备、审核、Slurm投递脚本、配置校验值与汇总结果。

> 这是2026-10-29已完成实验的源码快照。不是另一个注意力融合模型，也不是可直接在任意电脑运行的全量数据包。脚本保留当时u17015的路径与来源校验；缺少外部数据/缓存时会失败，不会静默重建划分或使用替代输入。

## 这次到底改了什么

原模型输入为 `[ProtT5(1024); MolT5(768); MACCS(167)]`，总计1959维。
本次将前1024维换成RAMER有效残基均值，其余935维逐值保留。

```text
原序列 → 大于1000时首500+尾500 → U/Z/O/B替换X
      → 冻结RAMER官方protein inference分支 → [有效残基数,1024]
      → float32 sum(dim=0)/残基数 → 1024维蛋白向量
      → 拼接原MolT5与MACCS → 原CataPro MLP → log10(kcat)
```

- **原model.py、utils.py、runner.py、tail_batch.py字节不变。** 原模型506113参数。
- 既有读取器仅将特征块名称`ProtT5`改为`RAMER`，见[逐行diff](docs/master_store.patch)。
- 新增[workflow.py](experiments/catapro_baseline_ramer_cluster40_20261029/workflow.py)：核验13192条RAMER缓存、均值池化、按蛋白键映射原27658行、替换特征、CPU/CUDA审核与结果复算。
- 新增[run_pipeline.py](experiments/catapro_baseline_ramer_cluster40_20261029/run_pipeline.py)：CPU准备/审核以及GPU审核→原预检→重新设seed正式训练→原valid/test→结果审核。
- 新增[manage.py](experiments/catapro_baseline_ramer_cluster40_20261029/manage.py)：原部署、带回执防重复sbatch提交、只读状态查询。
- 新增[collect_final.py](experiments/catapro_baseline_ramer_cluster40_20261029/collect_final.py)：下载预测、校验SHA并以Python标准库独立复算。不会加载模型或重新推断。

## 代码目录

| 目录 | 内容 |
|---|---|
| `experiments/catapro_baseline_ramer_cluster40_20261029/` | 本次原CataPro＋RAMER完整冻结执行代码、划分ID及投递回执 |
| `cache_origin/catapro_model2_ramer_cluster40_20261028/` | 本次复用的RAMER缓存的生成来源；完整44文件历史部署及成功的提取sbatch |
| `evidence/` | 本次CPU/CUDA检查、28轮history、protocol、最终summary与审核结果 |
| `docs/` | 改动、运行边界、历史投递与最终结果说明 |
| `third_party/` | CataPro与RAMER上游许可 |
| `scripts/verify_archive.py` | 只用标准库检查归档完整性及Python语法，不进行训练 |

**cache_origin内有历史模型2的model.py/train.py/train.sbatch。** 它们为完整保存当时44文件部署与SHA门禁而收录；它们不是本次原CataPro的模型或入口。本次只复用该目录生成的RAMER缓存，不能运行历史模型2训练来复现本仓库主结果。

同样，`runtime/residue_cache.py`与`support.py`保留原基线导入依赖，其中ESM工具未在这次实验执行；本次没有增加ESM特征。

## 参数与评估

| 设置 | 值 |
|---|---|
| train / valid / test | 21585 / 3037 / 3036 |
| seed | 668 |
| batch size | 8；训练最后8+1合成9，避免单样本BN |
| 优化器 | Adam，lr=1e-5，betas=(0.9,0.999)，weight_decay=0.01 |
| 损失 | 原RMSELoss，sqrt(MSE+1e-6) |
| dropout | 0 |
| 最大epoch / patience / min_delta | 150 / 20 / 0.001 |
| 选模 | 原EarlyStopping.check：逐批valid RMSELoss均值显著改善时保存；首轮counter=1 |
| workers | 8，保持原配置 |

这是一份既有固定cluster40单次划分，不是作者10折逐折训练，也没有新增seed。完整valid RMSE与逐批RMSELoss是不同统计口径。

## 投递与运行

详细路径、依赖和执行边界见[RUNNING.md](docs/RUNNING.md)。实际投递代码：

- [prepare.sbatch](experiments/catapro_baseline_ramer_cluster40_20261029/prepare.sbatch)：intel，4CPU/16G，池化全部缓存及CPU审核。
- [train.sbatch](experiments/catapro_baseline_ramer_cluster40_20261029/train.sbatch)：L40，1GPU/7CPU/32G/24h，CUDA审核及原训练/评估。
- [protein_activation_fixed.sbatch](cache_origin/catapro_model2_ramer_cluster40_20261028/protein_activation_fixed.sbatch)：此前实际成功使用的RAMER提取作业。旧`protein.sbatch`保留为历史，conda激活前`set -u`曾失败，成功版把nounset移到激活后。

全部正式计算必须sbatch，不能在登录节点运行批量提取或训练。仓库上传不代表启动新实验。

本地只读归档检查：

```bash
python scripts/verify_archive.py
```

## 已完成结果

GPU作业3184180于2026-10-29 11:38:11–11:43:12在gpu4025完成，退出码0。
28轮自然早停，主选第8轮，逐批valid RMSELoss **1.119881926026**，完整valid RMSE **1.269932302461**。

| 同一完整3036条test | RMSE | MAE | R² | PCC |
|---|---:|---:|---:|---:|
| 原CataPro＋ProtT5 | 1.256491 | 0.959728 | 0.308895 | 0.560341 |
| 原CataPro＋RAMER | 1.311748 | 0.993559 | 0.246772 | 0.510399 |

本次RMSE增加约4.40%，没有超过基线。仅一个seed，不据此宣称编码器普遍优劣。
源码、输入、选模记录、预测ID/标签和指标审计通过。
原runner没有每轮valid逐值预测，因此只能核对选中轮次valid汇总指标，不能称历史valid逐值重放。

stderr里的workers8>7及torch.load兼容提示均为UserWarning，实际作业正常结束。
报告见[最终结果](docs/CataPro_baseline_RAMER_results_20261029.md)。

## 数据、权重和许可

未包含：原序列/SMILES/标签表、embedding矩阵、逐残基NPY缓存、模型checkpoint、逐样本预测、SSH密钥或认证信息。包含固定划分ID和分箱信息、来源SHA以及聚合指标。

- CataPro作者代码：[zchwang/CataPro](https://github.com/zchwang/CataPro)，保留[MIT许可](third_party/CataPro-LICENSE.txt)。
- RAMER：[Ming-Ni-Group/RAMER](https://github.com/Ming-Ni-Group/RAMER)，固定commit `5f453a0f836b407c77ff666acb2f9dde7c595a30`，保留[CC BY-NC 4.0许可](third_party/RAMER-LICENSE.txt)。RAMER上游代码未修改。
- 本仓库没有把不同来源文件统一重新许可；各第三方文件遵循其原许可。更多归属见[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
