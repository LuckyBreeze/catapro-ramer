# 运行依赖与投递说明

## 原样归档，而非自动重新训练

本次实验已经完成。不要在原目录重复运行prepare/train或删除回执以重新提交。`manage.py submit_cpu/submit_gpu`发现已有提交回执会返回旧JobID；GPU提交还检查CPU审核与sacct退出状态。`manage.py deploy`在本归档已带deployment_manifest的情况下会拒绝再次冻结，不是自动安装器。

需要另一次实验时应在新的独立目录准备数据路径和对应SHA记录，并显式制定新的运行身份。本仓库没有开展这种新拟合，也不自动重写冻结源码的绝对路径。仅修改某个路径后直接运行会使完整性门禁失败。

## 外部文件

原服务器父目录：`/share/home/u17015/data/GraphKcat`。

| 外部目录 | 必需文件/用途 |
|---|---|
| `catapro_cluster40_six_seed668_20261025` | 原inputs/rows.csv、feature1959.npy、feature_store.json、rowspec.json、assignments.csv、原作者及运行源码；完成审计还读取基线test_predictions.csv |
| `catapro_residue_query_model2_20261027` | prepared/protein_items.json及prepared/manifest.json；缓存提取历史完整verify还依赖旧模型2源码/输入SHA |
| `catapro_model2_ramer_cluster40_20261028` | 完成的cache/ramer_protein/manifest.json和13192个NPY数组 |
| `/share/home/u17015/data/RAMER` | 仅重新提取RAMER时所需的ProtT5模型/tokenizer和RAMER.pth官方权重 |

具体SHA：主实验`workflow.py`中的EXPECTED/CACHE_SHA/ITEMS_SHA，`source_identity.json`；提取来源的`resource_identity.json`、`baseline_identity.json`、`upstream_sha256.json`。这些是原实验身份条件，不应为了跑通而删除。

## 环境

实际原训练环境`graphkcat_env`：Python3.9.23、torch2.8.0+cu128、numpy1.23.5、pandas2.3.3、scipy1.13.1。提取环境`cpdiffusion`：torch1.13.1、transformers4.35.2、peft0.7.1；完整实际兼容性检查见cache_origin/environment_audit.json。RAMER上游ramer.yml作为原作者环境参考保留，并不声称本次完整安装了该yml。

没有提供伪造的完整conda锁文件；这次复用已有环境，未安装或修改依赖。

## 当时的执行顺序

以下解释实际脚本的用途，不应向已完成目录重复投递。

1. 先前RAMER提取作业3169378用`protein_activation_fixed.sbatch`，严格加载270权重键、最长序列CUDA重复/官方inference一致后提取13192条有效残基FP32数组。
2. 本次CPU3183967用`prepare.sbatch`，调用`run_pipeline.py cpu`，依次`workflow.py prepare`、`review_cpu`。核验缓存SHA并池化，仅替换原1959维特征前1024列。
3. CPU审核成功后GPU3184180用`train.sbatch`，调用`run_pipeline.py gpu`：`review_cuda`→原runner `--preflight`→原runner正式训练与评估→`audit_result`。
4. 主训练原调用为：

```bash
python -B -u catapro_baseline/runner.py \
  --catapro-root author --split-dir inputs --feature-dir inputs \
  --seed 668 --output-dir run_seed668
```

该命令仅在sbatch分配的GPU作业内执行。正式训练重新设置seed，不继承预检参数更新。原runner选择完成后加载bestmodel.pth，进行最终valid和test各一次。

## 只读状态查询

Windows端具有既有SSH别名u17015时，可从主实验目录运行：

```powershell
python -B manage.py snapshot
```

Slurm只读查询：

```bash
squeue -j 3184180
sacct -j 3184180 --format=JobIDRaw,State,ExitCode,Elapsed,NodeList
```

服务器logs/train.3184180.out与.err包含完整日志；仓库保留汇总证据，不上传全部日志或逐样本预测。完成审计脚本collect_final.py会下载到新received_final目录，核对16文件SHA、源码、ID标签、选模和指标；需要上述原服务器产物，不会重新推断。
