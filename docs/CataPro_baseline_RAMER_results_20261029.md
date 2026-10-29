# 原始CataPro仅替换RAMER：最终结果

2026-10-29，cluster40/seed668。作业3184180于11:38:11–11:43:12在gpu4025完成，COMPLETED/0:0。CPU缓存核验3183967已完成。训练28轮自然早停，按原逐批valid RMSELoss与min_delta0.001规则选第8轮；随后原流程最终valid和test完成。没有重投或追加推断。

## 报错核查

完整stderr没有Traceback或运行失败，包含两类UserWarning：

- DataLoader设置8个worker，而Slurm分配7个CPU，PyTorch提示可能减慢或卡顿。为保持用户指定原参数，没有修改workers；本次实际完成全部训练和评估。
- 原runner未显式传torch.load的weights_only参数；既有TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1触发兼容提示。加载的是本次自行训练的checkpoint；strict加载无缺失或额外键，最终指标审计通过。

## 指标

第8轮主选逐批valid RMSELoss：1.1198819260259991。完整3037条valid RMSE：1.2699323024612865。

以下为同一完整3036条test，保存预测独立复算。标签均为log10(kcat)，RMSE越低越好。

| 模型 | RMSE | MAE | R² | PCC |
|---|---:|---:|---:|---:|
| 原CataPro＋ProtT5 | 1.256490612197 | 0.959727724120 | 0.308894720057 | 0.560340975737 |
| 原CataPro＋RAMER | 1.311748313139 | 0.993559048433 | 0.246771501178 | 0.510398576014 |

RAMER本次RMSE增加0.055257700942（4.39778%）。原基线本次复算与历史1.256490612280仅差8.3e-11，为CSV数值读取/舍入精度差异，不影响结论。本次结果未超过原基线；单seed不能据此断言编码器普遍优劣，没有进行新配对bootstrap。

## 忠实性与证据

原CataPro506113参数MLP、原model/utils/runner字节不变；仅蛋白前1024维改为已核验RAMER有效残基均值。MolT5+MACCS后935列、标签、划分、训练与选模参数保持。CUDA初态与CPU/原模型一致，实际两批更新和Adam回载及后续更新审核通过；预检后原runner重新set_seed正式拟合。

16下载文件SHA、17冻结源码本地/服务器SHA、原rows.csv SHA、运行protocol、28轮原早停与选模、完整valid/test行ID和FP32标签、四指标与逐批损失独立复算通过。实际checkpoint SHA：`cb6cdd43590f137999bfcfabcd7f593ba9ae5f4e38d262bd92afa1a9fd6ea207`。本地未加载checkpoint或重新推断。

原runner没有每轮valid逐值预测，所以仅核对最终valid汇总指标与选中历史轮次一致，不能称历史valid逐值重放。两份预测文件保存于服务器run_seed668及本地received_final/run_seed668。

本地证据：`work/catapro_baseline_ramer_cluster40_20261029/completed_local_audit.json`、`received_final/receipt.json`、`received_final/result_audit.json`、完整日志及预测CSV。服务器独立目录为`/share/home/u17015/data/GraphKcat/catapro_baseline_ramer_cluster40_20261029`。所有旧实验保留。
