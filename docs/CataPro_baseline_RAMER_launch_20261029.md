# 原始 CataPro 仅替换 RAMER 蛋白 embedding

**最终完成：3184180于11:43:12 COMPLETED/0:0，28轮/选第8轮，完整3036 test RMSE1.311748313139。独立审计通过；详见CataPro_baseline_RAMER_results_20261029.md。下方启动与排队记录为历史。**

用户授权：u17015，CataPro 原始数据、既有 cluster40 划分、seed668，保持原运行参数；仅替换蛋白 embedding，发现异常报告。不覆盖旧文件。

## 已执行与边界

- 最新11:38:24：3184180于11:38:11在gpu4025开始RUNNING，目前独立CUDA审核中，stderr空；尚无GPU审核/预检完成或正式训练轮次。此前排队预计18:36已失效，同一作业自动获卡，无重投。
- 新目录：`/share/home/u17015/data/GraphKcat/catapro_baseline_ramer_cluster40_20261029`；本地对应 `work/catapro_baseline_ramer_cluster40_20261029`。
- CPU 作业3183967：2026-10-29 10:35:30–10:43:02，cpui166，COMPLETED/0:0，stderr空。
- GPU 作业3184180：10:55:08提交，首次PENDING/Priority。1 L40/7CPU/32G/24h，排除gpu4015/4016/4020，无固定节点。后续状态以 `latest_snapshot.json` 为准；勿重复提交。
- 这是原始 CataPro 的506113参数MLP，不是此前的RAMER交叉注意力模型2。

## 唯一输入替换

复用先前完成且冻结的RAMER官方protein inference逐残基缓存。13,192个文件逐个SHA验证、形状/FP32/有限性验证通过。沿用原序列大于1000时首500+尾500、UZOB→X；有效残基求和除以残基数得到1024维向量，没有padding或EOS参与池化。与官方均值池化公式一致，3条独立float64参考最大误差4.44e-8。

原1959维矩阵仅替换前1024维；后935维原MolT5与MACCS逐值不变，标签不变。27658行全部覆盖，train21585/valid3037/test3036。划分、行ID及数据源SHA均核验。新feature SHA：`a02bb6922764e6a61c47ae45cfb7a01b90ab8cad08e5a126f44dbcdbafd0762a`。

## 原参数与训练程序

原model.py、utils.py、runner.py和尾批采样代码字节不变；输入读取器仅将蛋白块名称ProtT5改成RAMER。17部署文件SHA和10来源文件已验证。graphkcat_env保持原环境，无安装。

seed668；batch8且训练末8+1合9；Adam lr1e-5、betas(0.9,0.999)、weight_decay0.01；原RMSELoss与eps1e-6；dropout0；最大150轮、patience20、min_delta0.001；原逐批valid RMSELoss均值显著改善保存规则，首轮counter1；workers8。MolT5和MACCS不重新提取。

CPU审核通过：原模型初态一致、两批实际数据训练更新逐值一致、每BN更新2次、梯度有限、模型与Adam回载及下一次更新一致、训练行覆盖和尾批9、原选模逻辑检查。CPU初态SHA `76b98922a28e1155a26a2579a3f94051559f37f124d94d791f69ef7240f87498`。

GPU同一作业内先独立CUDA审核，再原runner预检，丢弃预检更新后由原runner重设seed正式训练。按原valid规则锁定checkpoint后执行原valid/test流程，末尾独立检查选模历史及预测指标。原runner不保存每轮valid逐值预测，因此最终审核只能将valid汇总指标与选中轮次核对，不能称逐值重放历史valid通过。

## 证据与查询

本地 `deployment_receipt.json`、`local_checks.json`、`submission_cpu.json`、`submission_gpu.json`、`latest_snapshot.json`。服务器 `prepare_result.json`、`review_cpu.json`、后续 `review_cuda.json`、`run_seed668/history.json`、`run_seed668/summary.json`、`result_audit.json` 与 `logs/*.out/*.err`。

只读查询：`python -B work/catapro_baseline_ramer_cluster40_20261029/manage.py snapshot`。GPU实际开训必须以CUDA审核、预检和训练history为依据，不能把提交成功当作训练完成。

已创建本聊天heartbeat `catapro-ramer-seed668`，每10分钟检查；正常排队/普通轮次推进保持安静，异常、需用户决定或最终结果完成时通知，完成审计后停用。10:55:55快照仍PENDING/Priority，尚无CUDA审核或正式训练结果。

## 11:23排队原因核查

用户要求排查并尽快运行。3184180仍为PENDING/Priority，依赖为空、无固定节点、10:55:08已eligible，脚本路径正确；没有发现提交失败或依赖阻塞。L40分区33节点，其中gpu4029/gpu4040为DOWN+DRAIN+NOT_RESPONDING，其余31节点每节点8张卡均已分配，即248/248张卡已占用。MIXED表示仍有部分其他资源，并不代表有空GPU。排除的4015/4016/4020当时也全部占卡，解除排除不能立即释放资源。

调度为priority/basic与sched/backfill，sprio不支持该优先级插件，因此不能提供多因素分值分解。11:23:33预计18:36:00在gpu4036启动；此前一次为22:04，说明仅为动态估计，并非预约或绑定节点。保持既有作业及排队年龄，未重复提交、改训练参数或缩短时限。证据：`work/catapro_baseline_ramer_cluster40_20261029/scheduling_diagnosis_20261029.json`。
