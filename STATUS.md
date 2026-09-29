# 实验状态（2026-09-29）

## 已完成

- RTX 3090 24 GB 云主机已通过 SSH 检查；Docker、Harbor 0.23.0、NVIDIA GPU 透传正常。
- 首次镜像构建时 Ubuntu 默认源连接失败，已切换到可用镜像源；模型权重按固定 revision 下载并校验 SHA256。
- DeepSeek API 连通性曾返回 HTTP 200；Agent 使用模型名 `deepseek-flash`。
- 正式 Agent run `jobs/2026-09-29__09-34-46` 已完成 LoRA 训练：586 步、28,892 条训练样本、训练耗时约 1,532 秒，训练 loss 约 0.80 → 0.38；adapter 已被 Harbor 持久化，约 141 MiB，可加载。
- Agent 在 4-shot GSM8K 派生开发集上做过 base 诊断。它发现基座经常在一个回答后继续生成多个 `####`；这是 Agent 选择训练终止行为的依据。该开发日志不是 GSM-Symbolic 正式成绩。
- GSM-Symbolic Main/P1/P2 三个固定源文件已下载到 `gsm_symbolic_posttrain/tests/data/`，SHA256 与 Apple 固定 revision 一致；固定评测抽样为每版 250 条，共 750 条。
- 更新任务以解决审查中指出的问题：Agent 阶段网络白名单包含 DeepSeek API 和 Harbor 安装 Agent CLI 所需的软件包主机（仅限环境安装）；Verifier 阶段断网；GSM8K train 切分为 7,223 条 train 和 250 条 development；要求提交 adapter、训练脚本、实际训练 JSONL 和运行摘要；Verifier 做数字归一化 n-gram 重合检查；adapter 缺失或加载失败会生成结构化 0 分；base 与 adapter 在同一最终题集上配对评测。Docker 的 train/dev 拆分使用 `Dataset.select()`，已通过实际 Harbor 镜像构建验证。
- 固定贪心解码改为 8 条一批，prompt、样本、max tokens 和答案抽取规则不变。静态检查、评分单测、shell 语法和任务 TOML 检查通过。

## 正式成绩与限制

- 原始 Agent run 在总时限 3,600 秒时超时：Agent 把大量时间花在全量 4-shot 开发评估上，导致 Harbor 没有进入 Verifier。因此原始 trial 没有正式 `reward.json`。
- 旧 trial 的 verifier 首次下载 GSM-Symbolic 时遇到 HTTP 429；镜像转址又遇到 HTTP 403。为此已把源数据固定文件随 verifier 包提供，并在运行时继续校验 SHA256。
- 历史 trial 的 paired regrade 第一次因 Harbor 拒绝含字符串字段的 reward JSON 而失败；修复后，Harbor regrade `gsm-symbolic-posttrain-adapter-only-v4` 已成功完成并接受结构化 reward。它复用了匹配的 base baseline，只推理 adapter；Verifier 从 `04:05:10Z` 到 `04:28:51Z`，耗时约 23 分 41 秒。
- 完整诊断分数：base `score=0.4373`（Main 0.656、P1 0.444、P2 0.212）；adapter `score=0.3947`（Main 0.668、P1 0.352、P2 0.164）；`delta=-0.0427`。adapter 在 Main 上略升，但 P1/P2 下降，等权总分下降。
- 这份历史产物没有 `train_script.py`、`train_data.jsonl` 和 `run_summary.json`；为补测能力分数显式关闭了证据门禁，所以这些准确率不能称为满足新证据要求的有效提交，`valid_submission=0`。审计原因另写入 `submission_status.json`，reward JSON 只包含数值字段，Harbor 已验证可以解析。
- 同协议 base 分数固定写入 `tests/base_baseline.json`，包含 base revision/权重 SHA256、评测 JSONL SHA256、解码协议和各项分数。adapter-only verifier 实测约 23 分 41 秒，因此 20 分钟上限偏紧；建议保留 2,400 秒（40 分钟）安全余量。
- 之前工具输出意外显示过 DeepSeek Key。下一次启动 Agent 前必须先在 DeepSeek 控制台轮换该 Key，再由用户更新主机上的 `~/.config/novix/deepseek_api_key`。不要将新 Key 发到对话或写入项目文件。

## 下一步

1. 轮换并更新 DeepSeek Key 后，再跑一次新 Harbor trial。新 trial 必须通过训练数据、脚本、摘要和重合检查，才可报告为有效正式结果。
2. Agent 阶段设为 4,800 秒：其中实际训练预算最多 3,600 秒，剩余 1,200 秒覆盖开发集有限比较、保存训练证据和整理提交。Harbor 的 Agent timeout 是整个 Agent 阶段的总时限，不是纯训练时限。
3. Verifier timeout 保持 2,400 秒（40 分钟）；本次 adapter-only 实测约 23 分 41 秒，20 分钟上限不足以覆盖真实耗时和波动。

固定子集只覆盖 Main/P1/P2 的模板 id 0–49、每个模板 5 个实例，不代表完整 GSM-Symbolic。尚未逐题人工核查所有题面与金标。
