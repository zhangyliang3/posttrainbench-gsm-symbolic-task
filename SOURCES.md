# 来源与改编说明

| 来源 | 在本题中的用途 | 我们的改编 |
|---|---|---|
| [PostTrainBench](https://github.com/aisa-group/PostTrainBench) | 借鉴“指定基座 + 预算，由 Agent 自主研究后训练，再固定评测”的任务范式 | 新目标设为 GSM-Symbolic 三档条件复杂度，提交协议和评分器重写为 Harbor 形式 |
| [Apple GSM-Symbolic 论文](https://arxiv.org/abs/2410.05229)与[数据](https://huggingface.co/datasets/apple/GSM-Symbolic) | Main/P1/P2 下游题目与金标 | 等量抽样三版、固定单次任务评分，不宣称重现原论文全部实验；官方数据按原样下载，题目与金标不改写 |
| [Apple 数据仓库](https://github.com/apple-aiml-research/ml-gsm-symbolic) | 字段说明、生成实例与原题关系、作者提示和答案提取方法 | 实现自己的固定推理协议和数字提取程序，逐版报告结果 |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | 只使用 train split 作为 Agent 可用训练资源 | 把 test split 与 GSM-Symbolic 留作评测；Agent 可从 train 自主合成新训练样本 |
| [Harbor](https://docs.harborframework.com/core-concepts/tasks/overview) | 任务文件格式、执行和 verifier 输出接口 | 编写本题 `instruction.md`、`task.toml`、环境、参考解和评分脚本 |
| [Qwen2.5-Math-1.5B Base](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) | 指定基座权重 | 仅接受从该基座导出的 LoRA adapter；不使用 Instruct 变体 |

Apple Hugging Face 数据集页面标记为 `cc-by-nc-nd-4.0`；本目录仅提供下载与抽样代码，不把官方题目文件复制进任务源码。若对外发布任务包或衍生数据，应重新核对各来源的许可证与用途。GSM8K 数据集说明采用 MIT 许可；模型权重也应遵守其仓库许可。

本题没有声称合成一个新的数学 benchmark。**合成的是一道新的 Agent 后训练研究任务**：研究问题、资源与数据边界、产物接口、三档均衡评分及 Harbor 执行链。
