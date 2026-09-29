# GSM-Symbolic 后训练任务：运行配置

本项目用 Harbor 运行一道 PostTrainBench-like 任务：执行 Agent 在一张 GPU 上后训练 `Qwen/Qwen2.5-Math-1.5B`，提交 LoRA adapter；独立 verifier 在 GSM-Symbolic Main、P1、P2 上评分。**DeepSeek API 只驱动执行 Agent，不是被微调的基座模型。**

> **面试提交摘要**：本仓库包含 Harbor 任务包、来源与实验设计、RTX 3090 上的验证记录，以及用于讲解的 `posttrain_task_synthesis.html`。任务环境、GPU 透传、Agent LoRA 训练、离线 verifier 与 reward 解析已验证；当前 `adapter=0.3947`、`base=0.4373` 的结果仅是历史产物的诊断性复评分，因缺少新版要求的训练证据，`valid_submission=0`，不能作为有效正式成绩。完整进展见 [STATUS.md](STATUS.md)；展示建议从 [posttrain_task_synthesis.html](posttrain_task_synthesis.html) 开始。

> **公开查看说明**：为保留最终评测的隔离性，公开仓库不包含 `gsm_symbolic_posttrain/tests/data/` 中的 GSM-Symbolic 题目与答案；仓库保留 verifier、固定文件哈希、抽样与评分逻辑。需要复跑的人应从受控的原始发布版本取得这些文件并通过 SHA256 校验。

## 先确认 AutoDL 实例类型

当前任务使用 `environment/Dockerfile`，Harbor 默认用 Docker 创建 Agent 和 verifier 沙箱。AutoDL 官方文档明确说明：**普通容器实例内不支持 Docker**。因此，普通 AutoDL 4090 容器即使能运行 `nvidia-smi`，也不能直接执行这里的 `harbor run`。需要使用可运行 Docker、且 Docker 容器可访问 GPU 的 Linux 主机，例如 AutoDL 官方建议的裸金属服务器，或者其他满足条件的 GPU 云主机。不要为此尝试在普通 AutoDL 容器内安装 Docker CLI；CLI 不等于可用的 Docker daemon。

如果已经租了普通 AutoDL 4090 容器，它仍可用于**手工训练/调试**，但这不等于完成了 Harbor trial。正式实验应在上述 Docker GPU 主机运行。

在租用或开始计费前，向供应商确认 Docker daemon、NVIDIA Container Toolkit/GPU 透传和至少一张具备 24 GB 显存的 NVIDIA GPU。当前租用主机为 RTX 3090 24 GB；任务容器申请 30 GiB 系统内存。建议预留 60 GB 以上可用磁盘。镜像构建和 Harbor/Agent 初始化阶段需要访问依赖源；Agent 阶段使用白名单网络（DeepSeek API 与仅供安装 CLI 的两个包源），最终 verifier 禁止网络访问。

## 1. 主机检查

在目标 Linux 主机执行：

```bash
nvidia-smi
docker info
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
df -h
```

最后一个 Docker GPU 检查应能在容器内看到实际租用的显卡（当前为 RTX 3090）。若 `docker info` 报无法连接 daemon，或 `--gpus all` 失败，应先修复宿主机的 Docker/GPU 运行时。`task.toml` 中的 `gpus = 1` 只申请一张卡；实际 GPU 型号由云主机决定。`memory_mb = 30720` 表示容器的系统内存上限（30 GiB），不是 GPU 显存。Harbor 0.23.0 自带的本地 Docker 环境暂不声明 GPU 能力，因此在这台主机运行时还需要项目中的 `harbor_gpu_env.py` 和 `gpu-compose.yaml`。

## 2. 上传项目并安装 Harbor

从本机上传目录；SSH 地址、端口以云平台控制台给出的实际值为准：

```bash
scp -r -P <SSH端口> <本地项目目录>/novix_project root@<主机地址>:<远程项目目录>/
```

登录服务器后：

```bash
cd /root/novix_project
python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
uv tool install harbor
harbor --version
```

如果主机已有 `uv`，可直接执行 `uv tool install harbor`。Harbor 的默认沙箱是 Docker；本机另外安装 Claude Code 不是必需的，Harbor 的 `claude-code` 集成会在任务沙箱中安装它。

## 3. 配置 DeepSeek Agent

在 [DeepSeek 开放平台](https://platform.deepseek.com/)创建 API Key。当前 V4.1 Flash 的 API 模型名是 **`deepseek-flash`**；DeepSeek 提供的 Anthropic 兼容入口为 `https://api.deepseek.com/anthropic`。这里选择 Harbor 已集成的 `claude-code` 执行 Agent，通过该入口调用 DeepSeek。

在运行 Harbor 的**云主机**上创建一个仅当前用户可读的密钥文件。只需保存一次；不要把密钥放进项目目录、`config.sh`、任务文件或 shell 历史：

```bash
mkdir -p "$HOME/.config/novix"
chmod 700 "$HOME/.config/novix"
read -r -s -p 'DeepSeek API Key: ' NOVIX_KEY_INPUT
printf '\n'
umask 077
printf '%s\n' "$NOVIX_KEY_INPUT" > "$HOME/.config/novix/deepseek_api_key"
unset NOVIX_KEY_INPUT
chmod 600 "$HOME/.config/novix/deepseek_api_key"
cd /root/novix_project
source ./config.sh
```

每次新开 shell，运行 `cd /root/novix_project && source ./config.sh`，再从**同一个 shell**运行 Harbor。`config.sh` 只读取密钥文件并导出环境变量，不含密钥本身。若想使用其他路径，可在 `source` 前设置 `NOVIX_DEEPSEEK_KEY_FILE`。`ANTHROPIC_AUTH_TOKEN` 是 DeepSeek 官方给 Claude Code 的变量名；同时设置 `ANTHROPIC_API_KEY` 是为了让 Harbor 识别并传递 Agent 凭据。下方 `--ae` 只传**非秘密**的模型与地址配置；不要在 `--ae` 参数中写入 Key，因为命令行参数可能进入进程列表或运行记录。独立 verifier 不需要 DeepSeek Key。

## 4. 冒烟检查和正式运行

先验证任务的参考流程，仅取每个模板 1 个实例，以缩短评分耗时：

```bash
cd /root/novix_project
PYTHONPATH=/root/novix_project EVAL_INSTANCES_PER_TEMPLATE=1 \
  harbor run -p ./gsm_symbolic_posttrain -a oracle \
  -e harbor_gpu_env:DockerGPUEnvironment \
  --extra-docker-compose /root/novix_project/gpu-compose.yaml \
  --environment-build-timeout-multiplier 6 \
  -n 1 -y
```

再运行正式参考流程。默认每个模板取 5 个实例，即 Main/P1/P2 各 250 题：

```bash
PYTHONPATH=/root/novix_project harbor run -p ./gsm_symbolic_posttrain -a oracle \
  -e harbor_gpu_env:DockerGPUEnvironment \
  --extra-docker-compose /root/novix_project/gpu-compose.yaml \
  --environment-build-timeout-multiplier 6 \
  -n 1 -y
```

最后运行 DeepSeek 驱动的真实 Agent：

```bash
PYTHONPATH=/root/novix_project harbor run -p ./gsm_symbolic_posttrain \
  -a claude-code -m deepseek-flash \
  -e harbor_gpu_env:DockerGPUEnvironment \
  --extra-docker-compose /root/novix_project/gpu-compose.yaml \
  --environment-build-timeout-multiplier 6 \
  --agent-setup-timeout-multiplier 10 \
  -n 1 -y \
  --ae ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic \
  --ae ANTHROPIC_MODEL=deepseek-flash \
  --ae ANTHROPIC_DEFAULT_OPUS_MODEL=deepseek-flash \
  --ae ANTHROPIC_DEFAULT_SONNET_MODEL=deepseek-flash \
  --ae ANTHROPIC_DEFAULT_HAIKU_MODEL=deepseek-flash \
  --ae CLAUDE_CODE_SUBAGENT_MODEL=deepseek-flash
```

首次运行会分别构建 Agent 和独立 verifier 的 Docker 镜像，并在镜像构建时下载固定版本的基座模型和 GSM8K train。GSM-Symbolic 固定评测文件随 verifier 任务包提供，逐文件校验 SHA256 后离线抽样；verifier 执行期间无需联网。Agent 执行期通过 Harbor egress allowlist 只允许 `api.deepseek.com`，以及 Harbor 安装 Agent CLI 所需的 `mirrors.aliyun.com` 和 `registry.npmjs.org`；后两者仅用于包安装，禁止借此获取数据、模型权重或 benchmark 答案。Agent 阶段总时限为 4,800 秒，其中训练必须在 3,600 秒内结束；最终独立 verifier 时限为 2,400 秒。使用 `tmux` 运行长任务。

云主机从 Hugging Face 下载 3 GB 权重时出现过 HTTP 429，因此 Dockerfile 从固定 Hugging Face revision 读取模型配置与 tokenizer，通过 ModelScope 下载相同的权重文件，并检查它与该 revision 的文件大小及 SHA256 完全一致。Docker BuildKit 缓存保留可续传的权重，Agent 与 verifier 镜像共用此缓存。**公开模型和数据不要求 HF Key**；首次构建较慢，所以命令中提高了环境构建超时。

## 5. 提交与审计

Agent 必须提交 `/logs/artifacts/adapter/`、`train_script.py`、实际使用的 `train_data.jsonl` 和 `run_summary.json`。Verifier 校验文件结构与记录数，并将训练问题同保留的 GSM8K development split 和固定 GSM-Symbolic 评测题做数字归一化的 n-gram 近重复检查；检测到重合时将提交标为无效。无效或无法加载的 adapter 会留下结构化 0 分 reward 和原因。

GSM8K train 前 7,223 条用于训练，末尾 250 条单独提供为 development set，只能用于策略比较，不能参与训练。Verifier 使用独立环境，网络模式为 `no-network`。

## 6. 看结果与比较

Harbor 在 `jobs/` 下生成 trial 目录。查看其 `verifier/reward.json`（或使用 `harbor view`），重点记录 adapter 的 `score`、`acc_main`、`acc_p1`、`acc_p2`，同协议基座的 `base_score` / 分项准确率、`delta_score`、`valid_submission` 和 overlap report。基座分数固定在 `tests/base_baseline.json`，并绑定固定评测 JSONL 哈希、模型 revision/权重哈希和解码协议；每个新 trial 因此只需推理 adapter。固定评测子集是 GSM-Symbolic 三个版本各 250 题，共 750 题；结果应称为固定子集结果。

首次建立基线时，Verifier 在同一批题、prompt 和贪心解码设置下同时评估 base 与 adapter；后续 trial 读取带哈希校验的固定 base 指标，再评估 adapter 并计算差值。development 指标仅供 Agent 迭代，不替代最终验证。不要把 `oracle` 分数当成基座分数，它是参考训练产物的分数。

## 当前验证状态

本地配置、静态评测数据 SHA256、抽样、评分单测和 shell/Python 语法均已检查。当前正式训练、regrade 和安全改进的阶段状态以 [STATUS.md](STATUS.md) 为准。旧 trial 的补评分会明确标为来源审计不完整；通过新 evidence 和 network gate 的 trial 才算完整有效结果。

## 文档依据

- [AutoDL 容器实例说明](https://www.autodl.com/docs/env/)：普通容器内不支持 Docker。
- [Harbor 安装与沙箱说明](https://docs.harborframework.com/getting-started/installation)、[运行 job](https://docs.harborframework.com/core-concepts/jobs/run-a-job)、[Agent 选项](https://docs.harborframework.com/core-concepts/agents/pre-integrated-agents)、[环境变量隔离](https://docs.harborframework.com/core-concepts/jobs/environment-variables)。
- [DeepSeek 接入 Claude Code](https://api-docs.deepseek.com/zh-cn/quick_start/agent_integrations/claude_code/)及 [API 模型名](https://api-docs.deepseek.com/guides/agent_integrations/openclaw)。
