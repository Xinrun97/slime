# CorePassRL：基于 slime + simple-coding-harness 训练自研编程大模型

## 一、整体思路与架构选择

**架构定位**：slime 主导整个训练循环，simple-coding-harness 只提供 agent 逻辑（工具调用、文件编辑、多轮对话管理）。LLM 推理完全由 slime 管理的 SGLang 引擎负责，harness 的 LLM 调用经由 shim 透明转发给 SGLang，harness 自身感知不到这一层替换。

**权重同步机制**（来自 `train.py:93`）：每个 rollout_id 循环末尾都执行 `actor_model.update_weights()`，把 Megatron 最新训练权重同步到 SGLang 推理引擎。因此每轮"生成→训练→生成→训练"循环中，下一轮生成用的就是当前最新的模型权重。

**模型说明**：
- **基础权重**：**Qwen3.6-27B**（HuggingFace 开放权重，训练起点）
- **架构 spec**：`slime_plugins.models.qwen3_5`（Qwen3.6 与 Qwen3.5 架构相同，复用该 spec）
- **使用方式**：`--spec qwen3_5` 描述模型结构，`--hf-checkpoint` 加载 Qwen3.6-27B 权重——spec 与权重独立，二者不需来自同一版本
- **harness 路由名**：`openai/qwen3.6-27b` 是 harness `model_routes.py` 中的 API 路由标识，指向我们的 shim，最终由 SGLang 提供服务

**关键发现**：simple-coding-harness 里已经有 `QWEN36_27B_API_BASE` / `QWEN36_27B_API_KEY` 这两个环境变量，专门为指向本地推理服务设计。`api_key` 会被 LiteLLM 作为 `Authorization: Bearer {key}` 发出——这和 `coding_agent_rl` 里把 session_id 塞进 `ANTHROPIC_AUTH_TOKEN` 的机制完全相同。

因此集成点极为干净：**不需要修改两个项目的任何一行代码**。

**训练大循环**（来自 `train.py`）：

```
for rollout_id in range(num_rollout):
    rollout_data = rollout_manager.generate(rollout_id)   # SGLang 推理 + 沙箱执行
    actor_model.train(rollout_data)                        # Megatron 训练
    actor_model.update_weights()                           # 权重同步到 SGLang ← 每步必执行
```

**核心结构**（含 AgentCore）：

```
┌─────────────────────────────────────────────────────────────────────────┐
│  slime 训练大循环（train.py）                     训练集群（自有 GPU）    │
│  ┌─────────────────┐   权重同步(每步)   ┌───────────────────────────┐   │
│  │ Megatron 训练    │ ────────────────→ │ SGLang 推理引擎             │   │
│  │ (Qwen3-27B)     │                   │ (args.sglang_router_*)     │   │
│  └─────────────────┘                   └─────────────┬─────────────┘   │
│                                                       │ /generate       │
│                         ┌─────────────────────────────▼───────────────┐ │
│                         │  harness_rl/shim.py（VPC 内网端口）           │ │
│                         │  · OpenAI /v1/chat/completions               │ │
│                         │  · 记录 TurnRecord + session 轨迹状态         │ │
│                         └─────────────────────────────┬───────────────┘ │
└───────────────────────────────────────────────────────│─────────────────┘
                          Bearer={session_id}，VPC 内网（同 Region）↑
┌───────────────────────────────────────────────────────│─────────────────┐
│  AWS AgentCore Runtime（托管 microVM，按需自动扩缩容）                    │
│                         ┌─────────────────────────────▼───────────────┐ │
│                         │  harness/agentcore_rl.py（已有代码，不改）    │ │
│                         │  @rollout_entrypoint → 后台执行，立刻返回     │ │
│                         │  · 设置 QWEN36_27B_API_BASE=shim公网地址      │ │
│                         │  · 设置 QWEN36_27B_API_KEY=session_id        │ │
│                         │  · 运行 run_cloud()（harness agent 主循环）   │ │
│                         │  · 计算 reward_breakdown，写入 S3            │ │
│                         └──────────────────────────────────────────────┘ │
│                 ↓ 结果自动保存到 S3（{exp_id}/{session_id}/result.json）  │
└─────────────────────────────────────────────────────────────────────────┘
         ↑ RolloutClient.invoke_async() / future.result_async()
         来自 harness_rl/rollout_fn.py 的 custom_generate
```

---

## 二、代码仓库布局

```
slime/
└── examples/
    └── harness_rl/
        ├── shim.py           # Component A：OpenAI shim + 轨迹记录（训练集群上运行）
        ├── rollout_fn.py     # Component B：--custom-generate-function-path
        ├── agentcore_client.py  # Component C：AgentCore RolloutClient 封装（替代 sandbox.py）
        ├── reward.py         # Component D：奖励函数包装（从 S3 结果取值）
        ├── dataset.py        # Component E：数据集格式转换
        ├── launch.sh         # 训练启动脚本（示例）
        └── README.md
```

**不修改任何项目代码**：
- `simple-coding-harness/src/services/sandbox/agentcore_rl.py` 已有 `@rollout_entrypoint`，直接用
- `simple-coding-harness/rl_data/collect_rollouts.py` 的 `RolloutClient` 用法是参考模板
- harness 的 Docker 镜像已部署到 AgentCore Runtime，包含所有依赖

---

## 三、Component A：Shim（`shim.py`）

**职责**：实现 OpenAI `/v1/chat/completions` API，内部调 SGLang `/generate`，记录每轮的 `TurnRecord`，最后按 session 汇出 `TurnSegment`。

**启动方式**：与 `coding_agent_rl/middleware.py` 完全相同——在 rollout 进程内以后台守护线程运行，只启动一次（单例）。

### 3.1 HTTP 路由

```
POST /v1/chat/completions     ← LiteLLM 调用入口
GET  /v1/models               ← LiteLLM 健康探测
GET  /healthz                 ← 启动就绪检查
```

### 3.2 每轮请求处理流程

```
1. 从 Authorization: Bearer {token} 提取 session_id
2. 获取该 session 的 Chain（对话状态机）
3. 检测是否发生 "wipe"（上下文压缩）：
   比较新 messages 与已记录的历史
   若不是已有历史的延伸 → 快照当前 turns 为 kind="wipe" TurnSegment
4. 用 tokenizer.apply_chat_template(messages, tools) 计算 prompt_ids
5. POST SGLang /generate：
   {"input_ids": prompt_ids, "return_logprob": true, "sampling_params": {...}}
6. 从响应提取 output_token_logprobs → output_ids, output_log_probs
7. 存储 TurnRecord(prompt_ids, output_ids, log_probs)
8. 解析 output_ids → reasoning + text + tool_calls（复用 slime.agent.parsing）
9. 构造 OpenAI response 返回给 LiteLLM
```

### 3.3 Wipe 检测

harness 在 token 数超过阈值时会调用 `_llm_compact_history()`，将旧消息替换为摘要。检测方式：

```python
def _is_wipe(chain: Chain, new_messages: list[dict]) -> bool:
    if chain.seen_msgs == 0:
        return False
    # 新 messages 的前 N 条不匹配已记录历史 → 发生了 wipe
    overlap = min(len(new_messages), len(chain.msg_hashes))
    for i in range(overlap):
        if _hash_msg(new_messages[i]) != chain.msg_hashes[i]:
            return True
    return False
```

检测到 wipe 时，把当前 `chain.turns` 快照为 `kind="wipe"` 的 `TurnSegment`，然后清空 `chain.turns` 重新开始。

### 3.4 SGLang 连接

shim 从 `args` 里直接读取 slime 自动注入的 SGLang 地址（与 `coding_agent_rl/generate.py:90` 相同）：

```python
sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
```

`args.sglang_router_ip` 和 `args.sglang_router_port` 由 slime 在启动 SGLang 引擎时自动设置，无需额外配置。

### 3.5 Session 管理 API（内部，供 rollout_fn 调用）

```python
def open_session(store, session_id, *, tokenizer, tools_schema, sampling_defaults): ...
def pop_session_split(store, session_id) -> list[TurnSegment]: ...  # 同 coding_agent_rl
def shutdown_session(store, session_id): ...
```

`pop_session_split` 和 `merge_turn_segments` 直接复用 `slime.agent.trajectory`，零重复代码。

### 3.6 与 coding_agent_rl/middleware.py 的对比

| | coding_agent_rl | harness_rl（本方案）|
|---|---|---|
| 对外 API | Anthropic Messages API | OpenAI Chat Completions API |
| 消息翻译 | Anthropic blocks → chat_template | OpenAI messages → chat_template（更简单）|
| Session 识别 | `ANTHROPIC_AUTH_TOKEN` | `QWEN36_27B_API_KEY`（Bearer token）|
| 推理引擎 | SGLang `/generate` | SGLang `/generate`（相同）|
| 轨迹记录 | TurnRecord + merge_turns | TurnRecord + merge_turns（复用）|

---

## 四、Component B：Rollout 函数（`rollout_fn.py`）

使用 `--custom-generate-function-path`，复用 slime 默认 rollout 外层循环。

### 4.1 签名

```python
async def custom_generate(args, sample: Sample, sampling_params: dict) -> list[Sample]:
```

### 4.2 完整流程（使用 AgentCore）

```python
SHIM_PUBLIC_URL = os.environ["SHIM_PUBLIC_URL"]   # 训练前 export，AgentCore 能访问的 shim 地址
SWE_TIME_BUDGET_SEC = int(os.environ.get("SWE_TIME_BUDGET_SEC", "600"))

async def custom_generate(args, sample, sampling_params):
    state = _State.get(args)          # 单例：初始化 tokenizer + shim + AgentCore client

    session_id = f"cagent-{sample.index}-{sample.group_index}-{secrets.token_hex(4)}"
    # HARNESS_TOOLS_SCHEMA：从 harness 的工具注册表导出的 OpenAI 格式 schema，
    # 供 shim 在 apply_chat_template 时注入工具定义，使模型输出合法的工具调用格式。
    # 初始化时调用：from src.tools import all_tools_openai; HARNESS_TOOLS_SCHEMA = all_tools_openai()
    open_session(state.store, session_id,
                 tokenizer=state.tokenizer,
                 tools_schema=HARNESS_TOOLS_SCHEMA,
                 sampling_defaults=sampling_params)

    # sample.metadata 由 dataset.py 构造，包含 config 和 env_vars 两个字段
    task_config = sample.metadata.get("config", {})
    task_env_vars = sample.metadata.get("env_vars", {})

    # 1. 提交到 AgentCore（异步，立刻返回 future）
    #    env_vars 直接设置 QWEN36_27B_API_BASE 和 QWEN36_27B_API_KEY，
    #    AgentCore 的 agentcore_rl.py 会把它们注入 harness 进程环境
    payload = {
        "action": "start",
        "config": task_config,
        "env_vars": {
            **task_env_vars,  # 任务自带的 env（如 GITHUB_TOKEN）
            "MODEL": "openai/qwen3.6-27b",
            "QWEN36_27B_API_BASE": f"{SHIM_PUBLIC_URL}/v1",
            "QWEN36_27B_API_KEY": session_id,  # shim 用此识别 session
            "HARNESS_EXTENDED_THINKING": "0",
            "HARNESS_LLM_NUM_RETRIES": "1",
        },
    }
    future = await state.agentcore_client.invoke_async(payload)

    # 2. 等待 AgentCore 完成（harness 执行 + 评估 + 奖励全在 agentcore_rl.py 里）
    try:
        s3_result = await future.result_async(timeout=SWE_TIME_BUDGET_SEC)
        reward = float(s3_result.get("rewards", 0.0))
        timed_out = False
    except TimeoutError:
        reward = -1.0
        s3_result = {}
        timed_out = True

    # 3. 取出 shim 记录的 TokenSegment（在 EC2 训练节点的进程内存中）
    segments = pop_session_split(state.store, session_id)
    shutdown_session(state.store, session_id)

    # 4. 构建训练样本
    # 超时或 shim 无轨迹（agent 未发任何请求）时返回 ABORTED 样本，
    # 而非空列表——slime 的 generate_and_rm_group 期望至少一个 Sample
    if not segments or timed_out:
        sample.status = Sample.Status.ABORTED
        return [sample]

    return fan_out_sample_segments(
        sample, segments, reward,
        state.tokenizer,
        metadata={"s3_result": s3_result},
    )
```

**关键点**：奖励由 AgentCore 内的 `agentcore_rl.py` 计算（调用 harness 自己的 `compose_reward`），通过 S3 返回。训练集群只取 `s3_result["rewards"]`，不需要重新运行评估。

---

## 五、Component C：AgentCore 客户端封装（`agentcore_client.py`）

封装 `RolloutClient`，在 slime 的 `_State` 单例里初始化：

```python
from agentcore_rl_toolkit import RolloutClient

AGENT_RUNTIME_ARN = os.environ["AGENTCORE_RUNTIME_ARN"]
S3_BUCKET         = os.environ["AGENTCORE_S3_BUCKET"]
EXP_ID            = os.environ.get("AGENTCORE_EXP_ID", f"slime-{uuid4().hex[:8]}")
MAX_CONCURRENT    = int(os.environ.get("AGENTCORE_MAX_CONCURRENT", "50"))

def build_rollout_client() -> RolloutClient:
    return RolloutClient(
        agent_runtime_arn=AGENT_RUNTIME_ARN,
        s3_bucket=S3_BUCKET,
        exp_id=EXP_ID,
        tps_limit=25,
        max_pool_connections=max(MAX_CONCURRENT, 10),
    )
```

`RolloutClient` 本身管理并发、限流、指数退避轮询 S3——这些原来需要 DockerSandbox 手动实现的逻辑全部由 AgentCore 提供。

---

## 六、Component D：奖励函数

奖励由 AgentCore 内的 `agentcore_rl.py` 在沙箱执行完毕后立即计算，写入 S3 结果：

```json
{ "rewards": 0.72, "reward_breakdown": {"r_task": 1.0, "r_quality_det": 0.44, ...} }
```

`custom_generate` 直接从 `s3_result["rewards"]` 取值，**无需在训练集群重复计算奖励**。

奖励权重通过 AgentCore Runtime 的容器环境变量控制（部署时配置）：
```
REWARD_WEIGHT_TASK=1.0
REWARD_WEIGHT_QUALITY_DET=0.5
REWARD_WEIGHT_QUALITY_LLM=0.0
REWARD_WEIGHT_COST=0.0          # 训练时不惩罚 token 成本
```

`custom_generate` 返回前预填 `sample.reward`，`sglang_rollout.py:297` 的 `if sample.reward is None` 判断会跳过 RM 调用，不需要设置 `--rm-type`。

---

## 七、Component E：数据集格式（`dataset.py`）

**关键约束**：`cloud_run.py`（AgentCore 内执行 harness 的入口）读取的是 `RunConfig` 格式，要求字段为 `repo_url`、`base_branch`、`agent_branch`、`task`、`test_command`。**SWE-bench 的 `base_commit` + `test_patch` + Docker image 格式与此不兼容，无法直接使用。**

**正确的训练任务格式**（与 harness 原生 RunConfig 对齐）：

```jsonl
{
  "instance_id": "django-issue-1234",
  "config": {
    "run_id": "rl-django-1234",
    "repo_url": "https://github.com/our-org/our-repo",
    "base_branch": "main",
    "agent_branch": "rl-fix/django-1234",
    "task": "Fix the bug where X causes Y when Z...",
    "test_command": "python -m pytest tests/test_foo.py -x",
    "setup_command": "pip install -e .",
    "max_fix_attempts": 3
  },
  "env_vars": {}
}
```

**注意**：`GITHUB_TOKEN` 不应放在 `env_vars` 字段（训练数据文件中），应在 AgentCore Runtime 容器层统一注入（AWS Secrets Manager 或容器环境变量），防止 token 泄露和轮转问题。

**slime JSONL 输出格式**（`--prompt-data` 参数）：

```jsonl
{
  "prompt": "Fix the bug where X causes Y when Z...",
  "label": "django-issue-1234",
  "metadata": {
    "instance_id": "django-issue-1234",
    "config": { ... },
    "env_vars": {}
  }
}
```

**关于奖励格式**：`cloud_run.py` 的 `RunResult` 使用 `lifecycle_outcome` 字段（`"ready_for_review"` 或 `"failed"`），而 `agentcore_rl.py` 的 `_compute_reward` 调用 `compose_reward` 时依赖 `result.get("resolved")`。**需要在 `agentcore_rl.py` 中添加格式适配**：
```python
# 在 _compute_reward 中：
result["resolved"] = (result.get("lifecycle_outcome") == "ready_for_review")
result["f2p_passed"] = len([t for t in result.get("test_results", {}).get("passed", [])])
```

转换命令：

```bash
python examples/harness_rl/dataset.py \
    --input /data/internal_tasks.jsonl \
    --output /data/slime_tasks.jsonl
```

---

## 八、AWS AgentCore 详解与集成要点

### 8.1 AgentCore 是什么

Amazon Bedrock AgentCore 是 AWS 的托管 agent 执行平台，核心是 **Runtime**：

- **microVM 隔离**：每个 session 独立 microVM，真正的沙箱隔离
- **自动扩缩容**：按需启动，最多支持 1000 个并发 session（可申请提升）
- **长运行支持**：单个 session 最长 8 小时（适合复杂编程任务）
- **无服务器**：无需管理 Docker 集群，按用量计费

### 8.2 RL Toolkit（`agentcore-rl-toolkit`）

`agentcore-rl-toolkit` 是 AWS 开源的 SDK，专为 RL 训练设计：

| 组件 | 位置 | 作用 |
|---|---|---|
| `AgentCoreRLApp` | agent 容器内 | 替换 `BedrockAgentCoreApp`，`@rollout_entrypoint` 让 agent 后台执行并自动保存结果到 S3 |
| `RolloutClient` | 训练客户端 | 批量提交 rollout、管理并发、轮询 S3 结果 |
| `RolloutFuture` | 训练客户端 | 单个 rollout 的 future，`result_async(timeout=...)` 等待 S3 结果 |

**harness 已有的 RL 集成**（`src/services/sandbox/agentcore_rl.py`）：
```python
@app.rollout_entrypoint
def invoke_agent(payload: dict) -> dict:
    # 1. 从 payload["env_vars"] 设置环境变量（包括 QWEN36_27B_API_BASE）
    # 2. 调用 run_cloud()（harness agent 主循环）
    # 3. 计算 reward_breakdown（compose_reward）
    # 4. 返回 {"rewards": float, "reward_breakdown": dict, "result": dict}
    # → @rollout_entrypoint 自动将返回值保存到 S3
```

### 8.3 session_id 传递机制

```
rollout_fn.py                    AgentCore microVM
  session_id = "cagent-xxx"
  payload["env_vars"] = {
    "QWEN36_27B_API_KEY": session_id  ──→  env var in harness process
  }                                         ↓
                                   harness LiteLLM call:
                                     api_key = QWEN36_27B_API_KEY = session_id
                                     → Authorization: Bearer cagent-xxx
                                         ↓ HTTP
                              shim.py（训练集群）
                              提取 Bearer token = session_id
                              → 关联到正确的 TurnRecord 列表
```

### 8.4 网络连通性（训练在 AWS EC2 上）

训练在 EC2 上进行，与 AgentCore Runtime 同在 AWS 网络内，连通性最简单：

```
EC2（slime 训练 + shim）
    └── VPC 私有子网
         ↕ 同 VPC 或 VPC Peering（内网，无需公网）
AgentCore Runtime（microVM）
         ↕ 同 Region
S3（rollout 结果）
```

**配置要点**：
- `SHIM_PUBLIC_URL` 设为 EC2 实例的**私有 IP**（或内网 DNS），不需要公网 IP
- Security Group：EC2 的 shim 端口（默认 18001）允许来自 AgentCore 的 VPC 流量入站
- IAM：EC2 实例角色授予 `bedrock-agentcore:InvokeAgentRuntime` 和 `s3:GetObject/PutObject` 权限
- 建议 EC2 和 AgentCore Runtime 在同一 AWS Region，降低延迟并避免跨区域数据传输费用

### 8.5 长尾任务与训练效率

SWE 任务执行时间分布极不均匀（10s ~ 600s），直接影响 GPU 利用率：

```
标准 train.py：
  rollout_batch = 128 个任务并发提交给 AgentCore
  → 等待最慢那个（P95 ≈ 580s）才能开始训练
  → 大部分 GPU 在 P50~P95 这段时间（约 460s）空转
  GPU 利用率 ≈ P50/P95 ≈ 120/580 ≈ 20%

train_async.py（本方案）：
  rollout N+1 在训练 rollout N 时已提前启动
  → 训练结束后大概率 rollout 已就绪
  GPU 利用率显著提升，但仍受最慢样本影响
```

进一步优化（V2）：使用 `--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async`，维持固定大小的 in-flight 任务池，训练步只取已完成的样本，彻底消除等待长尾的问题。适合稳定后的大规模训练。

### 8.6 并发控制

AgentCore 默认账号限制 1000 个并发 session。`RolloutClient(max_concurrent_sessions=N)` 控制并发上限。建议与 slime 的 `--rollout-batch-size` 对齐：

```
--rollout-batch-size 32 --n-samples-per-prompt 4
→ 同时最多 32×4 = 128 个 AgentCore session
→ AGENTCORE_MAX_CONCURRENT=128
```

---

## 九、多节点 EC2 训练配置

### 9.1 slime 多节点机制

slime 用 **Ray** 做跨节点调度，**Megatron** 做分布式训练，二者协同工作：

```
EC2 Head 节点（同时运行 shim）
  └── ray start --head → Ray 集群入口
        ├── EC2 Worker 1: ray start --address=HEAD:6379
        ├── EC2 Worker 2: ray start --address=HEAD:6379
        └── EC2 Worker N: ray start --address=HEAD:6379

ray job submit → train.py → Ray 自动跨节点分配 GPU bundles → Megatron 接管分布式训练
```

slime 的 `placement_group.py` 会跨节点收集所有 GPU（按节点 IP + GPU 编号排序），统一分配给 Megatron actor。Megatron 负责跨节点的 TP/PP/CP 并行通信（NCCL over EFA/InfiniBand）。

### 9.2 Qwen3.6-27B 并行策略（4节点 × 8×H100，colocate 模式）

**GPU 分配**：4×8=32 张，全部同时用于训练和推理（`--colocate` 时间复用）。

| 参数 | 值 | 说明 |
|---|---|---|
| `--actor-num-nodes` | 4 | EC2 节点数 |
| `--actor-num-gpus-per-node` | 8 | 每节点 GPU 数 |
| `--colocate` | 开启 | 训练与 SGLang 推理共用同一批 GPU（时间复用），无需额外推理节点 |
| `--tensor-model-parallel-size` | 4 | 节点内 TP（NVLink）|
| `--pipeline-model-parallel-size` | 2 | 跨节点 PP（EFA）|
| `--context-parallel-size` | 4 | 长序列 CP |
| `--rollout-num-gpus-per-engine` | 2 | colocate 模式下每个 SGLang 引擎 2 张 GPU（参考 run-qwen3.5-27B.sh）|

TP=4×PP=2=8 张一组，共 4 组数据并行。colocate 模式下训练完成后 SGLang 接管同一批 GPU 做推理，再同步权重后进入下一轮训练。

### 9.3 启动脚本（`launch.sh`）

沙箱相关配置通过**环境变量**传入（slime 的 argument parser 不接受未定义参数）：

```bash
#!/bin/bash
set -ex

# ── 必填：节点配置 ──────────────────────────────────────────
export MASTER_ADDR=<Head节点私有IP>
export ACTOR_NUM_NODES=4
export ACTOR_NUM_GPUS_PER_NODE=8
export HOSTFILE=/root/hostfile   # 每行一个 Worker 节点私有 IP

# ── 必填：AgentCore / shim（通过环境变量，不是 CLI 参数）──
export SHIM_PUBLIC_URL=http://${MASTER_ADDR}:18001  # AgentCore microVM 访问 shim 的地址（VPC 内网）
export SHIM_PORT=18001
export AGENTCORE_RUNTIME_ARN=arn:aws:bedrock-agentcore:us-west-2:...:runtime/...
export AGENTCORE_S3_BUCKET=your-rollout-bucket
export AGENTCORE_EXP_ID=corepass-run-01
export AGENTCORE_MAX_CONCURRENT=128   # 对齐 rollout-batch-size × n-samples-per-prompt
export SWE_TIME_BUDGET_SEC=600

# ── 网络：关闭代理，避免 Ray/NCCL 通信异常 ────────────────
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="127.0.0.1,${MASTER_ADDR}"

# ── Step 1：启动 Ray Head ──────────────────────────────────
ray stop --force; ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
    --dashboard-host=0.0.0.0 --dashboard-port=8265

# ── Step 2：SSH 启动所有 Worker 节点 ──────────────────────
for WORKER_IP in $(awk '{print $1}' "${HOSTFILE}"); do
    [[ "${WORKER_IP}" == "${MASTER_ADDR}" ]] && continue
    ssh root@"${WORKER_IP}" \
      "ray stop --force; ray start \
        --address=${MASTER_ADDR}:6379 \
        --num-gpus ${ACTOR_NUM_GPUS_PER_NODE} \
        --node-ip-address ${WORKER_IP}" &
done
wait

# ── Step 3：提交训练任务 ───────────────────────────────────
RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "MASTER_ADDR": "${MASTER_ADDR}",
    "GLOO_SOCKET_IFNAME": "eth0",
    "NCCL_SOCKET_IFNAME": "efa0",
    "FI_PROVIDER": "efa",
    "FI_EFA_USE_DEVICE_RDMA": "1",
    "PYTHONPATH": "/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "SHIM_PUBLIC_URL": "${SHIM_PUBLIC_URL}",
    "SHIM_PORT": "${SHIM_PORT}",
    "AGENTCORE_RUNTIME_ARN": "${AGENTCORE_RUNTIME_ARN}",
    "AGENTCORE_S3_BUCKET": "${AGENTCORE_S3_BUCKET}",
    "AGENTCORE_EXP_ID": "${AGENTCORE_EXP_ID}",
    "AGENTCORE_MAX_CONCURRENT": "${AGENTCORE_MAX_CONCURRENT}",
    "SWE_TIME_BUDGET_SEC": "${SWE_TIME_BUDGET_SEC}"
  }
}
EOF
)

# ── 前置步骤：把 Qwen3.6-27B HF checkpoint 转换为 torch_dist 格式 ──
# Qwen3.6 与 Qwen3.5 架构相同，使用 qwen3_5 spec 转换 Qwen3.6 权重
# python tools/convert_hf_to_torch_dist.py \
#     --hf-checkpoint /root/models/Qwen3.6-27B \
#     --output /root/models/Qwen3.6-27B_torch_dist

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train_async.py \
  --actor-num-nodes "${ACTOR_NUM_NODES}" \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
  --colocate \
  \
  # ── 模型架构：Qwen3.6-27B 权重 + qwen3_5 spec（架构相同）──
  --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec" \
  --num-layers 64 --hidden-size 5120 --ffn-hidden-size 17408 \
  --num-attention-heads 24 --num-query-groups 4 --kv-channels 256 \
  --vocab-size 248320 --swiglu --qk-layernorm --disable-bias-linear \
  --normalization RMSNorm --norm-epsilon 1e-6 \
  --position-embedding-type rope --rotary-percent 0.25 --rotary-base 10000000 \
  --untie-embeddings-and-output-weights --attention-output-gate \
  \
  # ── 检查点（权重来自 Qwen3.6-27B）──────────────────────────
  --hf-checkpoint /root/models/Qwen3.6-27B/ \
  --ref-load /root/models/Qwen3.6-27B_torch_dist/ \
  --load /root/models/Qwen3.6-27B_slime/ \
  --save /root/models/Qwen3.6-27B_slime/ \
  \
  # ── 并行策略 ────────────────────────────────────────────────
  --tensor-model-parallel-size 4 \
  --pipeline-model-parallel-size 2 \
  --context-parallel-size 4 \
  --sequence-parallel \
  \
  # ── 内存与效率（来自 run-qwen3.5-27B.sh 参考配置，架构相同适用）──
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --use-dynamic-batch-size --max-tokens-per-gpu 8192 \
  --calculate-per-token-loss \
  --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d \
  --use-precision-aware-optimizer \
  --optimizer adam --lr 1e-6 --lr-decay-style constant \
  --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
  --attention-dropout 0.0 --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 --attention-backend flash \
  \
  # ── 数据 ────────────────────────────────────────────────────
  --prompt-data /data/slime_tasks.jsonl \
  --input-key prompt --label-key label \
  --apply-chat-template --rollout-shuffle \
  --balance-data \
  \
  # ── 集成入口 ────────────────────────────────────────────────
  --custom-generate-function-path examples.harness_rl.rollout_fn.custom_generate \
  --dynamic-sampling-filter-path \
      slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std \
  \
  # ── SGLang ──────────────────────────────────────────────────
  --rollout-num-gpus-per-engine 2 \
  --sglang-tool-call-parser qwen25 \
  \
  # ── GRPO ────────────────────────────────────────────────────
  --advantage-estimator grpo \
  --kl-coef 0.00 \
  --kl-loss-coef 0.00 --kl-loss-type low_var_kl \
  --eps-clip 0.2 --eps-clip-high 0.28 \
  \
  # ── Rollout ─────────────────────────────────────────────────
  # global-batch-size 必须 = rollout-batch-size × n-samples-per-prompt
  # 32 × 4 = 128；若想用 256 需把 rollout-batch-size 改为 64
  --num-rollout 500 \
  --rollout-batch-size 32 \
  --n-samples-per-prompt 4 \
  --rollout-max-response-len 8192 \
  --rollout-temperature 1.0 \
  --global-batch-size 128 \
  \
  # ── 保存与评估 ───────────────────────────────────────────────
  --save-interval 50 \
  --eval-interval 50 \
  --eval-prompt-data /data/swebench_test.jsonl
```

关键参数说明：

| 参数 | 值 | 原因 |
|---|---|---|
| 入口 `train_async.py` | — | 长尾任务：训练与下一轮 rollout 重叠，减少 GPU 空转 |
| `--colocate` | 添加 | 32 张 GPU 全给训练，没有 GPU 剩余给 SGLang |
| `--spec` | `slime_plugins.models.qwen3_5` | 模型通过 `--spec` + 架构参数指定，不存在单一 model-type flag |
| `--balance-data` | 添加 | 平衡各 DP rank 的序列长度，避免长序列节点成为瓶颈 |
| `--recompute-*` | 添加 | 激活重计算，~40% 显存节省 |
| `--calculate-per-token-loss` | 添加 | GRPO 变长序列正确 loss 归一化 |
| `--optimizer-cpu-offload` | 添加 | 27B 优化器状态 offload 到 CPU |
| `--dynamic-sampling-filter-path` | 添加 | 过滤全零方差组（全失败或全成功），节省无效训练步 |
| `--sglang-tool-call-parser qwen25` | 添加 | Qwen3 工具调用格式解析 |
| `--global-batch-size` | 128 | 必须 = rollout_batch_size(32) × n_samples(4)；256 会超出单轮 rollout 的样本数 |
| `--rollout-temperature` | 1.0 | 编程任务需要更多探索 |
| `--kl-coef 0.00` + `--kl-loss-coef 0.00` | 双零 | 早期训练不加 KL 惩罚；二者互斥，参考脚本均设 0 |
| `--load` | 与 `--save` 相同 | 断点续训；若 `--load` 路径无 checkpoint 会自动 fallback 到 `--ref-load` |
| `FI_PROVIDER=efa` + `FI_EFA_USE_DEVICE_RDMA=1` | 添加 | EC2 多节点训练启用 EFA RDMA，跨节点通信提速 5-10× |

---

## 十、评估循环

**方式一：slime 内置 eval**

每 50 步自动跑 eval（已在 launch.sh 中配置）：

```bash
--eval-interval 50 \
--eval-prompt-data /data/swebench_test.jsonl
```

**方式二：直接用 harness 的 bench.py**

把训练好的 checkpoint 转回 HF 格式，启动 SGLang 服务，然后：

```bash
MODEL=openai/qwen3.6-27b \
QWEN36_27B_API_BASE=http://localhost:8000/v1 \
python bench.py --dataset swebench_verified --n 50
```

可直接得到 resolved rate，与同等规模商业模型横向对比。

---

## 十一、运营与成本估算

### 11.1 AWS 成本概估（500 rollout，4× p4de.24xlarge）

| 资源 | 计算 | 金额（On-demand） | 金额（Spot ~70% off）|
|---|---|---|---|
| EC2（4× p4de.24xlarge @~$32.8/hr）| ~80h（500轮 × 580s/轮）| ~$10,500 | ~$3,200 |
| AgentCore Runtime | 500×128 session × 200s_avg | ~$214 | ~$214 |
| S3 存储/传输 | rollout 结果 + checkpoint | ~$20 | ~$20 |
| **合计** | | **~$10,700** | **~$3,400** |

**建议**：先用 Spot 实例跑 50~100 rollout 的探索实验（~$340-680），验证 reward 有上升趋势后再扩到完整训练。p4de.24xlarge Spot 中断率相对较低，配合 `--save-interval 50` 可安全恢复。

### 11.2 数据集规模与多样性

**SWE-bench 的局限**：
- SWE-bench verified：~500 个任务，rollout_batch=32 × 500轮 = 一个数据集多次重复
- 同一任务反复出现 → 模型过拟合特定 repo/语言，而非泛化编程能力

**建议的数据混合策略**：

| 数据来源 | 规模 | 价值 |
|---|---|---|
| SWE-bench verified | ~500 | 标准 benchmark，便于横向对比 |
| 内部 bug 修复记录 | 按需 | 贴合实际业务场景 |
| SWE-bench-extra / open-source issues | ~2000+ | 更多任务多样性 |

简单任务（resolved rate > 20%）先暖启动，再引入困难任务，避免早期全零奖励。

### 11.3 n_samples_per_prompt 的成本效益

每个 AgentCore session 约 200s、成本约 $0.003。

| n_samples | 每任务成本 | GRPO 信号 | 建议 |
|---|---|---|---|
| 2 | ~$0.006 | 较弱（方差低）| 预算紧张时 |
| 4 | ~$0.012 | 均衡 | **推荐起始值** |
| 8 | ~$0.024 | 充足 | 验证有效后扩大 |

---

## 十二、长期兼容策略

### 11.1 对 slime 更新的兼容

我们只依赖 slime 的**公开 customization 接口**：

| 依赖点 | 位置 | 稳定性 | 策略 |
|---|---|---|---|
| `custom_generate` 签名 | `sglang_rollout.py:267-275` | 高（公开 API） | 无风险 |
| `Sample` 字段及 `sample.reward` 预填跳过 RM | `sglang_rollout.py:297-300` | 高（向后兼容） | 无风险 |
| `args.sglang_router_ip/port` 自动注入 | slime 启动时设置 | 高 | 无风险 |
| 每步 `update_weights()` 同步到 SGLang | `train.py:93` | 高（核心机制） | 无风险 |
| `TurnRecord`, `merge_turns` | `slime/agent/trajectory.py` | 中（内部但稳定） | slime 更新时跑 `test_agent_trajectory.py` 验证 |
| `fan_out_sample_segments` | `slime/agent/trajectory.py` | 中 | 同上 |

操作：用 git submodule 把 slime 锁定版本，按需升级。

### 11.2 对 harness 更新的兼容

**我们依赖 harness 的唯一合约**：

> harness 调用 `QWEN36_27B_API_BASE` 指向的 OpenAI `/v1/chat/completions` 端点

只要 harness 继续用 LiteLLM 发 HTTP 请求，shim 完全不受 harness 内部变动影响（工具改了、消息格式变了、新增功能……都无所谓）。

风险点：

| harness 变化 | 影响 | 应对 |
|---|---|---|
| 删除 `QWEN36_27B_API_BASE` 支持 | 高 | PR 给 harness 加一个通用 `CUSTOM_API_BASE` 机制（一行改动）|
| 切换 LiteLLM → 其他 HTTP 客户端 | 中 | 只要还用 Bearer token，shim 不需改；若换认证方式则改 shim 的 session_id 提取逻辑 |
| tools schema 格式大变 | 低 | 更新 `HARNESS_TOOLS_SCHEMA` 常量 |

操作：把 `simple-coding-harness` 作为 git submodule 管理，Docker 镜像里锁定版本；harness 有重大更新时，先跑 smoke test（单任务走完全流程）再升级。

### 11.3 双向更新流程

```
slime 有新版本：
  1. 跑 tests/test_agent_trajectory.py
  2. 跑 shim 的 smoke test（单任务端到端）
  3. 通过 → 升级 slime submodule

harness 有新版本：
  1. 检查 src/query/model_routes.py 是否变动
  2. 跑单任务端到端 smoke test
  3. 通过 → 升级 harness submodule
  4. 重新 build Docker 训练镜像
```

### 11.4 接口隔离图

```
examples/harness_rl/
├── shim.py              ← 只依赖 slime.agent.{trajectory,parsing}（内部稳定）
│                           + SGLang /generate API（sglang 稳定）
├── rollout_fn.py        ← 只依赖 slime 公开 custom_generate 接口
│                           + agentcore_client.py（自控）
├── agentcore_client.py  ← 只依赖 agentcore-rl-toolkit RolloutClient（AWS SDK，稳定）
│                           + 环境变量（AGENTCORE_RUNTIME_ARN 等）
└── dataset.py           ← 无运行时依赖，仅格式转换
```

奖励计算在 AgentCore 容器内由 harness 处理，训练集群侧无 reward 依赖。

---

## 十三、第一性原理审核

RL 训练模型在智能体场景下持续提升，需要以下条件全部成立。以下是逐条核查结果：

### 13.1 必要条件清单

| # | 条件 | 要求 | 当前状态 | 行动 |
|---|---|---|---|---|
| 1 | 基础模型可被 slime 加载 | Qwen3.6-27B 架构需有对应 spec | ⚠ 需确认 | 核查 HF model card，比对 qwen3_5 spec 参数 |
| 2 | 轨迹由当前策略生成 | shim 直连 SGLang，log_probs 来源于生成本身 | ✓ 架构正确 | 实现 shim 后验证 |
| 3 | loss_mask 正确 | 模型输出=1，工具结果/system=0 | ✓ merge_turns() 已实现 | 跑 test_agent_trajectory.py |
| 4 | 奖励有方差（基础模型非零成功率）| 至少 5% resolved rate | **✗ 未验证** | **训练前必须预测**，见下 |
| 5 | 任务格式与 cloud_run.py 兼容 | RunConfig 格式（repo_url+branch） | **✗ 格式错误** | 已在第七章更正 |
| 6 | 奖励字段格式一致 | agentcore_rl.py 需适配 RunResult 格式 | **✗ lifecycle_outcome ≠ resolved** | 已在第七章说明修改点 |
| 7 | 训练域与部署域一致 | 训练任务 ≈ 实际使用场景 | ⚠ 需要混合内部任务 | 见第十一章 |

### 13.2 阻断条件 #4：基础模型成功率（最重要）

**RL 只能从成功中学习**。如果基础模型 resolved rate ≈ 0%，GRPO 的所有 group 都是全失败（零方差），dynamic_sampling_filter 过滤掉所有 group，训练步数 = 0 梯度，$10,000 打水漂。

**必须在正式训练前做的预检（约 $50-100）：**

```bash
# 用基础模型（不做任何训练）跑 50-100 个训练任务
python rl_data/collect_rollouts.py \
    --agent_arn ${AGENTCORE_RUNTIME_ARN} \
    --s3_bucket ${AGENTCORE_S3_BUCKET} \
    --tasks_file /data/train_tasks_sample.jsonl \
    --output_dir ./preflight_eval \
    --limit 100 \
    --model_id openai/qwen3.6-27b
```

**判断标准**：
- resolved rate > 10%：可以开始 RL 训练
- resolved rate 3-10%：建议先 SFT 暖启动，再 RL
- resolved rate < 3%：任务太难；换更简单任务集，或先做 SFT

### 13.3 阻断条件 #1：Qwen3.6-27B 架构确认

从 slime 测试代码可知 Qwen3.6-35B-A3B 使用 `qwen3.5-35B-A3B` spec（即 qwen3_5 系列 spec 覆盖 Qwen3.6 MoE 架构）。Qwen3.6-27B 需要：

1. 查阅 HuggingFace 上 `Qwen/Qwen3.6-27B` 的 `config.json`，确认 `model_type` 和架构参数
2. 与 `slime_plugins/models/qwen3_5.py` + `scripts/models/qwen3.5-27B.sh` 的架构参数对比（若一致则直接复用 spec，仅替换 `--hf-checkpoint` 为 Qwen3.6-27B 路径）
3. 若维度一致（hidden_size, num_layers 等），直接复用 `qwen3_5` spec + 更新架构参数
4. 若 Qwen3.6-27B 是 MoE 而非 dense，参照 `qwen3.5-35B-A3B` 的配置方式

### 13.4 shim 的正确性验证策略（最高工程风险）

shim 是整个管道中唯一一个**错误不报警、但会静默污染训练数据**的组件。如果 loss_mask 有误或 token_ids 有偏移，模型会"训练"在错误的分布上，没有任何崩溃信号。

**验证方案（三步）：**

```
Step 1：单轮验证
  - 手动构造一条已知输入的对话（固定 messages + tools）
  - 记录 shim 输出的 prompt_ids 和 output_ids
  - 直接用 tokenizer.decode() 验证 ID 与文本一致
  - 验证 output_ids 与 SGLang 实际生成的文本对应

Step 2：loss_mask 验证  
  - 运行一条含工具调用的完整 trajectory（tool call + tool result + model reply）
  - 检查 TokenSegment.loss_mask：
    · 工具调用前的 assistant 文本 → 1
    · tool result 的 token → 0  
    · tool result 之后的 assistant 文本 → 1
  - 与 test_agent_trajectory.py 的用例交叉验证

Step 3：log_probs 验证
  - 对相同输入做两次独立生成（固定 temperature=0）
  - 验证 log_probs 一致（确认 return_logprob=True 工作正常）
  - 与 Megatron 的 ref_log_probs 计算对比（不能偏差超过 1e-4）
```

### 13.5 核心风险总结

| 风险 | 概率 | 对策 |
|---|---|---|
| 基础模型在训练任务上 resolved rate < 3% | 中 | **训练前预检**（见 13.2）；必要时先 SFT |
| Qwen3.6-27B 架构 slime 不支持 | 低 | 预检架构参数（见 13.3）；qwen3_5 spec 大概率覆盖 |
| shim 静默错误污染训练数据 | 中 | 严格执行三步验证（见 13.4）后再扩规模 |
| 任务格式与 cloud_run.py 不兼容 | ✓ 已发现 | 已在第七章修正为 RunConfig 格式 |
| AgentCore 并发/配额不足 | 低 | 默认 1000 session；可申请提升 |
| 训练域（内部任务）与评估域（SWE-bench）不一致 | 中 | 混合内部+公开任务；单独建立内部 eval 集 |
| 显存不足 | 低 | colocate + optimizer-cpu-offload + FP8 rollout |
| shim wipe 检测误判 | 低 | V1 可跳过 wipe 检测，损失压缩段但不影响正确性 |

---

## 十四、推进路线

```
第 0 周（预检，花 $50-100，避免后续浪费$10,000）：
  ├── 确认 Qwen3.6-27B HF config.json 架构参数 → 对比 qwen3_5 spec
  ├── 准备 50-100 条 RunConfig 格式的训练任务（内部 bug 库）
  ├── 用基础 Qwen3.6-27B 通过 collect_rollouts.py 跑预检评估
  ├── 统计 resolved rate：
  │   · > 10% → 直接 RL
  │   · 3-10% → 先收集成功轨迹做 SFT，再 RL
  │   · < 3%  → 任务太难，换更简单任务集后重试
  └── 确认 cloud_run.py → RunResult 格式 → agentcore_rl.py reward 链路正确

第 1 周：环境 + 连通性验证
  ├── 搭建 Qwen3.5/3.6-4B 的 slime 训练环境（验证 Megatron+SGLang colocate 能跑）
  ├── 验证 AgentCore Runtime 已部署（setup_agentcore_runtime.py 已跑通）
  ├── 验证 shim（EC2 Head 节点）← AgentCore microVM 的 VPC 内网连通性
  └── 准备 RunConfig 格式的 slime JSONL 训练集（基于预检通过的任务）

第 2 周：shim + rollout_fn（关键）
  ├── 实现 shim.py（重点：SGLang /generate + TurnRecord 记录 + --sglang-tool-call-parser）
  ├── 实现 rollout_fn.py
  │   · HARNESS_TOOLS_SCHEMA = all_tools_openai()（从 harness 导入）
  │   · RolloutClient.invoke_async + pop_session_split
  │   · 超时/空 segments → ABORTED sample
  ├── Smoke test：单条任务用 train_async.py 跑通一个完整 rollout+training step
  │   验证：shim 收到 AgentCore 发来的 /v1/chat/completions 请求
  │   验证：S3 结果中有 rewards 字段
  │   验证：TurnRecord 的 loss_mask 和 log_probs 正确
  │   验证：dynamic_sampling_filter 过滤掉全零 reward 的组
  └── 验证 reward 非零（用简单任务，confirm resolved rate > 0）

第 3 周：扩到 Qwen3.6-27B + 正式训练
  ├── 下载 Qwen3.6-27B HF checkpoint
  ├── 运行 convert_hf_to_torch_dist.py 生成 _torch_dist 格式（--ref-load 用）
  ├── 用 launch.sh 跑第一个 100-step 训练，验证 reward 有上升趋势
  └── 监控 GPU 利用率，确认 train_async.py 的 rollout/training 重叠有效

第 4 周+：迭代与优化
  ├── 用 bench.py 测 resolved rate（checkpoint 转回 HF 格式后直接测）
  ├── 如 GPU 利用率仍低（< 50%），切换到 fully_async_rollout（V2）
  ├── 调整奖励权重（quality_det vs task）
  └── 增加训练任务多样性（内部 bug 库 + SWE-bench）
```

---

## 附：相关项目路径

| 路径（相对于各自仓库根目录） | 说明 |
|---|---|
| `examples/coding_agent_rl/` | slime 参考实现（Claude Code + E2B + Anthropic API）|
| `slime/agent/trajectory.py` | TurnRecord / TokenSegment / merge_turns（直接复用）|
| `slime/agent/parsing.py` | 模型输出解析（直接复用）|
| `src/query/model_routes.py` | harness：Qwen3.6-27B 路由配置（核心集成点）|
| `rl_data/grpo/reward.py` | harness：奖励函数（AgentCore 内部调用）|
| `src/services/sandbox/agentcore_rl.py` | harness：AgentCore RL entrypoint（已有实现）|
| `rl_data/collect_rollouts.py` | harness：`RolloutClient` 用法参考模板 |
| `scripts/setup_agentcore_runtime.py` | harness：AgentCore Runtime 部署脚本 |
| `https://github.com/awslabs/agentcore-rl-toolkit` | AgentCore RL Toolkit 开源 SDK |
