# 论文检索 + 文献综述 Agent

输入一个研究问题 → Agent 自动检索 arXiv / Semantic Scholar → 用 LLM 对候选论文相关性打分排序 → 用户确认后下载全文 → 基于 [PaperQA2](https://github.com/Future-House/paper-qa) 做证据收集，生成带精确引用（页码级）的文献综述。

## 核心特性

- **两阶段 Agent 流程**：搜索阶段先把候选论文交给用户确认，避免把预算和时间浪费在无关论文的下载/索引上
- **真正的搜索 agent，不是写死的流水线**：搜索阶段由 Claude Sonnet 通过 tool-use 循环自主决策——自己选关键词、决定搜哪个源、判断候选池够不够好、要不要换个说法再搜一轮，而不是"提炼关键词→固定搜一次→固定打分"的单向脚本；打分这一步是 agent 自己发起的工具调用（转给便宜模型 Claude Haiku 执行），昂贵模型只负责编排和最终生成。迭代轮数、token 预算都有硬上限兜底，工具报错也会喂回给 agent 自己决定怎么应对，而不是代码里预先写好的降级路径。每轮决策前 agent 会先写一句中文"计划/判断"再调用工具，实时记进日志（"[Agent 计划] ..."），推理过程可见，不是黑盒
- **带引用证据可追溯**：最终答案不仅有参考文献列表，还能展开看每一条论断具体来自哪篇论文的哪一页、原始证据摘录是什么
- **引用幻觉自查，确定性检查 + LLM-as-judge 两层**：综述生成后先跑一道零成本的确定性检查——引用标记（如"(Smith2021 pages 2-3)"）是纯字符串，直接跟证据池的来源字段做集合比对，不存在的引用瞬间抓出来，不调用模型、没有随机性；再跑 LLM 判断"来源是真的，但证据内容其实不支持这句话"这类需要语义理解的问题（纯代码做不到）。两层结果合并展示，前端能看出每条问题是"确定性检查"还是"LLM判断"标出来的。核查本身有重试，两次都失败时前端会显式提示"未能完成"，跟"没查"和"查了没问题"区分开，不会被悄悄隐藏掉
- **Prompt injection 防御**：论文标题/摘要（来自 arXiv/S2）和证据片段（来自下载的 PDF 原文）都是外部不可信内容，会直接进入喂给 LLM 的 prompt——所有这类内容进 prompt 前都用明确分隔符包裹并声明"这是数据不是指令"，另加一道轻量启发式扫描作为可见信号（只标记不拦截，避免误伤真讨论"prompt injection"这个话题的论文）。实测跑了 10 次真实注入攻击调用（含 3 组防护前后配对对照）验证效果
- **Prompt caching**：搜索 agent 多轮 tool-use 循环里，system prompt + 工具定义是每轮都重复发送的静态前缀，累积的对话历史也是逐轮增长、前缀不变的——两处都打了 `cache_control` 断点，动态断点随对话增长每轮往后移。实测一次真实搜索（7次工具调用）缓存命中 13313 tokens、写入 8438 tokens，换算下来这次对话的有效输入成本降低了约 40-50%，跟已有的耗时/花费/token 可观测性直接打通——日志里就能看到每次的缓存命中数字
- **测试与 eval**：`backend/tests/` 下有两类：`test_prompt_injection.py` 是 prompt injection 防御的 pytest 回归套件（6种payload、8个防御断言+1个启发式命中率测量）；`eval_verifier.py` 是给引用自查功能做的合成注入 eval——在一篇真实生成的综述里故意植入已知的引用错误，量化 verifier 的 recall 和假阳性率，不需要人工标注 ground truth。这个 eval 在真正跑起来后意外挖出一个证据池较大时约 70% 单次失败率的真实可靠性 bug
- **回归基线对比**：固定阈值的测试只能抓"彻底坏了"，抓不住"悄悄变差了但还没跌破阈值"这种漂移。`backend/tests/capture_baseline.py` 把 verifier 的 recall/假阳性率、注入防御的检测命中率/攻击得分等指标的真实样本存成基线（滚动窗口，不是单次快照）；`check_regression.py` 重新测一遍，跟基线的历史区间比，只有超出噪声容忍度才报警，而不是精确比对
- **可观测性**：每次生成展示实际耗时、Claude API 花费（美元）、token 用量
- **健壮性**：外部依赖（arXiv/Semantic Scholar API）限流或报错时单独降级，不拖垮整个任务；Semantic Scholar 官方限速 1 请求/秒，用进程级限流器统一节流，不管并发多少任务都不超限；SQLite 落盘使任务历史跨重启保留，服务重启后自动清理僵死任务，历史记录支持删除
- **跨任务向量缓存**：所有任务共享同一个论文库和 paper-qa 索引，同一篇论文被不同任务选中时，第二次直接复用已有的 embedding，不重新解析+摘要（实测节省约30%生成耗时）
- **任务队列 + 进度可视化**：FastAPI 后台任务 + 轮询，前端实时展示搜索/选择/下载/生成各阶段状态

## 架构

```mermaid
flowchart TD
    User(["用户输入研究问题"]) --> Create["POST /api/jobs<br/>创建任务"]

    subgraph Stage1["阶段一 · 搜索 SEARCHING（search_agent.py）"]
        direction TB
        Agent["Claude Sonnet: tool-use 循环<br/>自主决定关键词/搜索源/是否再搜一轮"]
        ToolSearch["工具: search_arxiv / search_semantic_scholar<br/>按标题去重累积候选池，错误当作 tool_result 喂回给模型"]
        ToolScore["工具: score_candidates<br/>转给 Claude Haiku 打分（forced tool-use，非文本解析）"]
        ToolFinish["工具: finish_search<br/>agent 判断候选池够好后主动结束"]
        Cap["硬上限兜底：最多 N 轮迭代 / token 预算超限即停"]
        Agent -->|"调用工具"| ToolSearch --> Agent
        Agent -->|"调用工具"| ToolScore --> Agent
        Agent -->|"调用工具"| ToolFinish
        Cap -.-> Agent
    end

    Create --> Stage1
    Stage1 --> Wait(["AWAITING_SELECTION<br/>暂停，等待用户勾选论文"])
    Wait -->|"POST /api/jobs/:id/select"| Stage2

    subgraph Stage2["阶段二 · 生成 DOWNLOADING → GENERATING"]
        direction TB
        DL["下载用户选中的 PDF 到共享库<br/>papers/library/（已存在则跳过下载）"]
        Index["paper-qa: 增量建索引<br/>已索引过的论文跳过 embedding，只处理新论文"]
        Agent["paper-qa agent_query:<br/>证据收集 + 生成带页码引用的综述"]
        DL --> Index --> Agent
    end

    Stage2 --> Done(["DONE<br/>综述 + 引用 + 证据片段 + 成本/耗时统计"])

    DB[("SQLite jobs.db")]
    Stage1 -. 每次状态变更即落盘 .-> DB
    Stage2 -. 每次状态变更即落盘 .-> DB

    FE["前端每 1.5s 轮询<br/>GET /api/jobs/:id"] -.-> Create
    FE -.-> Wait
    FE -.-> Done
```

PaperQA2 内置的搜索工具只检索**本地已索引的目录**，不会自己联网找论文——因此"搜索 arXiv/Semantic Scholar 并下载 PDF"这一段是本项目自己实现的，PaperQA2 只负责拿到本地 PDF 之后的证据收集与生成引用综述（对应 `agent_query`）。

## 技术栈

| 层 | 选型 |
|---|---|
| 后端 | FastAPI + asyncio 后台任务 |
| LLM | Claude Sonnet 5（生成/关键词提炼）、Claude Haiku 4.5（候选论文相关性打分，控制成本） |
| RAG 引擎 | PaperQA2（`paper-qa` 库），证据收集 + 生成走 `agent_query` |
| Embedding | 本地 `sentence-transformers`，不依赖 OpenAI key |
| 论文来源 | arXiv API、Semantic Scholar Graph API |
| 持久化 | SQLite（标准库 `sqlite3`，无 ORM） |
| 前端 | 纯 HTML / CSS / JS，无框架，轮询获取任务状态 |

## 目录结构

- `backend/`
  - `main.py` — FastAPI 路由（`/api/jobs` 系列：POST/GET/DELETE）
  - `config.py` — 环境变量与全局配置（模型名、并发/迭代上限、路径等）
  - `job_manager.py` — 任务状态机（queued → searching → awaiting_selection → downloading → generating → done/failed）
  - `store.py` — SQLite 持久化，Job 状态变更自动落盘，支持删除
  - `pipeline.py` — 两阶段 pipeline：`run_search_stage` / `run_generate_stage`
  - `search_agent.py` — 搜索阶段的 agent：Claude Sonnet tool-use 循环，自主调用 `search_arxiv` / `search_semantic_scholar` / `score_candidates` / `finish_search`，打分工具内部转给 Claude Haiku（forced tool-use）
  - `search/`
    - `arxiv_search.py` — arXiv API 检索
    - `semantic_scholar_search.py` — Semantic Scholar Graph API 检索（进程级限流，1 请求/秒）
    - `merge.py` — 去重
  - `download.py` — 下载 PDF 到共享库，已存在（按 source+id 稳定命名）则跳过
  - `paperqa_engine.py` — 封装 paper-qa 的 `agent_query` 调用，指向共享库实现增量索引
  - `verifier.py` — 综述生成后的引用自查：确定性字符串比对 + 便宜模型逐条核对，两层
  - `security.py` — 外部内容（论文摘要/证据片段）的 prompt injection 防御：显式数据/指令分隔包装 + 启发式检测（仅提示不拦截）
  - `models.py` — `PaperCandidate` 数据结构
  - `tests/`
    - `test_prompt_injection.py` — 防御的 pytest 回归套件（真实 API 调用，非 mock）
    - `eval_verifier.py` — 引用自查的合成注入 eval，量化 recall / 假阳性率
    - `test_verifier_deterministic.py` — 确定性检查函数的纯单元测试，不调用 API
    - `fixtures/sample_review.json` — eval 用的真实综述样本
    - `baseline_metrics.py` / `capture_baseline.py` / `check_regression.py` — 把关键指标存成滚动基线，新结果超出历史区间才报警（非零退出码可接 CI）
    - `test_check_regression.py` — 比对逻辑本身的纯单元测试
- `frontend/`
  - `index.html` / `style.css` / `app.js` — 候选论文勾选、进度轮询、证据展开、历史任务列表
- `papers/library/` — 所有任务共享的 PDF + paper-qa 索引目录（跨任务复用）
- `jobs.db` — 任务历史持久化（本地文件，不入库）

## API

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/api/jobs` | 提交研究问题，触发搜索阶段 |
| GET | `/api/jobs` | 历史任务列表（摘要） |
| GET | `/api/jobs/{id}` | 单个任务完整状态（日志/候选论文/结果/证据） |
| POST | `/api/jobs/{id}/select` | 提交勾选的论文 id，触发下载+生成阶段 |
| DELETE | `/api/jobs/{id}` | 删除一条历史记录（仅限已完成/已失败的任务，不影响共享库里的 PDF） |

## 快速开始

复制 `.env.example` 为 `.env` 并填写：

- `ANTHROPIC_API_KEY`（必填）
- `SEMANTIC_SCHOLAR_API_KEY`（可选，不填也能跑，只是限速更低）
- 其余项有默认值，一般不用改

```bash
venv/Scripts/python.exe -m uvicorn backend.main:app --reload --port 8000
```

（必须从项目根目录启动，用 `backend.main:app` 而不是 `cd backend` 后 `main:app`——`backend/main.py` 里用的是相对导入 `from . import config`，从 `backend` 目录内直接启动会报 `ImportError: attempted relative import with no known parent package`。`.claude/launch.json` 里配的就是这个正确方式。）

浏览器打开 http://localhost:8000

## 工程亮点

跑通过程中几个靠实测才暴露、值得一提的坑：

- **上游依赖不兼容**：paper-qa 调用 `LiteLLMModel.get_router()`，但它依赖的 `fhlmi` 库新版本已删除该方法（两个包的 PyPI 最新版互不兼容）；二分定位到最后一个兼容版本并锁定依赖
- **模型参数兼容性**：Claude Sonnet 5 拒绝 `temperature` 参数，但 paper-qa 默认总会带上；`litellm.drop_params` 对新模型不生效，改为手动构造不含该字段的 `llm_config`
- **强制结构化输出≠符合 schema**：`tool_choice` 强制工具调用后，模型偶尔把数组字段返回成看起来像数组、实际截断了的字符串——因为仍是合法 JSON，SDK 不报错；加了显式类型校验（`list`/`dict`）而不只是"能否 parse"
- **Eval 逼出真实故障率**：给引用校验做合成注入 eval 时，反复运行意外发现证据池较大（15条）时结构化输出失败率高达 70%；定位到是"prompt 里塞的证据原文量"而非输出复杂度导致，改为只给完整原文给真正引用到的证据、其余只给标识列表，失败率回落到约 20% 的背景水平
- **跨任务缓存复用**：没有重新发明缓存逻辑，而是读 paper-qa 源码发现它自带按文件路径去重的增量索引——只要多个任务共享同一个论文目录，缓存复用就是免费的；实测冷/热启动对比省约 30% 生成耗时
- **Prompt caching 断点管理**：Anthropic 单次请求最多 4 个 `cache_control` 断点，若每轮新增不清理会线性超限；改为维护一个动态断点、每轮后移，真实搜索场景里换算约省 40-50% 有效输入成本
- **对抗测试量化了防御差异**：10 次真实注入攻击调用中均未成功突破，但加了防御后模型会稳定在回复中明确说明"识别到注入并已忽略"，这是一个可审计信号；把一次性测试打包成 pytest 回归套件后第一次跑就挖出启发式扫描器的多个真实漏检（尤其是中文场景）
- **回归基线而非固定阈值**：固定阈值测试抓不住"没跌破阈值但在变差"的漂移，加了 `capture_baseline.py`/`check_regression.py` 把关键指标存成滚动区间，超出历史噪声范围才报警

## 已知局限

- 共享论文库意味着 `agent_query` 检索证据时能看到所有历史任务下载过的论文，不只是本次选中的（已用相关性打分缓解，实测验证过）
- 单机内存态并发控制，未做分布式部署适配
- 没有鉴权，仅适合本地单用户使用
- Prompt injection 的启发式检测只是关键词正则，不是可靠防线——真正的防御是数据/指令分隔包装，检测仅作为可见信号
