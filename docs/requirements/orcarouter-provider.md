# OrcaRouter 可选 Provider 接入需求

状态：待实施。记录日期：2026-09-08。

本文记录下一轮需求，不代表项目已经支持 OrcaRouter、加入合作计划或获得合作收益。当前应用仍默认通过 DeepSeek 直连。

## 目标与范围

让用户自主选择 DeepSeek 直连或 OrcaRouter 网关，复用现有文档索引、知识图谱和问答能力。本期只接入文本 LLM；本地 Contriever 嵌入及 MPS/CPU 行为继续沿用。

技术接入与商业合作分别推进。官方 OSS 计划的申请、协议接受、收款账户配置由 maintainer 单独决定，本需求不包含这些操作。

## 已核对的接口与现状

- OrcaRouter 的 OpenAI 兼容基础地址为 `https://api.orcarouter.ai/v1`，模型采用 `vendor/model` 标识。[SDK 文档](https://docs.orcarouter.ai/compatibility/openai-sdk)
- 项目固定的 HippoRAG 提交已经包含 `OrcaRouterLLM`，使用 `ORCAROUTER_API_KEY`，并通过 `orcarouter/<vendor/model>` 选择后端；发送请求时去掉最外层前缀。实施时优先复用并测试该适配器。[固定版本源码](https://github.com/OSU-NLP-Group/HippoRAG/blob/1438aba3fc44ff10573e5a5e1e7cc3c7f9794aff/src/hipporag/llm/orcarouter_llm.py)
- 当前应用在 `backend.py` 中强制校验 DeepSeek 密钥、设置 DeepSeek 参数，因此不能仅凭上游已有适配器就宣称本项目已完成接入。
- 官方文档列有 `-free` 模型，但目录和可用容量可能变化，并有频率、每日额度和请求大小限制；免费请求的 429 不一定能靠重试解决。[免费模型文档](https://docs.orcarouter.ai/routing/free-models)

## 功能需求

### 1. 显式选择 Provider

- 基础版、高级版和单次问答入口统一支持 `--provider deepseek|orcarouter`。
- 拟增加 `LLM_PROVIDER` 环境变量，默认 `deepseek`，保持现有用户的运行方式。
- 配置优先级为命令行、shell 环境变量、项目 `.env`、内置默认值。
- 启动时显示实际 Provider 和模型；不输出密钥。

拟议配置（尚未实现，不应直接当作当前使用说明）：

| 配置 | 用途 |
| --- | --- |
| `LLM_PROVIDER` | 选择 Provider，默认 `deepseek` |
| `ORCAROUTER_API_KEY` | OrcaRouter 专用密钥 |
| `ORCAROUTER_API_URL` | 默认 `https://api.orcarouter.ai/v1` |
| `ORCAROUTER_MODEL` | 明确指定网关模型 ID；首期必填，避免隐式选择收费模型 |

`--model` 应覆盖所选 Provider 的模型配置。网关用户输入 `vendor/model`，适配层负责 HippoRAG 前缀转换，并拒绝空值或错误的重复前缀。

### 2. 独立凭据与请求参数

- 选择 OrcaRouter 时只要求 OrcaRouter 密钥，不要求 DeepSeek 密钥；反向同理。
- 密钥只能发送到当前选定 Provider 的客户端，不能沿用另一 Provider 的凭据。
- 不把目前 DeepSeek 专用的 `thinking` 参数无条件传给其他模型。按模型能力核对思考模式、输出 token 参数和结构化输出；首期验证一个明确的文本模型。
- 知识抽取、重排和问答均使用所选 Provider。保留超时、有限重试和资源释放。
- 成功响应必须包含有效文本及 HippoRAG 所需的 usage；缺失时给出明确错误，不生成看似成功的空答案。[推理参数文档](https://docs.orcarouter.ai/advanced/reasoning)

### 3. 索引与缓存隔离

- Provider、API 端点、模型或影响抽取的参数变化时，不得混用已有索引及 LLM 缓存。
- 新增 Provider 的默认索引目录应与 DeepSeek 隔离；用户指定同一目录且身份不匹配时明确提示重建。
- 同配置、同文档重启后继续复用索引，不重复调用抽取 API。

### 4. 免费模型与失败处理

- 允许用户显式选择目录中的免费模型；文档链接到实时目录，不承诺某型号永久免费或无限量。
- 区分认证失败、模型不可用、限流、每日额度耗尽和请求过长，给出下一步操作提示。
- 免费模型失败时不得静默切换为付费模型、其他 Provider 或自动路由。
- 请求过长不能无限重试；提示减少文档段落或检索上下文。

### 5. 合作归因与信息披露

- 在官方明确提供归因协议后，再确定是否需要项目 ID、请求头或推荐链接；不要猜测字段或填入未经确认的归因值。
- 合作归因不得包含文档正文、问题、用户个人信息或用户 API 密钥。
- 如果正式加入合作计划，应在相关使用说明中说明 maintainer 可能获得分成，且 Provider 始终由用户主动选择。
- 技术接入不依赖分成开通，缺少合作标识不能阻止用户正常使用自己的 OrcaRouter 密钥。

### 6. 文档与诊断

- 更新 README、`.env.template`、`--help` 和 `test_env.py`，提供两个 Provider 的独立配置示例。
- 说明选择网关后，相关文本将发送至 OrcaRouter 及实际模型服务方。
- 离线诊断仅检查配置和依赖，不访问模型目录、不调用 API。
- 提供可手动执行的最小在线验证步骤；记录真实验证过的型号、日期和限制。

## 验收标准

- [ ] 原有 DeepSeek 默认流程和测试全部通过。
- [ ] 两种入口及 `--query` 均遵守 Provider、模型与配置优先级。
- [ ] 只有所选 Provider 密钥时能初始化；缺失密钥在下载模型、建立索引或请求 API 前报错。
- [ ] API 边界测试验证 URL、密钥、模型前缀、思考参数和 token 参数正确，不发生跨 Provider 泄漏。
- [ ] 两个 Provider 的离线集成测试覆盖索引、问答、缓存复用、索引身份不匹配及初始化失败后的释放。
- [ ] 验证 401/403、429、模型不可用、缺少 usage、空答案等失败路径；无隐式付费回退。
- [ ] 免费模型额度或长度错误不会触发无效的无限重试。
- [ ] CPU CI 及可用实体 Mac 上的 MPS 测试通过。
- [ ] 使用 maintainer 提供的 OrcaRouter 测试密钥完成最小在线索引和问答，再将功能状态改为“可用”。

## 合作事项待确认

- 官方合作归因机制、项目标识和验收要求。
- 邀请中提到的消费分成方案：比例、有效期、适用消费范围、结算条件及退出机制，须以正式条款为准。
- 免费模型的当期目录、账户资格、限额与输出能力是否适合 OpenIE 和问答。
- 如对方提供 integration/PR，取得具体链接后审查代码、依赖和数据流，再决定合入。

[OSS 合作申请页](https://www.orcarouter.ai/zh-CN/built-with)本次未能成功读取，合作条款尚未核实。本文不复制邀请邮件、不发布联系人信息，也不将邮件中的规模或合作项目宣传作为已验证事实。
