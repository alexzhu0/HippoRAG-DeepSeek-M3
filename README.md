# HippoRAG + DeepSeek（Apple Silicon）

通过本地 Contriever 嵌入、HippoRAG 知识图谱检索和 DeepSeek API，对自己的文档进行问答。提供基础和高级两个命令行入口，支持 Apple Silicon MPS、CPU 和可用的 CUDA 设备。

> 文档的知识抽取与问答会把相关文本发送给配置的 DeepSeek API；嵌入模型在本机运行。首次索引会产生 API 费用，同一索引中的未变化文档可复用缓存。

## 本次维护更新

- 修复高级版未初始化、未索引就进入问答的问题。
- 统一两个入口的结果解析，支持 HippoRAG 2 的 `QuerySolution` 元组。
- 同时接受字符串文档和 `{ "content": "..." }`，校验空值并按内容去重。
- 显式控制嵌入设备；FP32、默认批大小 8，每批结果立即转回 CPU，降低加速器上的临时内存占用。
- 完整安装依赖，新增离线测试、环境诊断和 Linux/macOS CI。
- 高级版可用 `history` 查看本次会话记录、`export` 导出含参考资料的 CSV。

## 环境要求与版本选择

建议使用 **Python 3.11**；测试配置覆盖 Python 3.10–3.12。Apple Silicon 建议使用 macOS 14 或更新版本；CPU 模式也可用于 Linux。需要 Git、可访问 GitHub/PyPI/Hugging Face 的网络，以及 DeepSeek API 密钥。

本项目固定使用上游 [HippoRAG 提交 `1438aba`](https://github.com/OSU-NLP-Group/HippoRAG/tree/1438aba3fc44ff10573e5a5e1e7cc3c7f9794aff)（版本标识 `2.0.0a5`，预发布源码快照）。选用它的原因是 PyPI `2.0.0a4` 强制安装 Linux 专用的 vLLM，而该提交将其改为可选依赖，并增加索引配置校验。**不要直接用 `pip install --upgrade hipporag` 替换该版本。** 后续切换上游提交应重新运行本项目测试。

PyTorch `2.5.1`、Transformers `4.45.2` 等由该上游提交固定，本次没有强行突破其兼容约束。其余依赖的范围见 [requirements.txt](requirements.txt)。这些范围允许兼容更新，并不是所有传递依赖的完整锁文件。

## 快速开始

```bash
git clone https://github.com/alexzhu0/HippoRAG-DeepSeek-M3.git
cd HippoRAG-DeepSeek-M3
PYTHON_BIN=python3.11 ./setup_env.sh
source .venv/bin/activate
```

脚本创建项目内 `.venv`，安装完整依赖并执行离线检查；可重复执行，不会覆盖已有 `.env`。如果默认 `python3` 已是受支持版本，可直接运行 `./setup_env.sh`。已有 Conda 用户也可在 Python 3.10–3.12 环境中手动安装：

```bash
python -m pip install -r requirements.txt
cp .env.template .env  # 仅在尚未创建 .env 时执行
```

编辑 `.env`，填写：

```dotenv
DEEPSEEK_API_KEY=你的DeepSeek密钥
DEEPSEEK_API_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
EMBEDDING_DEVICE=auto
EMBEDDING_BATCH_SIZE=8
EMBEDDING_MODEL=facebook/contriever
```

只需一把 DeepSeek 密钥，无需 OpenAI 账户。上游使用 OpenAI 兼容客户端，因此适配器在构造客户端时临时映射 `OPENAI_API_KEY`，随即恢复原环境变量。不要在别的线程同时构造依赖该环境变量的第三方客户端。已设置的 shell 环境变量优先于 `.env`，命令行参数优先于两者。

```bash
python test_env.py                  # 检查依赖、示例数据和密钥是否填写，不调用 API
python app.py                      # 基础交互问答
python advanced_app.py             # 增加 history / export 命令
python app.py --query "什么是 HippoRAG？"
```

首次运行会从 Hugging Face 下载模型，并调用 DeepSeek 建立索引。`--help` 不会加载模型、建立索引或校验密钥。默认数据和输出路径以项目目录为基准，因此可在其他目录执行脚本。

## 文档格式与索引

默认读取 `data/documents.json`，以下格式均可，允许混合：

```json
[
  "HippoRAG 使用知识图谱关联文档中的实体。",
  {"id": "doc2", "content": "个性化 PageRank 可用于图谱检索。"}
]
```

`content` 必须是非空字符串；`id` 仅用于你自己的数据管理，不传给 HippoRAG。相同内容会去重并保留首次出现顺序。空数组、空文本或无效条目会在加载模型前报错。

每条记录建议是一段完整但较短的文本。当前本地编码器最多编码 **512 个 tokenizer token**，超长部分会被截断；请先将长文档切分为短段落，避免后半部分无法参与向量表示。这里不提供 PDF/Word 解析或自动分块。

```bash
python advanced_app.py --documents /path/to/documents.json --save-dir outputs/my-corpus
```

新版本默认写入 `outputs/v2`。旧版本产生的无 manifest 索引不能直接复用，请从原始文档重建；不要把旧图谱或 parquet 复制到新目录。仓库不再跟踪运行生成的索引和缓存。更换模型、API 端点或数据集时使用新的 `--save-dir`；配置不一致时，上游会拒绝混用已有索引。

同一目录下新增文档可增量索引。**从 JSON 删除记录不会自动删除旧索引中的文档**；需要精确同步删除时，请使用新的输出目录重建。

## 设备与内存

```bash
python app.py --device mps --batch-size 4
python app.py --device cpu --batch-size 2
python list_models.py
python app.py --embedding-model facebook/mcontriever --save-dir outputs/multilingual
```

- `auto` 优先选择可用的 MPS，其次 CUDA，最后 CPU；显式指定不可用设备会报错。
- 本项目适配 Contriever 系列的 mean pooling，也支持本地同格式模型目录，不是任意 Hugging Face 模型的通用适配器。
- MPS 运算失败时可显式切换 `--device cpu`。内存紧张时先降低批大小，或缩小文档集。
- 批大小限制临时张量，不会消除模型、图谱及向量存储本身的内存需求。高级版显示的是进程 RSS，不是精确 GPU 内存或 OOM 保护。
- 默认保留原来的 `facebook/contriever`。`facebook/mcontriever` 可用于多语言实验；本次未做中文检索质量或 M3 性能基准，因此不声称量化提速或质量提升。

## 历史与命令行

高级版的记录保存在当前进程内，退出后不保留；需要保存时输入 `export`。默认写入当前工作目录的 `qa_history.csv`，存在同名文件时会覆盖，可提前指定目标：

```bash
python advanced_app.py --history-file /path/to/session.csv
```

CSV 使用 UTF-8 BOM，包含时间、问题、回答、JSON 格式的参考资料和内存使用。`exit` / `quit` / Ctrl-D 正常退出，Ctrl-C 返回 130。单次 `--query`、初始化或索引失败返回非零退出码，适合脚本调用。

## 开发与验证

```bash
python -m pip install -r requirements-dev.txt
python -m pip check
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python test_env.py --offline
bash -n setup_env.sh
```

测试在本地生成小型 Transformer 模型，实际执行编码、图谱和向量索引、结果解析、持久化与重新加载。外部 LLM 响应在 API 边界替换为固定测试响应，测试不下载模型、不需要真实密钥，也不评价 DeepSeek 生成质量。CI 在 Linux Python 3.10/3.12 和 macOS Python 3.11 上执行 CPU 检查（`pytest -m "not mps"`）。GitHub 托管 macOS 的 [MPS 限制](https://github.com/actions/runner-images/issues/11899)可能导致原生算子崩溃，因此 MPS 硬件测试标记为 `mps`；在实体或自托管 Mac 上运行 `python -m pytest -m mps -q`。默认本地 `pytest` 仍包含 MPS 测试，设备不可用时跳过。

## 参考

- [HippoRAG 上游与论文](https://github.com/OSU-NLP-Group/HippoRAG)
- [DeepSeek API 文档](https://api-docs.deepseek.com/)
- [PyTorch MPS 文档](https://pytorch.org/docs/stable/notes/mps.html)

本项目沿用原 README 声明的 MIT 许可。上游代码和模型遵循各自许可证。
