# HippoRAG 演示应用 (M3芯片优化版 + DeepSeek)

这是一个针对Mac M3芯片优化的HippoRAG应用示例，使用DeepSeek Chat模型进行文档索引、知识关联和智能问答。

## 简介

HippoRAG是一个受人类海马体（负责长期记忆）启发的检索增强生成框架，能够帮助大型语言模型持续整合外部文档中的知识。该框架使用知识图谱和个性化PageRank技术，提升模型的关联性和意义构建能力。

本演示应用经过特别优化，可在Mac M3芯片上高效运行，并使用DeepSeek Chat模型进行问答。相比原始版本，本项目解决了以下问题：

- 解决了M3芯片上的兼容性问题
- 优化了内存使用，防止在资源受限环境中崩溃
- 改进了用户界面，提供更友好的交互体验
- 增加了对DeepSeek API的支持，替代OpenAI API

## 功能特点

- **文档索引和向量嵌入存储**：自动将文档转换为向量表示并存储
- **基于知识图谱的关联检索**：构建实体和关系的知识图谱，支持多跳推理
- **智能问答**：使用DeepSeek Chat API进行自然语言问答
- **Mac M3芯片MPS加速**：利用Metal Performance Shaders加速计算
- **内存使用监控**：实时跟踪内存占用，适用于资源受限环境
- **问答历史管理**：记录和导出问答历史，便于后续分析
- **美化的用户界面**：使用emoji和格式化输出提升用户体验

## 系统要求

- macOS 11.0+（M系列芯片优化）
- Python 3.10+
- Conda 环境管理
- 互联网连接（用于API调用）
- DeepSeek API密钥

## 快速开始

### 1. 环境设置

首先克隆本仓库并运行环境设置脚本：

```bash
git clone https://github.com/您的用户名/HippoRAG-M3.git
cd HippoRAG-M3
chmod +x setup_env.sh
./setup_env.sh
```

这将创建名为`hipporag`的conda环境，并安装所有必要依赖。

### 2. 配置API密钥

复制环境变量模板并编辑：

```bash
cp .env.template .env
```

然后编辑`.env`文件，添加您的DeepSeek API密钥：

```
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_API_URL=https://api.deepseek.com/v1
OPENAI_API_KEY=your_openai_api_key_here  # 用于HippoRAG内部调用
```

### 3. 准备文档

将您想要索引的文档放入`data/documents.json`文件中，格式如下：

```json
[
  {
    "id": "doc1",
    "content": "这是第一篇文档的内容..."
  },
  {
    "id": "doc2",
    "content": "这是第二篇文档的内容..."
  }
]
```

### 4. 运行应用

激活环境并运行基本应用：

```bash
conda activate hipporag
python app.py
```

或运行高级版本（带更多功能）：

```bash
python advanced_app.py
```

## 应用版本说明

### 基础版 (app.py)

- 简单直观的命令行界面
- 文档加载、索引和问答功能
- 美化的输出格式
- 适合快速测试和演示

### 高级版 (advanced_app.py)

- 包含基础版的所有功能
- 增加历史记录管理功能
- 实时内存使用监控
- 更完善的错误处理
- 模块化设计，便于扩展
- 支持"history"命令查看历史问答

## 文件结构

```
@hipporag/
├── app.py               # 基本应用
├── advanced_app.py      # 高级版应用（带更多功能）
├── data/
│   └── documents.json   # 示例文档集
├── outputs/             # 索引和知识图谱保存目录
│   ├── deepseek-chat_facebook_contriever/  # 模型特定输出
│   │   ├── chunk_embeddings/              # 文档块嵌入
│   │   ├── entity_embeddings/             # 实体嵌入
│   │   ├── fact_embeddings/               # 事实嵌入
│   │   └── graph.graphml                  # 知识图谱
│   ├── llm_cache/                         # LLM响应缓存
│   └── openie_results_ner_deepseek-chat.json  # 开放信息抽取结果
├── .env                 # 环境变量配置
├── .env.template        # 环境变量模板
├── setup_env.sh         # 环境设置脚本
├── test_env.py          # 环境测试脚本
└── README.md            # 本文档
```

## 示例问题

启动应用后，您可以尝试以下问题：

1. "什么是人工智能？"
2. "HippoRAG和传统RAG有什么区别？"
3. "知识图谱和PageRank如何在HippoRAG中结合？"
4. "深度学习和机器学习的关系是什么？"
5. "解释一下大型语言模型的工作原理"

## 技术细节

### M3芯片优化

本应用针对Apple Silicon M3芯片进行了以下优化：

1. **MPS加速**：使用PyTorch MPS后端（Metal Performance Shaders）代替CUDA
2. **轻量级模型**：使用`facebook/contriever`作为默认嵌入模型，平衡性能和资源消耗
3. **内存管理**：实现内存使用监控，防止OOM错误
4. **进程管理**：使用`spawn`方法解决多进程问题
5. **批处理优化**：移除不必要的批处理参数，提高稳定性

### DeepSeek API集成

本应用使用DeepSeek Chat模型代替OpenAI模型，优势包括：

1. **本地化服务**：提供更好的中文支持和本地化服务
2. **API兼容性**：与OpenAI API格式兼容，便于集成
3. **成本效益**：提供更经济的API调用选项
4. **多语言支持**：优秀的中英文双语能力

### HippoRAG核心技术

1. **知识图谱构建**：使用开放信息抽取（OpenIE）从文档中提取实体和关系
2. **个性化PageRank**：根据查询动态调整图谱节点重要性
3. **多模态检索**：结合实体、事实和文档块的检索结果
4. **上下文增强**：通过知识图谱关联扩展检索上下文

## 常见问题解答

### Q: 为什么选择DeepSeek而不是OpenAI？
A: DeepSeek提供了更好的中文支持，API格式与OpenAI兼容，且价格更经济实惠。

### Q: 如何添加自己的文档？
A: 将文档按JSON格式添加到`data/documents.json`文件中，每次运行应用时会自动索引。

### Q: 应用占用内存过多怎么办？
A: 可以尝试使用更小的嵌入模型，或减少同时处理的文档数量。高级版应用提供内存监控功能。

### Q: 如何导出问答历史？
A: 高级版应用会自动记录历史，可以使用"history"命令查看。

## 进一步探索

1. **添加更多文档**：扩展知识库，提高问答质量
2. **尝试不同模型**：测试其他嵌入模型和LLM
3. **开发Web界面**：使用Streamlit或Flask创建图形界面
4. **实现文档上传功能**：支持动态添加新文档
5. **添加多语言支持**：扩展到更多语言的文档处理

## 参考资源

- [HippoRAG官方仓库](https://github.com/OSU-NLP-Group/HippoRAG)
- [HippoRAG论文](https://arxiv.org/abs/2405.14831)
- [DeepSeek AI官方网站](https://deepseek.com/)
- [DeepSeek API文档](https://api-docs.deepseek.com/)
- [PyTorch MPS文档](https://pytorch.org/docs/stable/notes/mps.html)

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request，共同改进这个项目！

## 致谢

感谢OSU-NLP-Group开发的原始HippoRAG框架，以及DeepSeek AI提供的API服务。
