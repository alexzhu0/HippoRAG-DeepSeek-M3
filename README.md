# HippoRAG 演示应用 (M3芯片优化版 + DeepSeek)

这是一个针对Mac M3芯片优化的HippoRAG应用示例，使用DeepSeek Chat模型进行文档索引、知识关联和智能问答。

## 简介

HippoRAG是一个受人类海马体（负责长期记忆）启发的检索增强生成框架，能够帮助大型语言模型持续整合外部文档中的知识。该框架使用知识图谱和个性化PageRank技术，提升模型的关联性和意义构建能力。

本演示应用经过特别优化，可在Mac M3芯片上高效运行，并使用DeepSeek Chat模型进行问答。

## 功能特点

- 文档索引和向量嵌入存储
- 基于知识图谱的关联检索
- 智能问答（使用DeepSeek Chat API）
- Mac M3芯片MPS加速支持
- 内存使用监控（适用于资源受限环境）
- 问答历史导出功能

## 快速开始

### 1. 环境设置

首先运行环境设置脚本：

```bash
./setup_env.sh
```

这将创建名为`hipporag`的conda环境，并安装所有必要依赖。

### 2. 配置API密钥

编辑`.env`文件，添加您的DeepSeek API密钥：

```
DEEPSEEK_API_KEY=your_api_key_here
```

如果需要，也可以指定自定义的API URL：

```
DEEPSEEK_API_URL=https://api.deepseek.com/v1
```

### 3. 运行应用

激活环境并运行基本应用：

```bash
conda activate hipporag
python app.py
```

或运行高级版本：

```bash
python advanced_app.py
```

## 文件结构

```
@hipporag/
├── app.py               # 基本应用
├── advanced_app.py      # 高级版应用（带更多功能）
├── data/
│   └── documents.json   # 示例文档集
├── outputs/             # 索引和知识图谱保存目录
├── .env                 # 环境变量配置
├── .env.template        # 环境变量模板
├── setup_env.sh         # 环境设置脚本
└── README.md            # 本文档
```

## 示例问题

启动应用后，您可以尝试以下问题：

1. "什么是人工智能？"
2. "HippoRAG和传统RAG有什么区别？"
3. "知识图谱和PageRank如何在HippoRAG中结合？"

## M3芯片优化说明

本应用针对Apple Silicon M3芯片进行了以下优化：

1. 使用PyTorch MPS加速（Metal Performance Shaders）
2. 使用轻量级嵌入模型（sentence-transformers/all-MiniLM-L6-v2）
3. 批量处理索引以优化内存使用
4. 增加内存监控功能，防止资源耗尽

## DeepSeek Chat模型

本应用使用DeepSeek Chat模型代替OpenAI模型，优势包括：

1. 提供本地化的AI服务
2. 支持中文等多语言的优秀能力
3. 与标准OpenAI接口兼容，便于集成

## 进一步探索

1. 添加自己的文档到data目录
2. 尝试不同的嵌入模型
3. 调整批处理大小以找到最佳性能平衡点
4. 开发Web界面（如使用Streamlit）

## 参考资源

- [HippoRAG官方仓库](https://github.com/OSU-NLP-Group/HippoRAG)
- [HippoRAG论文](https://arxiv.org/abs/2405.14831)
- [DeepSeek AI官方网站](https://deepseek.com/)
- [PyTorch MPS文档](https://pytorch.org/docs/stable/notes/mps.html)
