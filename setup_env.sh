#!/bin/bash
# HippoRAG环境设置脚本 - M3芯片优化版 + DeepSeek

# 创建conda环境
echo "创建conda环境: hipporag..."
conda create -n hipporag python=3.10 -y

# 激活环境
echo "激活环境..."
eval "$(conda shell.bash hook)"
conda activate hipporag

# 安装基础依赖
echo "安装基础依赖..."
pip install python-dotenv pandas psutil requests

# 安装PyTorch (Apple Silicon优化版 - 稳定版)
echo "安装PyTorch (M3芯片优化版)..."
pip install --upgrade pip
pip install torch torchvision torchaudio

# 检查torch是否安装成功
echo "验证PyTorch安装..."
python -c "import torch; print(f'PyTorch已安装: {torch.__version__}, MPS可用: {torch.backends.mps.is_available()}')"

# 安装Transformers
echo "安装Transformers..."
pip install transformers sentence-transformers

# 安装HippoRAG依赖
echo "安装HippoRAG依赖..."
pip install networkx scikit-learn pyyaml fsspec

# 使用pip直接安装HippoRAG
echo "安装HippoRAG..."
pip install hipporag --no-deps

echo "===================================="
echo "环境设置完成！使用说明："
echo "1. 在.env文件中设置您的DeepSeek API密钥"
echo "2. 运行应用：python app.py"
echo "3. 高级版应用：python advanced_app.py"
echo "===================================="
