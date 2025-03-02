#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境测试脚本 - 检查安装情况和依赖项
"""
import os
import sys
import platform
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_separator(title):
    """打印分隔符"""
    print("\n" + "=" * 50)
    print(f" {title} ".center(50, "="))
    print("=" * 50)

def check_python():
    """检查Python版本"""
    print_separator("Python环境")
    
    print(f"Python版本: {platform.python_version()}")
    print(f"Python路径: {sys.executable}")
    print(f"平台信息: {platform.platform()}")
    
    if "arm64" in platform.platform().lower() and "darwin" in platform.platform().lower():
        print("✅ 检测到Apple Silicon架构 (M系列芯片)")
    else:
        print("⚠️ 未检测到Apple Silicon架构")

def check_torch():
    """检查PyTorch安装"""
    print_separator("PyTorch")
    
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        
        # 检查MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("✅ MPS可用 (Apple Metal性能着色器)")
        else:
            print("⚠️ MPS不可用")
            
        # 检查设备
        print(f"可用设备: CPU", end="")
        if torch.cuda.is_available():
            print(", CUDA", end="")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print(", MPS", end="")
        print()
        
    except ImportError:
        print("❌ PyTorch未安装")

def check_hipporag():
    """检查HippoRAG安装"""
    print_separator("HippoRAG")
    
    try:
        import hipporag
        print(f"HippoRAG已安装")
    except ImportError:
        print("❌ HippoRAG未安装")
        return
    
    # 检查必要依赖
    deps = ["transformers", "networkx", "sklearn", "pandas"]
    for dep in deps:
        try:
            if dep == "sklearn":
                import sklearn
            else:
                __import__(dep)
            print(f"✅ {dep} 已安装")
        except ImportError:
            print(f"❌ {dep} 未安装")

def check_env_file():
    """检查环境变量文件"""
    print_separator("环境配置")
    
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    
    if os.path.exists(env_path):
        print(f"✅ .env文件存在: {env_path}")
        
        # 检查API密钥
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
            
            if os.getenv("DEEPSEEK_API_KEY"):
                print("✅ DEEPSEEK_API_KEY已设置")
            else:
                print("❌ DEEPSEEK_API_KEY未设置")
                
            if os.getenv("DEEPSEEK_API_URL"):
                print(f"✅ DEEPSEEK_API_URL已设置: {os.getenv('DEEPSEEK_API_URL')}")
            else:
                print("⚠️ DEEPSEEK_API_URL未设置，将使用默认值")
                
        except ImportError:
            print("❌ python-dotenv未安装，无法验证API密钥")
    else:
        print(f"❌ .env文件不存在: {env_path}")

def check_data():
    """检查数据文件"""
    print_separator("数据文件")
    
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "documents.json")
    
    if os.path.exists(data_path):
        print(f"✅ 示例数据文件存在: {data_path}")
        
        try:
            import json
            with open(data_path, "r", encoding="utf-8") as f:
                docs = json.load(f)
            print(f"✅ 成功加载{len(docs)}篇文档")
        except Exception as e:
            print(f"❌ 数据文件加载失败: {e}")
    else:
        print(f"❌ 示例数据文件不存在: {data_path}")

def main():
    """主函数"""
    print_separator("HippoRAG环境测试")
    
    check_python()
    check_torch()
    check_hipporag()
    check_env_file()
    check_data()
    
    print_separator("测试完成")
    print("如果发现问题，请查看README.md获取解决方案")
    print("或者重新运行 ./setup_env.sh 脚本安装依赖")

if __name__ == "__main__":
    main()
