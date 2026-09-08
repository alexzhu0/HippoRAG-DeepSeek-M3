#!/usr/bin/env python3
"""List this demo's supported mean-pooling encoders without importing ML packages."""


def main():
    print("本演示支持的 Contriever 系列嵌入模型（非 HippoRAG 完整模型目录）：")
    print("- facebook/contriever：兼容原项目的默认模型")
    print("- facebook/mcontriever：多语言模型，可用于中文实验")
    print("通过 --embedding-model 配置；首次运行需要下载模型。")
    print("也可传入本地 Contriever 格式模型目录；其他架构需要适配器。")


if __name__ == "__main__":
    main()
