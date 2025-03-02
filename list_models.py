#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
列出HippoRAG支持的嵌入模型
"""
import multiprocessing
# 在Mac上使用spawn方法解决多进程问题
multiprocessing.set_start_method('spawn', force=True)

from hipporag.embedding_model import _get_embedding_model_class

# 尝试获取所有支持的嵌入模型类
try:
    embedding_models = _get_embedding_model_class()
    print("支持的嵌入模型：")
    for model_name, model_class in embedding_models.items():
        print(f"- {model_name}")
except Exception as e:
    print(f"获取支持的嵌入模型列表失败: {e}")

# 尝试导入hipporag.HippoRAG以便检查默认模型
try:
    from hipporag import HippoRAG
    from inspect import signature, Parameter
    
    sig = signature(HippoRAG.__init__)
    for param_name, param in sig.parameters.items():
        if param_name == 'embedding_model_name' and param.default is not Parameter.empty:
            print(f"\n默认嵌入模型: {param.default}")
except Exception as e:
    print(f"获取默认嵌入模型信息失败: {e}")
