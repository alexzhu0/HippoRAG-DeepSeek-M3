import os
import json
import torch
import logging
import multiprocessing
from dotenv import load_dotenv
from datetime import datetime

# 在Mac上使用spawn方法解决多进程问题
multiprocessing.set_start_method('spawn', force=True)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 检查MPS可用性（Apple Metal性能着色器）
try:
    if torch.backends.mps.is_available():
        logger.info("使用M3芯片的MPS加速")
    else:
        logger.info("使用CPU运算")
except:
    logger.warning("无法检测MPS可用性，使用CPU")

# 确保API密钥已设置
if not os.getenv("DEEPSEEK_API_KEY"):
    raise ValueError("请在.env文件中设置DEEPSEEK_API_KEY")

def main():
    # 在运行时导入HippoRAG，避免多进程初始化问题
    from hipporag import HippoRAG
    
    logger.info("加载文档...")
    # 加载示例文档
    try:
        with open("data/documents.json", "r", encoding="utf-8") as f:
            docs = json.load(f)
        
        logger.info(f"成功加载{len(docs)}篇文档")
    except Exception as e:
        logger.error(f"加载文档失败: {e}")
        return
    
    # 初始化HippoRAG (M3优化版本)
    logger.info("初始化HippoRAG...")
    try:
        hipporag = HippoRAG(
            save_dir="outputs",
            llm_model_name="deepseek-chat",  # 使用deepseek-chat模型
            llm_base_url=os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1"),  # deepseek API URL
            embedding_model_name="facebook/contriever"  # 使用完整的Hugging Face模型路径
        )
    except Exception as e:
        logger.error(f"初始化HippoRAG失败: {e}")
        logger.error("请确保安装了所有依赖，并检查API密钥是否正确")
        return
    
    # 索引文档
    logger.info("开始索引文档...")
    try:
        hipporag.index(docs=docs)  # 移除不支持的batch_size参数
        logger.info("索引完成")
    except Exception as e:
        logger.error(f"索引文档失败: {e}")
        return
    
    # 交互式问答循环
    print("\n=== HippoRAG问答系统 (基于DeepSeek, M3优化版) ===")
    print("输入问题进行询问，输入'exit'退出")
    
    while True:
        query = input("\n请输入您的问题: ")
        if query.lower() == 'exit':
            break
        
        print("思考中...")
        try:
            # 执行RAG问答
            results = hipporag.rag_qa(
                queries=[query]
            )
            
            # 美化输出结果
            print("\n" + "="*50)
            print("📝 问题：" + query)
            print("="*50)
            
            # 提取答案
            answer = None
            retrieved_docs = []
            
            if isinstance(results, list) and len(results) > 0:
                # 如果返回列表格式
                if 'answer' in results[0]:
                    answer = results[0]['answer']
                    if 'retrieved_nodes' in results[0]:
                        for doc in results[0]['retrieved_nodes']:
                            doc_content = doc['node']['content'] if isinstance(doc['node'], dict) else doc['node']
                            retrieved_docs.append(doc_content)
            elif isinstance(results, dict):
                # 如果返回字典格式
                if 'answers' in results:
                    answer = results['answers'][0]
                    if 'retrieved_nodes' in results:
                        for doc in results['retrieved_nodes']:
                            doc_content = doc['node']['content'] if isinstance(doc['node'], dict) else doc['node']
                            retrieved_docs.append(doc_content)
            
            # 如果上述方法都无法提取答案，尝试直接从原始结果中提取
            if answer is None and isinstance(results, tuple) and len(results) > 0:
                # 处理元组格式返回
                if isinstance(results[0], list) and len(results[0]) > 0:
                    if hasattr(results[0][0], 'answer'):
                        answer = results[0][0].answer
                        if hasattr(results[0][0], 'docs'):
                            retrieved_docs = results[0][0].docs
            
            # 显示答案
            if answer:
                print("\n🤖 回答：")
                print("-"*50)
                print(answer.strip())
                print("-"*50)
            else:
                print("\n🤖 回答：")
                print("-"*50)
                print("无法生成回答。")
                print("-"*50)
            
            # 显示来源
            if retrieved_docs:
                print("\n📚 参考资料：")
                print("-"*50)
                for i, doc in enumerate(retrieved_docs[:5]):  # 最多显示5个来源
                    print(f"{i+1}. {doc}")
                if len(retrieved_docs) > 5:
                    print(f"...还有 {len(retrieved_docs)-5} 个来源未显示")
                print("-"*50)
            
            # 显示时间戳
            print(f"\n⏱️ 回答时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*50)
        except Exception as e:
            logger.error(f"处理查询失败: {e}")
            print("处理您的问题时出错，请稍后重试")

if __name__ == "__main__":
    main()
