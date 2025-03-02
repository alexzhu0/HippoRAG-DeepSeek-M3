import os
import json
import torch
import pandas as pd
import logging
import multiprocessing
import psutil
from datetime import datetime
from dotenv import load_dotenv

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

class HippoRAGDemo:
    """高级HippoRAG演示类"""
    
    def __init__(self, save_dir="outputs"):
        self.save_dir = save_dir
        self.hipporag = None
        self.docs = []
        self.history = []
        
    def initialize(self, llm_model="deepseek-chat", embedding_model="facebook/contriever", llm_base_url=None):
        """初始化HippoRAG系统"""
        # 在运行时导入HippoRAG，避免多进程初始化问题
        from hipporag import HippoRAG
        
        try:
            # 设置API URL
            if not llm_base_url:
                llm_base_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
                
            # 初始化HippoRAG
            self.hipporag = HippoRAG(
                save_dir=self.save_dir,
                llm_model_name=llm_model,
                llm_base_url=llm_base_url,
                embedding_model_name=embedding_model
            )
            
            logger.info(f"HippoRAG初始化成功")
            return True
        except Exception as e:
            logger.error(f"初始化HippoRAG失败: {e}")
            return False
    
    def load_documents(self, file_path):
        """从文件加载文档"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.docs = json.load(f)
                
            logger.info(f"成功加载{len(self.docs)}篇文档")
            return True
        except Exception as e:
            logger.error(f"加载文档失败: {e}")
            return False
    
    def index_documents(self):
        """索引文档"""
        if self.hipporag is None:
            logger.warning("HippoRAG未初始化！")
            return False
            
        if not self.docs:
            logger.warning("没有文档可索引！")
            return False
            
        try:
            # 索引文档
            self.hipporag.index(docs=self.docs)
            logger.info("文档索引完成")
            return True
        except Exception as e:
            logger.error(f"索引文档失败: {e}")
            return False
        
    def ask(self, query):
        """提问并获取回答"""
        if self.hipporag is None:
            logger.warning("HippoRAG未初始化！")
            return None
            
        try:
            results = self.hipporag.rag_qa(
                queries=[query]
            )
            
            # 记录历史
            qa_record = {
                "query": query,
                "answer": "",
                "retrieved_docs": []
            }
            
            # 处理不同格式的结果
            if isinstance(results, list) and len(results) > 0:
                if 'answer' in results[0]:
                    qa_record["answer"] = results[0]['answer']
                    if 'retrieved_nodes' in results[0]:
                        qa_record["retrieved_docs"] = [
                            doc['node']['content'] if isinstance(doc['node'], dict) else doc['node']
                            for doc in results[0]['retrieved_nodes']
                        ]
            elif isinstance(results, dict):
                if 'answers' in results:
                    qa_record["answer"] = results['answers'][0]
                    if 'retrieved_nodes' in results:
                        qa_record["retrieved_docs"] = [
                            doc['node']['content'] if isinstance(doc['node'], dict) else doc['node']
                            for doc in results['retrieved_nodes']
                        ]
            
            # 添加到历史
            self.history.append({
                "query": qa_record["query"],
                "answer": qa_record["answer"],
                "retrieved_docs": qa_record["retrieved_docs"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "memory_usage": self.get_memory_usage(),
            })
            
            # 返回答案
            return qa_record
        except Exception as e:
            logger.error(f"提问失败: {e}")
            return None
    
    def export_history(self, file_path="qa_history.csv"):
        """导出问答历史"""
        if not self.history:
            logger.warning("没有历史记录可导出")
            return False
            
        try:
            # 创建DataFrame
            df = pd.DataFrame([
                {
                    "时间": record["timestamp"],
                    "问题": record["query"],
                    "回答": record["answer"],
                    "内存使用(MB)": record["memory_usage"]
                }
                for record in self.history
            ])
            
            # 导出到CSV
            df.to_csv(file_path, index=False, encoding="utf-8")
            logger.info(f"历史记录已导出到 {file_path}")
            return True
        except Exception as e:
            logger.error(f"导出历史失败: {e}")
            return False
    
    def get_memory_usage(self):
        """获取当前内存使用情况（帮助监控M3资源使用）"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # 转换为MB
        except:
            return 0

def main():
    """主函数"""
    print("\n=== HippoRAG高级问答系统 (基于DeepSeek, M3优化版) ===")
    print("输入问题进行询问，输入'exit'退出，输入'history'查看历史记录\n")
    
    # 初始化
    demo = HippoRAGDemo()
    
    # 交互式问答
    while True:
        query = input("请输入您的问题: ")
        if query.lower() == 'exit':
            break
        elif query.lower() == 'history':
            # 显示历史记录
            print("\n" + "="*60)
            print("📜 历史问答记录")
            print("="*60)
            
            if not demo.history:
                print("暂无历史记录")
            else:
                for i, record in enumerate(demo.history):
                    print(f"\n[{i+1}] 时间: {record.get('timestamp', '未知')}")
                    print(f"📝 问题: {record.get('query', '未知')}")
                    print(f"🤖 回答: {record.get('answer', '未知')[:100]}..." if len(record.get('answer', '')) > 100 else f"🤖 回答: {record.get('answer', '未知')}")
                    print(f"💾 内存使用: {record.get('memory_usage', '未知')} MB")
                    print("-"*40)
            continue
        
        print("思考中...")
        result = demo.ask(query)
        
        if result:
            # 美化输出结果
            print("\n" + "="*60)
            print("📝 问题：" + query)
            print("="*60)
            
            # 显示答案
            print("\n🤖 回答：")
            print("-"*50)
            print(result['answer'].strip())
            print("-"*50)
            
            # 显示来源
            if result['retrieved_docs']:
                print("\n📚 参考资料：")
                print("-"*50)
                for i, doc in enumerate(result['retrieved_docs'][:5]):  # 最多显示5个来源
                    print(f"{i+1}. {doc}")
                if len(result['retrieved_docs']) > 5:
                    print(f"...还有 {len(result['retrieved_docs'])-5} 个来源未显示")
                print("-"*50)
            
            # 显示内存使用
            memory_usage = demo.get_memory_usage()
            print(f"\n💾 内存使用: {memory_usage:.2f} MB")
            print(f"⏱️ 回答时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*60)
        else:
            print("处理您的问题时出错，请稍后重试")

if __name__ == "__main__":
    main()
