"""Command line shared by the basic and advanced entry points."""

import argparse
import logging
import multiprocessing
import os

from demo_core import DEFAULT_DOCUMENTS, DEFAULT_OUTPUT, ROOT


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def run_cli(demo_class, *, advanced=False, argv=None):
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description="HippoRAG + DeepSeek 本地文档问答")
    parser.add_argument("--documents", default=str(DEFAULT_DOCUMENTS), help="文档 JSON 路径")
    parser.add_argument("--save-dir", default=os.getenv("HIPPORAG_SAVE_DIR", str(DEFAULT_OUTPUT)))
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "facebook/contriever"))
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default=os.getenv("EMBEDDING_DEVICE", "auto")
    )
    parser.add_argument("--batch-size", type=positive_int, default=os.getenv("EMBEDDING_BATCH_SIZE", "8"))
    parser.add_argument("--query", help="执行一次问答后退出")
    parser.add_argument("--history-file", default="qa_history.csv", help="高级版 export 命令的目标文件")
    args = parser.parse_args(argv)
    if args.query is not None and not args.query.strip():
        parser.error("--query 不能为空")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method("spawn")
    demo = demo_class(save_dir=args.save_dir, device=args.device, batch_size=args.batch_size)
    try:
        if not demo.load_documents(args.documents):
            return 1
        if not demo.initialize(llm_model=args.model, embedding_model=args.embedding_model):
            return 1
        if not demo.index_documents():
            return 1
        if args.query is not None:
            return 0 if show_answer(demo, args.query, advanced) else 1
        print("\n=== HippoRAG + DeepSeek ===")
        print("输入问题，exit 退出" + ("；history 查看历史，export 导出 CSV" if advanced else ""))
        while True:
            try:
                query = input("\n请输入问题: ").strip()
            except EOFError:
                return 0
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                return 0
            if advanced and query.lower() == "history":
                for record in demo.history:
                    print(f"{record['timestamp']}\n问题：{record['query']}\n回答：{record['answer']}")
                if not demo.history:
                    print("暂无历史记录")
                continue
            if advanced and query.lower() == "export":
                if demo.export_history(args.history_file):
                    print(f"已导出到 {args.history_file}")
                continue
            show_answer(demo, query, advanced)
    except KeyboardInterrupt:
        print("\n已中断")
        return 130
    finally:
        demo.close()


def show_answer(demo, query, advanced):
    result = demo.ask(query)
    if result is None:
        print("本次问答失败，请查看错误信息后重试。")
        return False
    print(f"\n问题：{result['query']}\n\n回答：{result['answer']}")
    if result["retrieved_docs"]:
        print("\n参考资料：")
        for index, doc in enumerate(result["retrieved_docs"][:5], 1):
            print(f"{index}. {doc}")
    if advanced:
        print(f"\n进程内存：{demo.get_memory_usage():.1f} MB")
    return True
