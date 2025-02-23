import json
import os
import asyncio  # 移除 asyncio
from typing import List, Dict
from lightrag import LightRAG, QueryParam
from dataclasses import dataclass
import numpy as np
import argparse
from dotenv import load_dotenv
from lightrag.llm.siliconcloud import siliconcloud_embedding
from lightrag.utils import EmbeddingFunc
import nest_asyncio
from tenacity import retry, wait_exponential, stop_after_attempt
from time import sleep
from random import uniform


@dataclass
class QAEvalResult:
    """问答评估结果数据类"""
    question: str
    ground_truth: str
    naive_answer: str = ""
    local_answer: str = ""
    global_answer: str = ""
    mix_answer: str = ""
    
class RAGEvaluator:
    """RAG系统评估器"""
    
    def __init__(self, rag: LightRAG, eval_dir: str = "tests/evaluate"):
        """
        初始化评估器
        
        Args:
            rag: LightRAG实例
            eval_dir: 评估数据目录路径
        """
        self.rag = rag
        self.eval_dir = eval_dir
        self.results: Dict[str, List[QAEvalResult]] = {}
        
    def load_qa_pairs(self) -> None:
        """加载所有QA评估数据"""
        for filename in os.listdir(self.eval_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.eval_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    qa_pairs = json.load(f)
                    
                results = []
                for qa in qa_pairs:
                    # 添加字段存在性检查
                    if 'question' not in qa or 'answer' not in qa:
                        print(f"警告：文件 {filename} 中存在格式不正确的数据项：{qa}")
                        continue
                        
                    result = QAEvalResult(
                        question=qa['question'],
                        ground_truth=qa['answer']  # 确保使用正确的字段名
                    )
                    results.append(result)
                self.results[filename] = results
                
    def evaluate_single_question(self, qa_result: QAEvalResult) -> None:
        """评估单个问题的不同方法回答结果"""
        modes = [
            ("naive", "naive_answer"),
            ("local", "local_answer"), 
            ("global", "global_answer"),
            ("mix", "mix_answer")
        ]
        
        for mode, attr in modes:
            print(f"使用{mode}模式回答...")
            try:
                # 添加随机延迟 (0.5-1.5秒)
                sleep(uniform(0.5, 1.5))
                answer = self.rag.query(
                    qa_result.question,
                    param=QueryParam(mode=mode)
                )
                setattr(qa_result, attr, answer)
            except Exception as e:
                print(f"{mode}模式评估失败: {str(e)}")
                setattr(qa_result, attr, f"评估失败: {str(e)}")
                # 遇到错误时延长等待时间
                sleep(5)
        
    def evaluate_all(self) -> None:
        """评估所有问题"""
        self.load_qa_pairs()
        
        for filename, qa_results in self.results.items():
            print(f"正在评估文件: {filename}")
            for i, qa_result in enumerate(qa_results):
                print(f"正在评估第 {i+1}/{len(qa_results)} 个问题")
                self.evaluate_single_question(qa_result)
                
    def save_results(self) -> None:
        """保存评估结果"""
        # 创建结果目录
        results_dir = os.path.join(self.eval_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        for filename, qa_results in self.results.items():
            output_path = os.path.join(results_dir, f"eval_{filename}")
            
            output_data = []
            for result in qa_results:
                output_data.append({
                    "question": result.question,
                    "ground_truth": result.ground_truth,
                    "naive_answer": result.naive_answer,
                    "local_answer": result.local_answer, 
                    "global_answer": result.global_answer,
                    "mix_answer": result.mix_answer
                })
                
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
                
            print(f"评估结果已保存至: {output_path}")

def create_rag_sync(working_dir: str) -> LightRAG:
    """同步方式创建RAG实例"""
    # 加载环境变量
    load_dotenv()
    
    async def llm_model_func(
        prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
    ) -> str:
        from lightrag.llm.openai import openai_complete_if_cache
        @retry(wait=wait_exponential(multiplier=1, min=4, max=60), 
               stop=stop_after_attempt(5),
               retry_error_callback=lambda retry_state: "")
        async def _call_api():
            return await openai_complete_if_cache(
                "Qwen/Qwen2.5-14B-Instruct",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=os.getenv("SILICONFLOW_API_KEY"),
                base_url="https://api.siliconflow.cn/v1/",
                **kwargs,
            )
        
        try:
            return await _call_api()
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            return "无法获取回答"

    async def embedding_func(texts: list[str]) -> np.ndarray:
        return await siliconcloud_embedding(
            texts,
            model="netease-youdao/bce-embedding-base_v1",
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            max_token_size=512,
        )

    # 创建RAG实例（保持异步初始化）
    loop = asyncio.get_event_loop()
    rag = LightRAG(
        working_dir=working_dir,
        chunking_mode="markdown",
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=512,
            func=embedding_func
        ),
    )
    
    return rag

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估RAG系统的问答效果')
    parser.add_argument('--working-dir', type=str, default='tests/KG',
                      help='RAG工作目录路径 (默认: ./KG)')
    parser.add_argument('--eval-dir', type=str, default='tests/evaluate',
                      help='评估数据目录路径 (默认: tests/evaluate)')
    parser.add_argument('--input-file', type=str, default='kg9.md',
                      help='输入的知识文档 (默认: kg9.md)')
    
    args = parser.parse_args()
    
    # 确保工作目录存在
    if not os.path.exists(args.working_dir):
        os.makedirs(args.working_dir)
        
    # 创建RAG实例
    rag = create_rag_sync(args.working_dir)
    
    # 如果提供了输入文件，则插入文档
    if args.input_file and os.path.exists(args.input_file):
        print(f"正在插入文档: {args.input_file}")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
            loop = asyncio.get_event_loop()
            loop.run_until_complete(rag.insert(content))
    
    # 创建评估器
    evaluator = RAGEvaluator(rag, args.eval_dir)
    
    try:
        # 运行评估
        print("开始评估...")
        evaluator.evaluate_all()
        
        # 保存结果
        print("保存评估结果...")
        evaluator.save_results()
        
        print("评估完成!")
        
    except Exception as e:
        print(f"评估过程中出现严重错误: {str(e)}")
        # 保存已完成的评估结果
        if evaluator.results:
            print("尝试保存部分结果...")
            evaluator.save_results()

if __name__ == "__main__":
    # 运行主函数
    main() 