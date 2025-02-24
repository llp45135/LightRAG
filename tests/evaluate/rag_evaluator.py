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
    
    def __init__(self, rag: LightRAG, eval_dir: str = "tests/evaluate/kg_grounds", results_dir: str = "tests/evaluate/kg_grounds/results"):
        """
        初始化评估器
        
        Args:
            rag: LightRAG实例
            eval_dir: 评估数据目录路径
        """
        self.rag = rag
        self.eval_dir = eval_dir
        self.results_dir = results_dir
        self.results: Dict[str, List[QAEvalResult]] = {}
        
    def load_qa_pairs(self, filename: str) -> None:
        """加载单个QA评估数据文件
        
        Args:
            filename: 要加载的文件名
        """
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
                ground_truth=qa['answer']
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
        
    def evaluate_file(self, filename: str) -> None:
        """评估单个文件中的所有问题
        
        Args:
            filename: 要评估的文件名
        """
        file_path = os.path.join(self.eval_dir, filename)
        if not os.path.exists(file_path):
            print(f"警告：文件 {file_path} 不存在")
            return
        
        print(f"正在评估文件: {filename}")
        self.load_qa_pairs(filename)
        
        qa_results = self.results[filename]
        for i, qa_result in enumerate(qa_results):
            print(f"正在评估第 {i+1}/{len(qa_results)} 个问题")
            self.evaluate_single_question(qa_result)
            
        # 评估完成后立即保存该文件的结果
        self.save_results(filename)
        
        # 清理该文件的结果以释放内存
        del self.results[filename]

    def save_results(self, filename: str) -> None:
        """保存单个文件的评估结果
        
        Args:
            filename: 评估结果对应的原始文件名
        """
        # 创建结果目录
        results_dir = self.output
        os.makedirs(results_dir, exist_ok=True)

        output_path = os.path.join(results_dir, f"eval_{filename}")
        
        if filename not in self.results:
            print(f"警告：未找到 {filename} 的评估结果")
            return
            
        try:
            output_data = []
            for result in self.results[filename]:
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
        except Exception as e:
            print(f"保存结果到 {output_path} 时出错: {str(e)}")

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

def evaluate_rag_qa(
    working_dir: str = 'tests/KG',
    eval_dir: str = 'tests/evaluate/kg_grounds',
    results_dir: str = 'tests/evaluate/kg_grounds/results',
    input_file: str = 'kg9.md'
) -> None:
    """评估RAG系统的问答效果
    
    Args:
        working_dir: RAG工作目录路径
        eval_dir: 评估数据目录路径
        results_dir: 评估结果保存目录
        input_file: 输入的知识文档
    """
    # 确保工作目录存在
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
        
    # 创建RAG实例
    rag = create_rag_sync(working_dir)
    
    # 如果提供了输入文件，则插入文档
    if input_file and os.path.exists(input_file):
        print(f"正在插入文档: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
            loop = asyncio.get_event_loop()
            loop.run_until_complete(rag.insert(content))
    
    # 确保评估目录存在
    if not os.path.exists(eval_dir):
        print(f"错误：评估目录 {eval_dir} 不存在")
        return
        
    # 创建评估器
    evaluator = RAGEvaluator(rag, eval_dir, results_dir)
    
    try:
        # 遍历评估目录下的所有json文件
        for filename in os.listdir(eval_dir):
            if filename.endswith('.json'):
                try:
                    print(f"\n开始评估文件: {filename}")
                    evaluator.evaluate_file(filename)
                except Exception as e:
                    print(f"评估文件 {filename} 时出错: {str(e)}")
                    continue
        
        print("\n所有文件评估完成!")
        
    except Exception as e:
        print(f"评估过程中出现严重错误: {str(e)}")

def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='评估RAG系统的问答效果')
    parser.add_argument('--working-dir', type=str, default='tests/KG',
                      help='RAG工作目录路径 (默认: ./KG)')
    parser.add_argument('--eval-dir', type=str, default='tests/evaluate/kg_grounds',
                      help='评估数据目录路径 (默认: tests/evaluate)')
    parser.add_argument('--results_dir', type=str, default='tests/evaluate/kg_grounds/results',
                      help='评估结果的目录 (默认: tests/evaluate/kg_grounds/result)')
    parser.add_argument('--input-file', type=str, default='kg9.md',
                      help='输入的知识文档 (默认: kg9.md)')
    
    args = parser.parse_args()
    evaluate_rag_qa(
        working_dir=args.working_dir,
        eval_dir=args.eval_dir,
        results_dir=args.results_dir,
        input_file=args.input_file
    )

def analyze_evaluation_results(
    results_dir: str = 'tests/evaluate/kg_grounds/results',
    output_file: str = 'evaluation_summary.json'
) -> None:
    """分析评估结果文件，使用LLM对比答案质量并生成总结
    
    Args:
        results_dir: 评估结果目录
        output_file: 输出的总结文件名
    """
    if not os.path.exists(results_dir):
        print(f"错误：结果目录 {results_dir} 不存在")
        return
        
    # 加载环境变量
    load_dotenv()
    
    async def score_answer(question: str, ground_truth: str, answer: str) -> float:
        """使用LLM评分单个答案
        
        Args:
            question: 原始问题
            ground_truth: 标准答案
            answer: 待评估答案
            
        Returns:
            float: 1-10的评分
        """
        from lightrag.llm.openai import openai_complete_if_cache
        
        prompt = f"""请作为一位专业的评估者，评估以下答案的质量。

问题：{question}

标准答案：{ground_truth}

待评估答案：{answer}

请根据以下标准给出1-10分的评分：
1. 答案的准确性和完整性（4分）
2. 答案的相关性（3分）
3. 表述的清晰度和专业性（3分）

只需返回最终的总分（1-10的数字），不需要其他解释。"""

        try:
            score_text = await openai_complete_if_cache(
                "Qwen/Qwen2.5-14B-Instruct",
                prompt,
                api_key=os.getenv("SILICONFLOW_API_KEY"),
                base_url="https://api.siliconflow.cn/v1/",
            )
            return float(score_text.strip())
        except Exception as e:
            print(f"评分失败: {str(e)}")
            return 0.0

    async def analyze_file(filepath: str) -> dict:
        """分析单个评估结果文件
        
        Args:
            filepath: 评估结果文件路径
            
        Returns:
            dict: 分析结果
        """
        print(f"正在分析文件: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        file_summary = {
            "file_name": os.path.basename(filepath),
            "total_questions": len(results),
            "scores": {
                "naive": [],
                "local": [],
                "global": [],
                "mix": []
            },
            "question_details": []
        }
        
        for qa in results:
            question_scores = {
                "question": qa["question"],
                "scores": {}
            }
            
            # 评分每种方法的答案
            for method in ["naive", "local", "global", "mix"]:
                answer_key = f"{method}_answer"
                if answer_key in qa and qa[answer_key]:
                    score = await score_answer(
                        qa["question"],
                        qa["ground_truth"],
                        qa[answer_key]
                    )
                    file_summary["scores"][method].append(score)
                    question_scores["scores"][method] = score
                    # 添加随机延迟避免API限制
                    await asyncio.sleep(uniform(0.5, 1.5))
                    
            file_summary["question_details"].append(question_scores)
            
        # 计算平均分
        for method in file_summary["scores"]:
            scores = file_summary["scores"][method]
            if scores:
                file_summary[f"{method}_avg_score"] = sum(scores) / len(scores)
            else:
                file_summary[f"{method}_avg_score"] = 0
                
        return file_summary

    async def analyze_all_files() -> None:
        """分析所有评估结果文件"""
        all_results = []
        
        for filename in os.listdir(results_dir):
            if filename.endswith('.json') and filename.startswith('eval_'):
                filepath = os.path.join(results_dir, filename)
                try:
                    file_summary = await analyze_file(filepath)
                    all_results.append(file_summary)
                except Exception as e:
                    print(f"分析文件 {filename} 时出错: {str(e)}")
                    continue
        
        # 计算总体统计信息
        overall_summary = {
            "total_files": len(all_results),
            "total_questions": sum(r["total_questions"] for r in all_results),
            "method_scores": {
                "naive": [],
                "local": [],
                "global": [],
                "mix": []
            },
            "file_summaries": all_results
        }
        
        # 汇总所有方法的分数
        for result in all_results:
            for method in ["naive", "local", "global", "mix"]:
                if f"{method}_avg_score" in result:
                    overall_summary["method_scores"][method].append(
                        result[f"{method}_avg_score"]
                    )
        
        # 计算总体平均分
        for method in overall_summary["method_scores"]:
            scores = overall_summary["method_scores"][method]
            if scores:
                overall_summary[f"{method}_overall_avg"] = sum(scores) / len(scores)
            else:
                overall_summary[f"{method}_overall_avg"] = 0
        
        # 保存总结结果
        output_path = os.path.join(results_dir, output_file)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(overall_summary, f, ensure_ascii=False, indent=2)
            print(f"评估总结已保存至: {output_path}")
        except Exception as e:
            print(f"保存总结文件时出错: {str(e)}")
    
    # 运行异步分析函数
    loop = asyncio.get_event_loop()
    loop.run_until_complete(analyze_all_files())

if __name__ == "__main__":
    main() 