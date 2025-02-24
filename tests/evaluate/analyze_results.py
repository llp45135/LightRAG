import json
import os
import asyncio
from random import uniform
from dotenv import load_dotenv
from typing import Dict, List
import re
from tqdm import tqdm
from datetime import datetime

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

请只返回一个1到10之间的数字，不要包含任何其他文字。"""

    try:
        score_text = await openai_complete_if_cache(
            "Qwen/Qwen2.5-14B-Instruct",
            prompt,
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1/",
        )
        # 提取第一个数字
        score_match = re.search(r'\d+(?:\.\d+)?', score_text)
        if score_match:
            return float(score_match.group())
        print(f"无法从回答中提取分数: {score_text}")
        return 0.0
    except Exception as e:
        print(f"评分失败: {str(e)}")
        return 0.0

async def analyze_file(filepath: str, pbar: tqdm) -> dict:
    """分析单个评估结果文件
    
    Args:
        filepath: 评估结果文件路径
        
    Returns:
        dict: 分析结果
    """
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始分析文件: {filepath}")
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
    
    # 创建问题进度条
    question_pbar = tqdm(
        total=len(results),
        desc="评估问题进度",
        leave=False
    )
    
    for i, qa in enumerate(results, 1):
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 评估第 {i}/{len(results)} 个问题")
        print(f"问题: {qa['question']}")
        
        question_scores = {
            "question": qa["question"],
            "ground_truth": qa["ground_truth"],
            "answers": {},
            "scores": {}
        }
        
        # 评分每种方法的答案
        for method in ["naive", "local", "global", "mix"]:
            answer_key = f"{method}_answer"
            if answer_key in qa and qa[answer_key]:
                print(f"评估 {method} 方法的答案...")
                # 保存答案
                question_scores["answers"][method] = qa[answer_key]
                # 评分
                score = await score_answer(
                    qa["question"],
                    qa["ground_truth"],
                    qa[answer_key]
                )
                print(f"{method} 方法得分: {score}")
                file_summary["scores"][method].append(score)
                question_scores["scores"][method] = score
                # 添加随机延迟避免API限制
                await asyncio.sleep(uniform(0.5, 1.5))
                
        file_summary["question_details"].append(question_scores)
        question_pbar.update(1)
        pbar.update(1)
        
    question_pbar.close()
    
    # 计算平均分
    for method in file_summary["scores"]:
        scores = file_summary["scores"][method]
        if scores:
            avg_score = sum(scores) / len(scores)
            file_summary[f"{method}_avg_score"] = avg_score
            print(f"\n{method} 方法平均分: {avg_score:.2f}")
        else:
            file_summary[f"{method}_avg_score"] = 0
            
    return file_summary

async def analyze_all_files(results_dir: str, output_file: str) -> None:
    """分析所有评估结果文件
    
    Args:
        results_dir: 评估结果目录
        output_file: 输出的总结文件名
    """
    # 获取所有需要评估的文件
    eval_files = [
        f for f in os.listdir(results_dir)
        if f.endswith('.json') and f.startswith('eval_')
    ]
    
    if not eval_files:
        print("没有找到需要评估的文件")
        return
        
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始评估")
    print(f"找到 {len(eval_files)} 个文件需要评估")
    
    # 计算总问题数
    total_questions = 0
    for filename in eval_files:
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
            total_questions += len(results)
            
    print(f"总共需要评估 {total_questions} 个问题")
    
    # 创建总进度条
    with tqdm(total=total_questions, desc="总体评估进度") as pbar:
        all_results = []
        for filename in eval_files:
            filepath = os.path.join(results_dir, filename)
            try:
                file_summary = await analyze_file(filepath, pbar)
                all_results.append(file_summary)
            except Exception as e:
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 分析文件 {filename} 时出错: {str(e)}")
                continue
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 计算总体统计信息")
    
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
    print("\n各方法总体平均分：")
    for method in overall_summary["method_scores"]:
        scores = overall_summary["method_scores"][method]
        if scores:
            avg_score = sum(scores) / len(scores)
            overall_summary[f"{method}_overall_avg"] = avg_score
            print(f"{method}: {avg_score:.2f}")
        else:
            overall_summary[f"{method}_overall_avg"] = 0
            print(f"{method}: 0.00")
    
    # 保存总结结果
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 保存评估结果")
    
    if os.path.isabs(output_file):
        output_path = output_file
    else:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = output_file
        else:
            output_path = os.path.join(results_dir, output_file)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(overall_summary, f, ensure_ascii=False, indent=2)
        print(f"评估总结已保存至: {output_path}")
    except Exception as e:
        print(f"保存总结文件时出错: {str(e)}")
        
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 评估完成!")

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
    
    # 运行异步分析函数
    loop = asyncio.get_event_loop()
    loop.run_until_complete(analyze_all_files(results_dir, output_file))

def main():
    """命令行入口函数"""
    import argparse
    parser = argparse.ArgumentParser(description='分析RAG评估结果')
    parser.add_argument('--results-dir', type=str, 
                      default='tests/evaluate/kg_grounds/results',
                      help='评估结果目录路径')
    parser.add_argument('--output-file', type=str, 
                      default='tests/evaluate/kg_grounds/evaluation_summary.json',
                      help='输出的总结文件名')
    
    args = parser.parse_args()
    analyze_evaluation_results(args.results_dir, args.output_file)

if __name__ == "__main__":
    main() 