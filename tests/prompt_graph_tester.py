import os
import json
import logging
import yaml
import streamlit as st
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema, OutputFixingParser
from unstructured.partition.auto import partition
import networkx as nx
import matplotlib.pyplot as plt
import markdown

# 配置日志记录
logging.basicConfig(level=logging.INFO, 
                   handlers=[
                       logging.FileHandler("extraction.log", mode="w"),
                       logging.StreamHandler()  # 添加控制台处理器
                   ],
                   format="%(asctime)s - %(levelname)s - %(message)s")

# 加载配置文件
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 在导入部分添加
from dotenv import load_dotenv

# 在代码的开头加载环境变量
load_dotenv()

# Few-shot 示例
few_shot_examples = [
    {
        "file_theme": "城市公交客运规章",
        "kg_purpose": "智能问答",
        "entity_types": "乘客, 客运公司, 车辆, 线路, 站点, 票务, 规章条款, 违规行为, 处罚措施",
        "relation_types": "乘客-购买-票务, 规章条款-约束-乘客行为, 车辆-运行于-线路, 站点-属于-线路, 违规行为-导致-处罚措施"
    }
]

# 修改实体类型生成提示词模板
entity_type_generation_prompt_template = """
**Few-shot示例**
文件主题：城市公交客运规章
知识图谱用途：智能问答
实体类型列表：乘客, 客运公司, 车辆, 线路, 站点, 票务, 规章条款, 违规行为, 处罚措施

**当前文件主题**:{file_theme}
**当前知识图谱用途**:{kg_purpose}
请根据当前文件主题和知识图谱用途，生成初步的实体类型列表。

**待提取的客运规章文件内容:**
{document_content}

**输出格式要求:**
请直接返回实体类型列表，每个实体类型用逗号分隔。例如：
实体类型1, 实体类型2, 实体类型3
"""

entity_type_generation_prompt = PromptTemplate(
    template=entity_type_generation_prompt_template,
    input_variables=["file_theme", "kg_purpose", "document_content"]
)

# 修改关系类型生成提示词模板
relation_type_generation_prompt_template = """
**Few-shot示例**
文件主题：城市公交客运规章
知识图谱用途：智能问答
关系类型列表：乘客-购买-票务, 规章条款-约束-乘客行为, 车辆-运行于-线路, 站点-属于-线路, 违规行为-导致-处罚措施

**当前文件主题**:{file_theme}
**当前知识图谱用途**:{kg_purpose}
请根据当前文件主题和知识图谱用途，生成初步的关系类型列表。

**待提取的客运规章文件内容:**
{document_content}

**输出格式要求:**
请直接返回关系类型列表，每个关系类型用逗号分隔。例如：
主体-关系-客体, 主体-关系-客体
"""

relation_type_generation_prompt = PromptTemplate(
    template=relation_type_generation_prompt_template,
    input_variables=["file_theme", "kg_purpose", "document_content"]
)

# 信息提取提示词模板
extraction_prompt_template = """
请从以下客运规章文件中提取信息，构建知识图谱。
你需要识别文件中的**实体**和**实体之间的关系**，并将结果以结构化 JSON 数据形式输出。
请重点关注与 **{file_theme}** 相关的信息。

**知识图谱用途:** {kg_purpose}
**目标实体类型列表:** {entity_types}
**目标关系类型列表:** {relation_types}

**待提取的客运规章文件内容:**
{document_content}

**输出格式要求:**
{format_instructions}
"""
extraction_generation_prompt = PromptTemplate(
    template=extraction_prompt_template,
    input_variables=["file_theme", "kg_purpose","entity_types","relation_types","document_content","format_instructions"]
)



# 定义输出结构
response_schemas_extraction = [
    ResponseSchema(name="主体实体", description="主体实体，包含类型、名称和属性"),
    ResponseSchema(name="关系", description="关系类型"),
    ResponseSchema(name="客体实体", description="客体实体，包含类型、名称和属性"),
    ResponseSchema(name="来源", description="提取信息的来源")
]
output_parser_extraction = StructuredOutputParser.from_response_schemas(response_schemas_extraction)
format_instructions_extraction = output_parser_extraction.get_format_instructions()

# LLM 模型
llm = OpenAI(model_name="Qwen/Qwen2.5-14B-Instruct", 
             temperature=0,        
             api_key=os.getenv("SILICONFLOW_API_KEY"),
             base_url="https://api.siliconflow.cn/v1/",
             max_tokens=4096  # 添加最大token限制
)

# 加载文档
def load_document(file_path):
    try:
        if file_path.endswith('.md'):
            # 处理 Markdown 文件
            with open(file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
                html_content = markdown.markdown(md_content)
                return html_content  # 返回 HTML 内容
        else:
            # 处理其他文件类型
            elements = partition(filename=file_path)
            return "\n".join([str(el) for el in elements])
    except Exception as e:
        logging.error(f"加载文件失败: {e}")
        return None

# 修改实体类型生成函数
def generate_entity_types(file_theme, kg_purpose, document_content):
    try:
        prompt_value = entity_type_generation_prompt.format_prompt(file_theme=file_theme, kg_purpose=kg_purpose, document_content=document_content)
        output = llm.invoke(prompt_value.to_string())
        logging.info(f"实体类型生成 LLM 原始输出: {output}")
        
        try:
            # 如果输出已经是列表，直接使用
            if isinstance(output, list):
                entity_types = output
            else:
                # 清理和分割输出
                output = output.strip()
                # 移除可能的前缀说明文字
                if "：" in output:
                    output = output.split("：")[-1]
                if ":" in output:
                    output = output.split(":")[-1]
                    
                # 分割并清理每个实体类型
                entity_types = [et.strip() for et in output.split(",") if et.strip()]
            
            if not entity_types:
                logging.error("未能从输出中提取到实体类型")
                return []
                
            logging.info(f"成功提取实体类型: {entity_types}")
            return entity_types
            
        except Exception as e:
            logging.error(f"实体类型处理失败: {e}")
            logging.error(f"问题输出: {output}")
            return []
    except Exception as e:
        logging.error(f"实体类型生成失败，详细错误: {str(e)}")
        logging.error(f"错误类型: {type(e)}")
        logging.error(f"错误输出: {output}")
        return []

# 修改关系类型生成函数
def generate_relation_types(file_theme, kg_purpose, document_content):
    try:
        prompt_value = relation_type_generation_prompt.format_prompt(file_theme=file_theme, kg_purpose=kg_purpose, document_content=document_content)
        output = llm.invoke(prompt_value.to_string())
        logging.info(f"关系类型生成 LLM 原始输出: {output}")
        
        try:
            # 如果输出已经是列表，直接使用
            if isinstance(output, list):
                relation_types = output
            else:
                # 清理和分割输出
                output = output.strip()
                # 移除可能的前缀说明文字
                if "：" in output:
                    output = output.split("：")[-1]
                if ":" in output:
                    output = output.split(":")[-1]
                    
                # 分割并清理每个关系类型
                relation_types = [rt.strip() for rt in output.split(",") if rt.strip()]
            
            if not relation_types:
                logging.error("未能从输出中提取到关系类型")
                return []
                
            logging.info(f"成功提取关系类型: {relation_types}")
            return relation_types
            
        except Exception as e:
            logging.error(f"关系类型处理失败: {e}")
            logging.error(f"问题输出: {output}")
            return []
    except Exception as e:
        logging.error(f"关系类型生成失败，详细错误: {str(e)}")
        logging.error(f"错误类型: {type(e)}")
        logging.error(f"错误输出: {output}")
        return []

# 信息提取
def extract_information(document_content, file_theme, kg_purpose, entity_types, relation_types, format_instructions):
    try:
        # 修改输出格式说明
        format_instructions = """请以JSON格式返回提取的信息，每条记录包含以下字段：
        {
            "主体实体": {
                "类型": "实体类型",
                "名称": "实体名称",
                "属性": "实体属性"
            },
            "关系": "关系类型",
            "客体实体": {
                "类型": "实体类型",
                "名称": "实体名称",
                "属性": "实体属性"
            },
            "来源": "信息来源"
        }"""
        
        prompt_value = extraction_generation_prompt.format_prompt(
            document_content=document_content,
            file_theme=file_theme,
            kg_purpose=kg_purpose,
            entity_types=entity_types,
            relation_types=relation_types,
            format_instructions=format_instructions
        )
        output = llm.invoke(prompt_value.to_string())
        logging.info(f"LLM输出: {output}")  # 添加日志记录
        
        try:
            # 尝试直接解析JSON
            if isinstance(output, str):
                extracted_data = json.loads(output)
            else:
                extracted_data = output
                
            # 验证输出格式
            if not isinstance(extracted_data, list):
                extracted_data = [extracted_data]
                
            # 确保每个记录都有必要的字段
            for item in extracted_data:
                if not all(key in item for key in ["主体实体", "关系", "客体实体", "来源"]):
                    raise ValueError("输出格式不符合要求")
                    
            return extracted_data
            
        except Exception as e_json_parse:
            logging.warning(f"JSON 解析失败: {e_json_parse}")
            fixing_parser = OutputFixingParser.from_parser(parser=output_parser_extraction)
            extracted_data_fixed = fixing_parser.parse(output)
            return extracted_data_fixed
            
    except Exception as e_llm_call:
        logging.error(f"LLM 信息提取失败: {e_llm_call}")
        return None

# 可视化知识图谱
def visualize_knowledge_graph(data):
    G = nx.DiGraph()
    for item in data:
        G.add_node(item["主体实体"]["名称"], label=item["主体实体"]["类型"])
        G.add_node(item["客体实体"]["名称"], label=item["客体实体"]["类型"])
        G.add_edge(item["主体实体"]["名称"], item["客体实体"]["名称"], label=item["关系"])
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    st.pyplot(plt)

# 新函数：直接执行信息提取
def run_extraction_tool(file_path, file_theme, kg_purpose):
    document_content = load_document(file_path)
    if document_content:
        logging.info("文件加载成功")
        try:
            entity_types = generate_entity_types(file_theme, kg_purpose, document_content)
            relation_types = generate_relation_types(file_theme, kg_purpose, document_content)
            logging.info(f"生成的实体类型: {entity_types}")
            logging.info(f"生成的关系类型: {relation_types}")
            
            if not entity_types or not relation_types:
                logging.error("实体类型或关系类型生成失败")
                return
            
            format_instructions = """请以JSON格式返回提取的信息，每条记录包含以下字段：
            {
                "主体实体": {
                    "类型": "实体类型",
                    "名称": "实体名称",
                    "属性": "实体属性"
                },
                "关系": "关系类型",
                "客体实体": {
                    "类型": "实体类型",
                    "名称": "实体名称",
                    "属性": "实体属性"
                },
                "来源": "信息来源"
            }"""
            
            extracted_info = extract_information(document_content, file_theme, kg_purpose, entity_types, relation_types, format_instructions)
            if extracted_info:
                logging.info(f"提取结果: {extracted_info}")
                visualize_knowledge_graph(extracted_info)
            else:
                logging.error("信息提取失败")
        except Exception as e:
            logging.error(f"处理过程中出错: {str(e)}")
    else:
        logging.error("文件加载失败")

# 新函数：Streamlit 界面
def run_streamlit_interface():
    st.title("客运规章知识图谱提取工具")
    uploaded_file = st.file_uploader("上传文件", type=["pdf", "docx", "txt", "md"])
    if uploaded_file:
        with open("temp_file", "wb") as f:
            f.write(uploaded_file.getbuffer())
        document_content = load_document("temp_file")
        if document_content:
            st.write("文件加载成功")
            file_theme = st.text_input("文件主题", value=config["file_subject"])
            kg_purpose = st.text_input("知识图谱用途", value=config["knowledge_graph_purpose"])
            if st.button("生成实体和关系类型"):
                entity_types = generate_entity_types(file_theme, kg_purpose, document_content)
                relation_types = generate_relation_types(file_theme, kg_purpose, document_content)
                st.write("生成的实体类型:", ", ".join(entity_types))
                st.write("生成的关系类型:", ", ".join(relation_types))
                st.session_state["entity_types"] = entity_types
                st.session_state["relation_types"] = relation_types
            entity_types_input = st.text_area("实体类型（逗号分隔）", value=", ".join(st.session_state.get("entity_types", [])))
            relation_types_input = st.text_area("关系类型（逗号分隔）", value=", ".join(st.session_state.get("relation_types", [])))
            if st.button("执行信息提取"):
                entity_types = entity_types_input.split(", ")
                relation_types = relation_types_input.split(", ")
                format_instructions = """请以JSON格式返回提取的信息，每条记录包含以下字段：
                {
                    "主体实体": {
                        "类型": "实体类型",
                        "名称": "实体名称",
                        "属性": "实体属性"
                    },
                    "关系": "关系类型",
                    "客体实体": {
                        "类型": "实体类型",
                        "名称": "实体名称",
                        "属性": "实体属性"
                    },
                    "来源": "信息来源"
                }"""
                extracted_info = extract_information(document_content, file_theme, kg_purpose, entity_types, relation_types, format_instructions)
                if extracted_info:
                    st.write("提取结果:", extracted_info)
                    visualize_knowledge_graph(extracted_info)
                else:
                    st.write("信息提取失败")

# 示例调用
if __name__ == "__main__":
    # 选择运行模式
    # run_mode = input("选择运行模式 (1: Streamlit, 2: 直接执行): ")
    run_mode = "2"
    if run_mode == "1":
        run_streamlit_interface()
    elif run_mode == "2":
        run_extraction_tool("./tests/kg9.md", "登乘证的管理规定", "智能问答")
    else:
        print("无效的选择")