import os
import json
import logging
import yaml
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema, OutputFixingParser
from unstructured.partition.auto import partition
import networkx as nx
import matplotlib.pyplot as plt
import markdown

# 配置日志记录
logging.basicConfig(level=logging.INFO, filename="extraction.log", filemode="w",
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

# 实体类型生成提示词模板（加入Few-shot）
entity_type_generation_prompt_template = """
**Few-shot示例**
文件主题：城市公交客运规章
知识图谱用途：智能问答
实体类型列表：乘客, 客运公司, 车辆, 线路, 站点, 票务, 规章条款, 违规行为, 处罚措施

**当前文件主题**：{file_theme}
**当前知识图谱用途**：{kg_purpose}
请根据当前文件主题和知识图谱用途，生成初步的实体类型列表。
"""

entity_type_generation_prompt = PromptTemplate(
    template=entity_type_generation_prompt_template,
    input_variables=["file_theme", "kg_purpose"]
)

# 关系类型生成提示词模板（加入Few-shot）
relation_type_generation_prompt_template = """
**Few-shot示例**
文件主题：城市公交客运规章
知识图谱用途：智能问答
关系类型列表：乘客-购买-票务, 规章条款-约束-乘客行为, 车辆-运行于-线路, 站点-属于-线路, 违规行为-导致-处罚措施

**当前文件主题**：{file_theme}
**当前知识图谱用途**：{kg_purpose}
请根据当前文件主题和知识图谱用途，生成初步的关系类型列表。
"""

relation_type_generation_prompt = PromptTemplate(
    template=relation_type_generation_prompt_template,
    input_variables=["file_theme", "kg_purpose"]
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
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

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

# 生成实体类型
def generate_entity_types(file_theme, kg_purpose):
    try:
        prompt_value = entity_type_generation_prompt.format_prompt(file_theme=file_theme, kg_purpose=kg_purpose)
        output = llm(prompt_value.to_string())
        return output.strip().split(", ")
    except Exception as e:
        logging.error(f"实体类型生成失败: {e}")
        return []

# 生成关系类型
def generate_relation_types(file_theme, kg_purpose):
    try:
        prompt_value = relation_type_generation_prompt.format_prompt(file_theme=file_theme, kg_purpose=kg_purpose)
        output = llm(prompt_value.to_string())
        return output.strip().split(", ")
    except Exception as e:
        logging.error(f"关系类型生成失败: {e}")
        return []

# 信息提取
def extract_information(document_content, file_theme, kg_purpose, entity_types, relation_types,format_instructions):
    try:
        prompt_value = extraction_generation_prompt.format_prompt(
            document_content=document_content,
            file_theme=file_theme,
            kg_purpose=kg_purpose,
            entity_types=entity_types,
            relation_types=relation_types,
            format_instructions=format_instructions
        )
        output = llm(prompt_value.to_string())
        try:
            extracted_data = output_parser_extraction.parse(output)
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

# Streamlit 界面
def main():
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
                entity_types = generate_entity_types(file_theme, kg_purpose)
                relation_types = generate_relation_types(file_theme, kg_purpose)
                st.write("生成的实体类型:", ", ".join(entity_types))
                st.write("生成的关系类型:", ", ".join(relation_types))
                st.session_state["entity_types"] = entity_types
                st.session_state["relation_types"] = relation_types
            entity_types_input = st.text_area("实体类型（逗号分隔）", value=", ".join(st.session_state.get("entity_types", [])))
            relation_types_input = st.text_area("关系类型（逗号分隔）", value=", ".join(st.session_state.get("relation_types", [])))
            if st.button("执行信息提取"):
                entity_types = entity_types_input.split(", ")
                relation_types = relation_types_input.split(", ")
                format_instructions = """The output should be a JSON list where each element contains:
                {
                    "subject_entity": {
                        "type": "entity_type",
                        "name": "entity_name",
                        "attributes": "entity_attributes"
                    },
                    "relation": "relation_type",
                    "object_entity": {
                        "type": "entity_type",
                        "name": "entity_name", 
                        "attributes": "entity_attributes"
                    }
                }"""
                extracted_info = extract_information(document_content, file_theme, kg_purpose, entity_types, relation_types, format_instructions)
                if extracted_info:
                    st.write("提取结果:", extracted_info)
                    visualize_knowledge_graph(extracted_info)
                else:
                    st.write("信息提取失败")

if __name__ == "__main__":
    main()