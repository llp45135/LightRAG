import os
from langchain.document_loaders import PyPDFLoader  # 或者其他适合你的文件类型的加载器
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema, OutputFixingParser
import json

# ---------------------- 配置 OpenAI API ----------------------
# 请将 'YOUR_OPENAI_API_KEY' 替换成你自己的 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = 'YOUR_OPENAI_API_KEY'

# ---------------------- 定义输出结构 (信息提取) ----------------------
response_schemas_extraction = [
    ResponseSchema(name="主体实体", description="知识图谱三元组关系中的主体实体，包含类型、名称和属性"),
    ResponseSchema(name="关系", description="知识图谱三元组关系中的关系类型"),
    ResponseSchema(name="客体实体", description="知识图谱三元组关系中的客体实体，包含类型、名称和属性"),
    ResponseSchema(name="来源", description="提取信息的来源，例如：章节标题、条款编号")
]
output_parser_extraction = StructuredOutputParser.from_response_schemas(response_schemas_extraction)
format_instructions_extraction = output_parser_extraction.get_format_instructions()

# ---------------------- 定义提示词模板 (信息提取) ----------------------
extraction_prompt_template = """请从以下客运规章文件中提取信息，构建知识图谱。
你需要识别文件中的**实体**和**实体之间的关系**，并将结果以结构化 JSON 数据形式输出。
请尽可能提取文件中所有与客运规章相关的信息。

**文件主题:** {file_subject}
**知识图谱用途:** {knowledge_graph_purpose}
**目标实体类型列表:** {entity_types}
**目标关系类型列表:** {relation_types}

**待提取的客运规章文件内容:**
{document_content}

**输出格式要求:**
请严格按照以下格式输出 JSON 结构化数据:
{format_instructions}
"""

extraction_prompt = PromptTemplate(
    template=extraction_prompt_template,
    input_variables=["document_content", "file_subject", "knowledge_graph_purpose", "entity_types", "relation_types"],
    partial_variables={"format_instructions": format_instructions_extraction}
)

# ---------------------- 定义输出结构 (实体类型生成) ----------------------
response_schemas_entity_types = [
    ResponseSchema(name="实体类型", description="从文件中提取的潜在实体类型"),
    ResponseSchema(name="描述", description="实体类型的简要描述或解释")
]
output_parser_entity_types = StructuredOutputParser.from_response_schemas(response_schemas_entity_types)
format_instructions_entity_types = output_parser_entity_types.get_format_instructions()

# ---------------------- 定义提示词模板 (实体类型生成 - 示例提示词 2 细粒度) ----------------------
entity_type_generation_prompt_template = """请详细阅读以下客运规章文件，并识别、提取文件中所有重要的名词和名词短语。
这些名词和名词短语可能代表知识图谱中的实体类型。
请尽可能详细地列举，避免遗漏。
对于每个提取出的名词或名词短语，请简要解释它在客运规章文件中可能代表的实体类型意义。

**待分析的客运规章文件内容:**
{document_content}

**输出格式要求:**
请严格按照以下格式输出 JSON 结构化数据:
{format_instructions}
"""

entity_type_generation_prompt = PromptTemplate(
    template=entity_type_generation_prompt_template,
    input_variables=["document_content"],
    partial_variables={"format_instructions": format_instructions_entity_types}
)


# ---------------------- 定义输出结构 (关系类型生成) ----------------------
response_schemas_relation_types = [
    ResponseSchema(name="关系类型", description="从文件中提取的潜在关系类型"),
    ResponseSchema(name="解释", description="关系类型的简要解释和实体例子")
]
output_parser_relation_types = StructuredOutputParser.from_response_schemas(response_schemas_relation_types)
format_instructions_relation_types = output_parser_relation_types.get_format_instructions()


# ---------------------- 定义提示词模板 (关系类型生成 - 示例提示词 2 细粒度) ----------------------
relation_type_generation_prompt_template = """请详细阅读以下客运规章文件，并识别、提取文件中所有重要的动词和动词短语。
这些动词和动词短语可能代表知识图谱中实体之间存在的关系类型。
请尽可能详细地列举，并结合上下文理解，解释这些动词或动词短语在客运规章文件中可能代表的关系类型意义，并给出涉及到的实体例子。

**待分析的客运规章文件内容:**
{document_content}

**输出格式要求:**
请严格按照以下格式输出 JSON 结构化数据:
{format_instructions}
"""

relation_type_generation_prompt = PromptTemplate(
    template=relation_type_generation_prompt_template,
    input_variables=["document_content"],
    partial_variables={"format_instructions": format_instructions_relation_types}
)


# ---------------------- 定义 LLM 模型 ----------------------
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0) # 你可以根据需要选择其他模型

# ---------------------- 函数：加载文档 ----------------------
def load_document(file_path):
    """加载 PDF 文档，处理文件加载异常"""
    try:
        loader = PyPDFLoader(file_path) # 根据你的文件类型选择合适的加载器
        documents = loader.load()
        return documents[0].page_content # 假设文档只有一个页面，或者只处理第一页
    except Exception as e:
        print(f"**错误: 加载文件失败: {e}**")
        return None

# ---------------------- 函数：生成实体类型 ----------------------
def generate_entity_types(document_content):
    """使用 LLM 生成潜在的实体类型列表，处理 LLM 调用异常"""
    try:
        prompt_value = entity_type_generation_prompt.format_prompt(document_content=document_content)
        output = llm(prompt_value.to_string())
        extracted_data = output_parser_entity_types.parse(output)
        return extracted_data
    except Exception as e:
        print(f"**错误: 实体类型生成失败: {e}**")
        return None

# ---------------------- 函数：生成关系类型 ----------------------
def generate_relation_types(document_content):
    """使用 LLM 生成潜在的关系类型列表，处理 LLM 调用异常"""
    try:
        prompt_value = relation_type_generation_prompt.format_prompt(document_content=document_content)
        output = llm(prompt_value.to_string())
        extracted_data = output_parser_relation_types.parse(output)
        return extracted_data
    except Exception as e:
        print(f"**错误: 关系类型生成失败: {e}**")
        return None


# ---------------------- 函数：信息提取 ----------------------
def extract_information_from_document(document_content, file_subject, knowledge_graph_purpose, entity_types, relation_types):
    """使用 Langchain 和 LLM 从文档中提取信息，处理 LLM 调用和 JSON 解析异常"""
    try:
        prompt_value = extraction_prompt.format_prompt(
            document_content=document_content,
            file_subject=file_subject,
            knowledge_graph_purpose=knowledge_graph_purpose,
            entity_types=entity_types,
            relation_types=relation_types
        )
        output = llm(prompt_value.to_string())

        try:
            # 尝试解析 LLM 输出的 JSON
            extracted_data = output_parser_extraction.parse(output)
            return extracted_data
        except Exception as e_json_parse:
            # 如果 JSON 解析失败，尝试使用 OutputFixingParser 修复
            print(f"**JSON 解析失败，尝试修复: {e_json_parse}**")
            fixing_parser = OutputFixingParser.from_parser(parser=output_parser_extraction)
            extracted_data_fixed = fixing_parser.parse(output)
            return extracted_data_fixed

    except Exception as e_llm_call:
        print(f"**错误: LLM 信息提取调用失败: {e_llm_call}**")
        return None


# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    file_path = '广客发[2007]207号关于发布《广州铁路（集团）公司售票组织管理暂行办法》的通知.pdf' # 替换成你的文件路径
    document_content = load_document(file_path)

    if document_content: # 确保文档成功加载

        # ----- 半自动生成实体和关系类型 -----
        print("----- 正在半自动生成潜在的实体类型... -----")
        generated_entity_types_raw = generate_entity_types(document_content)
        if generated_entity_types_raw:
            print("----- 潜在的实体类型 (待人工审核): -----")
            print(json.dumps(generated_entity_types_raw, indent=4, ensure_ascii=False))
        else:
            print("**实体类型生成失败，请检查错误信息**")

        print("\n----- 正在半自动生成潜在的关系类型... -----")
        generated_relation_types_raw = generate_relation_types(document_content)
        if generated_relation_types_raw:
            print("----- 潜在的关系类型 (待人工审核): -----")
            print(json.dumps(generated_relation_types_raw, indent=4, ensure_ascii=False))
        else:
            print("**关系类型生成失败，请检查错误信息**")

        print("\n**请人工审核并精炼上面生成的实体类型和关系类型列表，然后在下面手动设置参数**")
        input("按 Enter 键继续信息提取...") # 暂停，等待人工审核

        # ----- 用户自定义参数 (需要根据人工审核结果和文件特点设置) -----
        file_subject = "广州铁路（集团）公司售票组织管理暂行办法"
        knowledge_graph_purpose = "智能问答、合规性检查、内部培训"
        entity_types = "铁路集团公司, 铁道部, 车站, 车务段, 客运段, 售票组织, 票额, 席位, 调度命令, 客票系统, 客运业务数据, 售票人员, 旅客, 车票, 规章条款, 违规行为, 处罚措施" #  这里使用了之前讨论的实体类型，请根据人工审核结果修改
        relation_types = "制定, 规定, 执行, 管理, 负责, 属于, 分配, 生成, 指导, 依据, 维护, 职责, 购买, 发售, 约束, 导致, 应用于" #  这里使用了之前讨论的关系类型，请根据人工审核结果修改


        # ----- 执行信息提取 -----
        print("\n----- 正在执行信息提取... -----")
        extracted_info = extract_information_from_document(
            document_content=document_content,
            file_subject=file_subject,
            knowledge_graph_purpose=knowledge_graph_purpose,
            entity_types=entity_types,
            relation_types=relation_types
        )

        if extracted_info: # 确保信息提取成功
            # ----- 打印 JSON 结果 -----
            print("\n----- 信息提取结果 (JSON 格式): -----")
            print(json.dumps(extracted_info, indent=4, ensure_ascii=False))
            print("\n----- 信息提取完成，请检查 JSON 输出结果 -----")
        else:
            print("**信息提取失败，请检查错误信息**")

        print("\n----- 提示词工程建议 -----")
        print("- 可以尝试修改和优化 'extraction_prompt_template', 'entity_type_generation_prompt_template', 'relation_type_generation_prompt_template' 提示词，以提高信息提取的准确性和完整性。")
        print("- 可以尝试调整 'OpenAI' 模型的参数，例如 'temperature'，探索不同的模型输出风格。")

        print("\n----- 未来改进方向 -----")
        print("- 可以开发用户界面，方便用户上传文件、设置参数、查看和编辑结果。")
        print("- 可以将提取的 JSON 数据转换为知识图谱的存储格式 (例如 RDF 三元组, Neo4j 图数据库格式)，并构建知识图谱。")
        print("- 可以加入更完善的错误处理和日志记录机制。")
        print("- 可以考虑使用更强大的 LLM 模型，例如 GPT-4，以获得更好的信息提取效果。")
