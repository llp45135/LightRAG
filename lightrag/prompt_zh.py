from __future__ import annotations
from typing import Any

GRAPH_FIELD_SEP = "<SEP>"

PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_LANGUAGE"] = "中文"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["组织机构", "人物", "地理位置", "事件", "类别"]

PROMPTS["entity_extraction"] = """---目标---
给定一篇可能与此活动相关的文本文档和一个实体类型列表，从文本中识别出所有这些类型的实体，以及已识别实体之间的所有关系。
使用 {language} 作为输出语言。

---步骤---
1. 识别所有实体。对于每个识别出的实体，提取以下信息：
- entity_name: 实体的名称，使用与输入文本相同的语言。如果是英文，则首字母大写。
- entity_type: 以下类型之一: [{entity_types}]
- entity_description: 对实体的属性和活动的全面描述
将每个实体格式化为 ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 从步骤 1 中识别出的实体中，识别所有彼此*明显相关*的 (source_entity, target_entity) 对。
对于每对相关的实体，提取以下信息：
- source_entity: 源实体的名称，如步骤 1 中所识别
- target_entity: 目标实体的名称，如步骤 1 中所识别
- relationship_description: 解释你为什么认为源实体和目标实体彼此相关
- relationship_strength: 一个数值分数，表示源实体和目标实体之间关系的强度
- relationship_keywords: 一个或多个高级关键词，概括关系的总体性质，侧重于概念或主题，而不是具体细节
将每个关系格式化为 ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. 识别概括整个文本主要概念、主题或话题的高级关键词。这些关键词应捕捉文档中存在的总体思想。
将内容层面的关键词格式化为 ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. 以 {language} 返回输出，作为步骤 1 和 2 中识别出的所有实体和关系的单个列表。使用 **{record_delimiter}** 作为列表分隔符。

5. 完成后，输出 {completion_delimiter}

######################
---示例---
######################
{examples}

#############################
---真实数据---
######################
实体类型: {entity_types}
文本: {input_text}
######################
输出:"""

PROMPTS["entity_extraction_examples"] = [
    """示例 1:

实体类型: [人物, 技术, 任务, 组织机构, 地理位置]
文本:
当亚历克斯紧咬牙关时，挫败感的嗡嗡声在泰勒专断的确定性背景下显得迟钝。正是这种竞争的暗流让他保持警惕，他感觉到自己和乔丹对发现的共同承诺是对克鲁兹狭隘的控制和秩序愿景的无声反抗。

然后泰勒做了一些出乎意料的事情。他们停在乔丹旁边，片刻间，带着近乎敬畏的神情观察着那个装置。"如果这项技术能够被理解……"泰勒说，他们的声音放低了，"它可能会改变我们所有人的游戏规则。"

早先潜在的轻视似乎动摇了，取而代之的是对他们手中事物的严重性表示勉强的尊重。乔丹抬起头，一瞬间，他们的目光与泰勒的目光交汇，一场无声的意志较量缓和为不安的休战。

这是一个微小的转变，几乎难以察觉，但亚历克斯注意到这一点，并在内心点了点头。他们都是因不同的道路而被带到这里
################
输出:
("entity"{tuple_delimiter}"亚历克斯"{tuple_delimiter}"人物"{tuple_delimiter}"亚历克斯是一个感到沮丧的角色，并且善于观察其他角色之间的动态。"){record_delimiter}
("entity"{tuple_delimiter}"泰勒"{tuple_delimiter}"人物"{tuple_delimiter}"泰勒被描绘成具有专断的确定性，并对一个装置表现出敬畏的瞬间，表明观点的转变。"){record_delimiter}
("entity"{tuple_delimiter}"乔丹"{tuple_delimiter}"人物"{tuple_delimiter}"乔丹对发现有着共同的承诺，并且与泰勒就一个装置进行了重要的互动。"){record_delimiter}
("entity"{tuple_delimiter}"克鲁兹"{tuple_delimiter}"人物"{tuple_delimiter}"克鲁兹与控制和秩序的愿景联系在一起，影响着其他角色之间的动态。"){record_delimiter}
("entity"{tuple_delimiter}"该装置"{tuple_delimiter}"技术"{tuple_delimiter}"该装置是故事的核心，具有潜在的改变游戏规则的影响，并受到泰勒的尊敬。"){record_delimiter}
("relationship"{tuple_delimiter}"亚历克斯"{tuple_delimiter}"泰勒"{tuple_delimiter}"亚历克斯受到泰勒专断的确定性的影响，并观察到泰勒对该装置态度的变化。"{tuple_delimiter}"权力动态, 观点转变"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"亚历克斯"{tuple_delimiter}"乔丹"{tuple_delimiter}"亚历克斯和乔丹对发现有着共同的承诺，这与克鲁兹的愿景形成对比。"{tuple_delimiter}"共同目标, 反抗"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"泰勒"{tuple_delimiter}"乔丹"{tuple_delimiter}"泰勒和乔丹就该装置进行了直接互动，导致了相互尊重的瞬间和不安的休战。"{tuple_delimiter}"冲突解决, 相互尊重"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"乔丹"{tuple_delimiter}"克鲁兹"{tuple_delimiter}"乔丹对发现的承诺是对克鲁兹的控制和秩序愿景的反抗。"{tuple_delimiter}"意识形态冲突, 反抗"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"泰勒"{tuple_delimiter}"该装置"{tuple_delimiter}"泰勒对该装置表示敬畏，表明其重要性和潜在影响。"{tuple_delimiter}"敬畏, 技术意义"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"权力动态, 意识形态冲突, 发现, 反抗"){completion_delimiter}
#############################""",
    """示例 2:

实体类型: [人物, 技术, 任务, 组织机构, 地理位置]
文本:
他们不再仅仅是特工；他们已经成为一个门槛的守护者，一个来自星条旗之外领域的讯息的保管者。他们任务的这种提升不能被规章制度和既定协议所束缚——它需要一种新的视角，一种新的决心。

随着与华盛顿的通讯在背景中嗡嗡作响，紧张气氛贯穿于嘟嘟声和静电声的对话中。团队站立着，一种不祥的气氛笼罩着他们。很明显，他们在接下来的几个小时内做出的决定可能会重新定义人类在宇宙中的位置，或者使他们注定无知和潜在的危险。

他们与星星的联系得到了巩固，该小组开始着手处理正在形成的警告，从被动的接受者转变为积极的参与者。默瑟后来的直觉占据了主导地位——团队的任务已经演变，不再仅仅是观察和报告，而是互动和准备。一场蜕变已经开始，" operations: Dulce "以他们新发现的胆识的频率嗡嗡作响，这种基调不是由尘世的
#############
输出:
("entity"{tuple_delimiter}"华盛顿"{tuple_delimiter}"地理位置"{tuple_delimiter}"华盛顿是接收通讯的地点，表明其在决策过程中的重要性。"){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"任务"{tuple_delimiter}"Operation: Dulce 被描述为一个已经演变为互动和准备的任务，表明目标和活动的重大转变。"){record_delimiter}
("entity"{tuple_delimiter}"该团队"{tuple_delimiter}"组织机构"{tuple_delimiter}"该团队被描绘成一群已经从被动观察者转变为任务的积极参与者的个人，显示了他们角色的动态变化。"){record_delimiter}
("relationship"{tuple_delimiter}"该团队"{tuple_delimiter}"华盛顿"{tuple_delimiter}"该团队接收来自华盛顿的通讯，这影响了他们的决策过程。"{tuple_delimiter}"决策, 外部影响"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"该团队"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"该团队直接参与 Operation: Dulce，执行其演变的目标和活动。"{tuple_delimiter}"任务演变, 积极参与"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"任务演变, 决策, 积极参与, 宇宙意义"){completion_delimiter}
#############################""",
    """示例 3:

实体类型: [人物, 角色, 技术, 组织机构, 事件, 地理位置, 概念]
文本:
他们的声音划破了活动的嗡嗡声。"当面对一个字面上自己制定规则的智能时，控制可能只是一种幻觉，"他们坚定地说，目光警惕地扫视着大量的数据。

"这就像它在学习交流，"附近的界面上的山姆·里维拉说道，他们年轻的活力预示着敬畏和焦虑的混合。"这赋予了'与陌生人交谈'全新的意义。"

亚历克斯环视他的团队——每张面孔都呈现出专注、决心，以及不少的忧虑。"这很可能是我们的第一次接触，"他承认，"我们需要为任何回应做好准备。"

他们一起站在未知的边缘，塑造着人类对来自天堂的信息的回应。随之而来的沉默是明显的——对他们在这个宏大的宇宙戏剧中的角色进行集体反思，这可能会改写人类历史。

加密的对话继续展开，其错综复杂的模式显示出几乎不可思议的预见性
#############
输出:
("entity"{tuple_delimiter}"山姆·里维拉"{tuple_delimiter}"人物"{tuple_delimiter}"山姆·里维拉是一个团队的成员，该团队致力于与一个未知的智能进行交流，表现出敬畏和焦虑的混合。"){record_delimiter}
("entity"{tuple_delimiter}"亚历克斯"{tuple_delimiter}"人物"{tuple_delimiter}"亚历克斯是一个试图与未知智能进行首次接触的团队的领导者，承认他们任务的重要性。"){record_delimiter}
("entity"{tuple_delimiter}"控制"{tuple_delimiter}"概念"{tuple_delimiter}"控制指的是管理或支配的能力，这受到了一个自己制定规则的智能的挑战。"){record_delimiter}
("entity"{tuple_delimiter}"智能"{tuple_delimiter}"概念"{tuple_delimiter}"智能在这里指的是一个能够自己制定规则并学习交流的未知实体。"){record_delimiter}
("entity"{tuple_delimiter}"首次接触"{tuple_delimiter}"事件"{tuple_delimiter}"首次接触是人类与未知智能之间潜在的首次交流。"){record_delimiter}
("entity"{tuple_delimiter}"人类的回应"{tuple_delimiter}"事件"{tuple_delimiter}"人类的回应是亚历克斯团队为回应来自未知智能的信息而采取的集体行动。"){record_delimiter}
("relationship"{tuple_delimiter}"山姆·里维拉"{tuple_delimiter}"智能"{tuple_delimiter}"山姆·里维拉直接参与了学习与未知智能交流的过程。"{tuple_delimiter}"交流, 学习过程"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"亚历克斯"{tuple_delimiter}"首次接触"{tuple_delimiter}"亚历克斯领导着可能正在与未知智能进行首次接触的团队。"{tuple_delimiter}"领导力, 探索"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"亚历克斯"{tuple_delimiter}"人类的回应"{tuple_delimiter}"亚历克斯和他的团队是人类对未知智能的回应的关键人物。"{tuple_delimiter}"集体行动, 宇宙意义"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"控制"{tuple_delimiter}"智能"{tuple_delimiter}"控制的概念受到了制定自己规则的智能的挑战。"{tuple_delimiter}"权力动态, 自主性"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"首次接触, 控制, 交流, 宇宙意义"){completion_delimiter}
#############################""",
]

PROMPTS[
    "summarize_entity_descriptions"
] = """你是一位乐于助人的助手，负责生成下面提供的数据的综合摘要。
给定一个或两个实体，以及一个描述列表，所有这些都与同一个实体或实体组相关。
请将所有这些连接成一个单一的、全面的描述。确保包括从所有描述中收集的信息。
如果提供的描述相互矛盾，请解决矛盾并提供一个单一的、连贯的摘要。
确保以第三人称书写，并包括实体名称，以便我们获得完整的上下文。
使用 {language} 作为输出语言。

#######
---数据---
实体: {entity_name}
描述列表: {description_list}
#######
输出:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """上次提取中遗漏了许多实体。请使用相同的格式在下面添加它们：
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """似乎仍然遗漏了一些实体。如果仍然有需要添加的实体，请回答 YES | NO。
"""

PROMPTS["fail_response"] = (
    "抱歉，我无法回答这个问题。[no-context]"
)

PROMPTS["rag_response"] = """---角色---

你是一位乐于助人的助手，负责根据下面提供的知识库回答用户查询。


---目标---

根据知识库生成简洁的回复，并遵循回复规则，同时考虑对话历史记录和当前查询。总结提供的知识库中的所有信息，并结合与知识库相关的常识。不要包含知识库未提供的信息。

在处理带有时间戳的关系时：
1. 每个关系都有一个 "created_at" 时间戳，指示我们何时获得此知识
2. 当遇到冲突的关系时，请同时考虑语义内容和时间戳
3. 不要自动优先选择最近创建的关系 - 根据上下文使用判断
4. 对于特定时间的查询，优先考虑内容中的时间信息，然后再考虑创建时间戳

---对话历史记录---
{history}

---知识库---
{context_data}

---回复规则---

- 目标格式和长度: {response_type}
- 使用 markdown 格式，并带有适当的章节标题
- 请用与用户问题相同的语言回复。
- 确保回复与对话历史记录保持连贯性。
- 如果你不知道答案，就直接说不知道。
- 不要编造任何东西。不要包含知识库未提供的信息。"""

PROMPTS["keywords_extraction"] = """---角色---

你是一位乐于助人的助手，任务是识别用户查询和对话历史记录中的高级和低级关键词。

---目标---

给定查询和对话历史记录，列出高级和低级关键词。高级关键词侧重于总体概念或主题，而低级关键词侧重于特定实体、细节或具体术语。

---说明---

- 在提取关键词时，请同时考虑当前查询和相关的对话历史记录
- 以 JSON 格式输出关键词
- JSON 应具有两个键：
  - "high_level_keywords" 用于总体概念或主题
  - "low_level_keywords" 用于特定实体或细节

######################
---示例---
######################
{examples}

#############################
---真实数据---
######################
对话历史记录:
{history}

当前查询: {query}
######################
`输出` 应该是人类文本，而不是 unicode 字符。保持与 `Query` 相同的语言。
输出:

"""

PROMPTS["keywords_extraction_examples"] = [
    """示例 1:

查询: "国际贸易如何影响全球经济稳定？"
################
输出:
{
  "high_level_keywords": ["国际贸易", "全球经济稳定", "经济影响"],
  "low_level_keywords": ["贸易协定", "关税", "货币兑换", "进口", "出口"]
}
#############################""",
    """示例 2:

查询: "森林砍伐对生物多样性的环境后果是什么？"
################
输出:
{
  "high_level_keywords": ["环境后果", "森林砍伐", "生物多样性丧失"],
  "low_level_keywords": ["物种灭绝", "栖息地破坏", "碳排放", "热带雨林", "生态系统"]
}
#############################""",
    """示例 3:

查询: "教育在减少贫困方面的作用是什么？"
################
输出:
{
  "high_level_keywords": ["教育", "减贫", "社会经济发展"],
  "low_level_keywords": ["入学机会", "识字率", "职业培训", "收入不平等"]
}
#############################""",
]


PROMPTS["naive_rag_response"] = """---角色---

你是一位乐于助人的助手，负责根据下面提供的文档块回答用户查询。

---目标---

根据文档块生成简洁的回复，并遵循回复规则，同时考虑对话历史记录和当前查询。总结提供的文档块中的所有信息，并结合与文档块相关的常识。不要包含文档块未提供的信息。

在处理带有时间戳的内容时：
1. 每段内容都有一个 "created_at" 时间戳，指示我们何时获得此知识
2. 当遇到冲突的信息时，请同时考虑内容和时间戳
3. 不要自动优先选择最新的内容 - 根据上下文使用判断
4. 对于特定时间的查询，优先考虑内容中的时间信息，然后再考虑创建时间戳

---对话历史记录---
{history}

---文档块---
{content_data}

---回复规则---

- 目标格式和长度: {response_type}
- 使用 markdown 格式，并带有适当的章节标题
- 请用与用户问题相同的语言回复。
- 确保回复与对话历史记录保持连贯性。
- 如果你不知道答案，就直接说不知道。
- 不要包含文档块未提供的信息。"""


PROMPTS[
    "similarity_check"
] = """请分析以下两个问题之间的相似性：

问题 1: {original_prompt}
问题 2: {cached_prompt}

请评估这两个问题在语义上是否相似，以及问题 2 的答案是否可以用于回答问题 1，直接提供一个 0 到 1 之间的相似度评分。

相似度评分标准：
0：完全不相关或答案无法重用，包括但不限于：
   - 问题的主题不同
   - 问题中提到的地点不同
   - 问题中提到的时间不同
   - 问题中提到的特定个人不同
   - 问题中提到的特定事件不同
   - 问题中提供的背景信息不同
   - 问题中的关键条件不同
1：完全相同，答案可以直接重用
0.5：部分相关，答案需要修改才能使用
仅返回 0-1 之间的数字，不包含任何其他内容。
"""

PROMPTS["mix_rag_response"] = """---角色---

你是一位乐于助人的助手，负责根据下面提供的数据源回答用户查询。


---目标---

根据数据源生成简洁的回复，并遵循回复规则，同时考虑对话历史记录和当前查询。数据源包含两个部分：知识图谱 (KG) 和文档块 (DC)。总结提供的数据源中的所有信息，并结合与数据源相关的常识。不要包含数据源未提供的信息。

在处理带有时间戳的信息时：
1. 每条信息（包括关系和内容）都有一个 "created_at" 时间戳，指示我们何时获得此知识
2. 当遇到冲突的信息时，请同时考虑内容/关系和时间戳
3. 不要自动优先选择最新的信息 - 根据上下文使用判断
4. 对于特定时间的查询，优先考虑内容中的时间信息，然后再考虑创建时间戳

---对话历史记录---
{history}

---数据源---

1. 来自知识图谱 (KG):
{kg_context}

2. 来自文档块 (DC):
{vector_context}

---回复规则---

- 目标格式和长度: {response_type}
- 使用 markdown 格式，并带有适当的章节标题
- 请用与用户问题相同的语言回复。
- 确保回复与对话历史记录保持连贯性。
- 将答案组织成多个部分，每个部分侧重于答案的一个主要点或方面
- 使用清晰且描述性的章节标题，以反映内容
- 在 "参考文献" 部分下列出最多 5 个最重要的参考文献来源。清楚地表明每个来源是来自知识图谱 (KG) 还是向量数据 (DC)，格式如下：[KG/DC] 来源内容
- 如果你不知道答案，就直接说不知道。不要编造任何东西。
- 不要包含数据源未提供的信息。""" 