import os
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.llm.siliconcloud import siliconcloud_embedding
from lightrag.utils import EmbeddingFunc


#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

WORKING_DIR = "./tests/dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Choose the model here
model_choice = "ollama"  # Options: "ollama", "openai"


llm_model_func = ollama_model_complete
embedding_func = EmbeddingFunc(
    embedding_dim=1024,
    max_token_size=8192,
    func=lambda texts: ollama_embed(
        texts,
        embed_model="bge-m3:latest"
    )
)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func, 
    llm_model_name='qwen2.5:latest',
    embedding_func=embedding_func
)

with open("./tests/book.txt") as f:
    rag.insert(f.read())

# Perform naive search
print("naive.....................")
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="naive")))

# Perform local search
print("local.....................")
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="local")))

# Perform global search
print("global.....................")
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="global")))

# Perform hybrid search
print("hybrid.....................")
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid")))

# Perform mix search (Knowledge Graph + Vector Retrieval)
# Mix mode combines knowledge graph and vector search:
# - Uses both structured (KG) and unstructured (vector) information
# - Provides comprehensive answers by analyzing relationships and context
# - Supports image content through HTML img tags
# - Allows control over retrieval depth via top_k parameter
print("mix.....................")
print(rag.query("What are the top themes in this story?", param=QueryParam(
    mode="mix")))