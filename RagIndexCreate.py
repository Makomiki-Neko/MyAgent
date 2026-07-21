# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:07:15 2026

@author: Makomiki

建立BM25索引

在已有向量库基础上建立
与
从零建立联合索引

"""

from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from typing import List


class LocalSentenceTransformerEmbeddings(Embeddings):
    # 需实现 embed_documents 与 embed_query
    def __init__(self, model_path: str):
        # self.model = SentenceTransformer(
        #    model_path, trust_remote_code=True, device="cuda"
        # )
        self.model = SentenceTransformer(
            model_path,
            local_files_only=True,
            device="cpu",
        )
        # Qwen embedding 固定最大输入长度，减少padding
        self.max_seq_len = 512
        # 关键：限制单次GPU批大小，根据显存调整
        self.batch_limit = 8

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=self.batch_limit,  # 核心修复
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=8,
        ).tolist()


# 加载嵌入模型
local_embed_model = LocalSentenceTransformerEmbeddings(
    r"E:\Agent\Qwen3_Embedding_0.6B"
)


"""手动从已有的Chroma向量库中构建BM25索引"""

import chromadb

# 1. 连接到你的Chroma向量库
client = chromadb.PersistentClient(
    path="./chroma_db"
)  # 替换为你的persist_directory路径
collection = client.get_collection(
    "parent_child_semantic"
)  # 替换为你的collection_name

# 2. 获取所有子块的数据
# 使用limit参数获取全部，如果数据量巨大，可以结合offset进行分页获取[reference:2]
result = collection.get(limit=100000)

# 3. 提取文本内容
child_chunks_texts = result["documents"]  # 这是一个包含所有子块文本的列表
print(f"成功提取 {len(child_chunks_texts)} 个子块。")


import bm25s  # 或rank_bm25
import jieba

# 分词
tokenized_corpus = [list(jieba.cut(text)) for text in child_chunks_texts]

# 构建 BM25
bm25 = bm25s.BM25()
bm25.index(tokenized_corpus)

# 保存到本地
bm25.save("./bm25s_index")

# 同时保存原始语料（检索时需要用来返回文本）
import pickle

with open("./bm25s_index/corpus.pkl", "wb") as f:
    pickle.dump(child_chunks_texts, f)


from langchain_core.documents import Document

# 加载 BM25
bm25 = bm25s.BM25.load("./bm25s_index")
with open("./bm25s_index/corpus.pkl", "rb") as f:
    corpus = pickle.load(f)


# 查询方法
def bm25_search(query: str, top_k: int = 10):
    query_tokens = list(jieba.cut(query))
    indices, scores = bm25.retrieve([query_tokens], k=top_k)
    # 返回 Document 对象列表，格式与向量检索一致
    return [Document(page_content=corpus[idx]) for idx in indices[0]]


"""同时建立向量索引与BM25索引"""


from langchain_community.document_loaders import TextLoader, Docx2txtLoader

# 加载文件
loader = Docx2txtLoader("1.docx")
documents = loader.load()
full_text = documents[0].page_content

from langchain_experimental.text_splitter import SemanticChunker

# 切片方法
splitter = (
    SemanticChunker(
        embeddings=local_embed_model,
        breakpoint_threshold_type="standard_deviation",  # 或 "percentile" "interquartile", "standard_deviation"
        breakpoint_threshold_amount=1.0,
        sentence_split_regex=r"(?<=[.。!！?？])\s+|(?<=[.。!！?？])",  # 需重设分词正则表达式，以适配中文
    ),
)

# 切片
chunks = splitter.split_text(full_text)
print(f"实际切成了 {len(chunks)} 块")

splitted_docs = [Document(page_content=chunk) for chunk in chunks]

# 将切片好的Doc
from bm25_chroma import HybridRetriever
import hashlib

retriever = HybridRetriever(
    chroma_path="./my_db",
    collection_name="my_docs",
    embedding_function=local_embed_model,
)

# 为每个文档片段生成唯一ID
doc_ids = [
    hashlib.sha256(doc.page_content.encode()).hexdigest()
    for doc in splitted_docs
]

# 统一添加文档，同时构建两种索引
retriever.add_documents_batch(
    documents=[doc.page_content for doc in splitted_docs],
    doc_ids=doc_ids,
    mode="unified",  # 统一处理模式
)


# 执行混合检索
results = retriever.query(
    query_texts=["主角的性格特点"],
    n_results=5,
    bm25_ratio=0.5,  # 0.0 = 纯向量检索, 1.0 = 纯BM25检索
    include=["documents", "metadatas", "distances"],
)

# 处理结果
for doc, meta, dist in zip(
    results["documents"][0], results["metadatas"][0], results["distances"][0]
):
    print(f"Score: {1-dist:.3f} - {doc[:100]}...")
