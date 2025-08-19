import re
from typing import Any

import numpy as np
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter


def extract_keywords(question):
    # Simple keyword extraction: remove stopwords, punctuation, and split
    stopwords = {"is", "the", "in", "of", "a", "an", "who", "what", "where", "when", "how", "to", "from", "and"}
    words = re.findall(r'\w+', question.lower())
    return [w for w in words if w not in stopwords]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



class FilteredRetriever(BaseRetriever):
    retriever:Any
    embeddings:Any
    similarity_threshold: float= 0.6
    def __init__(self, retriever, embeddings, similarity_threshold=0.7):
        super().__init__(retriever=retriever, embeddings=embeddings, similarity_threshold=similarity_threshold)

    def get_relevant_documents(self, query):
        retrieved = self.retriever.get_relevant_documents(query)
        query_emb = self.embeddings.embed_query(query)
        relevant = []
        for doc in retrieved:
            doc_emb = self.embeddings.embed_query(doc.page_content)
            sim = cosine_similarity(query_emb, doc_emb)
            if sim >= self.similarity_threshold:
                relevant.append(doc)
        return relevant

def create_retriever(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()

    # Use RecursiveCharacterTextSplitter for better chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n"]
    )
    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    # Optional: Use similarity threshold filtering
    # return db.as_retriever(search_type="similarity")
    # return db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 1})
    base_retriever = db.as_retriever(search_kwargs={"k": 5})
    llm= ChatOpenAI(model_name="gpt-4.1-nano-2025-04-14")
    compressor = LLMChainExtractor.from_llm(llm)
    compressor_retriever = ContextualCompressionRetriever(base_retriever=base_retriever, base_compressor=compressor)
    return compressor_retriever

def create_retriever2(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    for doc in docs:
        doc.metadata["source"] = "sample_doc.txt"
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db.as_retriever()

def create_retriever1(doc_path, question, similarity_threshold=0.6):
    loader = TextLoader(doc_path)
    raw_docs = loader.load()
    # keywords = extract_keywords(question)
    # docs = []
    # for doc in raw_docs:
    #     content = doc.page_content.lower()
    #     if all(kw in content for kw in keywords):
    #         topic = "_".join(keywords)
    #     else:
    #         topic = "general"
    #     docs.append(Document(page_content=doc.page_content, metadata={"topic": topic}))
    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0, separators=["\n"])
    split_docs = splitter.split_documents(raw_docs)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(split_docs, embeddings)
    # retriever = db.as_retriever(search_type="similarity_score_threshold",search_kwargs={"score_threshold": 0.8, "filter": {"topic": "_".join(keywords)}})
    retriever = db.as_retriever(search_type="similarity_score_threshold",search_kwargs={"score_threshold": 0.8, "k": 5})
    # return retriever
    return FilteredRetriever(retriever, embeddings, similarity_threshold)






























# def create_retriever(doc_path):
#     loader = TextLoader(doc_path)
#     documents = loader.load()
#     # Split by sentences for precise retrieval
#     splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0, separators=["\n"])
#     docs = splitter.split_documents(documents)
#     embeddings = OpenAIEmbeddings()
#     db = FAISS.from_documents(docs, embeddings)
#     # Retrieve only the top 1 most relevant chunk
#     return db.as_retriever(search_kwargs={"k":5})


# def create_retriever(doc_path):
#     loader = TextLoader(doc_path)
#     documents = loader.load()
#     splitter = CharacterTextSplitter(chunk_size=30, chunk_overlap=5)
#     docs = splitter.split_documents(documents)
#     embeddings = OpenAIEmbeddings()
#     db = FAISS.from_documents(docs, embeddings)
#     return db.as_retriever()
