import os
import pytesseract
import pandas as pd
from typing import List
from PIL import Image
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


def load_file(file_path: str) -> List[Document]:
    """Load different file formats into LangChain Documents"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        return TextLoader(file_path).load()
    elif ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext == ".csv":
        return CSVLoader(file_path).load()
    elif ext in (".png", ".jpg", ".jpeg"):
        text = pytesseract.image_to_string(Image.open(file_path))
        return [Document(page_content=text)]
    elif ext in (".xls", ".xlsx"):
        try:
            df = pd.read_excel(file_path, dtype=str, engine="openpyxl")  # Force openpyxl
            text_data = "\n".join(
                df.astype(str).apply(lambda row: " | ".join(row), axis=1)
            )
            if not text_data.strip():
                raise ValueError("Excel file is empty or unreadable.")
            return [Document(page_content=text_data, metadata={"source": file_path})]
        except Exception as e:
            raise ValueError(f"Error reading Excel file {file_path}: {e}")
    # elif ext == ".xlsx":
    #     df = pd.read_excel(file_path, dtype=str)  # read as string
    #     text_data = "\n".join(df.astype(str).apply(lambda row: " | ".join(row), axis=1))
    #     return [Document(page_content=text_data, metadata={"source": file_path})]
        # df = pd.read_excel(file_path)
        # text = df.to_string(index=False)
        # return [Document(page_content=text)]
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# def create_retriever(documents: List[Document], existing_db: FAISS = None):
#     """Create or update a FAISS retriever using OpenAI embeddings"""
#     embeddings = OpenAIEmbeddings()
#
#     splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     split_docs = splitter.split_documents(documents)
#
#     if existing_db:
#         existing_db.add_documents(split_docs)
#         retriever = existing_db.as_retriever(search_kwargs={"k": 4})
#         return retriever, existing_db
#     else:
#         db = FAISS.from_documents(split_docs, embeddings)
#         retriever = db.as_retriever(search_kwargs={"k": 4})
#         return retriever, db

def create_retriever(documents):

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = FAISS.from_documents(split_docs, embeddings)

    # retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    retriever = vector_db.as_retriever(search_type="similarity_score_threshold",
                                search_kwargs={"score_threshold": 0.7, "k": 5})

    return retriever, vector_db
