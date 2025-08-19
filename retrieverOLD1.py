import os
import tempfile
from typing import List

# from Tools.scripts.pep384_macrocheck import parse_file
from langchain.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import pandas as pd
from PIL import Image
from langchain_community.chat_models import ChatOpenAI
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter

from langchain.schema import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader

def load_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        return TextLoader(file_path).load()

    elif ext == ".pdf":
        return PyPDFLoader(file_path).load()

    elif ext == ".csv":
        return CSVLoader(file_path).load()

    elif ext == ".xlsx":
        df = pd.read_excel(file_path)
        text = "\n".join(df.astype(str).apply(lambda x: " | ".join(x), axis=1))
        return [Document(page_content=text)]

    elif ext in (".png", ".jpg", ".jpeg"):
        text = pytesseract.image_to_string(Image.open(file_path))
        return [Document(page_content=text)]

    else:
        raise ValueError(f"Unsupported file format: {ext}")

def load_fileOLD(file_paths: List[str]) -> List[Document]:
    all_docs = []

    for file_path in file_paths:
        print(f"📄 Processing file: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == ".txt":
                loader = TextLoader(file_path)
                docs = loader.load()

            elif ext == ".pdf":
                loader = PyPDFLoader(file_path)
                docs = loader.load()

            elif ext == ".csv":
                loader = CSVLoader(file_path)
                docs = loader.load()

            elif ext in [".png", ".jpg", ".jpeg"]:
                text = pytesseract.image_to_string(Image.open(file_path))
                docs = [Document(page_content=text)]

            else:
                raise ValueError(f"Unsupported file format: {ext}")

            all_docs.extend(docs)

        except Exception as e:
            print(f"❌ Error parsing {file_path}: {str(e)}")

    return all_docs

def load_fileOLD2(file_paths: List[str]) -> List[Document]:
    all_docs = []

    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        print(f"Processing file: {file_path} with extension: {ext}")

        if ext == ".txt":
            docs = TextLoader(file_path).load()
        elif ext == ".pdf":
            docs = PyPDFLoader(file_path).load()
        elif ext == ".csv":
            docs = CSVLoader(file_path).load()
        elif ext in ('.png', '.jpg', '.jpeg'):
            text = pytesseract.image_to_string(Image.open(file_path))
            docs = [{"page_content": text}]
        elif ext == ".xlsx":
            df = pd.read_excel(file_path)
            text = df.to_string(index=False)
            docs = [{"page_content": text}]
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        all_docs.extend(docs)

    return all_docs


def load_fileOLD1(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        return TextLoader(file_path).load()
    elif ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext == ".csv":
        return CSVLoader(file_path).load()
    elif ext in [".png", ".jpg", ".jpeg"]:
        text = pytesseract.image_to_string(Image.open(file_path))
        return [Document(page_content=text)]
    elif ext == ".xlsx":
        df = pd.read_excel(file_path)
        text = df.to_string(index=False)
        docs = [{"page_content": text}]
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def handle_uploaded_files_OLD(uploaded_files) -> List[Document]:
    documents = []
    for uploaded_file in uploaded_files:
        # Save to temp file
        # with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as temp:
            temp.write(uploaded_file.read())
            documents.append(temp.name)
            # temp_path = temp.name

        # Parse file and collect documents
        # try:
        #     docs = parse_file(temp_path)
        #     documents.extend(docs)
        # except Exception as e:
        #     print(f"❌ Error parsing {uploaded_file.name}: {e}")

    return documents

def handle_uploaded_files(uploaded_files) -> List[str]:
    file_paths = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp:
            temp.write(uploaded_file.read())
            file_paths.append(temp.name)
    return file_paths

def create_retrieverOLD(file_path):
    docs = load_file(file_path)
    print(f"✅ Loaded {len(docs)} documents.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    print(f"✅ Created {len(split_docs)} chunks.")
    if not split_docs:
        raise ValueError("No documents to index. Please upload valid files.")
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(split_docs, embeddings)
    base_retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model_name="gpt-4.1-nano-2025-04-14")
    compressor = LLMChainExtractor.from_llm(llm)
    compressor_retriever = ContextualCompressionRetriever(base_retriever=base_retriever, base_compressor=compressor)
    return compressor_retriever
    # return base_retriever

def create_retriever(documents: List[Document]):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(split_docs, embeddings)
    base_retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model_name="gpt-4.1-nano-2025-04-14")
    compressor = LLMChainExtractor.from_llm(llm)
    compressor_retriever = ContextualCompressionRetriever(base_retriever=base_retriever, base_compressor=compressor)
    return compressor_retriever
    # return db.as_retriever()