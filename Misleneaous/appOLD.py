import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from retrieverOLD import create_retriever
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("🤖 DocuBot")

# File Upload
uploaded_file = st.file_uploader("Upload a .txt file for context", type=["txt"])

if uploaded_file:
    file_path = f"data/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
else:
    file_path = "../data/sample_doc.txt"

# Initialize retriever
retriever = create_retriever(file_path)

# Initialize LLM and memory
llm = ChatOpenAI(model_name="gpt-4.1-nano-2025-04-14", temperature=0.3)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Prompt template
custom_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, respond with "I don't know."

Context:
{context}

Question:
{question}
""")

# Session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_input("Ask a question based on uploaded file:", key="input")

if user_input:
    # 1. Retrieve context
    docs = retriever.get_relevant_documents(user_input)
    print("\n[Retrieval Debug]")
    print(f"Question: {user_input}")
    print("Retrieved context:")
    for doc in docs:
        print("-", doc.page_content)
    context = "\n".join([doc.page_content for doc in docs]) if docs else "No context found."

    # 2. Format prompt
    prompt = custom_prompt.format(context=context, question=user_input)

    # 3. Get response
    response = llm.invoke(prompt)

    # Save to history
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", response.content))

# Display chat history
for role, msg in st.session_state.history:
    if role == "You":
        st.markdown(f"**🧑 {role}:** {msg}")
    else:
        st.markdown(f"**🤖 {role}:** {msg}")
