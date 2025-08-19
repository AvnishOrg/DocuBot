import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from agents import create_agent
from retriever import load_file, create_retriever

load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="RAG + Agentic AI Chatbot", layout="wide")
st.title("🤖 DocuRobo — AI Chatbot with RAG & Memory")

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "all_docs" not in st.session_state:
    st.session_state.all_docs = []

uploaded_files = st.file_uploader(
    "📂 Upload documents (PDF, TXT, CSV, XLSX, Images)",
    type=["txt", "pdf", "csv", "xlsx", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    # all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            docs = load_file(tmp_path)
            # all_docs.extend(docs)
            st.session_state.all_docs.extend(docs)
        except Exception as e:
            st.error(f"❌ Error parsing {uploaded_file.name}: {str(e)}")

    if st.session_state.all_docs:
        retriever, vector_db = create_retriever(st.session_state.all_docs)
        st.session_state.retriever = retriever
        st.session_state.vector_db = vector_db
    # if all_docs:
    #     st.session_state.retriever, st.session_state.vector_db = create_retriever(
    #         all_docs, st.session_state.vector_db
    #     )
        st.session_state.agent = create_agent(st.session_state.retriever)
        st.success("✅ Documents processed successfully!")


# Chat input
user_input = st.chat_input("Type your question...")
if user_input:
    if not st.session_state.agent:
        st.warning("Please upload a document first!")
    else:
        with st.spinner("🤖 Thinking..."):
            response = st.session_state.agent.invoke({"input": user_input})
            answer = response.get("output", "⚠ No answer generated")
            st.session_state.chat_history.append((user_input, answer))

# Display chat history
for q, a in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
