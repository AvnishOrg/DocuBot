import streamlit as st
import os
from agents import create_agent, llm
from retriever import load_file
from retrieverOLD1 import handle_uploaded_files

st.set_page_config(page_title="📄 RAG Chatbot", layout="centered")
st.title("🤖 DocuBot - Ask Your Docs")

TEMP_DIR = "temp_data"
os.makedirs(TEMP_DIR, exist_ok=True)

# Upload Files
uploaded_files = st.file_uploader("📁 Upload one or more documents", type=["pdf", "txt", "csv", "png", "jpg", "jpeg", "xlsx"],
                                  accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing uploaded files..."):
        temp_paths = handle_uploaded_files(uploaded_files)  # Save files temporarily
        parsed_documents = load_file(temp_paths)  # Parse and load documents

        if parsed_documents:
            st.session_state.agent = create_agent(parsed_documents)
            st.session_state.chat_history = []  # reset chat history
            st.success(f"✅ Loaded {len(parsed_documents)} documents.")
        else:
            st.error("❌ No valid content found in uploaded files.")

if not uploaded_files and "agent" not in st.session_state:
    from langchain.agents import initialize_agent, Tool
    from langchain.memory import ConversationBufferMemory

    def fallback_tool(input):
        return llm.predict(input)

    tools = [Tool(name="FallbackTool", func=fallback_tool, description="Default fallback tool")]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    st.session_state.agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="chat-conversational-react-description",
        memory=memory,
        verbose=True
    )


# Chat Interface
if "agent" in st.session_state:
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("💬 Ask a question:", key="chat_input")
        submitted = st.form_submit_button("Send")

    if submitted and user_input:
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.agent.invoke({"input": user_input})
                answer = response.get("output", "❌ No answer found.")
            except Exception as ex:
                answer = llm.predict(user_input)
            st.session_state.chat_history.append((user_input, answer))

    # Chat history
    if st.session_state.get("chat_history"):
        for i, (q, a) in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")
            st.markdown("---")
