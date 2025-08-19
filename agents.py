from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain_community.chat_models import ChatOpenAI


def create_agent(retriever):
    """Create an Agent with RAG + Memory + Fallback to OpenAI if no docs found"""

    # LLM
    llm = ChatOpenAI(model_name="gpt-4.1-nano-2025-04-14", temperature=0.3)

    # Conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Retrieval QA Chain (for RAG)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # Define tool for RAG
    def rag_with_fallback(query):
        """First try RAG, if no relevant docs → fallback to direct ChatGPT answer"""
        result = qa_chain.invoke({"query": query})

        # Check if we got relevant documents
        source_docs = result.get("source_documents", [])
        if source_docs and any(doc.page_content.strip() for doc in source_docs):
            return result.get("result", "⚠ No answer from RAG.")
        else:
            # Fallback: Ask ChatGPT directly (no context)
            return llm.invoke(query).content

    tools = [
        Tool(
            name="RAGRetriever",
            func=rag_with_fallback,
            description="Use this to answer questions from uploaded documents; falls back to OpenAI if no context found"
        )
    ]

    # Initialize the conversational agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="chat-conversational-react-description",
        memory=memory,
        verbose=True
    )

    return agent
