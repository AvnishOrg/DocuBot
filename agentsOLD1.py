import os

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

from retriever import create_retriever

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0.3, model_name="gpt-4.1-nano-2025-04-14")

def create_agent(documents):
    retriever = create_retriever(documents)  # already parsed
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     retriever=retriever,
    #     return_source_documents=True  # Optional: returns sources
    # )
    def rag_tool_function(question: str) -> str:
        docs = retriever.get_relevant_documents(question)
        if not docs:
            return "No relevant context found."

        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"""Context:
        {context}

        Question: {question}
        Provide answer based only on the context above. If not found any answer then provide answer from openAI"""""

        return prompt
    tools = [
        Tool(
            name="RAGRetriever",
            # func=qa_chain.run,
            func=rag_tool_function,
            description="Useful for answering questions from uploaded documents"
        )
    ]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="chat-conversational-react-description",
        memory=memory,
        verbose=True
    )
    return agent

def create_agentOLD(doc_path):
    retriever = create_retriever(doc_path)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    def rag_tool_function(question: str) -> str:
        docs = retriever.get_relevant_documents(question)
        print("\n[Retrieval Debug]")
        print(f"Question: {question}")
        print("Retrieved context:")
        for doc in docs:
            print("-", doc.page_content)
        if not docs:
            return "No relevant context found."

        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"""Context:
        {context}

        Question: {question}
        Please answer based on the context above. If not use open AI and provide answer."""""

        return prompt
    tools = [
        Tool(name="RAGRetriever", func=rag_tool_function, description="Answer using documents")
    ]

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # system_prompt = SystemMessagePromptTemplate.from_template(
    #     "Answer strictly using the provided context. If you do not know the answer, reference or quote the retrieved context."
    # )
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        memory=memory,
        agent="chat-conversational-react-description",
        verbose=True
    )

    return agent


