from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import SystemMessagePromptTemplate

from retrieverOLD import create_retriever
import os
load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model_name="gpt-4.1-nano-2025-04-14", temperature=0.3)
while True:
    question = input("You: ")
    if question.lower() == "exit":
        break
    retriever = create_retriever("../data/sample_doc.txt", question)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever,return_source_documents=True)


    def rag_tool_function(question: str) -> str:
        docs = retriever.get_relevant_documents(question)
        if not docs:
            return "No relevant context found."

        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"""Context:
        {context}

        Question: {question}
        Please answer based only on the context above. If not found, say "I don't know."""""

        return prompt
    tools = [
        Tool(
            name="RAGRetriever",
            # func=lambda q: qa_chain({"query": q})["result"],
            func=rag_tool_function,
            # func=qa_chain.run,
            description="Use to answer questions from internal documents"
        )
    ]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    system_prompt = SystemMessagePromptTemplate.from_template(
        "Answer strictly using the provided context. If you do not know the answer, reference or quote the retrieved context."
    )
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="chat-conversational-react-description",
        memory=memory,
        verbose=True
        # system_message=system_prompt
    )
    docs = retriever.get_relevant_documents(question)
    print("\n[Retrieval Debug]")
    print(f"Question: {question}")
    print("Retrieved context:")
    for doc in docs:
        print("-", doc.page_content)
    with get_openai_callback() as cb:
        response = agent.invoke({"input": question})
        print("Agent:", response["output"])
        print(f"- Token Usage:")
        print(f"  - Prompt Tokens: {cb.prompt_tokens}")
        print(f"  - Completion Tokens: {cb.completion_tokens}")
        print(f"  - Total Tokens: {cb.total_tokens}")
        print(f"  - Estimated Cost (USD): ${cb.total_cost:.6f}")
        print(f"  - Total (INR): ${cb.total_cost*85.80:.6f}")
