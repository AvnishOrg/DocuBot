import os

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from Misleneaous.retrieverOLD import create_retriever

load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model_name="gpt-4.1-nano-2025-04-14", temperature=0.3)
retriever = create_retriever("data/sample_doc.txt")

# Define custom prompt
custom_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, respond with "I don't know."

Context:
{context}

Question:
{question}
""")

# Main loop
while True:
    question = input("You: ")
    if question.lower() == "exit":
        break

    # Step 1: Retrieve relevant context
    docs = retriever.get_relevant_documents(question)
    print("\n[Retrieval Debug]")
    print(f"Question: {question}")
    print("Retrieved context:")
    for doc in docs:
        print("-", doc.page_content)
    context = "\n".join([doc.page_content for doc in docs]) if docs else "No context found."

    # Step 2: Format the prompt
    final_prompt = custom_prompt.format(context=context, question=question)

    # Step 3: Get response
    # with get_openai_callback() as cb:
    #     response = agent.invoke({"input": question})
    #     print("Agent:", response["output"])
    #     print(f"- Token Usage:")
    #     print(f"  - Prompt Tokens: {cb.prompt_tokens}")
    #     print(f"  - Completion Tokens: {cb.completion_tokens}")
    #     print(f"  - Total Tokens: {cb.total_tokens}")
    #     print(f"  - Estimated Cost (USD): ${cb.total_cost:.6f}")
    #     print(f"  - Total (INR): ${cb.total_cost*85.80:.6f}")

    response = llm.invoke(final_prompt)
    print("Bot:", response.content)
    # print("Bot:", response)  # Print the prompt used for debugging
