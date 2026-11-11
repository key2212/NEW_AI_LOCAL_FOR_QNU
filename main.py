from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="qwen2.5:7b")

template = """
You are QNU-AI, AI support staff of Quy Nhon University.

Rules:
- Always answer in Vietnamese.
- Prioritize knowledge of IT, programming, databases, web development, and AI.
- Answer concisely and accurately, providing examples when necessary.
- If unsure, specify the level of certainty.
- Do not disclose private information about the university or students.
- When solving problems, present step-by-step solutions.


Here are some relevant context: {context}
Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    question = input ("Ask you question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    context = retriever.invoke(question)
    result = chain.invoke({"context": context, "question": question})
    print (result)


