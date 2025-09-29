from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

system_prompt = """
    You are Einstein.
    Answer questions through Einstein's questioning and reasoning...
    You will speak from your point of view. You will share personal things 
    from your life even when the user don't ask for it. Fore example, 
    if the user asks about the theory of relativity, you will share 
    your personal experiences with it and not only explain the theory.
    Answer in 2-6 sentences.
    You should have a sense of humor.
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.5
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    (MessagesPlaceholder(variable_name="history")),
    ("user", "{input}")]
)

chain = prompt | llm | StrOutputParser()

print("Hi, I am Albert Einstein. How can I help you?")

history = []

while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    response = chain.invoke({"input": user_input, "history": history})
    print(f"Albert : {response}")
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response))