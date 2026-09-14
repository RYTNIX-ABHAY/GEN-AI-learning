from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
   # repo_id="zai-org/GLM-5.3",
    task="text-generation"

)
model=ChatHuggingFace(llm=llm)

chat_his=[SystemMessage(content="You are a helpful Ai assistant.")]

while True:
    user_ip=input('You : ')
    chat_his.append(HumanMessage(content=user_ip))
    if user_ip=='exit':
         break
    result=model.invoke(chat_his)
    chat_his.append(AIMessage(content=result.content))
    print("Ai :", result.content)

print(chat_his)
