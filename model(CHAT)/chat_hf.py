from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace


load_dotenv()

# Hugging Face Inference API
llm = HuggingFaceEndpoint(
    #repo_id="meta-llama/Llama-3.1-8B-Instruct",
    repo_id="zai-org/GLM-5.3",
    task="text-generation"

)
m=ChatHuggingFace(llm=llm)
r=m.invoke("What is  capital of India in one word?")

print(r.content)