from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
   # repo_id="meta-llama/Llama-3.1-8B-Instruct",
    repo_id="zai-org/GLM-5.3",
    task="text-generation",
)
model=ChatHuggingFace(llm=llm)
parser = JsonOutputParser()
temp= PromptTemplate(
    template="Give me a 5 intresting lines on the following topic {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = temp | model | parser
r=chain.invoke({'topic':'GenAi'})
print(r)