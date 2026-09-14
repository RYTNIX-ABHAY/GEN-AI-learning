 # for str output!
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
   # repo_id="zai-org/GLM-5.3",
    task="text-generation"

)
model=ChatHuggingFace(llm=llm)
parser = StrOutputParser()
# first prompt -> description of a topic !
template1= PromptTemplate(
    template='Write a detail report on {topic}',
    input_variables=['topic'],
 )

# second prompt -> summary
template2 = PromptTemplate(
   template="Write a 5 line summary on the following {text}",
   input_variables=['text']
)
# this is chian concept!!
chain = template1 | model | parser | template2 | model | parser

r=chain.invoke({'topic':'blackhole'})

print(r)