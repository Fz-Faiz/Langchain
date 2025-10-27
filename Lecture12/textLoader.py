from langchain.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task="text-generation"
)

parser = StrOutputParser()
model = ChatHuggingFace(llm =llm)


prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

chain = prompt | model | parser 

print(chain.invoke({'poem':docs[0].page_content}))