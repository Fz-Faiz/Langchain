from langchain_community.document_loaders import WebBaseLoader
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
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)


url = 'https://www.flipkart.com/apple-2022-macbook-air-m2-8-gb-256-gb-ssd-mac-os-monterey-mly33hn-a/p/itm0946c05e6335c'
loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({'question': 'What is the product that we are talking about?','text':docs[0].page_content})

print(result)