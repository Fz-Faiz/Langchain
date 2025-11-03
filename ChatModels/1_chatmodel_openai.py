from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model =  ChatOpenAI(model='gpt-4o-mini')

result = model.invoke("Suggest me five indian names")

print(result.content)