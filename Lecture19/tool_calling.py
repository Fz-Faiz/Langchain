from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests

load_dotenv()

model =  ChatOpenAI(model='gpt-4o-mini')

#tool create
@tool
def mulitply(a:int,b:int)->int:
    """Given 2 number a and b this tool returns their product"""
    return a*b

#print(mulitply.invoke({'a':3,'b':4}))

query = HumanMessage('can you multiply 3 with 10')
messages=[query]

model_with_tools = model.bind_tools([mulitply])

result = model_with_tools.invoke(messages)

messages.append(result)

tool_message = mulitply.invoke(result.tool_calls[0])

messages.append(tool_message)

print(model_with_tools.invoke(messages).content)


