from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")
search_tool = DuckDuckGoSearchRun(name='search')
@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

agent = create_agent(
    model=model,
    tools=[search],

)

result = agent.invoke({"messages":"Find the capital of Madhya Pradesh, then find its current weather condition."})
print(result)
