from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# ✅ Ensure your Gemini API key is set
# Either store it in .env as GOOGLE_API_KEY=your_key_here
# or set it manually below:
# os.environ["GOOGLE_API_KEY"] = "your_gemini_api_key_here"

# Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash"
)

# Example tool definition
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

# Invoke tool
result = multiply.invoke({"a": 3, "b": 3})
print(result)
