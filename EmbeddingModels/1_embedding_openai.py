from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize the embedding model (no dimensions argument)
embedding = OpenAIEmbeddings(model='text-embedding-3-large')

# Create embedding for a query
result = embedding.embed_query("Delhi is the capital of India")

print(result)
