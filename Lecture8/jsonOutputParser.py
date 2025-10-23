from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Create LLM endpoint
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it", 
    task="text-generation",
)

# Wrap it in a chat interface
model = ChatHuggingFace(llm=llm)

# JSON output parser
parser = JsonOutputParser()

# ✅ Fixed template
template1 = PromptTemplate(
    template=(
        "Create a fictional person and provide details in JSON format.\n"
        "{format_instructions}"
    ),
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Use the prompt directly
prompt = template1.format()

# Invoke the model
result = model.invoke(prompt)

# Parse JSON safely
try:
    final_result = parser.parse(result.content)
    print(final_result)
except Exception as e:
    print("Parsing failed. Model output was:\n", result.content)
