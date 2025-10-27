from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 300,
    chunk_overlap=0,
)

chunks = splitter.split_text(text)

print(chunks[0])