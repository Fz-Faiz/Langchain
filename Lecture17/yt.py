from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task="conversational"
)


model = ChatHuggingFace(llm = llm)

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

video_id = "Gfr50f6ZBvo" 
try:
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id) 
    transcript = " ".join(chunk.text for chunk in fetched_transcript)
except TranscriptsDisabled:
    print("No captions available for this video.")
    
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = splitter.create_documents([transcript])

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})



prompt = PromptTemplate(
    template="""
       You are a helpfull assistant.
       Answer ONLY from the provided transcript context.
       If the context is insufficent, just say you don't know.
       
       {context}
       Question: {question}
    """,
    input_variables=['context','question']
)


parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})


main_chain = parallel_chain | prompt | llm | parser
response = main_chain.invoke('Can you summarize the video')
print(response)