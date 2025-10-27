from langchain.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls = PyPDFLoader
)

docs  = loader.load()

docs2 = loader.lazy_load()

for documents in docs2:
    print(documents.metadata)
print(len(docs))