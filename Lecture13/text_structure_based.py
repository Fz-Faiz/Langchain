from langchain.text_splitter import RecursiveCharacterTextSplitter


text = """Zip the extension folder and host the zip on GitHub / Google Drive / your website. Provide a short README explaining how to unzip and Load unpacked as above.

Pros: easy to host, no store review.
Cons: manual steps for users; some may be wary of installing unpacked extensions.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=20,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

print(chunks)