from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(chunk_size=400, chunk_overlap=20)

loader = PyPDFLoader("resume.pdf")

pages = loader.load_and_split()

texts = text_splitter.split_documents(pages)

print(f"You have {len(texts)} documents")

print(texts)
