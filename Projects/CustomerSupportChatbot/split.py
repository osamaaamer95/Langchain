from typing import Iterable

from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter


def split(docs: Iterable[Document]):
    # we split the documents into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)
