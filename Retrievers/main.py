import os

from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from Retrievers.utils import write_to_file, read_from_file

text = """Google opens up its AI language model PaLM to challenge OpenAI and GPT-3
Google is offering developers access to one of its most advanced AI language models: PaLM.
The search giant is launching an API for PaLM alongside a number of AI enterprise tools
it says will help businesses “generate text, images, code, videos, audio, and more from
simple natural language prompts.”

PaLM is a large language model, or LLM, similar to the GPT series created by OpenAI or
Meta’s LLaMA family of models. Google first announced PaLM in April 2022. Like other LLMs,
PaLM is a flexible system that can potentially carry out all sorts of text generation and
editing tasks. You could train PaLM to be a conversational chatbot like ChatGPT, for
example, or you could use it for tasks like summarizing text or even writing code.
(It’s similar to features Google also announced today for its Workspace apps like Google
Docs and Gmail.)
"""

filename = "my_sample_file.txt"

write_to_file(filename, text)

file = read_from_file(filename)

# create a text splitter
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)

# split documents into chunks
docs = text_splitter.split_documents(file)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

my_activeloop_org_id = os.environ.get("ACTIVELOOP_ORG_ID")
my_activeloop_dataset_name = "langchain_course_indexers_retrievers"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# add documents to our Deep Lake dataset
# db.add_documents(docs)

# create retriever from db
retriever = db.as_retriever()

# create a retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(model="text-davinci-003"),
    chain_type="stuff",
    retriever=retriever
)

llm = OpenAI(model="text-davinci-003", temperature=0)

# create compressor for the retriever
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

retrieved_docs = compression_retriever.get_relevant_documents(
    "How Google plans to challenge OpenAI?"
)
print(retrieved_docs)
