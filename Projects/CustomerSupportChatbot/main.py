"""
Pre-processing:
1. Documents
2. Content scraping
3. Splitting into chunks
4. Embedding
5. Store in vector database

Documents retrieval:
1. User enters a query
2. Relevant results retrieved using vector database retriever
3. Prompt prepared
4. Answer generated and returned
"""

from os import environ

from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

from Projects.CustomerSupportChatbot.prompt import prompt
from Projects.CustomerSupportChatbot.scrape import scrape
from Projects.CustomerSupportChatbot.split import split
from Projects.CustomerSupportChatbot.urls import urls

# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
# create Deep Lake dataset
my_activeloop_org_id = environ.get("ACTIVELOOP_ORG_ID")
my_activeloop_dataset_name = "langchain_course_customer_support"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)


def preprocess():
    raw_documents = scrape(urls)

    # split raw article data
    docs = split(raw_documents)

    # add split documents to our Deep Lake dataset
    db.add_documents(docs)


def answer_query(query: str):
    # retrieve relevant chunks
    docs = db.similarity_search(query)
    retrieved_chunks = [doc.page_content for doc in docs]

    # format the prompt
    chunks_formatted = "\n\n".join(retrieved_chunks)
    prompt_formatted = prompt.format(chunks_formatted=chunks_formatted, query=query)

    # generate answer
    llm = OpenAI(model="text-davinci-003", temperature=0)
    answer = llm(prompt_formatted)
    print(answer)


# scrape, split, add to vector db
preprocess()

# retrieve similar docs, join into a single doc, use llm to generate an answer
answer_query("How to check disk usage in Linux?")
