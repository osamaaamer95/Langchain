from os import environ

from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

from enums import Model


class DeepLakeAgent:
    def __init__(self, model: Model, temperature: float):
        self.llm = OpenAI(model=model.value, temperature=temperature)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        activeloop_org_id = environ.get("ACTIVELOOP_ORG_ID")
        activeloop_dataset_name = environ.get("ACTIVELOOP_DATASET")

        # create Deep Lake dataset
        dataset_path = f"hub://{activeloop_org_id}/{activeloop_dataset_name}"
        self.db = DeepLake(dataset_path=dataset_path, embedding_function=self.embeddings)

    def add_documents_to_deeplake(self, texts: list[str]):
        # instantiate the LLM and embeddings models

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

        docs = text_splitter.create_documents(texts)

        # add documents to our Deep Lake dataset
        self.db.add_documents(docs)

    def ask_question_from_deeplake(self, question: str):
        retrieval_qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever()
        )

        tools = [
            Tool(
                name="Retrieval QA System",
                func=retrieval_qa.run,
                description="Useful for answering questions."
            ),
        ]

        agent = initialize_agent(
            tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

        return agent.run(question)
