from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

from enums import ChatModel


def summarize_pdf(path: str):
    # Initialize language model
    llm = ChatOpenAI(model_name=ChatModel.GPT3_5.value, temperature=0)

    # Load the summarization chain
    summarize_chain = load_summarize_chain(llm)

    # Load the document using PyPDFLoader
    document_loader = PyPDFLoader(file_path=path)
    document = document_loader.load()

    # Summarize the document
    summary = summarize_chain(document)
    print(summary['output_text'])
