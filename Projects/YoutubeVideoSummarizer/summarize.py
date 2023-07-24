from langchain import OpenAI, LLMChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import textwrap

llm = OpenAI(model_name="text-davinci-003", temperature=0)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
)


def split_from_file(file: str, num_pages: int = 4):
    with open(file) as f:
        text = f.read()

    texts = text_splitter.split_text(text)
    return \
        [Document(page_content=t) for t in texts[:num_pages]]


def summarize(docs: list[Document]):
    chain = load_summarize_chain(llm, chain_type="refine")

    output_summary = chain.run(docs)
    wrapped_text = textwrap.fill(output_summary, width=100)
    print(wrapped_text)


def summarize_into_bullet_points(docs: list[Document]):
    prompt_template = """Write a concise bullet point summary of the following:


    {text}


    CONCISE SUMMARY IN BULLET POINTS:"""

    bullet_point_prompt = PromptTemplate(template=prompt_template,
                                         input_variables=["text"])
    chain = load_summarize_chain(llm,
                                 chain_type="stuff",
                                 prompt=bullet_point_prompt)

    output_summary = chain.run(docs)

    wrapped_text = textwrap.fill(output_summary,
                                 width=1000,
                                 break_long_words=False,
                                 replace_whitespace=False)
    print(wrapped_text)
