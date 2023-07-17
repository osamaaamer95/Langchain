from typing import Any

from langchain import PromptTemplate
from langchain import HuggingFaceHub, LLMChain

template = """
Question: {question}
Answer: 
"""

prompt = PromptTemplate(
    template=template,
    input_variables=['question']
)

# initialize Hub LLM
hub_llm = HuggingFaceHub(
    repo_id='google/flan-t5-large',
    model_kwargs={'temperature': 0}
)

# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=prompt,
    llm=hub_llm
)


def ask_huggingface(query: str):
    # ask the user question about the capital of France
    print(llm_chain.run(query))


def ask_huggingface_using_list(query: list[dict, [str, Any]]):
    res = llm_chain.generate(query)
    print(res)


def ask_huggingface_using_template(qs_str: str):
    multi_template = """Answer the following questions one at a time.

    Questions:
    {questions}

    Answers:
    """
    long_prompt = PromptTemplate(template=multi_template, input_variables=["questions"])

    llm_chain = LLMChain(
        prompt=long_prompt,
        llm=hub_llm
    )

    print(llm_chain.run(qs_str))
