from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

from enums import OpenAIModel


def generate_company_names(model: OpenAIModel, temperature: float, query: str):
    # get reference to the llm we want to use
    llm = OpenAI(model=model.value, temperature=temperature)

    # initialize prompt template
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    return chain.run(query)
