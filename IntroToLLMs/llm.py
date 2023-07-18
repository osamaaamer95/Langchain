# LLMs and Chat Models each have their advantages and disadvantages. LLMs are powerful and flexible, capable of
# generating text for a wide range of tasks. However, their API is less structured compared to Chat Models.

from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = OpenAI(
    model_name="text-davinci-003",
    temperature=0.5
)

prompt = PromptTemplate(
    input_variables=["product"],
    template="Tell me something about {product} and how it helps humanity."
)

chain = LLMChain(
    llm=llm, prompt=prompt
)

print(chain.run("Laptops"))
