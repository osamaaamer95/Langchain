from langchain.llms import OpenAI
from enums import Model


def generate_openai_response(model: Model, temperature: float, prompt: str):
    # this is a simple way to call the model via API
    llm = OpenAI(model=model.value, temperature=temperature)

    # generate and print the output
    return llm(prompt)
