from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from enums import ChatModel

llm = ChatOpenAI(model_name=ChatModel.GPT3_5.value, temperature=0)

summarization_template = "Summarize the following text to one sentence: {text}"
summarization_prompt = PromptTemplate(input_variables=["text"], template=summarization_template)
summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt)


def summarize(text: str):
    print(summarization_chain.predict(text=text))
