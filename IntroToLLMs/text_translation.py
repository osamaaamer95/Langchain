from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from enums import ChatModel

llm = ChatOpenAI(model_name=ChatModel.GPT3_5.value, temperature=0)

translation_template = "Translate the following text from English to {target_language}: {text}"
translation_prompt = PromptTemplate(input_variables=["target_language", "text"],
                                    template=translation_template)
translation_chain = LLMChain(llm=llm, prompt=translation_prompt)


def translate(text: str, target_language: str):
    translated_text = translation_chain.predict(
        target_language=target_language,
        text=text
    )
    print(translated_text)
