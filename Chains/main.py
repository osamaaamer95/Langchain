from langchain import PromptTemplate, OpenAI, ConversationChain

from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()

llm = OpenAI(model_name="text-davinci-003", temperature=0)

template = """List all possible words as substitute for 'artificial' as comma separated.

Current conversation:
{history}

{input}"""

conversation = ConversationChain(
    llm=llm,
    prompt=PromptTemplate(template=template, input_variables=["history", "input"], output_parser=output_parser),
    memory=ConversationBufferMemory(),
    verbose=True)

conversation.predict(input="")
