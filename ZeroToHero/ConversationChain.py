from langchain import ConversationChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

from enums import OpenAIModel


def generate_conversation(model: OpenAIModel, temperature: float):
    # get reference to the llm we want to use
    llm = OpenAI(model=model.value, temperature=temperature)

    conversation = ConversationChain(
        llm=llm,
        verbose=True,
        memory=ConversationBufferMemory()
    )

    # Start the conversation
    conversation.predict(input="Tell me about yourself.")

    # Continue the conversation
    conversation.predict(input="What can you do?")
    conversation.predict(input="How can you help me with data analysis?")

    # Display the conversation
    print(conversation)
