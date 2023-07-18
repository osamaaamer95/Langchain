# Chat Models offer a more structured API and are better suited for conversational tasks. Also, they can remember
# previous exchanges with the user, making them more suitable for engaging in meaningful conversations. Additionally,
# they benefit from reinforcement learning from human feedback, which helps improve their responses. They still have
# some limitations in reasoning and may require careful handling to avoid hallucinations and generating inappropriate
# content.

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

chat = ChatOpenAI(
    model_name="gpt-4", temperature=0.5
)

# previous history
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="The capital of France is Paris.")
]

# new incoming chat
prompt = HumanMessage(
    content="I'd like to know more about the city you just mentioned."
)

# add to messages
messages.append(prompt)

# generate a response
response = chat(messages)
