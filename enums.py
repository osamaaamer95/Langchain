from enum import Enum


class OpenAIModel(Enum):
    DAVINCI_3 = "text-davinci-003"
    DAVINCI_2 = "text-davinci-002"
    # add more models as needed


class ChatModel(Enum):
    GPT4 = "gpt-4"
    GPT3 = "gpt-3"
    GPT3_5 = "gpt-3.5-turbo"
    # add more models as needed
