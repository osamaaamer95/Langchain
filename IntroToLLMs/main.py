from IntroToLLMs.few_shot_learning import few_shot_learning
from IntroToLLMs.ask_huggingface import ask_huggingface, ask_huggingface_using_list, ask_huggingface_using_template
from IntroToLLMs.summarization import summarize
from IntroToLLMs.text_translation import translate
from IntroToLLMs.tracking_token_usage import tell_joke_and_track_usage

from enums import ChatModel

tell_joke_and_track_usage()

few_shot_learning(ChatModel.GPT4, "What's the meaning of life?")

# Single question
ask_huggingface("What is the capital city of France?")

# Multiple questions
qa = [
    {'question': "What is the capital city of France?"},
    {'question': "What is the largest mammal on Earth?"},
    {'question': "Which gas is most abundant in Earth's atmosphere?"},
    {'question': "What color is a ripe banana?"}
]

ask_huggingface_using_list(qa)

qs_str = (
        "What is the capital city of France?\n" +
        "What is the largest mammal on Earth?\n" +
        "Which gas is most abundant in Earth's atmosphere?\n" +
        "What color is a ripe banana?\n"
)

ask_huggingface_using_template(qs_str)

summarize(
    "LangChain provides many modules that can be used to build language model applications. Modules can be combined "
    "to create more complex applications, or be used individually for simple applications. The most basic building "
    "block of LangChain is calling an LLM on some input. Let’s walk through a simple example of how to do this. For "
    "this purpose, let’s pretend we are building a service that generates a company name based on what the company "
    "makes.")

translate("Greece is a cool place to be!", target_language="Greek")
