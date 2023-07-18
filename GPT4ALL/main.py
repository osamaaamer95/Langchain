# TODO pip install -q langchain==0.0.152

import os

from langchain.llms import GPT4All
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# from download_weights import download_weights


# First download the model
# download_weights("./models/gpt4all-lora-quantized-ggml.bin")

# Then, transform to 4B
# git clone https://github.com/ggerganov/llama.cpp.git
# cd llama.cpp && git checkout 2b26469
# python3 llama.cpp/convert.py ./models/gpt4all-lora-quantized-ggml.bin

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

dir_path = os.path.dirname(os.path.realpath(__file__))
path_to_model = (dir_path + "/models/ggml-model-q4_0.bin")

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = GPT4All(model=path_to_model, callback_manager=callback_manager, verbose=True)
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What happens when it rains somewhere?"
llm_chain.run(question)
