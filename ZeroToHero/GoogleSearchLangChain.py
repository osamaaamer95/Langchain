from langchain import LLMChain, PromptTemplate
from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.utilities import GoogleSearchAPIWrapper

from enums import Model


class GoogleSearchAgent:
    def __init__(self, model: Model, temperature: float):
        self.llm = OpenAI(model=model.value, temperature=temperature)
        self.search = GoogleSearchAPIWrapper()
        self.summarize_chain = LLMChain(llm=self.llm, prompt=PromptTemplate(
            input_variables=["query"],
            template="Write a summary of the following text: {query}"
        ))

    def ask_question(self, question: str):
        tools = [
            Tool(
                name="google-search",
                func=self.search.run,
                description="useful for when you need to search google to answer questions about current events"
            ),
            Tool(
                name='Summarizer',
                func=self.summarize_chain.run,
                description='useful for summarizing texts'
            )
        ]

        agent = initialize_agent(
            tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

        return agent(question)
