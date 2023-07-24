from langchain.chains import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

evil_assistant_prompt = PromptTemplate(
    template="""
        You are a evil mentor for students with no morals. Give suggestions that are easiest and fastest to achieve the goal.
        Goal: {inquiry}
        Easiest way:""",
    input_variables=["inquiry"],
)

llm = OpenAI(model_name="text-davinci-003", temperature=0)
evil_assistant_chain = LLMChain(llm=llm, prompt=evil_assistant_prompt)

ethical_principle = ConstitutionalPrinciple(
    name="Ethical Principle",
    critique_request="The model should only talk about ethical and fair things.",
    revision_request="Rewrite the model's output to be both ethical and fair.",
)

fun_principle = ConstitutionalPrinciple(
    name="Be Funny",
    critique_request="The model responses must be funny and understandable for a 7th grader.",
    revision_request="Rewrite the model's output to be both funny and understandable for 7th graders.",
)

constitutional_chain = ConstitutionalChain.from_llm(
    chain=evil_assistant_chain,
    constitutional_principles=[ethical_principle, fun_principle],
    llm=llm,
    verbose=True,
)

result = constitutional_chain.run(inquiry="Getting full mark on my exams.")

print(result)
