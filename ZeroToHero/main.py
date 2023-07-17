from ZeroToHero.ConversationChain import generate_conversation
from ZeroToHero.DeepLake import DeepLakeAgent
from ZeroToHero.GoogleSearchLangChain import GoogleSearchAgent
from ZeroToHero.LLMChain import generate_company_names
from enums import Model
from ZeroToHero.SimpleAPICall import generate_openai_response

######
# Call a GPT model with a temperature and a prompt
######

print(generate_openai_response(Model.GPT3, 0.7, "Hello world!"))

######
# Simple prompt template to generate a company name based on a prompt
######

print(generate_company_names(Model.GPT2, 0.4, "blueberry icecream"))

######
# Simple conversation with buffer memory
######

generate_conversation(Model.GPT3, 0.7)

######
# Add documents to deep lake and ask questions using a LLM model
######

texts = [
    "Napoleon Bonaparte was born in 15 August 1769",
    "Louis XIV was born in 5 September 1638",
    "Lady Gaga was born in 28 March 1986",
    "Michael Jeffrey Jordan was born in 17 February 1963"
]

deeplake_agent = DeepLakeAgent(Model.GPT3, 0)

deeplake_agent.add_documents_to_deeplake(texts)

deeplake_agent.ask_question_from_deeplake("When was Napoleon born?")
deeplake_agent.ask_question_from_deeplake("When was Lady Gaga born?")
deeplake_agent.ask_question_from_deeplake("When was Michael Jordan born?")

######
# Ask questions from Google API
######

question = "What's the latest news about the Mars rover? Then please summarize the results for me."
google_agent = GoogleSearchAgent(Model.GPT3, 0)
google_agent.ask_question(question)
