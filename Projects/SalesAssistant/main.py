"""
Sales Copilot: transcribe sales audio in real-time and connects to a chatbot with
knowledge of the transcript so the salesperson can get help from AI during calls

Alternate approaches:
1. Pure LLM with no knowledge base
2. Split the knowledge base naively

Chosen approach: Intelligent splitting

Note: Splitting the text intelligently is super important!
"""
from Projects.SalesAssistant.DeepLakeLoader import DeepLakeLoader
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

objection = "It is too expensive"

# load some guidelines
db = DeepLakeLoader('sales.txt')

# fetch relevant guidelines from db
results = db.query_db(objection)

chat = ChatOpenAI()

system_message = SystemMessage(content=objection)

# ask chat to devise an answer based on the objection
human_message = HumanMessage(content=f'Customer objection: {objection} | Relevant guidelines: {results}')

response = chat([system_message, human_message])

print(response.content)
