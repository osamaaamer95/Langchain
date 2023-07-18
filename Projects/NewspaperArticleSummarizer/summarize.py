from langchain.schema import (
    SystemMessage, HumanMessage
)
from langchain.chat_models import ChatOpenAI


def get_messages(article):
    # prepare template for prompt
    template = """    
    Here's the article you will summarize:
    
    ==================
    Title: {article_title}
    
    {article_text}
    ==================
    
    Write a summary for this article as a bulleted list.
    """

    prompt = template.format(article_title=article['title'], article_text=article['text'])

    return [
        SystemMessage(content="You are an assistant that summarizes online articles."),
        HumanMessage(content=prompt)
    ]


def summarize(article):
    # load the model
    chat = ChatOpenAI(model_name="gpt-4", temperature=0)

    # get messages
    messages = get_messages(article)

    # generate
    summary = chat(messages)

    return summary.content
