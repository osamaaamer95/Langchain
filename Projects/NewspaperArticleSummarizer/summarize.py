from langchain.schema import (
    HumanMessage
)
from langchain.chat_models import ChatOpenAI


def get_messages(article):
    # prepare template for prompt
    template = """You are an assistant that summarizes online articles.
    
    Here's the article you will summarize:
    
    ==================
    Title: {article_title}
    
    {article_text}
    ==================
    
    Write a summary for this article.
    """

    prompt = template.format(article_title=article['title'], article_text=article['text'])

    return [HumanMessage(content=prompt)]


def summarize(article):
    # load the model
    chat = ChatOpenAI(model_name="gpt-4", temperature=0)

    # get messages
    messages = get_messages(article)

    # generate
    summary = chat(messages)

    return summary.content
