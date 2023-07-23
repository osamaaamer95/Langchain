from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser, OutputFixingParser
from langchain.schema import (
    SystemMessage, HumanMessage, OutputParserException
)
from langchain.chat_models import ChatOpenAI

from Projects.NewspaperArticleSummarizer.parser import ArticleSummary


def get_messages(article, parser):
    # prepare template for prompt
    template = """    
    Here's the article you will summarize:
    
    ==================
    Title: {article_title}
    
    {article_text}
    ==================
    
    Write a summary for this article as a bulleted list. 
    
    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["article_title", "article_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    formatted_prompt = prompt.format_prompt(article_title=article['title'], article_text=article['text'])

    return [
        SystemMessage(
            content="""
            You are an assistant that summarizes online articles into bulleted points.
            Here are a few examples of summarized articles:
            Example 1:
            Original Article: 'The Effects of Climate Change
            Summary:
            - Climate change is causing a rise in global temperatures.
            - This leads to melting ice caps and rising sea levels.
            - Resulting in more frequent and severe weather conditions.
            
            Example 2:
            Original Article: 'The Evolution of Artificial Intelligence
            Summary:
            - Artificial Intelligence (AI) has developed significantly over the past decade.
            - AI is now used in multiple fields such as healthcare, finance, and transportation.
            - The future of AI is promising but requires careful regulation.                        
            """
        ),
        HumanMessage(content=formatted_prompt.to_string())
    ]


def summarize(article):
    # load the model
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # set up output parser
    parser = PydanticOutputParser(pydantic_object=ArticleSummary)

    # get messages
    messages = get_messages(article, parser)

    # generate
    summary = chat(messages)

    try:

        parsed_output = parser.parse(summary.content).json()
        return parsed_output
    except OutputParserException as e:
        print("Output has less than three bullet points. Fixing output...")
        # output fixing parser
        outputfixing_parser = OutputFixingParser.from_llm(parser=parser, llm=chat)
        print(outputfixing_parser.parse(summary.content).json())
