from langchain import PromptTemplate, OpenAI
from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.schema import OutputParserException

from Parsers.suggestions_parser import Suggestions


def generate_suggestions():
    parser = PydanticOutputParser(pydantic_object=Suggestions)

    template = """
    Offer a list of suggestions to substitute the specified target_word based the presented context.
    {format_instructions}
    target_word={target_word}
    context={context}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["target_word", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    model_input = prompt.format(
        target_word="behaviour",
        context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to"
                "conduct the lesson."
    )

    model = OpenAI(model_name='text-davinci-003', temperature=0.5)

    output = model(model_input)

    try:
        parsed_output = parser.parse(output)
        print(parsed_output.json())
    except OutputParserException:
        # retry parser
        retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=model)
        retry_parser.parse_with_prompt(output, model_input)

        # output fixing parser
        outputfixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model)
        print(outputfixing_parser.parse(output).json())
