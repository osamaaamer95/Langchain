from typing import List

from pydantic import BaseModel, Field, validator


class Suggestion(BaseModel):
    word: str = Field(description="The substitute word")
    reason: str = Field(description="Reasoning behind why this word fits the context")

    @validator('word')
    def not_start_with_number(cls, field):
        if field[0].isnumeric():
            raise ValueError("The word can not start with numbers!")
        return field

    @validator('reason')
    def end_with_dot(cls, field):
        if field[-1] != ".":
            field += "."
        return field


# Define your desired data structure.
class Suggestions(BaseModel):
    words: List[Suggestion] = Field(description="List of substitute words based on context")

    # Throw error in case of receiving a numbered-list from API
    @validator('words')
    def cannot_be_empty(cls, field):
        if len(field) < 1:
            raise ValueError("The words cannot be empty")
        return field
