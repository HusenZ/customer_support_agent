from pydantic import BaseModel, Field
from typing import Literal, TypedDict

class ClassifierSchema(BaseModel):
    user_input:str=Field(description="Users query")
    query_type:Literal["simple", "critical"] = Field(..., description="users query type whether it is simple or critical")

class CustomerSupportState(TypedDict):
    user_query: str
    query_type:str
    bot_reply:str