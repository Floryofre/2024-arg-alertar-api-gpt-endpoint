from pydantic import BaseModel, Field

class Model(BaseModel):
    text: str = Field(min_lenght=10,max_length=100)