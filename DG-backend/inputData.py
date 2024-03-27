from datetime import datetime

from pydantic import BaseModel


class inputData(BaseModel):
    point_timestamp: str
    point_value: float