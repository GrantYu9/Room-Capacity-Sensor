import asyncio

from json import *

from psycopg import *

class DataSender:
    def __init__(self, url: str, password: str):
        self._SERVER = connect(url, password=password)

    async def send_data(data: dict) -> None:
        pass
