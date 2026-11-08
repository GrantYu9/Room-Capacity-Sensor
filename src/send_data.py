import asyncio

from json import *

from psycopg import connect

class DataSender:
    def __init__(self, url: str, password: str):
        self._SERVER = connect(url, password=password)

        format: str = '''
            !!!
        '''
        
        with self._SERVER.transaction():
            self._SERVER.execute(format)


    async def send_data(self, data: dict) -> None:
        with self._SERVER.transaction():
            await self._SERVER.commit(data)
