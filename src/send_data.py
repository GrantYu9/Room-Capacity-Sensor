import asyncio

from json import *
from socket import socket

class DataSender:
    def __init__(self, host: str, port: int):
        self._SERVER = socket(host, port)

    async def send_data(data: dict) -> None:
        pass
