import asyncio

from os import getenv
from process_image import ImageProcessor
from send_data import DataSender

from dotenv import load_dotenv

async def main() -> None:
    load_dotenv()

    data_sender: DataSender = DataSender(getenv("SERVER_URL"), getenv("SERVER_PASSWORD"))
    image_processor: ImageProcessor = ImageProcessor()

    wait_time: int = 10

    while True:
        await asyncio.sleep(wait_time)
        await data_sender.send_data(image_processor.process_image()) # !!!

if __name__ == "__main__":
    asyncio.run(main())
