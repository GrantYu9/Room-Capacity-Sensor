import asyncio

from process_image import ImageProcessor
from send_data import DataSender

async def main() -> None:
    data_sender: DataSender = DataSender()
    image_processor: ImageProcessor = ImageProcessor()

    wait_time: int = 10

    while True:
        await asyncio.sleep(wait_time)
        await data_sender.send_data(image_processor.process_image()) # !!!

if __name__ == "__main__":
    asyncio.run(main())
