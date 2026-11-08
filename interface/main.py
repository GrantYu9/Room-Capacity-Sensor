from os import getenv
from time import sleep

from dotenv import load_dotenv

from src.process_data import Processor
from src.send_data import Sender

async def main() -> None:
    load_dotenv()

    data_sender: Sender = Sender(getenv("SERVER_URL"), getenv("SERVER_PASSWORD"))
    # image_processor: ImageProcessor = ImageProcessor()

    wait_time: int = 10

    while True:
        sleep(wait_time)
        break
        # await data_sender.send_data(image_processor.process_image()) # !!!

if __name__ == "__main__":
    main()
