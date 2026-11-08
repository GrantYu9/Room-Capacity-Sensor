from psycopg import connect, Connection

from process_data import EnumType

class Sender:
    def __init__(self, id: int, url: str, password: str):
        self._SERVER: Connection = connect(url, password=password)

        format: str = '''
            CREATE TABLE IF NOT EXISTS room_data (
                id BIGSERIAL PRIMARY KEY,
                location TEXT,
                count INTEGER NOT NULL,
                decibels DOUBLE PRECISION
            )
        '''
        
        with self._SERVER.transaction():
            self._SERVER.execute(format)

        self._id: int = id

    def send_data(self, data: dict) -> None:
        query: str = '''
            INSERT INTO room_data (id, location, count, decibels, average_confidence)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                location = EXCLUDED.location,
                count = EXCLUDED.count,
                decibels = EXCLUDED.decibels
                average_confidence = EXCLUDED.average_confidence
        '''

        modified_data: tuple = (self._id, data[EnumType.LOCATION], data[EnumType.COUNT], data[EnumType.DECIBELS], data[EnumType.AVERAGE_CONFIDENCE])

        with self._SERVER.transaction():
            self._SERVER.execute(query, modified_data)
