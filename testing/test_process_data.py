from pathlib import Path

import pytest

from src.process_data import EnumType, Processor

class TestProcessor:
    _ROOT: Path = Path(__file__).parent.parent
    _DATA_TESTING: Path = _ROOT / "data" / "testing"


    @pytest.fixture
    def processor(self) -> Processor:
        return Processor()

    def test_process_data_sanity(self, processor: Processor) -> None:
        expected: dict = {
            EnumType.LOCATION: "",
            EnumType.COUNT: 6,
            EnumType.DECIBELS: -24
        }

        audio: Path = self._DATA_TESTING / "sample-9s.mp3"
        image: Path = self._DATA_TESTING / "room.jpg"
        location: Path = ""

        assert processor.process_data(audio, image, location) == expected
