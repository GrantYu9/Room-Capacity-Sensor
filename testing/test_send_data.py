from pathlib import Path

from unittest.mock import MagicMock

import pytest

from pytest_mock import MockerFixture

from src.process_data import EnumType
from src.send_data import Sender

class TestProcessor:
    _ROOT: Path = Path(__file__).parent.parent
    _DATA_TESTING: Path = _ROOT / "data" / "testing"
    _ID: int = 67
    _URL: str = "fake_url_123"
    _PASSWORD: str = "42_is_the_answer_to_everything"

    @pytest.fixture
    def mock_connect(self, mocker:MockerFixture) -> MagicMock:
        return mocker.patch("src.send_data.connect")

    @pytest.fixture
    def sender(self, mock_connect: MagicMock) -> Sender:
        return Sender(self._ID, self._URL, self._PASSWORD)

    def test_process_data_sanity(self, sender: Sender, mock_connect: MagicMock) -> None:
        data = {
            EnumType.LOCATION: "lab",
            EnumType.COUNT: 6,
            EnumType.AVERAGE_CONFIDENCE: pytest.approx(0.4955, abs=0.01),
            EnumType.DECIBELS: -24,
        }

        sender.send_data(data)

        connection = mock_connect.return_value
        parameters = connection.execute.call_args.args[1]

        assert parameters == (self._ID, "lab", 6, -24, 0.4955)
