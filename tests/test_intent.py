"""Unit tests for IntentParser."""
import pytest
from robot.voice.intent import IntentParser, IntentType


@pytest.fixture
def parser():
    return IntentParser()


class TestIntentParser:
    def test_follow_exact(self, parser):
        assert parser.parse("follow me").type == IntentType.FOLLOW

    def test_follow_in_sentence(self, parser):
        assert parser.parse("please follow me now").type == IntentType.FOLLOW

    def test_stop(self, parser):
        assert parser.parse("stop").type == IntentType.STOP

    def test_halt(self, parser):
        assert parser.parse("halt right now").type == IntentType.STOP

    def test_search(self, parser):
        assert parser.parse("search for me").type == IntentType.SEARCH

    def test_forward(self, parser):
        assert parser.parse("go forward").type == IntentType.FORWARD

    def test_backward(self, parser):
        assert parser.parse("go back").type == IntentType.BACKWARD

    def test_turn_left(self, parser):
        assert parser.parse("turn left please").type == IntentType.TURN_LEFT

    def test_turn_right(self, parser):
        assert parser.parse("turn right").type == IntentType.TURN_RIGHT

    def test_speed_up(self, parser):
        assert parser.parse("faster").type == IntentType.SPEED_UP

    def test_speed_down(self, parser):
        assert parser.parse("slow down").type == IntentType.SPEED_DOWN

    def test_unknown(self, parser):
        assert parser.parse("what is the weather").type == IntentType.UNKNOWN

    def test_case_insensitive(self, parser):
        assert parser.parse("STOP").type == IntentType.STOP
        assert parser.parse("Follow Me").type == IntentType.FOLLOW

    def test_speed_param_extracted(self, parser):
        intent = parser.parse("speed 80")
        assert intent.type == IntentType.SPEED_UP
        assert intent.params.get("speed") == 80

    def test_intent_to_dict(self, parser):
        intent = parser.parse("follow me")
        d = intent.to_dict()
        assert d["type"] == "follow_person"
        assert "raw_text" in d
        assert "confidence" in d

    def test_idle(self, parser):
        assert parser.parse("idle").type == IntentType.IDLE

    def test_empty_string(self, parser):
        assert parser.parse("").type == IntentType.UNKNOWN
