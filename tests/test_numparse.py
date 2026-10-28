"""Tests of the safe str2num replacement."""
import os
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE.parent / "src"))

import pytest

from pycba.gui.numparse import str2num


@pytest.mark.parametrize("text, expected", [
    ("12", [12.0]), ("  -3.5e2 ", [-350.0]), ("+4", [4.0]), ("2*3", [6.0]), ("1/4", [0.25]), ("2^3", [8.0]),
    ("(1+2)*3", [9.0]), ("1 2 3", [1.0, 2.0, 3.0]), ("[1,2]", [1.0, 2.0]), ("1;2", [1.0, 2.0]),
    ("pi", [3.141592653589793]), (".5", [0.5]), ("1e3", [1000.0]),
])
def test_numbers_and_expressions(text, expected):
    assert str2num(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "   ", "abc", "1,,x", "nan", "inf", "-inf", "1/0", "9**9**9**9", "1+"])
def test_invalid_or_non_finite_text_gives_nothing(text):
    assert str2num(text) == []


@pytest.mark.parametrize("text", [
    "__import__('os').system('echo pwned')", "open('x','w')", "(lambda: 1)()", "[x for x in range(3)]",
    "().__class__", "1 if 1 else 2", "True", "'a'*3",
])
def test_nothing_is_ever_executed(text):
    assert str2num(text) == []


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v", "-p", "no:cacheprovider"]))
