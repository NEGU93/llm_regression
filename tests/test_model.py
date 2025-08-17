from llm_regressor.model import *


def test_extract_price():
    assert extract_price("Price is $123.45") == 123.45
    assert extract_price("Price is $1,234.56") == 1234.56
    assert extract_price("Price is $0.99") == 0.99
    assert extract_price("No price here") == 0
    assert extract_price("Price is $-100.00") == -100.00
    assert extract_price("Price is $1000") == 1000.0
    assert extract_price("Price is $999 blah blah so cheap") == 999.0
