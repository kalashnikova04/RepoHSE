import pytest
from main import Item, predict_item


def test_predict_item():
    item = Item(text='it was cool')
    prediction = predict_item(item)
    # value = await prediction
    # print(value)
    assert isinstance(prediction, float)