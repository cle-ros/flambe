import pytest
import mock

from flambe.tokenizer import PipelineTokenizer


def test_pipeline_call():
    transformations = []
    for i in range(10):
        m = mock.Mock()
        m.return_value = 'test'
        transformations.append(m)

    tokenizer = PipelineTokenizer(transformations)
    tokenizer.tokenize('test')

    for m in transformations:
        m.assert_called()


def test_pipeline_order():
    transformations = []
    for i in list(range(10)):
        transformations.append(lambda x, i=i: f'{x}{i}')

    tokenizer = PipelineTokenizer(transformations)
    ret = tokenizer.tokenize('')

    assert ret == '0123456789'


def test_pipeline_none():
    with pytest.raises(ValueError):
        _ = PipelineTokenizer(None)
