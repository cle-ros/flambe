import torch

from flambe.nn import AvgPooling, SumPooling
from torch import allclose


def test_avg_pooling():
    layer = AvgPooling()
    average = layer.forward(build_tensor())
    assert allclose(average[0], torch.tensor([5 / 4, 7 / 4, 9 / 4], dtype=torch.float32))


def test_sum_pooling():
    layer = SumPooling()
    sum = layer.forward(build_tensor())
    assert allclose(sum[0], torch.tensor([5, 7, 9], dtype=torch.float32))


def build_tensor():
    batch_size = 32
    items = 4
    embedding_size = 3
    data = torch.zeros([batch_size, items, embedding_size], dtype=torch.float32)
    data[0][0] = torch.tensor([1, 2, 3])
    data[0][1] = torch.tensor([4, 5, 6])
    return data
