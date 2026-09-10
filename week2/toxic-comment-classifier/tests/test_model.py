import torch

from toxic_classifier.models.lstm import ToxicLSTM


def test_model_shape():
    model = ToxicLSTM(100, 16, 8, 1, 0.2, True, 6)
    x = torch.randint(0, 100, (4, 20))
    y = model(x)
    assert y.shape == (4, 6)
