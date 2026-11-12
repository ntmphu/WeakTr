"""CPU regression checks for AMP smoke acceptance; CUDA execution is separate."""
import pytest
import torch

from reproduction.smoke import check_amp_update


class CalibratingScaler:
    """Simulate overflow/backoff and skipped steps without requiring a GPU."""
    def __init__(self, skipped):
        self.skipped = skipped
        self.scale = 65536.0
        self.updates = 0

    def state_dict(self):
        return {'scale': self.scale}

    def __call__(self, loss, optimizer, parameters):
        parameters = list(parameters)
        assert all(p.grad is None for p in parameters)
        loss.backward()
        if self.skipped:
            self.skipped -= 1
            for p in parameters:
                p.grad.fill_(float('inf'))
            self.scale /= 2
        else:
            optimizer.step()
            self.updates += 1


@pytest.mark.parametrize('skipped', [0, 2])
def test_amp_requires_successful_update_after_calibration(skipped):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scaler = CalibratingScaler(skipped)
    before = model.weight.detach().clone()
    loss, attempts, scale = check_amp_update(
        model, optimizer, scaler, lambda: model(torch.ones(1, 2)).sum())
    assert attempts == skipped + 1
    assert scale == 65536 / (2 ** skipped)
    assert scaler.updates == 1
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before, model.weight)


def test_amp_rejects_persistent_overflow_without_updating_weights():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scaler = CalibratingScaler(10)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    with pytest.raises(ValueError, match='after 3 attempts'):
        check_amp_update(model, optimizer, scaler,
                         lambda: model(torch.ones(1, 2)).sum(), max_attempts=3)
    assert scaler.updates == 0
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)


def test_amp_rejects_nonfinite_forward_loss():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scaler = CalibratingScaler(0)
    with pytest.raises(ValueError, match='Nonfinite Phase-1 smoke loss'):
        check_amp_update(model, optimizer, scaler,
                         lambda: model(torch.ones(1, 2)).sum() * float('nan'))
    assert scaler.updates == 0
