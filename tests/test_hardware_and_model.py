import pytest

torch = pytest.importorskip("torch")

from cortex.hardware.profiler import can_train_locally, profile_hardware
from cortex.model.configs import CORTEX_2B, MODEL_STATUS, get_config
from cortex.model.transformer import CortexTransformer


def test_hardware_profile_is_real():
    p = profile_hardware()
    assert p.cpu_cores >= 1
    assert p.ram_total_gb > 0
    assert p.disk_free_gb > 0
    assert p.summary()


def test_2b_config_is_declared_not_trained():
    cfg = CORTEX_2B
    assert cfg.theoretical_param_count() >= 2_000_000_000
    assert MODEL_STATUS["cortex-2b"]["status"] == "declared"
    assert MODEL_STATUS["cortex-2b"]["checkpoint"] is None


def test_2b_not_trainable_on_this_host():
    p = profile_hardware()
    ok, reason = can_train_locally(p, CORTEX_2B.theoretical_param_count())
    if not p.has_cuda and p.ram_available_gb < 32:
        assert not ok
        assert reason


def test_param_count_math_matches_live_model():
    cfg = get_config("cortex-demo")
    model = CortexTransformer(cfg)
    assert model.actual_param_count() == cfg.theoretical_param_count()


def test_forward_and_loss_shapes():
    cfg = get_config("cortex-demo")
    model = CortexTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(loss)


def test_generation_runs():
    cfg = get_config("cortex-demo")
    model = CortexTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(x, max_new_tokens=4, temperature=1.0)
    assert out.shape[1] == 8
