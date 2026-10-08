import torch
from torch import nn

from src.lora_scratch import LoRALinear, inject_lora


def test_initial_output_equals_base():
    base = nn.Linear(32, 16)
    x = torch.randn(4, 32)
    expected = base(x)
    assert torch.allclose(LoRALinear(base, r=4)(x), expected)


def test_merge_is_equivalent():
    layer = LoRALinear(nn.Linear(32, 16), r=4, alpha=8)
    nn.init.normal_(layer.lora_B)
    x = torch.randn(4, 32)
    before = layer(x)
    layer.merge()
    assert torch.allclose(layer(x), before, atol=1e-5)
    layer.unmerge()
    assert torch.allclose(layer(x), before, atol=1e-5)


def test_only_lora_params_trainable():
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    model.q = nn.Linear(8, 8)
    inject_lora(model, ["q"], r=2)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"q.lora_A", "q.lora_B"}
