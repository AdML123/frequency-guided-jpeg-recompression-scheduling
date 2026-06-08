from pathlib import Path

import pytest
import torch


def test_cifar_resnet18_forward_returns_class_logits():
    from jpeg_defense.models import build_cifar_resnet18

    torch.manual_seed(0)
    model = build_cifar_resnet18(num_classes=10)

    with torch.no_grad():
        logits = model(torch.randn(2, 3, 32, 32))

    assert logits.shape == (2, 10)


def test_cifar_resnet18_uses_checkpoint_compatible_key_names():
    from jpeg_defense.models import build_cifar_resnet18

    keys = set(build_cifar_resnet18(num_classes=10).state_dict())

    expected_keys = {
        "conv1.weight",
        "bn1.weight",
        "layer1.0.conv1.weight",
        "layer2.0.shortcut.0.weight",
        "layer4.1.bn2.running_var",
        "linear.weight",
    }
    assert expected_keys.issubset(keys)


def test_normalize_cifar10_batch_uses_standard_channel_statistics():
    from jpeg_defense.models import normalize_cifar10_batch

    x = torch.zeros(1, 3, 1, 1)

    normalized = normalize_cifar10_batch(x)

    expected = torch.tensor(
        [
            -0.4914 / 0.2023,
            -0.4822 / 0.1994,
            -0.4465 / 0.2010,
        ]
    ).view(1, 3, 1, 1)
    assert normalized == pytest.approx(expected)


def test_load_state_dict_file_loads_plain_state_dict_and_sets_eval(tmp_path):
    from jpeg_defense.models import load_state_dict_file

    model = torch.nn.Linear(2, 1)
    checkpoint_path = tmp_path / "plain.pt"
    state = {
        "weight": torch.tensor([[1.0, -1.0]]),
        "bias": torch.tensor([0.5]),
    }
    torch.save(state, checkpoint_path)
    model.train()

    loaded = load_state_dict_file(model, checkpoint_path, device="cpu")

    assert loaded is model
    assert not loaded.training
    assert loaded.weight.detach() == pytest.approx(state["weight"])
    assert loaded.bias.detach() == pytest.approx(state["bias"])


def test_load_state_dict_file_loads_wrapped_state_dict(tmp_path):
    from jpeg_defense.models import load_state_dict_file

    model = torch.nn.Linear(2, 1)
    checkpoint_path = tmp_path / "wrapped.pt"
    state = {
        "weight": torch.tensor([[2.0, 3.0]]),
        "bias": torch.tensor([-4.0]),
    }
    torch.save({"state_dict": state, "epoch": 3}, checkpoint_path)

    loaded = load_state_dict_file(model, Path(checkpoint_path), device=torch.device("cpu"))

    assert not loaded.training
    assert loaded.weight.detach() == pytest.approx(state["weight"])
    assert loaded.bias.detach() == pytest.approx(state["bias"])


def test_model_registry_lists_core_reviewer_defense_models():
    from jpeg_defense.models import list_model_specs

    specs = {spec.name: spec for spec in list_model_specs()}

    assert specs["cifar_resnet18"].dataset == "cifar10"
    assert specs["robustbench_wong2020fast"].family == "robustbench_cifar"
    assert specs["robustbench_engstrom2019"].checkpoint_relpath.endswith(
        "Engstrom2019Robustness.pt"
    )
    assert specs["robustbench_rice2020"].input_size == 32
    assert specs["imagenet_vit_b_16"].dataset == "imagenet"
    assert specs["imagenet_swin_t"].input_size == 224
    assert specs["imagenet_deit_tiny"].family == "timm_imagenet"


def test_get_model_spec_returns_named_spec_and_rejects_unknown_name():
    from jpeg_defense.models import get_model_spec

    assert get_model_spec("imagenet_vit_b_16").dataset == "imagenet"

    with pytest.raises(ValueError, match="Unknown model"):
        get_model_spec("missing_model")


def test_build_model_from_spec_uses_cifar_builder_and_checkpoint_loader(
    monkeypatch, tmp_path
):
    from jpeg_defense import models
    from jpeg_defense.models import ModelSpec, build_model_from_spec

    checkpoint = tmp_path / "checkpoints" / "mock.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")

    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, x):
            return torch.zeros(x.shape[0], 10, device=x.device)

    load_calls = []
    monkeypatch.setattr(models, "build_cifar_resnet18", Dummy)
    monkeypatch.setattr(
        models,
        "load_state_dict_file",
        lambda model, path, device: load_calls.append((path, device)) or model.eval(),
    )
    spec = ModelSpec(
        name="mock",
        dataset="cifar10",
        family="unit",
        checkpoint_relpath="checkpoints/mock.pt",
        input_size=32,
        builder="cifar_resnet18",
    )

    model, normalize_fn = build_model_from_spec(spec, tmp_path, device="cpu")

    assert not model.training
    assert normalize_fn is None
    assert load_calls == [(checkpoint, "cpu")]
