from dataclasses import dataclass
import os
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    dataset: str
    family: str
    checkpoint_relpath: str
    input_size: int
    builder: str
    robustbench_name: str = ""


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class _CifarResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(block(self.in_planes, planes, block_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = torch.flatten(out, 1)
        return self.linear(out)


def build_cifar_resnet18(num_classes=10):
    return _CifarResNet(_BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def load_state_dict_file(model, checkpoint_path, device="cpu"):
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    return model.eval()


def normalize_cifar10_batch(x):
    mean = torch.tensor(CIFAR10_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def normalize_imagenet_batch(x):
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def list_model_specs():
    return (
        ModelSpec(
            name="cifar_resnet18",
            dataset="cifar10",
            family="standard_cifar",
            checkpoint_relpath="checkpoints/cifar/CIFAR10_ResNet18_epoch_20.pt",
            input_size=32,
            builder="cifar_resnet18",
        ),
        ModelSpec(
            name="robustbench_wong2020fast",
            dataset="cifar10",
            family="robustbench_cifar",
            checkpoint_relpath="checkpoints/robustbench/cifar10/Linf/Wong2020Fast.pt",
            input_size=32,
            builder="robustbench",
            robustbench_name="Wong2020Fast",
        ),
        ModelSpec(
            name="robustbench_engstrom2019",
            dataset="cifar10",
            family="robustbench_cifar",
            checkpoint_relpath="checkpoints/robustbench/cifar10/Linf/Engstrom2019Robustness.pt",
            input_size=32,
            builder="robustbench",
            robustbench_name="Engstrom2019Robustness",
        ),
        ModelSpec(
            name="robustbench_rice2020",
            dataset="cifar10",
            family="robustbench_cifar",
            checkpoint_relpath="checkpoints/robustbench/cifar10/Linf/Rice2020Overfitting.pt",
            input_size=32,
            builder="robustbench",
            robustbench_name="Rice2020Overfitting",
        ),
        ModelSpec(
            name="imagenet_vit_b_16",
            dataset="imagenet",
            family="torchvision_imagenet",
            checkpoint_relpath="checkpoints/torch_home/hub/checkpoints/vit_b_16-c867db91.pth",
            input_size=224,
            builder="torchvision_vit_b_16",
        ),
        ModelSpec(
            name="imagenet_swin_t",
            dataset="imagenet",
            family="torchvision_imagenet",
            checkpoint_relpath="checkpoints/torch_home/hub/checkpoints/swin_t-704ceda3.pth",
            input_size=224,
            builder="torchvision_swin_t",
        ),
        ModelSpec(
            name="imagenet_deit_tiny",
            dataset="imagenet",
            family="timm_imagenet",
            checkpoint_relpath="checkpoints/imagenet/deit_tiny_patch16_224.fb_in1k/model.safetensors",
            input_size=224,
            builder="timm_deit_tiny",
        ),
    )


def get_model_spec(name):
    for spec in list_model_specs():
        if spec.name == name:
            return spec
    known = ", ".join(spec.name for spec in list_model_specs())
    raise ValueError(f"Unknown model {name!r}; expected one of: {known}")


def build_model_from_spec(spec, data_root, device="cpu"):
    checkpoint = Path(data_root) / spec.checkpoint_relpath
    if spec.builder == "cifar_resnet18":
        model = build_cifar_resnet18()
        return load_state_dict_file(model, checkpoint, device=device), None
    if spec.builder == "robustbench":
        return _build_robustbench_model(spec, data_root, device), None
    if spec.builder == "torchvision_vit_b_16":
        import torchvision.models as tv_models

        model = tv_models.vit_b_16(weights=None)
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
        return model.to(device).eval(), normalize_imagenet_batch
    if spec.builder == "torchvision_swin_t":
        import torchvision.models as tv_models

        model = tv_models.swin_t(weights=None)
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
        return model.to(device).eval(), normalize_imagenet_batch
    if spec.builder == "timm_deit_tiny":
        import timm
        from safetensors.torch import load_file

        model = timm.create_model("deit_tiny_patch16_224.fb_in1k", pretrained=False)
        model.load_state_dict(load_file(str(checkpoint)))
        return model.to(device).eval(), normalize_imagenet_batch
    raise ValueError(f"Unsupported model builder {spec.builder!r}")


def _build_robustbench_model(spec, data_root, device):
    for env_name in ("AUTO_ATTACK_ROOT", "ROBUSTBENCH_ROOT"):
        env_path = os.environ.get(env_name)
        if env_path:
            _prepend_existing_sys_path(Path(env_path))
    from robustbench.utils import load_model

    model = load_model(
        spec.robustbench_name,
        model_dir=Path(data_root) / "checkpoints" / "robustbench",
        dataset="cifar10",
        threat_model="Linf",
    )
    return model.to(device).eval()


def _prepend_existing_sys_path(path):
    if path.is_dir():
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
