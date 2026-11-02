"""Backbones b_f(.) and CLIP encoders E_I / E_T.

Paper, App. D.2: CIFAR -> CLIP(RN50) image encoder; CUB -> ResNet-18 trained on CUB (imgclsmob/pytorchcv).
All models are run in float32: IP noise (sigma = 1e-3) would be partially rounded away in fp16.
"""
import torch
from torchvision import transforms


def _imagenet_preprocess():
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_clip(name: str, device: str):
    import clip  # pip install git+https://github.com/openai/CLIP.git
    model, preprocess = clip.load(name, device=device, jit=False)
    model = model.float().eval()
    return model, preprocess


def get_backbone(name: str, device: str):
    """Returns (feature_fn, preprocess). feature_fn: image batch -> (B, d) backbone features."""
    if name.startswith("clip_"):
        model, preprocess = load_clip(name[len("clip_"):], device)

        @torch.no_grad()
        def feature_fn(x):
            return model.encode_image(x).float()
        return feature_fn, preprocess

    if name == "resnet18_cub":
        from pytorchcv.model_provider import get_model as ptcv_get_model
        model = ptcv_get_model("resnet18_cub", pretrained=True).to(device).float().eval()

        @torch.no_grad()
        def feature_fn(x):
            return torch.flatten(model.features(x), 1).float()  # output of features.final_pool
        return feature_fn, _imagenet_preprocess()

    raise ValueError(f"Unknown backbone {name}")
