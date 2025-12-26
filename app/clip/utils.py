# app/clip/utils.py
import json
import os
import urllib
import urllib.request
from pathlib import Path
from typing import Union, List

import torch
from safetensors.torch import load_file
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode
from tqdm import tqdm

from . import _tokenizer
from .model import convert_weights, CLIP, restore_model

__all__ = ["load", "tokenize", "available_models", "image_transform", "load_from_name", "MODEL_INFO"]

_MODELS = {
    "ViT-B-16": "https://huggingface.co/TencentARC/QA-CLIP/resolve/main/QA-CLIP-base.pt",
    "ViT-L-14": "https://huggingface.co/TencentARC/QA-CLIP/resolve/main/QA-CLIP-large.pt",
    "RN50": "https://huggingface.co/TencentARC/QA-CLIP/resolve/main/QA-CLIP-RN50.pt",
}

MODEL_INFO = {
    "ViT-B-16": {
        "struct": "ViT-B-16@RoBERTa-wwm-ext-base-chinese",
        "input_resolution": 224
    },
    "ViT-L-14": {
        "struct": "ViT-L-14@RoBERTa-wwm-ext-base-chinese",
        "input_resolution": 224
    },
    "RN50": {
        "struct": "RN50@RBT3-chinese",
        "input_resolution": 224
    },
}

def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True,
                  unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))
    return download_target

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def available_models() -> List[str]:
    return list(_MODELS.keys())

def load_from_name(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
                   download_root: str = None, vision_model_name: str = None, text_model_name: str = None, input_resolution: int = None):

    model_path = None

    if name in _MODELS:
        # 优先查找 safetensors
        root_dir = download_root or os.path.expanduser("~/.cache/clip")
        pt_filename = os.path.basename(_MODELS[name])
        safe_filename = pt_filename.replace(".pt", ".safetensors")
        safe_path = os.path.join(root_dir, safe_filename)

        if os.path.exists(safe_path):
            model_path = safe_path
        else:
            # 如果没有 safetensors，才下载 .pt (但建议使用 download-models.py 预处理)
            model_path = _download(_MODELS[name], root_dir)

        model_name, model_input_resolution = MODEL_INFO[name]['struct'], MODEL_INFO[name]['input_resolution']
    elif os.path.isfile(name):
        model_path = name
        if not (vision_model_name and text_model_name and input_resolution):
            # 尝试从 name 中推断，或者保持原有的 assert
            assert vision_model_name and text_model_name and input_resolution, \
                "Please specify specific 'vision_model_name', 'text_model_name', and 'input_resolution'"
        model_name, model_input_resolution = f'{vision_model_name}@{text_model_name}', input_resolution
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    # Loading Logic: Safetensors vs PyTorch Pickle
    if model_path.endswith(".safetensors"):
        print(f"Loading weights from {model_path} (Safetensors)...")
        checkpoint = load_file(model_path)
    else:
        print(f"Loading weights from {model_path} (PyTorch default)...")
        with open(model_path, 'rb') as opened_file:
            checkpoint = torch.load(opened_file, map_location="cpu")

    model = create_model(model_name, checkpoint)
    if str(device) == "cpu":
        model.float()
    else:
        model.to(device)
    return model, image_transform(model_input_resolution)

def load(model, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", clip_path=None,
         bert_path=None):
    """Load CLIP and BERT model weights"""

    bert_state_dict = torch.load(bert_path, map_location="cpu") if bert_path else None

    clip_state_dict = None
    if clip_path:
        if clip_path.endswith(".safetensors"):
            clip_state_dict = load_file(clip_path)
        else:
            clip_state_dict = torch.load(clip_path, map_location="cpu")

    restore_model(model, clip_state_dict, bert_state_dict).to(device)

    if str(device) == "cpu":
        model.float()
    return model

def tokenize(texts: Union[str, List[str]], context_length: int = 52) -> torch.LongTensor:
    if isinstance(texts, str):
        texts = [texts]

    all_tokens = []
    for text in texts:
        all_tokens.append([_tokenizer.vocab['[CLS]']] + _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(text))[
            :context_length - 2] + [_tokenizer.vocab['[SEP]']])

    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        assert len(tokens) <= context_length
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

def _convert_to_rgb(image):
    return image.convert('RGB')

def image_transform(image_size=224):
    transform = Compose([
        Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        _convert_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    return transform

def create_model(model_name, checkpoint=None):
    vision_model, text_model = model_name.split('@')

    vision_model_config_file = Path(__file__).parent / f"model_configs/{vision_model.replace('/', '-')}.json"
    assert os.path.exists(vision_model_config_file)

    text_model_config_file = Path(__file__).parent / f"model_configs/{text_model.replace('/', '-')}.json"
    assert os.path.exists(text_model_config_file)

    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        for k, v in json.load(ft).items():
            model_info[k] = v
    if isinstance(model_info['vision_layers'], str):
        model_info['vision_layers'] = eval(model_info['vision_layers'])

    # Remove use_flash_attention if it exists in config json (just in case)
    if 'use_flash_attention' in model_info:
        del model_info['use_flash_attention']

    print('Model info', model_info)
    model = CLIP(**model_info)
    convert_weights(model)

    if checkpoint:
        # Handle both nested .pt dicts {"state_dict": ...} and flat .safetensors dicts
        sd = checkpoint
        if "state_dict" in sd:
            sd = sd["state_dict"]

        # Clean up DataParallel prefix if exists
        if next(iter(sd.items()))[0].startswith('module.'):
            sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}

        model.load_state_dict(sd, strict=False)

    return model