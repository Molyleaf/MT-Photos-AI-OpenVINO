# app/clip/__init__.py
from .bert_tokenizer import FullTokenizer

_tokenizer = FullTokenizer()

# 移除了不存在的 convert_state_dict 导入
from .utils import load_from_name, available_models, tokenize, image_transform, load
