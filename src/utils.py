import torch
import transformers
from typing import List

from typing import List, Set

def model2hfname(model: str) -> str:
    return {
        "bert-tiny": "prajjwal1/bert-tiny",
        "bert-med": "prajjwal1/bert-medium",
        "small": "gpt2",
        "med": "gpt2-medium",
        "large": "gpt2-large",
        "full": "gpt2-xl",
        "gpt2-sm": "gpt2",
        "gpt2-med": "gpt2-medium",
        "gpt2-lg": "gpt2-large",
        "gpt2": "gpt2-xl",
        "neo": "EleutherAI/gpt-neo-2.7B",
    }[model]

def get_model_and_tokenizer(model: str, Cls = transformers.AutoModelForCausalLM, **model_kwargs):
    hf_model_name = model2hfname(model)

    m = Cls.from_pretrained(hf_model_name, **model_kwargs)
    if isinstance(m, transformers.GPT2LMHeadModel):
        m.transformer.gradient_checkpointing_enable()

    tok = transformers.AutoTokenizer.from_pretrained(hf_model_name)

    if tok.pad_token_id is None:
        if Cls == transformers.AutoModelForCausalLM:
            tok.pad_token = tok.eos_token
        else:
            print("Adding pad token to tokenizer")
            tok.add_special_tokens({"pad_token": "[PAD]"})
            tok.pad_token = "[PAD]"
    return m, tok


def stop_tokens(tokenizer, stop_strings: Set[str] = set([])) -> List[int]:
    tokens = []
    for idx in range(len(tokenizer)):
        if tokenizer.decode(idx) in stop_strings:
            tokens.append(idx)
    print("Stop tokens:", tokens)
    return tokens

def ignore_tokens(tokenizer, stop_strings: Set[str] = set("\n")) -> List[int]:
    tokens = []
    for idx in range(len(tokenizer)):
        if tokenizer.decode(idx) in stop_strings:
            tokens.append(idx)
    print("Ignore tokens:", tokens)
    return tokens

def ignore_tokens_replace(tokenizer, stop_strings: Set[str] = set(" ")) -> List[int]:
    tokens = []
    for idx in range(len(tokenizer)):
        if tokenizer.decode(idx) in stop_strings:
            tokens.append(idx)
    print("Ignore tokens replaced by:", tokens)
    return tokens[0]

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)
