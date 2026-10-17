"""Shared Qwen helpers: loading, layer-k features, and running only the first k layers.

Feature definition (used everywhere): the feature at layer k is the model's final
RMSNorm applied to the residual stream after layer k, taken at the last prompt token.
This is exactly what a model cut down to k layers outputs, so features cached from one
full forward pass match what the fast path computes live.
"""

from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_LAYERS = [4, 8, 12, 16, 20, 24, 28]


def load_qwen(name="Qwen/Qwen3-0.6B", device="cuda"):
    """Load a frozen Qwen model in bfloat16. Returns (tokenizer, model)."""
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"  # so the last prompt token is always at position -1 in a batch
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device).eval()
    model.requires_grad_(False)
    return tok, model


def num_layers(model):
    return len(model.model.layers)


@torch.no_grad()
def layer_features(model, tok, prompts, layers=DEFAULT_LAYERS):
    """One full forward pass; returns {k: tensor [batch, hidden]} of last-token features."""
    batch = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    hs = model.model(**batch, output_hidden_states=True).hidden_states
    # hs[0] = embeddings, hs[k] = raw output of layer k, except hs[-1] already has the final norm.
    n = num_layers(model)
    return {k: (hs[k] if k == n else model.model.norm(hs[k]))[:, -1, :].float() for k in layers}


def layer_features_bucketed(model, tok, prompts, layers=DEFAULT_LAYERS, batch_size=128):
    """layer_features for prompts of varying length: batches hold equal-length prompts (no padding).

    Returns {k: tensor [len(prompts), hidden]} in the original prompt order.
    """
    lengths = [len(ids) for ids in tok(prompts).input_ids]
    order = sorted(range(len(prompts)), key=lambda i: lengths[i])
    out = {k: [None] * len(prompts) for k in layers}
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and j - i < batch_size and lengths[order[j]] == lengths[order[i]]:
            j += 1
        idx = order[i:j]
        feats = layer_features(model, tok, [prompts[n] for n in idx], layers)
        for k in layers:
            for row, n in enumerate(idx):
                out[k][n] = feats[k][row]
        i = j
    return {k: torch.stack(v) for k, v in out.items()}


@contextmanager
def first_k_layers(model, k):
    """Temporarily cut the model down to its first k layers (the fast path)."""
    all_layers = model.model.layers
    model.model.layers = torch.nn.ModuleList(all_layers[:k])
    try:
        yield model
    finally:
        model.model.layers = all_layers


@torch.no_grad()
def fast_features(model, tok, prompts, k):
    """Run only the first k layers; returns [batch, hidden] last-token features."""
    batch = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with first_k_layers(model, k):
        out = model.model(**batch).last_hidden_state
    return out[:, -1, :].float()
