"""M1a go/no-go: does a frozen LLM understand unseen instructions better than a small text encoder?

No simulator. We sample step-0 scenes (hand at its start, random puck, 3 markers, an instruction),
compute the true target, and train small MLP probes to predict the target's xy from different
representations. Metric: fraction of predictions within 5 cm, per split (test_paraphrase is the
one that matters).

Representations (numbers = per marker slot in prompt order: x, y, color one-hot; plus puck xy):
  qwen Lk               last-token layer-k features of the full prompt
  qwen Lk + numbers     same, plus the raw numbers (digits handled directly)
  qwen-instr Lk + num   layer-k features of the instruction alone, plus numbers (same input form
                        as the MiniLM baseline, so it compares the two encoders directly)
  minilm + numbers      all-MiniLM-L6-v2 mean-pooled instruction embedding, plus numbers
  bow + numbers         instruction word counts (train vocabulary), plus numbers
  numbers only          no language (floor)
  qwen-instr Lk only    instruction features without markers (checks for leaks)

Usage: python src/lang_probe.py [--models Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B] [--out outputs/lang_probe]
Needs a GPU.
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from torch import nn
from transformers import AutoModel, AutoTokenizer

from lang import COLORS, lang_prompt, sample_episode
from qwen import DEFAULT_LAYERS, layer_features_bucketed, load_qwen

warnings.filterwarnings("ignore")

SPLIT_SIZES = {"train": 4000, "val": 500, "test_iid": 500, "test_paraphrase": 1000, "test_compose": 500}
HAND = np.array([0.0, 0.6, 0.2])


def make_scenes(seed):
    """Step-0 scenes for every split. Returns list of dicts with prompt, instruction, numbers, target."""
    rng = np.random.default_rng(seed)
    scenes = []
    for split, n in SPLIT_SIZES.items():
        for _ in range(n):
            puck = np.array([rng.uniform(-0.1, 0.1), rng.uniform(0.6, 0.7), 0.02])
            ep = sample_episode(rng, "reach-v3", "train" if split == "val" else split, HAND[:2], puck[:2])
            obs = np.zeros(39)
            obs[0:3], obs[3], obs[4:7] = HAND, 1.0, puck
            slots = [v for c, x, y in ep["markers"] for v in [x, y] + [float(c == k) for k in COLORS]]
            scenes.append({
                "split": split, "prompt": lang_prompt(obs, ep["instruction"], ep["markers"]),
                "instruction": ep["instruction"], "family": ep["spec"]["family"],
                "numbers": slots + list(puck[:2]), "target": ep["target"][:2],
            })
    return scenes


@torch.no_grad()
def minilm_embed(texts, device):
    name = "sentence-transformers/all-MiniLM-L6-v2"
    tok, model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).to(device).eval()
    out = []
    for i in range(0, len(texts), 256):
        batch = tok(texts[i : i + 256], return_tensors="pt", padding=True, truncation=True).to(device)
        h = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        out.append(((h * mask).sum(1) / mask.sum(1)).cpu().numpy())
    return np.concatenate(out)


def fit_probe(x, y, split, device, epochs=300, seed=0):
    """MLP regression probe x -> target xy, best-val epoch. Returns predictions for all rows."""
    torch.manual_seed(seed)
    tr, va = split == "train", split == "val"
    mean, std = x[tr].mean(0), x[tr].std(0)
    std[std < 1e-6] = 1.0
    ym, ys = y[tr].mean(0), y[tr].std(0)
    xs = torch.tensor((x - mean) / std, dtype=torch.float32, device=device)
    yt = torch.tensor((y - ym) / ys, dtype=torch.float32, device=device)
    net = nn.Sequential(nn.Linear(x.shape[1], 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 512), nn.ReLU(),
                        nn.Linear(512, 2)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    tri = torch.tensor(np.where(tr)[0], device=device)
    vai = torch.tensor(np.where(va)[0], device=device)
    best, best_state = np.inf, None
    for _ in range(epochs):
        net.train()
        for idx in tri[torch.randperm(len(tri), device=device)].split(256):
            loss = nn.functional.mse_loss(net(xs[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            err = ((net(xs[vai]) - yt[vai]) * torch.tensor(ys, device=device)).norm(dim=1).mean().item()
        if err < best:
            best, best_state = err, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    with torch.no_grad():
        return net(xs).cpu().numpy() * ys + ym


def score(pred, y, split):
    dist = np.linalg.norm(pred - y, axis=1)
    return {s: {"within_5cm": float((dist[split == s] <= 0.05).mean()), "mean_err_cm": float(100 * dist[split == s].mean())}
            for s in SPLIT_SIZES if s not in ("train", "val")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"])
    parser.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/lang_probe")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    scenes = make_scenes(args.seed)
    split = np.array([s["split"] for s in scenes])
    y = np.array([s["target"] for s in scenes])
    numbers = np.array([s["numbers"] for s in scenes], dtype=np.float32)
    instructions = [s["instruction"] for s in scenes]
    print(f"{len(scenes)} scenes; example:\n{scenes[0]['prompt']}\n")

    reps = {"numbers only": numbers}
    reps["minilm + numbers"] = np.concatenate([minilm_embed(instructions, device), numbers], 1)
    bow = CountVectorizer(token_pattern=r"[a-z0-9-]+").fit([t for t, s in zip(instructions, split) if s == "train"])
    reps["bow + numbers"] = np.concatenate([bow.transform(instructions).toarray(), numbers], 1)

    results = {}
    for name, x in reps.items():
        results[name] = score(fit_probe(x, y, split, device), y, split)
        print(f"{name:<28}" + "  ".join(f"{s}: {r['within_5cm']:.0%}" for s, r in results[name].items()))

    for model_name in args.models:
        short = model_name.split("/")[-1]
        tok, model = load_qwen(model_name)
        start = time.time()
        full = layer_features_bucketed(model, tok, [s["prompt"] for s in scenes], args.layers)
        instr = layer_features_bucketed(model, tok, [f"Instruction: {t}." for t in instructions], args.layers)
        print(f"\n{short}: features in {time.time() - start:.0f}s")
        for k in args.layers:
            f, fi = full[k].cpu().numpy(), instr[k].cpu().numpy()
            for name, x in [(f"{short} L{k}", f), (f"{short} L{k} + numbers", np.concatenate([f, numbers], 1)),
                            (f"{short}-instr L{k} + numbers", np.concatenate([fi, numbers], 1)),
                            (f"{short}-instr L{k} only", fi)]:
                results[name] = score(fit_probe(x, y, split, device), y, split)
                print(f"{name:<28}" + "  ".join(f"{s}: {r['within_5cm']:.0%}" for s, r in results[name].items()))
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump({"args": vars(args), "split_sizes": SPLIT_SIZES, "results": results}, fh, indent=2)
    print(f"\nSaved {args.out}/results.json")


if __name__ == "__main__":
    main()
