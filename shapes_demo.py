"""
shapes_demo.py — print the tensor shape at every step of a nanoGPT forward pass.

Run:
    python shapes_demo.py

Needs torch only (no dataset, no GPU, no checkpoint required).

HOW IT WORKS
------------
This uses PyTorch *forward hooks*, so it works WITHOUT modifying model.py.
A hook is attached to every leaf sub-module (Embedding, Linear, LayerNorm,
GELU, Dropout). As a batch flows through the network, each hook prints the
shape of that module's input and output, in execution order.

WHAT IT CANNOT SHOW
-------------------
The internal attention tensors (q, k, v, the [B, nh, T, T] attention matrix)
live *inside* CausalSelfAttention.forward and are not their own nn.Modules,
so no hook fires for them. Flash attention also runs as a functional call.
To see those internal shapes, apply the small `dbg(...)` edits to model.py
described in nanogpt_explained.html -> "Print tensor shapes", then run any
script (this one, sample.py, or train.py) with:
    DEBUG_SHAPES=1 python shapes_demo.py
"""

import torch
from model import GPTConfig, GPT

# A deliberately tiny model so the printout is short and readable.
# (head size = n_embd / n_head = 32 / 4 = 8)
cfg = GPTConfig(
    vocab_size=65,   # tiny-shakespeare character vocab
    block_size=8,    # context length T
    n_layer=2,       # stack of 2 transformer blocks
    n_head=4,        # 4 attention heads
    n_embd=32,       # embedding width C
    dropout=0.0,
    bias=False,
)
B, T = 4, cfg.block_size          # batch of 4 sequences, each 8 tokens long

model = GPT(cfg)
model.eval()

# fake a batch of token ids and targets (so the loss branch runs too)
idx = torch.randint(0, cfg.vocab_size, (B, T))
targets = torch.randint(0, cfg.vocab_size, (B, T))


def shape(x):
    if isinstance(x, torch.Tensor):
        return str(tuple(x.shape))
    if isinstance(x, (tuple, list)):
        return str([shape(t) for t in x])
    return type(x).__name__


def make_hook(name):
    def hook(module, inp, out):
        kind = module.__class__.__name__
        print(f"{name:<24} {kind:<11} in={shape(inp[0]):<16} out={shape(out)}")
    return hook


print(f"\nconfig: B={B} T={T} n_embd={cfg.n_embd} n_head={cfg.n_head} "
      f"n_layer={cfg.n_layer} vocab={cfg.vocab_size} "
      f"head_size={cfg.n_embd // cfg.n_head}\n")
print(f"{'input idx (token ids)':<24} {'-':<11} in={'-':<16} out={shape(idx)}")
print("-" * 78)

# hook only leaf modules (those with no children) so we print real ops in order
for name, module in model.named_modules():
    if name and len(list(module.children())) == 0:
        module.register_forward_hook(make_hook(name))

with torch.no_grad():
    logits, loss = model(idx, targets)

print("-" * 78)
print(f"{'logits (all positions)':<24} {'Linear':<11} in={'-':<16} out={shape(logits)}")
print(f"{'loss (scalar)':<24} {'cross_entropy':<11} in={'-':<16} "
      f"out={shape(loss)}  value={loss.item():.4f}")
print()
