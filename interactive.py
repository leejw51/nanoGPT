"""
Interactive REPL for a trained or pretrained nanoGPT model.

Loads the model ONCE, then loops: read a prompt -> generate -> print -> repeat.
(sample.py reloads the model every run; this keeps it resident for fast turns.)

Usage:
    python interactive.py --out_dir=out-shakespeare-char --device=cpu   # your trained model
    python interactive.py --init_from=gpt2 --device=cpu                 # pretrained GPT-2
Type a prompt and press Enter. Quit with ':q', 'exit', or Ctrl-D / Ctrl-C.

This is a text-completion model, not a chatbot: it continues whatever you type.
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# ----- config (override via configurator.py / CLI, like sample.py) -----
init_from = 'resume'        # 'resume' (from out_dir) or a 'gpt2*' variant
out_dir = 'out'             # used when init_from == 'resume'
max_new_tokens = 300        # tokens generated per response
temperature = 0.8           # <1.0 safer/sharper, >1.0 wilder
top_k = 200                 # keep only the top_k most likely tokens
seed = 1337
device = 'cuda'             # 'cpu', 'cuda', 'mps'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
exec(open('configurator.py').read())
# -----------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ----- load the model once -----
print(f"loading model (init_from={init_from}, device={device})...")
checkpoint = None
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise SystemExit(
            f"no checkpoint at {ckpt_path}.\n"
            f"Train one first (make train), or use a pretrained model: "
            f"python interactive.py --init_from=gpt2 --device={device}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
else:
    raise SystemExit(f"unknown init_from={init_from!r} (use 'resume' or a 'gpt2*' variant)")

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# ----- set up the encoder/decoder (same logic as sample.py) -----
load_meta = False
if init_from == 'resume' and checkpoint is not None and 'config' in checkpoint \
        and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    raw_encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    char_level = True
else:
    enc = tiktoken.get_encoding("gpt2")
    raw_encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    char_level = False

def encode(s):
    """Encode, silently dropping characters a char-level model has never seen."""
    if not char_level:
        return raw_encode(s)
    kept = [c for c in s if c in stoi]
    if len(kept) != len(s):
        dropped = sorted(set(s) - set(stoi))
        print(f"  (note: ignoring chars not in this model's vocab: {dropped})")
    return [stoi[c] for c in kept]

# ----- example prompts to get the user started -----
if char_level:
    examples = ["ROMEO:", "To be, or not to be", "JULIET:\nO Romeo, Romeo!", "First Citizen:"]
    flavor = "tiny Shakespeare (character-level)"
else:
    examples = ["Once upon a time", "The meaning of life is",
                "In a shocking discovery, scientists", "def fibonacci(n):"]
    flavor = "GPT-2 (BPE tokens)"

print()
print("=" * 60)
print(f"  nanoGPT interactive REPL  -  {flavor}")
print(f"  max_new_tokens={max_new_tokens}  temperature={temperature}  top_k={top_k}")
print("=" * 60)
print("  Try one of these prompts (or write your own):")
for ex in examples:
    preview = ex.replace("\n", "\\n")
    print(f"     {preview}")
print("  Commands:  :q / exit / Ctrl-D  to quit")
print("=" * 60)

# ----- the read -> generate -> print loop -----
while True:
    try:
        prompt = input("\nprompt> ")
    except (EOFError, KeyboardInterrupt):
        print("\nbye!")
        break

    if prompt.strip().lower() in (":q", ":quit", "exit", "quit"):
        print("bye!")
        break
    if prompt == "":
        prompt = "\n"  # empty -> unconditional generation

    start_ids = encode(prompt)
    if not start_ids:
        start_ids = encode("\n")
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

    print("-" * 60)
    print(decode(y[0].tolist()))
    print("-" * 60)
