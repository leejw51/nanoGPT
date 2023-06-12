all: install prepare train generate
	
prepare:
	python3 data/shakespeare_char/prepare.py
train:
	python3 train.py config/train_shakespeare_char.py --device=mps --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
traincpu:
	python3 train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
generate:
	python3 sample.py --out_dir=out-shakespeare-char --device=mps
install:
	python3 -m pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
	python3 -m pip install transformers datasets tiktoken wandb tqdm
	@echo "use python 3.10"
