# Pokedex

local: torchrun --standalone --nproc_per_node=gpu PokeTrainer.py 5

global: torchrun --nproc_per_node=gpu --nnodes=3 --node_rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint={ip_addr:port} PokeTrainer.py 10