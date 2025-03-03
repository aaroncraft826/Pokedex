total_epochs=5
torchrun --nproc_per_node=1 --nnodes=1:3 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=10.0.1.169:12345 PokeTrainer.py $total_epochs