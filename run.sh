total_epochs=10
torchrun --nproc_per_node=1 --nnodes=1:3 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=10.0.1.243:12345 PokeTrainer.py $total_epochs