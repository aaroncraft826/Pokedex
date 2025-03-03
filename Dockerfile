FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime as build
WORKDIR /Pokedex

# Install the application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 12345
RUN chmod +x run.sh
ENTRYPOINT ["/bin/bash", "-c", "./run.sh"]
# ENTRYPOINT [ "torchrun" ]
# CMD [ "--nproc_per_node", "gpu", "--nnodes", "1:3", "-rdzv_id", "12345", "--rdzv_backend", "c10d", "--rdzv_endpoint", "10.0.1.169:12345", "PokeTrainer.py", "5" ]