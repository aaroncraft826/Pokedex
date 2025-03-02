FROM python:3.12-slim as build
WORKDIR /Pokedex

# Install the application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim as production
WORKDIR /Pokedex

# Copy in the source code
COPY . .

# Run torchrun
CMD [ "torchrun", "-nproc_per_node", "gpu", "--nnodes", "1:3", "-rdzv_id", "12345", "--rdzv_backend", "c10d", "--rdzv_endpoint", "{localhost:12345}", "PokeTrainer.py", "10" ]