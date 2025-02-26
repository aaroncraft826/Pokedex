FROM python:3.12-slim

WORKDIR /Pokedex

# Install the application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the source code
COPY . .

CMD ["python", "training.py"]

FROM python:3.12-slim
WORKDIR /Pokedex