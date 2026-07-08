# Slim base image — we don't need the full python:3.11 image with build tools
# we're not using; smaller image = faster deploys and smaller attack surface.
FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffers output immediately to
# stdout — makes container logs show up in real time instead of being buffered,
# which matters when you're watching logs on a cloud platform.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy and install dependencies BEFORE copying the rest of the code.
# Docker caches layers — as long as requirements-api.txt doesn't change,
# Docker reuses this layer on rebuilds instead of reinstalling everything,
# which makes iterating on code changes much faster.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Only copy what the API actually needs at runtime: the source code and the
# trained model checkpoint. We deliberately do NOT copy data/, notebooks/,
# tests/, or the training scripts — none of that belongs in a production image.
COPY src/ ./src/
COPY checkpoints/best_model.pth ./checkpoints/best_model.pth
COPY checkpoints/class_names.json ./checkpoints/class_names.json

EXPOSE 8000

# Bind to 0.0.0.0, not 127.0.0.1 — inside a container, the API must accept
# connections from outside the container itself, not just localhost within it.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
