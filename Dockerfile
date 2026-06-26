# syntax=docker/dockerfile:1

# ---- builder: compile Proxima, install deps, bake the model + serving index ----
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch keeps the image small (no CUDA).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[train,serve]" \
 && pip install "git+https://github.com/sushantlokhande14/proxima.git"

COPY config ./config
COPY scripts ./scripts
COPY backend ./backend

# Bake the trained two-tower and a MovieLens-only serving index. No TMDB key is
# needed at build time; the entrypoint refreshes live titles when a key is set.
ENV HF_HOME=/app/.hf_cache \
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN python scripts/train_two_tower.py --config config/default.yaml \
 && python scripts/refresh_tmdb.py  --config config/default.yaml

# ---- runtime: slim image with the venv, artifacts, and model cache ----
FROM python:3.12-slim
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache \
    KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app
COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /app
EXPOSE 8000
CMD ["/entrypoint.sh"]
