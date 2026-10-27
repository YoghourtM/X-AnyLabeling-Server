# Local SAM 3.1 server.
#
# The base image must provide CUDA PyTorch and the upstream `sam3` package
# with SAM 3.1 multiplex support (facebookresearch/sam3 >= 2026-03-27), e.g.
# an image built from that repository's instructions. Only the HTTP server
# dependencies are added here; the source tree is copied for standalone use
# and is usually bind-mounted over /app during development.
ARG BASE_IMAGE=cgd-sam3:2.10.0-cu128-local
FROM ${BASE_IMAGE}

# Keep the base image's numpy 1.x (opencv 4.12+ requires numpy 2).
RUN pip install --no-cache-dir \
        "fastapi[standard]>=0.115.0" \
        "pydantic>=2.12.0" \
        "opencv-python-headless>=4.11.0,<4.12" \
        "numpy<2" \
        loguru requests packaging pyyaml

WORKDIR /app
COPY app ./app
COPY configs ./configs

ENV PYTHONUNBUFFERED=1 \
    XANYLABELING_MODELS_CONFIG=/app/configs/models.sam31.yaml \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/cache

EXPOSE 8000
ENTRYPOINT ["python", "-m", "app.cli", "--host", "0.0.0.0", "--port", "8000"]
