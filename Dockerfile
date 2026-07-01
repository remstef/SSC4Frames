# start from a base image
FROM pytorch/pytorch:2.12.1-cuda13.2-cudnn9-runtime AS base

# ---- Keeps Python from generating .pyc files during runtime in the container ----
ENV PYTHONDONTWRITEBYTECODE=1

# ---- Turns off buffering for easier container logging ----
ENV PYTHONUNBUFFERED=1

# ---- pip environment settings, for faster building ----
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_ROOT_USER_ACTION=ignore

# ---- uv settings ----
ENV UV_LINK_MODE=copy

# ---- cuda/pytorch related environment variables ----
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# ---- set the workspace ----
WORKDIR /app

# ---- Install system dependencies ----
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        iputils-ping \
        wget \
        git \
        curl \
        tini

# ---- Create virtual environment, but allow access to system site packages (pytorch) ----
RUN uv venv --system-site-packages

# ---- Install deps (cached if requirements unchanged) ----
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pkg,target=pkg \
    --mount=type=bind,source=requirements-no-torch.txt,target=requirements-no-torch.txt \
    uv pip install -r requirements-no-torch.txt --no-cache-dir

# ---- Copy app ----
COPY . /app

# ---- Install ptgcl package, dependencies have been installed already ----
RUN uv pip install --no-cache-dir --no-editable --no-deps .

# ---- Cleanup system installs ----
RUN apt-get purge -y --auto-remove build-essential \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# ---- Make virtual environment accessible ----
ENV PATH="/app/.venv/bin:$PATH"

# Initialize DVC
RUN dvc init --no-scm

# Add a (local) remote storage which can be mounted
RUN mkdir -p ./dvcstore
RUN dvc remote add -d --local myremote ./dvcstore

# ---- Create non-root user with an explicit UID and adds permission to access the /app folder ----
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
RUN useradd -u 5678 -M -s /usr/sbin/nologin appuser && chown -R appuser /app
USER appuser

# run with tini
ENTRYPOINT [ "tini", "--", "ssc4frames" ]

# to be extended with docker run, e.g. the subcommand data with
#   docker run --rm -ti remstef/ssc4frames data 
CMD [ ]