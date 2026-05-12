# start from a base image
FROM python:3.12-slim

# install OS packages
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        iputils-ping \
        wget \
        git \
        curl \
        tini \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir uv

ENV UV_LINK_MODE=copy

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN uv venv

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pkg,target=pkg \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable

COPY . /app/

RUN uv sync --locked --no-editable

# make the virtual environment accessible
ENV PATH="/app/.venv/bin:$PATH"

# Initialize DVC
RUN dvc init --no-scm

# Add a (local) remote storage which can be mounted
RUN mkdir -p ./dvcstore
RUN dvc remote add -d --local myremote ./dvcstore

# run with tini
ENTRYPOINT [ "tini", "--", "python", "-O", "-m", "ssc4frames" ]

# to be extended with docker run, e.g. the subcommand data with
#   docker run --rm -ti remstef/ssc4frames data 
CMD [ ]