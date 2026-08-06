# accio: the API, the worker and the built frontend in one image.
#
# The stitcher is deliberately not in here. MediaSDK ships as its own amd64
# image with its own model tree, and accio drives it by talking to the host's
# Docker daemon through the mounted socket, so the container needs the docker
# client but not the daemon. That is also why the data root has to be mounted
# at the same path inside and out: see accio/settings.py.

# --- the frontend ----------------------------------------------------------
FROM node:22-alpine AS web
# Pinned because npm versions disagree about the lock file, not for taste:
# 11.12+ writes hoisted @emnapi/* entries that 11.6 strips out again, so a
# local `npm install` and a `npm ci` from a stock image reject each other's
# work. One npm owns the lock. Match it to whatever `npm -v` says on the
# machine that regenerates it, or the build stops with "Missing: ... from
# lock file", which is a version disagreement and not a missing package.
ARG NPM_VERSION=11.6.2
RUN npm i -g npm@${NPM_VERSION}
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# --- the API ---------------------------------------------------------------
FROM python:3.12-slim

# ffprobe reads the .insv's fps and frame count before anything is stitched
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# the client only. Pulling docker.io from apt would bring a daemon we never
# start, to talk to the one on the host we do.
COPY --from=docker:27-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# dependencies before source, so editing a stage does not re-resolve torch
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
RUN uv sync --frozen --no-dev
COPY --from=web /web/dist ./web/dist

ENV PATH="/app/.venv/bin:$PATH" \
    ACCIO_WEB_DIST=/app/web/dist \
    HF_HOME=/models

EXPOSE 8000
CMD ["uvicorn", "accio.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
