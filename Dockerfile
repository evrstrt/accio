# The stitcher (MediaSDK) is not in this image. accio starts it as a sibling
# container through the host's Docker socket, so only the docker client is
# needed here. See settings.py for why the data root must be mounted at the
# same path inside and out.
#
# Runs as root: the mounted socket is host root anyway, and a non-root user
# would need the host's docker group GID baked in.

FROM node:22-alpine AS web
# npm 11.12+ writes hoisted @emnapi/* lock entries that 11.6 strips again, so
# the lock file only round-trips with one npm version. Match whatever
# regenerated package-lock.json.
ARG NPM_VERSION=11.6.2
RUN npm i -g npm@${NPM_VERSION}
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY --from=docker:27-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

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
