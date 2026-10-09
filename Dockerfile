# Build stage: resolve the locked dependencies into a virtualenv with uv.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# Runtime stage: the virtualenv only, no build tools, unprivileged user.
FROM python:3.12-slim
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin vcmem \
    && mkdir /data && chown 10001:10001 /data
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    VCMEM_DATA_DIR=/data
USER 10001
WORKDIR /home/vcmem
EXPOSE 8765
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/healthz', timeout=4)"]
CMD ["vcmem", "serve", "--transport", "http", "--host", "0.0.0.0", "--port", "8765"]
