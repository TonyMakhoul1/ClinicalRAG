# ── Base image ────────────────────────────────────────────────────────────────
# python:3.12-slim is the official Python image stripped of everything that isn't
# needed to run Python — no compilers, no docs, no extras.
# "slim" cuts the base from ~1 GB (full image) down to ~130 MB.
# We pin to 3.12 to match the project's .python-version requirement.
FROM python:3.12-slim

# Copy the /uv folder from the ghcr... image, and place it in my image /usr...
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# This is the root of the project inside the container.
WORKDIR /app

# Copy the pyproject.toml and uv.lock files from the local machine
# to the current directory in the container
COPY pyproject.toml uv.lock ./

# uv's default HTTP timeout is 30s — too short for large packages like torch or scipy.
# 300s gives each download chunk up to 5 minutes before timing out.
ENV UV_HTTP_TIMEOUT=300

# Install all the dependencies from pyproject.toml and uv.lock
# --frozen (don't modify uv.lock automatically, if there is some missing or something get wrong)
# --no-install-project, install only the dependencies, don't install the project itself
RUN uv sync --frozen --no-install-project

# Copy the actual source code.
COPY src/ ./src/
COPY main.py ./

# With the source now present, this installs the project package into the
# same environment that already has all the dependencies.
RUN uv sync --frozen

# when the container starts, run this command
CMD ["uv", "run", "uvicorn", "src.backend_src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
