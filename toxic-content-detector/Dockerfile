FROM python:3.12-slim

WORKDIR /app

# Install uv in the container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project metadata first for Docker layer caching
COPY pyproject.toml .python-version ./

# Install project dependencies with uv
RUN uv sync

# Copy application code
COPY app ./app
COPY data ./data

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.port=8501"]
