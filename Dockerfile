FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml ./
COPY requirements.txt ./
COPY . .

RUN uv venv
RUN uv pip install -r requirements.txt

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONPATH="$VIRTUAL_ENV/lib/python3.12/site-packages:/app/src"

EXPOSE 8000

CMD ["/bin/bash", "-c", "source /app/.venv/bin/activate && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"]