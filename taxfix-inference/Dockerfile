FROM python:3.11-slim-buster AS builder

ENV POETRY_VERSION=1.8.5
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
RUN pip install poetry==$POETRY_VERSION

ENV PATH=/root/.poetry/bin:$PATH
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-ansi --only main --no-root


FROM python:3.11-slim-buster AS prod
RUN apt-get update -y && apt-get install -y curl
COPY --from=builder /app/.venv /app/.venv/
ENV PATH=/app/.venv/bin:$PATH
ENV VIRTUAL_ENV=/app/.venv
WORKDIR /app
COPY /src /app
