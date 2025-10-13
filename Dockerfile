# syntax=docker/dockerfile:1.4
FROM python:3.10-slim AS runtime

ARG POETRY_VERSION=1.6.1
ARG WITH_EXTRAS=""
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \ \
    && apt-get install --no-install-recommends -y build-essential curl \ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /workspace

COPY pyproject.toml poetry.lock* README.md ./
COPY neva ./neva
COPY examples ./examples
COPY tests ./tests
COPY requirements*.txt ./
COPY CONTRIBUTING.md CHANGELOG.md CODE_OF_CONDUCT.md LICENSE ./

RUN set -eux; \
    poetry config virtualenvs.create false; \
    extras=""; \
    if [ -n "${WITH_EXTRAS}" ]; then \
        for extra in $(echo "${WITH_EXTRAS}" | tr ',' ' '); do \
            extras="${extras} --extras ${extra}"; \
        done; \
    fi; \
    poetry install --no-interaction ${extras}

CMD ["python"]
