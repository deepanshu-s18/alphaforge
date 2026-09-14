FROM python:3.12-slim AS base

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY evals ./evals

RUN pip install --no-cache-dir -e . && mkdir -p /app/reports

# headless synthetic run by default; override args for other modes
ENTRYPOINT ["alphaforge"]
CMD ["post-earnings drift in megacap tech"]
