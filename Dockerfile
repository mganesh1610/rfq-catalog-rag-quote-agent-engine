FROM python:3.12-slim

WORKDIR /app
COPY src/ ./src/
COPY crm/ ./crm/
COPY docs/ ./docs/
COPY outputs/ ./outputs/
COPY tests/ ./tests/

ENV PYTHONUNBUFFERED=1
CMD ["python3", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"]
