FROM python:3.11-slim-bookworm

# HuggingFace Spaces runs the container as uid 1000 and expects the app on 7860.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

RUN useradd -m -u 1000 user
USER user
WORKDIR /app

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user:user . /app

# Artifacts, logs and the MLflow store are written at runtime and are not in the image.
RUN mkdir -p /app/Artifacts /app/logs /app/final_model

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
