FROM python:3.11-slim-bookworm
WORKDIR /app
COPY . /app

RUN apt-get update && pip install -r requirements.txt
CMD ["python3", "main.py"]
