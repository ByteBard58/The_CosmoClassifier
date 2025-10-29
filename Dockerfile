# Dockerfile (urbanecho)
FROM python:3.11-slim

# Avoid interactive prompts
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# copy constraints / requirements first for better caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# copy app source
COPY . .

# example env and port
ENV FLASK_APP=app.py
EXPOSE 5000

# run the flask app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--workers", "2"]
