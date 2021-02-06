FROM tiangolo/uvicorn-gunicorn-fastapi:python3.6

COPY ./app /app
COPY ./sentiment /app/sentiment
COPY requirements.txt /app

WORKDIR /app
RUN pip install -r requirements.txt 
EXPOSE 80