FROM python:3.8-slim-buster
RUN apt update && apt upgrade -y
RUN mkdir app
COPY requirements.txt .
RUN apt install -y libsndfile-dev
RUN pip install -r requirements.txt
ADD * /app/
WORKDIR /app
EXPOSE 7000
CMD ["uvicorn", "server:app", "--port", "7000", "--host", "0.0.0.0"]