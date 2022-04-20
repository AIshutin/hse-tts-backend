FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
RUN apt update && apt upgrade -y
RUN mkdir app
COPY requirements.txt .
RUN apt install -y libsndfile-dev
RUN pip install -r requirements.txt
ADD * /app/
WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "server:app", "--port", "8000", "--host", "0.0.0.0"]