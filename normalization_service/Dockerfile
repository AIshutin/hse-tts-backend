FROM gcc:12.1.0
RUN apt update && apt upgrade -y
RUN apt-get install -y python3 python3-pip python3-dev
RUN apt-get install -y libsndfile1 ffmpeg 
RUN mkdir app
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN python3 -m pip install git+https://github.com/NVIDIA/NeMo.git
ADD * /app/
WORKDIR /app
EXPOSE 9997
ENV FLASK_APP=/app/run_predict.py
CMD ["flask", "run", "--port", "9997", "--host", "0.0.0.0"]