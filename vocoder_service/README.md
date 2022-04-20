# How to run server locally

```
uvicorn server:app --host 0.0.0.0
```
# How to use in Docker
```
sudo docker build . --tag=vocoder
```

```
sudo docker run --gpus all -p 8000:8000 vocoder
```

Model code was taken from: https://github.com/jik876/hifi-gan