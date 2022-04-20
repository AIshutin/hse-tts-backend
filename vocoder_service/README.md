# How to run server locally?

```
uvicorn server:app --host 0.0.0.0
```
# How to use in Docker?

To build:
```
sudo docker build . --tag=vocoder
```

To run:
```
sudo docker run --gpus all -p 8000:8000 vocoder
```