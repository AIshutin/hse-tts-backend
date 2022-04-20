# How to run server locally?

```
uvicorn server:app --host 0.0.0.0 --port 7000
```
# How to use in Docker?

To build:
```
sudo docker build . --tag=dispatcher
```

To run (**unsecure**):
```
sudo docker run -p 7000:7000  --net=host dispatcher
```
