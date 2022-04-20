# How to use locally

```
uvicorn server:app --host 0.0.0.0 --port 7000
```
```
pytest server.py # to test
```


# How to use in Docker
```
sudo docker build . --tag=dispatcher
```

to run (**unsecure**):
```
sudo docker run -p 7000:7000  --net=host dispatcher
```

```
python3 run_query.py -h # to get cli options
python3 run_query.py # to run with default params
```
