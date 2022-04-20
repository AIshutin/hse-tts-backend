# How to use locally

```
uvicorn server:app --host 0.0.0.0 # to run
```

```
pytest server.py # to test
```

# How to use in Docker
```
sudo docker build . --tag=vocoder # to build
```

```
sudo docker run --gpus all -p 8000:8000 vocoder # to run
```

```
python3 run_query.py -h # to get cli options
python3 run_query.py # to run with default params
```


Model code was taken from: https://github.com/jik876/hifi-gan