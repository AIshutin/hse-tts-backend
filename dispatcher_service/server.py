from ast import Str
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from numpy import quantile
from pydantic.types import List
from pydantic import BaseModel
import io
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.testclient import TestClient
import logging
from scipy.io.wavfile import write
import numpy as np
import json
import os
import requests as rq
import re
import httpx
import asyncio
import redis
import time
import math
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import base64

def clean_text(text):
    return re.sub(r'[^A-Za-z ]+', '', text).lower()

app = FastAPI()

class AcousticException(Exception):
    pass

class VocoderException(Exception):
    pass

class RedisException(Exception):
    pass

class NormalizationException(Exception):
    pass

ACOUSTIC_URL = os.getenv('ACOUSTIC_URL', "http://acoustic_servise:5000/")
VOCODER_URL  = os.getenv("VOCODER_URL",  "http://vocoder_service:8000/")
NORMALIZATION_URL = os.getenv('ACOUSTIC_URL', "http://normalization_service:9997/")
r = redis.Redis(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"), db=0)

class StatisticsClass:
    queries_cnt = 0
    failed_queries_cnt = 0
    cached_queries_cnt = 0
    start_time = None
    records = []
    quantile = 0.95
    thrs = [25, 50, 100, 200, 400, 800]

    def __init__(self):
        self.start_time = time.time()

    def add_record(self, n, t):
        self.records.append((n, t))
    
    def calc_quantile(self, data):
        if len(data) == 0:
            return None
        ind = min(math.ceil(len(data) * self.quantile), len(data) - 1)
        data.sort()
        return data[ind]
    
    def get_quantiles(self):
        datas = [[] for i in range(len(self.thrs))]
        for el in self.records:
            for i in range(len(self.thrs)):
                if el[0] <= self.thrs[i]:
                    datas[i].append(el[1])
                    break
        return [(self.thrs[i], self.calc_quantile(datas[i])) for i in range(len(self.thrs))]

    def plot_distribution(self):
        lengths = [el[0] for el in self.records]
        ts = [el[1] for el in self.records]
        plt.plot(lengths, ts, 'o', color='black')
        plt.title("generation time vs length of text")
        plt.xlabel("Length of text")
        plt.ylabel("Generation time in seconds")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        encoded_string = base64.b64encode(buf.read())
        return encoded_string.decode("ascii")

statistics = StatisticsClass()

@app.get('/')
def tts(text: str):  # may be it's worth it to use async
    statistics.queries_cnt += 1
    start_T = time.time()
    N = len(text)
    try:
        with httpx.Client() as client:
            try:
                resp_normalization = client.get(NORMALIZATION_URL, params={"text": text})
                print('NORMALIZED', resp_normalization.text, flush=True)
                assert(resp_normalization.status_code == 200)
                text = clean_text(resp_normalization.text)
            except Exception as exp:
                raise AcousticException(resp_normalization.text)
            try:
                out_bytes = r.get(text)
            except Exception as exp:
                raise RedisException(exp.__repr__())
            if out_bytes is None:
                try:
                    resp_acoustic = client.get(ACOUSTIC_URL, params={"text": text})
                except Exception as exp:
                    raise AcousticException(exp.__repr__())
                if resp_acoustic.status_code != 200:
                    raise AcousticException(resp_acoustic.text)
                mel = [np.array(resp_acoustic.json()).T.tolist()]
                try:
                    resp_vocoder = client.post(VOCODER_URL, json=mel)
                except Exception as exp:
                    raise VocoderException(exp.__repr__())
                if resp_vocoder.status_code != 200: 
                    raise VocoderException(resp_vocoder.text)
                try:
                    r.set(text, resp_vocoder.content)
                except Exception as exp:
                    raise RedisException(exp.__repr__())
                out_bytes = resp_vocoder.content
                statistics.add_record(N, time.time() - start_T)
            else:
                statistics.cached_queries_cnt += 1
        return StreamingResponse(io.BytesIO(out_bytes), media_type="audio/wav")
    except Exception as exp:
        statistics.failed_queries_cnt += 1
        print(exp.__repr__())
        raise HTTPException(status_code=500, detail=exp.__repr__())

@app.get('/analytics')
def analytics():
    quantile_key = f"{statistics.quantile*100}% quantiles by query length"
    my_data = {
        "cache hit rate": f"{statistics.cached_queries_cnt / (statistics.queries_cnt + 1e-10) * 100:.3f}%",
        "total queries": statistics.queries_cnt,
        "uptime": datetime.fromtimestamp(int(time.time() - statistics.start_time)).strftime("%H:%M:%S"),
        "failed queries": statistics.failed_queries_cnt,
        "failure percent": f"{statistics.failed_queries_cnt / (statistics.queries_cnt + 1e-10)*100:.3f}%",
        "response time distr": statistics.plot_distribution()
    }
    my_data[quantile_key] = dict()
    for el in statistics.get_quantiles():
        my_data[quantile_key]['<= ' + str(el[0]) + " symbols length"] = f"{el[1]:.3f} s" if el[1] is not None else str(el[1])
    vocoder_key = 'vocoder configuration'
    try:
        my_data[vocoder_key] = rq.get(VOCODER_URL + 'config').json()
    except Exception as exp:
        my_data[vocoder_key] = exp.__repr__()
    return JSONResponse(content=my_data)

client = TestClient(app)

def test_make_audio():
    text = "Hello, world!"
    response = client.post("/", json={'text': text})
    assert response.status_code == 200
    open('out.wav', 'wb').write(response.content)