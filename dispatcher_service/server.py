from ast import Str
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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


def clean_text(text):
    return re.sub(r'[^A-Za-z ]+', '', text).lower()

app = FastAPI()

class AcousticException(Exception):
    pass

class VocoderException(Exception):
    pass

ACOUSTIC_URL = os.getenv('ACOUSTIC_URL', "http://0.0.0.0:5000")
VOCODER_URL  = os.getenv("VOCODER_URL",  "http://0.0.0.0:8000")

@app.get('/')
def hello_world(text: str):  # may be it's worth it to use async
    try:
        text = clean_text(text)
        with httpx.Client() as client:
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
            return StreamingResponse(io.BytesIO(resp_vocoder.content), media_type="audio/wav")
    except Exception as exp:
        raise HTTPException(status_code=500, detail=exp.__repr__())


client = TestClient(app)

def test_make_audio():
    text = "Hello, world!"
    response = client.post("/", json={'text': text})
    assert response.status_code == 200
    open('out.wav', 'wb').write(response.content)