from ast import Str
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic.types import List
from pydantic import BaseModel
from env import AttrDict
import io

from scipy.io.wavfile import write
import torch
import numpy as np
import json
import os

from meldataset import MAX_WAV_VALUE
from inference_e2e_2 import load_checkpoint, h, device
from config import checkpoint_file
from models import Generator


config_file = 'config.json'
with open(config_file) as f:
    data = f.read()

json_config = json.loads(data)
h = AttrDict(json_config)

torch.manual_seed(h.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'DEVICE is {device}')

generator = Generator(h).to(device)

print(checkpoint_file)
state_dict_g = load_checkpoint(checkpoint_file, device)
generator.load_state_dict(state_dict_g['generator'])

generator.eval()
generator.remove_weight_norm()


app = FastAPI()

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import logging

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(Request.body, 'body')

    exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
    # or logger.error(f'{exc}')
    print(request)
    # print(exc_str)
    logging.error(request, exc_str)
    content = {'status_code': 422, 'message': exc_str, 'data': None}
    return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

@app.get("/")
async def root():
    return {"message": "Hello World"}

from fastapi import Body
@app.post("/")
async def root(mel: List[List[List[float]]] = Body(...)):
    #print(mel)
    #print('inside query', flush=True)
    def generate_mels():
        print('generating', flush=True)
        with torch.no_grad():
            x = torch.tensor(mel, device=device)
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            f = io.BytesIO()
            write(f, h.sampling_rate, audio)
            yield from f
    return StreamingResponse(generate_mels(), media_type="audio/wav")

class Item(BaseModel):
    name: str
    price: float

@app.post("/items/")
async def create_item(item: Item):
    return item