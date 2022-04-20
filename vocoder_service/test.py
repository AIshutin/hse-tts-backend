import requests
import torch
import os
import numpy as np

host = "http://127.0.0.1:8000"

resp = requests.get(host)
print(resp.status_code)
# x = torch.load("LJ001-mel.pt")
x = torch.tensor(np.load('../audio_processing/FastPitch/output/LJSpeech-1.1_mels/LJ001-0001.npy').T).unsqueeze(0)
print(x.shape)
resp = requests.post(host, json=x.tolist())
print(resp.status_code)
# print(resp.text)
open('out.wav', 'wb').write(resp.content)