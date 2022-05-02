# Проверка работы сервиса
import json

import requests
import numpy as np
import torch
import os

ans = requests.get('http://127.0.0.1:5000/?text=Hello+I+am+text+And+it+works')
print("request to acoustic status:", ans.status_code)
print("Try connect to vocoder")
try:
    host = "http://127.0.0.1:8000"
    resp = requests.get(host)
    #print(np.array(ans.json()).T.shape)
    x = torch.from_numpy(np.array(ans.json()).T).unsqueeze(0)
    print(x.shape)
    obj = x.tolist()
    resp = requests.post(host, json=x.tolist())
    print("request to vocoder status:", resp.status_code)
    open('out.wav', 'wb').write(resp.content)
    print("audio saved to out.wav")
except:
    print("Can't connect to vocoder")