# Проверка работы сервиса
import json

import requests
import numpy as np
import torch
import os

resp = requests.get('http://127.0.0.1:7000/', params={'text': 'Hello I am text'})
print(resp.status_code)
if resp.status_code != 200:
    print(resp.text)
print(resp.text)
open('out.wav', 'wb').write(resp.content)
