import json

import requests
import numpy as np
import torch
import os

ans = requests.get('http://127.0.0.1:9997/?text=Hello+I+am+text+And+it+works+228')
host = "http://127.0.0.1:9997"
print(ans.text)