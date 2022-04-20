# Проверка работы сервиса
import json

import requests
import numpy as np
import torch
import os

import requests
import torch
import os
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Query dispatcher service")
    parser.add_argument("--url", default="http://127.0.0.1:7000", help="url to service")
    parser.add_argument("--text", default="Hello, world!!", help="text to pass")
    parser.add_argument("--out", default="./out.wav", help="out wav filename")
    args = parser.parse_args()
    resp = requests.get(args.url, params={'text': args.text})
    print(resp.status_code)
    if resp.status_code != 200:
        print(resp.text)
    open(args.out, 'wb').write(resp.content)
