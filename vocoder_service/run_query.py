import requests
import torch
import os
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Query vocoder service")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="url to service")
    parser.add_argument("--mel", default="./mel_example.pt", help="pt tensor with mel")
    parser.add_argument("--out", default="./out.wav", help="out wav filename")
    args = parser.parse_args()
    x = torch.load(args.mel)
    print(x.shape)
    resp = requests.post(args.url, json=x.tolist())
    print(resp.status_code)
    open(args.out, 'wb').write(resp.content)
