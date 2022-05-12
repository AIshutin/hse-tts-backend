from flask import Flask
from flask import jsonify
from flask import request
from argparse import ArgumentParser
from typing import List

from nemo_text_processing.text_normalization.normalize import Normalizer
import torch


'''
Runs normalization prediction on text data
'''

app = Flask(__name__)
normalizer = Normalizer(input_case="cased", lang="en")

@app.route('/', methods=['GET', 'POST'])
def normalization():
    text = request.args.get('text')
    answer = normalizer.normalize(text)
    print('OUT', answer, flush=True)
    return answer

if __name__ == "__main__":
    app.run()
